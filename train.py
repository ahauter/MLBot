#!/usr/bin/env python3
"""
YAML-Configured Training Loop
==============================
Algorithm-agnostic training loop driven by YAML config + importlib.

The training loop knows NOTHING about PPO, SAC, or any specific algorithm.
It only calls Algorithm ABC methods: select_action, store_transition,
should_update, update. Everything is resolved from YAML at runtime.

Usage
-----
    # Default config:
    python train.py --config configs/ppo_sparse.yaml

    # Override params:
    python train.py --config configs/ppo_sparse.yaml --seed 3 --total-steps 10000

    # Minimal run (no YAML, uses defaults):
    python train.py --total-steps 10000 --num-envs 2
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing
import multiprocessing.connection
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO))

from encoder import N_TOKENS, TOKEN_FEATURES


# ── universal defaults ──────────────────────────────────────────────────────

UNIVERSAL_DEFAULTS = {
    'seed': 0,
    'total_steps': 50_000_000,
    'eval_interval': 200_000,
    't_window': 8,
    'num_envs': 8,
    'reward_type': 'sparse',
    'model_dir': 'models/baseline',
    'log_interval': 10,
    'snapshot_dir': 'models/snapshots',
    'population': {
        'agents': 1,
        'generation_steps': 1_000_000,
        'generation_noise_scale': 0.01,
    },
}


# ── utility ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class(dotted_path: str):
    """Import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit('.', 1)
    return getattr(importlib.import_module(module_path), class_name)


def resolve_or_default(config: dict, section: str, default_class):
    """Resolve a class from config section, or use default."""
    section_cfg = config.get(section, {})
    if isinstance(section_cfg, dict) and 'class' in section_cfg:
        cls = load_class(section_cfg['class'])
    else:
        cls = default_class
    return cls


def _coerce_numbers(obj):
    """Recursively convert scientific-notation strings (e.g. '1e5') to int/float."""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        try:
            f = float(obj)
            return int(f) if f == int(f) else f
        except ValueError:
            pass
    return obj


def load_config(yaml_path: str, cli_overrides: Optional[dict] = None) -> dict:
    """Load YAML, resolve classes via importlib, merge defaults, apply CLI overrides."""
    with open(yaml_path) as f:
        config = _coerce_numbers(yaml.safe_load(f) or {})

    # Start with universal defaults, overlay YAML
    merged = _deep_merge(dict(UNIVERSAL_DEFAULTS), config)

    # Resolve algorithm class
    algo_path = merged.get('algorithm', {}).get('class')
    if algo_path:
        AlgoCls = load_class(algo_path)
        merged.setdefault('algorithm', {})['cls'] = AlgoCls
        # Merge class defaults with YAML overrides
        class_defaults = AlgoCls.default_params() if hasattr(
            AlgoCls, 'default_params') else {}
        yaml_params = merged.get('algorithm', {}).get('params', {})
        merged['algorithm']['params'] = {**class_defaults, **yaml_params}

    # Resolve opponent pool class
    pool_path = merged.get('opponent_pool', {}).get('class')
    if pool_path:
        PoolCls = load_class(pool_path)
        merged.setdefault('opponent_pool', {})['cls'] = PoolCls
        class_defaults = PoolCls.default_params() if hasattr(
            PoolCls, 'default_params') else {}
        yaml_params = merged.get('opponent_pool', {}).get('params', {})
        merged['opponent_pool']['params'] = {**class_defaults, **yaml_params}

    # Apply CLI overrides (highest priority)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                merged[k] = v

    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ── null ABC defaults ───────────────────────────────────────────────────────

class NullReplayProvider:
    """No expert data. Axis 2 cost = 0."""

    def load_demonstrations(self):
        return None

    def seed_algorithm(self, algo, demos):
        pass

    def get_metrics(self):
        return {}


class NullFeedbackProvider:
    """No human feedback. Axis 3 cost = 0."""

    def get_feedback_reward(self, *args):
        return None

    def should_query(self, step):
        return False

    def get_metrics(self):
        return {}


# ── SubprocVecEnv ───────────────────────────────────────────────────────────

def _env_worker(conn: multiprocessing.connection.Connection,
                t_window: int, reward_type: str,
                dense_reward_weights: Optional[dict] = None,
                env_class: Optional[str] = None):
    """Child process: owns one gym env, responds to commands."""
    if env_class:
        EnvCls = load_class(env_class)
    else:
        from training.environments.baseline_env import BaselineGymEnv
        EnvCls = BaselineGymEnv
    env = EnvCls(
        t_window=t_window,
        reward_type=reward_type,
        dense_reward_weights=dense_reward_weights,
    )
    # Opponent state for PPO snapshots
    _opponent_encoder = None
    _opponent_policy = None

    while True:
        try:
            cmd, data = conn.recv()
        except (EOFError, BrokenPipeError):
            break

        if cmd == 'reset':
            obs, info = env.reset()
            conn.send(('obs', obs, info))

        elif cmd == 'step':
            obs, reward, done, truncated, info = env.step(data)
            conn.send(('step', obs, reward, done, truncated, info))

        elif cmd == 'set_opponent_snapshot':
            # Load opponent from snapshot path for PPO self-play
            snap_path = data
            if snap_path is not None:
                from training.opponents.pool import load_opponent_from_snapshot
                weights = load_opponent_from_snapshot(snap_path, device='cpu')
                from encoder import SharedTransformerEncoder, D_MODEL
                from policy_head import StochasticPolicyHead
                if _opponent_encoder is None:
                    _opponent_encoder = SharedTransformerEncoder(
                        d_model=D_MODEL)
                    _opponent_policy = StochasticPolicyHead(d_model=D_MODEL)
                _opponent_encoder.load_state_dict(weights['encoder'])
                _opponent_policy.load_state_dict(weights['policy'])
                _opponent_encoder.eval()
                _opponent_policy.eval()

                # Monkey-patch the env's opponent action method
                import torch as _torch
                from encoder import ENTITY_TYPE_IDS_1V1 as _EIDS

                def _ppo_opponent_action(self_env):
                    stacked = np.stack(list(self_env._orange_buf), axis=0)
                    flat_obs = stacked.ravel().astype(np.float32)
                    with _torch.no_grad():
                        x = _torch.tensor(
                            flat_obs[np.newaxis], dtype=_torch.float32)
                        tokens = x.view(
                            1, self_env.t_window, -1, TOKEN_FEATURES)
                        eids = _torch.tensor(_EIDS, dtype=_torch.long)
                        emb = _opponent_encoder(tokens, eids)
                        action, _ = _opponent_policy.act_deterministic(emb)
                    return action[0].cpu().numpy().astype(np.float32)

                import types
                env._get_opponent_action = types.MethodType(
                    _ppo_opponent_action, env)
            conn.send(('ok',))

        elif cmd == 'close':
            env.close()
            conn.send(('closed',))
            break

        else:
            conn.send(('error', f'Unknown command: {cmd}'))


class SubprocVecEnv:
    """Vectorized environment using subprocesses."""

    def __init__(self, num_envs: int, t_window: int = 8,
                 reward_type: str = 'sparse',
                 dense_reward_weights: Optional[dict] = None,
                 env_class: Optional[str] = None):
        self.num_envs = num_envs
        self.parents: List[multiprocessing.connection.Connection] = []
        self.procs: List[multiprocessing.Process] = []

        for i in range(num_envs):
            parent_conn, child_conn = multiprocessing.Pipe()
            proc = multiprocessing.Process(
                target=_env_worker,
                args=(child_conn, t_window, reward_type, dense_reward_weights,
                      env_class),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.procs.append(proc)

    def reset_all(self) -> np.ndarray:
        """Reset all envs, return stacked observations (num_envs, obs_dim)."""
        for conn in self.parents:
            conn.send(('reset', None))
        obs_list = []
        for conn in self.parents:
            tag, obs, info = conn.recv()
            obs_list.append(obs)
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        """Step all envs. Returns (obs, rewards, dones, infos)."""
        for i, conn in enumerate(self.parents):
            conn.send(('step', actions[i]))
        obs_list, rewards, dones, infos = [], [], [], []
        for conn in self.parents:
            tag, obs, reward, done, truncated, info = conn.recv()
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (np.stack(obs_list), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32), infos)

    def set_opponent_snapshot(self, snap_path: Optional[str], worker_indices=None):
        """Set opponent snapshot for specified workers (or all)."""
        indices = worker_indices if worker_indices is not None else range(
            self.num_envs)
        for i in indices:
            self.parents[i].send(('set_opponent_snapshot', snap_path))
        for i in indices:
            self.parents[i].recv()

    def close(self):
        for conn in self.parents:
            try:
                conn.send(('close', None))
                conn.recv()
            except (EOFError, BrokenPipeError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()


# ── AsyncUpdater ────────────────────────────────────────────────────────────

class AsyncUpdater:
    """Background GPU thread for sequential gradient updates.

    Main thread calls trigger(agent) when buffer is full.
    GPU thread runs agent.update() in the background.
    Multiple agents queue up — processed FIFO, one at a time.
    """

    def __init__(self):
        self._queue: Queue = Queue()
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._results: List = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def trigger(self, agent, agent_id: int = 0):
        """Queue an agent for GPU update."""
        self._queue.put((agent_id, agent))

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def pop_metrics(self) -> List[tuple]:
        """Drain completed (agent_id, metrics) pairs."""
        with self._lock:
            results = list(self._results)
            self._results.clear()
        return results

    def stop(self):
        self._stop.set()
        self._queue.put(None)  # unblock
        self._thread.join(timeout=10)

    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                break
            agent_id, agent = item
            self._busy.set()
            try:
                _t0 = time.perf_counter()
                metrics = agent.update()
                metrics['update_wall_time'] = time.perf_counter() - _t0
                agent.buffer.reset()
                agent._buffer_ready.set()
                with self._lock:
                    self._results.append((agent_id, metrics))
            except Exception as e:
                print(f'[updater] Agent {agent_id} update failed: {e}',
                      file=sys.stderr)
                agent._buffer_ready.set()  # unblock collection even on error
            finally:
                self._busy.clear()


# ── metrics providers ───────────────────────────────────────────────────────

class RolloutMetricsProvider:
    """Tracks episode returns and lengths."""

    def __init__(self):
        self.episode_returns: deque = deque(maxlen=200)
        self.episode_lengths: deque = deque(maxlen=200)
        self.goals_scored: int = 0
        self.goals_conceded: int = 0

    def on_episode_end(self, episode_return: float, episode_length: int, goal: int):
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        if goal > 0:
            self.goals_scored += 1
        elif goal < 0:
            self.goals_conceded += 1

    def get_metrics(self) -> dict:
        if not self.episode_returns:
            return {}
        return {
            'mean_return': float(np.mean(self.episode_returns)),
            'mean_length': float(np.mean(self.episode_lengths)),
            'goals_scored': self.goals_scored,
            'goals_conceded': self.goals_conceded,
        }


class TimingMetricsProvider:
    """Tracks training speed."""

    def __init__(self):
        self._start = time.time()
        self._steps = 0

    def add_steps(self, n: int):
        self._steps += n

    def get_metrics(self) -> dict:
        elapsed = time.time() - self._start
        return {
            'steps_per_sec': self._steps / max(elapsed, 1e-6),
            'wall_time_hours': elapsed / 3600,
        }


# ── collection profiler ─────────────────────────────────────────────────────

class CollectionProfiler:
    """Per-collection-round wall-clock breakdown and transition counters.

    Follows the MetricsRegistry provider contract: get_metrics() -> dict.
    Uses time.perf_counter() for sub-microsecond monotonic timing.
    """

    def __init__(self):
        self._timers: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        self._history: deque = deque(maxlen=100)
        self._round_start: float = 0.0

    def start_round(self):
        """Call at the beginning of each collection round."""
        self._timers.clear()
        self._counters.clear()
        self._round_start = time.perf_counter()

    def add_time(self, name: str, elapsed: float):
        """Accumulate wall-clock seconds for a named category."""
        self._timers[name] = self._timers.get(name, 0.0) + elapsed

    def incr(self, name: str, n: int = 1):
        """Increment a named counter."""
        self._counters[name] = self._counters.get(name, 0) + n

    def end_round(self):
        """Call at the end of each collection round. Snapshots current data."""
        total = time.perf_counter() - self._round_start
        snapshot = {**self._timers, 'round_total_time': total, **self._counters}
        self._history.append(snapshot)

    def get_metrics(self) -> dict:
        """Return latest round's breakdown with absolute times and percentages."""
        if not self._history:
            return {}
        latest = self._history[-1]
        total = latest.get('round_total_time', 1e-9)
        metrics: Dict[str, Any] = {}
        for k, v in latest.items():
            if k.endswith('_time'):
                metrics[k] = v
                metrics[k + '_pct'] = 100.0 * v / total
            else:
                metrics[k] = v
        return metrics

    def generate_report(self, config: dict,
                        update_times: Optional[List[tuple]] = None) -> str:
        """Generate a formatted profiling report from collected history.

        Parameters
        ----------
        config : dict
            Training config (for system/hyperparameter context).
        update_times : list of (agent_id, wall_time) tuples, optional
            GPU update wall times collected during training.

        Returns
        -------
        str
            Markdown-formatted report suitable for pasting into a conversation.
        """
        import os

        if not self._history:
            return '# Profiling Report\n\nNo data collected.\n'

        history = list(self._history)

        # Collect all timer and counter keys across rounds
        timer_keys = sorted({k for snap in history for k in snap
                             if k.endswith('_time') and k != 'round_total_time'})
        counter_keys = sorted({k for snap in history for k in snap
                               if not k.endswith('_time')
                               and k != 'round_total_time'})

        def _stats(values):
            arr = np.array(values, dtype=np.float64)
            return float(np.mean(arr)), float(np.std(arr))

        lines: List[str] = []
        lines.append('# Training Performance Profile')
        lines.append('')

        # System info
        device = config.get('device', 'cpu')
        cuda_avail = torch.cuda.is_available()
        gpu_name = ''
        if cuda_avail and torch.cuda.device_count() > 0:
            gpu_name = torch.cuda.get_device_name(0)
        lines.append('## System')
        lines.append(f'- Device: `{device}`')
        lines.append(f'- CUDA available: {cuda_avail}')
        if gpu_name:
            lines.append(f'- GPU: {gpu_name}')
        lines.append(f'- CPU count: {os.cpu_count()}')
        lines.append(f'- Rounds collected: {len(history)}')
        lines.append('')

        # Config summary
        pop_cfg = config.get('population', {})
        algo_params = config.get('algorithm', {}).get('params', {})
        lines.append('## Config')
        lines.append(f'- Agents: {pop_cfg.get("agents", 1)}')
        lines.append(f'- Envs: {config.get("num_envs", 8)}')
        lines.append(f'- rollout_steps: {algo_params.get("rollout_steps", 2048)}')
        lines.append(f'- minibatch_size: {algo_params.get("minibatch_size", "N/A")}')
        lines.append(f'- ppo_epochs: {algo_params.get("ppo_epochs", "N/A")}')
        lines.append(f'- t_window: {config.get("t_window", 8)}')
        lines.append('')

        # Time breakdown table
        lines.append('## Per-Round Time Breakdown')
        lines.append('')
        lines.append('| Category | Mean (s) | Std (s) | Mean % |')
        lines.append('|----------|----------|---------|--------|')

        round_totals = [s.get('round_total_time', 0) for s in history]
        rt_mean, rt_std = _stats(round_totals)

        for key in timer_keys:
            vals = [s.get(key, 0.0) for s in history]
            mean, std = _stats(vals)
            pct = 100.0 * mean / rt_mean if rt_mean > 0 else 0
            label = key.replace('_time', '').replace('_', ' ')
            lines.append(f'| {label} | {mean:.4f} | {std:.4f} | {pct:.1f}% |')

        lines.append(f'| **round total** | **{rt_mean:.4f}** | **{rt_std:.4f}** | **100%** |')
        lines.append('')

        # Throughput
        if rt_mean > 0:
            rollout_steps = algo_params.get('rollout_steps', 2048)
            num_envs = config.get('num_envs', 8)
            steps_per_round = rollout_steps * num_envs
            sps = steps_per_round / rt_mean
            lines.append(f'## Throughput')
            lines.append(f'- Steps per round: {steps_per_round:,}')
            lines.append(f'- **Steps/sec: {sps:.1f}**')
            lines.append(f'- Projected steps/hour: {sps * 3600:,.0f}')
            lines.append('')

        # Counters
        lines.append('## Transition Stats (per round)')
        lines.append('')
        lines.append('| Counter | Mean | Std |')
        lines.append('|---------|------|-----|')
        for key in counter_keys:
            vals = [float(s.get(key, 0)) for s in history]
            mean, std = _stats(vals)
            label = key.replace('_', ' ')
            lines.append(f'| {label} | {mean:.1f} | {std:.1f} |')

        # Waste rate
        collected = [float(s.get('transitions_collected', 0)) for s in history]
        discarded = [float(s.get('transitions_discarded', 0)) for s in history]
        total_trans = [c + d for c, d in zip(collected, discarded)]
        waste_rates = [d / t if t > 0 else 0 for d, t in zip(discarded, total_trans)]
        if waste_rates:
            wr_mean, wr_std = _stats(waste_rates)
            lines.append(f'| **waste rate** | **{wr_mean*100:.1f}%** | {wr_std*100:.1f}% |')
        lines.append('')

        # Update times — aggregate per agent
        if update_times:
            agent_times: Dict[int, List[float]] = {}
            for agent_id, wt in update_times:
                agent_times.setdefault(agent_id, []).append(wt)
            lines.append('## GPU Update Wall Times')
            lines.append('')
            lines.append('| Agent | Count | Mean (s) | Std (s) | Min (s) | Max (s) |')
            lines.append('|-------|-------|----------|---------|---------|---------|')
            for aid in sorted(agent_times):
                vals = agent_times[aid]
                mean, std = _stats(vals)
                lines.append(
                    f'| agent_{aid} | {len(vals)} '
                    f'| {mean:.3f} | {std:.3f} '
                    f'| {min(vals):.3f} | {max(vals):.3f} |')
            lines.append('')

        return '\n'.join(lines)


# ── collect and train ───────────────────────────────────────────────────────

def collect_and_train(
    population,
    envs: SubprocVecEnv,
    updater: AsyncUpdater,
    rollout_metrics: RolloutMetricsProvider,
    config: dict,
    scheduler,
    profiler: Optional[CollectionProfiler] = None,
) -> int:
    """Collect rollouts and trigger GPU updates. Algorithm-agnostic.

    The scheduler controls which agents use which envs each step.
    Returns total environment steps collected.
    """
    from training.abstractions import ActionResult

    num_envs = envs.num_envs
    agents = population.agents
    rollout_steps = config.get('algorithm', {}).get(
        'params', {}).get('rollout_steps', 2048)

    # If all agents are mid-update, yield briefly rather than hot-spinning
    _t0 = time.perf_counter()
    while not any(a._buffer_ready.is_set() for a in agents):
        time.sleep(0.005)
    if profiler:
        profiler.add_time('idle_time', time.perf_counter() - _t0)

    _t0 = time.perf_counter()
    obs = envs.reset_all()  # (num_envs, obs_dim)
    if profiler:
        profiler.add_time('env_reset_time', time.perf_counter() - _t0)

    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)
    total_steps = 0
    prev_agents = None

    scheduler.on_round_start()

    for agent_workers in scheduler.iter_steps(rollout_steps):
        # Detect agent rotation (for serial scheduler) — reset envs
        current_agents = frozenset(agent_workers.keys())
        if prev_agents is not None and current_agents != prev_agents:
            _t0 = time.perf_counter()
            obs = envs.reset_all()
            episode_returns[:] = 0.0
            episode_lengths[:] = 0
            if profiler:
                profiler.add_time('env_reset_time', time.perf_counter() - _t0)
        prev_agents = current_agents

        # Action selection — only active agents this step
        actions = np.zeros((num_envs, 8), dtype=np.float32)
        agent_results: Dict[int, Any] = {}

        _t0 = time.perf_counter()
        for agent_idx, worker_ids in agent_workers.items():
            result = agents[agent_idx].select_action(obs[worker_ids])
            for local_i, wi in enumerate(worker_ids):
                actions[wi] = result.action[local_i]
            agent_results[agent_idx] = result
        if profiler:
            profiler.add_time('action_select_time', time.perf_counter() - _t0)

        # Step all envs
        _t0 = time.perf_counter()
        next_obs, rewards, dones, infos = envs.step(actions)
        if profiler:
            profiler.add_time('env_step_time', time.perf_counter() - _t0)

        # Store transitions — skip agents whose buffer is being updated
        _t0 = time.perf_counter()
        for agent_idx, worker_ids in agent_workers.items():
            agent = agents[agent_idx]
            if not agent._buffer_ready.is_set():
                if profiler:
                    profiler.incr('transitions_discarded', len(worker_ids))
                continue
            result = agent_results[agent_idx]
            agent.store_transition(
                obs[worker_ids], result,
                rewards[worker_ids], next_obs[worker_ids],
                dones[worker_ids], {})
            if profiler:
                profiler.incr('transitions_collected', len(worker_ids))
        if profiler:
            profiler.add_time('store_transition_time', time.perf_counter() - _t0)

        # Track episodes — credit the agent controlling each worker
        for wi in range(num_envs):
            episode_returns[wi] += rewards[wi]
            episode_lengths[wi] += 1
            if dones[wi]:
                goal = infos[wi].get('goal', 0)
                rollout_metrics.on_episode_end(
                    episode_returns[wi], int(episode_lengths[wi]), goal)
                for ai, wids in agent_workers.items():
                    if wi in wids:
                        population.add_score(ai, float(goal))
                        break
                episode_returns[wi] = 0.0
                episode_lengths[wi] = 0
                if profiler:
                    profiler.incr('episodes_completed')

        obs = next_obs
        total_steps += num_envs

        # Trigger updates for full buffers
        for agent_idx in agent_workers:
            agent = agents[agent_idx]
            if agent.should_update():
                agent._buffer_ready.clear()
                updater.trigger(agent, agent_idx)

    return total_steps


# ── main training loop ──────────────────────────────────────────────────────

def train(config: dict):
    """Main training entry point. Fully algorithm-agnostic."""
    seed = config.get('seed', 0)
    set_seed(seed)

    total_steps = config['total_steps']
    num_envs = config['num_envs']
    t_window = config.get('t_window', 8)
    reward_type = config.get('reward_type', 'sparse')
    model_dir = Path(config.get(
        'model_dir', 'models/baseline')) / f'seed_{seed}'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_interval = config.get('log_interval', 10)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[train] seed={seed} device={device} total_steps={total_steps:}')

    # ── resolve logger ──────────────────────────────────────────────────
    LoggerCls = resolve_or_default(config, 'logger', None)
    if LoggerCls is None:
        from training.loggers.wandb import WandbLogger
        LoggerCls = WandbLogger
    logger = LoggerCls()
    logger.init(config)

    from training.loggers.registry import MetricsRegistry
    registry = MetricsRegistry()

    # ── resolve reward function ─────────────────────────────────────────
    RewardCls = resolve_or_default(config, 'reward', None)
    if RewardCls is not None:
        reward_fn = RewardCls()
    else:
        reward_fn = None  # env handles reward internally

    # ── resolve replay / feedback providers ─────────────────────────────
    ReplayCls = resolve_or_default(config, 'replay', NullReplayProvider)
    replay_provider = ReplayCls()
    FeedbackCls = resolve_or_default(config, 'feedback', NullFeedbackProvider)
    feedback_provider = FeedbackCls()

    # ── resolve collection scheduler ───────────────────────────────────
    from training.schedulers import InterleavedScheduler
    SchedulerCls = resolve_or_default(config, 'scheduler', InterleavedScheduler)
    scheduler = SchedulerCls()

    # ── create population ───────────────────────────────────────────────
    from training.algorithms.ppo import Population
    pop_config = config.get('population', {})
    num_agents = pop_config.get('agents', 1)
    agent_envs = scheduler.envs_per_agent(num_envs, num_agents)
    population = Population(num_agents=num_agents, num_workers=num_envs,
                            config={**config, 'device': device},
                            envs_per_agent=agent_envs)
    scheduler.init(population, num_envs, config)

    # ── create opponent pool ────────────────────────────────────────────
    PoolCls = resolve_or_default(config, 'opponent_pool', None)
    pool = None
    if PoolCls is not None:
        pool_params = config.get('opponent_pool', {}).get('params', {})
        snapshot_dir = config.get('snapshot_dir', 'models/snapshots')
        pool = PoolCls(snapshot_dir=snapshot_dir, **pool_params)

    # ── create vectorized envs ──────────────────────────────────────────
    dense_weights = config.get('dense_reward_weights', None)
    env_class = config.get('env_class', None)
    envs = SubprocVecEnv(
        num_envs=num_envs,
        t_window=t_window,
        reward_type=reward_type,
        dense_reward_weights=dense_weights,
        env_class=env_class,
    )

    # ── seed from replay data ───────────────────────────────────────────
    demos = replay_provider.load_demonstrations()
    if demos:
        for agent in population.agents:
            replay_provider.seed_algorithm(agent, demos)

    # ── register metrics providers ──────────────────────────────────────
    rollout_metrics = RolloutMetricsProvider()
    timing_metrics = TimingMetricsProvider()
    profiler = CollectionProfiler()
    registry.register('rollout', rollout_metrics.get_metrics)
    registry.register('timing', timing_metrics.get_metrics)
    registry.register('perf', profiler.get_metrics)
    if pool:
        registry.register('opponent_pool', pool.get_metrics)
    registry.register('population', population.get_metrics)

    # ── profiling config ──────────────────────────────────────────────
    profiling_cfg = config.get('profiling', {})
    profiling_enabled = profiling_cfg.get('enabled', False)
    profiling_report_path = profiling_cfg.get('report', None)
    all_update_times: List[tuple] = []  # (agent_id, wall_time) for report

    # ── main loop ───────────────────────────────────────────────────────
    updater = AsyncUpdater()
    total_collected = 0
    collection_round = 0
    gen_steps = pop_config.get('generation_steps', 1_000_000)
    noise_scale = pop_config.get('generation_noise_scale', 0.01)
    last_gen_step = 0

    print(f'[train] {num_agents} agent(s), {num_envs} envs, '
          f'rollout_steps={config.get("algorithm", {}).get("params", {}).get("rollout_steps", 2048)}')

    from tqdm import tqdm
    pbar = tqdm(total=total_steps, unit='step', dynamic_ncols=True)
    try:
        while total_collected < total_steps:
            profiler.start_round()

            # Set opponents from pool
            if pool and pool.num_snapshots() > 0:
                snap_path = pool.sample_opponent()
                if snap_path:
                    _t0 = time.perf_counter()
                    envs.set_opponent_snapshot(snap_path)
                    profiler.add_time('opponent_load_time', time.perf_counter() - _t0)

            # Collect rollouts and trigger updates (non-blocking; per-agent
            # _buffer_ready gates writes while updates are in flight)
            steps = collect_and_train(
                population, envs, updater, rollout_metrics, config,
                scheduler, profiler=profiler)
            profiler.end_round()
            total_collected += steps
            timing_metrics.add_steps(steps)
            collection_round += 1
            pbar.update(steps)

            # Drain update metrics and log
            update_results = updater.pop_metrics()
            for agent_id, metrics in update_results:
                prefixed = {f'agent_{agent_id}/{k}': v for k,
                            v in metrics.items()}
                logger.log(total_collected, **prefixed)
                if profiling_enabled and 'update_wall_time' in metrics:
                    all_update_times.append(
                        (agent_id, metrics['update_wall_time']))

            # Periodic logging
            if collection_round % log_interval == 0:
                custom = registry.collect()
                logger.log(total_collected,
                           total_steps=total_collected,
                           **custom)
                rm = rollout_metrics.get_metrics()
                mean_ret = rm.get('mean_return', 0.0)
                sps = timing_metrics.get_metrics().get('steps_per_sec', 0.0)
                pbar.set_postfix(
                    ret=f'{mean_ret:.3f}',
                    sps=f'{sps:.0f}',
                    scored=rollout_metrics.goals_scored,
                    conceded=rollout_metrics.goals_conceded,
                )

            # Save snapshots to opponent pool
            if pool:
                best_idx = population.rank_agents()[0]
                best_agent = population.agents[best_idx]
                if pool.should_swap(total_collected):
                    pool.save_snapshot(best_agent, total_collected)

            # Population generation cycle — wait for all updates before mutating weights
            if num_agents > 1 and total_collected - last_gen_step >= gen_steps:
                for a in population.agents:
                    a._buffer_ready.wait()  # blocks until this agent's update finishes
                # Log per-agent stats before scores are cleared
                gen_metrics = {f'population/{k}': v
                               for k, v in population.get_metrics().items()}
                logger.log(total_collected, **gen_metrics)
                ranked = population.rank_agents()
                best = population.agents[ranked[0]]
                # Reset worst from best + noise
                if len(ranked) > 1:
                    worst_idx = ranked[-1]
                    population.agents[worst_idx].clone_from(
                        best, noise_scale=noise_scale)
                    print(f'[gen {population.generation}] '
                          f'best=agent_{ranked[0]}  worst=agent_{worst_idx} (reset)')
                population.reset_scores()
                last_gen_step = total_collected

            # Checkpoint
            if collection_round % (log_interval * 10) == 0:
                best_idx = population.rank_agents()[0]
                population.agents[best_idx].save_checkpoint(
                    model_dir / 'latest')

    except KeyboardInterrupt:
        print('\n[train] Interrupted.')
    finally:
        pbar.close()
        updater.stop()
        # Save final checkpoint
        best_idx = population.rank_agents()[0]
        population.agents[best_idx].save_checkpoint(model_dir / 'final')
        print(f'[train] Saved final checkpoint to {model_dir}/final')

        # Generate profiling report if enabled
        if profiling_enabled:
            report = profiler.generate_report(
                {**config, 'device': device},
                update_times=all_update_times or None)
            if profiling_report_path:
                report_path = Path(profiling_report_path)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(report)
                print(f'[train] Profiling report saved to {report_path}')
            else:
                print(report)

        logger.finish()
        envs.close()

    print(f'[train] Done. {total_collected:,} total steps.')


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='YAML-configured RL training')
    parser.add_argument('--config', default=None,
                        help='YAML config file path')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--total-steps', type=int, default=None)
    parser.add_argument('--num-envs', type=int, default=None)
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable W&B logging')
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config, cli_overrides={
            'seed': args.seed,
            'total_steps': args.total_steps,
            'num_envs': args.num_envs,
        })
    else:
        # No YAML — use universal defaults with PPO
        config = dict(UNIVERSAL_DEFAULTS)
        config['algorithm'] = {
            'class': 'training.algorithms.ppo.PPOAlgorithm',
            'params': {},
        }
        config['opponent_pool'] = {
            'class': 'training.opponents.pool.HistoricalOpponentPool',
            'params': {},
        }
        # Resolve
        from training.algorithms.ppo import PPOAlgorithm
        from training.opponents.pool import HistoricalOpponentPool
        config['algorithm']['cls'] = PPOAlgorithm
        config['algorithm']['params'] = PPOAlgorithm.default_params()
        config['opponent_pool']['cls'] = HistoricalOpponentPool
        config['opponent_pool']['params'] = HistoricalOpponentPool.default_params()
        # Apply CLI overrides
        if args.seed is not None:
            config['seed'] = args.seed
        if args.total_steps is not None:
            config['total_steps'] = args.total_steps
        if args.num_envs is not None:
            config['num_envs'] = args.num_envs

    if args.no_wandb:
        config.setdefault('logger', {})['params'] = {'enabled': False}

    train(config)


if __name__ == '__main__':
    main()
