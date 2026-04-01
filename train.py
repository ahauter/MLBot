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
import os
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
    'envs_per_worker': 1,
    'reward_type': 'sparse',
    'model_dir': 'models/baseline',
    'log_interval': 10,
    'snapshot_dir': 'models/snapshots',
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
                env_class: Optional[str] = None,
                envs_per_worker: int = 1):
    """Child process: owns one or more gym envs, responds to commands.

    When envs_per_worker > 1, multiple rlgym-sim arenas run in a single
    process.  Opponent inference is batched across all arenas so the
    encoder+policy forward pass runs once at batch=N instead of N times
    at batch=1.
    """
    if env_class:
        EnvCls = load_class(env_class)
    else:
        from training.environments.baseline_env import BaselineGymEnv
        EnvCls = BaselineGymEnv

    n = envs_per_worker
    envs = [
        EnvCls(
            t_window=t_window,
            reward_type=reward_type,
            dense_reward_weights=dense_reward_weights,
        )
        for _ in range(n)
    ]

    # Shared opponent state for PPO snapshots (one model for all envs)
    _opponent_encoder = None
    _opponent_policy = None
    _torch = None
    _EIDS = None

    while True:
        try:
            cmd, data = conn.recv()
        except (EOFError, BrokenPipeError):
            break

        if cmd == 'reset':
            obs_list = []
            for env in envs:
                obs, _info = env.reset()
                obs_list.append(obs)
            conn.send(('obs', np.stack(obs_list, axis=0)))

        elif cmd == 'step':
            # data is (n, 8) actions
            actions = data

            # Pre-compute all opponent actions in one batched forward pass
            if _opponent_encoder is not None and n > 1:
                flat_list = []
                for e in envs:
                    stacked = np.stack(list(e._orange_buf), axis=0)
                    flat_list.append(stacked.ravel().astype(np.float32))
                batch = np.stack(flat_list, axis=0)  # (n, T*N*F)

                with _torch.no_grad():
                    x = _torch.tensor(batch, dtype=_torch.float32)
                    tokens = x.view(
                        n, envs[0].t_window, -1, TOKEN_FEATURES)
                    eids = _torch.tensor(_EIDS, dtype=_torch.long)
                    emb = _opponent_encoder(tokens, eids)
                    opp_actions, _ = _opponent_policy.act_deterministic(emb)
                opp_np = opp_actions.cpu().numpy().astype(np.float32)
                for i, e in enumerate(envs):
                    e._cached_opp_action = opp_np[i]

            obs_list, rewards, dones, infos = [], [], [], []
            for i, env in enumerate(envs):
                obs, reward, done, truncated, info = env.step(actions[i])
                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            conn.send((
                'step',
                np.stack(obs_list, axis=0),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                infos,
            ))

        elif cmd == 'get_opponent_obs':
            obs_list = []
            for env in envs:
                obs_list.append(env.get_opponent_obs())
            conn.send(('opponent_obs', np.stack(obs_list, axis=0)))

        elif cmd == 'step_with_opp':
            blue_actions, opp_actions = data  # (n, 8) each
            obs_list, rewards, dones, infos = [], [], [], []
            for i, env in enumerate(envs):
                obs, reward, done, truncated, info = env.step_with_opponent_action(
                    blue_actions[i], opp_actions[i])
                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            conn.send((
                'step',
                np.stack(obs_list, axis=0),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                infos,
            ))

        elif cmd == 'set_opponent_snapshot':
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

                # Import torch/constants once for use in step handler
                if _torch is None:
                    import torch as _t
                    from encoder import ENTITY_TYPE_IDS_1V1 as _e
                    _torch = _t
                    _EIDS = _e

                if n > 1:
                    # Multi-env: monkey-patch to read cached action
                    # (computed in the step handler above)
                    import types

                    def _cached_opponent_action(self_env):
                        return self_env._cached_opp_action

                    for env in envs:
                        env._cached_opp_action = np.zeros(8, dtype=np.float32)
                        env._get_opponent_action = types.MethodType(
                            _cached_opponent_action, env)
                else:
                    # Single env: inline forward pass (no caching needed)
                    import types

                    def _single_opponent_action(self_env):
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

                    envs[0]._get_opponent_action = types.MethodType(
                        _single_opponent_action, envs[0])
            conn.send(('ok',))

        elif cmd == 'close':
            for env in envs:
                env.close()
            conn.send(('closed',))
            break

        else:
            conn.send(('error', f'Unknown command: {cmd}'))


class SubprocVecEnv:
    """Vectorized environment using subprocesses.

    Each worker process can manage multiple rlgym-sim arenas
    (controlled by ``envs_per_worker``).  This reduces IPC overhead
    and enables batched opponent inference within each worker.

    Parameters
    ----------
    num_envs : int
        Number of worker processes to spawn.
    envs_per_worker : int
        Number of arenas per subprocess.  Default 1 preserves the
        original one-process-per-env behaviour.  Total logical
        environments = ``num_envs * envs_per_worker``.
    """

    def __init__(self, num_envs: int, t_window: int = 8,
                 reward_type: str = 'sparse',
                 dense_reward_weights: Optional[dict] = None,
                 env_class: Optional[str] = None,
                 envs_per_worker: int = 1):
        self.num_workers = num_envs
        self.envs_per_worker = envs_per_worker
        self.num_envs = num_envs * envs_per_worker  # total logical envs
        self.parents: List[multiprocessing.connection.Connection] = []
        self.procs: List[multiprocessing.Process] = []

        for i in range(self.num_workers):
            parent_conn, child_conn = multiprocessing.Pipe()
            proc = multiprocessing.Process(
                target=_env_worker,
                args=(child_conn, t_window, reward_type, dense_reward_weights,
                      env_class, envs_per_worker),
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
            tag, obs_batch = conn.recv()
            obs_list.append(obs_batch)  # (envs_per_worker, obs_dim)
        return np.concatenate(obs_list, axis=0)  # (num_envs, obs_dim)

    def step(self, actions: np.ndarray):
        """Step all envs. Returns (obs, rewards, dones, infos)."""
        epw = self.envs_per_worker
        for i, conn in enumerate(self.parents):
            conn.send(('step', actions[i * epw:(i + 1) * epw]))
        obs_list, rew_list, done_list, infos = [], [], [], []
        for conn in self.parents:
            tag, obs_batch, rew_batch, done_batch, info_batch = conn.recv()
            obs_list.append(obs_batch)
            rew_list.append(rew_batch)
            done_list.append(done_batch)
            infos.extend(info_batch)
        return (np.concatenate(obs_list, axis=0),
                np.concatenate(rew_list, axis=0),
                np.concatenate(done_list, axis=0),
                infos)

    def get_opponent_obs(self) -> np.ndarray:
        """Get stacked orange observations from all envs. Returns (num_envs, obs_dim)."""
        for conn in self.parents:
            conn.send(('get_opponent_obs', None))
        obs_list = []
        for conn in self.parents:
            tag, obs_batch = conn.recv()
            obs_list.append(obs_batch)  # (envs_per_worker, obs_dim)
        return np.concatenate(obs_list, axis=0)  # (num_envs, obs_dim)

    def step_with_opponent_actions(self, blue_actions: np.ndarray,
                                   opp_actions: np.ndarray):
        """Step all envs with pre-computed opponent actions. Same return as step()."""
        epw = self.envs_per_worker
        for i, conn in enumerate(self.parents):
            conn.send(('step_with_opp', (
                blue_actions[i * epw:(i + 1) * epw],
                opp_actions[i * epw:(i + 1) * epw],
            )))
        obs_list, rew_list, done_list, infos = [], [], [], []
        for conn in self.parents:
            tag, obs_batch, rew_batch, done_batch, info_batch = conn.recv()
            obs_list.append(obs_batch)
            rew_list.append(rew_batch)
            done_list.append(done_batch)
            infos.extend(info_batch)
        return (np.concatenate(obs_list, axis=0),
                np.concatenate(rew_list, axis=0),
                np.concatenate(done_list, axis=0),
                infos)

    def set_opponent_snapshot(self, snap_path: Optional[str], worker_indices=None):
        """Set opponent snapshot for specified workers (or all).

        ``worker_indices`` are *env* indices (0..num_envs-1). The
        corresponding worker processes are deduced automatically.
        """
        if worker_indices is None:
            worker_ids = set(range(self.num_workers))
        else:
            worker_ids = {i // self.envs_per_worker for i in worker_indices}
        for wi in worker_ids:
            self.parents[wi].send(('set_opponent_snapshot', snap_path))
        for wi in worker_ids:
            self.parents[wi].recv()

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

    def __init__(self, profiler: Optional['CollectionProfiler'] = None):
        self._queue: Queue = Queue()
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._results: List = []
        self._lock = threading.Lock()
        self._profiler = profiler
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
                _t1 = time.perf_counter()
                metrics['update_wall_time'] = _t1 - _t0
                # Record GPU timeline event if profiler is attached
                if self._profiler is not None:
                    self._profiler.record_event(
                        _t0, _t1, 'gpu_update', thread='gpu',
                        agent_id=agent_id)
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
    """Training loop profiler with per-round timing, counters, and waterfall chart.

    Follows the MetricsRegistry provider contract: get_metrics() -> dict.
    Uses time.perf_counter() for sub-microsecond monotonic timing.

    Every phase of the training loop has a canonical timing key. The waterfall
    records absolute start/end timestamps for the first N rounds so a
    matplotlib chart can show main-thread vs GPU-thread concurrency.
    """

    # Canonical timing categories — defines report ordering.
    TIMING_KEYS = [
        'idle_time',              # waiting for agent buffers to free up
        'env_reset_time',         # env resets (including agent rotation resets)
        'action_select_time',     # GPU forward passes for action selection
        'env_step_time',          # physics sim + subprocess IPC
        'store_transition_time',  # writing transitions to rollout buffer
        'episode_tracking_time',  # episode return/length tracking + scoring
        'opponent_load_time',     # loading opponent snapshot into workers
        'metrics_drain_time',     # draining AsyncUpdater results + per-agent logging
        'logging_time',           # registry.collect() + logger.log() + tqdm
        'snapshot_save_time',     # saving snapshots to opponent pool
        'generation_time',        # PBT generation cycle (wait/rank/clone/reset)
        'checkpoint_time',        # saving model checkpoints
        'eval_time',              # inline evaluation episodes
    ]

    COUNTER_KEYS = [
        'transitions_collected',
        'transitions_discarded',
        'episodes_completed',
    ]

    def __init__(self, waterfall_rounds: int = 0):
        self._timers: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        self._history: deque = deque(maxlen=100)
        self._round_start: float = 0.0
        self._round_count: int = 0
        # Waterfall: list of (start, end, category, thread, agent_id) tuples
        self._waterfall_rounds = waterfall_rounds
        self._timeline: List[tuple] = []
        self._timeline_lock = threading.Lock()

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

    def record_event(self, start: float, end: float, category: str,
                     thread: str = 'main', agent_id: int = -1):
        """Record an absolute-time event for the waterfall chart.

        Only records during the first waterfall_rounds rounds.
        Thread-safe (called from both main and GPU threads).
        """
        if self._round_count < self._waterfall_rounds:
            with self._timeline_lock:
                self._timeline.append(
                    (start, end, category, thread, agent_id))

    def end_round(self):
        """Call at the end of each collection round. Snapshots current data."""
        total = time.perf_counter() - self._round_start
        snapshot = {**self._timers, 'round_total_time': total, **self._counters}
        self._history.append(snapshot)
        self._round_count += 1

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
                        update_times: Optional[List[tuple]] = None,
                        waterfall_path: Optional[str] = None) -> str:
        """Generate a formatted profiling report from collected history.

        Parameters
        ----------
        config : dict
            Training config (for system/hyperparameter context).
        update_times : list of (agent_id, wall_time) tuples, optional
            GPU update wall times collected during training.
        waterfall_path : str, optional
            If set, generate a waterfall PNG at this path and embed in report.
        """
        import os

        if not self._history:
            return '# Profiling Report\n\nNo data collected.\n'

        history = list(self._history)

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
        sched_cfg = config.get('scheduler', {})
        lines.append('## Config')
        lines.append(f'- Agents: {pop_cfg.get("agents", 1)}')
        _ne = config.get('num_envs', 8)
        _epw = config.get('envs_per_worker', 1)
        lines.append(f'- Workers: {_ne}, envs_per_worker: {_epw}, total_envs: {_ne * _epw}')
        lines.append(f'- Scheduler: {sched_cfg.get("class", "InterleavedScheduler")}')
        lines.append(f'- rollout_steps: {algo_params.get("rollout_steps", 2048)}')
        lines.append(f'- minibatch_size: {algo_params.get("minibatch_size", "N/A")}')
        lines.append(f'- ppo_epochs: {algo_params.get("ppo_epochs", "N/A")}')
        lines.append(f'- t_window: {config.get("t_window", 8)}')
        lines.append('')

        # Time breakdown table — use canonical order
        lines.append('## Per-Round Time Breakdown')
        lines.append('')
        lines.append('| Category | Mean (s) | Std (s) | Mean % |')
        lines.append('|----------|----------|---------|--------|')

        round_totals = [s.get('round_total_time', 0) for s in history]
        rt_mean, rt_std = _stats(round_totals)

        # Canonical keys first, then any extras
        seen_keys = set()
        all_timer_keys = list(self.TIMING_KEYS)
        for snap in history:
            for k in snap:
                if k.endswith('_time') and k != 'round_total_time' \
                        and k not in seen_keys:
                    if k not in all_timer_keys:
                        all_timer_keys.append(k)
                    seen_keys.add(k)

        for key in all_timer_keys:
            vals = [s.get(key, 0.0) for s in history]
            mean, std = _stats(vals)
            if mean < 0.00005 and key in self.TIMING_KEYS:
                continue  # skip canonical keys that are exactly zero
            pct = 100.0 * mean / rt_mean if rt_mean > 0 else 0
            label = key.replace('_time', '').replace('_', ' ')
            lines.append(f'| {label} | {mean:.4f} | {std:.4f} | {pct:.1f}% |')

        lines.append(f'| **round total** | **{rt_mean:.4f}** | **{rt_std:.4f}** | **100%** |')
        lines.append('')

        # Throughput
        if rt_mean > 0:
            collected_vals = [s.get('transitions_collected', 0) for s in history]
            if any(c > 0 for c in collected_vals):
                steps_per_round = int(np.mean(collected_vals))
            else:
                rollout_steps = algo_params.get('rollout_steps', 2048)
                _total = config.get('num_envs', 8) * config.get('envs_per_worker', 1)
                steps_per_round = rollout_steps * _total
            sps = steps_per_round / rt_mean
            lines.append(f'## Throughput')
            lines.append(f'- Steps per round: {steps_per_round:,}')
            lines.append(f'- **Steps/sec: {sps:.1f}**')
            lines.append(f'- Projected steps/hour: {sps * 3600:,.0f}')
            lines.append('')

        # Counters — canonical order, then extras
        lines.append('## Transition Stats (per round)')
        lines.append('')
        lines.append('| Counter | Mean | Std |')
        lines.append('|---------|------|-----|')
        seen_counters = set()
        all_counter_keys = list(self.COUNTER_KEYS)
        for snap in history:
            for k in snap:
                if not k.endswith('_time') and k != 'round_total_time' \
                        and k not in seen_counters:
                    if k not in all_counter_keys:
                        all_counter_keys.append(k)
                    seen_counters.add(k)

        for key in all_counter_keys:
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

        # Waterfall chart
        if waterfall_path and self._timeline:
            try:
                self._render_waterfall(waterfall_path)
                # Embed relative path in markdown
                wf_name = Path(waterfall_path).name
                lines.append('## Waterfall')
                lines.append('')
                lines.append(f'![Waterfall]({wf_name})')
                lines.append('')
            except Exception as e:
                lines.append(f'## Waterfall')
                lines.append('')
                lines.append(f'*(generation failed: {e})*')
                lines.append('')

        return '\n'.join(lines)

    def _render_waterfall(self, output_path: str) -> None:
        """Render a matplotlib waterfall chart of timeline events."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        events = list(self._timeline)
        if not events:
            return

        # Normalize times to start at 0
        t_origin = min(e[0] for e in events)
        events = [(s - t_origin, e - t_origin, cat, thr, aid)
                  for s, e, cat, thr, aid in events]

        # Color map — high contrast, visually distinct categories
        category_colors = {
            'idle': '#888888',
            'env_reset': '#17becf',
            'action_select': '#e6194b',
            'env_step': '#3cb44b',
            'store_transition': '#4363d8',
            'episode_tracking': '#f58231',
            'opponent_load': '#911eb4',
            'metrics_drain': '#42d4f4',
            'logging': '#f032e6',
            'snapshot_save': '#bfef45',
            'generation': '#fabed4',
            'checkpoint': '#ffe119',
            'gpu_update': '#ff6d00',
            'eval': '#dcbeff',
        }

        def _color(cat: str) -> str:
            key = cat.replace('_time', '')
            return category_colors.get(key, '#aaaaaa')

        # Assign y positions: main=1, gpu=0
        thread_y = {'main': 1.0, 'gpu': 0.0}

        fig, ax = plt.subplots(figsize=(16, 3.5))

        seen_cats = set()
        for start, end, cat, thread, agent_id in events:
            y = thread_y.get(thread, 0.5)
            duration = max(end - start, 0.001)  # avoid zero-width bars
            color = _color(cat)
            label_key = cat.replace('_time', '')
            ax.barh(y, duration, left=start, height=0.6,
                    color=color, edgecolor='white', linewidth=0.3)
            # Add agent label on GPU update bars
            if thread == 'gpu' and agent_id >= 0 and duration > 0.5:
                ax.text(start + duration / 2, y, f'a{agent_id}',
                        ha='center', va='center', fontsize=7, color='white',
                        fontweight='bold')
            seen_cats.add(label_key)

        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(['gpu', 'main'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Training Round Waterfall')
        ax.set_ylim(-0.5, 1.8)

        # Legend — only show categories that appeared
        patches = [mpatches.Patch(color=category_colors.get(c, '#aaa'), label=c)
                   for c in sorted(seen_cats) if c in category_colors]
        if patches:
            ax.legend(handles=patches, loc='upper right', fontsize=7,
                      ncol=min(len(patches), 4))

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# ── batched opponent inference ──────────────────────────────────────────────

@torch.no_grad()
def _opponent_inference(opp_obs: np.ndarray, encoder, policy,
                        device: str, t_window: int) -> np.ndarray:
    """Batched GPU forward pass for opponent actions.

    Parameters
    ----------
    opp_obs : (num_envs, obs_dim) flat orange observations
    encoder, policy : opponent's frozen encoder and policy on device
    device : torch device string
    t_window : number of stacked frames

    Returns
    -------
    (num_envs, 8) numpy array of opponent actions
    """
    from se3_field import SE3Encoder
    x = torch.tensor(opp_obs, dtype=torch.float32, device=device)
    if isinstance(encoder, SE3Encoder):
        emb = encoder(x)
    else:
        from encoder import ENTITY_TYPE_IDS_1V1
        tokens = x.view(x.shape[0], t_window, N_TOKENS, TOKEN_FEATURES)
        eids = torch.tensor(ENTITY_TYPE_IDS_1V1, dtype=torch.long, device=device)
        emb = encoder(tokens, eids)
    action, _ = policy.act_deterministic(emb)
    return action.cpu().numpy()


# ── collect and train ───────────────────────────────────────────────────────

def collect_and_train(
    population,
    envs: SubprocVecEnv,
    updater: AsyncUpdater,
    rollout_metrics: RolloutMetricsProvider,
    config: dict,
    scheduler,
    profiler: Optional[CollectionProfiler] = None,
    opponent_encoder=None,
    opponent_policy=None,
    opponent_loaded: bool = False,
) -> int:
    """Collect rollouts and trigger GPU updates. Algorithm-agnostic.

    The scheduler controls which agents use which envs each step.
    When opponent_loaded is True, opponent inference runs centrally
    on GPU via opponent_encoder/opponent_policy (batched).
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
    _t1 = time.perf_counter()
    if profiler:
        profiler.add_time('idle_time', _t1 - _t0)
        profiler.record_event(_t0, _t1, 'idle')

    _t0 = time.perf_counter()
    obs = envs.reset_all()  # (num_envs, obs_dim)
    _t1 = time.perf_counter()
    if profiler:
        profiler.add_time('env_reset_time', _t1 - _t0)
        profiler.record_event(_t0, _t1, 'env_reset')

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
            _t1 = time.perf_counter()
            if profiler:
                profiler.add_time('env_reset_time', _t1 - _t0)
                profiler.record_event(_t0, _t1, 'env_reset')
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
        _t1 = time.perf_counter()
        if profiler:
            profiler.add_time('action_select_time', _t1 - _t0)
            profiler.record_event(_t0, _t1, 'action_select')

        # Step all envs (with batched GPU opponent inference when available)
        _t0 = time.perf_counter()
        if opponent_loaded:
            opp_obs = envs.get_opponent_obs()
            opp_actions = _opponent_inference(
                opp_obs, opponent_encoder, opponent_policy,
                config.get('device', 'cpu'), config.get('t_window', 8))
            next_obs, rewards, dones, infos = envs.step_with_opponent_actions(
                actions, opp_actions)
        else:
            next_obs, rewards, dones, infos = envs.step(actions)
        _t1 = time.perf_counter()
        if profiler:
            profiler.add_time('env_step_time', _t1 - _t0)
            profiler.record_event(_t0, _t1, 'env_step')

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
        _t1 = time.perf_counter()
        if profiler:
            profiler.add_time('store_transition_time', _t1 - _t0)
            profiler.record_event(_t0, _t1, 'store_transition')

        # Track episodes — credit the agent controlling each worker
        _t0 = time.perf_counter()
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
        if profiler:
            profiler.add_time('episode_tracking_time',
                              time.perf_counter() - _t0)

        obs = next_obs
        total_steps += num_envs

        # Trigger updates for full buffers — gate on _buffer_ready to prevent
        # queuing multiple updates for the same agent while pos stays at capacity
        for agent_idx in agent_workers:
            agent = agents[agent_idx]
            if agent._buffer_ready.is_set() and agent.should_update():
                agent._buffer_ready.clear()
                updater.trigger(agent, agent_idx)

    return total_steps


# ── checkpoint save / load for resume ──────────────────────────────────────

def save_training_state(path: Path, population, total_collected: int,
                        collection_round: int, last_gen_step: int,
                        generation_pending: bool) -> None:
    """Save full training state (all agents + loop counters) atomically."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {
        'total_collected': total_collected,
        'collection_round': collection_round,
        'last_gen_step': last_gen_step,
        'generation_pending': generation_pending,
        'population_generation': population.generation,
        'population_last_snapshot_step': population._last_snapshot_step,
        'population_scores': population.scores,
        'num_agents': population.num_agents,
        'agents': [],
    }
    for agent in population.agents:
        state['agents'].append({
            'encoder': agent.encoder.state_dict(),
            'policy': agent.policy.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
        })
    tmp = path / 'training_state.pt.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path / 'training_state.pt')


def load_training_state(path: Path, population, device: str = 'cpu') -> dict:
    """Load training state from checkpoint, restore all agents and population."""
    ckpt = torch.load(Path(path) / 'training_state.pt',
                      map_location=device, weights_only=False)
    saved_agents = ckpt['num_agents']
    if saved_agents != population.num_agents:
        raise ValueError(
            f'Checkpoint has {saved_agents} agents but population has '
            f'{population.num_agents}')
    for i, agent_state in enumerate(ckpt['agents']):
        population.agents[i].encoder.load_state_dict(agent_state['encoder'])
        population.agents[i].policy.load_state_dict(agent_state['policy'])
        population.agents[i].optimizer.load_state_dict(agent_state['optimizer'])
    population._generation = ckpt['population_generation']
    population._last_snapshot_step = ckpt['population_last_snapshot_step']
    population.scores = ckpt['population_scores']
    return {
        'total_collected': ckpt['total_collected'],
        'collection_round': ckpt['collection_round'],
        'last_gen_step': ckpt['last_gen_step'],
        'generation_pending': ckpt['generation_pending'],
    }


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

    # ── create population from opponent_pool config ──────────────────────
    PopCls = config.get('opponent_pool', {}).get('cls')
    if PopCls is None:
        from training.opponents.population import Population as PopCls
    pool_params = config.get('opponent_pool', {}).get('params', {})
    num_agents = pool_params.get('agents', 1)
    snapshot_dir = config.get('snapshot_dir', 'models/snapshots')
    envs_per_worker = config.get('envs_per_worker', 1)
    total_envs = num_envs * envs_per_worker
    agent_envs = scheduler.envs_per_agent(total_envs, num_agents)
    population = PopCls(
        num_agents=num_agents, num_workers=total_envs,
        config={**config, 'device': device},
        envs_per_agent=agent_envs,
        snapshot_dir=snapshot_dir,
        **{k: v for k, v in pool_params.items() if k not in ('agents',)},
    )
    scheduler.init(population, total_envs, config)

    # ── create vectorized envs ──────────────────────────────────────────
    dense_weights = config.get('dense_reward_weights', None)
    env_class = config.get('env_class', None)
    envs = SubprocVecEnv(
        num_envs=num_envs,
        t_window=t_window,
        reward_type=reward_type,
        dense_reward_weights=dense_weights,
        env_class=env_class,
        envs_per_worker=envs_per_worker,
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
    registry.register('population', population.get_metrics)

    # ── profiling config ──────────────────────────────────────────────
    profiling_cfg = config.get('profiling', {})
    profiling_enabled = profiling_cfg.get('enabled', False)
    profiling_report_path = profiling_cfg.get('report', None)
    waterfall_path = profiling_cfg.get('waterfall', None)
    waterfall_rounds = profiling_cfg.get('waterfall_rounds', 2)
    if profiling_enabled and waterfall_path:
        profiler._waterfall_rounds = waterfall_rounds
    all_update_times: List[tuple] = []  # (agent_id, wall_time) for report

    # ── opponent model (centralized GPU inference) ───────────────────────
    from training.opponents.pool import load_opponent_from_snapshot

    _algo_cls = config.get('algorithm', {}).get('cls')
    _is_se3 = _algo_cls is not None and 'SE3' in _algo_cls.__name__

    if _is_se3:
        from se3_field import SE3Encoder
        from se3_policy import StochasticSE3Policy
        opponent_encoder = SE3Encoder().to(device)
        opponent_policy = StochasticSE3Policy().to(device)
    else:
        from encoder import SharedTransformerEncoder, D_MODEL
        from policy_head import StochasticPolicyHead
        opponent_encoder = SharedTransformerEncoder(d_model=D_MODEL).to(device)
        opponent_policy = StochasticPolicyHead(d_model=D_MODEL).to(device)

    opponent_encoder.eval()
    opponent_policy.eval()
    opponent_loaded = False

    # ── evaluation hook ────────────────────────────────────────────────
    from training.evaluation.sim_eval import SimEvaluationHook
    EvalCls = resolve_or_default(config, 'evaluation', SimEvaluationHook)
    eval_hook = None
    eval_interval = config.get('eval_interval', 0)
    if eval_interval > 0:
        eval_hook = EvalCls(config)
        print(f'[train] Eval hook enabled: {EvalCls.__name__}, '
              f'interval={eval_interval} steps')

    # ── main loop ───────────────────────────────────────────────────────
    updater = AsyncUpdater(profiler=profiler if profiling_enabled else None)
    total_collected = 0
    collection_round = 0
    gen_steps = pool_params.get('generation_steps', 1_000_000)
    noise_scale = pool_params.get('generation_noise_scale', 0.01)
    last_gen_step = 0
    generation_pending = False

    # ── resume from checkpoint ─────────────────────────────────────────
    resume_path = model_dir / 'latest'
    if config.get('resume', False) and (resume_path / 'training_state.pt').exists():
        resumed = load_training_state(resume_path, population, device=device)
        total_collected = resumed['total_collected']
        collection_round = resumed['collection_round']
        last_gen_step = resumed['last_gen_step']
        generation_pending = resumed['generation_pending']
        print(f'[train] Resumed from checkpoint: step {total_collected:,}, '
              f'round {collection_round}')
    elif config.get('resume', False):
        print('[train] --resume specified but no checkpoint found, starting fresh.')

    print(f'[train] {num_agents} agent(s), {num_envs} envs, '
          f'rollout_steps={config.get("algorithm", {}).get("params", {}).get("rollout_steps", 2048)}')

    from tqdm import tqdm
    pbar = tqdm(total=total_steps, initial=total_collected,
                unit='step', dynamic_ncols=True)
    try:
        while total_collected < total_steps:
            profiler.start_round()

            # Deferred generation cycle — executes at the TOP of the loop
            # so GPU updates from the previous round overlap with between-
            # round ops. By now the updates have had time to finish.
            _t0 = time.perf_counter()
            if generation_pending:
                for a in population.agents:
                    a._buffer_ready.wait()
                gen_metrics = {f'population/{k}': v
                               for k, v in population.get_metrics().items()}
                logger.log(total_collected, **gen_metrics)
                ranked = population.rank_agents()
                best = population.agents[ranked[0]]
                if len(ranked) > 1:
                    worst_idx = ranked[-1]
                    population.agents[worst_idx].clone_from(
                        best, noise_scale=noise_scale)
                    print(f'[gen {population.generation}] '
                          f'best=agent_{ranked[0]}  worst=agent_{worst_idx} (reset)')
                population.reset_scores()
                last_gen_step = total_collected
                generation_pending = False
            _t1 = time.perf_counter()
            profiler.add_time('generation_time', _t1 - _t0)
            profiler.record_event(_t0, _t1, 'generation')

            # Sample opponent snapshot and load onto GPU
            if population.num_snapshots() > 0:
                snap_path = population.sample_opponent()
                if snap_path:
                    _t0 = time.perf_counter()
                    weights = load_opponent_from_snapshot(snap_path, device=device)
                    opponent_encoder.load_state_dict(weights['encoder'])
                    opponent_policy.load_state_dict(weights['policy'])
                    opponent_loaded = True
                    profiler.add_time('opponent_load_time', time.perf_counter() - _t0)

            # Collect rollouts and trigger updates (non-blocking; per-agent
            # _buffer_ready gates writes while updates are in flight)
            steps = collect_and_train(
                population, envs, updater, rollout_metrics,
                {**config, 'device': device},
                scheduler, profiler=profiler,
                opponent_encoder=opponent_encoder,
                opponent_policy=opponent_policy,
                opponent_loaded=opponent_loaded)
            total_collected += steps
            timing_metrics.add_steps(steps)
            collection_round += 1
            pbar.update(steps)

            # Drain update metrics and log
            _t0 = time.perf_counter()
            update_results = updater.pop_metrics()
            for agent_id, metrics in update_results:
                prefixed = {f'agent_{agent_id}/{k}': v for k,
                            v in metrics.items()}
                logger.log(total_collected, **prefixed)
                if profiling_enabled and 'update_wall_time' in metrics:
                    all_update_times.append(
                        (agent_id, metrics['update_wall_time']))
            _t1 = time.perf_counter()
            profiler.add_time('metrics_drain_time', _t1 - _t0)
            profiler.record_event(_t0, _t1, 'metrics_drain')

            # Periodic logging
            _t0 = time.perf_counter()
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

                pass  # eval logging happens inline below
            _t1 = time.perf_counter()
            profiler.add_time('logging_time', _t1 - _t0)
            profiler.record_event(_t0, _t1, 'logging')

            # Save snapshots to opponent pool (population IS the pool)
            _t0 = time.perf_counter()
            best_idx = population.rank_agents()[0]
            best_agent = population.agents[best_idx]
            if population.should_swap(total_collected):
                population.save_snapshot(best_agent, total_collected)
            _t1 = time.perf_counter()
            profiler.add_time('snapshot_save_time', _t1 - _t0)
            profiler.record_event(_t0, _t1, 'snapshot_save')

            # Mark generation as pending (deferred to top of next iteration)
            if num_agents > 1 and total_collected - last_gen_step >= gen_steps:
                generation_pending = True

            # Checkpoint (full training state for resume)
            _t0 = time.perf_counter()
            if collection_round % (log_interval * 10) == 0:
                save_training_state(model_dir / 'latest', population,
                                    total_collected, collection_round,
                                    last_gen_step, generation_pending)
            _t1 = time.perf_counter()
            profiler.add_time('checkpoint_time', _t1 - _t0)
            profiler.record_event(_t0, _t1, 'checkpoint')

            # Inline evaluation — runs between collect_and_train() calls.
            # Next collect_and_train() resets all envs, so eval state
            # never contaminates training data.
            if eval_hook is not None and eval_hook.should_evaluate(total_collected):
                _t0 = time.perf_counter()
                best_idx = population.rank_agents()[0]
                best_agent = population.agents[best_idx]
                eval_results = eval_hook.evaluate_inline(
                    best_agent, envs, total_collected, device=device)
                _t1 = time.perf_counter()
                profiler.add_time('eval_time', _t1 - _t0)
                profiler.record_event(_t0, _t1, 'eval')

                eval_metrics = eval_hook.format_metrics(eval_results)
                if eval_metrics:
                    logger.log(total_collected, **eval_metrics)

            profiler.end_round()

    except KeyboardInterrupt:
        print('\n[train] Interrupted.')
    finally:
        pbar.close()
        updater.stop()
        # Save resumable checkpoint (all agents + loop state)
        save_training_state(model_dir / 'latest', population,
                            total_collected, collection_round,
                            last_gen_step, generation_pending)
        print(f'[train] Saved resumable checkpoint to {model_dir}/latest')
        # Save best-agent-only checkpoint for deployment
        best_idx = population.rank_agents()[0]
        population.agents[best_idx].save_checkpoint(model_dir / 'final')
        print(f'[train] Saved final checkpoint to {model_dir}/final')

        # Generate profiling report if enabled
        if profiling_enabled:
            report = profiler.generate_report(
                {**config, 'device': device},
                update_times=all_update_times or None,
                waterfall_path=waterfall_path)
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
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
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

    if args.resume:
        config['resume'] = True

    train(config)


if __name__ == '__main__':
    main()
