#!/usr/bin/env python3
"""
Unified Training Script (d3rlpy)
================================
Single entry point for all RL training. Uses d3rlpy for algorithm
implementations (AWAC, SAC, TD3, CQL, etc.) with our custom transformer
encoder and self-play opponent management.

Usage
-----
    # Baseline training (AWAC, sparse reward, self-play):
    python training/train.py

    # Specific seed:
    python training/train.py --seed 3

    # Swap algorithm:
    python training/train.py --algo SAC

    # Use Optuna-tuned hyperparameters:
    python training/train.py --params-from optuna_baseline.db

    # Short test run:
    python training/train.py --total-steps 10000 --eval-interval 5000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
from collections import deque
import multiprocessing.connection
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

import d3rlpy
from d3rlpy.algos.qlearning.explorers import NormalNoise
from d3rlpy.logging import WanDBAdapterFactory, FileAdapterFactory

from baseline_encoder_factory import TransformerEncoderFactory
from gym_env import BaselineGymEnv
from self_play import OpponentPool
from encoder import N_TOKENS, TOKEN_FEATURES


# ── configuration ────────────────────────────────────────────────────────────

ALGO_MAP = {
    'AWAC': d3rlpy.algos.AWACConfig,
    'SAC': d3rlpy.algos.SACConfig,
    'TD3': d3rlpy.algos.TD3Config,
    'CQL': d3rlpy.algos.CQLConfig,
    'IQL': d3rlpy.algos.IQLConfig,
    'TD3PlusBC': d3rlpy.algos.TD3PlusBCConfig,
}


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # ── seed & identity ──────────────────────────────────────────────────────
    seed: int = 0
    algo: str = 'AWAC'

    # ── budget ───────────────────────────────────────────────────────────────
    total_steps: int = 50_000_000
    eval_interval: int = 200_000       # env steps between Psyonix evaluations
    snapshot_interval: int = 10_000    # env steps between self-play snapshots

    # ── architecture ─────────────────────────────────────────────────────────
    t_window: int = 8
    obs_dim: int = 8 * N_TOKENS * TOKEN_FEATURES  # 800

    # ── AWAC / algorithm hyperparameters ─────────────────────────────────────
    batch_size: int = 256
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    awac_lambda: float = 1.0          # AWAC advantage temperature
    n_critics: int = 2
    explore_noise: float = 0.1        # Gaussian exploration std

    # ── replay buffer ────────────────────────────────────────────────────────
    buffer_capacity: int = 1_000_000
    random_steps: int = 10_000        # random actions before training starts

    # ── d3rlpy fit_online settings ───────────────────────────────────────────
    update_interval: int = 1          # gradient updates per env step (sequential path only)
    n_steps_per_epoch: int = 10_000   # steps per d3rlpy epoch (logging granularity)

    # ── async training (parallel path) ───────────────────────────────────────
    collection_buffer_size: int = 50_000  # transitions to collect before triggering training
    updates_per_swap: int = 500           # gradient steps per training trigger

    # ── parallel environments ─────────────────────────────────────────────
    num_envs: int = 1                 # parallel RLGym-sim envs (1 = sequential)

    # ── self-play ────────────────────────────────────────────────────────────
    max_snapshots: int = 20

    # ── reward ──────────────────────────────────────────────────────────────
    reward_type: str = 'sparse'       # 'sparse' or 'dense'
    dense_reward_weights: Optional[dict] = None  # component weight subset for sweep

    # ── convergence ──────────────────────────────────────────────────────────
    rookie_target_wr: float = 0.60    # win rate target vs Psyonix Rookie
    consecutive_evals_required: int = 2

    # ── paths ────────────────────────────────────────────────────────────────
    model_dir: str = 'models/baseline'
    snapshot_dir: str = 'models/baseline/snapshots'

    # ── experiment infrastructure ────────────────────────────────────────────
    intervention: str = 'none'                    # intervention name for metadata
    replay_seed_dir: Optional[str] = None         # pre-load replay data into buffer
    pretrained_encoder_path: Optional[str] = None  # load pre-trained encoder weights

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_project: str = 'rlbot-baseline'
    wandb_tags: list = field(default_factory=lambda: ['baseline', 'no-intervention'])
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    no_wandb: bool = False


# ── seed setup ───────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── d3rlpy algorithm builder ────────────────────────────────────────────────

def build_algo(config: TrainConfig) -> d3rlpy.algos.QLearningAlgoBase:
    """Build a d3rlpy algorithm from config."""
    encoder_factory = TransformerEncoderFactory(
        t_window=config.t_window,
        pretrained_weights_path=config.pretrained_encoder_path,
    )

    algo_cls = ALGO_MAP.get(config.algo)
    if algo_cls is None:
        raise ValueError(
            f'Unknown algorithm: {config.algo}. '
            f'Available: {list(ALGO_MAP.keys())}'
        )

    # Build kwargs common to all algorithms
    common = dict(
        batch_size=config.batch_size,
        gamma=config.gamma,
        actor_learning_rate=config.actor_lr,
        critic_learning_rate=config.critic_lr,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        tau=config.tau,
    )

    # Algorithm-specific params
    if config.algo == 'AWAC':
        common['lam'] = config.awac_lambda
        common['n_action_samples'] = 1
    if hasattr(algo_cls, '__dataclass_fields__') and 'n_critics' in algo_cls.__dataclass_fields__:
        common['n_critics'] = config.n_critics

    # Filter to only params the config class accepts
    valid_fields = {f.name for f in dataclasses.fields(algo_cls)}
    filtered = {k: v for k, v in common.items() if k in valid_fields}

    algo_config = algo_cls(**filtered)
    return algo_config.create(device=('cuda:0' if torch.cuda.is_available() else 'cpu'))


# ── callback ─────────────────────────────────────────────────────────────────

class TrainingCallback:
    """
    Callback for d3rlpy's fit_online. Handles self-play snapshots,
    evaluation, convergence detection, and W&B eval logging.
    """

    def __init__(self, config: TrainConfig, env: BaselineGymEnv, pool: OpponentPool,
                 axis_tracker=None):
        self.config = config
        self.env = env
        self.pool = pool
        self.axis_tracker = axis_tracker
        self.parallel_envs: Optional[SubprocVecEnv] = None  # set when num_envs > 1
        self.start_time = time.time()
        self.consecutive_wins = 0
        self.converged = False
        self._last_snapshot_step = 0
        self._last_eval_step = 0
        self._wandb = None

        # Initialize W&B for eval/metadata logging (separate from d3rlpy's logger)
        if not config.no_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                pass

    def log_metadata(self):
        """Log run metadata to W&B."""
        if self._wandb is None or self._wandb.run is None:
            return
        rc = self.axis_tracker.reward_components if self.axis_tracker else 0
        self._wandb.run.config.update({
            'meta/algorithm': self.config.algo,
            'meta/seed': self.config.seed,
            'meta/reward_components': rc if rc else (1 if self.config.reward_type == 'sparse' else 7),
            'meta/intervention': self.config.intervention,
            'meta/reference_bot_primary': 'Psyonix_Rookie',
            'meta/observation_dim': self.config.obs_dim,
            'meta/step_budget': self.config.total_steps,
            'meta/t_window': self.config.t_window,
            'meta/d3rlpy_version': d3rlpy.__version__,
        })

    def __call__(self, algo, epoch: int, total_step: int) -> None:
        # ── self-play snapshot ───────────────────────────────────────────
        if total_step - self._last_snapshot_step >= self.config.snapshot_interval:
            self._save_snapshot(algo, total_step)
            self._last_snapshot_step = total_step

        # ── evaluation ───────────────────────────────────────────────────
        if total_step - self._last_eval_step >= self.config.eval_interval:
            self._run_eval(algo, total_step)
            self._last_eval_step = total_step

    def _save_snapshot(self, algo, total_step: int) -> None:
        """Save current policy as self-play opponent snapshot."""
        snap_path = self.pool.save_snapshot(algo, total_step)

        if self.pool.num_snapshots() > 0:
            # Update single env's opponent (used in sequential mode)
            opponent_algo = self.pool.sample_opponent()
            self.env.set_opponent(opponent_algo)

            # Update all subprocess env opponents (used in parallel mode)
            if self.parallel_envs is not None:
                # Sample a random snapshot and send its path to all workers
                snap_dirs = sorted(self.pool.snapshot_dir.iterdir())
                if snap_dirs:
                    chosen = random.choice(snap_dirs)
                    model_path = str(chosen / 'model.pt')
                    self.parallel_envs.set_opponent_path(model_path)

    def _run_eval(self, algo, total_step: int) -> None:
        """Run evaluation against Psyonix tiers."""
        wall_clock = time.time() - self.start_time

        # For now, log placeholder — real Psyonix eval requires live RLBot
        # TODO: integrate training/evaluate.py when RLBot is available
        print(f'\n[step {total_step:,}] Evaluation checkpoint '
              f'(wall clock: {wall_clock/3600:.1f}h)')

        eval_metrics = {
            'eval/steps': total_step,
            'eval/wall_clock_seconds': int(wall_clock),
        }

        # Try to run evaluation if evaluate module is available
        try:
            from evaluate import run_evaluation
            model_dir = Path(self.config.model_dir) / 'eval_temp'
            model_dir.mkdir(parents=True, exist_ok=True)
            algo.save(str(model_dir / 'd3rlpy_model'))

            win_rates = run_evaluation(str(model_dir))
            eval_metrics.update({
                'eval/win_rate_beginner': win_rates.get('Beginner', 0.0),
                'eval/win_rate_rookie': win_rates.get('Rookie', 0.0),
                'eval/win_rate_pro': win_rates.get('Pro', 0.0),
                'eval/win_rate_allstar': win_rates.get('Allstar', 0.0),
            })

            rookie_wr = win_rates.get('Rookie', 0.0)
            print(f'  Rookie win rate: {rookie_wr:.1%}')

            # Convergence check
            if rookie_wr >= self.config.rookie_target_wr:
                self.consecutive_wins += 1
                print(f'  Target met ({self.consecutive_wins}/'
                      f'{self.config.consecutive_evals_required} consecutive)')
                if self.consecutive_wins >= self.config.consecutive_evals_required:
                    self.converged = True
                    print('  CONVERGED — stopping training.')
            else:
                self.consecutive_wins = 0

        except (ImportError, Exception) as e:
            print(f'  Evaluation skipped: {e}')

        # Log axis costs alongside eval metrics
        if self.axis_tracker is not None:
            self.axis_tracker.log(self._wandb, total_step)

        # Log to W&B
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.log(eval_metrics, step=total_step)


# ── subprocess vector environment ────────────────────────────────────────────

def _env_worker(
    pipe: multiprocessing.connection.Connection,
    t_window: int,
    tick_skip: int,
    max_steps: int,
    reward_type: str = 'sparse',
) -> None:
    """Persistent subprocess that owns a BaselineGymEnv."""
    env = BaselineGymEnv(t_window=t_window, tick_skip=tick_skip, max_steps=max_steps,
                         reward_type=reward_type)
    algo_builder = None
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'step':
                result = env.step(data)
                pipe.send(result)
            elif cmd == 'reset':
                result = env.reset()
                pipe.send(result)
            elif cmd == 'set_opponent_path':
                path, = data
                if algo_builder is not None:
                    env.load_opponent_from_path(path, algo_builder=algo_builder)
                pipe.send(None)
            elif cmd == 'set_algo_builder_args':
                # Receive args to reconstruct algo_builder in this process
                config_dict, = data
                import d3rlpy as _d3
                from baseline_encoder_factory import TransformerEncoderFactory as _TEF

                def _make_builder(cfg_d):
                    def _build():
                        # Minimal rebuild: only need predict(), so architecture must match
                        _cfg = TrainConfig(**cfg_d)
                        enc = _TEF(t_window=_cfg.t_window)
                        algo_cls = ALGO_MAP.get(_cfg.algo)
                        common = dict(
                            batch_size=_cfg.batch_size, gamma=_cfg.gamma,
                            actor_learning_rate=_cfg.actor_lr,
                            critic_learning_rate=_cfg.critic_lr,
                            actor_encoder_factory=enc, critic_encoder_factory=enc,
                            tau=_cfg.tau,
                        )
                        if _cfg.algo == 'AWAC':
                            common['lam'] = _cfg.awac_lambda
                            common['n_action_samples'] = 1
                        if hasattr(algo_cls, '__dataclass_fields__') and 'n_critics' in algo_cls.__dataclass_fields__:
                            common['n_critics'] = _cfg.n_critics
                        valid = {f.name for f in dataclasses.fields(algo_cls)}
                        filt = {k: v for k, v in common.items() if k in valid}
                        a = algo_cls(**filt).create(device='cpu')
                        a.build_with_env(env)
                        return a
                    return _build

                algo_builder = _make_builder(config_dict)
                pipe.send(None)
            elif cmd == 'close':
                break
    finally:
        env.close()
        pipe.close()


class SubprocVecEnv:
    """Manages N BaselineGymEnv instances in separate processes."""

    def __init__(self, num_envs: int, t_window: int = 8, tick_skip: int = 8,
                 max_steps: int = 4500, reward_type: str = 'sparse'):
        self.num_envs = num_envs
        self._t_window = t_window
        self._tick_skip = tick_skip
        self._max_steps = max_steps
        self._parents: List[multiprocessing.connection.Connection] = []
        self._procs: List[multiprocessing.Process] = []

        # Use 'spawn' context for Windows/macOS compatibility and CUDA safety
        ctx = multiprocessing.get_context('spawn')

        for _ in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_env_worker,
                args=(child_conn, t_window, tick_skip, max_steps, reward_type),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._parents.append(parent_conn)
            self._procs.append(proc)

    def reset(self) -> List:
        """Reset all envs, return list of (obs, info)."""
        for p in self._parents:
            p.send(('reset', None))
        return [p.recv() for p in self._parents]

    def reset_one(self, idx: int):
        """Reset a single env by index, return (obs, info)."""
        self._parents[idx].send(('reset', None))
        return self._parents[idx].recv()

    def send_reset(self, idx: int) -> None:
        """Send reset command without waiting for result (non-blocking)."""
        self._parents[idx].send(('reset', None))

    def recv_reset(self, idx: int):
        """Receive reset result from a previously sent reset command."""
        return self._parents[idx].recv()

    def step(self, actions: np.ndarray) -> List:
        """Step all envs in parallel with actions (num_envs, action_dim)."""
        for p, a in zip(self._parents, actions):
            p.send(('step', a))
        return [p.recv() for p in self._parents]

    def step_active(self, actions: np.ndarray, active: List[int]) -> dict:
        """Step only the given env indices. Returns {idx: result}."""
        for idx, a in zip(active, actions):
            self._parents[idx].send(('step', a))
        return {idx: self._parents[idx].recv() for idx in active}

    def set_algo_builder_args(self, config_dict: dict) -> None:
        """Send config dict so workers can build algo for opponent loading."""
        for p in self._parents:
            p.send(('set_algo_builder_args', (config_dict,)))
        for p in self._parents:
            p.recv()

    def set_opponent_path(self, path: str) -> None:
        """Tell all workers to load opponent from a snapshot path."""
        for p in self._parents:
            p.send(('set_opponent_path', (path,)))
        for p in self._parents:
            p.recv()

    def close(self) -> None:
        # Signal all workers to exit
        for p in self._parents:
            try:
                p.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        # Join processes — give workers time to exit cleanly
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()  # SIGTERM first — lets finally blocks run
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
        # Close parent pipe ends after workers have exited
        for p in self._parents:
            try:
                p.close()
            except OSError:
                pass

    def respawn_dead(self) -> int:
        """Respawn any dead worker processes. Returns number respawned."""
        ctx = multiprocessing.get_context('spawn')
        respawned = 0
        for i, proc in enumerate(self._procs):
            if not proc.is_alive():
                try:
                    self._parents[i].close()
                except OSError:
                    pass
                parent_conn, child_conn = ctx.Pipe()
                new_proc = ctx.Process(
                    target=_env_worker,
                    args=(child_conn, self._t_window, self._tick_skip, self._max_steps),
                    daemon=True,
                )
                new_proc.start()
                child_conn.close()
                self._parents[i] = parent_conn
                self._procs[i] = new_proc
                respawned += 1
        return respawned

    def assert_workers_alive(self) -> None:
        """Assert all worker processes are still alive. Call at start of each trial."""
        dead = [p.pid for p in self._procs if not p.is_alive()]
        assert not dead, f"SubprocVecEnv: workers died unexpectedly: pids={dead}"

    def assert_workers_dead(self) -> None:
        """Assert all worker processes have exited. Call after close()."""
        still_alive = [p.pid for p in self._procs if p.is_alive()]
        assert not still_alive, (
            f"SubprocVecEnv: workers failed to exit after close(): pids={still_alive}. "
            "Kill them manually (taskkill /F /PID <pid>) before restarting."
        )


# ── async gradient trainer ───────────────────────────────────────────────────

class AsyncTrainer:
    """
    Runs gradient updates in a background thread, decoupled from env collection.

    The main collection loop calls trigger() when enough transitions have been
    gathered. The training thread wakes up, does N gradient steps, then sleeps
    until the next trigger. Collection never blocks on GPU.
    """

    def __init__(self, algo, buffer, batch_size: int, lock: threading.Lock):
        self.algo = algo
        self.buffer = buffer
        self.batch_size = batch_size
        self.lock = lock
        self.last_loss: dict = {}

        self._total_updates = 0   # running count of gradient steps across all triggers
        self._pending_metrics: dict = {}
        self._metrics_lock = threading.Lock()

        self._trigger = threading.Event()
        self._stop = threading.Event()
        self._n_updates = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def trigger(self, n_updates: int) -> None:
        """Called from main thread: kick off n_updates gradient steps."""
        self._n_updates = n_updates
        self._trigger.set()

    def pop_metrics(self) -> dict:
        """Drain pending training metrics. Thread-safe. Call from main thread."""
        with self._metrics_lock:
            m = self._pending_metrics
            self._pending_metrics = {}
            return m

    def stop(self) -> None:
        self._stop.set()
        self._trigger.set()  # unblock if waiting
        self._thread.join()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._trigger.wait()
            self._trigger.clear()
            if self._stop.is_set():
                break
            losses, update_ms_list = [], []
            for _ in range(self._n_updates):
                with self.lock:
                    if self.buffer.transition_count < self.batch_size:
                        break
                    batch = self.buffer.sample_transition_batch(self.batch_size)
                update_start = time.time()
                self.last_loss = self.algo.update(batch)
                update_ms_list.append((time.time() - update_start) * 1000)
                self._total_updates += 1
                losses.append(self.last_loss)
            if losses:
                avg = {f'train/{k}': float(np.mean([l[k] for l in losses]))
                       for k in losses[0]}
                avg['train/total_gradient_updates'] = self._total_updates
                avg['train/update_ms'] = float(np.mean(update_ms_list))
                with self.lock:
                    avg['buffer/transition_count'] = self.buffer.transition_count
                with self._metrics_lock:
                    self._pending_metrics = avg


# ── parallel online training loop ────────────────────────────────────────────

def fit_online_parallel(
    algo,
    config: TrainConfig,
    envs: SubprocVecEnv,
    buffer,
    explorer,
    callback,
    on_episode_complete=None,
    axis_tracker=None,
) -> None:
    """
    Parallel replacement for d3rlpy's fit_online().

    Parameters
    ----------
    on_episode_complete : callable or None
        Called with (episode_return: float) when a blue episode finishes.
        Useful for reward tracking in tuning.

    Steps N environments simultaneously in subprocesses. Accumulates each
    env's episode locally and flushes completed episodes to d3rlpy's buffer
    (which only supports one active episode at a time). Both blue and orange
    perspectives are stored for self-play training signal.
    """
    from tqdm import trange
    num_envs = config.num_envs


    # W&B logging (active if wandb.run was initialized by caller)
    try:
        import wandb as _wandb_mod
        _wandb = _wandb_mod if _wandb_mod.run is not None else None
    except ImportError:
        _wandb = None

    start_wall = time.time()

    # Build algo if needed
    if algo.impl is None:
        # Need a dummy env to build with
        dummy = BaselineGymEnv(t_window=config.t_window, reward_type=config.reward_type)
        algo.build_with_env(dummy)
        dummy.close()

    # Reset all envs
    reset_results = envs.reset()
    observations = np.stack([r[0] for r in reset_results])  # (N, obs_dim)

    # Local episode accumulators per env
    local_blue = [[] for _ in range(num_envs)]
    local_orange = [[] for _ in range(num_envs)]
    rollout_returns = np.zeros(num_envs)
    recent_returns: deque = deque(maxlen=100)  # rolling window for avg reward logging
    recent_goals: deque = deque(maxlen=100)    # 1=blue scored, -1=orange scored, 0=timeout

    # Async trainer: GPU updates run in background, never blocking collection
    buf_lock = threading.Lock()
    trainer = AsyncTrainer(algo, buffer, config.batch_size, buf_lock)
    collected_since_trigger = 0

    total_step = 0
    pending_resets: set = set()  # env indices waiting for async reset
    pbar = trange(1, config.total_steps + 1, desc='Training')

    while total_step < config.total_steps:
        # ── collect any pending resets from previous iteration ─────────────
        for i in list(pending_resets):
            reset_obs, _ = envs.recv_reset(i)
            observations[i] = reset_obs
            pending_resets.discard(i)

        # ── action selection (batched for active envs) ────────────────────
        active = [i for i in range(num_envs) if i not in pending_resets]
        n_active = len(active)
        if n_active == 0:
            continue

        active_obs = observations[active]
        if total_step < config.random_steps:
            active_actions = np.random.uniform(-1, 1, size=(n_active, 8)).astype(np.float32)
        elif explorer:
            active_actions = explorer.sample(algo, active_obs, total_step)
        else:
            active_actions = algo.predict(active_obs)

        # ── step active envs in parallel ──────────────────────────────────
        step_results = envs.step_active(active_actions, active)

        for ai, i in enumerate(active):
            next_obs, reward, done, truncated, info = step_results[i]

            # Accumulate blue transition
            local_blue[i].append((observations[i].copy(), active_actions[ai].copy(), float(reward)))

            # Accumulate orange transition
            if 'orange_obs' in info and 'orange_action' in info:
                local_orange[i].append((
                    info['orange_obs'].copy(),
                    info['orange_action'].copy(),
                    info['orange_reward'],
                ))

            rollout_returns[i] += float(reward)

            if done:
                # Flush blue and orange episodes to buffer (locked for trainer thread safety)
                with buf_lock:
                    for obs, act, rew in local_blue[i]:
                        buffer.append(obs, act, rew)
                    buffer.clip_episode(bool(not truncated))

                    for obs, act, rew in local_orange[i]:
                        buffer.append(obs, act, rew)
                    buffer.clip_episode(bool(not truncated))

                if on_episode_complete is not None:
                    on_episode_complete(rollout_returns[i])

                recent_returns.append(rollout_returns[i])
                recent_goals.append(info.get('goal', 0))

                if _wandb:
                    _wandb.log({
                        'rollout/episode_return': rollout_returns[i],
                        'rollout/episode_length': len(local_blue[i]),
                        'rollout/goal': info.get('goal', 0),
                    }, step=total_step)

                local_blue[i] = []
                local_orange[i] = []
                rollout_returns[i] = 0.0

                # Async reset — send now, recv at start of next iteration
                envs.send_reset(i)
                pending_resets.add(i)
            else:
                observations[i] = next_obs

        total_step += n_active
        if axis_tracker is not None:
            axis_tracker.record_sim_steps(n_active)
        collected_since_trigger += n_active
        pbar.update(n_active)

        # ── trigger async training ────────────────────────────────────────
        if (
            total_step > config.random_steps
            and collected_since_trigger >= config.collection_buffer_size
        ):
            if _wandb is not None and _wandb.run is not None:
                pending = trainer.pop_metrics()
                if pending:
                    _wandb.log(pending, step=total_step)
            trainer.trigger(config.updates_per_swap)
            collected_since_trigger = 0

        # ── callback + axis logging (epoch boundaries) ────────────────────
        epoch = total_step // config.n_steps_per_epoch
        if callback and total_step % config.n_steps_per_epoch < num_envs:
            callback(algo, epoch, total_step)
            if _wandb is not None and _wandb.run is not None:
                elapsed = time.time() - start_wall
                with buf_lock:
                    buf_count = buffer.transition_count
                goals = list(recent_goals)
                _wandb.log({
                    # Rolling average reward and goal rates (last 100 episodes)
                    'rollout/avg_episode_return': float(np.mean(recent_returns)) if recent_returns else 0.0,
                    'rollout/score_rate': float(np.mean([g == 1 for g in goals])) if goals else 0.0,
                    'rollout/concede_rate': float(np.mean([g == -1 for g in goals])) if goals else 0.0,
                    # Axis 1 — simulation steps (key research metric)
                    'consumed_resources/env_steps': total_step,
                    # Axes 2-5 — zero for baseline, always logged for cross-run consistency
                    'consumed_resources/replays_loaded': 0,
                    'consumed_resources/labels_consumed': 0,
                    'consumed_resources/reward_components': 1,
                    'consumed_resources/pretrain_gpu_hours': 0.0,
                    # Collection health
                    'collect/buffer_transition_count': buf_count,
                    'collect/num_envs': num_envs,
                    # Throughput
                    'timing/steps_per_second': total_step / max(elapsed, 1e-6),
                    'timing/wall_clock_seconds': int(elapsed),
                }, step=total_step)

    pbar.close()

    # Stop async trainer thread
    trainer.stop()

    # Drain any pending async resets so pipes are clean for the next trial
    for i in list(pending_resets):
        envs.recv_reset(i)
    pending_resets.clear()

    # Clip any in-progress episodes
    for i in range(num_envs):
        if local_blue[i]:
            for obs, act, rew in local_blue[i]:
                buffer.append(obs, act, rew)
            buffer.clip_episode(False)
        if local_orange[i]:
            for obs, act, rew in local_orange[i]:
                buffer.append(obs, act, rew)
            buffer.clip_episode(False)


# ── main training function ───────────────────────────────────────────────────

def train(config: TrainConfig, axis_tracker=None) -> None:
    """Run training with the given configuration."""
    assert config.eval_interval > 0, "eval_interval must be positive"
    assert config.total_steps > 0, "total_steps must be positive"
    assert config.batch_size > 0, "batch_size must be positive"
    assert config.algo in ALGO_MAP, f"Unknown algo: {config.algo}"

    set_seed(config.seed)

    model_dir = Path(config.model_dir) / f'seed_{config.seed}'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    parallel = config.num_envs > 1

    print(f'Training config:')
    print(f'  Algorithm:  {config.algo}')
    print(f'  Reward:     {config.reward_type}')
    print(f'  Seed:       {config.seed}')
    print(f'  Steps:      {config.total_steps:,}')
    print(f'  Device:     {"cuda" if torch.cuda.is_available() else "cpu"}')
    print(f'  Eval every: {config.eval_interval:,} steps')
    print(f'  Num envs:   {config.num_envs}')
    print(f'  Model dir:  {model_dir}')

    # ── environment ──────────────────────────────────────────────────────
    env = BaselineGymEnv(
        t_window=config.t_window,
        reward_type=config.reward_type,
        dense_reward_weights=config.dense_reward_weights,
    )

    # ── d3rlpy algorithm ─────────────────────────────────────────────────
    algo = build_algo(config)

    # ── self-play opponent pool ──────────────────────────────────────────
    def _algo_builder():
        a = build_algo(config)
        a.build_with_env(env)
        return a

    pool = OpponentPool(
        snapshot_dir=config.snapshot_dir,
        algo_builder=_algo_builder,
        max_snapshots=config.max_snapshots,
    )

    # ── callback ─────────────────────────────────────────────────────────
    callback = TrainingCallback(config, env, pool, axis_tracker=axis_tracker)

    # ── W&B logging ──────────────────────────────────────────────────────
    if config.no_wandb:
        logger_adapter = FileAdapterFactory(root_dir=str(model_dir / 'logs'))
    else:
        logger_adapter = WanDBAdapterFactory(project=config.wandb_project)

    # ── exploration ──────────────────────────────────────────────────────
    explorer = NormalNoise(mean=0.0, std=config.explore_noise)

    # ── replay buffer ────────────────────────────────────────────────────
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=config.buffer_capacity,
        env=env,
    )

    # ── pre-load replay data into buffer (Axis 2) ──────────────────────
    if config.replay_seed_dir:
        from replay_dataset import load_replays_into_buffer
        n_eps = load_replays_into_buffer(config.replay_seed_dir, buffer)
        if axis_tracker is not None:
            # Count .npz files as replays loaded
            from pathlib import Path as _P
            n_files = len(list(_P(config.replay_seed_dir).glob('*.npz')))
            axis_tracker.record_replays(n_files)

    # ── run metadata ─────────────────────────────────────────────────────
    callback.log_metadata()

    # ── train ────────────────────────────────────────────────────────────
    print(f'\nStarting training...\n')

    if parallel:
        # Initialize W&B for parallel path (sequential path gets it from d3rlpy)
        if not config.no_wandb:
            import wandb
            run_name = config.wandb_run_name or f'{config.algo}_seed{config.seed}'
            init_kwargs = dict(
                project=config.wandb_project,
                name=run_name,
                config=dataclasses.asdict(config),
            )
            if config.wandb_group:
                init_kwargs['group'] = config.wandb_group
            wandb.init(**init_kwargs)

        envs = SubprocVecEnv(
            num_envs=config.num_envs,
            t_window=config.t_window,
            reward_type=config.reward_type,
        )
        # Send config to workers so they can build algo for opponent loading
        envs.set_algo_builder_args(dataclasses.asdict(config))
        # Wire callback to update opponents in all subprocess envs
        callback.parallel_envs = envs

        try:
            fit_online_parallel(
                algo=algo,
                config=config,
                envs=envs,
                buffer=buffer,
                explorer=explorer,
                callback=callback,
                axis_tracker=axis_tracker,
            )
        finally:
            envs.close()
            if not config.no_wandb:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
    else:
        algo.fit_online(
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_steps=config.total_steps,
            n_steps_per_epoch=config.n_steps_per_epoch,
            update_interval=config.update_interval,
            random_steps=config.random_steps,
            experiment_name=f'{config.algo}_seed{config.seed}',
            logger_adapter=logger_adapter,
            show_progress=True,
            callback=callback,
        )

    # ── save final model ─────────────────────────────────────────────────
    algo.save(str(model_dir / 'final_model'))
    print(f'\nTraining complete. Model saved to {model_dir}')

    if callback.converged:
        print(f'Converged at Rookie win rate >= {config.rookie_target_wr:.0%}')
    else:
        print(f'Did not converge within {config.total_steps:,} steps.')

    env.close()


# ── Optuna parameter loading ────────────────────────────────────────────────

def load_params_from_optuna(db_path: str, study_name: str = 'baseline-hparam-search') -> dict:
    """Load best hyperparameters from an Optuna study database."""
    try:
        import optuna
    except ImportError:
        raise ImportError('optuna required: pip install optuna')

    study = optuna.load_study(
        study_name=study_name,
        storage=f'sqlite:///{db_path}',
    )
    if not study.best_trial:
        raise RuntimeError(f'No completed trials in {db_path}')

    print(f'Loaded best params from Optuna (trial #{study.best_trial.number}):')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    return study.best_params


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train RL agent with d3rlpy.')

    # Core
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--algo', default='AWAC', choices=list(ALGO_MAP.keys()))
    parser.add_argument('--reward', default='sparse', choices=['sparse', 'dense'],
                        help='Reward function: sparse (goals only) or dense (shaped)')
    parser.add_argument('--total-steps', type=int, default=50_000_000)

    # Hyperparameters
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--awac-lambda', type=float, default=1.0)
    parser.add_argument('--explore-noise', type=float, default=0.1)
    parser.add_argument('--buffer-capacity', type=int, default=1_000_000)
    parser.add_argument('--random-steps', type=int, default=10_000)

    # Architecture
    parser.add_argument('--t-window', type=int, default=8)

    # Training loop
    parser.add_argument('--eval-interval', type=int, default=200_000)
    parser.add_argument('--snapshot-interval', type=int, default=10_000)
    parser.add_argument('--n-steps-per-epoch', type=int, default=10_000)
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Parallel RLGym-sim environments (default: 1 = sequential)')
    parser.add_argument('--collection-buffer-size', type=int, default=50_000,
                        help='Transitions to collect before triggering async training (parallel path)')
    parser.add_argument('--updates-per-swap', type=int, default=500,
                        help='Gradient steps per training trigger (parallel path)')

    # Paths
    parser.add_argument('--model-dir', default='models/baseline')

    # W&B
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', default='rlbot-baseline')

    # Optuna integration
    parser.add_argument('--params-from', default=None,
                        help='Load best hyperparams from Optuna SQLite DB')
    parser.add_argument('--study-name', default='baseline-hparam-search',
                        help='Optuna study name to load best params from (used with --params-from)')

    args = parser.parse_args()

    config = TrainConfig(
        seed=args.seed,
        algo=args.algo,
        reward_type=args.reward,
        total_steps=args.total_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        awac_lambda=args.awac_lambda,
        explore_noise=args.explore_noise,
        buffer_capacity=args.buffer_capacity,
        random_steps=args.random_steps,
        t_window=args.t_window,
        eval_interval=args.eval_interval,
        snapshot_interval=args.snapshot_interval,
        n_steps_per_epoch=args.n_steps_per_epoch,
        num_envs=args.num_envs,
        collection_buffer_size=args.collection_buffer_size,
        updates_per_swap=args.updates_per_swap,
        model_dir=args.model_dir,
        no_wandb=args.no_wandb,
        wandb_project=args.wandb_project,
    )

    # Override with Optuna-tuned params if requested
    if args.params_from:
        params = load_params_from_optuna(args.params_from, study_name=args.study_name)
        param_map = {
            'actor_lr': 'actor_lr',
            'critic_lr': 'critic_lr',
            'awac_lambda': 'awac_lambda',
            'tau': 'tau',
            'batch_size': 'batch_size',
            'gamma': 'gamma',
            'explore_noise': 'explore_noise',
        }
        for optuna_key, config_key in param_map.items():
            if optuna_key in params:
                setattr(config, config_key, params[optuna_key])

    train(config)


if __name__ == '__main__':
    main()
