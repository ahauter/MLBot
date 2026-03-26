#!/usr/bin/env python3
"""
AWAC Training Loop
==================
Trains a single policy head using Advantage-Weighted Actor-Critic (AWAC)
with the spatiotemporal encoder (sliding T-step observation window).

Both cars in each episode share the same encoder and policy head.
Gradients from blue and orange are summed in a single backward() pass.

Usage
-----
    # Train with rlgym-sim (fast, no live game required):
    python training/train_all.py --env rlgym

    # Train via RLBot (live game):
    python training/train_all.py --env rlbot

    # Override hyperparameters:
    python training/train_all.py --env rlgym --episodes 5000 --lr 1e-4 --awac-beta 2.0

    # Disable W&B (stdout only):
    python training/train_all.py --env rlgym --no-wandb

Architecture
------------
- Spatiotemporal SharedTransformerEncoder (sliding T_WINDOW-step window)
- Single PolicyHead shared across all scenarios and both cars
- AWAC loss: -log π(a|s) * exp(A/β)  with β=1.0
- SequenceReplayBuffer for off-policy replay
- Token entity-axis shuffling as data augmentation (entity_perm per update)

Environment abstraction
-----------------------
  GameEnv (ABC)
  ├── RLGymEnv  — wraps rlgym-sim; fast CPU rollouts
  └── RLBotEnv  — wraps the RLBot game runner
"""

from __future__ import annotations

from encoder import (
    SharedTransformerEncoder,
    TOKEN_FEATURES,
    ENTITY_TYPE_IDS_1V1,
    T_WINDOW,
    rlgym_obs_to_tokens,
    state_to_tokens,
)
from policy_head import PolicyHead
from scenarios.graders import ScenarioGrader, reward_for_event, step_reward
from scenarios.scenario_config import ScenarioConfig
from replay_buffer import SequenceReplayBuffer
from replay_dataset import load_replays_into_buffer
from train_config import TrainConfig
from logger import ExperimentLogger

import abc
import argparse
import math
import multiprocessing
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))


# ── config discovery ──────────────────────────────────────────────────────────

CONFIGS_DIR = Path(__file__).parent / 'scenarios' / 'configs'


def discover_all_configs() -> List[ScenarioConfig]:
    """Load every .yaml file under training/scenarios/configs/."""
    configs = []
    for yaml_path in sorted(CONFIGS_DIR.rglob('*.yaml')):
        try:
            configs.append(ScenarioConfig.from_yaml(yaml_path))
        except Exception as exc:
            print(f'[warn] Skipping {yaml_path.name}: {exc}')
    return configs


# ── discounted returns ────────────────────────────────────────────────────────

def compute_returns(rewards: List[float], gamma: float) -> np.ndarray:
    """Compute normalised discounted Monte Carlo returns."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    if len(returns) > 1:
        std = returns.std()
        if std > 1e-8:
            returns = (returns - returns.mean()) / std
    return returns


# ── AWAC loss ─────────────────────────────────────────────────────────────────

_LOG_STD = math.log(0.5)
_TWO_PI = 2.0 * math.pi


def compute_awac_loss(
    encoder:         SharedTransformerEncoder,
    policy_head:     PolicyHead,
    trajectory:      List[Tuple[np.ndarray, np.ndarray, float]],
    entity_type_ids: Union[torch.Tensor, List[int]],
    config:          TrainConfig,
    entity_perm:     Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute AWAC loss from one episode's trajectory.

    trajectory : list of (window_tokens, action, reward)
      window_tokens  (1, T_WINDOW, N, TOKEN_FEATURES)
      action         (ACTION_DIM,)
      reward         scalar

    Returns (total_loss, policy_loss, value_loss).
    Returns zeros for empty trajectories.
    """
    zero = torch.tensor(0.0)
    if not trajectory:
        return zero, zero, zero

    device = next(encoder.parameters()).device

    tokens_batch = torch.tensor(
        np.concatenate([t[0] for t in trajectory], axis=0),
        dtype=torch.float32, device=device,
    )   # (T_ep, T_WINDOW, N, TOKEN_FEATURES)
    actions_taken = torch.tensor(
        np.stack([t[1] for t in trajectory], axis=0),
        dtype=torch.float32, device=device,
    )   # (T_ep, ACTION_DIM)
    returns = torch.tensor(
        compute_returns([t[2] for t in trajectory], config.gamma),
        dtype=torch.float32, device=device,
    )   # (T_ep,)

    embeddings = encoder(tokens_batch, entity_type_ids,
                         entity_perm=entity_perm)
    policy, values = policy_head(embeddings)
    values = values.squeeze(-1)                         # (T_ep,)

    advantages = (returns - values.detach())            # (T_ep,)

    action_dim = float(policy.shape[1])
    log_probs = (
        -0.5 * ((actions_taken - policy) /
                math.exp(_LOG_STD)).pow(2).sum(dim=-1)
        - action_dim * (_LOG_STD + 0.5 * math.log(_TWO_PI))
    )   # (T_ep,)

    # AWAC weighting: exp(A / β), clipped to prevent explosion
    weights = torch.clamp(
        torch.exp(advantages / config.awac_beta), max=config.awac_max_weight
    ).detach()

    policy_loss = -(log_probs * weights).mean()
    value_loss = (returns - values).pow(2).mean()

    return policy_loss + 0.5 * value_loss, policy_loss, value_loss


# ── environment abstraction ───────────────────────────────────────────────────

_OBS_PAIR = Tuple[np.ndarray, np.ndarray]   # (blue_tokens, orange_tokens)
_STEP_OUT = Tuple[np.ndarray, np.ndarray, float, float, bool]


class GameEnv(abc.ABC):
    """
    Abstract game environment.
    All observations returned are (1, N, TOKEN_FEATURES) float32 arrays.
    """

    @abc.abstractmethod
    def reset(self, scenario: ScenarioConfig) -> _OBS_PAIR:
        """Reset to the given scenario. Returns (blue_obs, orange_obs)."""

    @abc.abstractmethod
    def step(
        self,
        blue_action:   np.ndarray,
        orange_action: np.ndarray,
    ) -> _STEP_OUT:
        """
        Apply actions and advance one tick.
        Returns blue_obs, orange_obs, blue_reward, orange_reward, done.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up resources."""


# ── RLGym-sim environment ─────────────────────────────────────────────────────

class RLGymEnv(GameEnv):
    """Wraps rlgym-sim for fast CPU rollouts — no live Rocket League needed."""

    def __init__(self) -> None:
        self._env = None
        self._scenario: Optional[ScenarioConfig] = None

        try:
            import rlgym_sim
            self._rlgym_sim = rlgym_sim
        except ImportError as exc:
            raise ImportError(
                'rlgym-sim is required for --env rlgym. '
                'Install it with:  pip install rlgym-sim'
            ) from exc

    def _build_env(self, scenario: ScenarioConfig):
        from rlgym_env import (
            TokenObsBuilder,
            ScenarioStateSetter,
            ScenarioRewardFn,
            ScenarioTerminalCondition,
        )
        return self._rlgym_sim.make(
            obs_builder=TokenObsBuilder(),
            action_parser=self._rlgym_sim.utils.action_parsers.ContinuousAction(),
            state_setter=ScenarioStateSetter(scenario),
            reward_fn=ScenarioRewardFn(scenario),
            terminal_conditions=[ScenarioTerminalCondition(scenario)],
            team_size=1,
            spawn_opponents=True,
            tick_skip=8,
        )

    def reset(self, scenario: ScenarioConfig) -> _OBS_PAIR:
        self._scenario = scenario
        if self._env is not None:
            self._env.close()
        self._env = self._build_env(scenario)

        obs_list = self._env.reset()
        if isinstance(obs_list, tuple) and len(obs_list) == 2:
            obs_list = obs_list[0]

        if isinstance(obs_list, np.ndarray) and obs_list.ndim == 2 and obs_list.shape[0] == 2:
            blue_obs, orange_obs = obs_list[0], obs_list[1]
        elif isinstance(obs_list, (list, tuple)) and len(obs_list) == 2:
            blue_obs, orange_obs = obs_list
        else:
            raise ValueError(
                f'Unexpected RLGym reset return: {type(obs_list)}. '
                'Expected two observations for blue and orange.'
            )

        return (
            rlgym_obs_to_tokens(blue_obs,   player_idx=0),
            rlgym_obs_to_tokens(orange_obs, player_idx=1),
        )

    def step(self, blue_action, orange_action) -> _STEP_OUT:
        actions = np.stack([blue_action, orange_action], axis=0)
        obs_list, rewards, terminated, truncated = self._env.step(actions)
        done = bool(terminated or truncated)
        return (
            rlgym_obs_to_tokens(obs_list[0], player_idx=0),
            rlgym_obs_to_tokens(obs_list[1], player_idx=1),
            float(rewards[0]),
            float(rewards[1]),
            done,
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


# ── RLBot live-game environment ───────────────────────────────────────────────

class RLBotEnv(GameEnv):
    """Wraps a live RLBot game runner for human play and expert-data collection."""

    def __init__(self, game_runner, boost_pad_tracker=None) -> None:
        self._runner = game_runner
        self._boost_pad_tracker = boost_pad_tracker
        self._scenario: Optional[ScenarioConfig] = None
        self._grader:   Optional[ScenarioGrader] = None
        self._episode_start: float = 0.0

    def _big_pads(self):
        return self._boost_pad_tracker.get_full_boosts() if self._boost_pad_tracker else None

    def reset(self, scenario: ScenarioConfig) -> _OBS_PAIR:
        self._scenario = scenario
        self._grader = ScenarioGrader(scenario)
        self._episode_start = time.monotonic()
        self._runner.reset(scenario)
        packet = self._runner.get_packet()
        if self._boost_pad_tracker:
            self._boost_pad_tracker.update_boost_status(packet)
        pads = self._big_pads()
        return (
            state_to_tokens(packet, car_idx=0, big_pads=pads),
            state_to_tokens(packet, car_idx=1, big_pads=pads),
        )

    def step(self, blue_action, orange_action) -> _STEP_OUT:
        self._runner.apply_actions(blue_action, orange_action)
        packet = self._runner.get_packet()
        if self._boost_pad_tracker:
            self._boost_pad_tracker.update_boost_status(packet)
        pads = self._big_pads()
        elapsed = time.monotonic() - self._episode_start
        event = self._grader.check(packet, elapsed)
        blue_r = step_reward(packet, self._scenario.reward.blue,   0)
        orange_r = step_reward(packet, self._scenario.reward.orange, 1)
        if event:
            blue_r += reward_for_event(self._scenario.reward.blue,   event)
            orange_r += reward_for_event(self._scenario.reward.orange, event)
        return (
            state_to_tokens(packet, car_idx=0, big_pads=pads),
            state_to_tokens(packet, car_idx=1, big_pads=pads),
            blue_r,
            orange_r,
            event is not None,
        )

    def close(self) -> None:
        pass


# ── per-episode trajectory collection ────────────────────────────────────────

def collect_episode(
    env:             GameEnv,
    scenario:        ScenarioConfig,
    encoder:         SharedTransformerEncoder,
    policy_head:     PolicyHead,
    entity_type_ids: Union[torch.Tensor, List[int]],
    config:          TrainConfig,
    t_window:        int = T_WINDOW,
    explore:         bool = True,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Run one episode and return (blue_trajectory, orange_trajectory).

    Each trajectory is a list of (window_tokens, action, reward) where
    window_tokens is (1, t_window, N, TOKEN_FEATURES).
    """
    blue_obs, orange_obs = env.reset(scenario)

    # obs: (1, N, F) — strip batch dim for buffer storage
    def _strip(obs: np.ndarray) -> np.ndarray:
        return obs[0]   # (N, F)

    init_blue = _strip(blue_obs)
    init_orange = _strip(orange_obs)

    # Sliding window buffers — warm up by replicating the initial frame
    blue_buf = deque([init_blue.copy()
                     for _ in range(t_window)], maxlen=t_window)
    orange_buf = deque([init_orange.copy()
                       for _ in range(t_window)], maxlen=t_window)

    encoder.eval()
    policy_head.eval()

    blue_traj:   List[Tuple] = []
    orange_traj: List[Tuple] = []

    done = False
    while not done:
        blue_window = np.stack(blue_buf)[np.newaxis]    # (1, T, N, F)
        orange_window = np.stack(orange_buf)[np.newaxis]  # (1, T, N, F)

        device = next(encoder.parameters()).device
        with torch.no_grad():
            blue_emb = encoder(
                torch.tensor(blue_window,   dtype=torch.float32,
                             device=device),
                entity_type_ids,
            ).cpu().numpy()   # (1, D_MODEL)
            orange_emb = encoder(
                torch.tensor(orange_window, dtype=torch.float32,
                             device=device),
                entity_type_ids,
            ).cpu().numpy()

        blue_action,   _ = policy_head.act(blue_emb)
        orange_action, _ = policy_head.act(orange_emb)

        if explore:
            blue_action[:5] += np.random.normal(0, config.explore_std, 5)
            orange_action[:5] += np.random.normal(0, config.explore_std, 5)
            np.clip(blue_action[:5],   -1.0, 1.0, out=blue_action[:5])
            np.clip(orange_action[:5], -1.0, 1.0, out=orange_action[:5])

        blue_obs, orange_obs, blue_r, orange_r, done = env.step(
            blue_action, orange_action)

        blue_buf.append(_strip(blue_obs))
        orange_buf.append(_strip(orange_obs))

        blue_traj.append((blue_window,   blue_action,   blue_r))
        orange_traj.append((orange_window, orange_action, orange_r))

    return blue_traj, orange_traj


# ── parallel episode collection ──────────────────────────────────────────

def _collect_episode_worker(args):
    """Subprocess worker: creates its own env + model, runs one episode."""
    scenario, encoder_state, policy_state, entity_ids, config, t_window, explore = args

    env = RLGymEnv()
    encoder = SharedTransformerEncoder()
    encoder.load_state_dict(encoder_state)
    encoder.to('cpu')
    policy_head = PolicyHead()
    policy_head.load_state_dict(policy_state)
    policy_head.to('cpu')
    entity_type_ids = torch.tensor(entity_ids, dtype=torch.long)

    try:
        return collect_episode(
            env, scenario, encoder, policy_head,
            entity_type_ids, config=config,
            t_window=t_window, explore=explore,
        )
    finally:
        env.close()


def collect_episodes_parallel(
    num_envs:        int,
    all_configs:     List[ScenarioConfig],
    encoder:         SharedTransformerEncoder,
    policy_head:     PolicyHead,
    entity_type_ids: torch.Tensor,
    config:          TrainConfig,
    pool:            multiprocessing.pool.Pool,
) -> List[Tuple[List[Tuple], List[Tuple]]]:
    """Collect *num_envs* episodes in parallel using a process pool."""
    encoder_state = {k: v.cpu() for k, v in encoder.state_dict().items()}
    policy_state = {k: v.cpu() for k, v in policy_head.state_dict().items()}
    entity_ids = entity_type_ids.cpu().tolist()

    args_list = [
        (random.choice(all_configs), encoder_state, policy_state,
         entity_ids, config, T_WINDOW, True)
        for _ in range(num_envs)
    ]

    return pool.map(_collect_episode_worker, args_list)


# ── gradient norm helper ──────────────────────────────────────────────────────

def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


# ── off-policy AWAC loss (batched buffer samples) ─────────────────────────────

def _awac_loss_from_batch(
    encoder:         SharedTransformerEncoder,
    policy_head:     PolicyHead,
    entity_type_ids: Union[torch.Tensor, List[int]],
    windows:         torch.Tensor,   # (B, T_WINDOW, N, TOKEN_FEATURES)
    actions:         torch.Tensor,   # (B, ACTION_DIM)
    returns:         torch.Tensor,   # (B,)  — MC returns from buffer
    config:          TrainConfig,
    entity_perm:     Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    AWAC loss over a pre-sampled batch from the replay buffer.

    Unlike compute_awac_loss(), this accepts tensors directly (no trajectory
    list, no return computation) so it works with SequenceReplayBuffer.sample().

    Returns (total_loss, policy_loss, value_loss).
    """
    embeddings = encoder(windows, entity_type_ids, entity_perm=entity_perm)
    policy, values = policy_head(embeddings)
    values = values.squeeze(-1)   # (B,)

    advantages = (returns - values.detach())

    action_dim = float(policy.shape[1])
    log_probs = (
        -0.5 * ((actions - policy) / math.exp(_LOG_STD)).pow(2).sum(dim=-1)
        - action_dim * (_LOG_STD + 0.5 * math.log(_TWO_PI))
    )

    weights = torch.clamp(
        torch.exp(advantages / config.awac_beta), max=config.awac_max_weight
    ).detach()

    policy_loss = -(log_probs * weights).mean()
    value_loss = (returns - values).pow(2).mean()

    return policy_loss + 0.5 * value_loss, policy_loss, value_loss


# ── main training function ────────────────────────────────────────────────────

def train(
    all_configs:     List[ScenarioConfig],
    env:             GameEnv,
    config:          Optional[TrainConfig] = None,
    logger:          Optional[ExperimentLogger] = None,
    entity_type_ids: Union[torch.Tensor, List[int]] = ENTITY_TYPE_IDS_1V1,
    # Optuna integration — pass trial for pruning support
    trial=None,
) -> float:
    """
    Train the AWAC agent and return the final smoothed episode reward.

    Parameters
    ----------
    all_configs : list of ScenarioConfig
    env : GameEnv  (used when num_envs == 1; ignored when num_envs > 1)
    config : TrainConfig  (uses defaults if None)
    logger : ExperimentLogger  (stdout-only if None)
    entity_type_ids : entity type index tensor
    trial : optuna.Trial or None  — if provided, enables ASHA pruning
    """
    if config is None:
        config = TrainConfig()
    if logger is None:
        logger = ExperimentLogger(config, enabled=False)

    model_path = Path(config.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    entity_type_ids = torch.tensor(
        entity_type_ids, dtype=torch.long, device=device)

    # ── 1. Build shared encoder + single policy head ──────────────────────────
    encoder = SharedTransformerEncoder()
    policy_head = PolicyHead()

    # ── 2. Try loading existing checkpoints ───────────────────────────────────
    enc_ckpt = model_path / 'encoder.pt'
    if enc_ckpt.exists():
        encoder = SharedTransformerEncoder.load_from(str(enc_ckpt))
        print(f'Loaded encoder from {enc_ckpt}')

    head_ckpt = model_path / 'policy.pt'
    if head_ckpt.exists():
        policy_head.load(str(head_ckpt))
        print(f'Loaded policy head from {head_ckpt}')

    encoder.to(device)
    policy_head.to(device)

    # ── 3. Single optimizer over ALL parameters ───────────────────────────────
    all_params = list(encoder.parameters()) + list(policy_head.parameters())
    optimizer = optim.Adam(all_params, lr=config.lr)

    # ── 4. Replay buffers (expert and sim kept separate for ratio control) ───
    _buf_kwargs = dict(
        capacity=config.buffer_capacity,
        t_window=T_WINDOW,
        action_dim=PolicyHead.ACTION_DIM,
        token_features=TOKEN_FEATURES,
    )
    expert_buffer = SequenceReplayBuffer(**_buf_kwargs)
    sim_buffer = SequenceReplayBuffer(**_buf_kwargs)

    # ── 4b. Pre-fill expert buffer with human replay episodes ─────────────────
    replay_data_dir = Path(__file__).parent / 'replay_data' / 'parsed'
    if replay_data_dir.exists():
        load_replays_into_buffer(replay_data_dir, expert_buffer)

    num_envs = config.num_envs
    parallel = num_envs > 1

    print(
        f'Configs: {len(all_configs)}   Episodes: {config.max_episodes}   '
        f'Expert buffer: {len(expert_buffer)} steps   '
        f'Expert ratio: {config.expert_replay_ratio:.2f}   '
        f'Parallel envs: {num_envs}'
    )

    # Rolling window for smoothed reward (used for Optuna objective)
    episode_rewards: deque = deque(maxlen=100)
    smoothed_reward = 0.0

    # ── 5. Episode loop ───────────────────────────────────────────────────────
    pool = None
    try:
        if parallel:
            pool = multiprocessing.Pool(num_envs)

        episode = 0
        while episode < config.max_episodes:
            # ── collect episode(s) ────────────────────────────────────────────
            if parallel:
                results = collect_episodes_parallel(
                    num_envs, all_configs, encoder, policy_head,
                    entity_type_ids, config, pool,
                )
            else:
                scenario = random.choice(all_configs)
                results = [collect_episode(
                    env, scenario, encoder, policy_head,
                    entity_type_ids, config=config, explore=True,
                )]

            # ── store trajectories & compute AWAC loss ────────────────────────
            encoder.train()
            policy_head.train()

            total_loss = torch.tensor(0.0, device=device)
            policy_loss = torch.tensor(0.0, device=device)
            value_loss = torch.tensor(0.0, device=device)
            N = None

            for blue_traj, orange_traj in results:
                sim_buffer.add_episode(
                    [(w[0, -1], a, r, i == len(blue_traj) - 1)
                     for i, (w, a, r) in enumerate(blue_traj)]
                )
                sim_buffer.add_episode(
                    [(w[0, -1], a, r, i == len(orange_traj) - 1)
                     for i, (w, a, r) in enumerate(orange_traj)]
                )

                # entity count from window shape (1,T,N,F)
                N = blue_traj[0][0].shape[2]
                entity_perm = torch.randperm(N, device=device)

                b_total, b_pol, b_val = compute_awac_loss(
                    encoder, policy_head, blue_traj,
                    entity_type_ids, config, entity_perm=entity_perm,
                )
                o_total, o_pol, o_val = compute_awac_loss(
                    encoder, policy_head, orange_traj,
                    entity_type_ids, config, entity_perm=entity_perm,
                )
                total_loss = total_loss + b_total + o_total
                policy_loss = policy_loss + b_pol + o_pol
                value_loss = value_loss + b_val + o_val

            # ── gradient update ───────────────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = _grad_norm(all_params)
            optimizer.step()

            # ── off-policy buffer update (expert + sim) ───────────────────────
            buf_loss_val = 0.0
            batch_size = config.buffer_batch_size
            has_expert = len(expert_buffer._valid_endpoints()) > 0
            has_sim = len(sim_buffer._valid_endpoints()) > 0

            if batch_size > 0 and (has_expert or has_sim) and N is not None:
                n_expert = int(
                    batch_size * config.expert_replay_ratio) if has_expert else 0
                n_sim = batch_size - n_expert if has_sim else 0
                # redistribute if one source is unavailable
                if not has_expert:
                    n_expert, n_sim = 0, batch_size
                if not has_sim:
                    n_expert, n_sim = batch_size, 0

                batches = []
                try:
                    if n_expert > 0:
                        batches.append(expert_buffer.sample(
                            n_expert, config.gamma))
                    if n_sim > 0:
                        batches.append(sim_buffer.sample(n_sim, config.gamma))
                except RuntimeError as _sample_err:
                    print(f'[warn] Skipping off-policy update: {_sample_err}',
                          file=sys.stderr)

                if batches:
                    buf_windows = torch.cat([b[0] for b in batches]).to(device)
                    buf_actions = torch.cat([b[1] for b in batches]).to(device)
                    buf_returns = torch.cat([b[2] for b in batches]).to(device)

                    encoder.train()
                    policy_head.train()
                    buf_total, _, _ = _awac_loss_from_batch(
                        encoder, policy_head, entity_type_ids,
                        buf_windows, buf_actions, buf_returns, config,
                        entity_perm=torch.randperm(N, device=device),
                    )
                    optimizer.zero_grad()
                    buf_total.backward()
                    optimizer.step()
                    buf_loss_val = float(buf_total)

            # ── metrics (averaged across parallel episodes) ───────────────────
            batch_rewards = []
            batch_lengths = []
            for blue_traj, orange_traj in results:
                blue_ep_reward = sum(r for _, _, r in blue_traj)
                orange_ep_reward = sum(r for _, _, r in orange_traj)
                batch_rewards.append((blue_ep_reward + orange_ep_reward) / 2.0)
                batch_lengths.append(len(blue_traj))

            for r in batch_rewards:
                episode_rewards.append(r)
            ep_reward = float(np.mean(batch_rewards))
            smoothed_reward = float(np.mean(episode_rewards))

            logger.log(
                episode,
                **{
                    'train/total_loss':            float(total_loss),
                    'train/policy_loss':           float(policy_loss),
                    'train/value_loss':            float(value_loss),
                    'train/buffer_loss':           buf_loss_val,
                    'train/episode_reward':        ep_reward,
                    'train/episode_length':        float(np.mean(batch_lengths)),
                    'train/grad_norm':             grad_norm,
                    'train/expert_buffer_size':    len(expert_buffer),
                    'train/sim_buffer_size':       len(sim_buffer),
                    'train/smoothed_reward_100ep': smoothed_reward,
                },
            )

            episode += num_envs

            # ── Optuna pruning hook ───────────────────────────────────────────
            if trial is not None and episode > 0 and episode % 200 < num_envs:
                trial.report(smoothed_reward, step=episode)
                if trial.should_prune():
                    import optuna
                    raise optuna.exceptions.TrialPruned()

            # ── checkpoint ────────────────────────────────────────────────────
            if episode > 0 and episode % config.save_every < num_envs:
                _save_all(encoder, policy_head, model_path)
                print(f'[ep {episode:05d}] Checkpointed.')

    except Exception as exc:
        # Re-raise unless it's an Optuna pruning signal
        try:
            import optuna
            if not isinstance(exc, optuna.exceptions.TrialPruned):
                raise
        except ImportError:
            raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        env.close()
        logger.finish()

    # ── 6. Final save ─────────────────────────────────────────────────────────
    _save_all(encoder, policy_head, model_path)

    return smoothed_reward


def _save_all(
    encoder:     SharedTransformerEncoder,
    policy_head: PolicyHead,
    model_path:  Path,
) -> None:
    torch.save(encoder.state_dict(),     str(model_path / 'encoder.pt'))
    torch.save(policy_head.state_dict(), str(model_path / 'policy.pt'))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Train with AWAC.')
    parser.add_argument(
        '--env', default='rlgym', choices=['rlgym', 'rlbot'],
        help='Game environment: rlgym (fast sim, default) or rlbot (live game)',
    )
    parser.add_argument('--episodes',    default=10000,  type=int)
    parser.add_argument('--save-every',  default=500,    type=int)
    parser.add_argument('--model-dir',   default='models/')
    # Hyperparameters
    parser.add_argument('--lr',          default=3e-4,   type=float)
    parser.add_argument('--awac-beta',   default=1.0,    type=float)
    parser.add_argument('--awac-max-weight', default=20.0, type=float)
    parser.add_argument('--gamma',       default=0.99,   type=float)
    parser.add_argument('--explore-std',       default=0.1,   type=float)
    parser.add_argument('--expert-ratio',      default=0.5,   type=float,
                        help='Fraction of off-policy batch from expert replays (0.0–1.0)')
    parser.add_argument('--buffer-batch-size', default=256,   type=int,
                        help='Off-policy batch size per episode update (0 to disable)')
    parser.add_argument('--num-envs',         default=1,     type=int,
                        help='Parallel RLGym-sim environments for episode collection')
    # W&B
    parser.add_argument('--no-wandb',    action='store_true',
                        help='Disable W&B logging (stdout only)')
    parser.add_argument('--wandb-project', default='mlbot')
    parser.add_argument('--wandb-run-name', default=None)
    args = parser.parse_args()

    config = TrainConfig(
        lr=args.lr,
        awac_beta=args.awac_beta,
        awac_max_weight=args.awac_max_weight,
        gamma=args.gamma,
        explore_std=args.explore_std,
        expert_replay_ratio=args.expert_ratio,
        buffer_batch_size=args.buffer_batch_size,
        max_episodes=args.episodes,
        save_every=args.save_every,
        model_dir=args.model_dir,
        num_envs=args.num_envs,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    configs = discover_all_configs()
    if not configs:
        print(f'No scenario configs found under {CONFIGS_DIR}')
        return

    if args.env == 'rlgym':
        env: GameEnv = RLGymEnv()
    else:
        raise NotImplementedError(
            '--env rlbot requires a configured game_runner. '
            'Instantiate RLBotEnv(game_runner, boost_pad_tracker) '
            'and call train() directly from your launcher script.'
        )

    logger = ExperimentLogger(config, enabled=not args.no_wandb)

    train(
        all_configs=configs,
        env=env,
        config=config,
        logger=logger,
    )


if __name__ == '__main__':
    main()
