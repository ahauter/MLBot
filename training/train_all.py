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
_TWO_PI  = 2.0 * math.pi


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

    tokens_batch = torch.tensor(
        np.concatenate([t[0] for t in trajectory], axis=0),
        dtype=torch.float32,
    )   # (T_ep, T_WINDOW, N, TOKEN_FEATURES)
    actions_taken = torch.tensor(
        np.stack([t[1] for t in trajectory], axis=0),
        dtype=torch.float32,
    )   # (T_ep, ACTION_DIM)
    returns = torch.tensor(
        compute_returns([t[2] for t in trajectory], config.gamma),
        dtype=torch.float32,
    )   # (T_ep,)

    embeddings = encoder(tokens_batch, entity_type_ids, entity_perm=entity_perm)
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
    value_loss  = (returns - values).pow(2).mean()

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
        blue_r   = step_reward(packet, self._scenario.reward.blue,   0)
        orange_r = step_reward(packet, self._scenario.reward.orange, 1)
        if event:
            blue_r   += reward_for_event(self._scenario.reward.blue,   event)
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
    t_window:        int  = T_WINDOW,
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

    init_blue   = _strip(blue_obs)
    init_orange = _strip(orange_obs)

    # Sliding window buffers — warm up by replicating the initial frame
    blue_buf   = deque([init_blue.copy()   for _ in range(t_window)], maxlen=t_window)
    orange_buf = deque([init_orange.copy() for _ in range(t_window)], maxlen=t_window)

    encoder.eval()
    policy_head.eval()

    blue_traj:   List[Tuple] = []
    orange_traj: List[Tuple] = []

    done = False
    while not done:
        blue_window   = np.stack(blue_buf)[np.newaxis]    # (1, T, N, F)
        orange_window = np.stack(orange_buf)[np.newaxis]  # (1, T, N, F)

        with torch.no_grad():
            blue_emb = encoder(
                torch.tensor(blue_window,   dtype=torch.float32),
                entity_type_ids,
            ).numpy()   # (1, D_MODEL)
            orange_emb = encoder(
                torch.tensor(orange_window, dtype=torch.float32),
                entity_type_ids,
            ).numpy()

        blue_action,   _ = policy_head.act(blue_emb)
        orange_action, _ = policy_head.act(orange_emb)

        if explore:
            blue_action[:5]   += np.random.normal(0, config.explore_std, 5)
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


# ── gradient norm helper ──────────────────────────────────────────────────────

def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


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
    env : GameEnv
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

    entity_type_ids = torch.tensor(entity_type_ids, dtype=torch.long)

    # ── 1. Build shared encoder + single policy head ──────────────────────────
    encoder     = SharedTransformerEncoder()
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

    # ── 3. Single optimizer over ALL parameters ───────────────────────────────
    all_params = list(encoder.parameters()) + list(policy_head.parameters())
    optimizer  = optim.Adam(all_params, lr=config.lr)

    # ── 4. Off-policy replay buffer ───────────────────────────────────────────
    replay_buf = SequenceReplayBuffer(
        capacity=config.buffer_capacity,
        t_window=T_WINDOW,
        action_dim=PolicyHead.ACTION_DIM,
        token_features=TOKEN_FEATURES,
    )

    # ── 4b. Pre-fill buffer with human replay episodes ────────────────────────
    replay_data_dir = Path(__file__).parent / 'replay_data' / 'parsed'
    if replay_data_dir.exists():
        load_replays_into_buffer(replay_data_dir, replay_buf)

    print(f'Configs: {len(all_configs)}   Episodes: {config.max_episodes}   Buffer: {len(replay_buf)} steps')

    # Rolling window for smoothed reward (used for Optuna objective)
    episode_rewards: deque = deque(maxlen=100)
    smoothed_reward = 0.0

    # ── 5. Episode loop ───────────────────────────────────────────────────────
    try:
        for episode in range(config.max_episodes):
            scenario = random.choice(all_configs)

            blue_traj, orange_traj = collect_episode(
                env, scenario, encoder, policy_head,
                entity_type_ids, config=config, explore=True,
            )

            # Store trajectories in replay buffer
            replay_buf.add_episode(
                [(w[0], a, r, i == len(blue_traj) - 1)
                 for i, (w, a, r) in enumerate(blue_traj)]
            )
            replay_buf.add_episode(
                [(w[0], a, r, i == len(orange_traj) - 1)
                 for i, (w, a, r) in enumerate(orange_traj)]
            )

            # ── gradient update ───────────────────────────────────────────────
            encoder.train()
            policy_head.train()

            # Random entity permutation for AWAC data augmentation
            N = blue_traj[0][0].shape[2]   # entity count from window shape (1,T,N,F)
            entity_perm = torch.randperm(N)

            blue_total, blue_pol, blue_val = compute_awac_loss(
                encoder, policy_head, blue_traj,
                entity_type_ids, config, entity_perm=entity_perm,
            )
            orange_total, orange_pol, orange_val = compute_awac_loss(
                encoder, policy_head, orange_traj,
                entity_type_ids, config, entity_perm=entity_perm,
            )
            total_loss  = blue_total + orange_total
            policy_loss = blue_pol  + orange_pol
            value_loss  = blue_val  + orange_val

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = _grad_norm(all_params)
            optimizer.step()

            # ── metrics ───────────────────────────────────────────────────────
            blue_ep_reward   = sum(r for _, _, r in blue_traj)
            orange_ep_reward = sum(r for _, _, r in orange_traj)
            ep_reward        = (blue_ep_reward + orange_ep_reward) / 2.0
            episode_rewards.append(ep_reward)
            smoothed_reward = float(np.mean(episode_rewards))

            logger.log(
                episode,
                **{
                    'train/total_loss':          float(total_loss),
                    'train/policy_loss':         float(policy_loss),
                    'train/value_loss':          float(value_loss),
                    'train/episode_reward':      ep_reward,
                    'train/episode_length':      len(blue_traj),
                    'train/grad_norm':           grad_norm,
                    'train/replay_buffer_size':  len(replay_buf),
                    'train/smoothed_reward_100ep': smoothed_reward,
                },
            )

            # ── Optuna pruning hook ───────────────────────────────────────────
            if trial is not None and episode > 0 and episode % 200 == 0:
                trial.report(smoothed_reward, step=episode)
                if trial.should_prune():
                    raise _TrialPruned()

            # ── checkpoint ────────────────────────────────────────────────────
            if episode > 0 and episode % config.save_every == 0:
                _save_all(encoder, policy_head, model_path)
                print(f'[ep {episode:05d}] Checkpointed.')

    except _TrialPruned:
        pass
    finally:
        env.close()
        logger.finish()

    # ── 6. Final save ─────────────────────────────────────────────────────────
    _save_all(encoder, policy_head, model_path)

    return smoothed_reward


class _TrialPruned(Exception):
    """Internal signal to cleanly exit the training loop on Optuna pruning."""


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
    parser.add_argument('--explore-std', default=0.1,    type=float)
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
        max_episodes=args.episodes,
        save_every=args.save_every,
        model_dir=args.model_dir,
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
