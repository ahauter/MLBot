#!/usr/bin/env python3
"""
Joint Multi-Task Training Loop
================================
Trains all skills simultaneously using adversarial scenario configs.

Both cars in each episode share the SAME encoder and the SAME skill_heads dict.
Gradients from blue's loss and orange's loss are summed in a single GradientTape
so the shared encoder learns features useful for all skills at once.

Usage
-----
    # Train with rlgym-sim (fast, no live game required):
    python training/train_all.py --env rlgym

    # Train via RLBot (live game, supports human play / expert data):
    python training/train_all.py --env rlbot

    # Fine-tune a single skill only:
    python training/train_all.py --env rlgym --skill shooting

    # Override number of episodes:
    python training/train_all.py --env rlgym --episodes 5000

Architecture notes
------------------
- Actor-Critic with Monte Carlo returns (A2C).  Upgrade path to PPO is
  scaffolded in the TODOs below.
- Policy: Gaussian distribution with fixed log_std = log(0.5).  The mean is
  the tanh output of the skill head.  Exploration comes from per-step additive
  Gaussian noise on the analog controls during trajectory collection.
- The GradientTape context wraps compute_ac_loss() which replays the episode
  as a batch — the encoder is called INSIDE the tape for training, separate
  from the inference calls during tick collection.

Environment abstraction
-----------------------
  GameEnv (ABC)
  ├── RLGymEnv  — wraps rlgym-sim; fast CPU rollouts, no live game needed
  └── RLBotEnv  — wraps the RLBot game runner; supports human play and
                  expert-data collection alongside the live game

  Both envs expose the same interface:
      obs_blue, obs_orange = env.reset(scenario)
      obs_blue, obs_orange, blue_r, orange_r, done = env.step(blue_action, orange_action)
      env.close()

  Observations are (1, N_TOKENS, TOKEN_FEATURES) float32 arrays — ready to
  pass straight into the encoder.
"""

from __future__ import annotations

import abc
import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from scenarios.scenario_config import ScenarioConfig
from scenarios.graders import ScenarioGrader, reward_for_event, step_reward
from encoder import (
    SharedTransformerEncoder,
    N_TOKENS,
    TOKEN_FEATURES,
    rlgym_obs_to_tokens,
    state_to_tokens,
)
from skills.skill_head import SkillHead
from skills.controller import KNNController

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

GAMMA = 0.99


def compute_returns(rewards: List[float]) -> np.ndarray:
    """Compute normalised discounted Monte Carlo returns."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + GAMMA * G
        returns[t] = G
    if len(returns) > 1:
        std = returns.std()
        if std > 1e-8:
            returns = (returns - returns.mean()) / std
    return returns


# ── Actor-Critic loss ─────────────────────────────────────────────────────────

_LOG_STD = tf.constant(math.log(0.5), dtype=tf.float32)
_TWO_PI  = tf.constant(2.0 * math.pi, dtype=tf.float32)


def compute_ac_loss(
    encoder:    SharedTransformerEncoder,
    skill_head: SkillHead,
    trajectory: List[Tuple[np.ndarray, np.ndarray, float]],
) -> tf.Tensor:
    """
    Compute Actor-Critic loss from one episode's trajectory.

    trajectory : list of (tokens, action, reward)
      tokens  (1, N_TOKENS, TOKEN_FEATURES)
      action  (ACTION_DIM,)  — action actually taken (including exploration noise)
      reward  scalar

    Must be called INSIDE a tf.GradientTape so encoder gradients are captured.
    Returns tf.constant(0.0) for empty trajectories.
    """
    if not trajectory:
        return tf.constant(0.0)

    tokens_batch  = tf.constant(
        np.concatenate([t[0] for t in trajectory], axis=0),
        dtype=tf.float32,
    )   # (T, N_TOKENS, TOKEN_FEATURES)
    actions_taken = tf.constant(
        np.stack([t[1] for t in trajectory], axis=0),
        dtype=tf.float32,
    )   # (T, ACTION_DIM)
    returns = tf.constant(
        compute_returns([t[2] for t in trajectory]),
        dtype=tf.float32,
    )   # (T,)

    embeddings     = encoder(tokens_batch, training=True)       # (T, D_MODEL)
    policy, values = skill_head(embeddings, training=True)      # (T, ACTION_DIM), (T, 1)
    values         = tf.squeeze(values, axis=-1)                # (T,)

    advantages = returns - tf.stop_gradient(values)

    action_dim = tf.cast(tf.shape(policy)[1], tf.float32)
    log_probs  = (
        -0.5 * tf.reduce_sum(
            tf.square((actions_taken - policy) / tf.exp(_LOG_STD)), axis=-1
        )
        - action_dim * (_LOG_STD + 0.5 * tf.math.log(_TWO_PI))
    )
    policy_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
    value_loss  =  tf.reduce_mean(tf.square(returns - values))

    # Differential entropy of Gaussian — encourages exploration
    entropy_bonus = 0.01 * 0.5 * action_dim * (
        1.0 + tf.math.log(_TWO_PI * tf.exp(2.0 * _LOG_STD))
    )

    return policy_loss + 0.5 * value_loss - entropy_bonus


# ── environment abstraction ───────────────────────────────────────────────────

_OBS_PAIR = Tuple[np.ndarray, np.ndarray]   # (blue_tokens, orange_tokens)
_STEP_OUT = Tuple[np.ndarray, np.ndarray, float, float, bool]


class GameEnv(abc.ABC):
    """
    Abstract game environment used by the training loop.

    All observations returned are (1, N_TOKENS, TOKEN_FEATURES) float32 arrays
    that can be fed directly into SharedTransformerEncoder.
    """

    @abc.abstractmethod
    def reset(self, scenario: ScenarioConfig) -> _OBS_PAIR:
        """Reset to the given scenario.  Returns (blue_obs, orange_obs)."""

    @abc.abstractmethod
    def step(
        self,
        blue_action:   np.ndarray,
        orange_action: np.ndarray,
    ) -> _STEP_OUT:
        """
        Apply actions and advance one tick.

        Returns
        -------
        blue_obs, orange_obs : (1, N_TOKENS, TOKEN_FEATURES)
        blue_reward          : float
        orange_reward        : float
        done                 : bool
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up resources."""


# ── RLGym-sim environment ─────────────────────────────────────────────────────

class RLGymEnv(GameEnv):
    """
    Wraps rlgym-sim for fast CPU rollouts — no live Rocket League needed.

    The observation builder (TokenObsBuilder) serialises the token matrix row-by-row
    into a flat array of length N_TOKENS * TOKEN_FEATURES.  rlgym_obs_to_tokens()
    in encoder.py reshapes it back into (1, N_TOKENS, TOKEN_FEATURES).

    rlgym-sim uses a 2-player environment where action arrays are stacked:
        actions = np.stack([blue_action, orange_action], axis=0)  # (2, ACTION_DIM)
    """

    def __init__(self) -> None:
        self._env  = None
        self._scenario: Optional[ScenarioConfig] = None
        self._grader:   Optional[ScenarioGrader] = None
        self._episode_start: float = 0.0
        self._last_obs: Optional[Tuple] = None

        try:
            import rlgym_sim
            self._rlgym_sim = rlgym_sim
        except ImportError as exc:
            raise ImportError(
                'rlgym-sim is required for --env rlgym.  '
                'Install it with:  pip install rlgym-sim'
            ) from exc

    # ------------------------------------------------------------------
    def _build_env(self, scenario: ScenarioConfig):
        """Construct a fresh rlgym-sim env from the given ScenarioConfig."""
        from rlgym_env import (   # local helper in training/
            TokenObsBuilder,
            ScenarioStateSetter,
            ScenarioRewardFn,
            ScenarioTerminalCondition,
        )
        obs_builder  = TokenObsBuilder()
        state_setter = ScenarioStateSetter(scenario)
        reward_fn    = ScenarioRewardFn(scenario)
        terminal     = ScenarioTerminalCondition(scenario)

        return self._rlgym_sim.make(
            obs_builder           = obs_builder,
            action_parser         = self._rlgym_sim.utils.action_parsers.ContinuousAction(),
            state_setter          = state_setter,
            reward_fn             = reward_fn,
            terminal_conditions   = [terminal],
            team_size             = 1,
            tick_skip             = 8,
        )

    # ------------------------------------------------------------------
    def reset(self, scenario: ScenarioConfig) -> _OBS_PAIR:
        self._scenario     = scenario
        self._grader       = ScenarioGrader(scenario)
        self._episode_start = time.monotonic()

        if self._env is not None:
            self._env.close()
        self._env = self._build_env(scenario)

        obs_list, _info = self._env.reset()   # list of flat obs per player
        blue_tokens   = rlgym_obs_to_tokens(obs_list[0], player_idx=0)
        orange_tokens = rlgym_obs_to_tokens(obs_list[1], player_idx=1)
        self._last_obs = (blue_tokens, orange_tokens)
        return blue_tokens, orange_tokens

    # ------------------------------------------------------------------
    def step(
        self,
        blue_action:   np.ndarray,
        orange_action: np.ndarray,
    ) -> _STEP_OUT:
        actions = np.stack([blue_action, orange_action], axis=0)  # (2, ACTION_DIM)
        obs_list, rewards, terminated, truncated, _info = self._env.step(actions)

        blue_tokens   = rlgym_obs_to_tokens(obs_list[0], player_idx=0)
        orange_tokens = rlgym_obs_to_tokens(obs_list[1], player_idx=1)

        done = bool(terminated or truncated)
        return blue_tokens, orange_tokens, float(rewards[0]), float(rewards[1]), done

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


# ── RLBot live-game environment ───────────────────────────────────────────────

class RLBotEnv(GameEnv):
    """
    Wraps a live RLBot game runner for human play and expert-data collection.

    This env requires an external game runner that provides:
        game_runner.reset(scenario) -> None
        game_runner.get_packet()    -> GameTickPacket
        game_runner.apply_actions(blue_controls, orange_controls) -> None

    Pass a configured game_runner instance to the constructor.
    The boost_pad_tracker must have been initialised with the field info
    before the first call to reset().
    """

    def __init__(self, game_runner, boost_pad_tracker=None) -> None:
        self._runner = game_runner
        self._boost_pad_tracker = boost_pad_tracker
        self._scenario: Optional[ScenarioConfig] = None
        self._grader:   Optional[ScenarioGrader] = None
        self._episode_start: float = 0.0

    # ------------------------------------------------------------------
    def _get_big_pads(self):
        if self._boost_pad_tracker is None:
            return None
        return self._boost_pad_tracker.get_full_boosts()

    # ------------------------------------------------------------------
    def reset(self, scenario: ScenarioConfig) -> _OBS_PAIR:
        self._scenario      = scenario
        self._grader        = ScenarioGrader(scenario)
        self._episode_start = time.monotonic()

        self._runner.reset(scenario)
        packet = self._runner.get_packet()

        if self._boost_pad_tracker is not None:
            self._boost_pad_tracker.update_boost_status(packet)

        big_pads      = self._get_big_pads()
        blue_tokens   = state_to_tokens(packet, car_idx=0, big_pads=big_pads)
        orange_tokens = state_to_tokens(packet, car_idx=1, big_pads=big_pads)
        return blue_tokens, orange_tokens

    # ------------------------------------------------------------------
    def step(
        self,
        blue_action:   np.ndarray,
        orange_action: np.ndarray,
    ) -> _STEP_OUT:
        self._runner.apply_actions(blue_action, orange_action)
        packet = self._runner.get_packet()

        if self._boost_pad_tracker is not None:
            self._boost_pad_tracker.update_boost_status(packet)

        big_pads      = self._get_big_pads()
        blue_tokens   = state_to_tokens(packet, car_idx=0, big_pads=big_pads)
        orange_tokens = state_to_tokens(packet, car_idx=1, big_pads=big_pads)

        elapsed = time.monotonic() - self._episode_start
        event   = self._grader.check(packet, elapsed)

        blue_r   = step_reward(packet, self._scenario.reward.blue,   0)
        orange_r = step_reward(packet, self._scenario.reward.orange, 1)
        if event:
            blue_r   += reward_for_event(self._scenario.reward.blue,   event)
            orange_r += reward_for_event(self._scenario.reward.orange, event)

        done = event is not None
        return blue_tokens, orange_tokens, blue_r, orange_r, done

    # ------------------------------------------------------------------
    def close(self) -> None:
        pass   # game runner lifetime is managed by the caller


# ── per-episode trajectory collection ────────────────────────────────────────

def collect_episode(
    env:         GameEnv,
    scenario:    ScenarioConfig,
    encoder:     SharedTransformerEncoder,
    skill_heads: Dict[str, SkillHead],
    explore:     bool = True,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Run one episode and return (blue_trajectory, orange_trajectory).

    Each trajectory is a list of (tokens, action, reward) tuples.
    """
    blue_skill   = scenario.initial_state.blue.skill
    orange_skill = scenario.initial_state.orange.skill

    blue_obs, orange_obs = env.reset(scenario)

    blue_traj:   List[Tuple] = []
    orange_traj: List[Tuple] = []

    done = False
    while not done:
        # ── inference (outside GradientTape) ──────────────────────────────────
        blue_emb   = encoder(tf.constant(blue_obs)).numpy()    # (1, D_MODEL)
        orange_emb = encoder(tf.constant(orange_obs)).numpy()

        blue_action,   _ = skill_heads[blue_skill].act(blue_emb)
        orange_action, _ = skill_heads[orange_skill].act(orange_emb)

        # ── exploration noise on analog dims (throttle, steer, pitch, yaw, roll)
        if explore:
            blue_action[:5]   += np.random.normal(0, 0.1, 5)
            orange_action[:5] += np.random.normal(0, 0.1, 5)
            np.clip(blue_action[:5],   -1.0, 1.0, out=blue_action[:5])
            np.clip(orange_action[:5], -1.0, 1.0, out=orange_action[:5])

        # ── environment step ──────────────────────────────────────────────────
        blue_obs, orange_obs, blue_r, orange_r, done = env.step(blue_action, orange_action)

        blue_traj.append((blue_obs,   blue_action,   blue_r))
        orange_traj.append((orange_obs, orange_action, orange_r))

    return blue_traj, orange_traj


# ── main training function ────────────────────────────────────────────────────

def train(
    all_configs:  List[ScenarioConfig],
    env:          GameEnv,
    skill_filter: Optional[str] = None,
    max_episodes: int           = 10000,
    save_every:   int           = 500,
    model_dir:    str           = 'models/',
) -> None:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Build shared encoder ───────────────────────────────────────────────
    encoder = SharedTransformerEncoder()

    # Collect all unique skill names across every config (both sides)
    all_skill_names: set = set()
    for cfg in all_configs:
        all_skill_names.add(cfg.initial_state.blue.skill)
        all_skill_names.add(cfg.initial_state.orange.skill)
    all_skill_names.discard('')

    # ── 2. Build one SkillHead per skill ──────────────────────────────────────
    skill_heads: Dict[str, SkillHead] = {
        name: SkillHead(name) for name in sorted(all_skill_names)
    }

    # Build all models with dummy inputs so weights are created
    _dummy = tf.zeros((1, N_TOKENS, TOKEN_FEATURES))
    encoder(_dummy)
    for head in skill_heads.values():
        head(tf.zeros((1, encoder.d_model)))

    # Try loading existing checkpoints
    enc_ckpt = model_path / 'encoder.weights.h5'
    if enc_ckpt.exists():
        encoder.load_weights(str(enc_ckpt))
        print(f'Loaded encoder from {enc_ckpt}')
    for name, head in skill_heads.items():
        head_ckpt = model_path / f'skill_{name}.weights.h5'
        if head_ckpt.exists():
            head.load_weights(str(head_ckpt))
            print(f'Loaded skill head: {name}')

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    # Filter to a single skill if requested
    active_configs = all_configs
    if skill_filter:
        active_configs = [
            c for c in all_configs
            if (c.initial_state.blue.skill   == skill_filter or
                c.initial_state.orange.skill == skill_filter)
        ]
        if not active_configs:
            raise ValueError(f'No configs found for skill: {skill_filter!r}')
        print(f'Fine-tuning skill: {skill_filter!r}  ({len(active_configs)} configs)')

    print(f'Skills: {sorted(all_skill_names)}')
    print(f'Configs: {len(active_configs)}   Episodes: {max_episodes}')

    # ── 3. Episode loop ───────────────────────────────────────────────────────
    try:
        for episode in range(max_episodes):
            scenario     = random.choice(active_configs)
            blue_skill   = scenario.initial_state.blue.skill
            orange_skill = scenario.initial_state.orange.skill

            # Collect trajectory from the live environment
            blue_traj, orange_traj = collect_episode(
                env, scenario, encoder, skill_heads, explore=True,
            )

            # ── 4. Joint gradient update ──────────────────────────────────────
            with tf.GradientTape() as tape:
                blue_loss   = compute_ac_loss(encoder, skill_heads[blue_skill],   blue_traj)
                orange_loss = compute_ac_loss(encoder, skill_heads[orange_skill], orange_traj)
                total_loss  = blue_loss + orange_loss

            # Collect variables; avoid duplicate entries when both sides share a skill
            vars_ep = (
                encoder.trainable_variables
                + skill_heads[blue_skill].trainable_variables
                + (skill_heads[orange_skill].trainable_variables
                   if orange_skill != blue_skill else [])
            )
            grads = tape.gradient(total_loss, vars_ep)
            optimizer.apply_gradients(zip(grads, vars_ep))

            # ── 5. Logging ────────────────────────────────────────────────────
            if episode % 100 == 0:
                print(
                    f'[ep {episode:05d}] loss={float(total_loss):.4f}  '
                    f'blue={blue_skill}  orange={orange_skill}  '
                    f'ticks={len(blue_traj)}'
                )

            # ── 6. Checkpoint ─────────────────────────────────────────────────
            if episode > 0 and episode % save_every == 0:
                _save_all(encoder, skill_heads, model_path)
                print(f'[ep {episode:05d}] Checkpointed.')

            # TODO: PPO upgrade — replace A2C trajectory collection with a
            #   rollout buffer and clipped surrogate objective.
            #
            # TODO: Expert data mixing (p=0.1) — with probability 0.1 replace
            #   one side's trajectory with a sampled expert trajectory from
            #   training/expert_data/<skill>/*.npz using BC loss instead of PG.

    finally:
        env.close()

    # ── 7. Final save ─────────────────────────────────────────────────────────
    _save_all(encoder, skill_heads, model_path)

    # ── 8. Build KNN index ────────────────────────────────────────────────────
    print('Building KNN index...')
    controller = KNNController(encoder, k=3)
    controller.build_index(all_configs, n_samples=20)
    knn_path = str(model_path / 'knn_index.npz')
    controller.save_index(knn_path)
    print(f'KNN index saved → {knn_path}')


def _save_all(
    encoder:     SharedTransformerEncoder,
    skill_heads: Dict[str, SkillHead],
    model_path:  Path,
) -> None:
    encoder.save_weights(str(model_path / 'encoder.weights.h5'))
    for name, head in skill_heads.items():
        head.save_weights(str(model_path / f'skill_{name}.weights.h5'))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Train all RL skills jointly.')
    parser.add_argument(
        '--env', default='rlgym', choices=['rlgym', 'rlbot'],
        help='Game environment: rlgym (fast sim, default) or rlbot (live game)',
    )
    parser.add_argument('--skill',      default=None,    help='Fine-tune a single skill')
    parser.add_argument('--episodes',   default=10000,   type=int)
    parser.add_argument('--save-every', default=500,     type=int)
    parser.add_argument('--model-dir',  default='models/')
    args = parser.parse_args()

    configs = discover_all_configs()
    if not configs:
        print(f'No scenario configs found under {CONFIGS_DIR}')
        return

    if args.env == 'rlgym':
        env: GameEnv = RLGymEnv()
    else:
        raise NotImplementedError(
            '--env rlbot requires a configured game_runner.  '
            'Instantiate RLBotEnv(game_runner, boost_pad_tracker) '
            'and call train() directly from your launcher script.'
        )

    train(
        all_configs  = configs,
        env          = env,
        skill_filter = args.skill,
        max_episodes = args.episodes,
        save_every   = args.save_every,
        model_dir    = args.model_dir,
    )


if __name__ == '__main__':
    main()
