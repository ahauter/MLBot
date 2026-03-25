#!/usr/bin/env python3
"""
Joint Multi-Task Training Loop
================================
Trains all skills simultaneously using adversarial scenario configs.

Both cars in each episode share the SAME encoder and the SAME skill_heads dict.
Gradients from blue's loss and orange's loss are summed in a single backward()
pass so the shared encoder learns features useful for all skills at once.

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
- Actor-Critic with Monte Carlo returns (A2C).
- Policy: Gaussian distribution with fixed log_std = log(0.5).  The mean is
  the tanh output of the skill head.  Exploration comes from per-step additive
  Gaussian noise on the analog controls during trajectory collection.

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
import torch
import torch.optim as optim

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

_LOG_STD = math.log(0.5)
_TWO_PI  = 2.0 * math.pi


def compute_ac_loss(
    encoder:    SharedTransformerEncoder,
    skill_head: SkillHead,
    trajectory: List[Tuple[np.ndarray, np.ndarray, float]],
) -> torch.Tensor:
    """
    Compute Actor-Critic loss from one episode's trajectory.

    trajectory : list of (tokens, action, reward)
      tokens  (1, N_TOKENS, TOKEN_FEATURES)
      action  (ACTION_DIM,)  — action actually taken (including exploration noise)
      reward  scalar

    Returns torch.tensor(0.0) for empty trajectories.
    """
    if not trajectory:
        return torch.tensor(0.0)

    tokens_batch  = torch.tensor(
        np.concatenate([t[0] for t in trajectory], axis=0),
        dtype=torch.float32,
    )   # (T, N_TOKENS, TOKEN_FEATURES)
    actions_taken = torch.tensor(
        np.stack([t[1] for t in trajectory], axis=0),
        dtype=torch.float32,
    )   # (T, ACTION_DIM)
    returns = torch.tensor(
        compute_returns([t[2] for t in trajectory]),
        dtype=torch.float32,
    )   # (T,)

    embeddings     = encoder(tokens_batch)              # (T, D_MODEL)
    policy, values = skill_head(embeddings)             # (T, ACTION_DIM), (T, 1)
    values         = values.squeeze(-1)                 # (T,)

    advantages = returns - values.detach()              # (T,)

    action_dim = float(policy.shape[1])
    log_probs  = (
        -0.5 * ((actions_taken - policy) / math.exp(_LOG_STD)).pow(2).sum(dim=-1)
        - action_dim * (_LOG_STD + 0.5 * math.log(_TWO_PI))
    )
    policy_loss   = -(log_probs * advantages.detach()).mean()
    value_loss    = (returns - values).pow(2).mean()
    entropy_bonus = 0.01 * 0.5 * action_dim * (1.0 + math.log(_TWO_PI * math.exp(2.0 * _LOG_STD)))

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

    The observation builder (TokenObsBuilder in training/rlgym_env.py) serialises
    the token matrix row-by-row into a flat array of length N_TOKENS * TOKEN_FEATURES.
    rlgym_obs_to_tokens() in encoder.py reshapes it back to (1, N_TOKENS, TOKEN_FEATURES).
    """

    def __init__(self, mesh_path: Optional[str] = None) -> None:
        self._env  = None
        self._scenario: Optional[ScenarioConfig] = None

        try:
            import rlgym_sim
            import RocketSim as rsim
            self._rlgym_sim = rlgym_sim
        except ImportError as exc:
            raise ImportError(
                'rlgym-sim is required for --env rlgym.  '
                'Install it with:  pip install rocketsim && '
                'pip install git+https://github.com/AechPro/rocket-league-gym-sim@main'
            ) from exc

        # Initialise the C++ physics engine with the collision mesh files.
        # These must be extracted from your Rocket League installation first.
        path = mesh_path or str(_REPO / 'meshes')
        rsim.init(path)

    def _build_env(self, scenario: ScenarioConfig):
        from rlgym_env import (
            TokenObsBuilder,
            ScenarioStateSetter,
            ScenarioRewardFn,
            ScenarioTerminalCondition,
        )
        return self._rlgym_sim.make(
            obs_builder         = TokenObsBuilder(),
            action_parser       = self._rlgym_sim.utils.action_parsers.ContinuousAction(),
            state_setter        = ScenarioStateSetter(scenario),
            reward_fn           = ScenarioRewardFn(scenario),
            terminal_conditions = [ScenarioTerminalCondition(scenario)],
            team_size           = 1,
            tick_skip           = 8,
        )

    def reset(self, scenario: ScenarioConfig) -> _OBS_PAIR:
        self._scenario = scenario
        if self._env is not None:
            self._env.close()
        self._env = self._build_env(scenario)
        obs_list, _info = self._env.reset()
        return (
            rlgym_obs_to_tokens(obs_list[0], player_idx=0),
            rlgym_obs_to_tokens(obs_list[1], player_idx=1),
        )

    def step(self, blue_action, orange_action) -> _STEP_OUT:
        actions = np.stack([blue_action, orange_action], axis=0)
        obs_list, rewards, terminated, truncated, _info = self._env.step(actions)
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
    """
    Wraps a live RLBot game runner for human play and expert-data collection.

    Requires a configured game_runner with:
        game_runner.reset(scenario) -> None
        game_runner.get_packet()    -> GameTickPacket
        game_runner.apply_actions(blue_controls, orange_controls) -> None

    Pass a configured BoostPadTracker instance (already initialised with field
    info) so that boost pad tokens are populated correctly.
    """

    def __init__(self, game_runner, boost_pad_tracker=None) -> None:
        self._runner            = game_runner
        self._boost_pad_tracker = boost_pad_tracker
        self._scenario: Optional[ScenarioConfig] = None
        self._grader:   Optional[ScenarioGrader] = None
        self._episode_start: float = 0.0

    def _big_pads(self):
        return self._boost_pad_tracker.get_full_boosts() if self._boost_pad_tracker else None

    def reset(self, scenario: ScenarioConfig) -> _OBS_PAIR:
        self._scenario      = scenario
        self._grader        = ScenarioGrader(scenario)
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
        pads    = self._big_pads()
        elapsed = time.monotonic() - self._episode_start
        event   = self._grader.check(packet, elapsed)
        blue_r  = step_reward(packet, self._scenario.reward.blue,   0)
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
    env:         GameEnv,
    scenario:    ScenarioConfig,
    encoder:     SharedTransformerEncoder,
    skill_heads: Dict[str, SkillHead],
    explore:     bool = True,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Run one episode and return (blue_trajectory, orange_trajectory)."""
    blue_skill   = scenario.initial_state.blue.skill
    orange_skill = scenario.initial_state.orange.skill

    blue_obs, orange_obs = env.reset(scenario)

    blue_traj:   List[Tuple] = []
    orange_traj: List[Tuple] = []

    encoder.eval()
    for head in skill_heads.values():
        head.eval()

    done = False
    while not done:
        with torch.no_grad():
            blue_emb   = encoder(torch.tensor(blue_obs,   dtype=torch.float32)).numpy()
            orange_emb = encoder(torch.tensor(orange_obs, dtype=torch.float32)).numpy()

        blue_action,   _ = skill_heads[blue_skill].act(blue_emb)
        orange_action, _ = skill_heads[orange_skill].act(orange_emb)

        if explore:
            blue_action[:5]   += np.random.normal(0, 0.1, 5)
            orange_action[:5] += np.random.normal(0, 0.1, 5)
            np.clip(blue_action[:5],   -1.0, 1.0, out=blue_action[:5])
            np.clip(orange_action[:5], -1.0, 1.0, out=orange_action[:5])

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

    all_skill_names: set = set()
    for cfg in all_configs:
        all_skill_names.add(cfg.initial_state.blue.skill)
        all_skill_names.add(cfg.initial_state.orange.skill)
    all_skill_names.discard('')

    # ── 2. Build one SkillHead per skill ──────────────────────────────────────
    skill_heads: Dict[str, SkillHead] = {
        name: SkillHead(name) for name in sorted(all_skill_names)
    }

    # ── 3. Try loading existing checkpoints ───────────────────────────────────
    enc_ckpt = model_path / 'encoder.pt'
    if enc_ckpt.exists():
        encoder.load_state_dict(torch.load(str(enc_ckpt), map_location='cpu'))
        print(f'Loaded encoder from {enc_ckpt}')
    for name, head in skill_heads.items():
        head_ckpt = model_path / f'skill_{name}.pt'
        if head_ckpt.exists():
            head.load_state_dict(torch.load(str(head_ckpt), map_location='cpu'))
            print(f'Loaded skill head: {name}')

    # ── 4. Single optimizer over ALL parameters ───────────────────────────────
    all_params = list(encoder.parameters())
    for head in skill_heads.values():
        all_params += list(head.parameters())
    optimizer = optim.Adam(all_params, lr=3e-4)

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

    # ── 5. Episode loop ───────────────────────────────────────────────────────
    try:
        for episode in range(max_episodes):
            scenario     = random.choice(active_configs)
            blue_skill   = scenario.initial_state.blue.skill
            orange_skill = scenario.initial_state.orange.skill

            # Collect trajectory (encoder in eval mode, no_grad)
            blue_traj, orange_traj = collect_episode(
                env, scenario, encoder, skill_heads, explore=True,
            )

            # ── gradient update ───────────────────────────────────────────────
            encoder.train()
            skill_heads[blue_skill].train()
            skill_heads[orange_skill].train()

            blue_loss   = compute_ac_loss(encoder, skill_heads[blue_skill],   blue_traj)
            orange_loss = compute_ac_loss(encoder, skill_heads[orange_skill], orange_traj)
            total_loss  = blue_loss + orange_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # ── logging ───────────────────────────────────────────────────────
            if episode % 100 == 0:
                print(
                    f'[ep {episode:05d}] loss={float(total_loss):.4f}  '
                    f'blue={blue_skill}  orange={orange_skill}  '
                    f'ticks={len(blue_traj)}'
                )

            # ── checkpoint ────────────────────────────────────────────────────
            if episode > 0 and episode % save_every == 0:
                _save_all(encoder, skill_heads, model_path)
                print(f'[ep {episode:05d}] Checkpointed.')

            # TODO: PPO upgrade — replace A2C with rollout buffer + clipped surrogate
            # TODO: Expert data mixing (p=0.1) — BC loss from expert_data/<skill>/*.npz

    finally:
        env.close()

    # ── 6. Final save ─────────────────────────────────────────────────────────
    _save_all(encoder, skill_heads, model_path)

    # ── 7. Build KNN index ────────────────────────────────────────────────────
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
    torch.save(encoder.state_dict(),    str(model_path / 'encoder.pt'))
    for name, head in skill_heads.items():
        torch.save(head.state_dict(), str(model_path / f'skill_{name}.pt'))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Train all RL skills jointly.')
    parser.add_argument(
        '--env', default='rlgym', choices=['rlgym', 'rlbot'],
        help='Game environment: rlgym (fast sim, default) or rlbot (live game)',
    )
    parser.add_argument('--skill',      default=None,   help='Fine-tune a single skill')
    parser.add_argument('--episodes',   default=10000,  type=int)
    parser.add_argument('--save-every', default=500,    type=int)
    parser.add_argument('--model-dir',  default='models/')
    parser.add_argument(
        '--meshes', default=None,
        help='Path to folder containing .cmf collision mesh files '
             '(default: <repo>/meshes/)',
    )
    args = parser.parse_args()

    configs = discover_all_configs()
    if not configs:
        print(f'No scenario configs found under {CONFIGS_DIR}')
        return

    if args.env == 'rlgym':
        env: GameEnv = RLGymEnv(mesh_path=args.meshes)
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
