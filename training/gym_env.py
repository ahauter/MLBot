"""
Gymnasium Environment Wrapper for Baseline Experiments
======================================================
Wraps rlgym-sim as a standard gymnasium.Env for use with d3rlpy.

Features:
- Frame-stacked observations (T consecutive frames → flat vector)
- Sparse goal-only reward (+1 score, -1 concede, 0 otherwise)
- Self-play: frozen opponent model runs inside the env
- Standard kickoff initial state

Observation: (T_WINDOW * N_TOKENS * TOKEN_FEATURES,) = (800,) float32
Action:      (8,) float32 — 5 analog [-1,1] + 3 binary (thresholded at 0.5)

Usage
-----
    env = BaselineGymEnv(t_window=8)
    obs, info = env.reset()         # (800,) flat
    obs, reward, term, trunc, info = env.step(action)
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from encoder import (
    SharedTransformerEncoder,
    TOKEN_FEATURES,
    N_TOKENS,
    ENTITY_TYPE_IDS_1V1,
    T_MAX,
)
from policy_head import PolicyHead


# ── goal detection ───────────────────────────────────────────────────────────
_GOAL_Y = 5124.0


class BaselineGymEnv(gym.Env):
    """
    Gymnasium wrapper for 1v1 baseline training with sparse reward.

    The agent controls blue (team 0). The opponent (team 1) is a frozen
    policy snapshot that runs inside the environment.

    Parameters
    ----------
    t_window : int
        Number of frames to stack in the observation.
    tick_skip : int
        Physics ticks per action (default 8 ≈ 15 actions/sec at 120Hz).
    max_steps : int
        Maximum steps per episode before timeout.
    opponent_encoder : SharedTransformerEncoder | None
        Frozen opponent encoder. If None, opponent takes random actions.
    opponent_policy : PolicyHead | None
        Frozen opponent policy head.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        t_window: int = 8,
        tick_skip: int = 8,
        max_steps: int = 4500,  # ~5 min at 15 steps/sec
        opponent_encoder: Optional[SharedTransformerEncoder] = None,
        opponent_policy: Optional[PolicyHead] = None,
    ):
        super().__init__()
        self.t_window = t_window
        self.tick_skip = tick_skip
        self.max_steps = max_steps

        obs_size = t_window * N_TOKENS * TOKEN_FEATURES
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self._opponent_encoder = opponent_encoder
        self._opponent_policy = opponent_policy
        self._entity_type_ids = torch.tensor(
            ENTITY_TYPE_IDS_1V1, dtype=torch.long
        )

        self._env = None
        self._rlgym_sim = None
        self._blue_buf: deque = deque(maxlen=t_window)
        self._orange_buf: deque = deque(maxlen=t_window)
        self._step_count = 0

    # ── public API ──────────────────────────────────────────────────────────

    def set_opponent(
        self,
        encoder: Optional[SharedTransformerEncoder],
        policy: Optional[PolicyHead],
    ) -> None:
        """Hot-swap the opponent model (e.g. from OpponentPool)."""
        self._opponent_encoder = encoder
        self._opponent_policy = policy

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0

        if self._env is None:
            self._build_env()

        obs_list = self._env.reset()
        blue_obs, orange_obs = self._parse_obs(obs_list)
        blue_tokens = self._to_tokens(blue_obs)
        orange_tokens = self._to_tokens(orange_obs)

        # Warm up frame buffers with initial observation
        self._blue_buf.clear()
        self._orange_buf.clear()
        for _ in range(self.t_window):
            self._blue_buf.append(blue_tokens.copy())
            self._orange_buf.append(orange_tokens.copy())

        return self._get_stacked_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Get opponent action
        opp_action = self._get_opponent_action()

        obs_list, rewards, terminated, truncated = self._env.step(
            np.stack([action, opp_action], axis=0)
        )
        blue_obs, orange_obs = self._parse_obs(obs_list)
        blue_tokens = self._to_tokens(blue_obs)
        orange_tokens = self._to_tokens(orange_obs)

        self._blue_buf.append(blue_tokens)
        self._orange_buf.append(orange_tokens)
        self._step_count += 1

        # Sparse reward: only on episode end (goal scored)
        reward = self._compute_sparse_reward(terminated or truncated)

        # Timeout
        timed_out = self._step_count >= self.max_steps
        done = bool(terminated or truncated or timed_out)

        return self._get_stacked_obs(), reward, done, False, {}

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ── internal ────────────────────────────────────────────────────────────

    def _build_env(self):
        import rlgym_sim
        from rlgym_sim.utils.action_parsers import ContinuousAction
        from rlgym_sim.utils.state_setters import DefaultState
        self._rlgym_sim = rlgym_sim

        self._env = rlgym_sim.make(
            obs_builder=self._make_obs_builder(),
            action_parser=ContinuousAction(),
            state_setter=DefaultState(),
            reward_fn=self._make_sparse_reward(),
            terminal_conditions=[self._make_terminal()],
            team_size=1,
            spawn_opponents=True,
            tick_skip=self.tick_skip,
        )

    def _make_obs_builder(self):
        from rlgym_env import TokenObsBuilder
        return TokenObsBuilder()

    def _make_sparse_reward(self):
        from rlgym_sim.utils import RewardFunction
        from rlgym_sim.utils.gamestates import GameState, PlayerData

        class SparseGoalReward(RewardFunction):
            """Sparse: 0 per step, +1/-1 on goal at episode end."""
            def reset(self, initial_state: GameState) -> None:
                pass

            def get_reward(self, player: PlayerData, state: GameState,
                           previous_action: np.ndarray) -> float:
                return 0.0

            def get_final_reward(self, player: PlayerData, state: GameState,
                                 previous_action: np.ndarray) -> float:
                ball_y = state.ball.position[1]
                scored_blue = ball_y > _GOAL_Y
                scored_orange = ball_y < -_GOAL_Y
                if player.team_num == 0:  # blue
                    if scored_blue:
                        return 1.0
                    elif scored_orange:
                        return -1.0
                else:  # orange
                    if scored_orange:
                        return 1.0
                    elif scored_blue:
                        return -1.0
                return 0.0  # timeout — no goal

        return SparseGoalReward()

    def _make_terminal(self):
        from rlgym_sim.utils import TerminalCondition
        from rlgym_sim.utils.gamestates import GameState

        class GoalTerminal(TerminalCondition):
            """End episode on goal scored only."""
            def reset(self, initial_state: GameState) -> None:
                pass

            def is_terminal(self, current_state: GameState) -> bool:
                return abs(current_state.ball.position[1]) > _GOAL_Y

        return GoalTerminal()

    def _parse_obs(self, obs_result) -> Tuple[np.ndarray, np.ndarray]:
        """Parse rlgym-sim obs return into (blue_flat, orange_flat)."""
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            obs_result = obs_result[0]
        if isinstance(obs_result, np.ndarray) and obs_result.ndim == 2:
            return obs_result[0], obs_result[1]
        if isinstance(obs_result, (list, tuple)) and len(obs_result) == 2:
            return obs_result[0], obs_result[1]
        raise ValueError(f'Unexpected obs format: {type(obs_result)}')

    def _to_tokens(self, flat_obs: np.ndarray) -> np.ndarray:
        """Flat obs (N*F,) → (N, F)."""
        return flat_obs.reshape(N_TOKENS, TOKEN_FEATURES).astype(np.float32)

    def _get_stacked_obs(self) -> np.ndarray:
        """Stack frame buffer into flat observation (T*N*F,)."""
        stacked = np.stack(list(self._blue_buf), axis=0)  # (T, N, F)
        return stacked.ravel().astype(np.float32)

    def _get_opponent_action(self) -> np.ndarray:
        """Get opponent action from frozen model or random."""
        if self._opponent_encoder is None or self._opponent_policy is None:
            return np.random.uniform(-1, 1, size=8).astype(np.float32)

        # Build opponent's stacked window
        window = np.stack(list(self._orange_buf), axis=0)  # (T, N, F)
        window_t = torch.tensor(
            window[np.newaxis], dtype=torch.float32  # (1, T, N, F)
        )

        with torch.no_grad():
            emb = self._opponent_encoder(window_t, self._entity_type_ids)
            action, _ = self._opponent_policy.act(emb.cpu().numpy())

        return action.astype(np.float32)

    def _compute_sparse_reward(self, done: bool) -> float:
        """
        Sparse reward from game state. Only nonzero when a goal is scored.

        We check the underlying rlgym-sim reward which uses get_final_reward
        on episode end. For mid-episode steps, reward is always 0.
        """
        if not done:
            return 0.0
        # rlgym-sim returns per-player rewards; index 0 = blue (our agent)
        # The reward was already computed by SparseGoalReward in the step call.
        # We need to extract it. Since rlgym_sim.step returns rewards,
        # we stored them in the step method above — but actually, the rewards
        # from rlgym_sim are step rewards + final rewards combined on the last step.
        # Let's just check ball position directly.
        try:
            state = self._env._prev_state
            ball_y = state.ball.position[1]
            if ball_y > _GOAL_Y:
                return 1.0   # blue scored
            elif ball_y < -_GOAL_Y:
                return -1.0  # orange scored
        except Exception:
            pass
        return 0.0  # timeout or error
