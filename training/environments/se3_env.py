"""
SE3 Gymnasium Environment
=========================
Wraps rlgym-sim as a gymnasium.Env for SE(3) spectral field training.

The env maintains SE3 field state (coefficients) internally and produces
SE3_OBS_DIM-dim observations: [raw_state (63) | coefficients (576)].

The SE3Encoder in the algorithm recomputes the coefficient update
differentiably during training.  The env's numpy update is used only
during rollout collection (no grad needed).

Observation: (639,) float32
Action:      (8,)   float32 — 5 analog [-1,1] + 3 binary (threshold 0.5)
"""
from __future__ import annotations

import math
import sys
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from pathlib import Path

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from encoder import (
    FIELD_X, FIELD_Y, CEILING_Z,
    MAX_VEL, MAX_ANG_VEL, MAX_BOOST, MAX_SCORE, MAX_TIME,
    TOKEN_FEATURES, N_TOKENS,
)
from se3_field import (
    SE3_OBS_DIM, RAW_STATE_DIM, COEFF_DIM, N_OBJECTS, K,
    make_initial_coefficients, update_coefficients_np,
    pack_observation, euler_to_quaternion_batch,
    _BALL_OFF, _EGO_OFF, _OPP_OFF, _PAD_OFF, _GS_OFF, _PREV_VEL_OFF,
    _PREV_EGO_VEL_OFF, _PREV_OPP_VEL_OFF,
)

_GOAL_Y = 5124.0


class SE3GymEnv(gym.Env):
    """
    Gymnasium wrapper for 1v1 SE3 spectral field training.

    Parameters
    ----------
    t_window : int
        Accepted for interface compat with train.py's _env_worker. Unused.
    reward_type : str
        'sparse' (default) or 'dense'.
    dense_reward_weights : dict or None
        Weights for dense reward components.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        t_window: int = 1,
        tick_skip: int = 8,
        max_steps: int = 4500,
        reward_type: str = 'sparse',
        dense_reward_weights: dict | None = None,
    ):
        super().__init__()
        self.t_window = t_window  # unused but stored for interface compat
        self.tick_skip = tick_skip
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.dense_reward_weights = dense_reward_weights

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(SE3_OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self._env = None
        self._step_count = 0

        # SE3 field state (blue / agent perspective)
        self._coefficients: np.ndarray = make_initial_coefficients()
        self._prev_ball_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_ego_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_opp_vel: np.ndarray = np.zeros(3, dtype=np.float32)

        # SE3 field state (orange / opponent perspective)
        self._orange_coefficients: np.ndarray = make_initial_coefficients()
        self._orange_prev_ball_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._orange_prev_ego_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._orange_prev_opp_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._last_orange_obs: np.ndarray = np.zeros(SE3_OBS_DIM, dtype=np.float32)

        # Encoder params (numpy, synced from algorithm)
        self._k_spatial: np.ndarray = np.random.randn(N_OBJECTS, K, 3).astype(np.float32) * 1.0
        self._quaternions: np.ndarray = np.random.randn(N_OBJECTS, K, 4).astype(np.float32)
        norms = np.linalg.norm(self._quaternions, axis=-1, keepdims=True)
        self._quaternions /= np.maximum(norms, 1e-8)
        self._lr: np.ndarray = np.full(N_OBJECTS, 0.05, dtype=np.float32)
        self._W_interact: np.ndarray = np.zeros((N_OBJECTS, N_OBJECTS), dtype=np.float32)

    def set_encoder_params(self, k_spatial: np.ndarray, quaternions: np.ndarray,
                           lr: np.ndarray,
                           W_interact: Optional[np.ndarray] = None) -> None:
        """Sync encoder parameters from the algorithm (called periodically)."""
        self._k_spatial = k_spatial.copy()
        self._quaternions = quaternions.copy()
        self._lr = lr.copy()
        if W_interact is not None:
            self._W_interact = W_interact.copy()

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

        # Reset field state
        self._coefficients = make_initial_coefficients()
        self._prev_ball_vel = np.zeros(3, dtype=np.float32)
        self._prev_ego_vel = np.zeros(3, dtype=np.float32)
        self._prev_opp_vel = np.zeros(3, dtype=np.float32)

        # Reset orange field state
        self._orange_coefficients = make_initial_coefficients()
        self._orange_prev_ball_vel = np.zeros(3, dtype=np.float32)
        self._orange_prev_ego_vel = np.zeros(3, dtype=np.float32)
        self._orange_prev_opp_vel = np.zeros(3, dtype=np.float32)

        # Build raw state from rlgym obs
        raw_state = self._token_obs_to_raw_state(
            blue_obs, self._prev_ball_vel, self._prev_ego_vel, self._prev_opp_vel)
        obs = pack_observation(raw_state, self._coefficients)

        # Cache orange obs
        orange_raw = self._token_obs_to_raw_state(
            orange_obs, self._orange_prev_ball_vel,
            self._orange_prev_ego_vel, self._orange_prev_opp_vel)
        self._last_orange_obs = pack_observation(orange_raw, self._orange_coefficients)

        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Random opponent
        opp_action = np.random.uniform(-1, 1, size=8).astype(np.float32)

        obs_list, rewards, terminated, _info = self._env.step(
            np.stack([action, opp_action], axis=0))
        blue_obs, orange_obs = self._parse_obs(obs_list)
        self._step_count += 1

        # Build raw state from rlgym tokens
        raw_state = self._token_obs_to_raw_state(
            blue_obs, self._prev_ball_vel, self._prev_ego_vel, self._prev_opp_vel)

        # Update coefficients (numpy, no grad)
        self._coefficients = update_coefficients_np(
            self._k_spatial, self._quaternions, self._lr,
            self._coefficients, raw_state, W_interact=self._W_interact)

        # Cache prev velocities for next step
        self._prev_ball_vel = raw_state[_BALL_OFF + 3:_BALL_OFF + 6].copy()
        self._prev_ego_vel = raw_state[_EGO_OFF + 3:_EGO_OFF + 6].copy()
        self._prev_opp_vel = raw_state[_OPP_OFF + 3:_OPP_OFF + 6].copy()

        obs = pack_observation(raw_state, self._coefficients)

        # Update orange perspective
        orange_raw = self._token_obs_to_raw_state(
            orange_obs, self._orange_prev_ball_vel,
            self._orange_prev_ego_vel, self._orange_prev_opp_vel)
        self._orange_coefficients = update_coefficients_np(
            self._k_spatial, self._quaternions, self._lr,
            self._orange_coefficients, orange_raw, W_interact=self._W_interact)
        self._orange_prev_ball_vel = orange_raw[_BALL_OFF + 3:_BALL_OFF + 6].copy()
        self._orange_prev_ego_vel = orange_raw[_EGO_OFF + 3:_EGO_OFF + 6].copy()
        self._orange_prev_opp_vel = orange_raw[_OPP_OFF + 3:_OPP_OFF + 6].copy()
        self._last_orange_obs = pack_observation(orange_raw, self._orange_coefficients)

        blue_reward = float(rewards[0])
        timed_out = self._step_count >= self.max_steps
        done = bool(terminated or timed_out)

        ball_y = _info['state'].ball.position[1]
        goal = (1 if ball_y > _GOAL_Y else -1) if terminated else 0
        info = {'goal': goal}

        return obs, blue_reward, done, False, info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def get_opponent_obs(self) -> np.ndarray:
        """Return current SE3 observation from the orange player's perspective."""
        return self._last_orange_obs.copy()

    def step_with_opponent_action(
        self, action: np.ndarray, opp_action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step with an externally-computed opponent action.

        Same as step() but uses the provided opp_action instead of random
        (for batched GPU inference in the training loop).
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        opp_action = np.clip(opp_action, -1.0, 1.0).astype(np.float32)

        obs_list, rewards, terminated, _info = self._env.step(
            np.stack([action, opp_action], axis=0))
        blue_obs, orange_obs = self._parse_obs(obs_list)
        self._step_count += 1

        raw_state = self._token_obs_to_raw_state(
            blue_obs, self._prev_ball_vel, self._prev_ego_vel, self._prev_opp_vel)
        self._coefficients = update_coefficients_np(
            self._k_spatial, self._quaternions, self._lr,
            self._coefficients, raw_state, W_interact=self._W_interact)
        self._prev_ball_vel = raw_state[_BALL_OFF + 3:_BALL_OFF + 6].copy()
        self._prev_ego_vel = raw_state[_EGO_OFF + 3:_EGO_OFF + 6].copy()
        self._prev_opp_vel = raw_state[_OPP_OFF + 3:_OPP_OFF + 6].copy()
        obs = pack_observation(raw_state, self._coefficients)

        orange_raw = self._token_obs_to_raw_state(
            orange_obs, self._orange_prev_ball_vel,
            self._orange_prev_ego_vel, self._orange_prev_opp_vel)
        self._orange_coefficients = update_coefficients_np(
            self._k_spatial, self._quaternions, self._lr,
            self._orange_coefficients, orange_raw, W_interact=self._W_interact)
        self._orange_prev_ball_vel = orange_raw[_BALL_OFF + 3:_BALL_OFF + 6].copy()
        self._orange_prev_ego_vel = orange_raw[_EGO_OFF + 3:_EGO_OFF + 6].copy()
        self._orange_prev_opp_vel = orange_raw[_OPP_OFF + 3:_OPP_OFF + 6].copy()
        self._last_orange_obs = pack_observation(orange_raw, self._orange_coefficients)

        blue_reward = float(rewards[0])
        timed_out = self._step_count >= self.max_steps
        done = bool(terminated or timed_out)

        ball_y = _info['state'].ball.position[1]
        goal = (1 if ball_y > _GOAL_Y else -1) if terminated else 0
        info = {'goal': goal}

        return obs, blue_reward, done, False, info

    # ── internal ────────────────────────────────────────────────────────────

    def _build_env(self):
        import rlgym_sim
        from rlgym_sim.utils.action_parsers import ContinuousAction
        from rlgym_sim.utils.state_setters import DefaultState
        from environments.rlgym_components import TokenObsBuilder

        self._env = rlgym_sim.make(
            obs_builder=TokenObsBuilder(),
            action_parser=ContinuousAction(),
            state_setter=DefaultState(),
            reward_fn=self._make_reward(),
            terminal_conditions=[self._make_terminal()],
            team_size=1,
            spawn_opponents=True,
            tick_skip=self.tick_skip,
        )

    def _make_reward(self):
        from rlgym_sim.utils import RewardFunction
        from rlgym_sim.utils.gamestates import GameState, PlayerData

        class SparseGoalReward(RewardFunction):
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
                if player.team_num == 0:
                    if scored_blue:
                        return 1.0
                    elif scored_orange:
                        return -1.0
                else:
                    if scored_orange:
                        return 1.0
                    elif scored_blue:
                        return -1.0
                return 0.0

        return SparseGoalReward()

    def _make_terminal(self):
        from rlgym_sim.utils import TerminalCondition
        from rlgym_sim.utils.gamestates import GameState

        class GoalTerminal(TerminalCondition):
            def reset(self, initial_state: GameState) -> None:
                pass

            def is_terminal(self, current_state: GameState) -> bool:
                return abs(current_state.ball.position[1]) > _GOAL_Y

        return GoalTerminal()

    def _parse_obs(self, obs_result) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            obs_result = obs_result[0]
        if isinstance(obs_result, np.ndarray) and obs_result.ndim == 2:
            return obs_result[0], obs_result[1]
        if isinstance(obs_result, (list, tuple)) and len(obs_result) == 2:
            return obs_result[0], obs_result[1]
        raise ValueError(f'Unexpected obs format: {type(obs_result)}')

    def _token_obs_to_raw_state(
        self, flat_obs: np.ndarray,
        prev_ball_vel: np.ndarray,
        prev_ego_vel: np.ndarray,
        prev_opp_vel: np.ndarray,
    ) -> np.ndarray:
        """Convert TokenObsBuilder output (100,) to SE3 raw state (63,).

        TokenObsBuilder format (10 tokens × 10 features):
          token 0: ball   [x,y,z, vx,vy,vz, avx,avy,avz, 0]
          token 1: own    [x,y,z, vx,vy,vz, yaw/pi, pitch/pi, roll/pi, boost/100]
          token 2: opp    [x,y,z, vx,vy,vz, yaw/pi, pitch/pi, roll/pi, 0]
          token 3-8: pads [x,y,z, active, 0,0,0,0,0,0]
          token 9: game   [score_diff, time_rem, overtime, 0,...,0]

        All values are already normalised to [-1, 1].
        """
        tokens = flat_obs.reshape(N_TOKENS, TOKEN_FEATURES)
        raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)

        # Ball: pos(3) + vel(3)  — already normalised
        raw[_BALL_OFF:_BALL_OFF + 3] = tokens[0, :3]
        raw[_BALL_OFF + 3:_BALL_OFF + 6] = tokens[0, 3:6]

        # Ego: pos(3) + vel(3) + quat(4) + boost(1)
        raw[_EGO_OFF:_EGO_OFF + 3] = tokens[1, :3]
        raw[_EGO_OFF + 3:_EGO_OFF + 6] = tokens[1, 3:6]
        # Convert normalised Euler (yaw/pi, pitch/pi, roll/pi) to quaternion
        yaw = tokens[1, 6] * math.pi
        pitch = tokens[1, 7] * math.pi
        roll = tokens[1, 8] * math.pi
        ego_q = euler_to_quaternion_batch(
            np.array([yaw]), np.array([pitch]), np.array([roll]))[0]
        raw[_EGO_OFF + 6:_EGO_OFF + 10] = ego_q
        raw[_EGO_OFF + 10] = tokens[1, 9]  # boost (already normalised)

        # Opponent: pos(3) + vel(3) + quat(4)
        raw[_OPP_OFF:_OPP_OFF + 3] = tokens[2, :3]
        raw[_OPP_OFF + 3:_OPP_OFF + 6] = tokens[2, 3:6]
        opp_yaw = tokens[2, 6] * math.pi
        opp_pitch = tokens[2, 7] * math.pi
        opp_roll = tokens[2, 8] * math.pi
        opp_q = euler_to_quaternion_batch(
            np.array([opp_yaw]), np.array([opp_pitch]), np.array([opp_roll]))[0]
        raw[_OPP_OFF + 6:_OPP_OFF + 10] = opp_q

        # Boost pads (6): pos(3) + active(1)
        for i in range(6):
            tok_idx = 3 + i
            off = _PAD_OFF + i * 4
            raw[off:off + 3] = tokens[tok_idx, :3]
            raw[off + 3] = tokens[tok_idx, 3]

        # Game state
        raw[_GS_OFF:_GS_OFF + 3] = tokens[9, :3]

        # Previous velocities (for contact detection + acceleration channel)
        raw[_PREV_VEL_OFF:_PREV_VEL_OFF + 3] = prev_ball_vel
        raw[_PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3] = prev_ego_vel
        raw[_PREV_OPP_VEL_OFF:_PREV_OPP_VEL_OFF + 3] = prev_opp_vel

        return raw
