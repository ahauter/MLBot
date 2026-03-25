"""
RLGym-sim environment components
=================================
Bridges rlgym-sim's API to the ScenarioConfig + encoder token format.

Components
----------
TokenObsBuilder          — serialises the token matrix to a flat numpy array
                           (N_TOKENS * TOKEN_FEATURES,) per player; decoded by
                           rlgym_obs_to_tokens() in encoder.py.
ScenarioStateSetter      — samples an initial state from a ScenarioConfig.
ScenarioRewardFn         — per-step and terminal rewards from a ScenarioConfig.
ScenarioTerminalCondition— ends episodes on goal / timeout from a ScenarioConfig.

All classes expect to be replaced/reset at the start of each episode via
their respective rlgym-sim reset() hooks, which accept the new scenario.
The training loop calls env.reset() with a new scenario each episode; the
RLGymEnv wrapper tears down and rebuilds the env so a fresh set of
components is instantiated.

rlgym-sim PhysicsObject conventions used here
----------------------------------------------
  obj.position          np.array([x, y, z])
  obj.linear_velocity   np.array([vx, vy, vz])
  obj.angular_velocity  np.array([avx, avy, avz])
  obj.euler_angles()    np.array([pitch, yaw, roll])   ← rlgym convention
                        (use .yaw() / .pitch() / .roll() helpers to avoid index ambiguity)

PlayerData fields
-----------------
  player.car_data       PhysicsObject (position, velocity, etc.)
  player.boost_amount   float 0..1  (multiply by 100 to get the 0-100 scale)
  player.team_num       0 = blue, 1 = orange
  player.inverted_car_data  PhysicsObject from the orange perspective
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

import numpy as np

# ── rlgym-sim base classes (module-level import so missing lib gives a clear error)
try:
    from rlgym_sim.utils import ObsBuilder, RewardFunction, TerminalCondition
    from rlgym_sim.utils.state_setters import StateSetter, StateWrapper
    from rlgym_sim.utils.gamestates import GameState, PlayerData
except ImportError as _err:
    raise ImportError(
        'rlgym-sim is required for the rlgym environment backend.\n'
        'Install it with:  pip install rlgym-sim'
    ) from _err

from scenarios.scenario_config import ScenarioConfig, RewardConfig

# Encoder constants (imported lazily to avoid circular imports at module level)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import (
    FIELD_X, FIELD_Y, CEILING_Z,
    MAX_VEL, MAX_ANG_VEL, MAX_BOOST,
    MAX_SCORE, MAX_TIME,
    N_TOKENS, TOKEN_FEATURES,
)

# ── big boost pad metadata ────────────────────────────────────────────────────
# Standard Rocket League arena — 6 large (100%) boost pads.
# Positions are in Unreal Engine units (1 uu ≈ 1 cm).
# Indices into rlgym-sim's boost_pads array (34 total pads in standard arena).
_BIG_PAD_INDICES = [3, 4, 15, 18, 29, 30]
_BIG_PAD_POSITIONS = np.array([
    [-3584.0,     0.0, 73.0],
    [ 3584.0,     0.0, 73.0],
    [-3072.0,  4096.0, 73.0],
    [ 3072.0,  4096.0, 73.0],
    [-3072.0, -4096.0, 73.0],
    [ 3072.0, -4096.0, 73.0],
], dtype=np.float32)

# ── goal detection ─────────────────────────────────────────────────────────────
_GOAL_Y     = 5124.0
_TIMEOUT_DEFAULT = 20.0  # seconds; overridden by scenario config


# ── TokenObsBuilder ───────────────────────────────────────────────────────────

class TokenObsBuilder(ObsBuilder):
    """
    Builds a flat (N_TOKENS * TOKEN_FEATURES,) observation per player.

    rlgym_obs_to_tokens() in encoder.py reshapes it back to
    (1, N_TOKENS, TOKEN_FEATURES) before it enters the encoder.

    Observation from each player's perspective:
      token 0  ball
      token 1  own car   (boost visible)
      token 2  opponent  (boost intentionally hidden → 0.0)
      3-8      big boost pads
      token 9  game state (score diff from own perspective, time rem, overtime)
    """

    def reset(self, initial_state: GameState) -> None:
        pass   # stateless builder

    def build_obs(
        self,
        player:          PlayerData,
        state:           GameState,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        own_team = player.team_num   # 0 = blue, 1 = orange

        # Identify opponent (first player on the other team)
        opponent: Optional[PlayerData] = None
        for p in state.players:
            if p.team_num != own_team:
                opponent = p
                break

        ball = state.ball

        # ── token 0: ball ─────────────────────────────────────────────────────
        ball_tok = np.array([
            ball.position[0]         / FIELD_X,
            ball.position[1]         / FIELD_Y,
            ball.position[2]         / CEILING_Z,
            ball.linear_velocity[0]  / MAX_VEL,
            ball.linear_velocity[1]  / MAX_VEL,
            ball.linear_velocity[2]  / MAX_VEL,
            ball.angular_velocity[0] / MAX_ANG_VEL,
            ball.angular_velocity[1] / MAX_ANG_VEL,
            ball.angular_velocity[2] / MAX_ANG_VEL,
            0.0,
        ], dtype=np.float32)

        # ── token 1: own car ──────────────────────────────────────────────────
        own       = player.car_data
        own_boost = player.boost_amount  # 0..1 — multiply by 100 for 0-100 scale

        own_tok = np.array([
            own.position[0]        / FIELD_X,
            own.position[1]        / FIELD_Y,
            own.position[2]        / CEILING_Z,
            own.linear_velocity[0] / MAX_VEL,
            own.linear_velocity[1] / MAX_VEL,
            own.linear_velocity[2] / MAX_VEL,
            own.yaw()              / math.pi,
            own.pitch()            / math.pi,
            own.roll()             / math.pi,
            (own_boost * 100.0)    / MAX_BOOST,
        ], dtype=np.float32)

        # ── token 2: opponent — boost hidden ──────────────────────────────────
        if opponent is not None:
            opp     = opponent.car_data
            opp_tok = np.array([
                opp.position[0]        / FIELD_X,
                opp.position[1]        / FIELD_Y,
                opp.position[2]        / CEILING_Z,
                opp.linear_velocity[0] / MAX_VEL,
                opp.linear_velocity[1] / MAX_VEL,
                opp.linear_velocity[2] / MAX_VEL,
                opp.yaw()              / math.pi,
                opp.pitch()            / math.pi,
                opp.roll()             / math.pi,
                0.0,   # <-- opponent boost intentionally hidden
            ], dtype=np.float32)
        else:
            opp_tok = np.zeros(TOKEN_FEATURES, dtype=np.float32)

        # ── tokens 3-8: big boost pads ────────────────────────────────────────
        pad_toks = []
        for i, idx in enumerate(_BIG_PAD_INDICES):
            active = float(state.boost_pads[idx]) if idx < len(state.boost_pads) else 0.0
            pos    = _BIG_PAD_POSITIONS[i]
            pad_toks.append(np.array([
                pos[0] / FIELD_X,
                pos[1] / FIELD_Y,
                pos[2] / CEILING_Z,
                active,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32))

        # ── token 9: game state ───────────────────────────────────────────────
        # rlgym-sim doesn't track score/time directly in GameState by default.
        # We expose placeholders; training/RLGymEnv injects a TimeLimit terminal
        # condition so the episode ends anyway.
        gs_tok = np.zeros(TOKEN_FEATURES, dtype=np.float32)
        # score_diff and time_remaining remain 0 (unknowable mid-rollout in sim)

        tokens = np.stack(
            [ball_tok, own_tok, opp_tok] + pad_toks + [gs_tok],
            axis=0,
        )   # (N_TOKENS, TOKEN_FEATURES)

        return tokens.ravel()   # (N_TOKENS * TOKEN_FEATURES,)


# ── ScenarioStateSetter ───────────────────────────────────────────────────────

class ScenarioStateSetter(StateSetter):
    """Sets the initial game state by sampling the given ScenarioConfig."""

    def __init__(self, scenario: ScenarioConfig) -> None:
        self._scenario = scenario

    def reset(self, state_wrapper: StateWrapper) -> None:
        cfg = self._scenario.initial_state

        # ── ball ──────────────────────────────────────────────────────────────
        state_wrapper.ball.position = [
            cfg.ball.location.x.sample(),
            cfg.ball.location.y.sample(),
            cfg.ball.location.z.sample(),
        ]
        state_wrapper.ball.linear_velocity = [
            cfg.ball.velocity.x.sample(),
            cfg.ball.velocity.y.sample(),
            cfg.ball.velocity.z.sample(),
        ]
        state_wrapper.ball.angular_velocity = [0.0, 0.0, 0.0]

        # ── blue car (index 0) ────────────────────────────────────────────────
        if len(state_wrapper.cars) > 0:
            blue = state_wrapper.cars[0]
            blue.position = [
                cfg.blue.location.x.sample(),
                cfg.blue.location.y.sample(),
                cfg.blue.location.z.sample(),
            ]
            blue.set_rot(pitch=0.0, yaw=cfg.blue.yaw.sample(), roll=0.0)
            blue.linear_velocity  = [0.0, 0.0, 0.0]
            blue.angular_velocity = [0.0, 0.0, 0.0]
            blue.boost = cfg.blue.boost.sample() / 100.0   # StateWrapper uses 0..1

        # ── orange car (index 1) ──────────────────────────────────────────────
        if len(state_wrapper.cars) > 1:
            orange = state_wrapper.cars[1]
            orange.position = [
                cfg.orange.location.x.sample(),
                cfg.orange.location.y.sample(),
                cfg.orange.location.z.sample(),
            ]
            orange.set_rot(pitch=0.0, yaw=cfg.orange.yaw.sample(), roll=0.0)
            orange.linear_velocity  = [0.0, 0.0, 0.0]
            orange.angular_velocity = [0.0, 0.0, 0.0]
            orange.boost = cfg.orange.boost.sample() / 100.0


# ── ScenarioRewardFn ──────────────────────────────────────────────────────────

class ScenarioRewardFn(RewardFunction):
    """
    Per-step reward from ScenarioConfig.reward.{blue,orange}.step events.

    Supported step event types:
      ball_toward_goal   — ball vy / MAX_VEL  (positive = toward orange goal)
      ball_from_goal     — -ball vy / MAX_VEL
      car_near_ball      — max(0, 1 - dist(car, ball) / 5000)

    Terminal rewards (goal_scored, timeout, etc.) are handled separately
    by the training loop via ScenarioTerminalCondition.
    """

    def __init__(self, scenario: ScenarioConfig) -> None:
        self._scenario = scenario

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self,
        player:          PlayerData,
        state:           GameState,
        previous_action: np.ndarray,
    ) -> float:
        if player.team_num == 0:
            reward_cfg: RewardConfig = self._scenario.reward.blue
        else:
            reward_cfg: RewardConfig = self._scenario.reward.orange

        ball = state.ball
        car  = player.car_data
        total = 0.0

        for evt in reward_cfg.step:
            w = evt.weight
            t = evt.type
            if t == 'ball_toward_goal':
                total += w * float(np.clip(ball.linear_velocity[1] / MAX_VEL, -1.0, 1.0))
            elif t == 'ball_from_goal':
                total += w * float(np.clip(-ball.linear_velocity[1] / MAX_VEL, -1.0, 1.0))
            elif t == 'car_near_ball':
                d = float(np.linalg.norm(car.position - ball.position))
                total += w * max(0.0, 1.0 - d / 5000.0)
        return total

    def get_final_reward(
        self,
        player:          PlayerData,
        state:           GameState,
        previous_action: np.ndarray,
    ) -> float:
        # Terminal reward (goal scored) — determine winner from ball position
        ball_y = state.ball.position[1]
        goal_scored_blue   = ball_y >  _GOAL_Y
        goal_scored_orange = ball_y < -_GOAL_Y

        if player.team_num == 0:
            reward_cfg = self._scenario.reward.blue
            scored = goal_scored_blue
        else:
            reward_cfg = self._scenario.reward.orange
            scored = goal_scored_orange

        total = 0.0
        for evt in reward_cfg.terminal:
            if evt.type == 'goal_scored' and scored:
                total += evt.value * evt.weight
        return total


# ── ScenarioTerminalCondition ─────────────────────────────────────────────────

class ScenarioTerminalCondition(TerminalCondition):
    """
    Ends an episode when a goal is scored or when the scenario timeout fires.

    Both conditions are evaluated for all players simultaneously (rlgym-sim
    shares one terminal flag across the episode).
    """

    def __init__(self, scenario: ScenarioConfig) -> None:
        self._scenario = scenario
        self._start_time: float = 0.0
        self._timeout: float = _TIMEOUT_DEFAULT
        self._load_timeout()

    def _load_timeout(self) -> None:
        """Extract the smallest timeout from the scenario reward configs."""
        vals = []
        for cfg in [self._scenario.reward.blue, self._scenario.reward.orange]:
            for evt in cfg.terminal:
                if evt.type == 'timeout' and hasattr(evt, 'seconds'):
                    vals.append(float(evt.seconds))
        if vals:
            self._timeout = min(vals)

    def reset(self, initial_state: GameState) -> None:
        self._start_time = time.monotonic()

    def is_terminal(self, current_state: GameState) -> bool:
        # Goal scored: ball past either goal line
        ball_y = current_state.ball.position[1]
        if abs(ball_y) > _GOAL_Y:
            return True

        # Timeout
        if time.monotonic() - self._start_time >= self._timeout:
            return True

        return False
