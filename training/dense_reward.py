"""
Dense Reward Function for Baseline Experiments
===============================================
Axis 4 intervention: hand-crafted dense reward shaping.

Composes multiple weighted reward components that provide per-step signal
in addition to the sparse terminal goal reward (+1/-1).

Selectable via ``--reward dense`` in train.py.  The sparse baseline remains
the default and is unaffected by this module.

Components
----------
1. ball_touch            — event reward on first contact with ball
2. shot_line             — proximity of car to the ball→own-goal line
3. ball_proximity        — distance from car to ball
4. offensive_goal_angle  — angle subtended by enemy goalposts at ball position
5. defensive_goal_angle  — angle subtended by own goalposts at ball position (penalty)
6. grounded              — binary: car on the ground
7. car_velocity          — car speed (normalized)
8. velocity_shot_axis    — car velocity projected onto ball→enemy-goal axis
9. boost_amount          — current boost level (normalized)
10. ball_toward_goal     — ball velocity projected onto ball→enemy-goal axis
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

# ── Arena constants ──────────────────────────────────────────────────────────
_GOAL_Y = 5124.0
_MAX_VEL = 2300.0
_MAX_BALL_VEL = 6000.0
_FIELD_DIAG = float(np.linalg.norm([4096.0, 5120.0, 2044.0]))
_GOAL_HALF_WIDTH = 893.0   # from scenario_visualizer.GOAL_HW

# ── Component thresholds ────────────────────────────────────────────────────
_TOUCH_RADIUS = 200.0       # car-ball distance to count as a touch
_BALL_PROX_MAX = 6000.0     # proximity reward falls to 0 at this distance
_SHOT_LINE_MAX = 3000.0     # shot-line reward falls to 0 at this distance
_GROUND_THRESH = 50.0       # car z below this counts as grounded

# ── Default weights ─────────────────────────────────────────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    'ball_touch':            0.05,
    'shot_line':             0.003,
    'ball_proximity':        0.005,
    'offensive_goal_angle':  0.010,
    'defensive_goal_angle':  0.003,
    'grounded':              0.001,
    'car_velocity':          0.001,
    'velocity_shot_axis':    0.003,
    'boost_amount':          0.001,
    'ball_toward_goal':      0.005,
}


# ── Individual component functions ──────────────────────────────────────────

def _enemy_goal(team: int) -> np.ndarray:
    """Center of the enemy goal for the given team."""
    return np.array([0.0, _GOAL_Y if team == 0 else -_GOAL_Y, 0.0])


def _own_goal(team: int) -> np.ndarray:
    """Center of own goal for the given team."""
    return np.array([0.0, -_GOAL_Y if team == 0 else _GOAL_Y, 0.0])


def ball_touch(car_pos: np.ndarray, ball_pos: np.ndarray,
               prev_dist: float) -> float:
    """1.0 on first frame of contact, 0.0 otherwise."""
    dist = float(np.linalg.norm(car_pos - ball_pos))
    if dist < _TOUCH_RADIUS and prev_dist >= _TOUCH_RADIUS:
        return 1.0
    return 0.0


def shot_line(car_pos: np.ndarray, ball_pos: np.ndarray,
              own_goal_pos: np.ndarray) -> float:
    """How close the car is to the line from ball to own goal. [0, 1]."""
    ball_to_goal = own_goal_pos - ball_pos
    ball_to_car = car_pos - ball_pos
    denom = float(np.dot(ball_to_goal, ball_to_goal))
    if denom < 1e-8:
        return 0.0
    t = np.clip(np.dot(ball_to_car, ball_to_goal) / denom, 0.0, 1.0)
    closest = ball_pos + float(t) * ball_to_goal
    dist = float(np.linalg.norm(car_pos - closest))
    return max(0.0, 1.0 - dist / _SHOT_LINE_MAX)


def ball_proximity(car_pos: np.ndarray, ball_pos: np.ndarray) -> float:
    """Proximity reward: 1 when on top of ball, 0 at MAX distance. [0, 1]."""
    dist = float(np.linalg.norm(car_pos - ball_pos))
    return max(0.0, 1.0 - dist / _BALL_PROX_MAX)


def goal_angle(ball_pos: np.ndarray, goal_y: float) -> float:
    """
    Angle (radians) subtended by the goal opening at ball_pos.
    Projects to the XY plane; returns a value in [0, π].
    Larger when the ball is closer to and more centered on the goal.
    """
    left_post = np.array([-_GOAL_HALF_WIDTH, goal_y])
    right_post = np.array([_GOAL_HALF_WIDTH, goal_y])
    bxy = ball_pos[:2]

    v_left = left_post - bxy
    v_right = right_post - bxy

    cos_theta = np.dot(v_left, v_right) / (
        np.linalg.norm(v_left) * np.linalg.norm(v_right) + 1e-8
    )
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def grounded(car_pos: np.ndarray) -> float:
    """1.0 if car is on the ground, 0.0 otherwise."""
    return 1.0 if car_pos[2] < _GROUND_THRESH else 0.0


def car_velocity(car_vel: np.ndarray) -> float:
    """Car speed normalized by max velocity. [0, 1]."""
    return min(1.0, float(np.linalg.norm(car_vel)) / _MAX_VEL)


def velocity_shot_axis(car_vel: np.ndarray, ball_pos: np.ndarray,
                       enemy_goal_pos: np.ndarray) -> float:
    """Car velocity projected onto ball→enemy-goal direction. [-1, 1]."""
    axis = enemy_goal_pos - ball_pos
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return 0.0
    axis_unit = axis / norm
    return float(np.clip(np.dot(car_vel, axis_unit) / _MAX_VEL, -1.0, 1.0))


def ball_toward_goal(ball_vel: np.ndarray, ball_pos: np.ndarray,
                     enemy_goal_pos: np.ndarray) -> float:
    """Ball velocity projected onto ball→enemy-goal axis. [-1, 1]."""
    axis = enemy_goal_pos - ball_pos
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return 0.0
    axis_unit = axis / norm
    return float(np.clip(np.dot(ball_vel, axis_unit) / _MAX_BALL_VEL, -1.0, 1.0))


# ── Composite reward function ──────────────────────────────────────────────

class DenseRewardFunction(RewardFunction):
    """
    Weighted sum of dense reward components + sparse terminal goal reward.

    Parameters
    ----------
    weights : dict, optional
        Override default component weights.  Keys not present use defaults.
        Set a weight to 0.0 to disable a component.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self._weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)
        # Per-player state for ball-touch detection: {player_id: prev_dist}
        self._prev_ball_dist: Dict[int, float] = {}

    @property
    def num_components(self) -> int:
        """Number of active (non-zero weight) step components."""
        return sum(1 for w in self._weights.values() if w != 0.0)

    @property
    def max_step_reward(self) -> float:
        """Theoretical maximum per-step reward (all components at their peak)."""
        return sum(max(0.0, v) for v in self._weights.values())

    def reset(self, initial_state: GameState) -> None:
        self._prev_ball_dist.clear()

    def get_reward(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
    ) -> float:
        car_pos = player.car_data.position
        car_vel = player.car_data.linear_velocity
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        team = player.team_num
        pid = id(player)

        enemy = _enemy_goal(team)
        own = _own_goal(team)

        # Ball-touch state tracking
        cur_dist = float(np.linalg.norm(car_pos - ball_pos))
        prev_dist = self._prev_ball_dist.get(pid, cur_dist)
        self._prev_ball_dist[pid] = cur_dist

        w = self._weights
        total = 0.0

        if w.get('ball_touch', 0.0):
            total += w['ball_touch'] * ball_touch(car_pos, ball_pos, prev_dist)

        if w.get('shot_line', 0.0):
            total += w['shot_line'] * shot_line(car_pos, ball_pos, own)

        if w.get('ball_proximity', 0.0):
            total += w['ball_proximity'] * ball_proximity(car_pos, ball_pos)

        if w.get('offensive_goal_angle', 0.0):
            total += w['offensive_goal_angle'] * goal_angle(ball_pos, enemy[1])

        if w.get('defensive_goal_angle', 0.0):
            total -= w['defensive_goal_angle'] * goal_angle(ball_pos, own[1])

        if w.get('grounded', 0.0):
            total += w['grounded'] * grounded(car_pos)

        if w.get('car_velocity', 0.0):
            total += w['car_velocity'] * car_velocity(car_vel)

        if w.get('velocity_shot_axis', 0.0):
            total += w['velocity_shot_axis'] * velocity_shot_axis(
                car_vel, ball_pos, enemy
            )

        if w.get('boost_amount', 0.0):
            total += w['boost_amount'] * (player.boost_amount / 100.0)

        if w.get('ball_toward_goal', 0.0):
            total += w['ball_toward_goal'] * ball_toward_goal(
                ball_vel, ball_pos, enemy
            )

        return total

    def get_final_reward(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
    ) -> float:
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
