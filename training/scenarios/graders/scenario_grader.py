"""
ScenarioGrader
==============
Pure-Python terminal-event detector for adversarial training scenarios.

Accepts any duck-typed object that looks like a RLBot GameTickPacket
(has .game_ball.physics.location, .game_cars[i].physics.location, etc.)
so it works with both live RLBot packets and dummy packets built from
ScenarioConfig during headless testing.

Usage
-----
    grader = ScenarioGrader(scenario_config)
    while True:
        event = grader.check(packet, elapsed_seconds)
        if event:
            reward = reward_for_event(scenario.reward.blue, event)
            break  # episode over
"""

from __future__ import annotations

import math
from typing import Optional

from scenarios.scenario_config import RewardConfig, ScenarioConfig

# ── arena constants ────────────────────────────────────────────────────────────

FIELD_X = 4096.0
FIELD_Y = 5120.0
GOAL_Y  = 5124.0

# Event detection thresholds
AERIAL_HIT_Z        = 300.0
AERIAL_HIT_DIST     = 150.0
BALL_GROUNDED_Z     = 110.0
DANGER_ZONE_Y       = 3500.0   # ball must pass here before save_made can fire
SAVE_MADE_DIST      = 300.0
SAVE_MADE_BALL_Y    = 3500.0
BALL_CLEARED_Y      = -500.0
PASS_RECEIVED_DIST  = 200.0
INTERCEPTION_DIST   = 200.0


# ── helpers ────────────────────────────────────────────────────────────────────

def _dist3(a, b) -> float:
    """Euclidean distance between two location objects with .x .y .z attributes."""
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _find_timeout(reward_cfg: RewardConfig):
    """Return the first timeout RewardEvent from a config, or None."""
    for evt in reward_cfg.terminal:
        if evt.type == 'timeout':
            return evt
    return None


def _unified_timeout(blue_evt, orange_evt) -> float:
    """Min timeout of both sides; math.inf if neither side has one."""
    vals = []
    if blue_evt is not None:
        vals.append(blue_evt.seconds)
    if orange_evt is not None:
        vals.append(orange_evt.seconds)
    return min(vals) if vals else math.inf


# ── public helpers used by training loop ──────────────────────────────────────

def reward_for_event(reward_cfg: RewardConfig, event_type: str) -> float:
    """
    Sum value * weight for all terminal events matching event_type.
    Returns 0.0 if event_type is not listed for this side.
    """
    total = 0.0
    for evt in reward_cfg.terminal:
        if evt.type == event_type:
            total += evt.value * evt.weight
    return total


def step_reward(packet, reward_cfg: RewardConfig, car_idx: int) -> float:
    """
    Compute the dense per-step reward for one car.

    Recognised step event types:
      ball_toward_goal      — ball.vy / 2300  (positive = toward orange goal, +Y)
      ball_from_goal        — -ball.vy / 2300
      car_near_ball         — max(0, 1 - dist(car, ball) / 5000)
      ball_toward_teammate  — approximate: same as ball_toward_goal

    Unknown types are silently ignored so new YAML step events don't crash training.
    """
    ball = packet.game_ball.physics
    car  = packet.game_cars[car_idx].physics
    total = 0.0

    for evt in reward_cfg.step:
        w = evt.weight
        t = evt.type
        if t == 'ball_toward_goal':
            total += w * _clamp(ball.velocity.y / 2300.0, -1.0, 1.0)
        elif t == 'ball_from_goal':
            total += w * _clamp(-ball.velocity.y / 2300.0, -1.0, 1.0)
        elif t == 'car_near_ball':
            d = _dist3(car.location, ball.location)
            total += w * max(0.0, 1.0 - d / 5000.0)
        elif t == 'ball_toward_teammate':
            total += w * _clamp(ball.velocity.y / 2300.0, -1.0, 1.0)
        # unknown types: intentionally ignored

    return total


# ── grader class ──────────────────────────────────────────────────────────────

class ScenarioGrader:
    """
    Stateful terminal-event detector for one episode.

    Instantiate fresh per episode. Call check() once per game tick in order.
    Returns the event_type string when a terminal event is detected, None otherwise.

    Priority order (important — goal_scored must precede ball_out_play):
      1  goal_scored
      2  ball_out_play
      3  aerial_hit
      4  ball_grounded   (guarded: only after ball was airborne)
      5  save_made       (guarded: only after ball entered danger zone)
      6  ball_cleared
      7  pass_received
      8  interception
      9  timeout
    """

    def __init__(self, config: ScenarioConfig) -> None:
        self._config = config

        # Internal state
        self._ball_was_in_danger_zone: bool = False
        self._ball_was_airborne:       bool = False

        blue_timeout   = _find_timeout(config.reward.blue)
        orange_timeout = _find_timeout(config.reward.orange)
        self._timeout_seconds: float = _unified_timeout(blue_timeout, orange_timeout)

    def check(self, packet, elapsed_seconds: float) -> Optional[str]:
        """
        Inspect packet and return a terminal event_type string, or None.

        packet must expose:
          packet.game_ball.physics.location  (.x .y .z)
          packet.game_ball.physics.velocity  (.x .y .z)
          packet.game_cars[0].physics.location
          packet.game_cars[1].physics.location
        """
        ball   = packet.game_ball.physics
        blue   = packet.game_cars[0].physics
        orange = packet.game_cars[1].physics

        bx = ball.location.x
        by = ball.location.y
        bz = ball.location.z

        # 1. goal_scored — check BEFORE ball_out_play (goal Y > field Y)
        if abs(by) > GOAL_Y:
            return 'goal_scored'

        # 2. ball_out_play
        if abs(bx) > FIELD_X or abs(by) > FIELD_Y:
            return 'ball_out_play'

        # 3. aerial_hit — ball high AND a car is near it
        if bz > AERIAL_HIT_Z:
            if (_dist3(blue.location,   ball.location) < AERIAL_HIT_DIST or
                    _dist3(orange.location, ball.location) < AERIAL_HIT_DIST):
                return 'aerial_hit'

        # 4. ball_grounded — only after ball was previously airborne
        #    (prevents tick-0 false positives when scenarios start with ball at z=100)
        if bz > BALL_GROUNDED_Z:
            self._ball_was_airborne = True
        if self._ball_was_airborne and bz < BALL_GROUNDED_Z:
            return 'ball_grounded'

        # 5. save_made — ball near orange goal AND orange car intercepted
        if by > DANGER_ZONE_Y:
            self._ball_was_in_danger_zone = True
        if (self._ball_was_in_danger_zone
                and by > SAVE_MADE_BALL_Y
                and _dist3(orange.location, ball.location) < SAVE_MADE_DIST):
            self._ball_was_in_danger_zone = False   # reset to avoid re-fire
            return 'save_made'

        # 6. ball_cleared — ball pushed back toward blue's end
        if by < BALL_CLEARED_Y:
            return 'ball_cleared'

        # 7. pass_received — ball in attack half and blue car near it
        if by > 0 and _dist3(blue.location, ball.location) < PASS_RECEIVED_DIST:
            return 'pass_received'

        # 8. interception — orange car near ball
        if _dist3(orange.location, ball.location) < INTERCEPTION_DIST:
            return 'interception'

        # 9. timeout
        if elapsed_seconds >= self._timeout_seconds:
            return 'timeout'

        return None
