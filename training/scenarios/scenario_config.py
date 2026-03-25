"""
Dataclasses representing a training scenario configuration.

Each scenario is adversarial: it defines TWO cars (blue + orange), each
training a different skill simultaneously.  E.g. a "shooting" scenario always
has an opposing "saving" car, so neither side is ever practised in isolation.

Schema overview
───────────────
  name / description          — human-readable labels
  initial_state
    ball                      — BallConfig  (location + velocity, each RangeOrFixed)
    blue                      — CarConfig   (skill + location + yaw + boost)
    orange                    — CarConfig   (adversary skill + location + yaw + boost)
  reward
    blue                      — RewardConfig  (terminal events + step signals)
    orange                    — RewardConfig
  training                    — TrainingConfig  (episodes, save cadence, model dir)

All position/velocity/boost fields accept either { fixed: N } or { min: A, max: B }.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml


# ── primitive: fixed value or uniform random range ────────────────────────────

@dataclass
class RangeOrFixed:
    fixed:   Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    random:  bool = False          # convenience flag for "any yaw"

    def is_range(self) -> bool:
        return self.min_val is not None and self.max_val is not None

    def center(self) -> float:
        if self.fixed is not None:
            return self.fixed
        if self.is_range():
            return (self.min_val + self.max_val) / 2
        return 0.0

    def sample(self, rng=None) -> float:
        import random as _rnd, math
        if self.random:
            return _rnd.uniform(-math.pi, math.pi)
        if self.fixed is not None:
            return self.fixed
        lo, hi = self.min_val, self.max_val
        return rng.uniform(lo, hi) if rng is not None else _rnd.uniform(lo, hi)

    def to_dict(self) -> dict:
        if self.random:
            return {'random': True}
        if self.is_range():
            return {'min': self.min_val, 'max': self.max_val}
        return {'fixed': self.fixed}

    @classmethod
    def from_dict(cls, d) -> RangeOrFixed:
        if isinstance(d, (int, float)):
            return cls(fixed=float(d))
        return cls(
            fixed=d.get('fixed'),
            min_val=d.get('min'),
            max_val=d.get('max'),
            random=d.get('random', False),
        )


# ── 3-D vector of RangeOrFixed ────────────────────────────────────────────────

@dataclass
class Vec3Config:
    x: RangeOrFixed
    y: RangeOrFixed
    z: RangeOrFixed

    def to_dict(self) -> dict:
        return {'x': self.x.to_dict(), 'y': self.y.to_dict(), 'z': self.z.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> Vec3Config:
        return cls(
            x=RangeOrFixed.from_dict(d.get('x', {'fixed': 0})),
            y=RangeOrFixed.from_dict(d.get('y', {'fixed': 0})),
            z=RangeOrFixed.from_dict(d.get('z', {'fixed': 0})),
        )


# ── ball ──────────────────────────────────────────────────────────────────────

@dataclass
class BallConfig:
    location: Vec3Config
    velocity: Vec3Config

    def to_dict(self) -> dict:
        return {'location': self.location.to_dict(), 'velocity': self.velocity.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> BallConfig:
        return cls(
            location=Vec3Config.from_dict(d.get('location', {})),
            velocity=Vec3Config.from_dict(d.get('velocity', {})),
        )


# ── car (used for both blue and orange) ───────────────────────────────────────

@dataclass
class CarConfig:
    skill:    str           # which skill this car is training
    location: Vec3Config
    yaw:      RangeOrFixed  # radians; random=True → any direction
    boost:    RangeOrFixed  # 0–100

    def to_dict(self) -> dict:
        d: dict = {}
        if self.skill:
            d['skill'] = self.skill
        d['location'] = self.location.to_dict()
        d['rotation']  = {'yaw': self.yaw.to_dict()}
        d['boost']     = self.boost.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CarConfig:
        rotation = d.get('rotation', {})
        return cls(
            skill=d.get('skill', ''),
            location=Vec3Config.from_dict(d.get('location', {})),
            yaw=RangeOrFixed.from_dict(rotation.get('yaw', {'fixed': 0})),
            boost=RangeOrFixed.from_dict(d.get('boost', {'fixed': 33})),
        )


# ── initial state (ball + two cars) ───────────────────────────────────────────

@dataclass
class InitialStateConfig:
    ball:   BallConfig
    blue:   CarConfig    # primary car (attacker / main skill)
    orange: CarConfig    # adversary car

    def to_dict(self) -> dict:
        return {
            'ball':   self.ball.to_dict(),
            'blue':   self.blue.to_dict(),
            'orange': self.orange.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> InitialStateConfig:
        # backward compat: accept old 'car' key as blue car
        blue_dict = d.get('blue') or d.get('car', {})
        return cls(
            ball=BallConfig.from_dict(d.get('ball', {})),
            blue=CarConfig.from_dict(blue_dict),
            orange=CarConfig.from_dict(d.get('orange', {})),
        )


# ── reward ────────────────────────────────────────────────────────────────────

@dataclass
class RewardEvent:
    type:    str
    value:   float = 0.0
    weight:  float = 1.0
    seconds: float = 0.0   # only used for type == 'timeout'

    def to_dict(self) -> dict:
        d: dict = {'type': self.type}
        if self.type == 'timeout':
            d['seconds'] = self.seconds
            d['value']   = self.value
        elif self.value != 0.0:
            d['value'] = self.value
        if self.weight != 1.0:
            d['weight'] = self.weight
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RewardEvent:
        return cls(
            type=d['type'],
            value=d.get('value', 0.0),
            weight=d.get('weight', 1.0),
            seconds=d.get('seconds', 0.0),
        )


@dataclass
class RewardConfig:
    terminal: List[RewardEvent] = field(default_factory=list)
    step:     List[RewardEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'terminal': [e.to_dict() for e in self.terminal],
            'step':     [e.to_dict() for e in self.step],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RewardConfig:
        return cls(
            terminal=[RewardEvent.from_dict(e) for e in d.get('terminal', [])],
            step=[RewardEvent.from_dict(e)     for e in d.get('step', [])],
        )


@dataclass
class AdversarialRewardConfig:
    """Holds independent reward configs for each side of the adversarial scenario."""
    blue:   RewardConfig
    orange: RewardConfig

    def to_dict(self) -> dict:
        return {'blue': self.blue.to_dict(), 'orange': self.orange.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> AdversarialRewardConfig:
        # backward compat: old flat format has 'terminal' / 'step' at top level
        if 'terminal' in d or 'step' in d:
            return cls(blue=RewardConfig.from_dict(d), orange=RewardConfig())
        return cls(
            blue=RewardConfig.from_dict(d.get('blue', {})),
            orange=RewardConfig.from_dict(d.get('orange', {})),
        )


# ── training hyper-params ─────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    max_episodes: int = 10000
    save_every:   int = 500
    model_path:   str = 'models/'

    def to_dict(self) -> dict:
        return {
            'max_episodes': self.max_episodes,
            'save_every':   self.save_every,
            'model_path':   self.model_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        return cls(
            max_episodes=d.get('max_episodes', 10000),
            save_every=d.get('save_every', 500),
            model_path=d.get('model_path', 'models/'),
        )


# ── top-level scenario ────────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    name:          str
    description:   str
    initial_state: InitialStateConfig
    reward:        AdversarialRewardConfig
    training:      TrainingConfig

    @property
    def skill(self) -> str:
        """Primary (blue) skill name — used for display and folder organisation."""
        return self.initial_state.blue.skill

    def to_dict(self) -> dict:
        return {
            'name':          self.name,
            'description':   self.description,
            'initial_state': self.initial_state.to_dict(),
            'reward':        self.reward.to_dict(),
            'training':      self.training.to_dict(),
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as fh:
            yaml.dump(self.to_dict(), fh,
                      default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> ScenarioConfig:
        with open(path) as fh:
            d = yaml.safe_load(fh)

        initial_state = InitialStateConfig.from_dict(d.get('initial_state', {}))

        # backward compat: old top-level 'skill' key → assign to blue car
        if 'skill' in d and not initial_state.blue.skill:
            initial_state = InitialStateConfig(
                ball=initial_state.ball,
                blue=CarConfig(
                    skill=d['skill'],
                    location=initial_state.blue.location,
                    yaw=initial_state.blue.yaw,
                    boost=initial_state.blue.boost,
                ),
                orange=initial_state.orange,
            )

        return cls(
            name=d['name'],
            description=d.get('description', ''),
            initial_state=initial_state,
            reward=AdversarialRewardConfig.from_dict(d.get('reward', {})),
            training=TrainingConfig.from_dict(d.get('training', {})),
        )
