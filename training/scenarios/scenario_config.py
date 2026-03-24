"""
Dataclasses representing a training scenario configuration.

Each scenario is stored as a YAML file and describes:
  - which skill is being trained (shooter, defender, passer, aerial)
  - the initial game state (ball + car positions, optionally randomised)
  - the reward structure (terminal events and per-step signals)
  - training hyper-parameters

All position/velocity/boost fields accept either a fixed value or a uniform
random range, expressed with the helpers below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml


# ── primitive: fixed value or uniform random range ────────────────────────────

@dataclass
class RangeOrFixed:
    """
    Represents a scalar that is either a fixed value or drawn uniformly from
    [min_val, max_val] each episode.  Set random=True for a full [-π, π] yaw.
    """
    fixed:   Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    random:  bool = False          # convenience flag for "any yaw"

    # ── queries ──

    def is_range(self) -> bool:
        return self.min_val is not None and self.max_val is not None

    def center(self) -> float:
        """Return the representative (mid-point) value, used for visualisation."""
        if self.fixed is not None:
            return self.fixed
        if self.is_range():
            return (self.min_val + self.max_val) / 2
        return 0.0

    def sample(self, rng=None) -> float:
        """Draw a concrete value for one episode."""
        import random as _rnd
        import math
        if self.random:
            return _rnd.uniform(-math.pi, math.pi)
        if self.fixed is not None:
            return self.fixed
        lo, hi = self.min_val, self.max_val
        return rng.uniform(lo, hi) if rng is not None else _rnd.uniform(lo, hi)

    # ── serialisation ──

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


# ── ball & car ────────────────────────────────────────────────────────────────

@dataclass
class BallConfig:
    location: Vec3Config
    velocity: Vec3Config

    def to_dict(self) -> dict:
        return {
            'location': self.location.to_dict(),
            'velocity': self.velocity.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> BallConfig:
        return cls(
            location=Vec3Config.from_dict(d.get('location', {})),
            velocity=Vec3Config.from_dict(d.get('velocity', {})),
        )


@dataclass
class CarConfig:
    location: Vec3Config
    yaw:   RangeOrFixed      # radians; random=True means any direction
    boost: RangeOrFixed      # 0–100

    def to_dict(self) -> dict:
        return {
            'location': self.location.to_dict(),
            'rotation': {'yaw': self.yaw.to_dict()},
            'boost':    self.boost.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CarConfig:
        rotation = d.get('rotation', {})
        return cls(
            location=Vec3Config.from_dict(d.get('location', {})),
            yaw=RangeOrFixed.from_dict(rotation.get('yaw', {'fixed': 0})),
            boost=RangeOrFixed.from_dict(d.get('boost', {'fixed': 33})),
        )


@dataclass
class InitialStateConfig:
    ball: BallConfig
    car:  CarConfig

    def to_dict(self) -> dict:
        return {'ball': self.ball.to_dict(), 'car': self.car.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> InitialStateConfig:
        return cls(
            ball=BallConfig.from_dict(d.get('ball', {})),
            car=CarConfig.from_dict(d.get('car', {})),
        )


# ── reward ────────────────────────────────────────────────────────────────────

@dataclass
class RewardEvent:
    """A single terminal or per-step reward signal."""
    type:    str
    value:   float = 0.0   # used for terminal events
    weight:  float = 1.0   # scaling for step rewards
    seconds: float = 0.0   # only for type == 'timeout'

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


# ── training hyper-params ─────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    max_episodes: int  = 10000
    save_every:   int  = 500
    model_path:   str  = ''

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
            model_path=d.get('model_path', ''),
        )


# ── top-level scenario ────────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    name:          str
    skill:         str
    description:   str
    initial_state: InitialStateConfig
    reward:        RewardConfig
    training:      TrainingConfig

    def to_dict(self) -> dict:
        return {
            'name':          self.name,
            'skill':         self.skill,
            'description':   self.description,
            'initial_state': self.initial_state.to_dict(),
            'reward':        self.reward.to_dict(),
            'training':      self.training.to_dict(),
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as fh:
            yaml.dump(
                self.to_dict(), fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> ScenarioConfig:
        with open(path) as fh:
            d = yaml.safe_load(fh)
        return cls(
            name=d['name'],
            skill=d['skill'],
            description=d.get('description', ''),
            initial_state=InitialStateConfig.from_dict(d.get('initial_state', {})),
            reward=RewardConfig.from_dict(d.get('reward', {})),
            training=TrainingConfig.from_dict(d.get('training', {})),
        )
