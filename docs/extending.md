# Extending the Training Framework

## Quick Start

Run an experiment with existing components:
```bash
python training/train.py --config configs/ppo_sparse.yaml
```

Run tests:
```bash
python -m pytest training/tests/test_ppo.py -v
```

## Architecture Overview

The framework is built on abstract base classes (ABCs) defined in `training/abstractions.py`. Every major component is swappable via YAML config. The training loop is algorithm-agnostic: it only calls `select_action`, `store_transition`, `should_update`, and `update`.

```
YAML Config
    |
    v
Algorithm  -----> Encoder (SharedTransformerEncoder)
    |                |
    v                v
PolicyHead <---- (batch, 64) embedding
    |
    v
ActionResult(action, aux)
    |
    v
Environment  --->  RewardFunction
    |
    v
store_transition -> update -> log metrics
```

## Creating New Components

Every experiment component is an abstract base class (ABC) in `training/abstractions.py`.
To create a new component:

1. Copy the ABC from `training/abstractions.py`
2. Implement all abstract methods
3. Save your implementation file
4. Point your YAML config at it

### Sample LLM Prompt for Generating a New Algorithm

Copy the contents of `training/abstractions.py` and add:

---

I need you to implement a new RL algorithm for this training framework.

The framework uses abstract base classes. My new algorithm should:
- [describe what you want, e.g. "implement SAC with automatic entropy tuning"]

Please implement a class that extends `Algorithm` from the ABCs above.
It must implement all abstract methods: `default_params`, `default_search_space`,
`select_action`, `store_transition`, `should_update`, `update`,
`save_checkpoint`, `load_checkpoint`, `get_weights`, `clone_from`.

Use `SharedTransformerEncoder` from `src/encoder.py` (input: batch, T, N, F -> output: batch, 64)
and a policy head. The action space is (8,): 5 analog [-1,1] + 3 binary {0,1}.
Observation is (800,) = 8 frames x 10 tokens x 10 features.

The encoder requires entity_type_ids when called:
```python
from encoder import SharedTransformerEncoder, ENTITY_TYPE_IDS_1V1, D_MODEL, N_TOKENS, TOKEN_FEATURES

encoder = SharedTransformerEncoder(d_model=D_MODEL)
entity_ids = torch.tensor(ENTITY_TYPE_IDS_1V1, dtype=torch.long)

# To encode a flat observation:
tokens = obs.view(batch, t_window, N_TOKENS, TOKEN_FEATURES)
embedding = encoder(tokens, entity_ids)  # (batch, 64)
```

For reference, see `training/ppo.py` which implements PPOAlgorithm using this pattern.

Save as `training/my_algorithm.py` and create a YAML config:
```yaml
algorithm:
  class: training.my_algorithm.MyAlgorithm
  params:
    lr: 3e-4
    gamma: 0.99
```

---

### Sample LLM Prompt for Generating a New Reward Function

Copy the `RewardFunction` ABC from `training/abstractions.py` and add:

---

I need a dense reward function for this training framework. It should:
- [describe what you want, e.g. "reward the agent for moving toward the ball and facing it"]

Please implement a class that extends `RewardFunction`.
It must implement: `compute_reward(obs, action, next_obs, done, info) -> float` and `on_reset()`.

The observation is a flat (800,) vector = 8 frames x 10 tokens x 10 features.
The most recent frame is the last 100 values. Token layout per frame (10 tokens x 10 features):
- Token 0: Ball [x, y, z, vx, vy, vz, av_x, av_y, av_z, 0] (normalized to [-1,1])
- Token 1: Own car [x, y, z, vx, vy, vz, yaw, pitch, roll, boost]
- Token 2: Opponent [x, y, z, vx, vy, vz, yaw, pitch, roll, 0]
- Tokens 3-8: Boost pads [x, y, z, active, 0, 0, 0, 0, 0, 0]
- Token 9: Game state [score_diff, time_rem, overtime, 0, ...]

Normalization constants: FIELD_X=4096, FIELD_Y=5120, CEILING_Z=2044, MAX_VEL=2300.

Also implement `get_metrics()` returning `{'reward_components': N}` where N is the
number of active reward terms (this tracks Axis 4 cost).

Save as `training/my_reward.py`.

---

### Sample LLM Prompt for Generating a New Opponent Pool

Copy the `OpponentPool` ABC from `training/self_play.py` and add:

---

I need a new self-play opponent pool strategy. It should:
- [describe what you want, e.g. "use Elo ratings to match opponents by skill level"]

Please implement a class that extends `OpponentPool` from `training/self_play.py`.
It must implement: `save_snapshot`, `sample_opponent`, `latest`, `num_snapshots`, `should_swap`.

For reference, see `PeriodicOpponentPool` in the same file (saves every N steps, samples uniformly)
and `FrozenOpponentPool` in `training/frozen_self_play.py` (swaps based on win rate).

Save as `training/my_opponent_pool.py`.

---

## ABC Reference Table

| ABC | Research Axis | What It Controls | Default Implementation |
|-----|--------------|-----------------|----------------------|
| Algorithm | Axis 1 (Simulation) | How the agent learns | PPOAlgorithm (`training/ppo.py`) |
| MetricLogger | -- | Where metrics are logged | WandbLogger (`training/logger.py`) |
| RewardFunction | Axis 4 (Reward) | What signal drives learning | SparseRewardFunction |
| EnvironmentProvider | Axis 1 (Simulation) | How envs are created | DefaultEnvironmentProvider |
| ReplayProvider | Axis 2 (Data) | Expert demonstration data | NullReplayProvider |
| FeedbackProvider | Axis 3 (Feedback) | Human reward signal | NullFeedbackProvider |
| EvaluationHook | -- | When to stop training | PsyonixEvaluationHook |
| OpponentPool | -- | Self-play opponent sampling | PeriodicOpponentPool (`training/self_play.py`) |

## YAML Config Reference

### Top-Level Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `algorithm.class` | str | `training.ppo.PPOAlgorithm` | Fully qualified class name for the algorithm |
| `algorithm.params` | dict | See `Algorithm.default_params()` | Algorithm-specific hyperparameters |
| `num_envs` | int | 1 | Number of parallel environment workers |
| `t_window` | int | 8 | Temporal window size (frames of history) |
| `total_steps` | int | 50000000 | Total environment steps to train |
| `eval_interval` | int | 200000 | Steps between evaluations |
| `seed` | int | 0 | Random seed |

### Algorithm Params (PPO)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lr` | float | 3e-4 | Learning rate for Adam optimizer |
| `gamma` | float | 0.99 | Discount factor |
| `gae_lambda` | float | 0.95 | GAE lambda for advantage estimation |
| `clip_epsilon` | float | 0.2 | PPO clip range |
| `vf_coef` | float | 0.5 | Value function loss coefficient |
| `ent_coef` | float | 0.01 | Entropy bonus coefficient |
| `max_grad_norm` | float | 0.5 | Max gradient norm for clipping |
| `rollout_steps` | int | 2048 | Steps per rollout before update |
| `ppo_epochs` | int | 4 | Gradient epochs per rollout |
| `minibatch_size` | int | 64 | Minibatch size for gradient updates |

### Experiment Config Keys (for `run_experiment.py`)

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Experiment name (used in W&B grouping) |
| `description` | str | Human-readable description |
| `intervention` | str | What is being tested |
| `base_config` | dict | Training parameters |
| `budget` | dict | Resource budgets (sim_steps, num_replays, etc.) |
| `seeds` | list[int] | Seeds to run (minimum 5 for publication) |
| `sweep` | list[dict] | Optional parameter sweep points |
| `wandb_project` | str | W&B project name |
| `wandb_tags` | list[str] | W&B run tags |

### Example YAML Config

```yaml
# PPO with sparse reward — baseline experiment
name: ppo_sparse_baseline
description: PPO with sparse reward, no interventions
intervention: none

algorithm:
  class: training.ppo.PPOAlgorithm
  params:
    lr: 3e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    vf_coef: 0.5
    ent_coef: 0.01
    rollout_steps: 2048
    ppo_epochs: 4
    minibatch_size: 64

num_envs: 8
t_window: 8
total_steps: 50000000
eval_interval: 200000

budget:
  sim_steps: 50000000
  num_replays: 0
  num_labels: 0
  reward_components: 0
  pretrain_gpu_hours: 0.0

seeds: [0, 1, 2, 3, 4]

wandb_project: rlbot-baseline
wandb_tags:
  - ppo
  - sparse
  - baseline
```

## Algorithm Interface Contract

Every algorithm must implement these methods (see `training/abstractions.py`):

```python
class Algorithm(ABC):
    @classmethod
    def default_params(cls) -> dict:
        """Default hyperparameters. YAML overrides these."""

    @classmethod
    def default_search_space(cls) -> dict:
        """Optuna search ranges for tuning."""

    def select_action(self, obs: np.ndarray) -> ActionResult:
        """Pick actions for a batch of observations.
        Returns ActionResult(action=(batch,8), aux=dict)."""

    def store_transition(self, obs, action_result, reward, next_obs, done, info):
        """Feed one transition. Algorithm manages its own buffer."""

    def should_update(self) -> bool:
        """Has enough data accumulated for a gradient update?"""

    def update(self) -> dict:
        """Run gradient update(s), return metrics dict for logging."""

    def save_checkpoint(self, path: Path) -> None:
        """Save full state to directory."""

    def load_checkpoint(self, path: Path) -> None:
        """Load full state from directory."""

    def get_weights(self) -> dict:
        """Return state_dicts for opponent loading."""

    def clone_from(self, other: 'Algorithm', noise_scale: float = 0.0):
        """Copy weights from another instance, optionally with noise."""
```

## Running Tests

```bash
# Run all PPO tests
python -m pytest training/tests/test_ppo.py -v

# Run specific test class
python -m pytest training/tests/test_ppo.py::TestPPOAlgorithm -v

# Run all tests
python -m pytest training/tests/ -v
```
