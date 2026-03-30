# CLAUDE.md

## What This Is

**This is not a Rocket League project.** This is a research framework for understanding how to reduce the resource cost of training RL agents in real-world domains — robotics, autonomous systems, any setting where simulation is expensive, data is scarce, and reward signals are sparse.

Rocket League is the testbed, not the subject. It was chosen precisely because its resources (simulation, replays, reward signals) are cheap and abundant — which means we can *artificially constrain* them to simulate the economics of expensive real-world domains. A robotics lab can't run millions of simulation steps cheaply or download thousands of expert demonstrations. In Rocket League, we can — but we deliberately treat these resources as expensive and meter them, so we can study what happens when you have limited budgets on each axis and must choose where to invest.

**The deliverable is a set of experimentally determined marginal return functions** across the five resource axes. Concretely: if you add 10 replays, how many simulation steps does that save? If you add 100, does it save proportionally more or have diminishing returns? These cross-axis substitution curves — "X units of Axis 2 investment displaces Y units of Axis 1 cost" — are the primary research output. They won't transfer perfectly to every domain, but they give practitioners in robotics, autonomous driving, industrial control, or any sparse-reward setting a quantitative starting point for resource allocation decisions. "Given X budget on Y axes, here's where to invest first and where you'll hit diminishing returns."

**Core research question:** How do you train effective RL models when data and compute are scarce, and which interventions reduce resource requirements most effectively?

Sparse reward is a feature, not a bug. Goals are rare in Rocket League, especially early in training. This mirrors real-world RL where dense reward signals don't exist and can't be easily engineered.

---

## Codebase Map

```
src/                          # Runtime bot (loads trained models, runs at ~120hz)
  bot.py                      # RLBot agent — loads encoder + policy, runs inference
  encoder.py                  # SharedTransformerEncoder (spatiotemporal, 64-dim output)
  policy_head.py              # PolicyHead (8-float action: 5 analog + 3 binary)
  skills/skill_head.py        # Legacy skill-specific head (not used in baseline)
  util/                       # Game-state utilities (vectors, orientation, boost tracking)

training/                     # All training, evaluation, and data collection
  train.py                    # Main entry point — d3rlpy AWAC/SAC/TD3/CQL/IQL training
  gym_env.py                  # Gymnasium wrapper over rlgym-sim (obs, action, reward, reset)
  rlgym_env.py                # rlgym-sim components (obs builder, reward fn, terminal cond)
  baseline_encoder_factory.py # d3rlpy EncoderFactory bridge for our transformer
  self_play.py                # OpponentPool — rotating pool of saved model snapshots
  evaluate.py                 # Psyonix bot evaluation protocol (Beginner/Rookie/Pro/Allstar)
  logger.py                   # W&B experiment logger with stdout fallback
  tune.py                     # Optuna hyperparameter search
  replay_sampler.py           # Replay buffer from ballchasing.com API
  collect_replays.py          # Download expert replays
  replay_dataset.py           # Parse replay data to (obs, action, reward) tuples
  scenario_builder.py         # YAML scenario → training setup
  scenario_visualizer.py      # Matplotlib scenario visualization
  profile_latency.py          # Inference speed profiling
  human_play.py               # Keyboard-controlled bot for demos
  tests/                      # test_baseline.py, test_baseline_integrity.py

training/scenarios/configs/   # Scenario YAML files (aerial, defending, passing, shooting)
models/                       # Training outputs (empty until first run)
eval_configs/                 # Match .toml configs for Psyonix evaluation tiers
```

### Key Commands

```bash
# Baseline training (AWAC, sparse reward, self-play)
python training/train.py

# Specific seed
python training/train.py --seed 3

# Swap algorithm
python training/train.py --algo SAC

# Short test run
python training/train.py --total-steps 10000 --eval-interval 5000

# Evaluate a saved model
python training/evaluate.py --model-dir models/baseline/seed_0

# Hyperparameter search
python training/tune.py

# Profile inference latency
python training/profile_latency.py

# Run tests
python -m pytest training/tests/
```

---

## Architecture

### Observation Space

Flat vector: **(800,)** = 8 frames x 10 tokens x 10 features, all normalized to [-1, 1].

**Token layout (1v1, 10 tokens) — stored format:**

| Slot | Entity | Features (TOKEN_FEATURES=10) |
|------|--------|----------|
| 0 | Ball | x, y, z, vx, vy, vz, av_x, av_y, av_z, 0 |
| 1 | Own car | x, y, z, vx, vy, vz, yaw, pitch, roll, boost |
| 2 | Opponent car | x, y, z, vx, vy, vz, yaw, pitch, roll, 0 |
| 3-8 | Big boost pads (6) | x, y, z, active, 0, 0, 0, 0, 0, 0 |
| 9 | Game state | score_diff, time_rem, overtime, 0, ... |

**GPU-side rotation expansion:** Inside `encoder.forward()`, car token Euler angles
(features 6-8) are converted to forward+up unit vectors (6 components) under
`torch.no_grad()`. This expands each token from 10 to 13 features
(ENCODER_INPUT_DIM=13) before the linear projection. The expansion avoids
wraparound discontinuities and gimbal lock inherent in Euler angles. Non-car
tokens receive zero-padding in the extra 3 slots.

| Slot | Internal features after expansion (ENCODER_INPUT_DIM=13) |
|------|----------|
| 1, 2 (cars) | x, y, z, vx, vy, vz, **fwd_x, fwd_y, fwd_z, up_x, up_y, up_z**, boost |

Entity type IDs: ball=0, own_car=1, opp_car=2, boost_pad=3, game_state=4.

### Encoder

`SharedTransformerEncoder` in `src/encoder.py`:
- Input: (batch, T, N, F) where T=8 frames, N=10 tokens, F=10 features
- GPU-side Euler → forward+up expansion: F=10 → 13 (no backprop through trig)
- Linear projection 13 → D_MODEL (64)
- Entity type embedding (5 types) + time embedding (T positions)
- 2 transformer layers, 4 heads, FFN dim 128, pre-norm
- Output: **(batch, 64)** — mean pool over most-recent timestep tokens
- Supports entity permutation augmentation during training

### Policy Head

`PolicyHead` in `src/policy_head.py`:
- Input: (batch, 64) embedding
- Hidden: 64 → 64 ReLU
- Analog head: → 5 floats (throttle, steer, pitch, yaw, roll) via tanh
- Binary head: → 3 floats (jump, boost, handbrake) via sigmoid, threshold 0.5
- Value head: → 1 float (for critic in AWAC)

### Inference Constraint

**~120hz hard limit.** All architecture choices must be profiled for latency. Tree search methods are off the table. Use `training/profile_latency.py` to verify.

---

## Training Infrastructure

### Algorithms (via d3rlpy)

| Algorithm | Type | Config Class | Role |
|-----------|------|-------------|------|
| **AWAC** | Offline/online hybrid | `d3rlpy.algos.AWACConfig` | Current baseline |
| SAC | Online off-policy | `d3rlpy.algos.SACConfig` | Simulation-dominant anchor |
| IQL | Offline | `d3rlpy.algos.IQLConfig` | Data-dominant anchor |
| TD3 | Online off-policy | `d3rlpy.algos.TD3Config` | Available |
| CQL | Offline | `d3rlpy.algos.CQLConfig` | Available |
| TD3PlusBC | Offline+online | `d3rlpy.algos.TD3PlusBCConfig` | Available |

All algorithms use the same `TransformerEncoderFactory` (in `training/baseline_encoder_factory.py`) which wraps our `SharedTransformerEncoder` for d3rlpy's interface.

### Default Hyperparameters (AWAC Baseline)

```
batch_size=256, gamma=0.99, actor_lr=3e-4, critic_lr=3e-4
tau=0.005, awac_lambda=1.0, n_critics=2, explore_noise=0.1
buffer_capacity=1M, random_steps=10K, total_steps=50M
eval_interval=200K steps, snapshot_interval=10K steps
```

### Self-Play

`OpponentPool` in `training/self_play.py` — saves model snapshots during training, randomly samples past versions as opponents. Max 20 snapshots, oldest evicted first.

### Evaluation Protocol

`training/evaluate.py` runs live matches against Psyonix bots:
- 50 episodes vs Beginner (sanity check)
- **100 episodes vs Rookie (convergence criterion: >=60% win rate)**
- 50 episodes vs Pro (ceiling check)
- 50 episodes vs Allstar (upper ceiling)

Convergence requires 2 consecutive evaluation intervals meeting the Rookie target.

### Reward (Baseline)

**Sparse only:** +1 for scoring, -1 for conceding, 0 otherwise. This is intentional — dense reward shaping is an intervention to be measured, not a default.

---

## Design Principle: Factory / Plugin Architecture

**The codebase must be structured for rapid experimentation.** Every major component should be swappable independently:

- **Encoder:** `TransformerEncoderFactory` is the pattern. New encoders (MLP, CNN, pretrained variants) implement the same d3rlpy `EncoderFactory` interface. Swap via config, not code surgery.
- **Policy head:** `PolicyHead` is a module. Variants (skill-decomposed heads, distributional outputs) drop in with the same input/output contract.
- **Reward function:** `ScenarioRewardFn` in `rlgym_env.py` is the base. Dense reward, learned reward models, and intrinsic reward all implement the same interface.
- **Algorithm:** Already swappable via `--algo` flag and `ALGO_MAP` in `train.py`.
- **Data pipeline:** Replay loading, filtering, and curriculum logic are isolated in `replay_sampler.py` / `replay_dataset.py`.

When adding new components, follow the existing factory pattern. If an experiment requires a new encoder architecture, create a new `EncoderFactory` subclass — don't modify the existing one. This keeps experiments isolated and reproducible.

---

## Research Framework

### The Five Resource Axes

Every experiment is measured against all five axes simultaneously. Resources that are cheap in Rocket League (simulation steps, replays) are deliberately budgeted and metered as if they were expensive — because in the target domains (robotics, industrial control), they are. An intervention's value is its total cost reduction across axes minus its own cost.

| Axis | What It Measures | Unit |
|------|-----------------|------|
| 1. Simulation | Environment steps for RL training | Env steps |
| 2. Real-world data | Replay collection and processing | Replays processed |
| 3. Human feedback | Labels, rankings, annotations | Labels or annotation-minutes |
| 4. Reward engineering | Expert time designing reward functions | Reward components or eng-hours |
| 5. Pre-training compute | Upfront compute for self-supervised objectives | GPU-hours |

These units are intentionally not converted to a common currency — that conversion is itself a research output.

### Experimental Method

1. **Define an intervention** (technique applied to one or more axes)
2. **Train to the fixed skill target** (>=60% vs Rookie)
3. **Record cost on every axis**
4. **Compare to the naive baseline**

**Isolation before combination.** Each intervention is tested alone first. Combinations are studied only after individual effects are characterized.

### The Naive Baseline

AWAC with no interventions: raw observations, sparse reward (goals only), no pre-training, no replay data, no human labels. This is the zero point. Every intervention is a delta from it.

**A naive baseline that struggles or fails is valid data.** It quantifies the sparse-reward penalty and establishes the cost floor. This is arguably the most informative outcome — it motivates every intervention that follows.

### Anchor Conditions

Four experiments that establish the frontier by isolating one axis each:

| Anchor | Dominant Resource | Algorithm |
|--------|------------------|-----------|
| Simulation-dominant | High step budget, sparse reward, no data/labels | SAC |
| Data-dominant | Offline RL on replay data, zero simulation | IQL |
| Human feedback-dominant | Learned reward from rank labels, minimal sim | AWAC + learned reward |
| Reward engineering-dominant | Dense hand-crafted reward, minimal everything | AWAC + dense reward |

### Intervention Catalogue (Condensed)

**Axis 1 — Simulation:**
- Pure online SAC from scratch (lower bound reference)

**Axis 2 — Real-world data:**
- Offline RL on replay data (IQL)
- Curriculum on difficulty-sequenced replay data

**Axis 3 — Human feedback:**
- Reward model from rank labels (RLHF-style, using ballchasing.com rank metadata)
- Skill categorization labels (richer per-label, fewer total)
- Expert vs. novice labeler comparison
- Unsupervised labeling assistance (pre-cluster then human-correct)

**Axis 4 — Reward engineering:**
- Dense shaped reward (upper bound reference)
- Next-goal predictor as learned reward

**Axis 5 — Pre-training compute:**
- Pre-train transformer encoder on replay data (masked entity prediction, next-state prediction)
- MLP baseline comparison (architecture ablation)
- Behavioral cloning warm start

### Three Substitution Archetypes (Hypotheses)

1. **Pre-training compute buys simulation efficiency** (Axis 5 → Axis 1 savings)
2. **Feedback quality buys feedback quantity** (better labels → fewer labels needed)
3. **Data structure buys reward signal** (structured data → less reward engineering)

These are hypotheses to be confirmed by experiment. The goal is to characterize each as a **marginal return curve**: how much of Axis A does one unit of Axis B buy, and where do diminishing returns set in? These curves are the primary transferable output.

---

## Logging Requirements

Every training run must log from day one. Minimum **5 seeds** per experimental condition. Report mean and 95% CI.

### Per evaluation interval
- Episode win rate vs each Psyonix tier
- Mean episode return and std dev

### Per training step (cumulative)
- Axis 1: total environment steps
- Axis 2: total replays loaded
- Axis 3: total human labels consumed (0 if N/A)
- Axis 4: number of active reward components (metadata)
- Axis 5: pre-training GPU-hours (0 if N/A)

### Per update
- Critic loss, actor loss, entropy
- Replay buffer size
- Wall clock time per update

### Run metadata
- Algorithm, hyperparameters, seed
- Intervention name and description
- Reference bot identifier
- Observation dimensionality

W&B project: `rlbot-baseline`. Logger: `training/logger.py`.

---

## Current Status

### Built and ready
- Full RLBot v5 bot (encoder + policy head + inference loop)
- d3rlpy training with AWAC/SAC/TD3/CQL/IQL
- Spatiotemporal transformer encoder (8-frame history, 10 tokens, 64-dim output)
- Self-play opponent pool
- Psyonix evaluation protocol (4 tiers)
- Hyperparameter tuning (Optuna)
- W&B experiment tracking
- Replay collection from ballchasing.com
- Scenario system (YAML configs for structured training)
- Test suites

### Not yet done
- **No trained models exist** — the naive baseline has not been run
- Five-axis cost logging not fully wired in W&B (axes 2-5 need structured tracking)
- Anchor conditions not yet implemented
- No intervention experiments yet

---

## Conventions

### Code style
- Python, PyTorch. Single-file implementations preferred (CleanRL/CORL philosophy).
- Factory/plugin pattern for swappable components. New experiments = new factory subclass, not modified existing code.
- Dataclasses for configuration. CLI args via argparse.

### Experimental discipline
- **One intervention at a time.** Isolate before combining.
- **5 seeds minimum.** No single-seed claims.
- **Fixed skill target.** Don't use training-entangled metrics (shaped reward score) as stopping criteria.
- **Log all five axes.** Even if an axis reads zero for a given experiment.
- **Track inference latency.** Every architecture variant must report it.

### What not to do
- Don't optimize for Rocket League performance at the expense of experimental clarity
- Don't combine interventions before isolating individual effects
- Don't use shaped reward as a convergence criterion
- Don't add features beyond what the current experiment requires
- Don't skip W&B logging even for quick test runs

### Dependencies
Key packages: `d3rlpy`, `rlgym-sim`, `rocketsim`, `torch`, `wandb`, `optuna`, `gymnasium`.
Full list in `requirements.txt`. Note: `rlgym-sim` is installed from GitHub, not PyPI.

---

## For Claude Sessions

- **Rocket League is incidental.** The domain is a cheap testbed. Every decision should be evaluated by whether it produces transferable insight about resource-constrained RL, not whether it makes a better Rocket League bot.
- **This is research.** Clean experimental isolation matters more than polish.
- **The five axes are the organizing principle.** Every change should be understood in terms of its axis costs and savings. The question is always: "Would this insight help a robotics researcher allocate their training budget?"
- **Factory pattern.** When building new components, make them swappable. Follow the `TransformerEncoderFactory` pattern. The whole point is rapid experimentation — a new architecture or reward function should be a config change, not a refactor.
- **A failing baseline is informative.** Don't paper over poor performance with ad-hoc fixes — document it as a data point. A naive baseline that can't learn under sparse reward is arguably the most valuable result: it quantifies the penalty that every intervention is trying to reduce.
- **Claude.ai handles exploration and design; Claude Code handles implementation.** Each session should produce either a sharpened hypothesis, a design document, or working code.
