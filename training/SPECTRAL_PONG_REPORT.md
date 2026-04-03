# Spectral Pong Experiment Report

## What We Built

A Pong environment where game objects (ball, paddles, walls) are represented as **spectral wavepackets** — Fourier-basis amplitude fields satisfying the PMF constraint ∫F²=1. Instead of feeding the RL controller raw positions and velocities, we derive all features from spectral interactions between wavepackets:

- **Normalized Inner Products (NIPs):** scalar proximity/alignment between two fields (~[0,1])
- **Cross Products:** signed 2D displacement vectors between fields
- **Deviation (∫F²−1):** PMF violations reveal environmental interactions (walls, paddles)
- **Reward Wavepacket:** 1D spectral field that learns where goals happen along x-axis

The RL controller sees a 10-dimensional state vector: 3 NIPs + 6 NIP-weighted cross products + 1 reward prediction. No Cartesian positions, no velocities, no collision flags.

## Key Result

**The spectral features contain sufficient information for perfect play.** A hand-coded linear policy `action = tanh(nip_x_padL_y × 50)` achieves **0 goals in 6000 frames** — perfect blocking. This proves the spectral representation encodes the right information for the task.

**However, RL cannot learn this mapping from sparse reward.** Across 8 algorithm variants, 6 hyperparameter sweeps, and 5 seeds per condition (12,000 frames each), no configuration achieved statistically significant improvement in goal intervals over time. The best was 1.01x (noise).

## Bugs Found and Fixed (Chronological)

| # | Bug | Impact | How Found |
|---|-----|--------|-----------|
| 1 | **Inverted cross product signs** | Paddles moved AWAY from ball | Hand-coded policy did worse than random (36 > 29 goals) |
| 2 | **TD step before wavepacket updates** | next_state ≈ current_state, TD error ≈ 0 | All LR configs produced identical trajectories per seed |
| 3 | **Reward never reaching critic** | Critic weights stayed at zero forever | Instrumented weights — all zeros after 5000 frames |
| 4 | **Weight decay erasing learning** | Actor weights shrunk 0.158→0.104 in 5000 frames | Weight magnitude tracked declining monotonically |
| 5 | **TD(0) too slow for value propagation** | Critic weights O(0.01) after 5000 frames | Value propagates 1 frame per goal; need 200+ goals for meaningful V(s) |

Each bug was only visible after fixing the previous one. This cascade is itself a finding.

## Why RL Can't Learn (Root Cause)

**Credit assignment gap exceeds trace memory.**

- Average rally length: ~100 frames (ball crosses court in ~100 frames at 2.0 units/s)
- Eligibility trace decay: γλ = 0.95 × 0.9 = 0.855 per frame
- Trace magnitude at 100 frames: 0.855¹⁰⁰ ≈ 1.6 × 10⁻⁷

The paddle decision that determines whether a goal happens (positioning at the moment of arrival) occurs ~100 frames before the reward signal. By then, the eligibility trace has decayed to effectively zero. The RL algorithm cannot assign credit to the correct action.

This is not a hyperparameter problem. Even with optimal learning rates, the information channel from reward to the responsible action has capacity ~10⁻⁷ bits per goal event.

## Attempted Solutions That Didn't Work

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| REINFORCE (sparse) | 0.96x | Only ~25 gradient updates in 6000 frames |
| TD(0) actor-critic | 0.88x | Critic weights stuck at zero (bug #3) |
| TD(λ) actor-critic | 0.95x | Traces decay before they bridge the rally gap |
| Feature normalization | 0.95x | Features were O(0.05); scaling to O(1) didn't help because the traces still decay |
| Higher LR (lr_a=0.1) | 0.88x | Faster learning but noisier; same trace problem |
| Reward wavepacket | 1.01x | Gives critic a reward-position signal, but V(s)≈0 in symmetric self-play |
| Various γ (0.90-0.99) | 0.86-0.95x | Higher γ helps trace memory but hurts value estimate stability |

## What Would Work (Hypotheses)

1. **Shorter court / faster ball:** Reduce rally length to ~20 frames. Trace at 20 frames: 0.855²⁰ ≈ 0.04 — still weak but 10⁵× better.

2. **Dense reward shaping:** Give small per-frame reward for paddle-ball y-alignment. This collapses the credit assignment gap to 1 frame. But it's "cheating" — it encodes domain knowledge that the spectral representation is supposed to discover.

3. **Asymmetric training:** Train one paddle against a fixed (random or tracking) opponent. Breaks the V(s)≈0 symmetry so the critic can learn non-trivial values.

4. **Evolutionary / population methods:** Black-box optimization (CMA-ES, OpenAI-ES) over the 10+1 actor weights. These methods don't rely on per-frame credit assignment — they evaluate entire trajectories. With only 11 parameters, population methods are feasible.

5. **Behavioral cloning from the hand-coded policy:** Generate expert demonstrations with the gain=50 policy, then fit the linear weights via supervised learning. This bypasses RL entirely for the initial policy.

## Insights for the Broader Research

### The spectral representation works — RL doesn't (yet)

The wavepacket system successfully encodes spatial relationships, proximity, direction, and environmental structure (walls) using only spectral features. The PMF constraint (∫F²=1) keeps all fields on comparable scales, making inner products meaningful cross-field attention weights. The NIP-weighted cross products correctly encode "which direction should this entity move to reach that entity, weighted by how close they are."

**But spectral features are smooth and slowly-varying.** The NIP between ball and paddle changes by ~0.01 per frame. The cross products shift by ~0.001 per frame. This smoothness is a feature for environment modeling but makes RL hard — the value landscape is extremely flat between interaction events, giving the critic almost nothing to learn from.

### The bug cascade is the real lesson

Five bugs, each hidden by the previous one. Inverted signs → wrong TD ordering → lost rewards → weight decay → insufficient propagation. In a real research setting, any one of these would be enough to produce a "negative result" paper claiming spectral features don't work. The finding that the features DO work (hand-coded policy proof) would never have been reached without systematic debugging.

**Implication for the broader project:** When an RL intervention appears to fail, the failure may be in the training loop, not the representation. The research framework needs a way to separate "representation quality" (can a hand-coded policy use these features?) from "learnability" (can RL discover the mapping?).

### Sparse reward + long episodes = fundamental hardness

This isn't specific to spectral features. Any representation that operates in a domain where the reward is sparse and the episode length is long will face the same credit assignment problem. The spectral representation doesn't make RL harder — it faithfully encodes the underlying difficulty.

**Connection to Axis 1 (Simulation) vs Axis 4 (Reward Engineering):** This experiment quantifies the cost of sparse reward. With dense reward (hand-coded gain), training is instant (0 samples needed — it's a known mapping). With sparse reward, RL cannot learn in 12,000 frames. The gap between these two conditions is the "reward engineering tax" — the price of not investing in Axis 4.

### The PMF constraint as a universal normalization

The ∫F²=1 constraint on all fields turned out to be crucial for making NIPs comparable across different entity types. Without it, the inner product between ball and paddle would be on a different scale than ball and env, and the soft-attention deviation attribution wouldn't work. This is analogous to batch normalization in neural networks — it keeps activations on comparable scales across layers.

## Files Modified

- `training/spectral_pong_viz.py` — Main viz with RL controller, deviation attribution, reward wavepacket
- `training/spectral_pong_tune.py` — Headless tuning script for sweeps

## Raw Data Summary

| Condition | Configs | Seeds | Frames | Best Improvement |
|-----------|---------|-------|--------|-----------------|
| REINFORCE baseline | 8 | 3 | 6,000 | 0.96x |
| Actor-critic TD(0) | 8 | 3 | 6,000 | 0.91x |
| TD(0) + feature norm | 8 | 3 | 6,000 | 1.77x* |
| TD(0) + norm (12k) | 1 | 5 | 12,000 | 0.77x |
| TD(λ) all fixes | 6 | 5 | 12,000 | 0.95x |
| TD(λ) + reward wp | 6 | 5 | 12,000 | 1.01x |
| **Hand-coded policy** | 1 | 1 | 6,000 | **∞ (0 goals)** |

*The 1.77x in the 6k norm run was noise — it collapsed to 0.77x at 12k frames.
