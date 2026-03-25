#!/usr/bin/env python3
"""
Joint Multi-Task Training Loop
================================
Trains all skills simultaneously using adversarial scenario configs.

Both cars in each episode share the SAME encoder and the SAME skill_heads dict.
Gradients from blue's loss and orange's loss are summed in a single GradientTape
so the shared encoder learns features useful for all skills at once.

Usage
-----
    # Train all discovered skills:
    python training/train_all.py

    # Fine-tune a single skill only:
    python training/train_all.py --skill shooting

    # Override number of episodes:
    python training/train_all.py --episodes 5000

Architecture notes
------------------
- Actor-Critic with Monte Carlo returns (A2C).  Upgrade path to PPO is
  scaffolded in the TODOs below.
- Policy: Gaussian distribution with fixed log_std = log(0.5).  The mean is
  the tanh output of the skill head.  Exploration comes from per-step additive
  Gaussian noise on the analog controls during trajectory collection.
- The GradientTape context wraps compute_ac_loss() which replays the episode
  as a batch — the encoder is called INSIDE the tape for training, separate
  from the inference calls during tick collection.

RLGym integration
-----------------
  The game tick loop is currently a stub (no game runner is wired up here).
  When RLGym compiles, replace game_runner.get_packet() with:
      obs, info = env.step(action)
      packet    = rlgym_obs_to_tokens(obs, player_idx)   # adapter in encoder.py
  Everything else stays unchanged.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from scenarios.scenario_config import ScenarioConfig
from scenarios.graders import ScenarioGrader, reward_for_event, step_reward
from encoder import SharedTransformerEncoder, state_to_tokens
from skills.skill_head import SkillHead
from skills.controller import KNNController

# ── config discovery (mirrors discover_skills() in scenario_builder.py) ───────

CONFIGS_DIR = Path(__file__).parent / 'scenarios' / 'configs'


def discover_all_configs() -> List[ScenarioConfig]:
    """Load every .yaml file under training/scenarios/configs/."""
    configs = []
    for yaml_path in sorted(CONFIGS_DIR.rglob('*.yaml')):
        try:
            configs.append(ScenarioConfig.from_yaml(yaml_path))
        except Exception as exc:
            print(f'[warn] Skipping {yaml_path.name}: {exc}')
    return configs


# ── discounted returns ────────────────────────────────────────────────────────

GAMMA = 0.99


def compute_returns(rewards: List[float]) -> np.ndarray:
    """Compute normalised discounted Monte Carlo returns."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + GAMMA * G
        returns[t] = G
    if len(returns) > 1:
        std = returns.std()
        if std > 1e-8:
            returns = (returns - returns.mean()) / std
    return returns


# ── Actor-Critic loss ──────────────────────────────────────────────────────────

# Fixed log standard deviation for the Gaussian policy (log(0.5) ≈ -0.693)
_LOG_STD = tf.constant(math.log(0.5), dtype=tf.float32)
_TWO_PI  = tf.constant(2.0 * math.pi, dtype=tf.float32)


def compute_ac_loss(
    encoder:    SharedTransformerEncoder,
    skill_head: SkillHead,
    trajectory: List[Tuple[np.ndarray, np.ndarray, float]],
) -> tf.Tensor:
    """
    Compute Actor-Critic loss from one episode's trajectory.

    trajectory: list of (tokens, action, reward)
      tokens: (1, N_TOKENS, 8)
      action: (8,)  — action actually taken (including exploration noise)
      reward: scalar

    Must be called INSIDE a tf.GradientTape so encoder gradients are captured.
    Returns tf.constant(0.0) for empty trajectories.
    """
    if not trajectory:
        return tf.constant(0.0)

    tokens_batch  = tf.constant(
        np.concatenate([t[0] for t in trajectory], axis=0),   # (T, N_TOKENS, 8)
        dtype=tf.float32,
    )
    actions_taken = tf.constant(
        np.stack([t[1] for t in trajectory], axis=0),          # (T, 8)
        dtype=tf.float32,
    )
    returns = tf.constant(
        compute_returns([t[2] for t in trajectory]),            # (T,)
        dtype=tf.float32,
    )

    # Forward pass through shared encoder + skill head (inside GradientTape)
    embeddings     = encoder(tokens_batch, training=True)       # (T, 64)
    policy, values = skill_head(embeddings, training=True)      # (T,8), (T,1)
    values         = tf.squeeze(values, axis=-1)                # (T,)

    advantages = returns - tf.stop_gradient(values)             # (T,)

    # Policy loss: Gaussian log-likelihood  N(policy, exp(LOG_STD))
    action_dim = tf.cast(tf.shape(policy)[1], tf.float32)
    log_probs  = (
        -0.5 * tf.reduce_sum(
            tf.square((actions_taken - policy) / tf.exp(_LOG_STD)), axis=-1
        )
        - action_dim * (_LOG_STD + 0.5 * tf.math.log(_TWO_PI))
    )
    policy_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))

    # Value loss: MSE
    value_loss = tf.reduce_mean(tf.square(returns - values))

    # Entropy bonus: encourages exploration (differential entropy of Gaussian)
    entropy_bonus = 0.01 * 0.5 * action_dim * (
        1.0 + tf.math.log(_TWO_PI * tf.exp(2.0 * _LOG_STD))
    )

    return policy_loss + 0.5 * value_loss - entropy_bonus


# ── main training function ────────────────────────────────────────────────────

def train(
    all_configs:  List[ScenarioConfig],
    skill_filter: Optional[str] = None,
    max_episodes: int           = 10000,
    save_every:   int           = 500,
    model_dir:    str           = 'models/',
) -> None:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Build shared encoder ───────────────────────────────────────────────
    encoder = SharedTransformerEncoder()

    # Collect all unique skill names across every config (both sides)
    all_skill_names: set = set()
    for cfg in all_configs:
        all_skill_names.add(cfg.initial_state.blue.skill)
        all_skill_names.add(cfg.initial_state.orange.skill)
    all_skill_names.discard('')

    # ── 2. Build one SkillHead per skill ──────────────────────────────────────
    skill_heads: Dict[str, SkillHead] = {
        name: SkillHead(name) for name in sorted(all_skill_names)
    }

    # Build all models with dummy inputs so weights are created
    _dummy_tokens = tf.zeros((1, 3, 8))
    encoder(_dummy_tokens)
    for head in skill_heads.values():
        head(tf.zeros((1, 64)))

    # Try loading existing checkpoints
    enc_ckpt = model_path / 'encoder.weights.h5'
    if enc_ckpt.exists():
        encoder.load_weights(str(enc_ckpt))
        print(f'Loaded encoder from {enc_ckpt}')
    for name, head in skill_heads.items():
        head_ckpt = model_path / f'skill_{name}.weights.h5'
        if head_ckpt.exists():
            head.load_weights(str(head_ckpt))
            print(f'Loaded skill head: {name}')

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    # Filter to a single skill if requested
    active_configs = all_configs
    if skill_filter:
        active_configs = [
            c for c in all_configs
            if (c.initial_state.blue.skill   == skill_filter or
                c.initial_state.orange.skill == skill_filter)
        ]
        if not active_configs:
            raise ValueError(f'No configs found for skill: {skill_filter!r}')
        print(f'Fine-tuning skill: {skill_filter!r}  ({len(active_configs)} configs)')

    print(f'Skills: {sorted(all_skill_names)}')
    print(f'Configs: {len(active_configs)}   Episodes: {max_episodes}')

    # ── 3. Episode loop ───────────────────────────────────────────────────────
    for episode in range(max_episodes):
        scenario    = random.choice(active_configs)
        blue_skill  = scenario.initial_state.blue.skill
        orange_skill = scenario.initial_state.orange.skill

        grader        = ScenarioGrader(scenario)
        episode_start = time.monotonic()

        blue_traj:   List[Tuple] = []
        orange_traj: List[Tuple] = []

        # ── per-tick logic ────────────────────────────────────────────────────
        # This loop is a stub; wire up a game runner (RLBot or RLGym) here.
        #
        # Structure for each tick:
        #
        #   blue_tokens   = state_to_tokens(packet, car_idx=0)     # (1, N_TOKENS, 8)
        #   orange_tokens = state_to_tokens(packet, car_idx=1)
        #
        #   blue_emb   = encoder(tf.constant(blue_tokens)).numpy()  # (1, 64)
        #   orange_emb = encoder(tf.constant(orange_tokens)).numpy()
        #
        #   blue_action,   _ = skill_heads[blue_skill].act(blue_emb)
        #   orange_action, _ = skill_heads[orange_skill].act(orange_emb)
        #
        #   # Exploration noise on analog dims (training only)
        #   blue_action[:5]   += np.random.normal(0, 0.1, 5)
        #   orange_action[:5] += np.random.normal(0, 0.1, 5)
        #   np.clip(blue_action[:5],   -1, 1, out=blue_action[:5])
        #   np.clip(orange_action[:5], -1, 1, out=orange_action[:5])
        #
        #   elapsed = time.monotonic() - episode_start
        #   event   = grader.check(packet, elapsed)
        #
        #   blue_r   = step_reward(packet, scenario.reward.blue,   0)
        #   orange_r = step_reward(packet, scenario.reward.orange, 1)
        #   if event:
        #       blue_r   += reward_for_event(scenario.reward.blue,   event)
        #       orange_r += reward_for_event(scenario.reward.orange, event)
        #
        #   blue_traj.append((blue_tokens, blue_action, blue_r))
        #   orange_traj.append((orange_tokens, orange_action, orange_r))
        #   if event:
        #       break
        #
        # TODO: replace stub with:
        #   game_runner.reset(scenario)
        #   while not done:
        #       packet = game_runner.get_packet()
        #       ... (loop body above) ...
        #
        # TODO: RLGym integration:
        #   obs, _ = env.reset(); done = False
        #   while not done:
        #       obs, reward, done, _ = env.step(action)
        #   (add rlgym_obs_to_tokens() adapter in encoder.py)
        #
        # TODO: PPO upgrade — replace A2C trajectory collection with a
        #   rollout buffer and clipped surrogate objective.
        #
        # TODO: Expert data mixing (p=0.1) — with probability 0.1 replace
        #   one side's trajectory with a sampled expert trajectory from
        #   training/expert_data/<skill>/*.npz using BC loss instead of PG.

        # ── 4. Joint gradient update ──────────────────────────────────────────
        with tf.GradientTape() as tape:
            blue_loss   = compute_ac_loss(encoder, skill_heads[blue_skill],   blue_traj)
            orange_loss = compute_ac_loss(encoder, skill_heads[orange_skill], orange_traj)
            total_loss  = blue_loss + orange_loss

        # Collect variables; avoid duplicate entries when both sides share a skill
        vars_ep = (
            encoder.trainable_variables
            + skill_heads[blue_skill].trainable_variables
            + (skill_heads[orange_skill].trainable_variables
               if orange_skill != blue_skill else [])
        )
        grads = tape.gradient(total_loss, vars_ep)
        optimizer.apply_gradients(zip(grads, vars_ep))

        # ── 5. Logging ────────────────────────────────────────────────────────
        if episode % 100 == 0:
            print(
                f'[ep {episode:05d}] loss={float(total_loss):.4f}  '
                f'blue={blue_skill}  orange={orange_skill}'
            )

        # ── 6. Checkpoint ─────────────────────────────────────────────────────
        if episode > 0 and episode % save_every == 0:
            _save_all(encoder, skill_heads, model_path)
            print(f'[ep {episode:05d}] Checkpointed.')

    # ── 7. Final save ─────────────────────────────────────────────────────────
    _save_all(encoder, skill_heads, model_path)

    # ── 8. Build KNN index ────────────────────────────────────────────────────
    print('Building KNN index...')
    controller = KNNController(encoder, k=3)
    controller.build_index(all_configs, n_samples=20)
    knn_path = str(model_path / 'knn_index.npz')
    controller.save_index(knn_path)
    print(f'KNN index saved → {knn_path}')


def _save_all(
    encoder:     SharedTransformerEncoder,
    skill_heads: Dict[str, SkillHead],
    model_path:  Path,
) -> None:
    encoder.save_weights(str(model_path / 'encoder.weights.h5'))
    for name, head in skill_heads.items():
        head.save_weights(str(model_path / f'skill_{name}.weights.h5'))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Train all RL skills jointly.')
    parser.add_argument('--skill',    default=None, help='Fine-tune a single skill')
    parser.add_argument('--episodes', default=10000, type=int)
    parser.add_argument('--save-every', default=500, type=int)
    parser.add_argument('--model-dir', default='models/')
    args = parser.parse_args()

    configs = discover_all_configs()
    if not configs:
        print(f'No scenario configs found under {CONFIGS_DIR}')
        return

    train(
        all_configs  = configs,
        skill_filter = args.skill,
        max_episodes = args.episodes,
        save_every   = args.save_every,
        model_dir    = args.model_dir,
    )


if __name__ == '__main__':
    main()
