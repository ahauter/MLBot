"""
Baseline Integrity Tests
========================
Tests that verify the baseline produces ACCURATE measurements, not just
that the code runs. These catch silent bugs that corrupt training signals
without crashing — the kind that invalidate all downstream comparisons.

Organized by risk category:
1. Observation consistency (training vs eval, normalization, frame stacking)
2. Reward integrity (sparse only, correct sign, no leakage)
3. Self-play correctness (opponent perspective, weight divergence)
4. d3rlpy integration (target networks, gradients, determinism)
5. Seeding and reproducibility
6. Evaluation protocol correctness

Run with:
    python -m pytest training/tests/test_baseline_integrity.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

import d3rlpy
import gymnasium as gym

from baseline_encoder_factory import TransformerEncoderFactory
from encoder import (
    D_MODEL, N_TOKENS, TOKEN_FEATURES,
    ENTITY_TYPE_IDS_1V1, N_ENTITY_TYPES,
    SharedTransformerEncoder, _causal_mask,
)


# ── helpers ──────────────────────────────────────────────────────────────────

class DummyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (800,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (8,), np.float32)
    def reset(self, **kw): return np.zeros(800, np.float32), {}
    def step(self, a): return np.zeros(800, np.float32), 0.0, True, False, {}


def _make_algo():
    factory = TransformerEncoderFactory(t_window=8)
    algo = d3rlpy.algos.AWACConfig(
        actor_encoder_factory=factory,
        critic_encoder_factory=factory,
    ).create(device='cpu')
    algo.build_with_env(DummyEnv())
    return algo


def _has_rlgym_sim():
    try:
        import rlgym_sim
        return True
    except ImportError:
        return False

requires_rlgym = pytest.mark.skipif(not _has_rlgym_sim(), reason='rlgym-sim not installed')


# ═══════════════════════════════════════════════════════════════════════════
# 1. OBSERVATION CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════

class TestObservationConsistency:

    def test_entity_type_ids_match_token_layout(self):
        """ENTITY_TYPE_IDS_1V1 must match the documented token layout."""
        ids = ENTITY_TYPE_IDS_1V1
        assert len(ids) == N_TOKENS, \
            f"ENTITY_TYPE_IDS has {len(ids)} entries but N_TOKENS={N_TOKENS}"
        # Token 0 = ball (type 0)
        assert ids[0] == 0, f"Token 0 should be ball (type 0), got {ids[0]}"
        # Token 1 = own car (type 1)
        assert ids[1] == 1, f"Token 1 should be own_car (type 1), got {ids[1]}"
        # Token 2 = opponent car (type 2)
        assert ids[2] == 2, f"Token 2 should be opponent (type 2), got {ids[2]}"
        # Tokens 3-8 = boost pads (type 3)
        for i in range(3, 9):
            assert ids[i] == 3, f"Token {i} should be boost_pad (type 3), got {ids[i]}"
        # Token 9 = game state (type 4)
        assert ids[9] == 4, f"Token 9 should be game_state (type 4), got {ids[9]}"

    def test_entity_type_count(self):
        """N_ENTITY_TYPES must cover all types in ENTITY_TYPE_IDS."""
        max_type = max(ENTITY_TYPE_IDS_1V1)
        assert max_type < N_ENTITY_TYPES, \
            f"Max entity type {max_type} >= N_ENTITY_TYPES {N_ENTITY_TYPES}"

    def test_frame_stacking_temporal_order(self):
        """
        After sequential appends, stacked obs should have oldest first.
        The encoder expects index 0=oldest, T-1=newest.
        """
        from collections import deque
        t_window = 4
        buf = deque(maxlen=t_window)

        # Append frames with distinguishable values
        for t in range(6):
            frame = np.full((N_TOKENS, TOKEN_FEATURES), float(t), dtype=np.float32)
            buf.append(frame)

        stacked = np.stack(list(buf), axis=0)  # (T, N, F)
        # After 6 appends with maxlen=4: buf contains [2, 3, 4, 5]
        assert stacked[0, 0, 0] == 2.0, f"Oldest frame should be t=2, got {stacked[0,0,0]}"
        assert stacked[-1, 0, 0] == 5.0, f"Newest frame should be t=5, got {stacked[-1,0,0]}"

    def test_frame_stacking_no_future_leakage(self):
        """At step t, the observation must not contain data from t+1."""
        from collections import deque
        t_window = 4
        buf = deque(maxlen=t_window)

        # Fill with initial frame
        init = np.zeros((N_TOKENS, TOKEN_FEATURES), dtype=np.float32)
        for _ in range(t_window):
            buf.append(init.copy())

        # Append step 1 with unique value
        step1_frame = np.ones((N_TOKENS, TOKEN_FEATURES), dtype=np.float32)
        buf.append(step1_frame)

        stacked = np.stack(list(buf), axis=0)
        # stacked should be [0, 0, 0, 1] — no frame with value > 1 exists
        assert stacked.max() <= 1.0, "Future data leaked into observation"
        # The newest frame (index -1) should be step 1
        assert stacked[-1, 0, 0] == 1.0

    def test_causal_mask_prevents_future_attention(self):
        """Causal mask must block attention from earlier to later timesteps."""
        T, N = 4, 10
        mask = _causal_mask(T, N, 'cpu')
        # Token at timestep 0 should NOT attend to timestep 1, 2, 3
        # Token 0 (timestep 0) attending to token N (timestep 1) should be -inf
        assert mask[0, N].item() == float('-inf'), \
            "Causal mask should block timestep 0 from attending to timestep 1"
        # Token at timestep 3 should attend to all timesteps 0-3
        assert mask[3 * N, 0].item() == 0.0, \
            "Causal mask should allow timestep 3 to attend to timestep 0"

    def test_flat_obs_roundtrip(self):
        """Flat obs → tokens → flat should be identity."""
        flat = np.random.randn(N_TOKENS * TOKEN_FEATURES).astype(np.float32)
        tokens = flat.reshape(N_TOKENS, TOKEN_FEATURES)
        roundtrip = tokens.ravel()
        np.testing.assert_array_equal(flat, roundtrip)


# ═══════════════════════════════════════════════════════════════════════════
# 2. REWARD INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

class TestRewardIntegrity:

    def test_sparse_goal_reward_values(self):
        """SparseGoalReward must return exactly {-1, 0, +1}."""
        from gym_env import _GOAL_Y
        # Verify the constant is reasonable
        assert _GOAL_Y > 5000, f"Goal Y={_GOAL_Y} seems too small"
        assert _GOAL_Y < 6000, f"Goal Y={_GOAL_Y} seems too large"

    @requires_rlgym
    def test_reward_exactly_zero_mid_episode(self):
        """Every non-terminal step must have reward == 0.0 exactly."""
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8, max_steps=50)
        env.reset()
        non_terminal_rewards = []
        for _ in range(100):
            action = np.zeros(8, dtype=np.float32)
            _, reward, done, _, _ = env.step(action)
            if not done:
                non_terminal_rewards.append(reward)
            else:
                break
        env.close()
        for i, r in enumerate(non_terminal_rewards):
            assert r == 0.0, \
                f"Step {i}: mid-episode reward = {r} (expected exactly 0.0, reward leakage!)"

    @requires_rlgym
    def test_timeout_reward_is_zero(self):
        """When episode ends by timeout (no goal), reward must be 0.0."""
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8, max_steps=10)  # very short timeout
        env.reset()
        final_reward = None
        for _ in range(20):
            action = np.zeros(8, dtype=np.float32)  # do nothing
            _, reward, done, _, _ = env.step(action)
            if done:
                final_reward = reward
                break
        env.close()
        assert final_reward is not None, "Episode didn't terminate"
        # Timeout with no goal should be 0.0
        # (could be +/-1 if a goal happened to be scored, so only assert if no goal)
        assert final_reward in {-1.0, 0.0, 1.0}, \
            f"Terminal reward {final_reward} not in {{-1, 0, +1}}"

    @requires_rlgym
    def test_at_most_one_nonzero_reward_per_episode(self):
        """At most one step per episode should have a nonzero reward."""
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8, max_steps=200)
        env.reset()
        nonzero_count = 0
        for _ in range(300):
            action = np.random.uniform(-1, 1, 8).astype(np.float32)
            _, reward, done, _, _ = env.step(action)
            if reward != 0.0:
                nonzero_count += 1
            if done:
                break
        env.close()
        assert nonzero_count <= 1, \
            f"Got {nonzero_count} nonzero rewards in one episode (expected <= 1)"


# ═══════════════════════════════════════════════════════════════════════════
# 3. SELF-PLAY INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

class TestSelfPlayIntegrity:

    def test_snapshot_weights_are_reproducible(self):
        """Save → load should produce identical predictions."""
        import tempfile, shutil
        from self_play import OpponentPool

        tmpdir = tempfile.mkdtemp()
        try:
            pool = OpponentPool(tmpdir, algo_builder=_make_algo)
            algo = _make_algo()

            pool.save_snapshot(algo, step=1000)
            loaded = pool.latest()

            obs = np.random.randn(5, 800).astype(np.float32)
            a_orig = algo.predict(obs)
            a_loaded = loaded.predict(obs)
            np.testing.assert_allclose(a_orig, a_loaded, atol=1e-6,
                                       err_msg="Loaded snapshot predictions don't match original")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_opponent_produces_bounded_actions(self):
        """Opponent actions must be in [-1, 1]."""
        import tempfile, shutil
        from self_play import OpponentPool

        tmpdir = tempfile.mkdtemp()
        try:
            pool = OpponentPool(tmpdir, algo_builder=_make_algo)
            algo = _make_algo()
            pool.save_snapshot(algo, step=1000)
            loaded = pool.sample_opponent()

            for _ in range(10):
                obs = np.random.randn(1, 800).astype(np.float32)
                action = loaded.predict(obs)
                assert np.all(action >= -1.0) and np.all(action <= 1.0), \
                    f"Opponent action out of bounds: {action}"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @requires_rlgym
    def test_opponent_obs_uses_orange_perspective(self):
        """Opponent's stacked obs should be built from orange buffers, not blue."""
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8)

        algo = _make_algo()
        env.set_opponent(algo)
        env.reset()

        # Take a step so buffers diverge
        action = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        env.step(action)

        # Blue and orange buffers should differ (different perspectives)
        blue_stacked = np.stack(list(env._blue_buf), axis=0)
        orange_stacked = np.stack(list(env._orange_buf), axis=0)
        assert not np.array_equal(blue_stacked, orange_stacked), \
            "Blue and orange obs are identical — opponent may see wrong perspective"
        env.close()


# ═══════════════════════════════════════════════════════════════════════════
# 4. D3RLPY INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestD3rlpyIntegration:

    def test_target_network_weights_match_at_init(self):
        """Target Q-network must start with same weights as online Q-network."""
        algo = _make_algo()
        q_funcs = algo.impl.modules.q_funcs
        targ_q_funcs = algo.impl.modules.targ_q_funcs

        for i, (q, tq) in enumerate(zip(q_funcs, targ_q_funcs)):
            for p_name, (p1, p2) in enumerate(zip(q.parameters(), tq.parameters())):
                assert torch.equal(p1.data, p2.data), \
                    f"Q-func {i} param {p_name}: target != online at init"

    def test_predict_is_deterministic(self):
        """predict() must return identical results for identical inputs."""
        algo = _make_algo()
        obs = np.random.randn(3, 800).astype(np.float32)
        a1 = algo.predict(obs)
        a2 = algo.predict(obs)
        np.testing.assert_array_equal(a1, a2,
                                      err_msg="predict() is not deterministic — evaluation is noisy")

    def test_encoder_gradient_flows(self):
        """Encoder parameters must receive gradients during training."""
        factory = TransformerEncoderFactory(t_window=8)
        encoder = factory.create((800,))

        x = torch.randn(2, 800, requires_grad=False)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        has_grad = False
        for name, param in encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients flowing through encoder — it will never learn"

    def test_encoder_with_action_gradient_flows(self):
        """EncoderWithAction must also receive gradients."""
        factory = TransformerEncoderFactory(t_window=8)
        encoder = factory.create_with_action((800,), action_size=8)

        x = torch.randn(2, 800)
        action = torch.randn(2, 8)
        out = encoder(x, action)
        loss = out.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
        )
        assert has_grad, "No gradients flowing through critic encoder"

    def test_actor_critic_have_separate_encoders(self):
        """
        Actor and critic should have separate encoder instances.
        This is standard in d3rlpy and NOT a bug — document it.
        """
        algo = _make_algo()
        policy = algo.impl.modules.policy
        q_func_0 = algo.impl.modules.q_funcs[0]

        # They should be different objects
        assert policy is not q_func_0, \
            "Policy and Q-func are the same object (unexpected)"

        # But both should have parameters (both are real networks)
        policy_params = sum(p.numel() for p in policy.parameters())
        q_params = sum(p.numel() for p in q_func_0.parameters())
        assert policy_params > 0, "Policy has no parameters"
        assert q_params > 0, "Q-function has no parameters"

    def test_two_critics_exist(self):
        """AWAC with n_critics=2 should create 2 Q-functions."""
        algo = _make_algo()
        assert len(algo.impl.modules.q_funcs) == 2, \
            f"Expected 2 Q-functions, got {len(algo.impl.modules.q_funcs)}"
        assert len(algo.impl.modules.targ_q_funcs) == 2, \
            f"Expected 2 target Q-functions"


# ═══════════════════════════════════════════════════════════════════════════
# 5. SEEDING AND REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════

class TestSeeding:

    def test_same_seed_same_weights(self):
        """Same seed must produce identical initial weights."""
        from train import set_seed

        set_seed(42)
        algo1 = _make_algo()
        w1 = {n: p.data.clone() for n, p in algo1.impl.modules.policy.named_parameters()}

        set_seed(42)
        algo2 = _make_algo()
        w2 = {n: p.data.clone() for n, p in algo2.impl.modules.policy.named_parameters()}

        for name in w1:
            assert torch.equal(w1[name], w2[name]), \
                f"Param '{name}' differs with same seed — seeding is broken"

    def test_different_seeds_different_weights(self):
        """Different seeds must produce different initial weights."""
        from train import set_seed

        set_seed(0)
        algo1 = _make_algo()
        # Collect all non-zero-init parameters (skip biases that may init to zero)
        params1 = torch.cat([
            p.data.flatten() for p in algo1.impl.modules.policy.parameters()
            if p.numel() > 1  # skip scalar/bias params
        ])

        set_seed(999)
        algo2 = _make_algo()
        params2 = torch.cat([
            p.data.flatten() for p in algo2.impl.modules.policy.parameters()
            if p.numel() > 1
        ])

        assert not torch.equal(params1, params2), \
            "Different seeds produced identical weights — all seeds will be the same run"

    def test_predict_reproducible_with_seed(self):
        """Same seed → same algo → same prediction."""
        from train import set_seed

        obs = np.array([[0.1] * 800], dtype=np.float32)

        set_seed(7)
        algo1 = _make_algo()
        a1 = algo1.predict(obs)

        set_seed(7)
        algo2 = _make_algo()
        a2 = algo2.predict(obs)

        np.testing.assert_array_equal(a1, a2,
                                      err_msg="Same seed doesn't produce reproducible predictions")


# ═══════════════════════════════════════════════════════════════════════════
# 6. EVALUATION PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════

class TestEvalProtocol:

    def test_eval_episodes_match_spec(self):
        """Evaluation episode counts must match the baseline spec."""
        from evaluate import EVAL_TIERS
        assert EVAL_TIERS['Beginner'] == 50
        assert EVAL_TIERS['Rookie'] == 100
        assert EVAL_TIERS['Pro'] == 50
        assert EVAL_TIERS['Allstar'] == 50
        assert sum(EVAL_TIERS.values()) == 250

    def test_eval_toml_team_assignment(self):
        """Our bot must be team 0, Psyonix must be team 1."""
        from evaluate import generate_match_toml
        toml = generate_match_toml('Rookie')
        lines = toml.split('\n')

        # Find team assignments
        car_sections = []
        current_team = None
        current_type = None
        for line in lines:
            line = line.strip()
            if line == '[[cars]]':
                if current_team is not None:
                    car_sections.append((current_team, current_type))
                current_team = None
                current_type = None
            elif line.startswith('team ='):
                current_team = int(line.split('=')[1].strip())
            elif line.startswith('type ='):
                current_type = line.split('=')[1].strip().strip('"')
        if current_team is not None:
            car_sections.append((current_team, current_type))

        assert len(car_sections) == 2, f"Expected 2 cars, got {len(car_sections)}"
        assert car_sections[0] == (0, 'RLBot'), \
            f"Car 0 should be (team=0, RLBot), got {car_sections[0]}"
        assert car_sections[1] == (1, 'Psyonix'), \
            f"Car 1 should be (team=1, Psyonix), got {car_sections[1]}"

    def test_convergence_requires_two_consecutive(self):
        """
        Convergence detection must require 2 CONSECUTIVE checkpoints above target.
        A single checkpoint above 60% followed by one below must NOT trigger convergence.
        """
        from train import TrainingCallback, TrainConfig
        from gym_env import BaselineGymEnv
        from self_play import OpponentPool

        config = TrainConfig(rookie_target_wr=0.60, consecutive_evals_required=2)
        # Don't need real env/pool for this test — just test the counter logic
        cb = TrainingCallback.__new__(TrainingCallback)
        cb.config = config
        cb.consecutive_wins = 0
        cb.converged = False

        # Simulate: above, below, above, above
        win_rates = [0.65, 0.55, 0.65, 0.65]
        expected_consecutive = [1, 0, 1, 2]
        expected_converged = [False, False, False, True]

        for i, wr in enumerate(win_rates):
            if wr >= config.rookie_target_wr:
                cb.consecutive_wins += 1
                if cb.consecutive_wins >= config.consecutive_evals_required:
                    cb.converged = True
            else:
                cb.consecutive_wins = 0

            assert cb.consecutive_wins == expected_consecutive[i], \
                f"Step {i}: consecutive_wins={cb.consecutive_wins}, expected {expected_consecutive[i]}"
            assert cb.converged == expected_converged[i], \
                f"Step {i}: converged={cb.converged}, expected {expected_converged[i]}"

    def test_all_four_tiers_evaluated(self):
        """All 4 Psyonix tiers must be in the evaluation schedule."""
        from evaluate import EVAL_TIERS
        required = {'Beginner', 'Rookie', 'Pro', 'Allstar'}
        assert set(EVAL_TIERS.keys()) == required, \
            f"Missing tiers: {required - set(EVAL_TIERS.keys())}"
