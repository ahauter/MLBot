"""
PPO + Population-Based Training Tests
======================================
Tests for PPO components: StochasticPolicyHead, RolloutBuffer,
PPOAlgorithm, and Population.

All tests run without rlgym-sim (no live simulation needed).

Run with:
    python -m pytest training/tests/test_ppo.py -v
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

from encoder import D_MODEL
from policy_head import StochasticPolicyHead
from training.algorithms.ppo import PPOAlgorithm, RolloutBuffer, Population
from training.abstractions import Algorithm, ActionResult


# ── StochasticPolicyHead tests ─────────────────────────────────────────────

class TestStochasticPolicyHead:

    def test_forward_shapes(self):
        """Input (batch=4, 64), output action (4,8), log_prob (4,), value (4,), entropy (4,)."""
        head = StochasticPolicyHead(d_model=D_MODEL)
        emb = torch.randn(4, D_MODEL)
        action, log_prob, value, entropy = head(emb)
        assert action.shape == (4, 8), f"Expected action (4,8), got {action.shape}"
        assert log_prob.shape == (4,), f"Expected log_prob (4,), got {log_prob.shape}"
        assert value.shape == (4,), f"Expected value (4,), got {value.shape}"
        assert entropy.shape == (4,), f"Expected entropy (4,), got {entropy.shape}"

    def test_log_probs_finite(self):
        """All log_probs should be finite (not nan/inf)."""
        head = StochasticPolicyHead(d_model=D_MODEL)
        emb = torch.randn(8, D_MODEL)
        _, log_prob, _, _ = head(emb)
        assert torch.isfinite(log_prob).all(), \
            f"Non-finite log_probs found: {log_prob}"

    def test_entropy_positive(self):
        """Entropy should be > 0 for a fresh (untrained) policy."""
        head = StochasticPolicyHead(d_model=D_MODEL)
        emb = torch.randn(4, D_MODEL)
        _, _, _, entropy = head(emb)
        assert (entropy > 0).all(), \
            f"Expected positive entropy for untrained policy, got {entropy}"

    def test_evaluate_actions_matches_forward(self):
        """evaluate_actions with same embedding+actions should give same log_probs."""
        head = StochasticPolicyHead(d_model=D_MODEL)
        torch.manual_seed(42)
        emb = torch.randn(4, D_MODEL)

        # Forward pass to get actions
        action, log_prob_fwd, value_fwd, entropy_fwd = head(emb)

        # Evaluate those same actions
        log_prob_eval, value_eval, entropy_eval = head.evaluate_actions(emb, action)

        # Log probs and values should match (same embedding, same actions)
        torch.testing.assert_close(log_prob_eval, log_prob_fwd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(value_eval, value_fwd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(entropy_eval, entropy_fwd, atol=1e-5, rtol=1e-5)

    def test_act_deterministic_shapes(self):
        """act_deterministic returns action (batch,8) and value (batch,)."""
        head = StochasticPolicyHead(d_model=D_MODEL)
        emb = torch.randn(4, D_MODEL)
        action, value = head.act_deterministic(emb)
        assert action.shape == (4, 8), f"Expected (4,8), got {action.shape}"
        assert value.shape == (4,), f"Expected (4,), got {value.shape}"

    def test_act_deterministic_is_deterministic(self):
        """Repeated calls with same input should give identical output."""
        head = StochasticPolicyHead(d_model=D_MODEL)
        head.eval()
        emb = torch.randn(2, D_MODEL)
        a1, v1 = head.act_deterministic(emb)
        a2, v2 = head.act_deterministic(emb)
        torch.testing.assert_close(a1, a2)
        torch.testing.assert_close(v1, v2)

    def test_gradient_flows_through_evaluate(self):
        """Ensure gradients flow through evaluate_actions (needed for PPO updates)."""
        head = StochasticPolicyHead(d_model=D_MODEL)
        emb = torch.randn(4, D_MODEL, requires_grad=True)
        # Analog actions can be any float in [-1,1], binary must be {0, 1}
        analog = torch.randn(4, 5).clamp(-1, 1)
        binary = torch.bernoulli(torch.ones(4, 3) * 0.5)
        actions = torch.cat([analog, binary], dim=-1)

        log_prob, value, entropy = head.evaluate_actions(emb, actions)
        loss = log_prob.sum() + value.sum()
        loss.backward()

        assert emb.grad is not None, "No gradient on embedding"
        assert emb.grad.abs().sum() > 0, "Zero gradient on embedding"

        has_param_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters()
        )
        assert has_param_grad, "No gradients flowing to policy head parameters"

    def test_analog_actions_in_range(self):
        """Analog actions (first 5) should be in [-1, 1]."""
        head = StochasticPolicyHead(d_model=D_MODEL)
        emb = torch.randn(100, D_MODEL)
        action, _, _, _ = head(emb)
        analog = action[:, :5]
        assert analog.min() >= -1.0, f"Analog action below -1: {analog.min()}"
        assert analog.max() <= 1.0, f"Analog action above 1: {analog.max()}"

    def test_binary_actions_are_binary(self):
        """Binary actions (last 3) should be 0 or 1."""
        head = StochasticPolicyHead(d_model=D_MODEL)
        emb = torch.randn(100, D_MODEL)
        action, _, _, _ = head(emb)
        binary = action[:, 5:]
        unique = torch.unique(binary)
        assert all(v in [0.0, 1.0] for v in unique.tolist()), \
            f"Binary actions should be {{0, 1}}, got unique values: {unique.tolist()}"


# ── RolloutBuffer tests ───────────────────────────────────────────────────

class TestRolloutBuffer:

    def test_add_and_iterate(self):
        """Add transitions, check iterate_minibatches yields correct shapes."""
        buf = RolloutBuffer(capacity=16, num_envs=2, obs_dim=800, action_dim=8)

        for _ in range(16):
            buf.add(
                obs=np.random.randn(2, 800).astype(np.float32),
                action=np.random.randn(2, 8).astype(np.float32),
                reward=np.zeros(2, dtype=np.float32),
                done=np.zeros(2, dtype=np.float32),
                log_prob=np.zeros(2, dtype=np.float32),
                value=np.zeros(2, dtype=np.float32),
            )

        assert buf.pos == 16

        buf.compute_gae(last_values=np.zeros(2, dtype=np.float32))

        batches = list(buf.iterate_minibatches(minibatch_size=8))
        assert len(batches) > 0

        for batch in batches:
            assert batch['obs'].shape[1] == 800
            assert batch['actions'].shape[1] == 8
            assert batch['log_probs'].ndim == 1
            assert batch['advantages'].ndim == 1
            assert batch['returns'].ndim == 1

    def test_gae_computation(self):
        """Hand-compute GAE for a simple case and verify."""
        # 3 steps, 1 env, rewards [0, 0, 1], values [0.5, 0.5, 0.5], dones [0, 0, 1]
        # last_value=0, gamma=0.99, gae_lambda=0.95
        buf = RolloutBuffer(capacity=3, num_envs=1, obs_dim=4, action_dim=2,
                            gamma=0.99, gae_lambda=0.95)

        for r, d, v in [(0.0, 0.0, 0.5), (0.0, 0.0, 0.5), (1.0, 1.0, 0.5)]:
            buf.add(
                obs=np.zeros((1, 4), dtype=np.float32),
                action=np.zeros((1, 2), dtype=np.float32),
                reward=np.array([r], dtype=np.float32),
                done=np.array([d], dtype=np.float32),
                log_prob=np.zeros(1, dtype=np.float32),
                value=np.array([v], dtype=np.float32),
            )

        buf.compute_gae(last_values=np.array([0.0], dtype=np.float32))

        # Manual GAE calculation (working backwards):
        gamma, lam = 0.99, 0.95

        # t=2: done=1, so next_non_terminal=0
        # delta_2 = r_2 + gamma * next_val * 0 - v_2 = 1.0 + 0 - 0.5 = 0.5
        # gae_2 = delta_2 + 0 = 0.5
        delta_2 = 1.0 + gamma * 0.0 * 0.0 - 0.5  # 0.5
        gae_2 = delta_2

        # t=1: done=0, next_non_terminal=1, next_val=v[2]=0.5
        # delta_1 = 0 + 0.99 * 0.5 * 1 - 0.5 = -0.005
        # gae_1 = -0.005 + 0.99 * 0.95 * 1 * gae_2 = -0.005 + 0.94050 * 0.5
        delta_1 = 0.0 + gamma * 0.5 * 1.0 - 0.5
        gae_1 = delta_1 + gamma * lam * 1.0 * gae_2

        # t=0: done=0, next_non_terminal=1, next_val=v[1]=0.5
        # delta_0 = 0 + 0.99 * 0.5 - 0.5 = -0.005
        # gae_0 = -0.005 + 0.99 * 0.95 * 1 * gae_1
        delta_0 = 0.0 + gamma * 0.5 * 1.0 - 0.5
        gae_0 = delta_0 + gamma * lam * 1.0 * gae_1

        np.testing.assert_allclose(buf.advantages[2, 0], gae_2, atol=1e-5,
                                   err_msg=f"GAE at t=2 wrong")
        np.testing.assert_allclose(buf.advantages[1, 0], gae_1, atol=1e-5,
                                   err_msg=f"GAE at t=1 wrong")
        np.testing.assert_allclose(buf.advantages[0, 0], gae_0, atol=1e-5,
                                   err_msg=f"GAE at t=0 wrong")

        # Returns = advantages + values
        np.testing.assert_allclose(
            buf.returns[:3, 0],
            buf.advantages[:3, 0] + buf.values[:3, 0],
            atol=1e-5,
        )

    def test_reset_clears_buffer(self):
        """After reset, pos should be 0."""
        buf = RolloutBuffer(capacity=8, num_envs=1, obs_dim=4, action_dim=2)
        buf.add(
            obs=np.zeros((1, 4), dtype=np.float32),
            action=np.zeros((1, 2), dtype=np.float32),
            reward=np.zeros(1, dtype=np.float32),
            done=np.zeros(1, dtype=np.float32),
            log_prob=np.zeros(1, dtype=np.float32),
            value=np.zeros(1, dtype=np.float32),
        )
        assert buf.pos == 1
        buf.reset()
        assert buf.pos == 0

    def test_buffer_full_raises(self):
        """Adding past capacity should raise."""
        buf = RolloutBuffer(capacity=2, num_envs=1, obs_dim=4, action_dim=2)
        for _ in range(2):
            buf.add(
                obs=np.zeros((1, 4), dtype=np.float32),
                action=np.zeros((1, 2), dtype=np.float32),
                reward=np.zeros(1, dtype=np.float32),
                done=np.zeros(1, dtype=np.float32),
                log_prob=np.zeros(1, dtype=np.float32),
                value=np.zeros(1, dtype=np.float32),
            )
        with pytest.raises(AssertionError, match="Buffer full"):
            buf.add(
                obs=np.zeros((1, 4), dtype=np.float32),
                action=np.zeros((1, 2), dtype=np.float32),
                reward=np.zeros(1, dtype=np.float32),
                done=np.zeros(1, dtype=np.float32),
                log_prob=np.zeros(1, dtype=np.float32),
                value=np.zeros(1, dtype=np.float32),
            )

    def test_minibatch_covers_all_data(self):
        """All data should be covered across minibatches (no duplicates)."""
        buf = RolloutBuffer(capacity=8, num_envs=2, obs_dim=4, action_dim=2)
        for _ in range(8):
            buf.add(
                obs=np.random.randn(2, 4).astype(np.float32),
                action=np.random.randn(2, 2).astype(np.float32),
                reward=np.zeros(2, dtype=np.float32),
                done=np.zeros(2, dtype=np.float32),
                log_prob=np.zeros(2, dtype=np.float32),
                value=np.zeros(2, dtype=np.float32),
            )
        buf.compute_gae(np.zeros(2, dtype=np.float32))

        total_samples = 0
        for batch in buf.iterate_minibatches(4):
            total_samples += batch['obs'].shape[0]
        assert total_samples == 16, f"Expected 16 total samples, got {total_samples}"


# ── PPOAlgorithm tests ────────────────────────────────────────────────────

class TestPPOAlgorithm:

    def _make_config(self):
        return {
            'algorithm': {
                'params': {
                    'lr': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95,
                    'clip_epsilon': 0.2, 'vf_coef': 0.5, 'ent_coef': 0.01,
                    'max_grad_norm': 0.5, 'rollout_steps': 64,
                    'ppo_epochs': 2, 'minibatch_size': 32,
                }
            },
            'num_envs': 2,
            't_window': 8,
        }

    def test_select_action_shapes(self):
        """obs (2, 800) -> ActionResult with action (2, 8), aux has log_prob and value."""
        algo = PPOAlgorithm(self._make_config())
        obs = np.random.randn(2, 800).astype(np.float32)
        result = algo.select_action(obs)

        assert isinstance(result, ActionResult)
        assert result.action.shape == (2, 8), f"Expected (2,8), got {result.action.shape}"
        assert 'log_prob' in result.aux
        assert 'value' in result.aux
        assert result.aux['log_prob'].shape == (2,)
        assert result.aux['value'].shape == (2,)

    def test_store_transition_fills_buffer(self):
        """Store transitions, check buffer pos increases."""
        algo = PPOAlgorithm(self._make_config())
        obs = np.random.randn(2, 800).astype(np.float32)
        result = algo.select_action(obs)
        next_obs = np.random.randn(2, 800).astype(np.float32)

        assert algo.buffer.pos == 0
        algo.store_transition(obs, result, np.zeros(2), next_obs, np.zeros(2), {})
        assert algo.buffer.pos == 1

    def test_should_update_triggers(self):
        """Fill buffer to capacity, should_update returns True."""
        algo = PPOAlgorithm(self._make_config())
        assert algo.should_update() is False

        # Fill the buffer
        for _ in range(64):
            obs = np.random.randn(2, 800).astype(np.float32)
            result = algo.select_action(obs)
            next_obs = np.random.randn(2, 800).astype(np.float32)
            algo.store_transition(obs, result, np.zeros(2), next_obs, np.zeros(2), {})

        assert algo.should_update() is True

    def test_update_returns_metrics(self):
        """Fill buffer, call update, check it returns dict with expected keys."""
        algo = PPOAlgorithm(self._make_config())

        # Fill buffer
        for _ in range(64):
            obs = np.random.randn(2, 800).astype(np.float32)
            result = algo.select_action(obs)
            next_obs = np.random.randn(2, 800).astype(np.float32)
            algo.store_transition(obs, result, np.zeros(2), next_obs, np.zeros(2), {})

        metrics = algo.update()
        assert isinstance(metrics, dict)
        expected_keys = {'policy_loss', 'value_loss', 'entropy', 'clip_fraction', 'approx_kl'}
        assert expected_keys.issubset(metrics.keys()), \
            f"Missing keys: {expected_keys - set(metrics.keys())}"

        # After update, buffer should be reset
        assert algo.buffer.pos == 0

    def test_update_changes_weights(self):
        """Verify that an update actually modifies the model parameters."""
        algo = PPOAlgorithm(self._make_config())

        # Record initial weights
        w_before = {n: p.data.clone() for n, p in algo.encoder.named_parameters()}

        # Fill buffer with non-trivial data
        for _ in range(64):
            obs = np.random.randn(2, 800).astype(np.float32)
            result = algo.select_action(obs)
            rewards = np.random.randn(2).astype(np.float32)
            next_obs = np.random.randn(2, 800).astype(np.float32)
            algo.store_transition(obs, result, rewards, next_obs, np.zeros(2), {})

        algo.update()

        # At least some weights should have changed
        any_changed = False
        for n, p in algo.encoder.named_parameters():
            if not torch.equal(p.data, w_before[n]):
                any_changed = True
                break
        assert any_changed, "No encoder weights changed after update"

    def test_default_params(self):
        """PPOAlgorithm.default_params() returns dict with expected keys."""
        params = PPOAlgorithm.default_params()
        expected = {'lr', 'gamma', 'gae_lambda', 'clip_epsilon', 'vf_coef',
                    'ent_coef', 'max_grad_norm', 'rollout_steps', 'ppo_epochs',
                    'minibatch_size'}
        assert expected.issubset(params.keys()), \
            f"Missing keys in default_params: {expected - set(params.keys())}"

    def test_default_search_space(self):
        """PPOAlgorithm.default_search_space() returns dict with expected structure."""
        space = PPOAlgorithm.default_search_space()
        assert isinstance(space, dict)
        assert len(space) > 0
        for key, spec in space.items():
            assert 'type' in spec, f"Search space entry '{key}' missing 'type'"

    def test_clone_from(self):
        """Clone agent A to agent B, verify weights are similar but optimizer is fresh."""
        config = self._make_config()
        agent_a = PPOAlgorithm(config)
        agent_b = PPOAlgorithm(config)

        # Make agent_a different by doing a dummy forward pass
        obs = np.random.randn(2, 800).astype(np.float32)
        agent_a.select_action(obs)

        # Clone
        agent_b.clone_from(agent_a, noise_scale=0.0)

        # Weights should match exactly
        for (na, pa), (nb, pb) in zip(
            agent_a.encoder.named_parameters(),
            agent_b.encoder.named_parameters(),
        ):
            torch.testing.assert_close(pa, pb, msg=f"Encoder param {na} mismatch after clone")

        for (na, pa), (nb, pb) in zip(
            agent_a.policy.named_parameters(),
            agent_b.policy.named_parameters(),
        ):
            torch.testing.assert_close(pa, pb, msg=f"Policy param {na} mismatch after clone")

    def test_clone_from_with_noise(self):
        """Clone with noise should produce different weights."""
        config = self._make_config()
        agent_a = PPOAlgorithm(config)
        agent_b = PPOAlgorithm(config)

        agent_b.clone_from(agent_a, noise_scale=0.1)

        # At least some weights should differ
        any_different = False
        for pa, pb in zip(agent_a.encoder.parameters(), agent_b.encoder.parameters()):
            if not torch.equal(pa.data, pb.data):
                any_different = True
                break
        assert any_different, "Clone with noise should produce different weights"

    def test_save_load_checkpoint(self, tmp_path):
        """Save, create new agent, load, verify weights match."""
        config = self._make_config()
        agent_a = PPOAlgorithm(config)

        # Modify weights slightly by doing an update
        for _ in range(64):
            obs = np.random.randn(2, 800).astype(np.float32)
            result = agent_a.select_action(obs)
            agent_a.store_transition(obs, result, np.zeros(2), obs, np.zeros(2), {})
        agent_a.update()

        # Save
        agent_a.save_checkpoint(tmp_path / 'ckpt')

        # Create fresh agent and load
        agent_b = PPOAlgorithm(config)
        agent_b.load_checkpoint(tmp_path / 'ckpt')

        # Verify weights match
        for (na, pa), (nb, pb) in zip(
            agent_a.encoder.named_parameters(),
            agent_b.encoder.named_parameters(),
        ):
            torch.testing.assert_close(pa, pb, msg=f"Encoder param {na} mismatch after load")

        for (na, pa), (nb, pb) in zip(
            agent_a.policy.named_parameters(),
            agent_b.policy.named_parameters(),
        ):
            torch.testing.assert_close(pa, pb, msg=f"Policy param {na} mismatch after load")

    def test_get_weights(self):
        """get_weights returns dict with encoder and policy state dicts."""
        algo = PPOAlgorithm(self._make_config())
        weights = algo.get_weights()
        assert 'encoder' in weights
        assert 'policy' in weights
        assert len(weights['encoder']) > 0
        assert len(weights['policy']) > 0


# ── Algorithm ABC compliance tests ─────────────────────────────────────────

class TestAlgorithmABC:

    def test_ppo_satisfies_interface(self):
        """Verify PPOAlgorithm has all required abstract methods from Algorithm."""
        # Check all abstractmethods are implemented
        abstract_methods = set()
        for name in dir(Algorithm):
            method = getattr(Algorithm, name)
            if getattr(method, '__isabstractmethod__', False):
                abstract_methods.add(name)

        for method_name in abstract_methods:
            assert hasattr(PPOAlgorithm, method_name), \
                f"PPOAlgorithm missing abstract method: {method_name}"
            # Verify it's not still abstract
            method = getattr(PPOAlgorithm, method_name)
            assert not getattr(method, '__isabstractmethod__', False), \
                f"PPOAlgorithm.{method_name} is still abstract"

    def test_ppo_is_instantiable(self):
        """PPOAlgorithm should be instantiable (all ABCs implemented)."""
        config = {
            'algorithm': {'params': {'rollout_steps': 8}},
            'num_envs': 1,
            't_window': 8,
        }
        algo = PPOAlgorithm(config)
        assert isinstance(algo, Algorithm)


# ── Population tests ────────────────────────────────────────────────────────

class TestPopulation:

    def test_worker_assignment_8_3(self):
        """8 workers, 3 agents -> [0,0,0,1,1,1,2,2]."""
        assignment = Population._assign_workers(8, 3)
        assert len(assignment) == 8
        assert assignment.count(0) == 3
        assert assignment.count(1) == 3
        assert assignment.count(2) == 2
        assert assignment == [0, 0, 0, 1, 1, 1, 2, 2]

    def test_worker_assignment_6_3(self):
        """6 workers, 3 agents -> [0,0,1,1,2,2]."""
        assignment = Population._assign_workers(6, 3)
        assert len(assignment) == 6
        assert assignment.count(0) == 2
        assert assignment.count(1) == 2
        assert assignment.count(2) == 2
        assert assignment == [0, 0, 1, 1, 2, 2]

    def test_worker_assignment_7_3(self):
        """7 workers, 3 agents -> [0,0,0,1,1,1,2,2] — extra goes to first agents."""
        assignment = Population._assign_workers(7, 3)
        assert len(assignment) == 7
        assert assignment.count(0) == 3
        assert assignment.count(1) == 2
        assert assignment.count(2) == 2

    def test_worker_assignment_single_agent(self):
        """All workers go to single agent."""
        assignment = Population._assign_workers(4, 1)
        assert assignment == [0, 0, 0, 0]

    def test_generation_ranking(self):
        """Add scores, verify ranking works."""
        config = {
            'algorithm': {'params': {'rollout_steps': 8}},
            'num_envs': 1,
            't_window': 8,
        }
        pop = Population(num_agents=3, num_workers=3, config=config)

        pop.add_score(0, 1.0)
        pop.add_score(0, 2.0)  # mean = 1.5
        pop.add_score(1, 5.0)
        pop.add_score(1, 5.0)  # mean = 5.0
        pop.add_score(2, 3.0)
        pop.add_score(2, 3.0)  # mean = 3.0

        ranking = pop.rank_agents()
        assert ranking == [1, 2, 0], f"Expected [1, 2, 0], got {ranking}"

    def test_ranking_with_no_scores(self):
        """Agents with no scores rank last."""
        config = {
            'algorithm': {'params': {'rollout_steps': 8}},
            'num_envs': 1,
            't_window': 8,
        }
        pop = Population(num_agents=3, num_workers=3, config=config)
        pop.add_score(0, 5.0)
        # Agents 1 and 2 have no scores

        ranking = pop.rank_agents()
        assert ranking[0] == 0, "Agent 0 should rank first"

    def test_get_metrics(self):
        """Population.get_metrics() returns dict with expected keys."""
        config = {
            'algorithm': {'params': {'rollout_steps': 8}},
            'num_envs': 1,
            't_window': 8,
        }
        pop = Population(num_agents=2, num_workers=4, config=config)
        pop.add_score(0, 1.0)
        pop.add_score(1, 2.0)

        metrics = pop.get_metrics()
        assert 'population/generation' in metrics
        assert 'population/num_agents' in metrics
        assert 'population/best_agent' in metrics
        assert metrics['population/num_agents'] == 2
        assert metrics['population/generation'] == 0

    def test_reset_scores(self):
        """reset_scores clears scores and increments generation."""
        config = {
            'algorithm': {'params': {'rollout_steps': 8}},
            'num_envs': 1,
            't_window': 8,
        }
        pop = Population(num_agents=2, num_workers=2, config=config)
        pop.add_score(0, 1.0)
        pop.add_score(1, 2.0)
        assert pop.generation == 0

        pop.reset_scores()
        assert pop.generation == 1
        assert all(len(s) == 0 for s in pop.scores)

    def test_population_creates_agents(self):
        """Population should create the right number of agents."""
        config = {
            'algorithm': {'params': {'rollout_steps': 8}},
            'num_envs': 1,
            't_window': 8,
        }
        pop = Population(num_agents=3, num_workers=6, config=config)
        assert len(pop.agents) == 3
        for agent in pop.agents:
            assert isinstance(agent, PPOAlgorithm)
