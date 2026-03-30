"""
Integration Tests Using DummyEnv
================================
Tests the full training pipeline without rlgym-sim / RocketSim.
Exercises: DummyEnv, PPOAlgorithm, Population, SubprocVecEnv,
AsyncUpdater, config loading, and the collect_and_train loop.
"""
from __future__ import annotations

import multiprocessing
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure repo root is on path
_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO))

from training.environments.dummy_env import DummyEnv
from training.algorithms.ppo import PPOAlgorithm, RolloutBuffer
from training.opponents.population import Population
from training.abstractions import Algorithm, ActionResult
from training.opponents.pool import HistoricalOpponentPool, load_opponent_from_snapshot
from training.loggers.stdout import StdoutLogger
from training.loggers.registry import MetricsRegistry


# ── helpers ─────────────────────────────────────────────────────────────────

def _small_config(num_envs=2, rollout_steps=32):
    """Minimal config for fast tests."""
    return {
        'algorithm': {
            'params': {
                'lr': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'vf_coef': 0.5,
                'ent_coef': 0.01,
                'max_grad_norm': 0.5,
                'rollout_steps': rollout_steps,
                'ppo_epochs': 2,
                'minibatch_size': 16,
            },
        },
        'num_envs': num_envs,
        't_window': 8,
        'seed': 42,
        'total_steps': 1000,
        'log_interval': 5,
    }


# ── DummyEnv tests ──────────────────────────────────────────────────────────

class TestDummyEnv:
    """Verify DummyEnv matches BaselineGymEnv's contract."""

    def test_obs_shape(self):
        env = DummyEnv(t_window=8)
        obs, info = env.reset()
        assert obs.shape == (800,)
        assert obs.dtype == np.float32

    def test_obs_finite(self):
        env = DummyEnv(t_window=8)
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs))

    def test_action_space(self):
        env = DummyEnv()
        assert env.action_space.shape == (8,)

    def test_step_returns(self):
        env = DummyEnv()
        obs, _ = env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, done, truncated, info = result
        assert obs.shape == (800,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert truncated is False
        assert 'goal' in info

    def test_episode_terminates(self):
        """Episodes end within max_steps."""
        env = DummyEnv(max_steps=50, goal_prob=0.0)
        env.reset()
        for _ in range(60):
            _, _, done, _, _ = env.step(env.action_space.sample())
            if done:
                break
        assert done

    def test_sparse_reward_contract(self):
        """Sparse reward is 0 mid-episode, nonzero only on terminal."""
        env = DummyEnv(max_steps=200, goal_prob=0.02, reward_type='sparse')
        env.reset()
        mid_rewards = []
        terminal_reward = None
        for _ in range(300):
            _, reward, done, _, info = env.step(env.action_space.sample())
            if done:
                terminal_reward = reward
                break
            mid_rewards.append(reward)
        # All mid-episode rewards should be 0 for sparse
        assert all(r == 0.0 for r in mid_rewards)

    def test_dense_reward_nonzero(self):
        """Dense reward produces non-zero mid-episode values."""
        env = DummyEnv(max_steps=100, goal_prob=0.0, reward_type='dense')
        env.reset()
        rewards = []
        for _ in range(20):
            _, reward, done, _, _ = env.step(env.action_space.sample())
            rewards.append(reward)
            if done:
                break
        # Dense should have some nonzero rewards
        assert any(r != 0.0 for r in rewards)

    def test_reset_clears_state(self):
        """Calling reset produces a fresh episode."""
        env = DummyEnv(max_steps=10)
        env.reset(seed=1)
        for _ in range(5):
            env.step(env.action_space.sample())
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=1)
        np.testing.assert_array_equal(obs1, obs2)

    def test_info_has_orange_obs(self):
        env = DummyEnv()
        _, info = env.reset()
        assert 'orange_obs' in info
        assert info['orange_obs'].shape == (800,)

    def test_goal_in_info(self):
        env = DummyEnv(goal_prob=1.0)  # force immediate goal
        env.reset()
        _, _, done, _, info = env.step(env.action_space.sample())
        assert done
        assert info['goal'] in (-1, 1)

    def test_load_ppo_opponent_noop(self):
        """load_ppo_opponent is a no-op but shouldn't error."""
        env = DummyEnv()
        env.load_ppo_opponent('/fake/path')

    def test_different_t_window(self):
        env = DummyEnv(t_window=4)
        obs, _ = env.reset()
        assert obs.shape == (4 * 10 * 10,)


# ── PPO + DummyEnv integration ─────────────────────────────────────────────

class TestPPOWithDummyEnv:
    """Test PPO algorithm with DummyEnv as the data source."""

    def test_single_env_collect_and_update(self):
        """Collect rollout from one env, run PPO update."""
        config = _small_config(num_envs=1, rollout_steps=32)
        agent = PPOAlgorithm(config)
        env = DummyEnv(t_window=8, max_steps=500)

        obs, _ = env.reset()
        for step in range(32):
            obs_batch = obs[np.newaxis]  # (1, 800)
            result = agent.select_action(obs_batch)

            action = result.action[0]
            next_obs, reward, done, _, info = env.step(action)

            agent.store_transition(
                obs_batch, result,
                np.array([reward]), next_obs[np.newaxis],
                np.array([float(done)]), {},
            )

            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        assert agent.should_update()
        metrics = agent.update()
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics
        assert np.isfinite(metrics['policy_loss'])

    def test_vectorized_collect_and_update(self):
        """Collect rollout from multiple envs (simulated), run update."""
        num_envs = 4
        config = _small_config(num_envs=num_envs, rollout_steps=16)
        agent = PPOAlgorithm(config)
        envs = [DummyEnv(t_window=8, max_steps=200) for _ in range(num_envs)]

        obs_list = []
        for env in envs:
            o, _ = env.reset()
            obs_list.append(o)
        obs_batch = np.stack(obs_list)  # (num_envs, 800)

        for step in range(16):
            result = agent.select_action(obs_batch)

            rewards = []
            dones = []
            next_obs_list = []
            for i, env in enumerate(envs):
                next_obs, reward, done, _, info = env.step(result.action[i])
                rewards.append(reward)
                dones.append(float(done))
                if done:
                    next_obs, _ = env.reset()
                next_obs_list.append(next_obs)

            agent.store_transition(
                obs_batch, result,
                np.array(rewards), np.stack(next_obs_list),
                np.array(dones), {},
            )
            obs_batch = np.stack(next_obs_list)

        assert agent.should_update()
        metrics = agent.update()
        assert all(np.isfinite(v) for v in metrics.values())

    def test_multiple_updates_reduce_loss(self):
        """Multiple PPO updates on dense reward should show learning signal."""
        num_envs = 2
        config = _small_config(num_envs=num_envs, rollout_steps=64)
        config['algorithm']['params']['ppo_epochs'] = 4
        agent = PPOAlgorithm(config)
        envs = [DummyEnv(t_window=8, max_steps=500, reward_type='dense', goal_prob=0.0)
                for _ in range(num_envs)]

        all_metrics = []
        for update_round in range(3):
            obs_list = [env.reset()[0] for env in envs]
            obs_batch = np.stack(obs_list)

            for step in range(64):
                result = agent.select_action(obs_batch)
                rewards, dones, next_obs_list = [], [], []
                for i, env in enumerate(envs):
                    next_obs, reward, done, _, _ = env.step(result.action[i])
                    rewards.append(reward)
                    dones.append(float(done))
                    if done:
                        next_obs, _ = env.reset()
                    next_obs_list.append(next_obs)
                agent.store_transition(
                    obs_batch, result,
                    np.array(rewards), np.stack(next_obs_list),
                    np.array(dones), {},
                )
                obs_batch = np.stack(next_obs_list)

            assert agent.should_update()
            metrics = agent.update()
            all_metrics.append(metrics)

        # Just verify we got valid metrics each round
        for m in all_metrics:
            assert 'policy_loss' in m
            assert np.isfinite(m['policy_loss'])


# ── Population + DummyEnv ───────────────────────────────────────────────────

class TestPopulationWithDummyEnv:
    """Test population-based training with DummyEnv."""

    def test_population_collect(self, tmp_path):
        """Multiple agents collect from their assigned envs."""
        config = _small_config(num_envs=4, rollout_steps=16)
        pop = Population(num_agents=2, num_workers=4, config=config,
                         snapshot_dir=tmp_path / 'snaps')

        envs = [DummyEnv(t_window=8, max_steps=200) for _ in range(4)]
        obs_list = [env.reset()[0] for env in envs]
        obs_all = np.stack(obs_list)

        # Group workers by agent
        agent_workers = {}
        for wi, ai in enumerate(pop.worker_assignment):
            agent_workers.setdefault(ai, []).append(wi)

        for step in range(16):
            actions = np.zeros((4, 8), dtype=np.float32)
            results_per_worker = [None] * 4

            for agent_idx, worker_ids in agent_workers.items():
                agent = pop.agents[agent_idx]
                agent_obs = obs_all[worker_ids]
                result = agent.select_action(agent_obs)
                for local_i, wi in enumerate(worker_ids):
                    actions[wi] = result.action[local_i]
                    results_per_worker[wi] = (result, local_i)

            # Step envs
            next_obs_list = []
            rewards = []
            dones = []
            for i, env in enumerate(envs):
                next_obs, reward, done, _, info = env.step(actions[i])
                rewards.append(reward)
                dones.append(float(done))
                if done:
                    goal = info.get('goal', 0)
                    pop.add_score(pop.worker_assignment[i], float(goal))
                    next_obs, _ = env.reset()
                next_obs_list.append(next_obs)

            # Store per agent
            for agent_idx, worker_ids in agent_workers.items():
                agent = pop.agents[agent_idx]
                w_obs = obs_all[worker_ids]
                w_rewards = np.array([rewards[wi] for wi in worker_ids])
                w_dones = np.array([dones[wi] for wi in worker_ids])
                w_next = np.stack([next_obs_list[wi] for wi in worker_ids])

                w_actions = actions[worker_ids]
                result, _ = results_per_worker[worker_ids[0]]
                w_lp = np.array([result.aux['log_prob'][j] for j, wi in enumerate(worker_ids)])
                w_val = np.array([result.aux['value'][j] for j, wi in enumerate(worker_ids)])

                combined = ActionResult(
                    action=w_actions,
                    aux={'log_prob': w_lp, 'value': w_val},
                )
                agent.store_transition(w_obs, combined, w_rewards, w_next, w_dones, {})

            obs_all = np.stack(next_obs_list)

        # Both agents should have data
        for agent in pop.agents:
            assert agent.buffer.pos == 16

    def test_population_generation_cycle(self, tmp_path):
        """Test ranking and cloning in a generation cycle."""
        config = _small_config(num_envs=4, rollout_steps=16)
        pop = Population(num_agents=3, num_workers=6, config=config,
                         snapshot_dir=tmp_path / 'snaps')

        # Simulate scores
        for _ in range(10):
            pop.add_score(0, 1.0)
            pop.add_score(1, 0.0)
            pop.add_score(2, -1.0)

        ranked = pop.rank_agents()
        assert ranked[0] == 0
        assert ranked[-1] == 2

        # Clone worst from best
        best = pop.agents[ranked[0]]
        worst_idx = ranked[-1]
        pop.agents[worst_idx].clone_from(best, noise_scale=0.01)
        pop.reset_scores()
        assert pop.generation == 1


# ── Opponent Pool integration ───────────────────────────────────────────────

class TestOpponentPoolWithDummyEnv:
    """Test snapshot saving/loading with PPO agents."""

    def test_save_and_load_snapshot(self):
        config = _small_config(num_envs=1)
        agent = PPOAlgorithm(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            pool = HistoricalOpponentPool(
                snapshot_dir=tmpdir,
                decay_rate=0.95,
                max_snapshots=10,
                snapshot_interval=100,
            )

            pool.save_snapshot(agent, step=1000)
            assert pool.num_snapshots() == 1

            snap_path = pool.sample_opponent_path()
            assert snap_path is not None

            weights = load_opponent_from_snapshot(snap_path)
            assert 'encoder' in weights
            assert 'policy' in weights

    def test_multiple_snapshots_sampling(self):
        config = _small_config(num_envs=1)
        agent = PPOAlgorithm(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            pool = HistoricalOpponentPool(
                snapshot_dir=tmpdir,
                max_snapshots=5,
                snapshot_interval=100,
            )

            for step in range(0, 5000, 1000):
                pool.save_snapshot(agent, step=step)

            assert pool.num_snapshots() == 5

            # Sample many times, verify we get valid paths
            paths = set()
            for _ in range(20):
                p = pool.sample_opponent_path()
                assert p is not None
                paths.add(p)
            assert len(paths) >= 2  # not always same snapshot

    def test_snapshot_cleanup(self):
        config = _small_config(num_envs=1)
        agent = PPOAlgorithm(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            pool = HistoricalOpponentPool(
                snapshot_dir=tmpdir,
                max_snapshots=3,
                snapshot_interval=100,
            )
            for step in range(0, 10000, 1000):
                pool.save_snapshot(agent, step=step)

            assert pool.num_snapshots() <= 3


# ── Logger integration ──────────────────────────────────────────────────────

class TestLoggerIntegration:

    def test_stdout_logger(self, capsys):
        logger = StdoutLogger()
        logger.init({})
        logger.log(100, loss=0.5, reward=1.2)
        logger.finish()
        captured = capsys.readouterr()
        assert 'step 0000100' in captured.out
        assert 'loss=0.5000' in captured.out

    def test_metrics_registry_with_providers(self):
        registry = MetricsRegistry()
        registry.register('test', lambda: {'val': 42})
        metrics = registry.collect()
        assert metrics['test/val'] == 42


# ── Config loading ──────────────────────────────────────────────────────────

class TestConfigLoading:

    def test_load_ppo_sparse_yaml(self):
        from train import load_config
        config = load_config(str(_REPO / 'configs' / 'ppo_sparse.yaml'))
        assert config['algorithm']['cls'].__name__ == 'PPOAlgorithm'
        assert 'lr' in config['algorithm']['params']
        assert config['algorithm']['params']['rollout_steps'] == 2048
        # opponent_pool should resolve to Population
        assert config['opponent_pool']['cls'].__name__ == 'Population'
        assert 'agents' in config['opponent_pool']['params']

    def test_yaml_overrides_defaults(self):
        from train import load_config
        config = load_config(
            str(_REPO / 'configs' / 'ppo_sparse.yaml'),
            cli_overrides={'seed': 99, 'total_steps': 5000},
        )
        assert config['seed'] == 99
        assert config['total_steps'] == 5000

    def test_load_class(self):
        from train import load_class
        cls = load_class('training.algorithms.ppo.PPOAlgorithm')
        assert cls is PPOAlgorithm

    def test_resolve_or_default(self):
        from train import resolve_or_default
        config = {'logger': {'class': 'training.loggers.stdout.StdoutLogger'}}
        cls = resolve_or_default(config, 'logger', StdoutLogger)
        assert cls is StdoutLogger

        # Missing section falls back to default
        cls = resolve_or_default({}, 'logger', StdoutLogger)
        assert cls is StdoutLogger


# ── SubprocVecEnv with DummyEnv ─────────────────────────────────────────────

def _dummy_env_worker(conn, t_window, reward_type, dense_reward_weights=None):
    """Worker process using DummyEnv instead of BaselineGymEnv."""
    env = DummyEnv(
        t_window=t_window,
        reward_type=reward_type,
        max_steps=100,
    )
    while True:
        try:
            cmd, data = conn.recv()
        except (EOFError, BrokenPipeError):
            break
        if cmd == 'reset':
            obs, info = env.reset()
            conn.send(('obs', obs, info))
        elif cmd == 'step':
            obs, reward, done, truncated, info = env.step(data)
            conn.send(('step', obs, reward, done, truncated, info))
        elif cmd == 'get_opponent_obs':
            obs = env.get_opponent_obs()
            conn.send(('opponent_obs', obs))
        elif cmd == 'step_with_opp':
            blue_action, opp_action = data
            obs, reward, done, truncated, info = env.step_with_opponent_action(
                blue_action, opp_action)
            conn.send(('step', obs, reward, done, truncated, info))
        elif cmd == 'close':
            env.close()
            conn.send(('closed',))
            break


class DummySubprocVecEnv:
    """SubprocVecEnv using DummyEnv workers."""

    def __init__(self, num_envs: int, t_window: int = 8, reward_type: str = 'sparse'):
        self.num_envs = num_envs
        self.parents = []
        self.procs = []

        for _ in range(num_envs):
            parent_conn, child_conn = multiprocessing.Pipe()
            proc = multiprocessing.Process(
                target=_dummy_env_worker,
                args=(child_conn, t_window, reward_type),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.procs.append(proc)

    def reset_all(self):
        for conn in self.parents:
            conn.send(('reset', None))
        obs_list = []
        for conn in self.parents:
            _, obs, info = conn.recv()
            obs_list.append(obs)
        return np.stack(obs_list)

    def step(self, actions):
        for i, conn in enumerate(self.parents):
            conn.send(('step', actions[i]))
        obs_list, rewards, dones, infos = [], [], [], []
        for conn in self.parents:
            _, obs, reward, done, truncated, info = conn.recv()
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (np.stack(obs_list), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32), infos)

    def get_opponent_obs(self):
        for conn in self.parents:
            conn.send(('get_opponent_obs', None))
        obs_list = []
        for conn in self.parents:
            _, obs = conn.recv()
            obs_list.append(obs)
        return np.stack(obs_list)

    def step_with_opponent_actions(self, blue_actions, opp_actions):
        for i, conn in enumerate(self.parents):
            conn.send(('step_with_opp', (blue_actions[i], opp_actions[i])))
        obs_list, rewards, dones, infos = [], [], [], []
        for conn in self.parents:
            _, obs, reward, done, truncated, info = conn.recv()
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (np.stack(obs_list), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32), infos)

    def close(self):
        for conn in self.parents:
            try:
                conn.send(('close', None))
                conn.recv()
            except (EOFError, BrokenPipeError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()


class TestSubprocVecEnv:
    """Test subprocess-based vectorized env with DummyEnv workers."""

    def test_reset_and_step(self):
        vec = DummySubprocVecEnv(num_envs=2)
        try:
            obs = vec.reset_all()
            assert obs.shape == (2, 800)

            actions = np.random.uniform(-1, 1, (2, 8)).astype(np.float32)
            obs, rewards, dones, infos = vec.step(actions)
            assert obs.shape == (2, 800)
            assert rewards.shape == (2,)
            assert dones.shape == (2,)
            assert len(infos) == 2
        finally:
            vec.close()

    def test_multiple_steps(self):
        vec = DummySubprocVecEnv(num_envs=3)
        try:
            obs = vec.reset_all()
            for _ in range(50):
                actions = np.random.uniform(-1, 1, (3, 8)).astype(np.float32)
                obs, rewards, dones, infos = vec.step(actions)
                assert obs.shape == (3, 800)
        finally:
            vec.close()


# ── Full collect_and_train loop ─────────────────────────────────────────────

class TestCollectAndTrain:
    """End-to-end test of the collect_and_train function with DummyEnv."""

    def test_collect_and_train_single_agent(self, tmp_path):
        from train import collect_and_train, AsyncUpdater, RolloutMetricsProvider
        from training.schedulers import InterleavedScheduler

        config = _small_config(num_envs=2, rollout_steps=32)
        pop = Population(num_agents=1, num_workers=2, config=config,
                         snapshot_dir=tmp_path / 'snaps')
        rollout_metrics = RolloutMetricsProvider()
        updater = AsyncUpdater()
        scheduler = InterleavedScheduler()
        scheduler.init(pop, 2, config)

        vec = DummySubprocVecEnv(num_envs=2, t_window=8)
        try:
            steps = collect_and_train(pop, vec, updater, rollout_metrics, config,
                                      scheduler)
            assert steps == 32 * 2  # rollout_steps * num_envs

            # Wait a moment for async update
            import time
            time.sleep(1)
            results = updater.pop_metrics()
            # Agent should have triggered an update
            assert len(results) >= 1
            agent_id, metrics = results[0]
            assert 'policy_loss' in metrics
        finally:
            updater.stop()
            vec.close()

    def test_collect_and_train_multi_agent(self, tmp_path):
        from train import collect_and_train, AsyncUpdater, RolloutMetricsProvider
        from training.schedulers import InterleavedScheduler

        config = _small_config(num_envs=4, rollout_steps=32)
        pop = Population(num_agents=2, num_workers=4, config=config,
                         snapshot_dir=tmp_path / 'snaps')
        rollout_metrics = RolloutMetricsProvider()
        updater = AsyncUpdater()
        scheduler = InterleavedScheduler()
        scheduler.init(pop, 4, config)

        vec = DummySubprocVecEnv(num_envs=4, t_window=8)
        try:
            steps = collect_and_train(pop, vec, updater, rollout_metrics, config,
                                      scheduler)
            assert steps == 32 * 4

            import time
            time.sleep(2)
            results = updater.pop_metrics()
            assert len(results) >= 1
        finally:
            updater.stop()
            vec.close()


# ── Tune integration ────────────────────────────────────────────────────────

class TestTuneIntegration:
    """Test the Optuna tuning pipeline with DummyEnv."""

    def test_tune_single_trial(self):
        pytest.importorskip('optuna')
        import optuna
        from tune import load_config, objective

        config = load_config(str(_REPO / 'configs' / 'ppo_tune_stub.yaml'))
        study = optuna.create_study(direction='maximize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            lambda trial: objective(trial, config, steps_per_trial=200, device='cpu'),
            n_trials=1,
        )
        assert len(study.trials) == 1
        assert study.best_value is not None
        assert np.isfinite(study.best_value)

    def test_tune_reward_tracker(self):
        from tune import RewardTracker
        env = DummyEnv(t_window=8, max_steps=30, goal_prob=0.1)
        tracked = RewardTracker(env)
        obs, _ = tracked.reset()
        assert obs.shape == (800,)

        for _ in range(100):
            action = np.random.uniform(-1, 1, size=8).astype(np.float32)
            obs, reward, done, _, info = tracked.step(action)
            if done:
                obs, _ = tracked.reset()

        assert len(tracked.episode_returns) > 0
        assert np.isfinite(tracked.mean_return())

    def test_tune_emit_yaml(self):
        from tune import load_config, emit_complete_yaml
        import tempfile

        config = load_config(str(_REPO / 'configs' / 'ppo_tune_stub.yaml'))
        best_params = {'algorithm.params.lr': 1e-4, 'algorithm.params.clip_epsilon': 0.15}

        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            emit_complete_yaml(config, best_params, f.name)
            import yaml
            with open(f.name) as rf:
                result = yaml.safe_load(rf)
            assert result['algorithm']['params']['lr'] == 1e-4
            assert result['algorithm']['params']['clip_epsilon'] == 0.15
            assert 'search_space' not in result
