"""
Comprehensive hyperparameter sweep for spectral encoder vs raw baseline.

Wide sweep across gamma, lr, hidden, std, lr_k with heavy per-episode and
per-checkpoint instrumentation. Outputs JSON for sweep_report.py graphing.

Usage:
    python training/spectral_sweep.py                    # full sweep
    python training/spectral_sweep.py --phase 1          # gamma sweep only
    python training/spectral_sweep.py --summary          # print results table
    python training/spectral_sweep.py --episodes 500     # shorter runs
"""

import json
import os
import sys
import time
import numpy as np

training_dir = os.path.dirname(os.path.abspath(__file__))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from pong_trainer import PongEnv, REINFORCEAgent


# ---------------------------------------------------------------------------
# Diagnostics collectors
# ---------------------------------------------------------------------------

def collect_agent_diagnostics(agent, obs_samples):
    """Per-layer activation stats from the agent's MLP."""
    d = {}

    if agent.hidden > 0:
        d['W1_norm'] = float(np.linalg.norm(agent.W1))
        d['W1_mean'] = float(agent.W1.mean())
        d['W1_std'] = float(agent.W1.std())
        d['W1_absmax'] = float(np.abs(agent.W1).max())
        d['b1_norm'] = float(np.linalg.norm(agent.b1))
        d['b1_mean'] = float(agent.b1.mean())
        d['W2_norm'] = float(np.linalg.norm(agent.W2))
        d['W2_mean'] = float(agent.W2.mean())
        d['W2_std'] = float(agent.W2.std())
        d['W2_absmax'] = float(np.abs(agent.W2).max())
        d['b2'] = float(agent.b2)

        if len(obs_samples) > 0:
            obs_arr = np.array(obs_samples)
            pre = obs_arr @ agent.W1.T + agent.b1
            post = np.maximum(pre, 0)

            d['pre_act_mean'] = float(pre.mean())
            d['pre_act_std'] = float(pre.std())
            d['pre_act_min'] = float(pre.min())
            d['pre_act_max'] = float(pre.max())
            d['post_act_mean'] = float(post.mean())
            d['post_act_std'] = float(post.std())

            fired = (post > 0).any(axis=0)
            d['dead_neurons'] = int((~fired).sum())
            d['alive_neurons'] = int(fired.sum())
            d['dead_frac'] = float((~fired).mean())

            neuron_means = post.mean(axis=0)
            d['neuron_act_max'] = float(neuron_means.max())
            d['neuron_act_min'] = float(neuron_means.min())
            d['neuron_act_std'] = float(neuron_means.std())
            # Per-neuron activation histogram (10 bins)
            d['neuron_act_hist'] = np.histogram(
                neuron_means, bins=10)[0].tolist()

            out = post @ agent.W2 + agent.b2
            d['output_mean'] = float(out.mean())
            d['output_std'] = float(out.std())
            d['output_min'] = float(out.min())
            d['output_max'] = float(out.max())
    else:
        d['w_norm'] = float(np.linalg.norm(agent.w))
        d['w_mean'] = float(agent.w.mean())
        d['w_std'] = float(agent.w.std())
        d['w_absmax'] = float(np.abs(agent.w).max())
        d['b'] = float(agent.b)

        if len(obs_samples) > 0:
            obs_arr = np.array(obs_samples)
            out = obs_arr @ agent.w + agent.b
            d['output_mean'] = float(out.mean())
            d['output_std'] = float(out.std())
            d['output_min'] = float(out.min())
            d['output_max'] = float(out.max())

    d['wv_norm'] = float(np.linalg.norm(agent.w_v))
    d['wv_mean'] = float(agent.w_v.mean())
    d['wv_std'] = float(agent.w_v.std())
    d['bv'] = float(agent.b_v)

    return d


def collect_spectral_diagnostics(env):
    """Strided conv layer stats + wavepacket state."""
    d = {}
    if env.obs_mode != 'spectral':
        return d

    sc = env._strided_conv
    d['conv_W1_norm'] = float(np.linalg.norm(sc.W1))
    d['conv_W2_norm'] = float(np.linalg.norm(sc.W2))

    try:
        fmaps = env._compute_fmaps()
        d['fmap_mean'] = float(fmaps.mean())
        d['fmap_std'] = float(fmaps.std())
        d['fmap_absmax'] = float(np.abs(fmaps).max())
        for ch in range(min(fmaps.shape[0], 6)):
            d[f'fmap_ch{ch}_mean'] = float(fmaps[ch].mean())
            d[f'fmap_ch{ch}_std'] = float(fmaps[ch].std())
            d[f'fmap_ch{ch}_absmax'] = float(np.abs(fmaps[ch]).max())

        # Strided conv activations
        h1 = sc._strided_conv2d(fmaps, sc.W1, sc.b1, sc.stride)
        h1r = np.maximum(h1, 0)
        h2 = sc._strided_conv2d(h1r, sc.W2, sc.b2, sc.stride)
        h2r = np.maximum(h2, 0)

        d['conv_h1_mean'] = float(h1r.mean())
        d['conv_h1_std'] = float(h1r.std())
        d['conv_h1_max'] = float(h1r.max())
        d['conv_h1_dead'] = int((h1r.max(axis=(1, 2)) == 0).sum())
        d['conv_h2_mean'] = float(h2r.mean())
        d['conv_h2_std'] = float(h2r.std())
        d['conv_h2_max'] = float(h2r.max())
        d['conv_h2_dead'] = int((h2r.max(axis=(1, 2)) == 0).sum())

        obs_flat = h2r.ravel()
        d['obs_nonzero'] = int((obs_flat > 0).sum())
        d['obs_total'] = len(obs_flat)
        d['obs_nonzero_frac'] = float((obs_flat > 0).mean())
        d['obs_mean'] = float(obs_flat.mean())
        d['obs_std'] = float(obs_flat.std())
        # Obs histogram (10 bins from 0 to max)
        if obs_flat.max() > 0:
            d['obs_hist'] = np.histogram(
                obs_flat, bins=10, range=(0, obs_flat.max()))[0].tolist()
    except Exception as e:
        d['error'] = str(e)

    # Wavepacket state
    try:
        wp_ball = env._wp_ball
        wp_env = env._wp_env
        wp_pl = env._wp_pl
        wp_pr = env._wp_pr

        for name, wp in [('ball', wp_ball), ('env', wp_env),
                         ('padL', wp_pl), ('padR', wp_pr)]:
            for ax in range(min(wp.ndim, 3)):
                cn = float(np.linalg.norm(wp.c_cos[:, ax]))
                sn = float(np.linalg.norm(wp.c_sin[:, ax]))
                d[f'wp_{name}_c{ax}_norm'] = cn + sn

        # Frequencies
        d['wp_ball_freqs'] = wp_ball.k.tolist()
        d['wp_env_freqs'] = wp_env.k.tolist()

        # NIPs (attention weights)
        d['nip_ball_env'] = float(abs(
            wp_ball.normalized_inner_product(wp_env)))
        d['nip_ball_padL'] = float(abs(
            wp_ball.normalized_inner_product(wp_pl)))
        d['nip_ball_padR'] = float(abs(
            wp_ball.normalized_inner_product(wp_pr)))
    except Exception as e:
        d['wp_error'] = str(e)

    return d


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config, n_episodes=2000, diag_interval=250,
                   max_steps=2000):
    """Run one experiment with full instrumentation."""
    np.random.seed(config.get('seed', 0))
    env = PongEnv(
        opp_skill=config.get('opp_skill', 0.0),
        reward_mode=config['reward_mode'],
        obs_mode=config['obs_mode'],
        lr_k_env=config.get('lr_k', 0.0),
    )
    agent = REINFORCEAgent(
        obs_dim=env.obs_dim,
        hidden=config['hidden'],
        lr=config['lr'],
        lr_baseline=config.get('lr_baseline', 1e-2),
        gamma=config['gamma'],
        std=config['std'],
    )

    # Per-episode data (compact — arrays not dicts)
    ep_wins = []
    ep_losses = []
    ep_lengths = []
    ep_touches = []
    ep_returns = []
    ep_action_means = []
    ep_action_stds = []
    ep_value_errors = []
    ep_advantage_stds = []
    ep_obs_stds = []

    # Checkpoint diagnostics
    diag_history = []
    obs_collector = []

    t0 = time.time()

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            action = agent.act(obs)
            if len(obs_collector) < 300:
                obs_collector.append(obs.copy())
            obs, reward, done = env.step(action)
            agent.record_reward(reward)
            ep_reward += reward
            if done:
                break
        ep_return = agent.end_episode()

        # Record per-episode stats
        ep_wins.append(1 if ep_reward > 0 else 0)
        ep_losses.append(1 if ep_reward < 0 else 0)
        ep_lengths.append(step + 1)
        ep_touches.append(env.agent_touches)
        ep_returns.append(float(ep_return))

        # Agent episode stats
        stats = getattr(agent, 'last_ep_stats', {})
        ep_action_means.append(stats.get('action_mean', 0.0))
        ep_action_stds.append(stats.get('action_std', 0.0))
        ep_value_errors.append(stats.get('value_error', 0.0))
        ep_advantage_stds.append(stats.get('advantage_std', 0.0))
        ep_obs_stds.append(stats.get('obs_std', 0.0))

        # Checkpoint diagnostics
        if (ep + 1) % diag_interval == 0:
            agent_diag = collect_agent_diagnostics(agent, obs_collector)
            spectral_diag = collect_spectral_diagnostics(env)
            diag_history.append({
                'ep': ep + 1,
                'wall_time': round(time.time() - t0, 1),
                'agent': agent_diag,
                'spectral': spectral_diag,
            })
            obs_collector.clear()

    wall_time = time.time() - t0

    # Compute windowed metrics for eval curve (100-ep windows)
    eval_curve = []
    window = 100
    for start in range(0, n_episodes, window):
        end = min(start + window, n_episodes)
        w = sum(ep_wins[start:end])
        l = sum(ep_losses[start:end])
        n = end - start
        eval_curve.append({
            'ep': end,
            'win_rate': round(w / n, 3),
            'loss_rate': round(l / n, 3),
            'avg_touches': round(np.mean(ep_touches[start:end]), 2),
            'avg_len': round(np.mean(ep_lengths[start:end]), 1),
            'avg_return': round(np.mean(ep_returns[start:end]), 4),
            'avg_action_mean': round(np.mean(ep_action_means[start:end]), 4),
            'avg_action_std': round(np.mean(ep_action_stds[start:end]), 4),
            'avg_value_error': round(np.mean(ep_value_errors[start:end]), 4),
            'avg_advantage_std': round(
                np.mean(ep_advantage_stds[start:end]), 4),
            'avg_obs_std': round(np.mean(ep_obs_stds[start:end]), 4),
        })

    # Final summary (last 500 episodes)
    tail = min(500, n_episodes)
    final_wins = sum(ep_wins[-tail:])
    final_losses = sum(ep_losses[-tail:])
    final_wr = round(final_wins / tail, 3)

    # Peak win rate (best 100-ep window)
    peak_wr = max(e['win_rate'] for e in eval_curve) if eval_curve else 0.0

    return {
        'config': config,
        'wall_time': round(wall_time, 1),
        'n_episodes': n_episodes,
        'final_win_rate': final_wr,
        'final_wins': final_wins,
        'final_losses': final_losses,
        'peak_win_rate': peak_wr,
        'eval_curve': eval_curve,
        'diagnostics': diag_history,
    }


# ---------------------------------------------------------------------------
# Sweep grid builder
# ---------------------------------------------------------------------------

def build_sweep_configs(phases=None):
    """Build prioritized sweep grid.

    Phases:
      1: Gamma sweep (spectral vs raw × paddle & goal) — 32 experiments
      2: LR sweep at key gammas — 24 experiments
      3: Hidden size sweep — 20 experiments
      4: Std (exploration noise) sweep — 20 experiments
      5: lr_k (frequency learning) sweep — 8 experiments
    """
    if phases is None:
        phases = [1, 2, 3, 4, 5]

    configs = []

    # Defaults
    default_lr = 1e-2
    default_hidden = 32
    default_std = 0.5
    default_seed = 0

    # Phase 1: Gamma sweep — the core question
    if 1 in phases:
        for gamma in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            for obs_mode in ['spectral', 'raw']:
                for reward_mode in ['paddle', 'goal']:
                    configs.append({
                        'name': f'p1_gamma{gamma}_{obs_mode}_{reward_mode}',
                        'phase': 1,
                        'obs_mode': obs_mode,
                        'reward_mode': reward_mode,
                        'gamma': gamma,
                        'lr': default_lr,
                        'hidden': default_hidden,
                        'std': default_std,
                        'seed': default_seed,
                    })

    # Phase 2: LR sweep at interesting gammas for spectral
    if 2 in phases:
        for gamma in [0.0, 0.5, 0.9]:
            for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
                for reward_mode in ['paddle', 'goal']:
                    configs.append({
                        'name': f'p2_lr{lr}_g{gamma}_spec_{reward_mode}',
                        'phase': 2,
                        'obs_mode': 'spectral',
                        'reward_mode': reward_mode,
                        'gamma': gamma,
                        'lr': lr,
                        'hidden': default_hidden,
                        'std': default_std,
                        'seed': default_seed,
                    })

    # Phase 3: Hidden size sweep
    if 3 in phases:
        for gamma in [0.0, 0.5]:
            for hidden in [0, 16, 32, 64, 128]:
                for reward_mode in ['paddle', 'goal']:
                    configs.append({
                        'name': f'p3_h{hidden}_g{gamma}_spec_{reward_mode}',
                        'phase': 3,
                        'obs_mode': 'spectral',
                        'reward_mode': reward_mode,
                        'gamma': gamma,
                        'lr': default_lr,
                        'hidden': hidden,
                        'std': default_std,
                        'seed': default_seed,
                    })

    # Phase 4: Exploration noise sweep
    if 4 in phases:
        for gamma in [0.0, 0.5]:
            for std in [0.1, 0.3, 0.5, 0.8, 1.0]:
                for reward_mode in ['paddle', 'goal']:
                    configs.append({
                        'name': f'p4_std{std}_g{gamma}_spec_{reward_mode}',
                        'phase': 4,
                        'obs_mode': 'spectral',
                        'reward_mode': reward_mode,
                        'gamma': gamma,
                        'lr': default_lr,
                        'hidden': default_hidden,
                        'std': std,
                        'seed': default_seed,
                    })

    # Phase 5: Frequency learning rate
    if 5 in phases:
        for lr_k in [1e-4, 1e-3, 1e-2, 0.1]:
            for reward_mode in ['paddle', 'goal']:
                configs.append({
                    'name': f'p5_lrk{lr_k}_spec_{reward_mode}',
                    'phase': 5,
                    'obs_mode': 'spectral',
                    'reward_mode': reward_mode,
                    'gamma': 0.5,  # will use best from phase 1
                    'lr': default_lr,
                    'hidden': default_hidden,
                    'std': default_std,
                    'lr_k': lr_k,
                    'seed': default_seed,
                })

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        if c['name'] not in seen:
            seen.add(c['name'])
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(n_episodes=2000, phases=None, output_path=None, resume=True):
    """Run the full sweep with incremental saving."""
    configs = build_sweep_configs(phases)

    if output_path is None:
        output_path = os.path.join(training_dir, 'sweep_results.json')

    # Resume: skip already-completed experiments
    completed = set()
    all_results = []
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            all_results = json.load(f)
        completed = {r['config']['name'] for r in all_results}
        print(f'Resuming: {len(completed)} experiments already done')

    remaining = [c for c in configs if c['name'] not in completed]
    print(f'Sweep: {len(remaining)} experiments to run '
          f'({len(configs)} total, {len(completed)} done), '
          f'{n_episodes} episodes each')
    print()

    for i, config in enumerate(remaining):
        name = config['name']
        phase = config.get('phase', '?')
        print(f'[{len(completed)+i+1}/{len(configs)}] P{phase} {name} ...',
              end=' ', flush=True)

        t0 = time.time()
        result = run_experiment(config, n_episodes=n_episodes)
        dt = time.time() - t0
        wr = result['final_win_rate']
        peak = result['peak_win_rate']
        print(f'{dt:.0f}s — WR={wr:.0%} peak={peak:.0%}')

        # Print compact learning curve
        evals = result['eval_curve']
        pts = evals[::max(1, len(evals)//6)]  # ~6 points
        curve = ' '.join(f"{e['win_rate']:.0%}" for e in pts)
        print(f'  curve: {curve}')

        all_results.append(result)

        # Save incrementally
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=1)

    print(f'\nSweep complete. {len(all_results)} total results '
          f'saved to {output_path}')
    return all_results


def print_summary(results_path=None):
    """Print ranked summary table."""
    if results_path is None:
        results_path = os.path.join(training_dir, 'sweep_results.json')
    with open(results_path) as f:
        results = json.load(f)

    print(f'\n{"#":<3} {"Name":<50} {"WR":>6} {"Peak":>6} '
          f'{"Time":>6} {"Phase":>5}')
    print('-' * 80)
    for i, r in enumerate(sorted(results,
                                  key=lambda x: -x['final_win_rate'])):
        c = r['config']
        print(f'{i+1:<3} {c["name"]:<50} '
              f'{r["final_win_rate"]:>5.0%} '
              f'{r["peak_win_rate"]:>5.0%} '
              f'{r["wall_time"]:>5.0f}s '
              f'P{c.get("phase", "?"):>3}')

    # Group by obs_mode
    print('\n--- By obs_mode ---')
    for mode in ['spectral', 'raw']:
        subset = [r for r in results if r['config']['obs_mode'] == mode]
        if subset:
            best = max(subset, key=lambda x: x['final_win_rate'])
            avg = np.mean([r['final_win_rate'] for r in subset])
            print(f'{mode:>10}: n={len(subset)}, '
                  f'avg_WR={avg:.0%}, '
                  f'best_WR={best["final_win_rate"]:.0%} '
                  f'({best["config"]["name"]})')

    # Group by gamma
    print('\n--- By gamma (spectral only) ---')
    spec = [r for r in results if r['config']['obs_mode'] == 'spectral']
    gammas = sorted(set(r['config']['gamma'] for r in spec))
    for g in gammas:
        subset = [r for r in spec if r['config']['gamma'] == g]
        avg = np.mean([r['final_win_rate'] for r in subset])
        best = max(subset, key=lambda x: x['final_win_rate'])
        print(f'  gamma={g:<5}: n={len(subset)}, avg_WR={avg:.0%}, '
              f'best={best["final_win_rate"]:.0%}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Spectral encoder hyperparameter sweep')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--phase', type=int, nargs='+', default=None,
                        help='Run specific phases (1-5)')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    if args.summary:
        print_summary(args.output)
    else:
        run_sweep(
            n_episodes=args.episodes,
            phases=args.phase,
            output_path=args.output,
            resume=not args.no_resume,
        )
