"""
Generate analysis plots from spectral_sweep.py results.

Usage:
    python training/sweep_report.py                          # all plots
    python training/sweep_report.py --input sweep_results.json
    python training/sweep_report.py --plots gamma,lr         # specific plots
"""

import json
import os
import sys
import numpy as np

training_dir = os.path.dirname(os.path.abspath(__file__))

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_results(path=None):
    if path is None:
        path = os.path.join(training_dir, 'sweep_results.json')
    with open(path) as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def filter_results(results, **kwargs):
    """Filter results by config keys."""
    out = []
    for r in results:
        c = r['config']
        if all(c.get(k) == v for k, v in kwargs.items()):
            out.append(r)
    return out


def get_eval_series(result, key='win_rate'):
    """Extract time series from eval_curve."""
    curve = result.get('eval_curve', [])
    eps = [e['ep'] for e in curve]
    vals = [e[key] for e in curve]
    return eps, vals


# ---------------------------------------------------------------------------
# Plot 1: Gamma comparison (spectral vs raw)
# ---------------------------------------------------------------------------

def plot_gamma_comparison(results, out_dir):
    """Win rate vs gamma for spectral vs raw, paddle and goal."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, reward_mode in zip(axes, ['paddle', 'goal']):
        for obs_mode, color, marker in [('spectral', '#38bdf8', 'o'),
                                         ('raw', '#f87171', 's')]:
            subset = filter_results(results, obs_mode=obs_mode,
                                    reward_mode=reward_mode)
            if not subset:
                continue
            gammas = sorted(set(r['config']['gamma'] for r in subset))
            wrs = []
            peaks = []
            for g in gammas:
                match = [r for r in subset if r['config']['gamma'] == g]
                if match:
                    wrs.append(match[0]['final_win_rate'])
                    peaks.append(match[0]['peak_win_rate'])
            ax.plot(gammas, wrs, f'{marker}-', color=color,
                    label=f'{obs_mode} final', linewidth=2, markersize=8)
            ax.plot(gammas, peaks, f'{marker}--', color=color,
                    label=f'{obs_mode} peak', linewidth=1, alpha=0.6)

        ax.set_xlabel('Gamma', fontsize=12)
        ax.set_ylabel('Win Rate', fontsize=12)
        ax.set_title(f'Gamma Sweep — {reward_mode} reward', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_gamma_comparison.png'), dpi=150)
    plt.close(fig)
    print('  01_gamma_comparison.png')


# ---------------------------------------------------------------------------
# Plot 2: Learning curves by gamma
# ---------------------------------------------------------------------------

def plot_learning_curves_gamma(results, out_dir):
    """Win rate over episodes for each gamma, spectral vs raw."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for row, reward_mode in enumerate(['paddle', 'goal']):
        for col, obs_mode in enumerate(['spectral', 'raw']):
            ax = axes[row, col]
            subset = filter_results(results, obs_mode=obs_mode,
                                    reward_mode=reward_mode)
            gammas = sorted(set(r['config']['gamma'] for r in subset))
            cmap = plt.cm.viridis(np.linspace(0, 1, len(gammas)))

            for g, color in zip(gammas, cmap):
                match = [r for r in subset if r['config']['gamma'] == g]
                if match:
                    eps, wrs = get_eval_series(match[0], 'win_rate')
                    ax.plot(eps, wrs, color=color, label=f'γ={g}',
                            linewidth=1.5)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Win Rate (100-ep window)')
            ax.set_title(f'{obs_mode} / {reward_mode}')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    fig.suptitle('Learning Curves by Gamma', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_learning_curves_gamma.png'), dpi=150)
    plt.close(fig)
    print('  02_learning_curves_gamma.png')


# ---------------------------------------------------------------------------
# Plot 3: Touch rate curves
# ---------------------------------------------------------------------------

def plot_touch_curves(results, out_dir):
    """Touches per episode over training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, obs_mode in zip(axes, ['spectral', 'raw']):
        subset = filter_results(results, obs_mode=obs_mode,
                                reward_mode='paddle')
        gammas = sorted(set(r['config']['gamma'] for r in subset))
        cmap = plt.cm.viridis(np.linspace(0, 1, len(gammas)))

        for g, color in zip(gammas, cmap):
            match = [r for r in subset if r['config']['gamma'] == g]
            if match:
                eps, touches = get_eval_series(match[0], 'avg_touches')
                ax.plot(eps, touches, color=color, label=f'γ={g}',
                        linewidth=1.5)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg Touches/Episode')
        ax.set_title(f'{obs_mode} — paddle reward')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Touch Rate by Gamma', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_touch_curves.png'), dpi=150)
    plt.close(fig)
    print('  03_touch_curves.png')


# ---------------------------------------------------------------------------
# Plot 4: LR × Gamma heatmap (spectral)
# ---------------------------------------------------------------------------

def plot_lr_gamma_heatmap(results, out_dir):
    """Win rate heatmap: lr vs gamma for spectral."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, reward_mode in zip(axes, ['paddle', 'goal']):
        subset = filter_results(results, obs_mode='spectral',
                                reward_mode=reward_mode)
        if len(subset) < 4:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{reward_mode}')
            continue

        gammas = sorted(set(r['config']['gamma'] for r in subset))
        lrs = sorted(set(r['config']['lr'] for r in subset))

        grid = np.full((len(lrs), len(gammas)), np.nan)
        for r in subset:
            gi = gammas.index(r['config']['gamma'])
            li = lrs.index(r['config']['lr'])
            grid[li, gi] = r['final_win_rate']

        im = ax.imshow(grid, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=1, origin='lower')
        ax.set_xticks(range(len(gammas)))
        ax.set_xticklabels([str(g) for g in gammas], fontsize=8)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f'{l:.0e}' for l in lrs], fontsize=8)
        ax.set_xlabel('Gamma')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'Spectral WR — {reward_mode}')

        # Annotate cells
        for li in range(len(lrs)):
            for gi in range(len(gammas)):
                v = grid[li, gi]
                if not np.isnan(v):
                    ax.text(gi, li, f'{v:.0%}', ha='center', va='center',
                            fontsize=8, color='black' if v > 0.3 else 'white')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('LR × Gamma Heatmap (Spectral)', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_lr_gamma_heatmap.png'), dpi=150)
    plt.close(fig)
    print('  04_lr_gamma_heatmap.png')


# ---------------------------------------------------------------------------
# Plot 5: Hidden size effect
# ---------------------------------------------------------------------------

def plot_hidden_effect(results, out_dir):
    """Win rate vs hidden layer size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, reward_mode in zip(axes, ['paddle', 'goal']):
        subset = filter_results(results, obs_mode='spectral',
                                reward_mode=reward_mode)
        gammas = sorted(set(r['config']['gamma'] for r in subset))
        cmap = plt.cm.Set1(np.linspace(0, 1, len(gammas)))

        for g, color in zip(gammas, cmap):
            match = sorted(
                [r for r in subset if r['config']['gamma'] == g],
                key=lambda r: r['config']['hidden'])
            if len(match) < 2:
                continue
            hiddens = [r['config']['hidden'] for r in match]
            wrs = [r['final_win_rate'] for r in match]
            ax.plot(hiddens, wrs, 'o-', color=color, label=f'γ={g}',
                    linewidth=2, markersize=8)

        ax.set_xlabel('Hidden Size')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'Hidden Size Effect — {reward_mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '05_hidden_effect.png'), dpi=150)
    plt.close(fig)
    print('  05_hidden_effect.png')


# ---------------------------------------------------------------------------
# Plot 6: Exploration noise effect
# ---------------------------------------------------------------------------

def plot_std_effect(results, out_dir):
    """Win rate vs exploration std."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, reward_mode in zip(axes, ['paddle', 'goal']):
        subset = filter_results(results, obs_mode='spectral',
                                reward_mode=reward_mode)
        gammas = sorted(set(r['config']['gamma'] for r in subset))
        cmap = plt.cm.Set1(np.linspace(0, 1, len(gammas)))

        for g, color in zip(gammas, cmap):
            match = sorted(
                [r for r in subset if r['config']['gamma'] == g],
                key=lambda r: r['config']['std'])
            stds_seen = set()
            filtered = []
            for r in match:
                s = r['config']['std']
                if s not in stds_seen:
                    stds_seen.add(s)
                    filtered.append(r)
            if len(filtered) < 2:
                continue
            stds = [r['config']['std'] for r in filtered]
            wrs = [r['final_win_rate'] for r in filtered]
            ax.plot(stds, wrs, 'o-', color=color, label=f'γ={g}',
                    linewidth=2, markersize=8)

        ax.set_xlabel('Exploration Std')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'Exploration Noise — {reward_mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '06_std_effect.png'), dpi=150)
    plt.close(fig)
    print('  06_std_effect.png')


# ---------------------------------------------------------------------------
# Plot 7: Activation evolution
# ---------------------------------------------------------------------------

def plot_activation_evolution(results, out_dir):
    """Weight norms, dead neurons, output std over training."""
    # Pick experiments with diagnostics and hidden > 0
    spectral = [r for r in results
                if r['config']['obs_mode'] == 'spectral'
                and r['config']['hidden'] > 0
                and len(r.get('diagnostics', [])) > 0]

    if len(spectral) == 0:
        print('  07_activation_evolution.png — SKIPPED (no data)')
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ('W1_norm', 'W1 Norm'),
        ('dead_neurons', 'Dead Neurons'),
        ('output_std', 'Output Std'),
        ('wv_norm', 'Value Baseline Norm'),
        ('post_act_mean', 'Post-ReLU Mean'),
        ('pre_act_std', 'Pre-activation Std'),
    ]

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        n_plotted = 0
        for r in spectral[:15]:  # limit to 15 experiments
            diags = r['diagnostics']
            eps_d = [d['ep'] for d in diags]
            vals = [d['agent'].get(key, 0) for d in diags]
            if any(v != 0 for v in vals):
                gamma = r['config']['gamma']
                lr = r['config']['lr']
                label = f'γ={gamma} lr={lr}'
                ax.plot(eps_d, vals, 'o-', markersize=3,
                        label=label if n_plotted < 8 else None,
                        alpha=0.7)
                n_plotted += 1
        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.set_title(title)
        if n_plotted <= 8:
            ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Agent Activation Evolution (Spectral)', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '07_activation_evolution.png'), dpi=150)
    plt.close(fig)
    print('  07_activation_evolution.png')


# ---------------------------------------------------------------------------
# Plot 8: Spectral encoder diagnostics
# ---------------------------------------------------------------------------

def plot_spectral_diagnostics(results, out_dir):
    """Feature map and conv layer stats over time."""
    spectral = [r for r in results
                if r['config']['obs_mode'] == 'spectral'
                and len(r.get('diagnostics', [])) > 0]

    if len(spectral) == 0:
        print('  08_spectral_diagnostics.png — SKIPPED')
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ('fmap_std', 'Feature Map Std'),
        ('conv_h1_mean', 'Conv L1 Activation Mean'),
        ('conv_h2_mean', 'Conv L2 Activation Mean'),
        ('obs_nonzero_frac', 'Obs Nonzero Fraction'),
        ('nip_ball_env', 'NIP Ball↔Env'),
        ('nip_ball_padL', 'NIP Ball↔PadL'),
    ]

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        n_plotted = 0
        for r in spectral[:15]:
            diags = r['diagnostics']
            eps_d = [d['ep'] for d in diags]
            vals = [d['spectral'].get(key, 0) for d in diags]
            if any(v != 0 for v in vals):
                gamma = r['config']['gamma']
                label = f'γ={gamma}'
                ax.plot(eps_d, vals, 'o-', markersize=3,
                        label=label if n_plotted < 8 else None,
                        alpha=0.7)
                n_plotted += 1
        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.set_title(title)
        if n_plotted <= 8:
            ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Spectral Encoder Diagnostics', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '08_spectral_diagnostics.png'), dpi=150)
    plt.close(fig)
    print('  08_spectral_diagnostics.png')


# ---------------------------------------------------------------------------
# Plot 9: Per-channel feature map evolution
# ---------------------------------------------------------------------------

def plot_per_channel_fmaps(results, out_dir):
    """Per-channel feature map std over time."""
    spectral = [r for r in results
                if r['config']['obs_mode'] == 'spectral'
                and len(r.get('diagnostics', [])) > 0]

    if len(spectral) == 0:
        print('  09_per_channel_fmaps.png — SKIPPED')
        return

    ch_labels = ['ball', 'env', 'padL', 'padR', 'rew', 'b×r']
    ch_colors = ['#38bdf8', '#c084fc', '#4ade80', '#f87171',
                 '#fb923c', '#facc15']

    # Pick up to 4 experiments with different gammas
    gammas_seen = set()
    selected = []
    for r in spectral:
        g = r['config']['gamma']
        if g not in gammas_seen and len(selected) < 4:
            gammas_seen.add(g)
            selected.append(r)

    fig, axes = plt.subplots(1, len(selected), figsize=(6*len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, r in zip(axes, selected):
        gamma = r['config']['gamma']
        diags = r['diagnostics']
        eps_d = [d['ep'] for d in diags]

        for ch in range(6):
            key = f'fmap_ch{ch}_std'
            vals = [d['spectral'].get(key, 0) for d in diags]
            ax.plot(eps_d, vals, color=ch_colors[ch],
                    label=ch_labels[ch], linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Channel Std')
        ax.set_title(f'γ={gamma}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Per-Channel Feature Map Std Evolution', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '09_per_channel_fmaps.png'), dpi=150)
    plt.close(fig)
    print('  09_per_channel_fmaps.png')


# ---------------------------------------------------------------------------
# Plot 10: Gradient flow indicators
# ---------------------------------------------------------------------------

def plot_gradient_flow(results, out_dir):
    """Return variance, advantage std, value error over training."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ('avg_advantage_std', 'Advantage Std'),
        ('avg_value_error', 'Value Prediction Error'),
        ('avg_return', 'Average Return'),
    ]

    spectral = [r for r in results
                if r['config']['obs_mode'] == 'spectral'
                and len(r.get('eval_curve', [])) > 0]

    for ax, (key, title) in zip(axes, metrics):
        n_plotted = 0
        for r in spectral[:12]:
            curve = r['eval_curve']
            eps_d = [e['ep'] for e in curve]
            vals = [e.get(key, 0) for e in curve]
            gamma = r['config']['gamma']
            ax.plot(eps_d, vals, alpha=0.6,
                    label=f'γ={gamma}' if n_plotted < 8 else None)
            n_plotted += 1

        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.set_title(title)
        if n_plotted <= 8:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Gradient Flow Indicators (Spectral)', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '10_gradient_flow.png'), dpi=150)
    plt.close(fig)
    print('  10_gradient_flow.png')


# ---------------------------------------------------------------------------
# Plot 11: Summary dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(results, out_dir):
    """4-panel summary: best curves, gamma sensitivity, health, bar chart."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 2, figure=fig)

    # Panel 1: Best spectral vs best raw learning curves
    ax1 = fig.add_subplot(gs[0, 0])
    for obs_mode, color in [('spectral', '#38bdf8'), ('raw', '#f87171')]:
        subset = [r for r in results if r['config']['obs_mode'] == obs_mode]
        if subset:
            best = max(subset, key=lambda x: x['final_win_rate'])
            eps, wrs = get_eval_series(best, 'win_rate')
            c = best['config']
            label = (f'{obs_mode} (γ={c["gamma"]}, lr={c["lr"]}, '
                     f'h={c["hidden"]}) WR={best["final_win_rate"]:.0%}')
            ax1.plot(eps, wrs, color=color, linewidth=2, label=label)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Best Spectral vs Best Raw')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    # Panel 2: Gamma sensitivity (spectral paddle)
    ax2 = fig.add_subplot(gs[0, 1])
    for reward_mode, color, marker in [('paddle', '#4ade80', 'o'),
                                        ('goal', '#fb923c', 's')]:
        subset = filter_results(results, obs_mode='spectral',
                                reward_mode=reward_mode)
        if not subset:
            continue
        gammas = sorted(set(r['config']['gamma'] for r in subset))
        # Average across lr/hidden/std for each gamma
        for g in gammas:
            match = [r for r in subset if r['config']['gamma'] == g]
            wrs = [r['final_win_rate'] for r in match]
            ax2.scatter([g] * len(wrs), wrs, color=color, alpha=0.4,
                       s=30, marker=marker)
        avg_wrs = [np.mean([r['final_win_rate'] for r in subset
                            if r['config']['gamma'] == g])
                   for g in gammas]
        ax2.plot(gammas, avg_wrs, f'{marker}-', color=color,
                label=f'{reward_mode} (mean)', linewidth=2, markersize=8)
    ax2.set_xlabel('Gamma')
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Gamma Sensitivity (All Spectral Experiments)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Activation health indicators
    ax3 = fig.add_subplot(gs[1, 0])
    spectral_with_diag = [r for r in results
                          if r['config']['obs_mode'] == 'spectral'
                          and r['config']['hidden'] > 0
                          and len(r.get('diagnostics', [])) > 0]
    if spectral_with_diag:
        for r in spectral_with_diag[:10]:
            diags = r['diagnostics']
            if not diags:
                continue
            last = diags[-1]['agent']
            gamma = r['config']['gamma']
            wr = r['final_win_rate']
            dead = last.get('dead_frac', 0)
            out_std = last.get('output_std', 0)
            ax3.scatter(dead, out_std, c=wr, cmap='RdYlGn',
                       vmin=0, vmax=1, s=100, edgecolors='black',
                       linewidths=0.5)
            ax3.annotate(f'γ={gamma}', (dead, out_std), fontsize=7,
                        textcoords='offset points', xytext=(5, 5))
        sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                                    norm=plt.Normalize(0, 1))
        plt.colorbar(sm, ax=ax3, label='Win Rate')
    ax3.set_xlabel('Dead Neuron Fraction')
    ax3.set_ylabel('Output Std')
    ax3.set_title('Activation Health vs Performance')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Top-15 configs bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_results = sorted(results, key=lambda x: -x['final_win_rate'])[:15]
    names = [r['config']['name'] for r in sorted_results]
    wrs = [r['final_win_rate'] for r in sorted_results]
    colors = ['#38bdf8' if r['config']['obs_mode'] == 'spectral'
              else '#f87171' for r in sorted_results]
    bars = ax4.barh(range(len(names)), wrs, color=colors)
    ax4.set_yticks(range(len(names)))
    ax4.set_yticklabels(names, fontsize=7)
    ax4.set_xlabel('Win Rate')
    ax4.set_title('Top 15 Configurations')
    ax4.set_xlim(0, 1.05)
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    # Legend
    from matplotlib.patches import Patch
    ax4.legend([Patch(color='#38bdf8'), Patch(color='#f87171')],
               ['Spectral', 'Raw'], loc='lower right')

    fig.suptitle('Spectral Encoder Sweep — Summary Dashboard', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, '11_summary_dashboard.png'), dpi=150)
    plt.close(fig)
    print('  11_summary_dashboard.png')


# ---------------------------------------------------------------------------
# Plot 12: Observation distribution comparison
# ---------------------------------------------------------------------------

def plot_obs_distribution(results, out_dir):
    """Obs mean and std over training for spectral vs raw."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, key, title in [(axes[0], 'avg_obs_std', 'Obs Std'),
                            (axes[1], 'avg_action_mean', 'Action Mean')]:
        for obs_mode, color in [('spectral', '#38bdf8'), ('raw', '#f87171')]:
            subset = [r for r in results
                      if r['config']['obs_mode'] == obs_mode
                      and len(r.get('eval_curve', [])) > 0]
            for r in subset[:6]:
                curve = r['eval_curve']
                eps = [e['ep'] for e in curve]
                vals = [e.get(key, 0) for e in curve]
                gamma = r['config']['gamma']
                ax.plot(eps, vals, alpha=0.5, color=color,
                        label=f'{obs_mode} γ={gamma}')

        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7)

    fig.suptitle('Observation & Action Distributions', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '12_obs_distribution.png'), dpi=150)
    plt.close(fig)
    print('  12_obs_distribution.png')


# ---------------------------------------------------------------------------
# Plot 13: Frequency learning (lr_k)
# ---------------------------------------------------------------------------

def plot_freq_learning(results, out_dir):
    """Win rate for different lr_k values."""
    subset = [r for r in results if r['config'].get('lr_k', 0) > 0]
    baseline = [r for r in results
                if r['config']['obs_mode'] == 'spectral'
                and r['config'].get('lr_k', 0) == 0
                and r['config']['gamma'] == 0.5]

    if not subset:
        print('  13_freq_learning.png — SKIPPED (no lr_k experiments)')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, reward_mode in zip(axes, ['paddle', 'goal']):
        # Baseline
        bl = [r for r in baseline if r['config']['reward_mode'] == reward_mode]
        if bl:
            eps, wrs = get_eval_series(bl[0], 'win_rate')
            ax.plot(eps, wrs, 'k--', label='lr_k=0 (baseline)', linewidth=2)

        # lr_k experiments
        lrk_sub = [r for r in subset
                    if r['config']['reward_mode'] == reward_mode]
        for r in sorted(lrk_sub, key=lambda x: x['config']['lr_k']):
            lrk = r['config']['lr_k']
            eps, wrs = get_eval_series(r, 'win_rate')
            ax.plot(eps, wrs, label=f'lr_k={lrk}', linewidth=1.5)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'Frequency Learning — {reward_mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '13_freq_learning.png'), dpi=150)
    plt.close(fig)
    print('  13_freq_learning.png')


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_text_report(results, out_dir):
    """Write a structured text summary."""
    path = os.path.join(out_dir, 'sweep_report.txt')
    lines = []
    lines.append('=' * 80)
    lines.append('SPECTRAL ENCODER HYPERPARAMETER SWEEP REPORT')
    lines.append('=' * 80)
    lines.append(f'Total experiments: {len(results)}')
    total_time = sum(r['wall_time'] for r in results)
    lines.append(f'Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)')
    lines.append('')

    # Top 20 by win rate
    lines.append('--- TOP 20 CONFIGURATIONS ---')
    lines.append(f'{"#":<3} {"Name":<50} {"WR":>6} {"Peak":>6}')
    lines.append('-' * 70)
    for i, r in enumerate(sorted(results,
                                  key=lambda x: -x['final_win_rate'])[:20]):
        c = r['config']
        lines.append(f'{i+1:<3} {c["name"]:<50} '
                     f'{r["final_win_rate"]:>5.0%} '
                     f'{r["peak_win_rate"]:>5.0%}')

    lines.append('')

    # Spectral vs Raw
    lines.append('--- SPECTRAL vs RAW ---')
    for mode in ['spectral', 'raw']:
        sub = [r for r in results if r['config']['obs_mode'] == mode]
        if sub:
            wrs = [r['final_win_rate'] for r in sub]
            best = max(sub, key=lambda x: x['final_win_rate'])
            lines.append(f'{mode:>10}: n={len(sub)}, '
                         f'mean={np.mean(wrs):.0%}, '
                         f'median={np.median(wrs):.0%}, '
                         f'best={max(wrs):.0%} ({best["config"]["name"]})')

    lines.append('')

    # Gamma analysis
    lines.append('--- GAMMA ANALYSIS (spectral only) ---')
    spec = [r for r in results if r['config']['obs_mode'] == 'spectral']
    gammas = sorted(set(r['config']['gamma'] for r in spec))
    for g in gammas:
        sub = [r for r in spec if r['config']['gamma'] == g]
        wrs = [r['final_win_rate'] for r in sub]
        lines.append(f'  γ={g:<5}: n={len(sub):>3}, '
                     f'mean={np.mean(wrs):.0%}, '
                     f'std={np.std(wrs):.0%}, '
                     f'best={max(wrs):.0%}, worst={min(wrs):.0%}')

    lines.append('')

    # Activation health summary
    lines.append('--- ACTIVATION HEALTH (final checkpoint) ---')
    for r in sorted(results, key=lambda x: -x['final_win_rate'])[:10]:
        c = r['config']
        if c['obs_mode'] != 'spectral' or c['hidden'] == 0:
            continue
        diags = r.get('diagnostics', [])
        if not diags:
            continue
        last = diags[-1]
        ag = last['agent']
        lines.append(f'  {c["name"]}:')
        lines.append(f'    WR={r["final_win_rate"]:.0%}, '
                     f'W1_norm={ag.get("W1_norm", 0):.3f}, '
                     f'dead={ag.get("dead_neurons", 0)}/{ag.get("alive_neurons", 0)}, '
                     f'out_std={ag.get("output_std", 0):.4f}')

    report = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(report)
    print(f'  sweep_report.txt')
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_plots(results_path=None, out_dir=None):
    """Generate all plots from sweep results."""
    results = load_results(results_path)

    if out_dir is None:
        out_dir = os.path.join(training_dir, 'sweep_plots')
    ensure_dir(out_dir)

    print(f'Generating plots from {len(results)} experiments...')
    print(f'Output: {out_dir}/')

    plot_gamma_comparison(results, out_dir)
    plot_learning_curves_gamma(results, out_dir)
    plot_touch_curves(results, out_dir)
    plot_lr_gamma_heatmap(results, out_dir)
    plot_hidden_effect(results, out_dir)
    plot_std_effect(results, out_dir)
    plot_activation_evolution(results, out_dir)
    plot_spectral_diagnostics(results, out_dir)
    plot_per_channel_fmaps(results, out_dir)
    plot_gradient_flow(results, out_dir)
    plot_summary_dashboard(results, out_dir)
    plot_obs_distribution(results, out_dir)
    plot_freq_learning(results, out_dir)
    report = write_text_report(results, out_dir)

    print(f'\nDone! {len(os.listdir(out_dir))} files in {out_dir}/')
    print('\n' + report)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate sweep analysis plots')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots')
    args = parser.parse_args()
    generate_all_plots(args.input, args.output)
