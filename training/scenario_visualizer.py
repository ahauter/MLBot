"""
Rocket League field visualiser for training scenarios.

Renders a top-down 2-D view of the field showing:
  - Ball spawn position (point) or spawn region (shaded rectangle)
  - Ball initial velocity (arrow, if non-zero)
  - Car spawn position or region
  - Car facing direction / yaw (arrow, if not fully random)
  - Reward summary text
  - Boost pad locations (approximate)

Usage:
    from scenario_visualizer import visualize_scenario, visualize_all

    cfg = ScenarioConfig.from_yaml('scenarios/configs/shooting/power_shot_center.yaml')
    visualize_scenario(cfg)                          # interactive window
    visualize_scenario(cfg, save_path='out.png')     # save to file

    visualize_all(configs)                           # grid of thumbnails
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scenarios.scenario_config import RangeOrFixed, ScenarioConfig


# ── Rocket League field constants (Unreal / game units) ───────────────────────
FIELD_X        = 4096   # half-width
FIELD_Y        = 5120   # half-length
GOAL_HW        = 893    # goal half-width
GOAL_DEPTH     = 880    # depth of goal box
CENTER_RADIUS  = 1024   # centre-circle radius

# Approximate big-boost pad positions
BIG_BOOSTS = [
    (-3072, -4096), ( 3072, -4096),
    (-3072,  4096), ( 3072,  4096),
    (-3584,     0), ( 3584,     0),
]

# ── colour palette ────────────────────────────────────────────────────────────
BG_COLOR    = '#12121f'
FIELD_COLOR = '#2d5a27'
LINE_COLOR  = 'white'
BALL_COLOR  = '#38bdf8'
CAR_COLOR   = '#fb923c'

SKILL_COLORS = {
    'shooter':  '#ef4444',
    'defender': '#60a5fa',
    'passer':   '#4ade80',
    'aerial':   '#c084fc',
}


# ── internal drawing helpers ──────────────────────────────────────────────────

def _draw_field(ax: plt.Axes) -> None:
    """Draw grass, walls, goals, centre line, centre circle, boost pads."""
    # Grass rectangle
    ax.add_patch(patches.FancyBboxPatch(
        (-FIELD_X, -FIELD_Y), 2 * FIELD_X, 2 * FIELD_Y,
        boxstyle='square,pad=0',
        linewidth=2, edgecolor=LINE_COLOR, facecolor=FIELD_COLOR, zorder=0,
    ))

    # Subtle stripe pattern
    for y in range(-FIELD_Y, FIELD_Y, 1024):
        ax.add_patch(patches.Rectangle(
            (-FIELD_X, y), 2 * FIELD_X, 512,
            facecolor='#295224', edgecolor='none', alpha=0.4, zorder=0,
        ))

    # Blue goal (bottom, Y = -FIELD_Y)
    ax.add_patch(patches.Rectangle(
        (-GOAL_HW, -FIELD_Y - GOAL_DEPTH), 2 * GOAL_HW, GOAL_DEPTH,
        linewidth=1.5, edgecolor='#93c5fd', facecolor='#1e3a8a', alpha=0.75, zorder=1,
    ))
    ax.text(0, -FIELD_Y - GOAL_DEPTH / 2, 'BLUE\nGOAL',
            ha='center', va='center', color='white',
            fontsize=7, fontweight='bold', zorder=2)

    # Orange goal (top, Y = +FIELD_Y)
    ax.add_patch(patches.Rectangle(
        (-GOAL_HW, FIELD_Y), 2 * GOAL_HW, GOAL_DEPTH,
        linewidth=1.5, edgecolor='#fdba74', facecolor='#7c2d12', alpha=0.75, zorder=1,
    ))
    ax.text(0, FIELD_Y + GOAL_DEPTH / 2, 'ORANGE\nGOAL',
            ha='center', va='center', color='white',
            fontsize=7, fontweight='bold', zorder=2)

    # Centre line
    ax.plot([-FIELD_X, FIELD_X], [0, 0],
            color=LINE_COLOR, linewidth=1, alpha=0.35, zorder=1)

    # Centre circle
    ax.add_patch(plt.Circle(
        (0, 0), CENTER_RADIUS,
        color=LINE_COLOR, fill=False, linewidth=1, alpha=0.3, zorder=1,
    ))

    # Big boost pads
    for bx, by in BIG_BOOSTS:
        ax.plot(bx, by, 'o', color='#facc15', markersize=5, alpha=0.55, zorder=2)


def _draw_spawn(
    ax:          plt.Axes,
    x_cfg:       RangeOrFixed,
    y_cfg:       RangeOrFixed,
    label:       str,
    color:       str,
    marker:      str = 'o',
    marker_size: int = 10,
    z:           int = 5,
) -> tuple[float, float]:
    """
    Draw a point or shaded range rectangle, return the centre (cx, cy)
    used as the anchor for velocity/yaw arrows.
    """
    x_rng = x_cfg.is_range()
    y_rng = y_cfg.is_range()

    if x_rng or y_rng:
        x0 = x_cfg.min_val if x_rng else x_cfg.fixed
        x1 = x_cfg.max_val if x_rng else x_cfg.fixed
        y0 = y_cfg.min_val if y_rng else y_cfg.fixed
        y1 = y_cfg.max_val if y_rng else y_cfg.fixed

        # Give degenerate (fixed) axis a tiny visible extent
        if not x_rng:
            x0, x1 = x0 - 60, x0 + 60
        if not y_rng:
            y0, y1 = y0 - 60, y0 + 60

        # Filled region
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=color, alpha=0.18, zorder=z,
        ))
        # Dashed border
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.5, edgecolor=color, facecolor='none',
            linestyle='--', zorder=z + 1,
        ))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    else:
        cx, cy = x_cfg.fixed, y_cfg.fixed

    ax.plot(cx, cy, marker, color=color, markersize=marker_size,
            markeredgecolor='white', markeredgewidth=0.8, zorder=z + 2)
    ax.annotate(label, (cx, cy),
                textcoords='offset points', xytext=(9, 6),
                color=color, fontsize=9, fontweight='bold', zorder=z + 3)
    return cx, cy


def _draw_velocity_arrow(
    ax: plt.Axes, cx: float, cy: float, vx: float, vy: float, color: str,
) -> None:
    """Arrow from the ball centre showing launch velocity (scaled for display)."""
    if abs(vx) < 1 and abs(vy) < 1:
        return
    scale = 0.6
    ax.annotate(
        '', xy=(cx + vx * scale, cy + vy * scale), xytext=(cx, cy),
        arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
        zorder=6,
    )


def _draw_yaw_arrow(
    ax: plt.Axes, cx: float, cy: float, yaw_cfg: RangeOrFixed, color: str,
) -> None:
    """Arrow showing car facing direction when yaw is not fully random."""
    if yaw_cfg.random:
        # Draw a small circular arc to indicate "any direction"
        theta = np.linspace(0, 2 * math.pi, 64)
        r = 320
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                color=color, linewidth=1, linestyle=':', alpha=0.5, zorder=6)
        return

    angle = yaw_cfg.center()
    length = 420
    dx, dy = math.cos(angle) * length, math.sin(angle) * length
    ax.annotate(
        '', xy=(cx + dx, cy + dy), xytext=(cx, cy),
        arrowprops=dict(arrowstyle='->', color=color, lw=2.5),
        zorder=6,
    )


def _reward_summary(config: ScenarioConfig) -> str:
    lines = ['Reward']
    for ev in config.reward.terminal:
        if ev.type == 'timeout':
            lines.append(f'  ✓ timeout {ev.seconds}s → {ev.value:+.1f}')
        else:
            lines.append(f'  ✓ {ev.type}: {ev.value:+.1f}')
    for ev in config.reward.step:
        lines.append(f'  ~ {ev.type} ×{ev.weight}')
    return '\n'.join(lines)


# ── public API ────────────────────────────────────────────────────────────────

def visualize_scenario(
    config:    ScenarioConfig,
    ax:        Optional[plt.Axes] = None,
    show:      bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render one scenario onto a top-down field view.

    Parameters
    ----------
    config    : the scenario to draw
    ax        : existing Axes to draw into (creates a new figure if None)
    show      : call plt.show() when done (only when ax is None)
    save_path : if given, save the figure to this path
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 9))
        fig.patch.set_facecolor(BG_COLOR)
    else:
        fig = ax.get_figure()

    ax.set_facecolor(BG_COLOR)
    _draw_field(ax)

    skill_color = SKILL_COLORS.get(config.skill, '#ffffff')
    state = config.initial_state

    # ── ball ──
    ball_cx, ball_cy = _draw_spawn(
        ax,
        state.ball.location.x, state.ball.location.y,
        'ball', BALL_COLOR, marker='o', marker_size=11, z=5,
    )
    _draw_velocity_arrow(
        ax, ball_cx, ball_cy,
        state.ball.velocity.x.center(),
        state.ball.velocity.y.center(),
        BALL_COLOR,
    )

    # ── car ──
    car_cx, car_cy = _draw_spawn(
        ax,
        state.car.location.x, state.car.location.y,
        'car', CAR_COLOR, marker='^', marker_size=11, z=5,
    )
    _draw_yaw_arrow(ax, car_cx, car_cy, state.car.yaw, CAR_COLOR)

    # ── axis limits ──
    ax.set_xlim(-FIELD_X - 700, FIELD_X + 700)
    ax.set_ylim(-FIELD_Y - GOAL_DEPTH - 500, FIELD_Y + GOAL_DEPTH + 500)
    ax.set_aspect('equal')

    # ── title ──
    ax.set_title(config.name, color='white', fontsize=11, fontweight='bold', pad=6)
    ax.text(
        0.5, 1.005, f'skill: {config.skill.upper()}',
        transform=ax.transAxes, ha='center', va='bottom',
        color=skill_color, fontsize=8,
    )

    # ── reward legend (bottom-left) ──
    ax.text(
        0.02, 0.02, _reward_summary(config),
        transform=ax.transAxes, va='bottom', ha='left',
        color='white', fontsize=6.5, alpha=0.85, family='monospace',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#00000077', edgecolor='none'),
        zorder=10,
    )

    # ── boost label (bottom-right) ──
    boost = state.car.boost
    boost_str = (f'boost {int(boost.min_val)}–{int(boost.max_val)}'
                 if boost.is_range() else f'boost {int(boost.fixed)}')
    ax.text(
        0.98, 0.02, boost_str,
        transform=ax.transAxes, va='bottom', ha='right',
        color=CAR_COLOR, fontsize=7, alpha=0.9,
    )

    # ── description (top, under title) ──
    if config.description:
        ax.text(
            0.5, 0.975, config.description,
            transform=ax.transAxes, ha='center', va='top',
            color='#aaaaaa', fontsize=6.5, style='italic',
            wrap=True,
        )

    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.tick_params(colors='#555', labelsize=6)

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)

    if show and standalone:
        plt.tight_layout()
        plt.show()

    return fig


def visualize_all(
    configs:   List[ScenarioConfig],
    cols:      int = 3,
    show:      bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a grid of scenario thumbnails.

    Parameters
    ----------
    configs   : list of ScenarioConfig objects
    cols      : number of columns in the grid
    show      : display the figure interactively
    save_path : optional path to save the composite image
    """
    n = len(configs)
    if n == 0:
        raise ValueError('No configs provided.')

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 8))
    fig.patch.set_facecolor(BG_COLOR)

    axes_flat: list = np.array(axes).flatten().tolist() if n > 1 else [axes]

    for i, cfg in enumerate(configs):
        visualize_scenario(cfg, ax=axes_flat[i], show=False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    skill_label = ', '.join(sorted({c.skill for c in configs}))
    fig.suptitle(
        f'Training Scenarios  ·  {skill_label}',
        color='white', fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight', facecolor=BG_COLOR)

    if show:
        plt.show()

    return fig
