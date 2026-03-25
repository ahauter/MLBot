"""
Rocket League field visualiser for training scenarios.

Each standalone figure has two panels:
  Top   — top-down field view (X-Y plane): ball & car spawn positions / ranges,
          velocity arrow (ball), yaw arrow (car), boost pads, goals.
  Bottom — elevation cross-section (X-Z plane): side-wall profile showing
           how high the ball / car spawn off the ground.  Useful for wall-ball
           and aerial scenarios.  The elevation strip is always shown so even
           "Z=92" ground scenarios confirm that ball is on the floor.

In grid mode (when an existing Axes is passed) the elevation panel is skipped
and a small "z=NNN" suffix is added to ball/car labels instead.

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
CEILING_Z      = 2044   # arena height (floor → ceiling)

# Approximate big-boost pad positions
BIG_BOOSTS = [
    (-3072, -4096), ( 3072, -4096),
    (-3072,  4096), ( 3072,  4096),
    (-3584,     0), ( 3584,     0),
]

# ── colour palette ────────────────────────────────────────────────────────────
BG_COLOR         = '#12121f'
FIELD_COLOR      = '#2d5a27'
LINE_COLOR       = 'white'
BALL_COLOR       = '#38bdf8'
BLUE_CAR_COLOR   = '#60a5fa'   # RL blue team
ORANGE_CAR_COLOR = '#fb923c'   # RL orange team

SKILL_COLORS = {
    'shooting':     '#ef4444',
    'saving':       '#60a5fa',
    'defending':    '#3b82f6',
    'attacking':    '#f97316',
    'passing':      '#4ade80',
    'intercepting': '#fb923c',
    'aerial':       '#c084fc',
    'dribbling':    '#fbbf24',
    'challenging':  '#f97316',
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


def _z_label(z_cfg: RangeOrFixed) -> str:
    """Return a z-annotation string for grid mode; empty string when near ground."""
    z = z_cfg.center()
    if z_cfg.is_range():
        return f'\nz {int(z_cfg.min_val)}–{int(z_cfg.max_val)}'
    return f'\nz={int(z)}' if z > 200 else ''


def _draw_spawn(
    ax:           plt.Axes,
    x_cfg:        RangeOrFixed,
    y_cfg:        RangeOrFixed,
    label:        str,
    color:        str,
    marker:       str = 'o',
    marker_size:  int = 10,
    z:            int = 5,
    label_suffix: str = '',
) -> tuple[float, float]:
    """
    Draw a point or shaded range rectangle, return the centre (cx, cy)
    used as the anchor for velocity/yaw arrows.

    label_suffix is appended to the annotation text — used in grid mode to
    show Z height when there is no elevation panel.
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
    ax.annotate(label + label_suffix, (cx, cy),
                textcoords='offset points', xytext=(9, 6),
                color=color, fontsize=9, fontweight='bold', zorder=z + 3)
    return cx, cy


def _draw_elevation(
    ax:         plt.Axes,
    ball_loc:   'Vec3Config',
    ball_vel:   'Vec3Config',
    blue_loc:   'Vec3Config',
    orange_loc: 'Vec3Config',
) -> None:
    """
    Draw a compact X-Z cross-section ("front-on" view of the side wall).

    Horizontal axis = X (−FIELD_X … +FIELD_X)
    Vertical axis   = Z (0 … CEILING_Z)
    """
    ax.set_facecolor(BG_COLOR)

    # ── arena outline ──
    # Floor
    ax.axhline(0, color=LINE_COLOR, linewidth=1.5, alpha=0.6, zorder=1)
    # Ceiling (dashed)
    ax.axhline(CEILING_Z, color=LINE_COLOR, linewidth=1, linestyle='--', alpha=0.4, zorder=1)
    ax.text(FIELD_X + 100, CEILING_Z, f'ceiling\n{CEILING_Z}',
            va='center', ha='left', color='#888', fontsize=6, zorder=2)
    # Side walls
    ax.axvline(-FIELD_X, color=LINE_COLOR, linewidth=1.5, alpha=0.5, zorder=1)
    ax.axvline( FIELD_X, color=LINE_COLOR, linewidth=1.5, alpha=0.5, zorder=1)
    # Light fill for the arena interior
    ax.add_patch(patches.Rectangle(
        (-FIELD_X, 0), 2 * FIELD_X, CEILING_Z,
        facecolor=FIELD_COLOR, alpha=0.25, edgecolor='none', zorder=0,
    ))

    # ── ball ──
    bx, bz = _draw_spawn(
        ax, ball_loc.x, ball_loc.z,
        'ball', BALL_COLOR, marker='o', marker_size=9, z=5,
    )
    # Z-velocity arrow (vertical component)
    vz = ball_vel.z.center()
    if abs(vz) > 1:
        scale = 0.4
        ax.annotate(
            '', xy=(bx, bz + vz * scale), xytext=(bx, bz),
            arrowprops=dict(arrowstyle='->', color=BALL_COLOR, lw=1.8),
            zorder=6,
        )

    # ── blue car ──
    _draw_spawn(ax, blue_loc.x,   blue_loc.z,   'blue',   BLUE_CAR_COLOR,   marker='v', marker_size=9, z=5)
    # ── orange car ──
    _draw_spawn(ax, orange_loc.x, orange_loc.z, 'orange', ORANGE_CAR_COLOR, marker='^', marker_size=9, z=5)

    # ── axis style ──
    ax.set_xlim(-FIELD_X - 700, FIELD_X + 700)
    ax.set_ylim(-80, CEILING_Z + 300)
    ax.set_aspect('equal')
    ax.set_xlabel('X', color='#666', fontsize=7)
    ax.set_ylabel('Z  (height)', color='#666', fontsize=7)
    ax.tick_params(colors='#555', labelsize=6)
    ax.set_title('elevation  (X-Z)', color='#888', fontsize=8, pad=3)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')


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
    def _fmt(side: str, rc) -> list:
        out = [f'{side}:']
        for ev in rc.terminal:
            if ev.type == 'timeout':
                out.append(f'  ✓ timeout {ev.seconds}s→{ev.value:+.1f}')
            else:
                out.append(f'  ✓ {ev.type}: {ev.value:+.1f}')
        for ev in rc.step:
            out.append(f'  ~ {ev.type} ×{ev.weight}')
        return out

    lines = _fmt('blue', config.reward.blue) + [''] + _fmt('orange', config.reward.orange)
    return '\n'.join(lines)


def _boost_str(b) -> str:
    return f'{int(b.min_val)}–{int(b.max_val)}' if b.is_range() else str(int(b.fixed))


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
        fig, (ax, ax_elev) = plt.subplots(
            2, 1, figsize=(6, 10.5),
            gridspec_kw={'height_ratios': [5, 1.5]},
        )
        fig.patch.set_facecolor(BG_COLOR)
    else:
        ax_elev = None
        fig = ax.get_figure()

    ax.set_facecolor(BG_COLOR)
    _draw_field(ax)

    state = config.initial_state
    blue_skill_color   = SKILL_COLORS.get(state.blue.skill,   BLUE_CAR_COLOR)
    orange_skill_color = SKILL_COLORS.get(state.orange.skill, ORANGE_CAR_COLOR)

    # In grid mode append z height to labels (no elevation panel available)
    ball_z_sfx   = '' if standalone else _z_label(state.ball.location.z)
    blue_z_sfx   = '' if standalone else _z_label(state.blue.location.z)
    orange_z_sfx = '' if standalone else _z_label(state.orange.location.z)

    # ── ball ──
    ball_cx, ball_cy = _draw_spawn(
        ax,
        state.ball.location.x, state.ball.location.y,
        'ball', BALL_COLOR, marker='o', marker_size=11, z=5,
        label_suffix=ball_z_sfx,
    )
    _draw_velocity_arrow(
        ax, ball_cx, ball_cy,
        state.ball.velocity.x.center(),
        state.ball.velocity.y.center(),
        BALL_COLOR,
    )

    # ── blue car ──
    blue_cx, blue_cy = _draw_spawn(
        ax,
        state.blue.location.x, state.blue.location.y,
        'blue', BLUE_CAR_COLOR, marker='v', marker_size=11, z=5,
        label_suffix=blue_z_sfx,
    )
    _draw_yaw_arrow(ax, blue_cx, blue_cy, state.blue.yaw, BLUE_CAR_COLOR)

    # ── orange car ──
    orange_cx, orange_cy = _draw_spawn(
        ax,
        state.orange.location.x, state.orange.location.y,
        'orange', ORANGE_CAR_COLOR, marker='^', marker_size=11, z=5,
        label_suffix=orange_z_sfx,
    )
    _draw_yaw_arrow(ax, orange_cx, orange_cy, state.orange.yaw, ORANGE_CAR_COLOR)

    # ── axis limits ──
    ax.set_xlim(-FIELD_X - 700, FIELD_X + 700)
    ax.set_ylim(-FIELD_Y - GOAL_DEPTH - 500, FIELD_Y + GOAL_DEPTH + 500)
    ax.set_aspect('equal')

    # ── title ──
    ax.set_title(config.name, color='white', fontsize=11, fontweight='bold', pad=6)
    # Two coloured skill labels side-by-side
    ax.text(0.38, 1.005, f'blue: {state.blue.skill}',
            transform=ax.transAxes, ha='right', va='bottom',
            color=blue_skill_color, fontsize=8)
    ax.text(0.50, 1.005, '↔',
            transform=ax.transAxes, ha='center', va='bottom',
            color='#666', fontsize=8)
    ax.text(0.62, 1.005, f'orange: {state.orange.skill}',
            transform=ax.transAxes, ha='left', va='bottom',
            color=orange_skill_color, fontsize=8)

    # ── reward legend (bottom-left) ──
    ax.text(
        0.02, 0.02, _reward_summary(config),
        transform=ax.transAxes, va='bottom', ha='left',
        color='white', fontsize=5.5, alpha=0.85, family='monospace',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#00000077', edgecolor='none'),
        zorder=10,
    )

    # ── boost labels (bottom-right) ──
    ax.text(
        0.98, 0.02,
        f'boost  blue {_boost_str(state.blue.boost)}  orange {_boost_str(state.orange.boost)}',
        transform=ax.transAxes, va='bottom', ha='right',
        color='#aaa', fontsize=6.5, alpha=0.9,
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

    # ── elevation strip (standalone only) ──
    if ax_elev is not None:
        _draw_elevation(ax_elev, state.ball.location, state.ball.velocity,
                        state.blue.location, state.orange.location)

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
