#!/usr/bin/env python3
"""
MLBot Scenario Builder
======================
Interactive CLI for creating, browsing, and visualising training scenario configs.

Usage
-----
    # Launch the interactive menu
    python training/scenario_builder.py

    # Directly visualise a single YAML file
    python training/scenario_builder.py --view training/scenarios/configs/shooting/power_shot_center.yaml

    # Visualise every scenario in a folder (or sub-folders)
    python training/scenario_builder.py --visualize training/scenarios/configs/shooting/

    # Save all visualisations as PNGs
    python training/scenario_builder.py --visualize training/scenarios/configs/ --save
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

# Make `scenarios` importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from scenarios.scenario_config import (
    BallConfig,
    CarConfig,
    InitialStateConfig,
    RangeOrFixed,
    RewardConfig,
    RewardEvent,
    ScenarioConfig,
    TrainingConfig,
    Vec3Config,
)

CONFIGS_DIR = Path(__file__).parent / 'scenarios' / 'configs'
PREVIEWS_DIR = Path(__file__).parent / 'scenarios' / 'previews'

# ── skill catalogue ───────────────────────────────────────────────────────────

SKILLS = {
    '1': ('shooter',  'Score a goal'),
    '2': ('defender', 'Prevent / clear the ball'),
    '3': ('passer',   'Deliver ball to a teammate zone'),
    '4': ('aerial',   'Contact an airborne ball'),
}

# Default reward blueprints per skill (humans can override in YAML later)
DEFAULT_REWARDS: dict[str, dict] = {
    'shooter': {
        'terminal': [
            {'type': 'goal_scored',   'value':  1.0},
            {'type': 'ball_out_play', 'value': -0.5},
            {'type': 'timeout',       'seconds': 8.0,  'value': -1.0},
        ],
        'step': [{'type': 'ball_toward_goal', 'weight': 0.01}],
    },
    'defender': {
        'terminal': [
            {'type': 'ball_cleared',  'value':  1.0},
            {'type': 'goal_scored',   'value': -1.0},
            {'type': 'timeout',       'seconds': 10.0, 'value': -0.5},
        ],
        'step': [{'type': 'ball_from_goal', 'weight': 0.01}],
    },
    'passer': {
        'terminal': [
            {'type': 'pass_received', 'value':  1.0},
            {'type': 'ball_out_play', 'value': -0.5},
            {'type': 'timeout',       'seconds': 8.0,  'value': -1.0},
        ],
        'step': [{'type': 'ball_toward_teammate', 'weight': 0.01}],
    },
    'aerial': {
        'terminal': [
            {'type': 'aerial_hit',    'value':  1.0},
            {'type': 'ball_grounded', 'value': -0.5},
            {'type': 'timeout',       'seconds': 6.0,  'value': -1.0},
        ],
        'step': [{'type': 'car_near_ball', 'weight': 0.005}],
    },
}


# ── terminal colours ──────────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code."""
    return f'\033[{code}m{text}\033[0m'

def _bold(t: str)   -> str: return _c(t, '1')
def _cyan(t: str)   -> str: return _c(t, '36')
def _yellow(t: str) -> str: return _c(t, '33')
def _green(t: str)  -> str: return _c(t, '32')
def _red(t: str)    -> str: return _c(t, '31')
def _dim(t: str)    -> str: return _c(t, '2')


def _section(title: str) -> None:
    print(f'\n{_dim("─" * 52)}')
    print(f'  {_bold(_cyan(title))}')
    print(_dim('─' * 52))


def _prompt(label: str, default: str = '') -> str:
    hint = f' {_dim(f"[{default}]")}' if default else ''
    try:
        raw = input(f'  {label}{hint}: ').strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return raw if raw else default


def _prompt_float(label: str, default: float) -> float:
    while True:
        raw = _prompt(label, str(default))
        try:
            return float(raw)
        except ValueError:
            print(_red('  ✗  Please enter a number.'))


def _prompt_int(label: str, default: int) -> int:
    while True:
        raw = _prompt(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print(_red('  ✗  Please enter an integer.'))


def _prompt_range_or_fixed(label: str, default_val: float) -> RangeOrFixed:
    """
    Ask the user whether this dimension is a fixed value or a random range,
    then collect the appropriate numbers.
    """
    print(f'\n  {_yellow(label)}')
    choice = _prompt('  Fixed or range?  (f / r)', 'f').lower()
    if choice == 'r':
        lo = _prompt_float('    min', round(default_val - 500, 1))
        hi = _prompt_float('    max', round(default_val + 500, 1))
        return RangeOrFixed(min_val=lo, max_val=hi)
    return RangeOrFixed(fixed=_prompt_float('    value', default_val))


def _prompt_yaw() -> RangeOrFixed:
    """Prompt for car yaw: random, fixed angle, or uniform range."""
    print(f'\n  {_yellow("Car yaw  (facing direction)")}')
    print(f'  {_dim("r = random   f = fixed angle   ra = angle range")}')
    choice = _prompt('  Mode', 'r').lower()
    if choice == 'f':
        angle = _prompt_float(
            '    yaw in radians  (0=east  π/2≈1.57=north)', round(math.pi / 2, 4)
        )
        return RangeOrFixed(fixed=angle)
    if choice == 'ra':
        lo = _prompt_float('    min (radians)', round(-math.pi, 4))
        hi = _prompt_float('    max (radians)', round(math.pi, 4))
        return RangeOrFixed(min_val=lo, max_val=hi)
    return RangeOrFixed(random=True)


# ── create ────────────────────────────────────────────────────────────────────

def create_scenario() -> Optional[ScenarioConfig]:
    _section('Create New Scenario')

    # Skill
    print()
    for key, (name, desc) in SKILLS.items():
        print(f'  {_yellow(key)}.  {_bold(name.capitalize())} — {_dim(desc)}')
    skill_key = _prompt('\n  Skill type', '1')
    if skill_key not in SKILLS:
        print(_red('  ✗  Invalid choice.'))
        return None
    skill, _ = SKILLS[skill_key]

    name        = _prompt('  Scenario name', f'my_{skill}_scenario')
    description = _prompt('  Short description (optional)', '')

    # ── ball ──
    _section('Ball — Initial Position')
    ball_x = _prompt_range_or_fixed('Ball X  (field width: –4096 to 4096)',  0.0)
    ball_y = _prompt_range_or_fixed('Ball Y  (field length: –5120 to 5120)', 3000.0)
    ball_z = _prompt_range_or_fixed('Ball Z  (height, min ≈ 92)',           100.0)

    _section('Ball — Initial Velocity')
    set_vel = _prompt('Set initial ball velocity?  (y / n)', 'n').lower()
    if set_vel == 'y':
        vel_x = _prompt_range_or_fixed('Velocity X (game units/s)', 0.0)
        vel_y = _prompt_range_or_fixed('Velocity Y (game units/s)', 0.0)
        vel_z = _prompt_range_or_fixed('Velocity Z (game units/s)', 0.0)
    else:
        vel_x = vel_y = vel_z = RangeOrFixed(fixed=0.0)

    # ── car ──
    _section('Car — Initial Position & State')
    car_x   = _prompt_range_or_fixed('Car X   (–4096 to 4096)',   0.0)
    car_y   = _prompt_range_or_fixed('Car Y   (–5120 to 5120)',   1500.0)
    car_yaw = _prompt_yaw()
    car_boost = _prompt_range_or_fixed('Boost amount (0–100)',     33.0)

    # ── reward ──
    _section('Reward & Training')
    print(f'  Using default reward template for {_yellow(skill)}.')
    print(_dim('  (You can edit the YAML file afterwards to customise further.)'))
    timeout  = _prompt_float('  Episode timeout (seconds)', 8.0)
    max_eps  = _prompt_int('  Max training episodes',     10000)

    # Patch the timeout value into the default template
    terminal_events: list[RewardEvent] = []
    for ev_dict in DEFAULT_REWARDS[skill]['terminal']:
        ev = dict(ev_dict)
        if ev['type'] == 'timeout':
            ev['seconds'] = timeout
        terminal_events.append(RewardEvent.from_dict(ev))
    step_events = [
        RewardEvent.from_dict(e) for e in DEFAULT_REWARDS[skill]['step']
    ]

    # ── assemble ──
    config = ScenarioConfig(
        name=name,
        skill=skill,
        description=description,
        initial_state=InitialStateConfig(
            ball=BallConfig(
                location=Vec3Config(x=ball_x, y=ball_y, z=ball_z),
                velocity=Vec3Config(x=vel_x,  y=vel_y,  z=vel_z),
            ),
            car=CarConfig(
                location=Vec3Config(x=car_x, y=car_y, z=RangeOrFixed(fixed=0.0)),
                yaw=car_yaw,
                boost=car_boost,
            ),
        ),
        reward=RewardConfig(terminal=terminal_events, step=step_events),
        training=TrainingConfig(
            max_episodes=max_eps,
            save_every=500,
            model_path=f'models/{skill}.weights',
        ),
    )

    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    save_path = CONFIGS_DIR / skill / f'{safe_name}.yaml'
    config.save_yaml(save_path)
    print(_green(f'\n  ✓  Saved → {save_path}'))
    return config


# ── list ──────────────────────────────────────────────────────────────────────

def list_scenarios() -> List[Path]:
    paths = sorted(CONFIGS_DIR.rglob('*.yaml'))
    if not paths:
        print(_dim('  No scenarios found in ' + str(CONFIGS_DIR)))
        return []

    _section('Saved Scenarios')
    prev_folder = None
    for i, p in enumerate(paths):
        folder = p.parent.name
        if folder != prev_folder:
            print(f'\n  {_dim(folder + "/")}')
            prev_folder = folder
        rel = p.stem
        print(f'    {_yellow(str(i + 1).rjust(2))}.  {rel}')
    return paths


def _pick_scenario(paths: List[Path]) -> Optional[ScenarioConfig]:
    if not paths:
        return None
    raw = _prompt('\n  Select number', '1')
    try:
        idx = int(raw) - 1
        if not 0 <= idx < len(paths):
            raise ValueError
        return ScenarioConfig.from_yaml(paths[idx])
    except ValueError:
        print(_red('  ✗  Invalid selection.'))
        return None


# ── visualise ─────────────────────────────────────────────────────────────────

def _vis_single(config: ScenarioConfig, save: bool = False) -> None:
    from scenario_visualizer import visualize_scenario
    sp = None
    if save:
        PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
        safe = config.name.lower().replace(' ', '_')
        sp = str(PREVIEWS_DIR / f'{safe}.png')
    visualize_scenario(config, show=True, save_path=sp)
    if sp:
        print(_green(f'  ✓  Preview saved → {sp}'))


def _vis_folder(folder_path: Path, save: bool = False) -> None:
    from scenario_visualizer import visualize_all
    paths = sorted(folder_path.rglob('*.yaml'))
    if not paths:
        print(_red('  No YAML files found in that folder.'))
        return
    configs = [ScenarioConfig.from_yaml(p) for p in paths]
    print(f'  Rendering {len(configs)} scenario(s)…')
    sp = None
    if save:
        PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
        folder_label = folder_path.name
        sp = str(PREVIEWS_DIR / f'{folder_label}_all.png')
    visualize_all(configs, cols=min(3, len(configs)), show=True, save_path=sp)
    if sp:
        print(_green(f'  ✓  Grid saved → {sp}'))


# ── main menu ─────────────────────────────────────────────────────────────────

def _menu() -> None:
    print(_bold(_cyan('\n╔══════════════════════════════════╗')))
    print(_bold(_cyan(  '║    MLBot  Scenario  Builder      ║')))
    print(_bold(_cyan(  '╚══════════════════════════════════╝')))

    while True:
        print(f"""
  {_yellow("1")}.  Create new scenario
  {_yellow("2")}.  View & visualise a scenario
  {_yellow("3")}.  List all saved scenarios
  {_yellow("4")}.  Visualise all scenarios in a skill folder
  {_yellow("5")}.  Visualise ALL scenarios (grid)
  {_yellow("q")}.  Quit""")

        choice = _prompt('\n  Choice', '1').lower()

        if choice == '1':
            config = create_scenario()
            if config:
                ans = _prompt('\n  Visualise this scenario?  (y / n)', 'y').lower()
                if ans == 'y':
                    save = _prompt('  Save PNG?  (y / n)', 'n').lower() == 'y'
                    _vis_single(config, save=save)

        elif choice == '2':
            paths = list_scenarios()
            config = _pick_scenario(paths)
            if config:
                save = _prompt('  Save PNG?  (y / n)', 'n').lower() == 'y'
                _vis_single(config, save=save)

        elif choice == '3':
            list_scenarios()

        elif choice == '4':
            print()
            for key, (skill, _) in SKILLS.items():
                folder = CONFIGS_DIR / skill
                count = len(list(folder.rglob('*.yaml'))) if folder.exists() else 0
                print(f'  {_yellow(key)}.  {skill.capitalize()}  {_dim(f"({count} scenario(s))")}')
            key = _prompt('\n  Skill folder', '1')
            if key in SKILLS:
                skill, _ = SKILLS[key]
                save = _prompt('  Save PNG?  (y / n)', 'n').lower() == 'y'
                _vis_folder(CONFIGS_DIR / skill, save=save)
            else:
                print(_red('  ✗  Invalid choice.'))

        elif choice == '5':
            save = _prompt('  Save PNG?  (y / n)', 'n').lower() == 'y'
            _vis_folder(CONFIGS_DIR, save=save)

        elif choice == 'q':
            print(_cyan('\n  Goodbye!\n'))
            break

        else:
            print(_red('  ✗  Unknown option — try again.'))


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='MLBot Scenario Builder — create & visualise training scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--view', metavar='YAML',
        help='Visualise a single scenario YAML file and exit.',
    )
    parser.add_argument(
        '--visualize', metavar='FOLDER',
        help='Visualise all scenarios in a folder (recursive) and exit.',
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save rendered figures as PNG files (used with --view or --visualize).',
    )
    args = parser.parse_args()

    if args.view:
        from scenario_visualizer import visualize_scenario
        cfg = ScenarioConfig.from_yaml(args.view)
        sp = None
        if args.save:
            PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
            safe = cfg.name.lower().replace(' ', '_')
            sp = str(PREVIEWS_DIR / f'{safe}.png')
        visualize_scenario(cfg, show=True, save_path=sp)
        if sp:
            print(_green(f'Saved → {sp}'))
        return

    if args.visualize:
        _vis_folder(Path(args.visualize), save=args.save)
        return

    _menu()


if __name__ == '__main__':
    main()
