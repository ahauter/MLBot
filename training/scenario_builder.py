#!/usr/bin/env python3
"""
MLBot Scenario Builder
======================
Interactive CLI for creating, browsing, and visualising training scenario configs.

Skills are discovered dynamically from the configs/ directory — no hardcoded list.
To add a new skill, just create a new subfolder and drop YAML files into it.

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

CONFIGS_DIR  = Path(__file__).parent / 'scenarios' / 'configs'
PREVIEWS_DIR = Path(__file__).parent / 'scenarios' / 'previews'

# Common reward event types shown as hints in the wizard
COMMON_TERMINAL_TYPES = [
    'goal_scored', 'ball_out_play', 'ball_cleared',
    'aerial_hit', 'pass_received', 'timeout',
]
COMMON_STEP_TYPES = [
    'ball_toward_goal', 'ball_from_goal', 'car_near_ball',
    'ball_toward_teammate',
]


# ── terminal colours ──────────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
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
    print(f'\n  {_yellow(label)}')
    choice = _prompt('  Fixed or range?  (f / r)', 'f').lower()
    if choice == 'r':
        lo = _prompt_float('    min', round(default_val - 500, 1))
        hi = _prompt_float('    max', round(default_val + 500, 1))
        return RangeOrFixed(min_val=lo, max_val=hi)
    return RangeOrFixed(fixed=_prompt_float('    value', default_val))


def _prompt_yaw() -> RangeOrFixed:
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


# ── skill discovery ───────────────────────────────────────────────────────────

def discover_skills() -> dict[str, int]:
    """
    Scan configs/ subdirectories and return {skill_name: scenario_count}.
    Sorted alphabetically.
    """
    if not CONFIGS_DIR.exists():
        return {}
    skills: dict[str, int] = {}
    for d in sorted(CONFIGS_DIR.iterdir()):
        if d.is_dir():
            count = len(list(d.rglob('*.yaml')))
            if count > 0:
                skills[d.name] = count
    return skills


def _pick_skill() -> str:
    """
    Show existing skills discovered from configs/, let user pick one or type a new name.
    Returns the skill name string.
    """
    known = discover_skills()
    print()
    if known:
        print(f'  {_bold("Existing skills:")}')
        for i, (name, count) in enumerate(known.items(), 1):
            print(f'    {_yellow(str(i))}.  {name}  {_dim(f"({count} scenario(s))")}')
        print()
    print(f'  Enter a number to pick an existing skill,')
    print(f'  or type a {_bold("new skill name")} to create one.')
    raw = _prompt('\n  Skill', '1' if known else '').strip()

    # numeric selection from existing list
    if raw.isdigit():
        idx = int(raw) - 1
        names = list(known.keys())
        if 0 <= idx < len(names):
            return names[idx]
        print(_red(f'  ✗  No skill #{raw}. Treating as new skill name.'))

    # free-text name
    safe = raw.lower().replace(' ', '_').replace('-', '_')
    if not safe:
        print(_red('  ✗  Empty name; defaulting to "custom".'))
        return 'custom'
    return safe


# ── reward wizard ─────────────────────────────────────────────────────────────

def _build_reward_events(kind: str) -> List[RewardEvent]:
    """
    Interactively build a list of reward events of the given kind ('terminal' or 'step').
    """
    hints = COMMON_TERMINAL_TYPES if kind == 'terminal' else COMMON_STEP_TYPES
    print(f'\n  {_dim("Common " + kind + " types:")}  {", ".join(hints)}')
    events: List[RewardEvent] = []
    while True:
        etype = _prompt(f'  {kind.capitalize()} event type  (blank to finish)', '')
        if not etype:
            break
        if kind == 'terminal':
            if etype == 'timeout':
                secs  = _prompt_float('    timeout seconds', 8.0)
                value = _prompt_float('    reward value',   -1.0)
                events.append(RewardEvent(type='timeout', seconds=secs, value=value))
            else:
                value = _prompt_float('    reward value', 1.0 if 'score' in etype or 'hit' in etype or 'received' in etype or 'cleared' in etype else -0.5)
                events.append(RewardEvent(type=etype, value=value))
        else:
            weight = _prompt_float('    weight (step multiplier)', 0.01)
            events.append(RewardEvent(type=etype, weight=weight))
        print(_green(f'    ✓  added'))
    return events


# ── create ────────────────────────────────────────────────────────────────────

def create_scenario() -> Optional[ScenarioConfig]:
    _section('Create New Scenario')

    skill = _pick_skill()
    print(f'\n  Skill: {_bold(_cyan(skill))}')

    name        = _prompt('  Scenario name', f'my_{skill}_scenario')
    description = _prompt('  Short description (optional)', '')

    # ── ball ──
    _section('Ball — Initial Position')
    ball_x = _prompt_range_or_fixed('Ball X  (field width: –4096 to 4096)',  0.0)
    ball_y = _prompt_range_or_fixed('Ball Y  (field length: –5120 to 5120)', 3000.0)
    ball_z = _prompt_range_or_fixed('Ball Z  (height — ground ≈ 92, wall mid ≈ 1022)', 100.0)

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
    car_z   = _prompt_range_or_fixed('Car Z   (ground = 0)',       0.0)
    car_yaw = _prompt_yaw()
    car_boost = _prompt_range_or_fixed('Boost amount (0–100)',     33.0)

    # ── reward ──
    _section('Reward Events')
    print(f'  Define terminal and per-step reward events for {_yellow(skill)}.')
    print(f'  {_dim("Leave blank to finish each section.")}')

    print(f'\n  {_bold("Terminal events")} (episode ends):')
    terminal = _build_reward_events('terminal')

    print(f'\n  {_bold("Step events")} (every tick):')
    step = _build_reward_events('step')

    _section('Training')
    max_eps = _prompt_int('  Max training episodes', 10000)

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
                location=Vec3Config(x=car_x, y=car_y, z=car_z),
                yaw=car_yaw,
                boost=car_boost,
            ),
        ),
        reward=RewardConfig(terminal=terminal, step=step),
        training=TrainingConfig(
            max_episodes=max_eps,
            save_every=500,
            model_path=f'models/skill_{skill}',
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
        print(f'    {_yellow(str(i + 1).rjust(2))}.  {p.stem}')
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
        known = discover_skills()
        skill_summary = '  '.join(f'{n}({c})' for n, c in known.items()) if known else 'none yet'
        print(f'\n  {_dim("skills: " + skill_summary)}')
        print(f"""
  {_yellow("1")}.  Create new scenario
  {_yellow("2")}.  View & visualise a scenario
  {_yellow("3")}.  List all saved scenarios
  {_yellow("4")}.  Visualise all scenarios for a skill
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
            known = discover_skills()
            if not known:
                print(_dim('  No skills found yet.'))
                continue
            print()
            for i, (name, count) in enumerate(known.items(), 1):
                print(f'  {_yellow(str(i))}.  {name}  {_dim(f"({count} scenario(s))")}')
            raw = _prompt('\n  Skill number', '1')
            names = list(known.keys())
            try:
                idx = int(raw) - 1
                if not 0 <= idx < len(names):
                    raise ValueError
                skill = names[idx]
            except ValueError:
                print(_red('  ✗  Invalid choice.'))
                continue
            save = _prompt('  Save PNG?  (y / n)', 'n').lower() == 'y'
            _vis_folder(CONFIGS_DIR / skill, save=save)

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
