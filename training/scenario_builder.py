#!/usr/bin/env python3
"""
MLBot Scenario Builder
======================
Interactive CLI for creating, browsing, and visualising adversarial training
scenario configs.

Every scenario has TWO cars — blue (primary skill) and orange (adversary skill)
— so both sides are always trained simultaneously and neither develops a bias.

Usage
-----
    python training/scenario_builder.py                        # interactive menu
    python training/scenario_builder.py --view   <yaml>       # visualise one scenario
    python training/scenario_builder.py --visualize <folder>  # visualise a folder
    python training/scenario_builder.py --play   <yaml> --side blue   # launch match
    python training/scenario_builder.py --save   (add to --view / --visualize / --play)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from scenarios.scenario_config import (
    AdversarialRewardConfig,
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

COMMON_TERMINAL_TYPES = [
    'goal_scored', 'save_made', 'ball_out_play', 'ball_cleared',
    'aerial_hit', 'pass_received', 'interception', 'timeout',
]
COMMON_STEP_TYPES = [
    'ball_toward_goal', 'ball_from_goal', 'car_near_ball',
    'ball_toward_teammate',
]


# ── terminal colours ──────────────────────────────────────────────────────────

def _c(t: str, code: str) -> str: return f'\033[{code}m{t}\033[0m'
def _bold(t):   return _c(t, '1')
def _cyan(t):   return _c(t, '36')
def _yellow(t): return _c(t, '33')
def _green(t):  return _c(t, '32')
def _red(t):    return _c(t, '31')
def _blue(t):   return _c(t, '34;1')
def _orange(t): return _c(t, '33;1')
def _dim(t):    return _c(t, '2')


def _section(title: str) -> None:
    print(f'\n{_dim("─" * 52)}')
    print(f'  {_bold(_cyan(title))}')
    print(_dim('─' * 52))


def _prompt(label: str, default: str = '') -> str:
    hint = f' {_dim(f"[{default}]")}' if default else ''
    try:
        raw = input(f'  {label}{hint}: ').strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
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
    print(f'\n  {_yellow("Yaw  (facing direction)")}')
    print(f'  {_dim("r=random   f=fixed angle   ra=angle range")}')
    choice = _prompt('  Mode', 'r').lower()
    if choice == 'f':
        a = _prompt_float('    radians  (0=east  π/2≈1.57=north)', round(math.pi / 2, 4))
        return RangeOrFixed(fixed=a)
    if choice == 'ra':
        lo = _prompt_float('    min (radians)', round(-math.pi, 4))
        hi = _prompt_float('    max (radians)', round(math.pi, 4))
        return RangeOrFixed(min_val=lo, max_val=hi)
    return RangeOrFixed(random=True)


# ── skill discovery ───────────────────────────────────────────────────────────

def discover_skills() -> dict[str, int]:
    if not CONFIGS_DIR.exists():
        return {}
    return {
        d.name: len(list(d.rglob('*.yaml')))
        for d in sorted(CONFIGS_DIR.iterdir())
        if d.is_dir() and list(d.rglob('*.yaml'))
    }


def _pick_skill(prompt_label: str = 'Skill') -> str:
    known = discover_skills()
    print()
    if known:
        print(f'  {_bold("Existing skills:")}')
        for i, (name, count) in enumerate(known.items(), 1):
            print(f'    {_yellow(str(i))}.  {name}  {_dim(f"({count} scenario(s))")}')
        print()
    print(f'  Enter a number to pick, or type a {_bold("new skill name")}.')
    raw = _prompt(f'\n  {prompt_label}', '1' if known else '').strip()

    if raw.isdigit():
        idx = int(raw) - 1
        names = list(known.keys())
        if 0 <= idx < len(names):
            return names[idx]

    safe = raw.lower().replace(' ', '_').replace('-', '_')
    return safe if safe else 'custom'


# ── reward wizard ─────────────────────────────────────────────────────────────

def _build_reward(side_label: str) -> RewardConfig:
    print(f'\n  {_dim("Common terminal types:")}  {", ".join(COMMON_TERMINAL_TYPES)}')
    terminal: List[RewardEvent] = []
    while True:
        etype = _prompt(f'  {side_label} terminal event  (blank to finish)', '')
        if not etype:
            break
        if etype == 'timeout':
            secs  = _prompt_float('    timeout seconds', 8.0)
            value = _prompt_float('    reward value', -1.0)
            terminal.append(RewardEvent(type='timeout', seconds=secs, value=value))
        else:
            value = _prompt_float('    reward value', 1.0)
            terminal.append(RewardEvent(type=etype, value=value))
        print(_green('    ✓ added'))

    print(f'\n  {_dim("Common step types:")}  {", ".join(COMMON_STEP_TYPES)}')
    step: List[RewardEvent] = []
    while True:
        etype = _prompt(f'  {side_label} step event  (blank to finish)', '')
        if not etype:
            break
        weight = _prompt_float('    weight', 0.01)
        step.append(RewardEvent(type=etype, weight=weight))
        print(_green('    ✓ added'))

    return RewardConfig(terminal=terminal, step=step)


# ── car wizard ────────────────────────────────────────────────────────────────

def _build_car(side_label: str, color_fn, default_y: float) -> tuple[str, CarConfig]:
    """Prompt for one car config (skill + position + yaw + boost).  Returns (skill, CarConfig)."""
    skill = _pick_skill(f'{side_label} skill')
    print(f'\n  {side_label} skill: {color_fn(_bold(skill))}')

    car_x   = _prompt_range_or_fixed(f'{side_label} X  (–4096 to 4096)',   0.0)
    car_y   = _prompt_range_or_fixed(f'{side_label} Y  (–5120 to 5120)',   default_y)
    car_z   = _prompt_range_or_fixed(f'{side_label} Z  (ground = 0)',       0.0)
    car_yaw = _prompt_yaw()
    car_boost = _prompt_range_or_fixed(f'{side_label} boost (0–100)',       33.0)

    return skill, CarConfig(
        skill=skill,
        location=Vec3Config(x=car_x, y=car_y, z=car_z),
        yaw=car_yaw,
        boost=car_boost,
    )


# ── create ────────────────────────────────────────────────────────────────────

def create_scenario() -> Optional[ScenarioConfig]:
    _section('Create New Adversarial Scenario')

    name        = _prompt('  Scenario name', 'my_scenario')
    description = _prompt('  Short description (optional)', '')

    _section('Ball — Initial Position')
    ball_x = _prompt_range_or_fixed('Ball X  (–4096 to 4096)',   0.0)
    ball_y = _prompt_range_or_fixed('Ball Y  (–5120 to 5120)',   3000.0)
    ball_z = _prompt_range_or_fixed('Ball Z  (ground ≈ 92)',     100.0)

    _section('Ball — Initial Velocity')
    if _prompt('Set initial ball velocity?  (y / n)', 'n').lower() == 'y':
        vel_x = _prompt_range_or_fixed('Velocity X (units/s)', 0.0)
        vel_y = _prompt_range_or_fixed('Velocity Y (units/s)', 0.0)
        vel_z = _prompt_range_or_fixed('Velocity Z (units/s)', 0.0)
    else:
        vel_x = vel_y = vel_z = RangeOrFixed(fixed=0.0)

    _section(f'Blue Car  {_blue("(primary skill)")}')
    blue_skill, blue_car = _build_car('Blue', _blue, 1500.0)

    _section(f'Orange Car  {_orange("(adversary skill)")}')
    orange_skill, orange_car = _build_car('Orange', _orange, -1500.0)

    _section(f'Blue Reward  {_blue(f"({blue_skill})")}')
    blue_reward = _build_reward(_blue('blue'))

    _section(f'Orange Reward  {_orange(f"({orange_skill})")}')
    orange_reward = _build_reward(_orange('orange'))

    _section('Training')
    max_eps = _prompt_int('  Max training episodes', 10000)

    config = ScenarioConfig(
        name=name,
        description=description,
        initial_state=InitialStateConfig(
            ball=BallConfig(
                location=Vec3Config(x=ball_x, y=ball_y, z=ball_z),
                velocity=Vec3Config(x=vel_x,  y=vel_y,  z=vel_z),
            ),
            blue=blue_car,
            orange=orange_car,
        ),
        reward=AdversarialRewardConfig(blue=blue_reward, orange=orange_reward),
        training=TrainingConfig(max_episodes=max_eps, save_every=500, model_path='models/'),
    )

    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    save_path = CONFIGS_DIR / blue_skill / f'{safe_name}.yaml'
    config.save_yaml(save_path)
    print(_green(f'\n  ✓  Saved → {save_path}'))
    return config


# ── list / pick ───────────────────────────────────────────────────────────────

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
        # show both skills if the YAML has them
        try:
            cfg = ScenarioConfig.from_yaml(p)
            skill_pair = f'{cfg.initial_state.blue.skill} ↔ {cfg.initial_state.orange.skill}'
        except Exception:
            skill_pair = ''
        print(f'    {_yellow(str(i + 1).rjust(2))}.  {p.stem}  {_dim(skill_pair)}')
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
        sp = str(PREVIEWS_DIR / f'{folder_path.name}_all.png')
    visualize_all(configs, cols=min(3, len(configs)), show=True, save_path=sp)
    if sp:
        print(_green(f'  ✓  Grid saved → {sp}'))


# ── human play ────────────────────────────────────────────────────────────────

def _play_scenario(config: ScenarioConfig, side: str) -> None:
    """Launch a human vs bot match for this scenario."""
    try:
        import human_play
        human_play.launch(config, side)
    except ImportError:
        print(_red('  ✗  human_play.py not found.'))
    except Exception as e:
        print(_red(f'  ✗  Error launching match: {e}'))


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
  {_yellow("6")}.  Play scenario against bot  {_dim("(validate + record expert data)")}
  {_yellow("q")}.  Quit""")

        choice = _prompt('\n  Choice', '1').lower()

        if choice == '1':
            config = create_scenario()
            if config:
                if _prompt('\n  Visualise?  (y / n)', 'y').lower() == 'y':
                    _vis_single(config, save=_prompt('  Save PNG?  (y / n)', 'n').lower() == 'y')

        elif choice == '2':
            paths = list_scenarios()
            config = _pick_scenario(paths)
            if config:
                _vis_single(config, save=_prompt('  Save PNG?  (y / n)', 'n').lower() == 'y')

        elif choice == '3':
            list_scenarios()

        elif choice == '4':
            known = discover_skills()
            if not known:
                print(_dim('  No skills found yet.')); continue
            print()
            for i, (name, count) in enumerate(known.items(), 1):
                print(f'  {_yellow(str(i))}.  {name}  {_dim(f"({count})")}')
            raw = _prompt('\n  Skill number', '1')
            names = list(known.keys())
            try:
                skill = names[int(raw) - 1]
            except (ValueError, IndexError):
                print(_red('  ✗  Invalid choice.')); continue
            _vis_folder(CONFIGS_DIR / skill, save=_prompt('  Save PNG?  (y / n)', 'n').lower() == 'y')

        elif choice == '5':
            _vis_folder(CONFIGS_DIR, save=_prompt('  Save PNG?  (y / n)', 'n').lower() == 'y')

        elif choice == '6':
            paths = list_scenarios()
            config = _pick_scenario(paths)
            if config:
                blue_skill   = config.initial_state.blue.skill   or 'blue'
                orange_skill = config.initial_state.orange.skill or 'orange'
                print(f'\n  {_blue("b")}.  Play as blue    ({blue_skill})')
                print(f'  {_orange("o")}.  Play as orange  ({orange_skill})')
                side = _prompt('  Side', 'b').lower()
                side = 'blue' if side == 'b' else 'orange'
                _play_scenario(config, side)

        elif choice == 'q':
            print(_cyan('\n  Goodbye!\n')); break

        else:
            print(_red('  ✗  Unknown option.'))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='MLBot Scenario Builder')
    parser.add_argument('--view',      metavar='YAML',   help='Visualise a single scenario and exit.')
    parser.add_argument('--visualize', metavar='FOLDER', help='Visualise all scenarios in a folder and exit.')
    parser.add_argument('--play',      metavar='YAML',   help='Launch a human vs bot match and exit.')
    parser.add_argument('--side',      choices=['blue', 'orange'], default='blue',
                        help='Which side to play as (used with --play).')
    parser.add_argument('--save',      action='store_true', help='Save rendered figures as PNG.')
    args = parser.parse_args()

    if args.view:
        from scenario_visualizer import visualize_scenario
        cfg = ScenarioConfig.from_yaml(args.view)
        sp = None
        if args.save:
            PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
            sp = str(PREVIEWS_DIR / f'{cfg.name.lower().replace(" ", "_")}.png')
        visualize_scenario(cfg, show=True, save_path=sp)
        if sp: print(_green(f'Saved → {sp}'))
        return

    if args.visualize:
        _vis_folder(Path(args.visualize), save=args.save)
        return

    if args.play:
        cfg = ScenarioConfig.from_yaml(args.play)
        _play_scenario(cfg, args.side)
        return

    _menu()


if __name__ == '__main__':
    main()
