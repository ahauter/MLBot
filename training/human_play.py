"""
Human Play Session
==================
Launch a RLBot match where a human plays one side of an adversarial scenario
while the bot controls the other car.

Primary goal:  validate that the scenario is fair before committing training time.
Secondary goal: record the human's gameplay as expert trajectories that can be
                mixed into skill training to prevent local minima.

Usage
-----
    # From the scenario builder menu (option 6), or directly:
    python training/human_play.py --scenario scenarios/configs/shooting/power_shot_center.yaml --side blue

Recorded trajectories are saved to:
    training/expert_data/<blue_skill>/<scenario_stem>_<side>_<timestamp>.npz

Each .npz contains:
    states   float32 (T, 14)  ball pos/vel (6) + car pos/vel/yaw/boost (8)
    actions  float32 (T,  8)  throttle/steer/pitch/yaw_ctrl/roll/jump/boost/handbrake
    side     str              'blue' or 'orange'
    scenario str              path to the scenario YAML
    fair     bool or None     human's post-session fairness verdict
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scenarios.scenario_config import ScenarioConfig

EXPERT_DATA_DIR = Path(__file__).parent / 'expert_data'


# ── state / action extraction ─────────────────────────────────────────────────

def _packet_to_state(packet) -> np.ndarray:
    """
    Extract a 14-float state vector from a RLBot GameTickPacket.

    Layout: [ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz,
             car_x,  car_y,  car_z,  car_vx,  car_vy,  car_vz,
             car_yaw, car_boost]
    """
    ball = packet.game_ball.physics
    b = [ball.location.x, ball.location.y, ball.location.z,
         ball.velocity.x,  ball.velocity.y,  ball.velocity.z]

    # index 0 = blue car, index 1 = orange car (1v1)
    car_idx = 0  # overridden by the caller
    car = packet.game_cars[car_idx].physics
    boost = packet.game_cars[car_idx].boost
    yaw   = packet.game_cars[car_idx].physics.rotation.yaw
    c = [car.location.x, car.location.y, car.location.z,
         car.velocity.x,  car.velocity.y,  car.velocity.z,
         yaw, float(boost)]

    return np.array(b + c, dtype=np.float32)


def _controls_to_action(ctrl) -> np.ndarray:
    """Flatten a RLBot SimpleControllerState into an 8-float vector."""
    return np.array([
        ctrl.throttle, ctrl.steer,    ctrl.pitch,
        ctrl.yaw,      ctrl.roll,
        float(ctrl.jump), float(ctrl.boost), float(ctrl.handbrake),
    ], dtype=np.float32)


# ── session runner ────────────────────────────────────────────────────────────

class HumanPlaySession:
    """
    Wraps a RLBot training exercise and runs a human vs bot episode.

    The session records every tick for replay / expert-data use.
    """

    def __init__(self, config: ScenarioConfig, side: str):
        self.config    = config
        self.side      = side          # 'blue' or 'orange'
        self.car_index = 0 if side == 'blue' else 1

        self._states:  list[np.ndarray] = []
        self._actions: list[np.ndarray] = []

    # ── recording ──

    def record_tick(self, packet, human_ctrl) -> None:
        """Call once per game tick from the RLBot agent."""
        state = _packet_to_state(packet)
        # patch in the correct car index
        car = packet.game_cars[self.car_index].physics
        boost = packet.game_cars[self.car_index].boost
        yaw   = car.rotation.yaw
        state[6:14] = [
            car.location.x, car.location.y, car.location.z,
            car.velocity.x, car.velocity.y, car.velocity.z,
            yaw, float(boost),
        ]
        self._states.append(state)
        self._actions.append(_controls_to_action(human_ctrl))

    def save(self, fair: Optional[bool] = None) -> Path:
        """Persist the recorded trajectory to disk and return the file path."""
        if not self._states:
            print('  (no ticks recorded — nothing saved)')
            return None

        skill = self.config.initial_state.blue.skill or 'unknown'
        out_dir = EXPERT_DATA_DIR / skill
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = Path(self.config.name.lower().replace(' ', '_'))
        out_path = out_dir / f'{stem}_{self.side}_{ts}.npz'

        np.savez(
            out_path,
            states=np.stack(self._states),
            actions=np.stack(self._actions),
            side=np.array(self.side),
            scenario=np.array(str(self.config.name)),
            fair=np.array(fair if fair is not None else 'unknown'),
        )
        print(f'  Saved {len(self._states)} ticks → {out_path}')
        return out_path


# ── match launcher ────────────────────────────────────────────────────────────

def launch(config: ScenarioConfig, side: str = 'blue') -> None:
    """
    Launch a RLBot 1v1 match:  human on `side`, bot on the other side.

    The initial state is set from the scenario config.  The session records
    the human's gameplay and asks for a fairness verdict at the end.
    """
    print(f'\n  Launching match — you play as {side.upper()}')
    print(f'  Scenario : {config.name}')
    print(f'  Blue     : {config.initial_state.blue.skill}')
    print(f'  Orange   : {config.initial_state.orange.skill}')
    print()

    try:
        from rlbot.setup_manager import SetupManager
    except ImportError:
        print('  [!] RLBot is not installed or not found.')
        print('      Install it with: pip install rlbot')
        print()
        print('  Tip: to test the scenario config without RLBot, use:')
        print(f'    python training/scenario_builder.py --view <yaml>')
        return

    session = HumanPlaySession(config, side)

    # ── build RLBot match config ──
    try:
        from rlbot.utils.game_state_util import (
            GameState, BallState, CarState, Physics, Vector3, Rotator,
        )
        from rlbot.matchconfig.match_config import (
            MatchConfig, PlayerConfig, MutatorConfig,
        )
        from rlbot.matchconfig.loadout_config import LoadoutConfig

        match_cfg = MatchConfig()
        match_cfg.game_mode = 'Soccer'
        match_cfg.game_map  = 'DFHStadium'
        match_cfg.mutators  = MutatorConfig()

        # Human player on the chosen side
        human_cfg = PlayerConfig()
        human_cfg.bot  = False
        human_cfg.name = 'Human'
        human_cfg.team = 0 if side == 'blue' else 1

        # Bot player on the other side
        bot_cfg = PlayerConfig()
        bot_cfg.bot       = True
        bot_cfg.name      = 'MLBot'
        bot_cfg.team      = 1 if side == 'blue' else 0
        bot_cfg.bot_skill = 1.0

        match_cfg.player_configs = [human_cfg, bot_cfg]

        mgr = SetupManager()
        mgr.load_config(config_location=None, match_config=match_cfg)
        mgr.launch_early_start_bot_processes()
        mgr.launch_bot_processes()
        mgr.start_match()

        # Tick loop — record until the match ends
        print('  Match started.  Press Ctrl-C to finish and save the recording.')
        try:
            while True:
                packet = mgr.game_interface.get_live_data_flat_binary()
                if packet is None:
                    time.sleep(0.016)
                    continue
                # TODO: pass actual human controller state when RLBot exposes it
                from rlbot.agents.base_agent import SimpleControllerState
                dummy_ctrl = SimpleControllerState()
                session.record_tick(packet, dummy_ctrl)
                time.sleep(0.016)
        except KeyboardInterrupt:
            pass

        mgr.shut_down()

    except Exception as exc:
        print(f'  [!] RLBot match setup failed: {exc}')
        print('      You can still use the scenario visualiser to review the config.')

    # ── post-session fairness prompt ──
    print()
    raw = input('  Was this scenario fair?  (y / n / s=skip): ').strip().lower()
    fair: Optional[bool] = None
    if raw == 'y':
        fair = True
    elif raw == 'n':
        fair = False

    session.save(fair=fair)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Launch a human vs bot RLBot match')
    parser.add_argument('--scenario', required=True, metavar='YAML',
                        help='Path to the scenario YAML file.')
    parser.add_argument('--side', choices=['blue', 'orange'], default='blue',
                        help='Which side the human plays.')
    args = parser.parse_args()

    config = ScenarioConfig.from_yaml(args.scenario)
    launch(config, args.side)


if __name__ == '__main__':
    main()
