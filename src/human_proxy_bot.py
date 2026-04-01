"""
Human Proxy Bot
===============
RLBot Bot that reads a physical gamepad via evdev and forwards inputs
as ControllerState.  Records every tick for offline SAC training.

The human plays Rocket League through this bot — it adds <1ms of
latency (gamepad poll <1μs, tokenisation ~0.5ms, list append ~μs).

Usage
-----
    # RLBot config points to this bot for the human's car.
    # The bot auto-detects the first gamepad via evdev.

    # Or specify a device explicitly:
    GAMEPAD_DEVICE=/dev/input/event5 python src/human_proxy_bot.py

Requires
--------
    pip install evdev
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_SRC  = Path(__file__).parent
_REPO = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO / 'training'))

from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot

from transcriber import GameTranscriber


# ── Gamepad Reader (evdev, non-blocking) ──────────────────────────────────

class GamepadReader:
    """Non-blocking evdev gamepad reader.  <1μs per poll() call.

    Maintains axis/button state updated from kernel events each tick.
    Axes are normalised to [-1, 1] using the device-reported min/max.
    Triggers are normalised to [0, 1].

    Parameters
    ----------
    device_path : str, optional
        Explicit /dev/input/eventN path.  If None, auto-detects the
        first device with analog stick capabilities (EV_ABS).
    deadzone : float
        Stick deadzone as a fraction of full range.  Values within
        ±deadzone of centre are clamped to 0.
    """

    def __init__(self, device_path: str | None = None, deadzone: float = 0.08):
        from evdev import InputDevice, ecodes  # noqa: delayed import

        self._ecodes = ecodes
        path = device_path or self._find_gamepad()
        self.device = InputDevice(path)
        self.device.grab()  # exclusive access — prevents double-reads
        print(f'[gamepad] Grabbed {self.device.name} @ {path}')

        self.deadzone = deadzone

        # Cache axis info (min/max) and set initial values
        self._axis_info: dict[int, tuple[int, int]] = {}  # code → (min, max)
        self._axis_raw: dict[int, int] = {}                # code → raw int
        caps = self.device.capabilities(absinfo=True)
        for code, absinfo in caps.get(ecodes.EV_ABS, []):
            self._axis_info[code] = (absinfo.min, absinfo.max)
            self._axis_raw[code] = absinfo.value

        self._buttons: dict[int, bool] = {}

    # ── polling ───────────────────────────────────────────────────────

    def poll(self) -> None:
        """Drain all pending kernel events.  Non-blocking."""
        ecodes = self._ecodes
        while True:
            event = self.device.read_one()
            if event is None:
                break
            if event.type == ecodes.EV_ABS:
                self._axis_raw[event.code] = event.value
            elif event.type == ecodes.EV_KEY:
                self._buttons[event.code] = bool(event.value)

    # ── axis / button access ──────────────────────────────────────────

    def get_axis(self, code: int) -> float:
        """Axis value in [-1.0, 1.0] with deadzone applied."""
        raw = self._axis_raw.get(code, 0)
        info = self._axis_info.get(code)
        if info is None:
            return 0.0
        lo, hi = info
        mid = (hi + lo) / 2.0
        half_range = (hi - lo) / 2.0
        if half_range == 0:
            return 0.0
        val = (raw - mid) / half_range
        val = max(-1.0, min(1.0, val))
        if abs(val) < self.deadzone:
            return 0.0
        return val

    def get_trigger(self, code: int) -> float:
        """Trigger value in [0.0, 1.0] (no deadzone — triggers rest at 0)."""
        raw = self._axis_raw.get(code, 0)
        info = self._axis_info.get(code)
        if info is None:
            return 0.0
        lo, hi = info
        rng = hi - lo
        if rng == 0:
            return 0.0
        return max(0.0, min(1.0, (raw - lo) / rng))

    def get_button(self, code: int) -> bool:
        return self._buttons.get(code, False)

    # ── action vector ─────────────────────────────────────────────────

    def to_action(self) -> np.ndarray:
        """Current gamepad state → 8-float action vector.

        Layout matches src/bot.py translate_controls():
          [0] throttle   [-1, 1]   left stick Y (inverted)
          [1] steer      [-1, 1]   left stick X
          [2] pitch      [-1, 1]   right stick Y (inverted)
          [3] yaw        [-1, 1]   right stick X
          [4] roll       [-1, 1]   right trigger − left trigger
          [5] jump       {0, 1}    A / BTN_SOUTH
          [6] boost      {0, 1}    B / BTN_EAST
          [7] handbrake  {0, 1}    X / BTN_WEST
        """
        ec = self._ecodes
        throttle  = -self.get_axis(ec.ABS_Y)
        steer     =  self.get_axis(ec.ABS_X)
        pitch     = -self.get_axis(ec.ABS_RY)
        yaw       =  self.get_axis(ec.ABS_RX)
        roll      =  self.get_trigger(ec.ABS_RZ) - self.get_trigger(ec.ABS_Z)
        jump      =  float(self.get_button(ec.BTN_SOUTH))
        boost     =  float(self.get_button(ec.BTN_EAST))
        handbrake =  float(self.get_button(ec.BTN_WEST))
        return np.array(
            [throttle, steer, pitch, yaw, roll, jump, boost, handbrake],
            dtype=np.float32,
        )

    # ── auto-detect ───────────────────────────────────────────────────

    def _find_gamepad(self) -> str:
        """Find the first input device with analog stick capabilities."""
        from evdev import InputDevice, list_devices, ecodes

        for path in list_devices():
            try:
                dev = InputDevice(path)
                caps = dev.capabilities()
                if ecodes.EV_ABS in caps:
                    abs_codes = [c for c, _ in caps[ecodes.EV_ABS]]
                    # Must have at least left stick (ABS_X + ABS_Y)
                    if ecodes.ABS_X in abs_codes and ecodes.ABS_Y in abs_codes:
                        print(f'[gamepad] Auto-detected: {dev.name} @ {path}')
                        dev.close()
                        return path
                dev.close()
            except (PermissionError, OSError):
                continue
        raise RuntimeError(
            'No gamepad found.  Plug in a controller or set '
            'GAMEPAD_DEVICE=/dev/input/eventN'
        )

    # ── cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self.device.ungrab()
        except OSError:
            pass
        self.device.close()


# ── Human Proxy Bot ───────────────────────────────────────────────────────

class HumanProxyBot(Bot):
    """RLBot agent that mirrors a physical gamepad as ControllerState.

    Per-tick cost: ~0.6ms (gamepad <1μs + tokenise ~0.5ms + bookkeeping).
    Budget: 8.33ms (120 Hz).  Headroom: ~7.7ms.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.gamepad: GamepadReader | None = None
        self.transcriber: GameTranscriber | None = None

    def initialize(self) -> None:
        device_path = os.environ.get('GAMEPAD_DEVICE', None)
        self.gamepad = GamepadReader(device_path=device_path)
        self.transcriber = GameTranscriber('humanproxy')

    def get_output(self, packet: GamePacket) -> ControllerState:
        # 1. Drain pending gamepad events (<1μs)
        self.gamepad.poll()

        # 2. Build 8-float action from current state (<1μs)
        action = self.gamepad.to_action()

        # 3. Record tick for training (~0.5ms — tokenisation dominates)
        self.transcriber.record_tick(packet, self.index, action)

        # 4. Convert to ControllerState (<1μs)
        return self._translate_controls(action)

    def retire(self) -> None:
        """Called by RLBot on shutdown.  Save transcript and release device."""
        if self.transcriber is not None:
            self.transcriber.save()
        if self.gamepad is not None:
            self.gamepad.close()

    @staticmethod
    def _translate_controls(action: np.ndarray) -> ControllerState:
        """Map 8-float action to ControllerState (same as MyBot.translate_controls)."""
        ctrl           = ControllerState()
        ctrl.throttle  = float(np.clip(action[0], -1.0, 1.0))
        ctrl.steer     = float(np.clip(action[1], -1.0, 1.0))
        ctrl.pitch     = float(np.clip(action[2], -1.0, 1.0))
        ctrl.yaw       = float(np.clip(action[3], -1.0, 1.0))
        ctrl.roll      = float(np.clip(action[4], -1.0, 1.0))
        ctrl.jump      = bool(action[5] >= 0.5)
        ctrl.boost     = bool(action[6] >= 0.5)
        ctrl.handbrake = bool(action[7] >= 0.5)
        return ctrl


if __name__ == '__main__':
    HumanProxyBot('austin/humanproxy').run()
