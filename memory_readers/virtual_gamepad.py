from __future__ import annotations

import time

import evdev
from evdev import UInput, ecodes as e


# Axis ranges match a standard DualShock2 / SDL gamepad profile.
# PCSX2's SDL input backend will pick this device up automatically.
_AXIS_MIN = 0
_AXIS_MAX = 255
_AXIS_CENTER = 128

# ---------------------------------------------------------------------------
# SDL3 game controller mapping
# ---------------------------------------------------------------------------
# GUID is derived from the uinput device identity set below:
#   bustype=BUS_USB(0x03)  vendor=0x0000  product=0x0000  version=0x0003
# Format: bustype(LE16)+pad + vendor(LE16)+pad + product(LE16)+pad + version(LE16)+pad
SDL_GUID = "03000000000000000000000003000000"

# Button indices (SDL joystick scan order, gaps from absent BTN_C/BTN_Z/BTN_MODE skipped):
#   b0=BTN_SOUTH(Cross)  b1=BTN_EAST(Circle)  b2=BTN_NORTH(Triangle)  b3=BTN_WEST(Square)
#   b4=BTN_TL(L1)  b5=BTN_TR(R1)  b6=BTN_TL2  b7=BTN_TR2
#   b8=BTN_SELECT  b9=BTN_START  b10=BTN_THUMBL(L3)  b11=BTN_THUMBR(R3)
# Axis indices (ABS code order):
#   a0=ABS_X(leftx)  a1=ABS_Y(lefty)  a2=ABS_Z(L2 trigger)
#   a3=ABS_RX(rightx)  a4=ABS_RY(righty)  a5=ABS_RZ(R2 trigger)
#   h0=ABS_HAT0X/Y (d-pad)
SDL_MAPPING = (
    f"{SDL_GUID},rocm-racer Virtual Gamepad,"
    "a:b0,b:b1,y:b2,x:b3,"
    "leftshoulder:b4,rightshoulder:b5,"
    "back:b8,start:b9,"
    "leftstick:b10,rightstick:b11,"
    "leftx:a0,lefty:a1,lefttrigger:+a2,"
    "rightx:a3,righty:a4,righttrigger:+a5,"
    "dpup:h0.1,dpright:h0.2,dpdown:h0.4,dpleft:h0.8,"
    "platform:Linux,"
)
_TRIGGER_MIN = 0
_TRIGGER_MAX = 255

_CAPABILITIES = {
    e.EV_KEY: [
        e.BTN_SOUTH,   # Cross   → SDL FaceSouth / a:b0
        e.BTN_EAST,    # Circle  → SDL FaceEast  / b:b1
        e.BTN_NORTH,   # Triangle→ SDL FaceNorth / y:b2
        e.BTN_WEST,    # Square  → SDL FaceWest  / x:b3
        e.BTN_TL,      # L1      → SDL LeftShoulder  / b4
        e.BTN_TR,      # R1      → SDL RightShoulder / b5
        e.BTN_TL2,     # L2 (digital fallback)        / b6
        e.BTN_TR2,     # R2 (digital fallback)        / b7
        e.BTN_SELECT,  # Select  → SDL Back       / b8
        e.BTN_START,   # Start   → SDL Start      / b9
        e.BTN_THUMBL,  # L3      → SDL LeftStick  / b10
        e.BTN_THUMBR,  # R3      → SDL RightStick / b11
    ],
    e.EV_ABS: [
        # Left stick
        (e.ABS_X,  evdev.AbsInfo(value=_AXIS_CENTER, min=_AXIS_MIN, max=_AXIS_MAX, fuzz=4, flat=8, resolution=0)),
        (e.ABS_Y,  evdev.AbsInfo(value=_AXIS_CENTER, min=_AXIS_MIN, max=_AXIS_MAX, fuzz=4, flat=8, resolution=0)),
        # Right stick
        (e.ABS_RX, evdev.AbsInfo(value=_AXIS_CENTER, min=_AXIS_MIN, max=_AXIS_MAX, fuzz=4, flat=8, resolution=0)),
        (e.ABS_RY, evdev.AbsInfo(value=_AXIS_CENTER, min=_AXIS_MIN, max=_AXIS_MAX, fuzz=4, flat=8, resolution=0)),
        # Analog triggers — L2 = ABS_Z, R2 = ABS_RZ
        (e.ABS_Z,  evdev.AbsInfo(value=0, min=_TRIGGER_MIN, max=_TRIGGER_MAX, fuzz=0, flat=0, resolution=0)),
        (e.ABS_RZ, evdev.AbsInfo(value=0, min=_TRIGGER_MIN, max=_TRIGGER_MAX, fuzz=0, flat=0, resolution=0)),
        # D-pad
        (e.ABS_HAT0X, evdev.AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)),
        (e.ABS_HAT0Y, evdev.AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)),
    ],
}


class VirtualGamepad:
    """
    Linux uinput virtual DualShock2-like gamepad for PCSX2 action injection.

    Axis mapping:
      steering  [-1.0,  1.0]  → ABS_X  (left stick horizontal)
      throttle  [ 0.0,  1.0]  → ABS_RZ (R2 analog trigger)
      brake     [ 0.0,  1.0]  → ABS_Z  (L2 analog trigger)

    Requires write access to /dev/uinput. Add your user to the `input` group
    or run with sufficient privileges:
        sudo usermod -aG input $USER
    """

    def __init__(self, name: str = "rocm-racer Virtual Gamepad", settle_seconds: float = 0.5) -> None:
        self.name = name
        self.settle_seconds = settle_seconds
        self._uinput: UInput | None = None

    def open(self) -> None:
        if self._uinput is not None:
            return
        self._uinput = UInput(
            _CAPABILITIES,
            name=self.name,
            bustype=e.BUS_USB,   # 0x03 → matches SDL_GUID
            vendor=0x0000,
            product=0x0000,
            version=0x0003,
        )
        # Give the OS and PCSX2's SDL backend time to enumerate the new device.
        time.sleep(self.settle_seconds)

    def close(self) -> None:
        if self._uinput is None:
            return
        self.center()
        self._uinput.close()
        self._uinput = None

    def __enter__(self) -> "VirtualGamepad":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def send(self, steering: float, throttle: float, brake: float) -> None:
        """
        Push one frame of analog input.

        Args:
            steering: [-1.0, 1.0] — negative = left, positive = right
            throttle: [ 0.0, 1.0] — R2 pressure
            brake:    [ 0.0, 1.0] — L2 pressure
        """
        if self._uinput is None:
            raise RuntimeError("VirtualGamepad is not open. Call open() first.")

        steer_raw = int(_AXIS_CENTER + steering * (_AXIS_MAX - _AXIS_CENTER))
        steer_raw = max(_AXIS_MIN, min(_AXIS_MAX, steer_raw))

        throttle_raw = int(throttle * _TRIGGER_MAX)
        throttle_raw = max(_TRIGGER_MIN, min(_TRIGGER_MAX, throttle_raw))

        brake_raw = int(brake * _TRIGGER_MAX)
        brake_raw = max(_TRIGGER_MIN, min(_TRIGGER_MAX, brake_raw))

        ui = self._uinput
        ui.write(e.EV_ABS, e.ABS_X,  steer_raw)
        ui.write(e.EV_ABS, e.ABS_RZ, throttle_raw)
        ui.write(e.EV_ABS, e.ABS_Z,  brake_raw)
        ui.syn()

    def press_button(self, button: int, duration: float = 0.05) -> None:
        """Press and release a single digital button (evdev BTN_* keycode)."""
        if self._uinput is None:
            raise RuntimeError("VirtualGamepad is not open. Call open() first.")
        ui = self._uinput
        ui.write(e.EV_KEY, button, 1)
        ui.syn()
        time.sleep(duration)
        ui.write(e.EV_KEY, button, 0)
        ui.syn()

    def center(self) -> None:
        """Return all axes to their neutral/resting positions."""
        if self._uinput is None:
            return
        ui = self._uinput
        ui.write(e.EV_ABS, e.ABS_X,  _AXIS_CENTER)
        ui.write(e.EV_ABS, e.ABS_Y,  _AXIS_CENTER)
        ui.write(e.EV_ABS, e.ABS_RX, _AXIS_CENTER)
        ui.write(e.EV_ABS, e.ABS_RY, _AXIS_CENTER)
        ui.write(e.EV_ABS, e.ABS_Z,  0)
        ui.write(e.EV_ABS, e.ABS_RZ, 0)
        ui.syn()


__all__ = ["VirtualGamepad"]
