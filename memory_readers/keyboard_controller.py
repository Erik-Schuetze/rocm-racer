from __future__ import annotations

import subprocess
import time


class KeyboardController:
    """Send keypresses to a PCSX2 window by process ID using xdotool.

    Uses ``xdotool key --window <id> <key>`` so PCSX2 does not need
    to be the focused window. Requires xdotool to be installed
    (pacman -S xdotool) but needs no special OS permissions.

    Args:
        pid: PID of the PCSX2 process. Pass None to send to the
             currently focused window (not recommended for training).
        settle_seconds: How long to wait after construction before
             the first keypress, giving PCSX2 time to create its
             renderer window.
    """

    def __init__(self, pid: int | None = None, settle_seconds: float = 2.0) -> None:
        self.pid = pid
        self._window_id: str | None = None
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        if pid is not None:
            self._window_id = self._find_window(pid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_window(self, pid: int) -> str:
        """Return the X11 window ID for the PCSX2 renderer window."""
        result = subprocess.run(
            ["xdotool", "search", "--pid", str(pid)],
            capture_output=True,
            text=True,
        )
        ids = result.stdout.strip().splitlines()
        if not ids:
            raise RuntimeError(
                f"[rocm-racer] xdotool found no X11 windows for pid={pid}. "
                "Is PCSX2 running and has it created its render window yet?"
            )
        # The last window ID is the game renderer; earlier ones are Qt shell windows.
        window_id = ids[-1]
        print(f"[rocm-racer] Targeting PCSX2 window id={window_id} (pid={pid})")
        return window_id

    def _xdotool_key(self, key: str) -> None:
        cmd = ["xdotool", "key"]
        if self._window_id:
            cmd += ["--window", self._window_id]
        cmd.append(key)
        subprocess.run(cmd, check=True)

    def _xdotool_key_down(self, key: str) -> None:
        cmd = ["xdotool", "keydown"]
        if self._window_id:
            cmd += ["--window", self._window_id]
        cmd.append(key)
        subprocess.run(cmd, check=True)

    def _xdotool_key_up(self, key: str) -> None:
        cmd = ["xdotool", "keyup"]
        if self._window_id:
            cmd += ["--window", self._window_id]
        cmd.append(key)
        subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def press(self, key: str, duration: float = 0.1) -> None:
        """Press and release a key, holding it for ``duration`` seconds."""
        self._xdotool_key_down(key)
        time.sleep(duration)
        self._xdotool_key_up(key)

    def tap(self, key: str) -> None:
        """Single instantaneous keypress (down + up in one xdotool call)."""
        self._xdotool_key(key)

    def hold(self, key: str) -> None:
        """Hold a key down (call release() to let go)."""
        self._xdotool_key_down(key)

    def release(self, key: str) -> None:
        """Release a previously held key."""
        self._xdotool_key_up(key)


__all__ = ["KeyboardController"]
