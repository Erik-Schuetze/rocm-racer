"""
Frame capture from PCSX2 on Wayland via grim + hyprctl.

No portal consent dialogs, no persistent streams — works unattended
across gym episode resets.

Capture flow:
  hyprctl clients -j → find pcsx2-qt window → (x, y, w, h) region
  grim -g "x,y wxh" -t ppm - → Pillow (PPM, fast decode) → numpy
  → crop top 1/6 (HUD) → resize → grayscale
  → push into deque(maxlen=stack_size) → return (N, H, W) uint8
"""

import json
import subprocess
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class FrameCaptureConfig:
    width: int = 96
    height: int = 96
    grayscale: bool = True
    frame_stack_size: int = 4
    crop_top_fraction: float = 1 / 6


class FrameCapture:
    """
    Captures frames from the PCSX2 window using grim (Wayland screenshot
    tool) and hyprctl for automatic window discovery.

    Usage:
        fc = FrameCapture(config)
        fc.open()           # finds PCSX2 window via hyprctl, no dialog
        obs = fc.step()     # (N, H, W) uint8 numpy array
        fc.close()          # no-op, nothing to release
    """

    def __init__(self, config: Optional[FrameCaptureConfig] = None):
        self.cfg = config or FrameCaptureConfig()
        self._stack: deque[np.ndarray] = deque(maxlen=self.cfg.frame_stack_size)
        self._region: Optional[str] = None  # "x,y wxh" string for grim -g

    # ── lifecycle ──────────────────────────────────────────────────────────

    def open(self) -> None:
        """Find the PCSX2 window and initialise the frame stack."""
        x, y, w, h = self._find_pcsx2_window()
        self._region = f"{x},{y} {w}x{h}"
        print(f"[frame-capture] PCSX2 window at {self._region}")
        self.reset_stack()

    def close(self) -> None:
        """No persistent resources — nothing to release."""
        self._region = None

    def reset_stack(self) -> None:
        """Clear the frame stack and fill with zero frames."""
        blank = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
        self._stack.clear()
        for _ in range(self.cfg.frame_stack_size):
            self._stack.append(blank)

    # ── capture ────────────────────────────────────────────────────────────

    def grab(self) -> np.ndarray:
        """
        Capture one processed frame: grim → crop → resize → grayscale.
        Returns (H, W) uint8.
        """
        if self._region is None:
            raise RuntimeError("FrameCapture not open — call open() first")

        raw = self._grim_capture(self._region)
        return self._process(raw)

    def step(self) -> np.ndarray:
        """
        Capture one frame, push to stack, return stacked observation.
        Returns (frame_stack_size, H, W) uint8.
        """
        frame = self.grab()
        self._stack.append(frame)
        return np.stack(list(self._stack), axis=0)

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of the stacked observation: (N, H, W)."""
        return (self.cfg.frame_stack_size, self.cfg.height, self.cfg.width)

    # ── internals ──────────────────────────────────────────────────────────

    def _find_pcsx2_window(self) -> tuple[int, int, int, int]:
        """
        Query hyprctl for the PCSX2 window position and size.
        Returns (x, y, width, height).
        Raises RuntimeError if PCSX2 is not running.
        """
        try:
            raw = subprocess.check_output(
                ["hyprctl", "clients", "-j"],
                timeout=2.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"hyprctl not available: {e}") from e

        clients = json.loads(raw)
        for client in clients:
            if client.get("class") == "pcsx2-qt":
                x, y = client["at"]
                w, h = client["size"]
                return x, y, w, h

        raise RuntimeError(
            "PCSX2 window not found via hyprctl. "
            "Is PCSX2 running and visible on screen?"
        )

    def _grim_capture(self, region: str) -> np.ndarray:
        """
        Capture a screen region with grim, return as RGB numpy array.
        PPM format avoids PNG compression overhead (~25 fps for 1891x1502).
        """
        result = subprocess.run(
            ["grim", "-g", region, "-t", "ppm", "-"],
            capture_output=True,
            timeout=2.0,
        )
        if result.returncode != 0:
            raise RuntimeError(f"grim failed: {result.stderr.decode()}")
        img = Image.open(BytesIO(result.stdout))
        return np.array(img)

    def _process(self, raw: np.ndarray) -> np.ndarray:
        """
        Apply the image pipeline: crop top 1/6 → resize → grayscale.
        Input: (H, W, 3) RGB uint8. Returns (H, W) uint8.
        """
        img = Image.fromarray(raw, mode="RGB")

        # Crop top 1/6 (HUD: money counter, messages)
        w, h = img.size
        crop_top = int(h * self.cfg.crop_top_fraction)
        img = img.crop((0, crop_top, w, h))

        img = img.resize(
            (self.cfg.width, self.cfg.height),
            resample=Image.BILINEAR,
        )

        if self.cfg.grayscale:
            img = img.convert("L")

        return np.array(img, dtype=np.uint8)
