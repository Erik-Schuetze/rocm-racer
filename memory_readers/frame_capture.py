"""
Frame capture from PCSX2 on Wayland via PipeWire portal.

Primary path:  pipewire-capture (PortalCapture → CaptureStream)
Fallback path: grim subprocess → Pillow

The capture flow:
  PCSX2 window (BGRA) → crop top 1/6 (HUD) → resize → grayscale
  → push into deque(maxlen=stack_size) → return (N, H, W) uint8 array
"""

import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
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
    Captures frames from the PCSX2 window, processes them into a
    stacked observation tensor.

    Usage:
        fc = FrameCapture(config)
        fc.open()               # shows portal consent dialog once
        obs = fc.step()         # (N, H, W) uint8 numpy array
        fc.close()
    """

    def __init__(self, config: Optional[FrameCaptureConfig] = None):
        self.cfg = config or FrameCaptureConfig()
        self._stack: deque[np.ndarray] = deque(maxlen=self.cfg.frame_stack_size)
        self._session = None
        self._stream = None
        self._use_pipewire: bool = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the capture stream. Triggers portal consent dialog once."""
        try:
            from pipewire_capture import CaptureStream, PortalCapture, is_available

            if not is_available():
                raise RuntimeError("PipeWire ScreenCast portal not available")

            portal = PortalCapture()
            self._session = portal.select_window()
            if self._session is None:
                raise RuntimeError("No window selected in portal dialog")

            self._stream = CaptureStream(
                self._session.fd,
                self._session.node_id,
                self._session.width,
                self._session.height,
            )
            self._stream.start()

            # Wait briefly for the first frame to arrive
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if self._stream.get_frame() is not None:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("PipeWire stream started but no frame arrived within 5s")

            self._use_pipewire = True
            print("[frame-capture] PipeWire portal stream open.")

        except Exception as e:
            print(f"[frame-capture] PipeWire unavailable ({e}), falling back to grim.")
            self._use_pipewire = False

        self.reset_stack()

    def close(self) -> None:
        """Stop the capture stream and release resources."""
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
        if self._session is not None:
            self._session.close()
            self._session = None

    def reset_stack(self) -> None:
        """Clear the frame stack and fill with zero frames."""
        blank = np.zeros(
            (self.cfg.height, self.cfg.width), dtype=np.uint8
        )
        self._stack.clear()
        for _ in range(self.cfg.frame_stack_size):
            self._stack.append(blank)

    # ── capture ────────────────────────────────────────────────────────────

    def grab(self) -> np.ndarray:
        """
        Capture one processed frame: crop → resize → grayscale.
        Returns a (H, W) uint8 array (grayscale) or (H, W, 3) RGB array.
        """
        if self._use_pipewire:
            raw = self._grab_pipewire()
        else:
            raw = self._grab_grim()

        return self._process(raw)

    def step(self) -> np.ndarray:
        """
        Capture one frame, push to stack, return stacked observation.
        Returns (frame_stack_size, H, W) uint8 array.
        """
        frame = self.grab()
        self._stack.append(frame)
        return np.stack(list(self._stack), axis=0)

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of the stacked observation: (N, H, W)."""
        return (self.cfg.frame_stack_size, self.cfg.height, self.cfg.width)

    # ── internals ──────────────────────────────────────────────────────────

    def _grab_pipewire(self) -> np.ndarray:
        """Get the latest frame from the PipeWire stream as BGRA numpy array."""
        if self._stream is None or self._stream.window_invalid:
            raise RuntimeError("PipeWire stream is closed or window was destroyed")
        frame = self._stream.get_frame()
        if frame is None:
            # No new frame yet — reuse last stack entry (caller gets stale frame)
            last = self._stack[-1] if self._stack else np.zeros(
                (self.cfg.height, self.cfg.width), dtype=np.uint8
            )
            return last
        # frame is (H, W, 4) BGRA — convert to PIL for uniform processing
        return frame  # returned as-is; _process handles conversion

    def _grab_grim(self) -> np.ndarray:
        """Capture current focused window via grim (fallback, Wayland)."""
        result = subprocess.run(
            ["grim", "-"],
            capture_output=True,
            timeout=2.0,
        )
        if result.returncode != 0:
            raise RuntimeError(f"grim failed: {result.stderr.decode()}")
        img = Image.open(BytesIO(result.stdout))
        return np.array(img)

    def _process(self, raw: np.ndarray) -> np.ndarray:
        """
        Apply the image pipeline:
          BGRA/RGBA/RGB numpy → PIL → crop top fraction → resize → (grayscale)
        Returns (H, W) uint8 for grayscale, (H, W, 3) uint8 for colour.
        """
        if raw.ndim == 2:
            # Already a processed frame (stale frame reuse path)
            return raw

        # Convert BGRA (PipeWire) or RGBA (grim) to PIL RGB
        if raw.shape[2] == 4:
            # BGRA → RGB
            img = Image.fromarray(raw[:, :, [2, 1, 0]], mode="RGB")
        else:
            img = Image.fromarray(raw, mode="RGB")

        # Crop top 1/6 (HUD: money, messages)
        w, h = img.size
        crop_top = int(h * self.cfg.crop_top_fraction)
        img = img.crop((0, crop_top, w, h))

        # Resize
        img = img.resize(
            (self.cfg.width, self.cfg.height),
            resample=Image.BILINEAR,
        )

        if self.cfg.grayscale:
            img = img.convert("L")

        return np.array(img, dtype=np.uint8)
