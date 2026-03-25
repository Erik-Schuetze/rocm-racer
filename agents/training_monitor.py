"""
Training observability for rocm-racer PPO training.

TrainingMonitorCallback provides:
  - One log line per completed episode with key statistics
  - Live OpenCV preview window showing what the model sees + its current action

Episode log format:
  [ep  42] 47.3s  182 steps  R=+34.2  avg=72 km/h  max=115 km/h  dist=312m  reason=crash

Preview window (384×384, updated every N steps):
  - Upscaled latest 96×96 grayscale frame (last frame in the stack)
  - Green line from bottom-center: steering angle (±1 → ±45° fan)
  - Red vertical bar on right edge: accel (up from mid) / brake (down from mid)
"""
from __future__ import annotations

import os
import time
from typing import Any

# OpenCV ships its own Qt that only has the XCB (X11) plugin.
# On Wayland sessions QT_QPA_PLATFORM is typically set to "wayland",
# which crashes cv2. Force xcb before cv2 is imported.
os.environ["QT_QPA_PLATFORM"] = "xcb"
# Suppress "Cannot find font directory" spam from Qt (harmless, fonts are unused)
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMonitorCallback(BaseCallback):
    """
    SB3 callback that logs episode summaries and renders a live preview window.

    Supports vectorized environments (N parallel envs). Episode stats are
    tracked per-env. Preview always shows env 0.

    Parameters
    ----------
    preview : bool
        Enable the OpenCV preview window (default True).
    preview_scale : int
        Scale factor applied to the 96×96 frame (default 4 → 384×384).
    preview_interval : int
        Render the preview every N steps (default 5).
    """

    def __init__(
        self,
        preview: bool = True,
        preview_scale: int = 4,
        preview_interval: int = 5,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._preview = preview
        self._preview_scale = preview_scale
        self._preview_interval = preview_interval

        # Per-env episode tracking (initialised in _on_training_start)
        self._num_envs = 1
        self._ep_count: list[int] = []
        self._ep_start: list[float] = []
        self._ep_reward: list[float] = []
        self._ep_speeds: list[list[float]] = []
        self._ep_max_dist: list[float] = []
        self._last_reason: list[str] = []

        # Gradient update tracking
        self._update_count = 0

        # Current action (updated each step for the preview overlay)
        self._last_action: np.ndarray = np.zeros(2, dtype=np.float32)

    # ── SB3 lifecycle ─────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        self._num_envs = self.training_env.num_envs
        now = time.monotonic()
        self._ep_count = [0] * self._num_envs
        self._ep_start = [now] * self._num_envs
        self._ep_reward = [0.0] * self._num_envs
        self._ep_speeds = [[] for _ in range(self._num_envs)]
        self._ep_max_dist = [0.0] * self._num_envs
        self._last_reason = [""] * self._num_envs

        n_steps = getattr(self.model, "n_steps", "?")
        rollout_total = f"{n_steps} × {self._num_envs} = {n_steps * self._num_envs}" if self._num_envs > 1 else str(n_steps)
        print(f"[monitor] Rollout size: {rollout_total} steps per gradient update")
        if self._preview:
            cv2.namedWindow("rocm-racer", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                "rocm-racer",
                96 * self._preview_scale,
                96 * self._preview_scale,
            )

    def _on_rollout_end(self) -> None:
        self._update_count += 1
        total_ts = self.num_timesteps
        print(
            f"[update {self._update_count:3d}]"
            f"  {total_ts:,} total steps"
            f"  — gradient update complete"
        )

    def _on_step(self) -> bool:
        # Pull step data from SB3 locals (set by on_policy_algorithm.py)
        rewards: np.ndarray = self.locals.get("rewards", np.zeros(self._num_envs))
        dones: np.ndarray = self.locals.get("dones", np.zeros(self._num_envs, dtype=bool))
        infos: list[dict[str, Any]] = self.locals.get("infos", [{}] * self._num_envs)
        actions: np.ndarray = self.locals.get("actions", np.zeros((self._num_envs, 2)))

        for env_i in range(self._num_envs):
            self._ep_reward[env_i] += float(rewards[env_i])
            info = infos[env_i] if env_i < len(infos) else {}

            speed_kph: float = info.get("speed_kph", 0.0)
            self._ep_speeds[env_i].append(speed_kph)

            dist: float = info.get("distance_from_start", 0.0)
            if dist > self._ep_max_dist[env_i]:
                self._ep_max_dist[env_i] = dist

            reason = info.get("terminated_reason", "")
            if reason:
                self._last_reason[env_i] = reason

            if dones[env_i]:
                self._ep_count[env_i] += 1
                elapsed = time.monotonic() - self._ep_start[env_i]

                ep_info = info.get("episode", {})
                total_reward = ep_info.get("r", self._ep_reward[env_i])
                ep_steps = ep_info.get("l", len(self._ep_speeds[env_i]))

                avg_speed = (
                    sum(self._ep_speeds[env_i]) / len(self._ep_speeds[env_i])
                    if self._ep_speeds[env_i] else 0.0
                )
                max_speed = max(self._ep_speeds[env_i]) if self._ep_speeds[env_i] else 0.0

                n_steps = getattr(self.model, "n_steps", 2048)
                rollout_size = n_steps * self._num_envs
                rollout_pos = self.num_timesteps % rollout_size
                rollout_remaining = rollout_size - rollout_pos if rollout_pos > 0 else 0

                total_eps = sum(self._ep_count)
                env_tag = f"/e{env_i}" if self._num_envs > 1 else ""
                print(
                    f"[ep {total_eps:4d}{env_tag}]"
                    f"  {elapsed:6.1f}s"
                    f"  {int(ep_steps):4d} steps"
                    f"  R={total_reward:+7.1f}"
                    f"  avg={avg_speed:5.0f} km/h"
                    f"  max={max_speed:5.0f} km/h"
                    f"  dist={self._ep_max_dist[env_i]:6.0f}m"
                    f"  reason={self._last_reason[env_i] or 'truncated'}"
                    f"  [{rollout_remaining:4d} to update]"
                )

                # Reset for next episode
                self._ep_start[env_i] = time.monotonic()
                self._ep_reward[env_i] = 0.0
                self._ep_speeds[env_i] = []
                self._ep_max_dist[env_i] = 0.0
                self._last_reason[env_i] = ""

        # Track env 0 action for preview overlay
        if actions is not None and len(actions) > 0:
            self._last_action = np.asarray(actions[0], dtype=np.float32)

        # Render preview window (env 0 only)
        if self._preview and (self.n_calls % self._preview_interval == 0):
            self._render_preview(infos)

        return True

    def _on_training_end(self) -> None:
        if self._preview:
            cv2.destroyWindow("rocm-racer")

    # ── preview rendering ─────────────────────────────────────────────────

    def _render_preview(self, infos: list[dict[str, Any]]) -> None:
        """
        Render the latest 96×96 grayscale frame upscaled with action overlays.

        Green steering line: radiates from bottom-center, ±1 → ±45°.
        Red accel/brake bar: right edge vertical bar, mid = 0, up = throttle, down = brake.
        """
        # Try to get the latest frame from the frame_capture attribute on the env
        try:
            frame_captures = self.training_env.get_attr("frame_capture")
            fc = frame_captures[0] if frame_captures else None
        except Exception:
            fc = None

        if fc is None:
            return

        # Grab the last frame from the stack (most recent)
        try:
            stack = fc._stack  # deque of (H, W) uint8 frames
            if not stack:
                return
            frame = np.array(stack[-1], dtype=np.uint8)  # (96, 96)
        except Exception:
            return

        scale = self._preview_scale
        h, w = frame.shape  # 96, 96
        out_h, out_w = h * scale, w * scale

        # Upscale to colour BGR for drawing
        img = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # ── Steering line (green) ──────────────────────────────────────────
        steering = float(self._last_action[0]) if len(self._last_action) > 0 else 0.0
        steering = max(-1.0, min(1.0, steering))

        # Base: bottom center
        bx, by = out_w // 2, out_h - 1
        # Angle: 90° is straight up; ±1 → ±45° (so range is 45° … 135°)
        angle_deg = 90.0 - steering * 45.0
        angle_rad = np.deg2rad(angle_deg)
        line_len = out_h // 3
        ex = int(bx + line_len * np.cos(angle_rad))
        ey = int(by - line_len * np.sin(angle_rad))
        ex = max(0, min(out_w - 1, ex))
        ey = max(0, min(out_h - 1, ey))
        cv2.line(img, (bx, by), (ex, ey), color=(0, 255, 0), thickness=1)

        # ── Accel/brake bar (red) ──────────────────────────────────────────
        accel_brake = float(self._last_action[1]) if len(self._last_action) > 1 else 0.0
        accel_brake = max(-1.0, min(1.0, accel_brake))

        bar_x = out_w - 8          # 8px wide bar on right edge
        bar_w = 7
        mid_y = out_h // 2
        max_bar = out_h // 2 - 4   # pixels from center to top/bottom edge

        bar_len = int(abs(accel_brake) * max_bar)
        if accel_brake >= 0:
            # Throttle: bar goes upward from center
            y1, y2 = mid_y - bar_len, mid_y
        else:
            # Brake: bar goes downward from center
            y1, y2 = mid_y, mid_y + bar_len

        # Draw background track
        cv2.rectangle(img, (bar_x, 4), (bar_x + bar_w, out_h - 4),
                      color=(40, 40, 40), thickness=-1)
        # Draw center tick
        cv2.line(img, (bar_x, mid_y), (bar_x + bar_w, mid_y),
                 color=(100, 100, 100), thickness=1)
        # Draw filled bar
        if bar_len > 0:
            cv2.rectangle(img, (bar_x, y1), (bar_x + bar_w, y2),
                          color=(0, 0, 220), thickness=-1)

        # ── Speed label ───────────────────────────────────────────────────
        info = infos[0] if infos else {}
        speed_kph: float = info.get("speed_kph", 0.0)
        cv2.putText(
            img,
            f"{speed_kph:.0f} km/h",
            (4, out_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 * scale / 4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("rocm-racer", img)
        cv2.waitKey(1)
