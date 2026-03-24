from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from memory_readers.nfsu2_memory import NFSU2MemoryReader, TelemetrySample
from memory_readers.virtual_gamepad import VirtualGamepad


@dataclass(frozen=True)
class RewardConfig:
    speed_weight: float = 0.02       # per-step: reward = speed_kph * speed_weight (uncapped)
    stuck_penalty: float = -10.0     # one-time penalty on stuck termination
    slow_timeout_penalty: float = -5.0  # one-time penalty on slow_timeout termination
    success_bonus: float = 50.0      # applied when distance > success_distance_m


@dataclass(frozen=True)
class PCSX2EnvConfig:
    step_interval_seconds: float = 0.1
    max_episode_steps: int = 2_000
    device: str = "cuda"
    reward: RewardConfig = field(default_factory=RewardConfig)
    savestate_slot: int = 0
    savestate_settle_s: float = 1.0  # wait time after load_state() before reading
    # Termination thresholds
    stuck_speed_threshold_kph: float = 3.0
    stuck_timeout_s: float = 2.0     # no grace — should never be at 0 km/h for 2s
    slow_speed_threshold_kph: float = 15.0
    slow_speed_timeout_s: float = 5.0
    slow_speed_grace_s: float = 5.0  # ignore slow speed for first N seconds (acceleration time)
    success_distance_m: float = 1000.0


@dataclass(frozen=True)
class ControlAction:
    # NFS Underground 2 PS2 control scheme (EA Black Box, NTSC-U)
    # ------------------------------------------------------------
    # Analog RL inputs:
    #   steering    [-1.0, 1.0]  → ABS_X  left stick horizontal
    #   accel_brake [-1.0, 1.0]  → ABS_RY right stick Y
    #     > 0 = throttle (stick up), < 0 = brake (stick down)
    steering: float     # [-1.0, 1.0]
    accel_brake: float  # [-1.0, 1.0]  (+throttle / -brake)


class PCSX2RacerEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        memory_reader: NFSU2MemoryReader | None = None,
        gamepad: VirtualGamepad | None = None,
        config: PCSX2EnvConfig | None = None,
        frame_capture=None,   # Optional[FrameCapture]
        sleep_fn=time.sleep,
    ) -> None:
        super().__init__()
        self.config = config or PCSX2EnvConfig()
        self.memory_reader = memory_reader or NFSU2MemoryReader()
        self.gamepad = gamepad or VirtualGamepad()
        self.frame_capture = frame_capture
        self.sleep_fn = sleep_fn
        self.device = torch.device(self.config.device)

        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Telemetry observation: [speed_kph, x, y, z, track_progress]
        telemetry_space = spaces.Box(
            low=np.asarray(
                [0.0, -1_000_000.0, -1_000_000.0, -1_000_000.0, 0.0],
                dtype=np.float32,
            ),
            high=np.asarray(
                [500.0, 1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        if self.frame_capture is not None:
            n, h, w = self.frame_capture.observation_shape
            self.observation_space = spaces.Dict({
                "image": spaces.Box(0, 255, shape=(n, h, w), dtype=np.uint8),
                "telemetry": telemetry_space,
            })
        else:
            self.observation_space = telemetry_space

        self._episode_steps = 0
        self._last_telemetry: TelemetrySample | None = None

        # Episode state for termination tracking
        self._start_position: tuple[float, float, float] | None = None
        self._slow_speed_elapsed: float = 0.0
        self._stuck_elapsed: float = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self.memory_reader.open()
        self.gamepad.open()
        self.gamepad.center()

        # Reload savestate for a clean episode
        self.memory_reader.load_state(self.config.savestate_slot)
        self.sleep_fn(self.config.savestate_settle_s)

        # Reset all episode tracking state
        self._episode_steps = 0
        self._slow_speed_elapsed = 0.0
        self._stuck_elapsed = 0.0

        self._last_telemetry = self.memory_reader.read_telemetry()
        self._start_position = self._last_telemetry.position

        obs = self._build_obs(self._last_telemetry, reset=True)
        info = self._build_info(self._last_telemetry, reward=0.0, reward_terms={},
                                distance=0.0, terminated_reason="")
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | dict, float, bool, bool, dict[str, Any]]:
        clipped = np.asarray(action, dtype=np.float32).clip(
            self.action_space.low, self.action_space.high
        )
        control = ControlAction(
            steering=float(clipped[0]),
            accel_brake=float(clipped[1]),
        )
        self._apply_action(control)
        self.sleep_fn(self.config.step_interval_seconds)

        telemetry = self.memory_reader.read_telemetry()
        self._episode_steps += 1

        # --- Compute per-step reward ---
        speed_term = self.config.reward.speed_weight * telemetry.speed_kph
        reward_terms: dict[str, float] = {"speed": speed_term}

        # --- Distance from start ---
        distance = self._euclidean_distance(telemetry.position)

        # --- Termination conditions ---
        terminated = False
        terminated_reason = ""
        episode_time = self._episode_steps * self.config.step_interval_seconds

        # Stuck detection (no grace period — should never sit at ~0 km/h)
        if telemetry.speed_kph < self.config.stuck_speed_threshold_kph:
            self._stuck_elapsed += self.config.step_interval_seconds
        else:
            self._stuck_elapsed = 0.0
        if self._stuck_elapsed >= self.config.stuck_timeout_s:
            terminated = True
            terminated_reason = "stuck"
            reward_terms["stuck"] = self.config.reward.stuck_penalty

        # Slow-speed timeout (only after grace period for initial acceleration)
        if not terminated and episode_time > self.config.slow_speed_grace_s:
            if telemetry.speed_kph < self.config.slow_speed_threshold_kph:
                self._slow_speed_elapsed += self.config.step_interval_seconds
            else:
                self._slow_speed_elapsed = 0.0
            if self._slow_speed_elapsed >= self.config.slow_speed_timeout_s:
                terminated = True
                terminated_reason = "slow_timeout"
                reward_terms["slow_timeout"] = self.config.reward.slow_timeout_penalty

        # Success
        if not terminated and distance >= self.config.success_distance_m:
            terminated = True
            terminated_reason = "success"
            reward_terms["success"] = self.config.reward.success_bonus

        truncated = (not terminated) and (self._episode_steps >= self.config.max_episode_steps)
        reward = float(sum(reward_terms.values()))

        obs = self._build_obs(telemetry)
        info = self._build_info(telemetry, reward=reward, reward_terms=reward_terms,
                                distance=distance, terminated_reason=terminated_reason)

        self._last_telemetry = telemetry
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.gamepad.close()
        self.memory_reader.close()
        if self.frame_capture is not None:
            self.frame_capture.close()

    # ── internals ─────────────────────────────────────────────────────────

    def _apply_action(self, action: ControlAction) -> None:
        ab = action.accel_brake
        self.gamepad.send(
            steering=action.steering,
            throttle=max(0.0, ab),
            brake=max(0.0, -ab),
        )

    def _euclidean_distance(self, pos: tuple[float, float, float]) -> float:
        """Euclidean distance from episode start position."""
        if self._start_position is None:
            return 0.0
        dx = pos[0] - self._start_position[0]
        dy = pos[1] - self._start_position[1]
        dz = pos[2] - self._start_position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _build_obs(
        self,
        telemetry: TelemetrySample,
        reset: bool = False,
    ) -> np.ndarray | dict:
        if self.frame_capture is not None:
            if reset:
                self.frame_capture.reset_stack()
            image = self.frame_capture.step()
            return {
                "image": image,
                "telemetry": telemetry.as_observation(),
            }
        return telemetry.as_observation()

    def _build_info(
        self,
        telemetry: TelemetrySample,
        *,
        reward: float,
        reward_terms: dict[str, float],
        distance: float,
        terminated_reason: str,
    ) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "episode_steps": self._episode_steps,
            "speed_kph": telemetry.speed_kph,
            "position": telemetry.position,
            "track_progress": telemetry.track_progress,
            "distance_from_start": distance,
            "terminated_reason": terminated_reason,
            "stuck_elapsed": self._stuck_elapsed,
            "slow_speed_elapsed": self._slow_speed_elapsed,
            "reward": reward,
            "reward_terms": reward_terms,
        }


__all__ = [
    "ControlAction",
    "PCSX2EnvConfig",
    "PCSX2RacerEnv",
    "RewardConfig",
]
