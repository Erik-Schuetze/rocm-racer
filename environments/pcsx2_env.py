from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from memory_readers.nfsu2_memory import NFSU2MemoryReader, TelemetrySample
from memory_readers.virtual_gamepad import VirtualGamepad


@dataclass(frozen=True)
class RewardConfig:
    target_speed_kph: float = 100.0
    speed_weight: float = 2.0
    progress_weight: float = 10.0
    collision_penalty: float = -25.0
    reverse_penalty: float = -15.0
    zero_speed_penalty: float = -5.0
    zero_speed_threshold_kph: float = 1.0


@dataclass(frozen=True)
class PCSX2EnvConfig:
    step_interval_seconds: float = 0.1
    max_episode_steps: int = 2_000
    device: str = "cuda"
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(frozen=True)
class ControlAction:
    # NFS Underground 2 PS2 control scheme (EA Black Box, NTSC-U)
    # ------------------------------------------------------------
    # Digital (test mode):
    #   X       (BTN_SOUTH) = Accelerate
    #   Circle  (BTN_EAST)  = Brake / Reverse
    #   Square  (BTN_WEST)  = Handbrake
    #   Triangle(BTN_NORTH) = Nitrous
    # Analog (RL training via right stick Y, ABS_RY):
    #   throttle [0,1] → right stick UP   (SDL-0/-RightY)
    #   brake    [0,1] → right stick DOWN (SDL-0/+RightY)
    #   VirtualGamepad.send() combines both onto ABS_RY (net = throttle - brake)
    # NOTE: R2/L2 = shift up/down — never used for throttle/brake.
    steering: float   # [-1.0, 1.0]  → ABS_X  left stick horizontal
    throttle: float   # [ 0.0, 1.0]  → ABS_RY right stick up
    brake: float      # [ 0.0, 1.0]  → ABS_RY right stick down


class PCSX2RacerEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        memory_reader: NFSU2MemoryReader | None = None,
        gamepad: VirtualGamepad | None = None,
        config: PCSX2EnvConfig | None = None,
        sleep_fn=time.sleep,
    ) -> None:
        super().__init__()
        self.config = config or PCSX2EnvConfig()
        self.memory_reader = memory_reader or NFSU2MemoryReader()
        self.gamepad = gamepad or VirtualGamepad()
        self.sleep_fn = sleep_fn
        self.device = torch.device(self.config.device)

        self.action_space = spaces.Box(
            low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.asarray(
                [-500.0, -1_000_000.0, -1_000_000.0, -1_000_000.0, -360.0, -360.0, -360.0, 0.0],
                dtype=np.float32,
            ),
            high=np.asarray(
                [500.0, 1_000_000.0, 1_000_000.0, 1_000_000.0, 360.0, 360.0, 360.0, 1_000_000.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self._episode_steps = 0
        self._last_telemetry: TelemetrySample | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self.memory_reader.open()
        self.gamepad.open()
        self.gamepad.center()
        self._episode_steps = 0
        self._last_telemetry = self.memory_reader.read_telemetry()

        observation = self._last_telemetry.as_observation()
        info = self._build_info(self._last_telemetry, reward=0.0, reward_terms={})
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.asarray(action, dtype=np.float32).clip(
            self.action_space.low, self.action_space.high
        )
        control = ControlAction(
            steering=float(clipped_action[0]),
            throttle=float(clipped_action[1]),
            brake=float(clipped_action[2]),
        )

        self._apply_action(control)
        self.sleep_fn(self.config.step_interval_seconds)

        telemetry = self.memory_reader.read_telemetry()
        reward, reward_terms = self._calculate_reward(telemetry)

        self._episode_steps += 1
        terminated = bool(telemetry.wall_collision_flag)
        truncated = self._episode_steps >= self.config.max_episode_steps
        observation = telemetry.as_observation()
        info = self._build_info(telemetry, reward=reward, reward_terms=reward_terms)

        self._last_telemetry = telemetry
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.gamepad.close()
        self.memory_reader.close()

    def observation_tensor(self, telemetry: TelemetrySample | None = None) -> torch.Tensor:
        sample = telemetry or self._last_telemetry
        if sample is None:
            raise RuntimeError("No telemetry is available yet. Call reset() first.")
        return torch.as_tensor(sample.as_observation(), device=self.device)

    def _apply_action(self, action: ControlAction) -> None:
        self.gamepad.send(
            steering=action.steering,
            throttle=action.throttle,
            brake=action.brake,
        )

    def _calculate_reward(self, telemetry: TelemetrySample) -> tuple[float, dict[str, float]]:
        reward_cfg = self.config.reward
        previous_progress = (
            self._last_telemetry.track_progress if self._last_telemetry is not None else telemetry.track_progress
        )
        progress_delta = telemetry.track_progress - previous_progress

        speed_term = reward_cfg.speed_weight * min(
            telemetry.speed_kph / reward_cfg.target_speed_kph,
            1.0,
        )
        progress_term = reward_cfg.progress_weight * progress_delta
        collision_term = (
            reward_cfg.collision_penalty if telemetry.wall_collision_flag else 0.0
        )
        reverse_term = reward_cfg.reverse_penalty if telemetry.reverse_flag else 0.0
        zero_speed_term = (
            reward_cfg.zero_speed_penalty
            if abs(telemetry.speed_kph) <= reward_cfg.zero_speed_threshold_kph
            else 0.0
        )

        reward_terms = {
            "speed": speed_term,
            "progress": progress_term,
            "collision": collision_term,
            "reverse": reverse_term,
            "zero_speed": zero_speed_term,
        }
        return float(sum(reward_terms.values())), reward_terms

    def _build_info(
        self,
        telemetry: TelemetrySample,
        *,
        reward: float,
        reward_terms: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "episode_steps": self._episode_steps,
            "speed_kph": telemetry.speed_kph,
            "position": telemetry.position,
            "rotation": telemetry.rotation,
            "track_progress": telemetry.track_progress,
            "reverse_flag": telemetry.reverse_flag,
            "wall_collision_flag": telemetry.wall_collision_flag,
            "reward": reward,
            "reward_terms": reward_terms,
        }


__all__ = [
    "ControlAction",
    "PCSX2EnvConfig",
    "PCSX2RacerEnv",
    "RewardConfig",
]
