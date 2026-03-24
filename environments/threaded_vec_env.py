"""
ThreadedVecEnv — SB3-compatible vectorized environment using threads.

Ideal for I/O-bound environments (PCSX2 + PINE + grim capture) where each
env.step() spends most of its time in sleep / socket reads / subprocess
calls that release the GIL.  Avoids the pickling/serialisation issues of
SubprocVecEnv with UInput gamepad handles.

Usage:
    envs = [env0, env1, env2, env3]  # pre-built PCSX2RacerEnv instances
    vec = ThreadedVecEnv(envs)
    model = PPO("MultiInputPolicy", vec, ...)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)


class ThreadedVecEnv(VecEnv):
    """Vectorized env that steps N gymnasium envs in parallel via threads."""

    def __init__(self, envs: list[gym.Env]) -> None:
        if not envs:
            raise ValueError("Need at least one environment")

        self.envs = envs
        env0 = envs[0]
        super().__init__(
            num_envs=len(envs),
            observation_space=env0.observation_space,
            action_space=env0.action_space,
        )

        self._executor = ThreadPoolExecutor(max_workers=len(envs))
        self._actions: np.ndarray | None = None

    # ── VecEnv interface ───────────────────────────────────────────────────

    def reset(self) -> VecEnvObs:
        """Reset all environments in parallel."""
        futures = [self._executor.submit(env.reset) for env in self.envs]
        results = [f.result() for f in futures]
        obs_list = [r[0] for r in results]
        return self._stack_obs(obs_list)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")

        actions = self._actions
        self._actions = None

        def _step_one(idx: int):
            obs, reward, terminated, truncated, info = self.envs[idx].step(actions[idx])
            done = terminated or truncated
            reset_obs = None
            if done:
                info["terminal_observation"] = obs
                reset_obs, _ = self.envs[idx].reset()
            return obs, reward, terminated, truncated, info, done, reset_obs

        futures = [self._executor.submit(_step_one, i) for i in range(self.num_envs)]
        results = [f.result() for f in futures]

        obs_list = []
        info_list = []
        rewards = np.empty(self.num_envs, dtype=np.float64)
        dones = np.empty(self.num_envs, dtype=bool)

        for i, (obs, reward, term, trunc, info, done, reset_obs) in enumerate(results):
            obs_list.append(reset_obs if done else obs)
            rewards[i] = reward
            dones[i] = done
            info_list.append(info)

        return self._stack_obs(obs_list), rewards, dones, info_list

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        self._executor.shutdown(wait=False)

    def seed(self, seed: int | None = None) -> list[int | None]:
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * self.num_envs

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        target_envs = self._get_target_envs(indices)
        return [getattr(env, method_name)(*method_args, **method_kwargs) for env in target_envs]

    def get_attr(self, attr_name: str, indices=None):
        target_envs = self._get_target_envs(indices)
        return [getattr(env, attr_name) for env in target_envs]

    def set_attr(self, attr_name: str, value, indices=None):
        target_envs = self._get_target_envs(indices)
        for env in target_envs:
            setattr(env, attr_name, value)

    # ── helpers ────────────────────────────────────────────────────────────

    def _get_target_envs(self, indices) -> list[gym.Env]:
        if indices is None:
            return self.envs
        if isinstance(indices, int):
            return [self.envs[indices]]
        return [self.envs[i] for i in indices]

    def _stack_obs(self, obs_list: list) -> VecEnvObs:
        """Stack observations from N envs into VecEnv format."""
        sample = obs_list[0]
        if isinstance(sample, dict):
            return {
                key: np.stack([o[key] for o in obs_list], axis=0)
                for key in sample
            }
        return np.stack(obs_list, axis=0)
