"""
Multimodal feature extractor for rocm-racer PPO agent.

Combines:
  CNN branch  — processes (4, 96, 96) grayscale frame stack
  MLP branch  — processes (5,) telemetry vector [speed, x, y, z, track_progress]

Both outputs are flattened and concatenated into a single feature vector
consumed by the SB3 PPO actor-critic heads.

Usage:
    from agents.feature_extractor import MultimodalExtractor

    policy_kwargs = dict(
        features_extractor_class=MultimodalExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, ...)
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultimodalExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observation space:
      "image":     Box(0, 255, (N, H, W), uint8)  — grayscale frame stack
      "telemetry": Box(shape=(5,), float32)        — speed + xyz + track_progress

    CNN architecture mirrors the DQN/Atari architecture adapted for 96x96:
      Conv2d(N→32, kernel=8, stride=4) → ReLU
      Conv2d(32→64, kernel=4, stride=2) → ReLU
      Conv2d(64→64, kernel=3, stride=1) → ReLU → Flatten

    MLP:
      Linear(5→64) → ReLU → Linear(64→64) → ReLU

    Fusion:
      Concatenate(CNN_out, MLP_out) → Linear(→features_dim) → ReLU
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
    ) -> None:
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        n_stack, h, w = image_space.shape  # e.g. (4, 96, 96)
        telemetry_dim = observation_space["telemetry"].shape[0]  # 5

        # ── CNN branch ─────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN output dimension by doing a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, n_stack, h, w)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # ── MLP branch ─────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        mlp_out_dim = 64

        # ── Fusion head ────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + mlp_out_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Image: SB3 passes uint8 tensors; normalise to [0, 1]
        image = observations["image"].float() / 255.0
        telemetry = observations["telemetry"].float()

        cnn_features = self.cnn(image)
        mlp_features = self.mlp(telemetry)

        return self.fusion(torch.cat([cnn_features, mlp_features], dim=1))
