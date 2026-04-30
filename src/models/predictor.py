#!/usr/bin/env python3
"""
Multi-task acquisition-parameter predictor.

Input:
- MRI slice tensor: [B, 1, H, W]

Outputs:
- manufacturer_logits: [B, num_manufacturers]
- field_strength_logits: [B, num_field_strengths]
- echo_time_norm: [B]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SmallCNNBackbone(nn.Module):
    """
    Small CNN backbone for the first predictor baseline.

    This is intentionally simple so you can debug the complete training pipeline
    before moving to ResNet18 or a larger architecture.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 256) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.out_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x


class AcquisitionPredictor(nn.Module):
    """
    Multi-task predictor with a shared image backbone and three output heads.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_manufacturers: int = 2,
        num_field_strengths: int = 2,
        feature_dim: int = 256,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.backbone = SmallCNNBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )

        self.shared = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.manufacturer_head = nn.Linear(feature_dim, num_manufacturers)
        self.field_strength_head = nn.Linear(feature_dim, num_field_strengths)
        self.echo_time_head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.shared(features)

        manufacturer_logits = self.manufacturer_head(features)
        field_strength_logits = self.field_strength_head(features)
        echo_time_norm = self.echo_time_head(features).squeeze(1)

        return {
            "manufacturer_logits": manufacturer_logits,
            "field_strength_logits": field_strength_logits,
            "echo_time_norm": echo_time_norm,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
