#!/usr/bin/env python3
"""
Lightweight residual U-Net for binary foreground segmentation of breast MRI.

Architecture:
    - Input:  [B, 1, 128, 128] grayscale MRI slice
    - Output: [B, 1, 128, 128] sigmoid probability map (foreground probability)
    - ~1.5M parameters

Design choices:
    - Residual blocks in each encoder/decoder stage for stable training
    - Skip connections (classic U-Net style)
    - Instance norm instead of batch norm (works better with small batches)
    - Simple 4-level hierarchy (fits 128x128 input)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: Conv-IN-ReLU-Conv-IN + skip."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class DownBlock(nn.Module):
    """Downsample (stride-2 conv) then residual block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True)
        self.res  = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.down(x)))
        return self.res(x)


class UpBlock(nn.Module):
    """Bilinear upsample + skip concat + residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True)
        self.res  = ResBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm(self.conv(x)))
        return self.res(x)


class SegmentationUNet(nn.Module):
    """
    Lightweight residual U-Net for breast MRI ROI segmentation.

    Input:  [B, 1, 128, 128]
    Output: [B, 1, 128, 128]  (logits, apply sigmoid for probability)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()
        c = base_channels  # 32

        # ── Encoder ──────────────────────────────────────────────
        # stem: 128x128 → 128x128
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True),
            ResBlock(c),
        )

        self.down1 = DownBlock(c,     c * 2)    # 128 → 64,  32→64
        self.down2 = DownBlock(c * 2, c * 4)    # 64  → 32,  64→128
        self.down3 = DownBlock(c * 4, c * 8)    # 32  → 16,  128→256

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ResBlock(c * 8),
            ResBlock(c * 8),
        )

        # ── Decoder ──────────────────────────────────────────────
        self.up3 = UpBlock(c * 8, c * 4, c * 4)   # 16 → 32
        self.up2 = UpBlock(c * 4, c * 2, c * 2)   # 32 → 64
        self.up1 = UpBlock(c * 2, c,     c)        # 64 → 128

        # ── Head ─────────────────────────────────────────────────
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)     # [B, 32,  128, 128]
        s1 = self.down1(s0)   # [B, 64,   64,  64]
        s2 = self.down2(s1)   # [B, 128,  32,  32]
        s3 = self.down3(s2)   # [B, 256,  16,  16]

        b  = self.bottleneck(s3)

        u3 = self.up3(b,  s2)  # [B, 128, 32,  32]
        u2 = self.up2(u3, s1)  # [B,  64, 64,  64]
        u1 = self.up1(u2, s0)  # [B,  32, 128, 128]

        return self.head(u1)   # [B, 1,  128, 128]  logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability map."""
        return torch.sigmoid(self.forward(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SegmentationUNet(base_channels=32)
    dummy = torch.randn(2, 1, 128, 128)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")
