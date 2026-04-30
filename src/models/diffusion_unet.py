import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_group_count(num_channels: int, max_groups: int = 8) -> int:
    """
    Returns the largest valid GroupNorm group count up to max_groups.
    GroupNorm requires num_channels % num_groups == 0.
    """
    for groups in range(max_groups, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal timestep embedding used in diffusion models.
    Input:
        timesteps: [B]
    Output:
        embedding: [B, embedding_dim]
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must have shape [B], got {timesteps.shape}")

        device = timesteps.device
        half_dim = self.embedding_dim // 2

        if half_dim == 0:
            return timesteps.float().unsqueeze(1)

        exponent = -math.log(10000.0) * torch.arange(
            half_dim, device=device, dtype=torch.float32
        )
        exponent = exponent / max(half_dim - 1, 1)

        frequencies = torch.exp(exponent)
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualBlock(nn.Module):
    """
    Residual convolution block with additive time+condition conditioning.

    Input:
        x: [B, in_channels, H, W]
        time_condition: [B, time_condition_dim]

    Output:
        h: [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_condition_dim: int,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            num_groups=get_group_count(in_channels),
            num_channels=in_channels,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_condition_proj = nn.Linear(time_condition_dim, out_channels)

        self.norm2 = nn.GroupNorm(
            num_groups=get_group_count(out_channels),
            num_channels=out_channels,
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        time_condition: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        condition = self.time_condition_proj(self.activation(time_condition))
        h = h + condition[:, :, None, None]

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample(nn.Module):
    """
    Downsamples HxW to H/2 x W/2.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsamples HxW to 2H x 2W.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        )
        return self.conv(x)


class MRIConditionalUNet(nn.Module):
    """
    Lightweight conditional U-Net for DDPM noise prediction.

    Inputs:
        noisy_image: [B, 1, 128, 128]
        timesteps: [B]
        condition_embedding: [B, condition_dim]

    Output:
        predicted_noise: [B, 1, 128, 128]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = None,
        time_embedding_dim: int = 128,
        condition_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4]

        if len(channel_multipliers) != 3:
            raise ValueError(
                "This first U-Net version expects exactly 3 channel multipliers, "
                f"got {channel_multipliers}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_embedding_dim = time_embedding_dim
        self.condition_dim = condition_dim

        c1 = base_channels * channel_multipliers[0]
        c2 = base_channels * channel_multipliers[1]
        c3 = base_channels * channel_multipliers[2]

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        time_condition_dim = time_embedding_dim + condition_dim

        self.input_conv = nn.Conv2d(
            in_channels,
            c1,
            kernel_size=3,
            padding=1,
        )

        # 128 x 128
        self.down1_res1 = ResidualBlock(c1, c1, time_condition_dim, dropout)
        self.down1_res2 = ResidualBlock(c1, c1, time_condition_dim, dropout)
        self.downsample1 = Downsample(c1)

        # 64 x 64
        self.down2_res1 = ResidualBlock(c1, c2, time_condition_dim, dropout)
        self.down2_res2 = ResidualBlock(c2, c2, time_condition_dim, dropout)
        self.downsample2 = Downsample(c2)

        # 32 x 32
        self.down3_res1 = ResidualBlock(c2, c3, time_condition_dim, dropout)
        self.down3_res2 = ResidualBlock(c3, c3, time_condition_dim, dropout)

        # bottleneck
        self.mid_res1 = ResidualBlock(c3, c3, time_condition_dim, dropout)
        self.mid_res2 = ResidualBlock(c3, c3, time_condition_dim, dropout)

        # 32 -> 64
        self.upsample2 = Upsample(c3, c2)
        self.up2_res1 = ResidualBlock(c2 + c2, c2, time_condition_dim, dropout)
        self.up2_res2 = ResidualBlock(c2, c2, time_condition_dim, dropout)

        # 64 -> 128
        self.upsample1 = Upsample(c2, c1)
        self.up1_res1 = ResidualBlock(c1 + c1, c1, time_condition_dim, dropout)
        self.up1_res2 = ResidualBlock(c1, c1, time_condition_dim, dropout)

        self.output_norm = nn.GroupNorm(
            num_groups=get_group_count(c1),
            num_channels=c1,
        )
        self.output_activation = nn.SiLU()
        self.output_conv = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_image.ndim != 4:
            raise ValueError(
                f"noisy_image must have shape [B, C, H, W], got {noisy_image.shape}"
            )

        batch_size = noisy_image.shape[0]

        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size)

        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must have shape [B], got {timesteps.shape}")

        if timesteps.shape[0] != batch_size:
            raise ValueError("timesteps batch size does not match noisy_image batch size.")

        if condition_embedding.ndim != 2:
            raise ValueError(
                "condition_embedding must have shape [B, condition_dim], "
                f"got {condition_embedding.shape}"
            )

        if condition_embedding.shape[0] != batch_size:
            raise ValueError(
                "condition_embedding batch size does not match noisy_image batch size."
            )

        if condition_embedding.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected condition_dim={self.condition_dim}, "
                f"got {condition_embedding.shape[1]}"
            )

        time_embedding = self.time_embedding(timesteps)
        time_condition = torch.cat([time_embedding, condition_embedding], dim=1)

        h = self.input_conv(noisy_image)

        h = self.down1_res1(h, time_condition)
        h = self.down1_res2(h, time_condition)
        skip1 = h

        h = self.downsample1(h)

        h = self.down2_res1(h, time_condition)
        h = self.down2_res2(h, time_condition)
        skip2 = h

        h = self.downsample2(h)

        h = self.down3_res1(h, time_condition)
        h = self.down3_res2(h, time_condition)

        h = self.mid_res1(h, time_condition)
        h = self.mid_res2(h, time_condition)

        h = self.upsample2(h)

        if h.shape[-2:] != skip2.shape[-2:]:
            h = F.interpolate(h, size=skip2.shape[-2:], mode="nearest")

        h = torch.cat([h, skip2], dim=1)
        h = self.up2_res1(h, time_condition)
        h = self.up2_res2(h, time_condition)

        h = self.upsample1(h)

        if h.shape[-2:] != skip1.shape[-2:]:
            h = F.interpolate(h, size=skip1.shape[-2:], mode="nearest")

        h = torch.cat([h, skip1], dim=1)
        h = self.up1_res1(h, time_condition)
        h = self.up1_res2(h, time_condition)

        h = self.output_norm(h)
        h = self.output_activation(h)
        predicted_noise = self.output_conv(h)

        return predicted_noise