from typing import Optional

import torch
import torch.nn as nn


def extract_values(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size):
    """
    Extract values for a batch of timesteps and reshape for image broadcasting.

    values: [T]
    timesteps: [B]
    target_shape: [B, C, H, W]
    """
    batch_size = timesteps.shape[0]
    out = values.gather(0, timesteps)
    return out.view(batch_size, *((1,) * (len(target_shape) - 1)))


class DDPMScheduler(nn.Module):
    """
    Minimal DDPM scheduler for training.

    Training objective:
        x0 = clean image
        noise = epsilon
        xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        model predicts epsilon
        loss = MSE(predicted_noise, noise)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
    ):
        super().__init__()

        if prediction_type != "epsilon":
            raise ValueError("Only epsilon prediction is supported in this first version.")

        if beta_schedule != "linear":
            raise ValueError("Only linear beta schedule is supported in this first version.")

        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        betas = torch.linspace(
            beta_start,
            beta_end,
            num_train_timesteps,
            dtype=torch.float32,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Adds noise to clean images x_start at timestep t.

        x_start: [B, C, H, W], usually in [-1, 1]
        timesteps: [B]
        noise: [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = extract_values(
            self.sqrt_alphas_cumprod,
            timesteps,
            x_start.shape,
        )

        sqrt_one_minus_alpha_bar = extract_values(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            x_start.shape,
        )

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise