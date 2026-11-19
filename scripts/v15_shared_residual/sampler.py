"""Ancestral sampling with the guide residual shared across acquisition labels."""
import math

import torch

from scripts.v14_dropout_audit.shared_residual import parent_response_score


@torch.inference_mode()
def sample(conditioner, parent, guided, scheduler, images, manufacturer,
           field_strength, echo, seed, timestep, guidance):
    """Mirror the established reverse kernel and draw order with a new score.

    The caller binds the guide to its source and reuses it for both requests.
    Passing guided=None uses the frozen parent directly.
    """
    if not 0 <= timestep < scheduler.num_train_timesteps:
        raise ValueError("Timestep outside the diffusion schedule")
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Guidance must be finite and nonnegative")
    if conditioner.training or parent.training or (guided is not None and guided.training):
        raise ValueError("Sampler requires evaluation mode")

    device = images.device
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(images.shape, device=device, generator=generator)
    steps = torch.full((len(images),), timestep, device=device, dtype=torch.long)
    noisy = scheduler.q_sample(images, steps, noise)
    condition = conditioner(manufacturer, field_strength, echo)
    unconditional = conditioner(manufacturer, field_strength, echo, force_unconditional=True)

    for step in range(timestep, -1, -1):
        steps = torch.full((len(images),), step, device=device, dtype=torch.long)
        if guided is None:
            parent_null = parent(noisy, steps, unconditional)
            parent_label = parent(noisy, steps, condition)
            epsilon = parent_null + guidance * (parent_label - parent_null)
        else:
            guided_null = guided(noisy, steps, unconditional)
            if guidance == 0:
                epsilon = guided_null
            else:
                parent_null = parent(noisy, steps, unconditional)
                parent_label = parent(noisy, steps, condition)
                epsilon = parent_response_score(guided_null, parent_label, parent_null, guidance)

        beta = scheduler.betas[step]
        alpha = scheduler.alphas[step]
        alpha_bar = scheduler.alphas_cumprod[step]
        noisy = torch.rsqrt(alpha) * (noisy - beta / torch.sqrt(1 - alpha_bar) * epsilon)
        if step:
            noise = torch.randn(images.shape, device=device, generator=generator)
            noisy = noisy + torch.sqrt(beta) * noise
    if not torch.isfinite(noisy).all():
        raise FloatingPointError("Nonfinite reverse trajectory")
    return ((noisy.clamp(-1, 1) + 1) / 2).cpu().numpy().astype("float32")
