"""Keep the original ancestral update and switch only the scoring rule."""
import math

import torch

from scripts.v14_dropout_audit.shared_residual import parent_response_score


@torch.inference_mode()
def hybrid(conditioner, parent, guided, scheduler, images, manufacturer, field_strength,
           echo, seed, timestep, guidance, switch_step):
    """Use the shared score above switch_step and ordinary CFG from it onward.

    None runs the shared score throughout. A switch at the starting timestep
    runs ordinary CFG throughout. Both use the same ordered noise draws.
    """
    if not 0 <= timestep < scheduler.num_train_timesteps:
        raise ValueError("Timestep outside the diffusion schedule")
    if switch_step is not None and not 0 <= switch_step <= timestep:
        raise ValueError("Switch outside the trajectory")
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Guidance must be finite and nonnegative")
    if any(model.training for model in (conditioner, parent, guided)):
        raise ValueError("Models must be in evaluation mode")
    device = images.device
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = images.shape
    noise = torch.randn(shape, device=device, generator=generator)
    steps = torch.full((len(images),), timestep, device=device, dtype=torch.long)
    noisy = scheduler.q_sample(images, steps, noise)
    condition = conditioner(manufacturer, field_strength, echo)
    unconditional = conditioner(manufacturer, field_strength, echo, force_unconditional=True)
    for step in range(timestep, -1, -1):
        steps = torch.full((len(images),), step, device=device, dtype=torch.long)
        guided_null = guided(noisy, steps, unconditional)
        if switch_step is not None and step <= switch_step:
            conditional = guided(noisy, steps, condition)
            epsilon = guided_null + guidance * (conditional - guided_null)
        elif guidance == 0:
            epsilon = guided_null
        else:
            parent_null = parent(noisy, steps, unconditional)
            conditional = parent(noisy, steps, condition)
            epsilon = parent_response_score(guided_null, conditional, parent_null, guidance)
        beta = scheduler.betas[step]
        alpha = scheduler.alphas[step]
        alpha_bar = scheduler.alphas_cumprod[step]
        noisy = torch.rsqrt(alpha) * (noisy - beta / torch.sqrt(1 - alpha_bar) * epsilon)
        if step:
            noisy = noisy + torch.sqrt(beta) * torch.randn(shape, device=device, generator=generator)
    if not torch.isfinite(noisy).all():
        raise FloatingPointError("Nonfinite reverse trajectory")
    return ((noisy.clamp(-1, 1) + 1) / 2).cpu().numpy().astype("float32")
