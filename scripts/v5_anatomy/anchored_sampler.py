"""Experimental coarse-source anchoring of x0 estimates; no learned weights.

This is an unvalidated sampling hypothesis, not a preservation guarantee.
The zero-strength path calls the original sampler without numerical changes.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scripts.v4_convergence.core import sample as reference_sample


def lowpass(value, sigma=2.):
    radius = int(math.ceil(3*sigma))
    if sigma <= 0 or 2*radius >= min(value.shape[-2:]):
        raise ValueError('Invalid low-pass sigma for image size')
    positions = torch.arange(-radius, radius+1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-.5*(positions/sigma)**2)
    kernel = kernel/kernel.sum()
    kernel2d = (kernel[:, None]*kernel[None, :])[None, None]
    return F.conv2d(F.pad(value, (radius, radius, radius, radius), mode='reflect'), kernel2d)


@torch.inference_mode()
def sample_anchored(c, u, sched, cond, seeds, timestep, guidance, images,
                    anchor_strength=0., sigma=2.):
    if not np.isfinite(anchor_strength) or not 0 <= anchor_strength <= 1:
        raise ValueError('anchor_strength must be in [0,1]')
    images = np.asarray(images, dtype=np.float32)
    n = len(seeds)
    if images.shape != (n, 128, 128) or not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('Expected finite source images [N,128,128] in [0,1]')
    if any(np.asarray(cond[k]).shape != (n,) for k in ['mfg', 'fs', 'echo']):
        raise ValueError('All conditions must align with images and seeds')
    if not 0 <= timestep < sched.num_train_timesteps or not np.isfinite(guidance) or guidance < 0:
        raise ValueError('Invalid timestep or guidance')
    if anchor_strength == 0:
        return reference_sample(c, u, sched, cond, seeds, timestep, guidance, images)
    c.eval(); u.eval()
    device = next(u.parameters()).device
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    def noise():
        return torch.stack([torch.randn((1,128,128), device=device, generator=g) for g in generators])
    ce = c(*(torch.as_tensor(cond[k], device=device) for k in ['mfg', 'fs', 'echo']))
    ue = c(*(torch.as_tensor(cond[k], device=device) for k in ['mfg', 'fs', 'echo']), force_unconditional=True)
    clean = torch.as_tensor(images.copy(), device=device).unsqueeze(1)*2-1
    coarse_source = lowpass(clean, sigma)
    x = sched.q_sample(clean, torch.full((n,), timestep, device=device), noise())
    for t in range(timestep, -1, -1):
        ts = torch.full((n,), t, device=device, dtype=torch.long)
        if guidance == 0:
            eps = u(x, ts, ue)
        elif guidance == 1:
            eps = u(x, ts, ce)
        else:
            uncond, conditional = u(torch.cat([x,x]), torch.cat([ts,ts]), torch.cat([ue,ce])).chunk(2)
            eps = uncond + guidance*(conditional-uncond)
        beta, alpha, abar = sched.betas[t], sched.alphas[t], sched.alphas_cumprod[t]
        mean = torch.rsqrt(alpha)*(x-beta/torch.sqrt(1-abar)*eps)
        x0 = (x-torch.sqrt(1-abar)*eps)/torch.sqrt(abar)
        abar_previous = sched.alphas_cumprod[t-1] if t else torch.ones_like(abar)
        # Correct only the coarse component of x0. The DDPM posterior's x0
        # coefficient propagates that correction into the reverse mean.
        # Ramp late in the chain; preserve the ancestral RNG ordering exactly.
        weight = anchor_strength*(1-t/(timestep+1))
        correction = weight*(coarse_source-lowpass(x0, sigma))
        mean = mean + beta*torch.sqrt(abar_previous)/(1-abar)*correction
        x = mean + torch.sqrt(beta)*noise() if t else mean
    result = ((x.clamp(-1,1)+1)/2).squeeze(1).cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError('Non-finite anchored output')
    return result
