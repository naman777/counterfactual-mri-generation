"""EMA-aware adapter and B3-compatible sampler. Does not alter src/ or checkpoints.

The sampling equations and RNG ordering follow scripts/v4_convergence/core.py:sample.
Frames are intermediate noisy states, never simulated progress or saved final images.
"""
import threading
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from api.data import ROOT, B2
from src.inference.counterfactual_editor import CounterfactualEditor
from src.explainability.validation import PredictorValidator

MODEL_PATHS = {
    "converged": ROOT / "checkpoints/diffusion/ax_t1_128_v4/b1_20261016_01/selected_epoch_0110.pt",
    "development": ROOT / "checkpoints/diffusion/ax_t1_128/latest.pt",
}
# All inference is serialised, including CPU classifier probes and model construction.
INFERENCE_LOCK = threading.Lock()
torch.set_num_threads(4)


class EMAEditor(CounterfactualEditor):
    def _load_checkpoint(self):
        ck = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "ema_unet_state_dict" in ck:
            if "ema_conditioning_state_dict" not in ck:
                raise ValueError("Incomplete EMA checkpoint")
            self.unet.load_state_dict(ck["ema_unet_state_dict"])
            self.conditioner.load_state_dict(ck["ema_conditioning_state_dict"])
            weights = "EMA"
        else:
            self.unet.load_state_dict(ck["unet_state_dict"])
            self.conditioner.load_state_dict(ck["conditioning_state_dict"])
            weights = "raw (development checkpoint)"
        self.checkpoint_metadata = dict(epoch=ck.get("epoch"), weights=weights)


@lru_cache(maxsize=2)
def editor(name):
    return EMAEditor(checkpoint_path=str(MODEL_PATHS[name]))


class HeldoutJudge(nn.Module):
    """Exact ResNet-18 architecture used by the frozen held-out checkpoints."""
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.backbone.fc = nn.Identity()
        self.drop = nn.Dropout(.3)
        self.mfg = nn.Linear(512, 2)
        self.fs = nn.Linear(512, 2)

    def forward(self, x):
        features = self.drop(self.backbone((x - .5) / .25))
        return dict(mfg=self.mfg(features), fs=self.fs(features))


@lru_cache(maxsize=2)
def classifiers(legacy=False):
    models = {"Original predictor": PredictorValidator(device="cpu").model}
    for seed in range(3):
        path = B2 / f"training/orientation_robust_seed{seed}/judge_heldout_resnet18_seed{seed}.pt"
        model = HeldoutJudge()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[f"Held-out judge {seed}"] = model.eval()
    if legacy:
        model = HeldoutJudge()
        path = B2 / "training/legacy_comparison_seed0/judge_heldout_resnet18_seed0.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models["Legacy judge 0 (orientation-sensitive)"] = model.eval()
    return models


@torch.inference_mode()
def probabilities(images, legacy=False):
    x = torch.from_numpy(np.stack(images).astype(np.float32)).unsqueeze(1)
    result = []
    for name, model in classifiers(legacy).items():
        p = model(x)
        original = name == "Original predictor"
        result.append(dict(name=name, independent=name.startswith("Held-out"),
                           mfg=p["manufacturer_logits" if original else "mfg"].softmax(1).tolist(),
                           fs=p["field_strength_logits" if original else "fs"].softmax(1).tolist()))
    return result


@torch.inference_mode()
def sample(model, image, mfg, fs, echo, timestep, guidance, seed, progress=None):
    device = model.device
    generator = torch.Generator(device=device).manual_seed(seed)
    def noise():
        return torch.randn((1, 1, 128, 128), device=device, generator=generator)
    m, f, e = model.make_condition_tensors(mfg, fs, echo)
    ce = model.conditioner(m, f, e)
    ue = model.conditioner(m, f, e, force_unconditional=True)
    x = noise()
    clean = torch.from_numpy(image.copy()).to(device).view(1, 1, 128, 128) * 2 - 1
    x = model.scheduler.q_sample(clean, torch.tensor([timestep], device=device), x)
    sched = model.scheduler
    for t in range(timestep, -1, -1):
        ts = torch.tensor([t], device=device)
        if guidance == 0:
            eps = model.unet(x, ts, ue)
        elif guidance == 1:
            eps = model.unet(x, ts, ce)
        else:
            uncond, cond = model.unet(torch.cat([x, x]), torch.cat([ts, ts]), torch.cat([ue, ce])).chunk(2)
            eps = uncond + guidance * (cond - uncond)
        beta, alpha, abar = sched.betas[t], sched.alphas[t], sched.alphas_cumprod[t]
        x = torch.rsqrt(alpha) * (x - beta / torch.sqrt(1 - abar) * eps)
        if t > 0:
            x = x + torch.sqrt(beta) * noise()
        if progress and (t % max(1, timestep // 12) == 0):
            progress(t, ((x.clamp(-1, 1) + 1) / 2).squeeze().cpu().numpy())
    result = ((x.clamp(-1, 1) + 1) / 2).squeeze().cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model output")
    return result
