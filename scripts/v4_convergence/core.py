"""Numerical B1/B3 primitives. No Modal calls or output side effects at import."""
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.conditioning import AcquisitionConditionEmbedding
from src.models.diffusion_unet import MRIConditionalUNet
from src.models.ddpm_scheduler import DDPMScheduler
from src.models.predictor import AcquisitionPredictor
from scripts.v2_quickwins.judge_model import load_judge

AXES = {"manufacturer": "mfg", "field_strength": "fs"}


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Scientific outputs are append-only: a completed artifact is never replaced.
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(cfg, device):
    c, m = cfg["condition"], cfg["model"]
    conditioner = AcquisitionConditionEmbedding(
        **{k: c[k] for k in ("num_manufacturers", "num_field_strengths", "manufacturer_embedding_dim",
                             "field_strength_embedding_dim", "echo_embedding_dim", "condition_dim")},
        dropout_prob=c["condition_dropout_prob"]).to(device)
    unet = MRIConditionalUNet(**m, condition_dim=c["condition_dim"]).to(device)
    d = cfg["diffusion"]
    sched = DDPMScheduler(num_train_timesteps=d["num_train_timesteps"], beta_start=d["beta_start"],
                          beta_end=d["beta_end"], beta_schedule=d["beta_schedule"],
                          prediction_type=d["prediction_type"]).to(device)
    return conditioner, unet, sched


def load_models(path, device="cuda", raw=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c, u, s = build(ck["config"], device)
    c.load_state_dict(ck.get("raw_conditioning_state_dict", ck["conditioning_state_dict"]) if raw else ck["conditioning_state_dict"])
    u.load_state_dict(ck.get("raw_unet_state_dict", ck["unet_state_dict"]) if raw else ck["unet_state_dict"])
    return c.eval(), u.eval(), s, ck


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@torch.no_grad()
def ema_update(ema, model, decay):
    for k, value in model.state_dict().items():
        dest = ema.state_dict()[k]
        if dest.is_floating_point():
            dest.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            dest.copy_(value)


def load_pack(path, splits=None):
    with np.load(path, allow_pickle=False) as pack:
        return {k: pack[k] for k in pack.files if splits is None or k.split("_", 1)[0] in splits}


def select_indices(data, split, n=100, seed=42, max_per_patient=10):
    # Same algorithm, pandas shuffle and order as select_diverse_examples.
    order = pd.DataFrame({"idx": np.arange(len(data[f"{split}_pid"]))}).sample(frac=1.0, random_state=seed)
    counts, selected = {}, []
    for idx in order.idx:
        pid = str(data[f"{split}_pid"][idx])
        if counts.get(pid, 0) >= max_per_patient:
            continue
        selected.append(int(idx))
        counts[pid] = counts.get(pid, 0) + 1
        if len(selected) == n:
            break
    if len(selected) != n:
        raise ValueError(f"Cannot select {n} {split} examples with patient cap")
    return np.array(selected)


def conditions(data, split, idx):
    return {k: np.asarray(data[f"{split}_{k}"][idx]) for k in ("mfg", "fs", "echo")}


def target(src, axis):
    result = {k: v.copy() for k, v in src.items()}
    result[AXES[axis]] = 1 - result[AXES[axis]]
    return result


@torch.inference_mode()
def sample(c, u, sched, cond, seeds, timestep, guidance, images=None):
    """Full-precision batched ancestral sampler, separate RNG stream per slice.

    Uses the project's beta variance and CFG formula exactly. For editing, the
    first noise draw is q(x_t|x_0). With images=None it starts from pure noise.
    No shortened sampler, autocast, cross-attention or denoising clipping added.
    """
    c.eval()
    u.eval()
    device = next(u.parameters()).device
    b = len(seeds)
    if b != len(cond["mfg"]) or not 0 <= timestep < sched.num_train_timesteps:
        raise ValueError("Invalid sampling dimensions or timestep")
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    def noise():
        return torch.stack([torch.randn((1, 128, 128), device=device, generator=g) for g in generators])
    m = torch.as_tensor(cond["mfg"], device=device, dtype=torch.long)
    f = torch.as_tensor(cond["fs"], device=device, dtype=torch.long)
    e = torch.as_tensor(cond["echo"], device=device, dtype=torch.float32)
    ce = c(m, f, e)
    ue = c(m, f, e, force_unconditional=True)
    x = noise()
    if images is not None:
        clean = torch.as_tensor(np.asarray(images).copy(), device=device, dtype=torch.float32).unsqueeze(1) * 2 - 1
        ts = torch.full((b,), timestep, device=device, dtype=torch.long)
        x = sched.q_sample(clean, ts, x)
    for t in range(timestep, -1, -1):
        ts = torch.full((b,), t, device=device, dtype=torch.long)
        if guidance == 0:
            eps = u(x, ts, ue)
        elif guidance == 1:
            eps = u(x, ts, ce)
        else:
            uncond, conditional = u(torch.cat([x, x]), torch.cat([ts, ts]), torch.cat([ue, ce])).chunk(2)
            eps = uncond + guidance * (conditional - uncond)
        beta, alpha, abar = sched.betas[t], sched.alphas[t], sched.alphas_cumprod[t]
        x = torch.rsqrt(alpha) * (x - beta / torch.sqrt(1 - abar) * eps)
        if t > 0:
            x = x + torch.sqrt(beta) * noise()
    out = ((x.clamp(-1, 1) + 1) / 2).squeeze(1).cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("Nonfinite generated image")
    return out.astype(np.float32)


def judges(asset_dir, device="cuda", robust_only=False):
    asset_dir = Path(asset_dir)
    result = {}
    if not robust_only:
        predictor = AcquisitionPredictor().to(device)
        ck = torch.load(asset_dir / "predictor.pt", map_location="cpu", weights_only=False)
        predictor.load_state_dict(ck["model_state_dict"])
        result["original_predictor"] = predictor.eval()
    for family in (["robust"] if robust_only else ["robust", "plain"]):
        for seed in range(3):
            result[f"{family}{seed}"] = load_judge(asset_dir / f"{family}{seed}.pt", device)
    return result


@torch.inference_mode()
def probabilities(model, images, original=False, batch=128):
    device = next(model.parameters()).device
    result = {"mfg": [], "fs": []}
    for i in range(0, len(images), batch):
        x = torch.as_tensor(np.asarray(images[i:i + batch]).copy(), dtype=torch.float32, device=device).unsqueeze(1)
        pred = model(x)
        for k in result:
            key = ("manufacturer_logits" if k == "mfg" else "field_strength_logits") if original else k
            result[k].append(pred[key].softmax(1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in result.items()}


def gate_conditions(data, n):
    if n % 4:
        raise ValueError("Gate sample count must be a multiple of four")
    out = {k: [] for k in ("mfg", "fs", "echo")}
    for m in (0, 1):
        for f in (0, 1):
            mask = (data["train_mfg"] == m) & (data["train_fs"] == f)
            if not mask.any():
                raise ValueError("Unsupported training condition in gate")
            out["mfg"].extend([m] * (n // 4))
            out["fs"].extend([f] * (n // 4))
            out["echo"].extend([float(np.median(data["train_echo"][mask]))] * (n // 4))
    return {k: np.asarray(v) for k, v in out.items()}


def gate_rank(result):
    return (min(result["mean_mfg_hit"] / 0.9, result["mean_fs_hit"] / 0.8),
            (result["mean_mfg_hit"] + result["mean_fs_hit"]) / 2,
            -result["epoch"], -result["guidance"])


def choose_validation(rows, axis, protocol):
    rule = protocol["validation"]
    rows = [r for r in rows if r["axis"] == axis]
    if not rows:
        raise ValueError("No validation results")
    feasible = [r for r in rows if r["ssim"] >= rule["min_ssim"][axis] and
                r["stability"] >= rule["min_other_axis_stability"]]
    if feasible:
        chosen = max(feasible, key=lambda r: (r["hit"], r["gain"], r["ssim"], -r["t"], -r["w"]))
    else:
        chosen = max(rows, key=lambda r: (r["ssim"], r["stability"], r["hit"], -r["t"], -r["w"]))
    return {**chosen, "feasible": bool(feasible), "selection_split": "val"}


def plateau(gates, epoch):
    if epoch < 100:
        return False
    best_by_epoch = {}
    for r in gates:
        if r["epoch"] > 10:
            best_by_epoch[r["epoch"]] = max(best_by_epoch.get(r["epoch"], -1), gate_rank(r)[0])
    scores = [best_by_epoch[e] for e in sorted(best_by_epoch)]
    return (len(scores) >= 10 and max(scores[-5:]) - min(scores[-5:]) <= .02 and
            max(scores[-5:]) - max(scores[-10:-5]) <= .01)


def loss_plateau(losses):
    if len(losses) < 10 or not np.isfinite(losses).all():
        return False
    old, new = min(losses[-10:-5]), min(losses[-5:])
    return ((old - new) / max(old, 1e-12) <= .01 and
            (max(losses[-5:]) - new) / max(new, 1e-12) <= .03)


def clone_eval(module):
    result = copy.deepcopy(module).eval()
    result.requires_grad_(False)
    return result
