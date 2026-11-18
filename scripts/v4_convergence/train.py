"""Resume the existing DDPM with EMA and immutable, resumable checkpoints."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.v4_convergence.core import clone_eval, cpu_state, ema_update, json_write, load_models, seed_all


def train_block(data, cfg, checkpoint, output, end_epoch, max_steps=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c, u, sched, ck = load_models(checkpoint, raw=True)
    ec, eu = clone_eval(c), clone_eval(u)
    if "ema_unet_state_dict" in ck:
        ec.load_state_dict(ck["ema_conditioning_state_dict"])
        eu.load_state_dict(ck["ema_unet_state_dict"])
    opt = torch.optim.AdamW(list(c.parameters()) + list(u.parameters()),
                           lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    opt.load_state_dict(ck["optimizer_state_dict"])
    for group in opt.param_groups:
        group["lr"] = cfg["training"]["learning_rate"]
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if "scaler_state_dict" in ck:
        scaler.load_state_dict(ck["scaler_state_dict"])
    seed_all(cfg["project"]["seed"])
    if "rng_torch" in ck:
        torch.set_rng_state(ck["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
    bs = cfg["training"]["batch_size"]
    X = torch.from_numpy(data["train_x"]).unsqueeze(1).cuda() * 2 - 1
    labels = {k: torch.as_tensor(data[f"train_{k}"], device="cuda") for k in ("mfg", "fs", "echo")}
    n = len(X)
    start_epoch = int(ck["epoch"])
    step = int(ck["global_step"])
    initial_step = step
    rows, timings = [], []
    start_time = time.perf_counter()
    params = list(c.parameters()) + list(u.parameters())
    for epoch in range(start_epoch + 1, end_epoch + 1):
        c.train()
        u.train()
        perm = torch.randperm(n, device="cuda")
        total, count, skipped = 0.0, 0, 0
        epoch_time = time.perf_counter()
        complete = True
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < bs:
                break  # same drop_last=True as the existing trainer
            before = time.perf_counter()
            x = X[idx]
            t = torch.randint(0, sched.num_train_timesteps, (len(idx),), device="cuda")
            noise = torch.randn_like(x)
            xt = sched.q_sample(x, t, noise)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                emb = c(labels["mfg"][idx], labels["fs"][idx], labels["echo"][idx])
                loss = F.mse_loss(u(xt, t, emb), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch={epoch}, step={step}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(params, cfg["training"]["grad_clip_norm"])
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale:
                ema_update(ec, c, cfg["training"]["ema_decay"])
                ema_update(eu, u, cfg["training"]["ema_decay"])
            else:
                skipped += 1
            value = float(loss)
            total += value * len(idx)
            count += len(idx)
            step += 1
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - before)
            if max_steps is not None and step - initial_step >= max_steps:
                complete = i + 2 * bs > n
                break
        # Fixed validation subset, timesteps and noise; no training RNG consumption.
        val_loss = validate(data, ec, eu, sched)
        row = {"epoch": epoch, "global_step": step, "train_loss": total / count,
               "ema_val_loss": val_loss, "steps": count // bs, "complete_epoch": complete,
               "amp_skips": skipped, "seconds": time.perf_counter() - epoch_time}
        json_write(output / f"epoch_{epoch:04d}.json", row)
        rows.append(row)
        print("EPOCH", row, flush=True)
        if max_steps is not None:
            break
    # Always save raw and EMA separately. Standard inference keys contain EMA.
    path = output / f"epoch_{epoch:04d}.pt"
    if path.exists():
        raise FileExistsError(path)
    result = {"epoch": epoch, "global_step": step, "config": cfg,
              "conditioning_state_dict": cpu_state(ec), "unet_state_dict": cpu_state(eu),
              "ema_conditioning_state_dict": cpu_state(ec), "ema_unet_state_dict": cpu_state(eu),
              "raw_conditioning_state_dict": cpu_state(c), "raw_unet_state_dict": cpu_state(u),
              "optimizer_state_dict": opt.state_dict(), "scaler_state_dict": scaler.state_dict(),
              "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
              "complete_epoch": complete, "smoke": max_steps is not None,
              "best_val_loss": min([r["ema_val_loss"] for r in rows]),
              "parent_checkpoint": str(checkpoint), "initial_global_step": initial_step}
    torch.save(result, path)
    summary = {"checkpoint": str(path), "epoch": epoch, "global_step": step,
               "seconds": time.perf_counter() - start_time, "rows": rows,
               "median_step_seconds_after_warmup": float(np.median(timings[2:] or timings)),
               "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "smoke": max_steps is not None}
    json_write(output / f"block_{start_epoch:04d}_{epoch:04d}.json", summary)
    return summary


@torch.no_grad()
def validate(data, c, u, sched):
    c.eval()
    u.eval()
    idx = np.random.default_rng(913).permutation(len(data["val_x"]))[:256]
    g = torch.Generator(device="cuda").manual_seed(58121)
    total = 0.0
    for start in range(0, len(idx), 32):
        take = idx[start:start + 32]
        x = torch.as_tensor(data["val_x"][take], device="cuda").unsqueeze(1) * 2 - 1
        t = torch.randint(0, sched.num_train_timesteps, (len(take),), device="cuda", generator=g)
        noise = torch.randn(x.shape, device="cuda", generator=g)
        emb = c(*(torch.as_tensor(data[f"val_{k}"][take], device="cuda") for k in ("mfg", "fs", "echo")))
        with torch.autocast("cuda", dtype=torch.float16):
            pred = u(sched.q_sample(x, t, noise), t, emb)
            loss = F.mse_loss(pred, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        total += float(loss) * len(take)
    return total / len(idx)
