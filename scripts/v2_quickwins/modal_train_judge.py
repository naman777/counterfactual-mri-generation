"""Train an independent ResNet-18 acquisition judge on Modal (GPU).

Usage: python -m modal run scripts/v2_quickwins/modal_train_judge.py --data <splits128.npz> --out <dir>
Heads: manufacturer (2), field strength (2). Class-weighted CE.
Augmentation: hflip, small shift, contrast/brightness, gamma, Gaussian noise sigma~U(0,0.05).
"""
import io
import json
import os
from pathlib import Path

import modal

app = modal.App("capstone-robust-judge")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1", "torchvision==0.20.1", "numpy<2.2", "scikit-learn"
)
vol = modal.Volume.from_name("capstone-judge-data", create_if_missing=True)


def build_model():
    import torch.nn as nn
    import torchvision

    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Identity()

    class Judge(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = m
            self.drop = nn.Dropout(0.3)
            self.mfg = nn.Linear(512, 2)
            self.fs = nn.Linear(512, 2)

        def features(self, x):
            return self.backbone((x - 0.5) / 0.25)

        def forward(self, x):
            f = self.drop(self.features(x))
            return {"mfg": self.mfg(f), "fs": self.fs(f)}

    return Judge()


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def train(seed: int, epochs: int = 40, noise_max: float = 0.05, aug: str = "robust"):
    """aug="robust": flip, shift, contrast, gamma, noise. aug="plain": flip only (like the original predictor)."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import balanced_accuracy_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    d = np.load("/data/splits128.npz")
    dev = "cuda"
    X = {s: torch.from_numpy(d[f"{s}_x"].astype(np.float32)).unsqueeze(1) for s in ["train", "val", "test"]}
    Ym = {s: torch.from_numpy(d[f"{s}_mfg"]) for s in X}
    Yf = {s: torch.from_numpy(d[f"{s}_fs"]) for s in X}

    def cw(y):
        c = torch.bincount(y, minlength=2).float()
        return (c.sum() / (2 * c)).to(dev)

    ce_m = nn.CrossEntropyLoss(weight=cw(Ym["train"]))
    ce_f = nn.CrossEntropyLoss(weight=cw(Yf["train"]))

    def augment(x):
        b = x.shape[0]
        flip = torch.rand(b, device=dev) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        if aug == "plain":
            return x
        # small integer shifts (+-6 px)
        dx, dy = np.random.randint(-6, 7, 2)
        x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(-2, -1))
        alpha = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
        beta = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.05, 0.05)
        x = (x * alpha + beta).clamp(0, 1)
        g = torch.exp(torch.empty(b, 1, 1, 1, device=dev).uniform_(np.log(0.8), np.log(1.25)))
        x = x.clamp_min(1e-6) ** g
        sig = torch.empty(b, 1, 1, 1, device=dev).uniform_(0, noise_max)
        use = (torch.rand(b, 1, 1, 1, device=dev) < 0.7).float()
        x = x + use * sig * torch.randn_like(x)
        return x.clamp(0, 1)

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    bs = 64
    steps = epochs * ((len(X["train"]) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.15)

    @torch.no_grad()
    def predict(x, noise=0.0, gen_seed=0):
        model.eval()
        pm, pf = [], []
        g = torch.Generator(device=dev).manual_seed(gen_seed)
        for i in range(0, len(x), 256):
            xb = x[i:i + 256].to(dev)
            if noise > 0:
                xb = (xb + noise * torch.randn(xb.shape, device=dev, generator=g)).clamp(0, 1)
            o = model(xb)
            pm.append(F.softmax(o["mfg"], 1).cpu())
            pf.append(F.softmax(o["fs"], 1).cpu())
        return torch.cat(pm).numpy(), torch.cat(pf).numpy()

    def scores(split, noise=0.0):
        pm, pf = predict(X[split], noise)
        ym, yf = Ym[split].numpy(), Yf[split].numpy()
        return {
            "mfg_acc": float((pm.argmax(1) == ym).mean()),
            "fs_acc": float((pf.argmax(1) == yf).mean()),
            "mfg_bacc": float(balanced_accuracy_score(ym, pm.argmax(1))),
            "fs_bacc": float(balanced_accuracy_score(yf, pf.argmax(1))),
        }

    best, best_state, hist = -1, None, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X["train"]))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = augment(X["train"][idx].to(dev))
            o = model(xb)
            loss = ce_m(o["mfg"], Ym["train"][idx].to(dev)) + ce_f(o["fs"], Yf["train"][idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        v = scores("val")
        # select on clean + noisy (sigma=0.03) val balanced accuracy
        vn = scores("val", noise=0.03)
        sel = (v["mfg_bacc"] + v["fs_bacc"] + vn["mfg_bacc"] + vn["fs_bacc"]) / 4
        hist.append({"epoch": ep, "train_loss": tot / len(perm), **{f"val_{k}": x for k, x in v.items()},
                     **{f"valnoise03_{k}": x for k, x in vn.items()}, "select": sel})
        print(json.dumps(hist[-1]))
        if sel > best:
            best = sel
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = {"seed": seed, "aug": aug, "best_val_select": best, "history": hist,
           "val": scores("val"), "test": scores("test"),
           "test_noise03": scores("test", 0.03), "test_noise05": scores("test", 0.05)}
    buf = io.BytesIO()
    torch.save(best_state, buf)
    return res, buf.getvalue()


@app.local_entrypoint()
def main(data: str, out: str, seeds: str = "0,1,2", epochs: int = 40, aug: str = "robust"):
    with vol.batch_upload(force=True) as b:
        b.put_file(data, "/splits128.npz")
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    seeds_l = [int(s) for s in seeds.split(",")]
    for res, state in train.starmap([(s, epochs, 0.05, aug) for s in seeds_l]):
        s = res["seed"]
        tag = "" if aug == "robust" else f"_{aug}"
        (outd / f"judge_resnet18{tag}_seed{s}.pt").write_bytes(state)
        (outd / f"judge_resnet18{tag}_seed{s}_train.json").write_text(json.dumps(res, indent=2))
        print("seed", s, "test", res["test"], "noise03", res["test_noise03"])
