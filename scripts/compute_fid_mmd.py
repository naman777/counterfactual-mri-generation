#!/usr/bin/env python3
"""
P0.3 — Compute FID and MMD for all three control axes.

FID (Fréchet Inception Distance) and MMD (Maximum Mean Discrepancy)
measure distributional similarity between real target-condition images
and counterfactual images edited toward that condition.

Feature extractor: Project's own SmallCNNBackbone (256-dim) with
pretrained predictor weights — avoids torchvision conflicts and
produces domain-relevant features.

Usage:
    python scripts/compute_fid_mmd.py --all_axes
    python scripts/compute_fid_mmd.py --control field_strength
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from scipy.linalg import sqrtm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictor import SmallCNNBackbone


# ─── Feature Extractor ────────────────────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    """
    Domain-pretrained feature extractor using the project's SmallCNNBackbone.
    Returns 256-dim vectors from 128×128 grayscale MRI slices.
    """

    def __init__(self, device: torch.device,
                 predictor_checkpoint: str = "checkpoints/predictor/ax_t1_128/best.pt"):
        super().__init__()
        self.backbone = SmallCNNBackbone(in_channels=1, feature_dim=256)
        self.device = device

        # Load domain-pretrained weights from predictor checkpoint
        ckpt_path = PROJECT_ROOT / predictor_checkpoint
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            backbone_state = {
                k.replace("backbone.", ""): v
                for k, v in state.items()
                if k.startswith("backbone.")
            }
            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=False)
                print(f"  Loaded backbone weights from: {ckpt_path.name}")
            else:
                print("  No backbone.* keys in checkpoint, using random init")
        else:
            print(f"  Checkpoint not found at {ckpt_path}, using random init")

        self.to(device)
        self.eval()

    @torch.no_grad()
    def extract(self, images_01: list[np.ndarray]) -> np.ndarray:
        """
        Extract 256-dim features from a list of [H, W] float32 images in [0,1].
        Returns np.ndarray of shape [N, 256].
        """
        feats = []
        for img in images_01:
            t = torch.from_numpy(np.clip(img, 0.0, 1.0)).float()
            t = t.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]
            f = self.backbone(t)                              # [1, 256]
            feats.append(f.squeeze().cpu().numpy())
        return np.stack(feats, axis=0)  # [N, 256]


# ─── FID & MMD ────────────────────────────────────────────────────────────────

def compute_fid(real_feats: np.ndarray, cf_feats: np.ndarray) -> float:
    """FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2*(Σ_r Σ_g)^{1/2})"""
    mu_r, mu_g   = real_feats.mean(0), cf_feats.mean(0)
    sig_r, sig_g = np.cov(real_feats, rowvar=False), np.cov(cf_feats, rowvar=False)
    diff = mu_r - mu_g
    sqrt_prod, _ = sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real
    fid = float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * sqrt_prod))
    return fid


def compute_mmd(real_feats: np.ndarray, cf_feats: np.ndarray,
                bandwidth: float = 1.0) -> float:
    """Unbiased MMD² with RBF kernel k(x,y)=exp(-||x-y||²/2σ²)."""
    def rbf(X, Y):
        XX = np.sum(X**2, 1, keepdims=True)
        YY = np.sum(Y**2, 1, keepdims=True)
        return np.exp(-(XX + YY.T - 2 * X @ Y.T) / (2 * bandwidth**2))

    n, m = len(real_feats), len(cf_feats)
    Krr  = rbf(real_feats, real_feats)
    Kgg  = rbf(cf_feats,   cf_feats)
    Krg  = rbf(real_feats, cf_feats)
    mmd2 = (
        (Krr.sum() - np.trace(Krr)) / (n * (n - 1)) +
        (Kgg.sum() - np.trace(Kgg)) / (m * (m - 1)) -
        2.0 * Krg.mean()
    )
    return float(max(mmd2, 0.0))


# ─── Data Loading ─────────────────────────────────────────────────────────────

CONTROL_CONFIG = {
    "field_strength": {
        "results_csv":        "reports/phase12/field_strength/field_strength_results.csv",
        "counterfactuals_dir": "reports/phase12/field_strength/counterfactuals",
    },
    "manufacturer": {
        "results_csv":        "reports/phase12/manufacturer/manufacturer_results.csv",
        "counterfactuals_dir": "reports/phase12/manufacturer/counterfactuals",
    },
    "echo_time": {
        "results_csv":        "reports/phase12/echo_time/echo_time_results.csv",
        "counterfactuals_dir": "reports/phase12/echo_time/counterfactuals",
    },
}


def load_images(control: str, project_root: Path) -> tuple[list, list]:
    """Load real source images and their counterfactuals for one axis."""
    cfg    = CONTROL_CONFIG[control]
    csv_p  = project_root / cfg["results_csv"]
    cf_dir = project_root / cfg["counterfactuals_dir"]

    if not csv_p.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_p}")
    if not cf_dir.exists():
        raise FileNotFoundError(f"Counterfactuals dir not found: {cf_dir}")

    df = pd.read_csv(csv_p)

    real_images, cf_images = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        img_path = project_root / row["image_path"]
        cf_path  = cf_dir / f"example_{i:03d}_counterfactual.npy"
        if img_path.exists() and cf_path.exists():
            real_images.append(np.load(img_path).astype(np.float32))
            cf_images.append(np.load(cf_path).astype(np.float32))

    print(f"  Loaded {len(real_images)} pairs")
    return real_images, cf_images


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_axis(control: str, extractor: CNNFeatureExtractor,
             project_root: Path, output_dir: Path,
             bandwidth: float = 1.0) -> dict:
    print(f"\n{'─'*55}")
    print(f"Control: {control.upper()}")
    print(f"{'─'*55}")

    real_images, cf_images = load_images(control, project_root)

    if len(real_images) < 2:
        return {"control": control, "fid": None, "mmd": None, "error": "no_data"}

    print(f"  Extracting features (real)...")
    real_feats = extractor.extract(real_images)
    print(f"  Extracting features (CF)...")
    cf_feats   = extractor.extract(cf_images)

    fid = compute_fid(real_feats, cf_feats)
    mmd = compute_mmd(real_feats, cf_feats, bandwidth=bandwidth)

    print(f"  FID = {fid:.6f}")
    print(f"  MMD = {mmd:.2e}")

    return {
        "control":          control,
        "n_images":         len(real_images),
        "feature_dim":      int(real_feats.shape[1]),
        "fid":              float(fid),
        "mmd":              float(mmd),
        "kernel_bandwidth": float(bandwidth),
        "feature_extractor": "SmallCNNBackbone (domain-pretrained, 256-dim)",
        "note": (
            "Feature extractor uses the project's own predictor backbone with "
            "Duke Breast MRI domain pretraining. FID and MMD are relative measures: "
            "lower = better distributional alignment of counterfactuals with real images."
        ),
    }


def generate_latex_table(results: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Distributional quality of counterfactual images (FID $\downarrow$, "
        r"MMD $\downarrow$) per control axis, computed using domain-pretrained CNN features.}",
        r"\label{tab:fid_mmd}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Control Axis & FID $\downarrow$ & MMD $\downarrow$ \\",
        r"\midrule",
    ]
    for r in results:
        if r.get("fid") is None:
            continue
        name = r["control"].replace("_", " ").title()
        lines.append(f"  {name} & {r['fid']:.4f} & {r['mmd']:.2e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="P0.3 — FID & MMD")
    parser.add_argument("--control", choices=list(CONTROL_CONFIG.keys()))
    parser.add_argument("--all_axes", action="store_true")
    parser.add_argument("--output_dir", default="reports/phase12/fid_mmd/")
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.all_axes and not args.control:
        parser.error("Provide --control <axis> or --all_axes")

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading feature extractor...")
    extractor = CNNFeatureExtractor(device)
    print("  ✅ Feature extractor ready")

    axes = list(CONTROL_CONFIG.keys()) if args.all_axes else [args.control]
    all_results = []

    for ctrl in axes:
        try:
            result = run_axis(ctrl, extractor, PROJECT_ROOT, out_dir, args.bandwidth)
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping {ctrl}: {e}")
            all_results.append({"control": ctrl, "fid": None, "mmd": None, "error": str(e)})

    # Save JSON
    json_path = out_dir / "fid_mmd_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    # Save LaTeX
    latex_path = out_dir / "fid_mmd_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(generate_latex_table(all_results))
    print(f"LaTeX saved: {latex_path}")

    # Summary
    print(f"\n{'='*55}")
    print("FID / MMD SUMMARY")
    print(f"{'='*55}")
    print(f"{'Control':<20} {'FID':>12} {'MMD':>14}")
    print("-" * 48)
    for r in all_results:
        fid_s = f"{r['fid']:.6f}" if r.get("fid") is not None else "ERROR"
        mmd_s = f"{r['mmd']:.2e}" if r.get("mmd") is not None else "ERROR"
        print(f"{r['control']:<20} {fid_s:>12} {mmd_s:>14}")
    print(f"\n✅ Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
