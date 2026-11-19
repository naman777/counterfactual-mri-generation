#!/usr/bin/env python3
"""
P0.1 — Step 4: OOD Evaluation of segmentation models.

Evaluates each trained model on:
    - In-distribution (ID) test set  — same manufacturer as training
    - Out-of-distribution (OOD) test set — opposite manufacturer

Metrics:
    - Dice score (foreground)
    - Precision, Recall
    - Statistical significance: Wilcoxon signed-rank test
      (paired test: augmented model vs baseline, per-slice Dice)

Output:
    reports/segmentation/segmentation_results.csv      (per-slice metrics)
    reports/segmentation/segmentation_summary.json     (aggregated metrics + p-values)
    reports/segmentation/segmentation_table.tex        (LaTeX table for paper)

Usage:
    python scripts/evaluate_segmentation_ood.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.segmentation_unet import SegmentationUNet

GE_LABEL      = 0
SIEMENS_LABEL = 1

SETTINGS_META = {
    "ge_only": {
        "train_manufacturer": GE_LABEL,
        "id_manufacturer":    GE_LABEL,
        "ood_manufacturer":   SIEMENS_LABEL,
        "baseline_for":       "ge_augmented",
    },
    "siemens_only": {
        "train_manufacturer": SIEMENS_LABEL,
        "id_manufacturer":    SIEMENS_LABEL,
        "ood_manufacturer":   GE_LABEL,
        "baseline_for":       "siemens_augmented",
    },
    "ge_augmented": {
        "train_manufacturer": GE_LABEL,
        "id_manufacturer":    GE_LABEL,
        "ood_manufacturer":   SIEMENS_LABEL,
        "baseline_for":       None,
    },
    "siemens_augmented": {
        "train_manufacturer": SIEMENS_LABEL,
        "id_manufacturer":    SIEMENS_LABEL,
        "ood_manufacturer":   GE_LABEL,
        "baseline_for":       None,
    },
}


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_dice(pred_binary: np.ndarray, target: np.ndarray) -> float:
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    if union == 0:
        return 1.0 if pred_binary.sum() == 0 else 0.0
    return float(2.0 * intersection / union)


def compute_precision_recall(pred_binary: np.ndarray, target: np.ndarray):
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    precision = float(tp / (tp + fp + 1e-8))
    recall    = float(tp / (tp + fn + 1e-8))
    return precision, recall


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: SegmentationUNet,
    test_df: pd.DataFrame,
    masks_dir: Path,
    manufacturer_filter: int,
    device: torch.device,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Evaluate a trained model on one subset of the test set.
    Returns per-slice metric dicts.
    """
    model.eval()
    sub = test_df[test_df["manufacturer_label"] == manufacturer_filter]

    results = []
    for _, row in sub.iterrows():
        img_path  = PROJECT_ROOT / row["image_path"]
        mask_name = Path(row["image_path"]).name
        mask_path = masks_dir / mask_name

        if not img_path.exists() or not mask_path.exists():
            continue

        img  = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # Skip slices with no foreground in mask
        if mask.sum() == 0:
            continue

        img_t = torch.from_numpy(np.clip(img, 0, 1)).unsqueeze(0).unsqueeze(0).to(device)
        logit = model(img_t)
        prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
        pred  = (prob > threshold).astype(np.float32)

        dice = compute_dice(pred, mask)
        prec, rec = compute_precision_recall(pred, mask)

        results.append({
            "patient_id":          row["patient_id"],
            "image_path":          row["image_path"],
            "manufacturer_label":  int(manufacturer_filter),
            "dice":                dice,
            "precision":           prec,
            "recall":              rec,
        })

    return results


def load_model(ckpt_path: Path, device: torch.device) -> SegmentationUNet:
    model = SegmentationUNet(base_channels=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ─── LaTeX Table ──────────────────────────────────────────────────────────────

def generate_latex_table(summary: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Segmentation Dice scores (mean $\pm$ std) on ID and OOD test sets, "
        r"with and without counterfactual augmentation. "
        r"$^\dagger$: $p < 0.05$ (Wilcoxon signed-rank test vs. baseline).}",
        r"\label{tab:segmentation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Training Setting & ID Dice $\uparrow$ & OOD Dice $\uparrow$ & $\Delta$ OOD & $p$-value \\",
        r"\midrule",
    ]

    for key, s in summary.items():
        id_dice  = s.get("id_dice_mean",  0)
        id_std   = s.get("id_dice_std",   0)
        ood_dice = s.get("ood_dice_mean", 0)
        ood_std  = s.get("ood_dice_std",  0)
        delta    = s.get("ood_delta",     None)
        pval     = s.get("wilcoxon_p",    None)

        name_map = {
            "ge_only":           r"GE only (baseline)",
            "siemens_only":      r"Siemens only (baseline)",
            "ge_augmented":      r"GE + CF$\rightarrow$Siemens",
            "siemens_augmented": r"Siemens + CF$\rightarrow$GE",
        }
        name = name_map.get(key, key)

        delta_str = f"{delta:+.3f}" if delta is not None else "---"
        if pval is not None:
            sig = r"$^\dagger$" if pval < 0.05 else ""
            pval_str = f"{pval:.3f}{sig}"
        else:
            pval_str = "---"

        lines.append(
            f"  {name} & {id_dice:.3f}$\\pm${id_std:.3f} & "
            f"{ood_dice:.3f}$\\pm${ood_std:.3f} & "
            f"{delta_str} & {pval_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P0.1 — OOD Segmentation Evaluation")
    parser.add_argument("--test_csv",    default="data/processed/ax_t1_128/splits/test.csv")
    parser.add_argument("--masks_dir",   default="data/processed/ax_t1_128/seg_masks/")
    parser.add_argument("--ckpt_dir",    default="checkpoints/segmentation/")
    parser.add_argument("--output_dir",  default="reports/segmentation/")
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    test_csv  = PROJECT_ROOT / args.test_csv
    masks_dir = PROJECT_ROOT / args.masks_dir
    ckpt_dir  = PROJECT_ROOT / args.ckpt_dir
    out_dir   = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_csv)
    print(f"Test set: {len(test_df)} slices from {test_df['patient_id'].nunique()} patients")

    all_rows  = []
    setting_results = {}

    for setting, cfg in SETTINGS_META.items():
        ckpt_path = ckpt_dir / f"seg_{setting}_best.pt"
        if not ckpt_path.exists():
            print(f"\n  ⚠ Checkpoint not found for {setting}: {ckpt_path}")
            print(f"    Run train_segmentation.py --setting {setting} first.")
            continue

        print(f"\n{'─'*55}")
        print(f"Evaluating: {setting}")
        print(f"{'─'*55}")
        model = load_model(ckpt_path, device)

        # ID evaluation
        id_results  = evaluate_model(
            model, test_df, masks_dir,
            manufacturer_filter=cfg["id_manufacturer"],
            device=device,
            threshold=args.threshold,
        )
        # OOD evaluation
        ood_results = evaluate_model(
            model, test_df, masks_dir,
            manufacturer_filter=cfg["ood_manufacturer"],
            device=device,
            threshold=args.threshold,
        )

        id_dices  = [r["dice"] for r in id_results]
        ood_dices = [r["dice"] for r in ood_results]

        id_mean  = float(np.mean(id_dices))  if id_dices  else 0.0
        id_std   = float(np.std(id_dices))   if id_dices  else 0.0
        ood_mean = float(np.mean(ood_dices)) if ood_dices else 0.0
        ood_std  = float(np.std(ood_dices))  if ood_dices else 0.0

        print(f"  ID  Dice: {id_mean:.4f} ± {id_std:.4f}  (n={len(id_dices)})")
        print(f"  OOD Dice: {ood_mean:.4f} ± {ood_std:.4f}  (n={len(ood_dices)})")

        # Tag results and accumulate
        for r in id_results:
            r["setting"] = setting
            r["split"]   = "id"
        for r in ood_results:
            r["setting"] = setting
            r["split"]   = "ood"

        all_rows.extend(id_results)
        all_rows.extend(ood_results)

        setting_results[setting] = {
            "id_dice_mean":  id_mean,
            "id_dice_std":   id_std,
            "id_n":          len(id_dices),
            "ood_dice_mean": ood_mean,
            "ood_dice_std":  ood_std,
            "ood_n":         len(ood_dices),
            "id_dices":      id_dices,
            "ood_dices":     ood_dices,
        }

    # ── Wilcoxon signed-rank tests ──────────────────────────────────────────
    # Compare augmented vs baseline OOD Dice (paired on slice index)
    comparison_pairs = [
        ("ge_augmented",      "ge_only"),
        ("siemens_augmented", "siemens_only"),
    ]

    for aug_key, base_key in comparison_pairs:
        if aug_key not in setting_results or base_key not in setting_results:
            continue

        aug_ood  = setting_results[aug_key]["ood_dices"]
        base_ood = setting_results[base_key]["ood_dices"]

        n = min(len(aug_ood), len(base_ood))
        if n < 5:
            print(f"\n  ⚠ Not enough pairs for Wilcoxon ({aug_key} vs {base_key}): n={n}")
            continue

        aug_arr  = np.array(aug_ood[:n])
        base_arr = np.array(base_ood[:n])
        diff = aug_arr - base_arr

        if np.all(diff == 0):
            p_val = 1.0
        else:
            _, p_val = wilcoxon(aug_arr, base_arr, alternative="greater")

        delta = float(np.mean(aug_ood) - np.mean(base_ood))
        setting_results[aug_key]["ood_delta"]   = delta
        setting_results[aug_key]["wilcoxon_p"]  = float(p_val)
        setting_results[aug_key]["wilcoxon_n"]  = n

        print(f"\n  Wilcoxon ({aug_key} vs {base_key}):")
        print(f"    OOD Δ Dice = {delta:+.4f}")
        print(f"    p-value    = {p_val:.4f} {'✅ significant' if p_val < 0.05 else '(not significant)'}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_csv = out_dir / "segmentation_results.csv"
    pd.DataFrame(all_rows).to_csv(results_csv, index=False)
    print(f"\nPer-slice results saved: {results_csv}")

    # Remove raw dice lists before saving JSON (too large)
    summary_clean = {}
    for k, v in setting_results.items():
        summary_clean[k] = {kk: vv for kk, vv in v.items()
                            if kk not in ("id_dices", "ood_dices")}

    summary_json = out_dir / "segmentation_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary_clean, f, indent=2)
    print(f"Summary JSON saved: {summary_json}")

    # LaTeX table
    latex = generate_latex_table(summary_clean)
    latex_path = out_dir / "segmentation_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved: {latex_path}")

    # Print console summary
    print(f"\n{'='*60}")
    print("SEGMENTATION OOD EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Setting':<25} {'ID Dice':>10} {'OOD Dice':>10} {'Δ OOD':>8} {'p-val':>8}")
    print("-" * 65)
    for key, s in summary_clean.items():
        delta_str = f"{s.get('ood_delta', float('nan')):+.4f}" \
                    if "ood_delta" in s else "   ---"
        pval_str  = f"{s.get('wilcoxon_p', float('nan')):.4f}" \
                    if "wilcoxon_p" in s else "   ---"
        print(f"{key:<25} {s.get('id_dice_mean',0):>10.4f} "
              f"{s.get('ood_dice_mean',0):>10.4f} "
              f"{delta_str:>8} {pval_str:>8}")


if __name__ == "__main__":
    main()
