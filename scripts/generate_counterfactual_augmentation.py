#!/usr/bin/env python3
"""
P0.1 — Step 2: Generate counterfactual augmentation images for training set.

For each training image, generate a counterfactual edited toward the
OPPOSITE manufacturer's acquisition parameters. This gives us paired
(real, counterfactual) training data for the augmentation experiment.

Strategy:
    - GE training images  → generate CF toward Siemens IAP
    - Siemens training images → generate CF toward GE IAP

Output:
    data/augmented/
        ge_to_siemens/   <image_filename>.npy  (original GE → CF Siemens)
        siemens_to_ge/   <image_filename>.npy  (original Siemens → CF GE)
        augmented_metadata.csv                  (index of all CF images)

Usage:
    python scripts/generate_counterfactual_augmentation.py \\
        --max_per_class 300 \\
        --edit_timestep 100 \\
        --guidance_scale 3.0
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.counterfactual_editor import CounterfactualEditor

# Label constants (from label_maps)
GE_LABEL      = 0
SIEMENS_LABEL = 1

# Default target IAP for cross-manufacturer CF
# We keep field_strength and echo_time at their training-set medians (neutral)
# and only flip the manufacturer label.
GE_TARGET_FS       = 0   # 1.5T (most common in GE subset)
SIEMENS_TARGET_FS  = 1   # 3.0T (most common in Siemens subset)
NEUTRAL_ECHO_NORM  = 0.5 # midpoint; manufacturer edit should dominate


def get_manufacturer_median_iap(df: pd.DataFrame, manufacturer_label: int) -> dict:
    """Get the median field_strength and echo_time for a manufacturer."""
    sub = df[df["manufacturer_label"] == manufacturer_label]
    fs_mode = int(sub["field_strength_label"].mode()[0])
    echo_median = float(sub["echo_time_norm"].median())
    return {"field_strength_label": fs_mode, "echo_time_norm": echo_median}


def main():
    parser = argparse.ArgumentParser(
        description="P0.1 — Generate counterfactual augmentation images"
    )
    parser.add_argument("--train_csv", type=str,
                        default="data/processed/ax_t1_128/splits/train.csv")
    parser.add_argument("--diffusion_checkpoint", type=str,
                        default="checkpoints/diffusion/ax_t1_128/latest.pt")
    parser.add_argument("--config_path", type=str,
                        default="configs/diffusion.yaml")
    parser.add_argument("--output_dir", type=str,
                        default="data/augmented/")
    parser.add_argument("--max_per_class", type=int, default=300,
                        help="Max CFs to generate per direction (GE→Siemens, Siemens→GE)")
    parser.add_argument("--edit_timestep", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_csv  = PROJECT_ROOT / args.train_csv
    output_dir = PROJECT_ROOT / args.output_dir

    ge_dir      = output_dir / "ge_to_siemens"
    siemens_dir = output_dir / "siemens_to_ge"
    ge_dir.mkdir(parents=True, exist_ok=True)
    siemens_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Counterfactual Augmentation Generator")
    print("=" * 60)
    print(f"  Edit timestep:  {args.edit_timestep}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Max per class:  {args.max_per_class}")

    # Load training data
    df = pd.read_csv(train_csv)
    print(f"\nTraining set: {len(df)} slices from {df['patient_id'].nunique()} patients")

    # Compute target IAPs per manufacturer
    ge_iap      = get_manufacturer_median_iap(df, SIEMENS_LABEL)  # target for GE→Siemens
    siemens_iap = get_manufacturer_median_iap(df, GE_LABEL)       # target for Siemens→GE

    print(f"  GE→Siemens target IAP:  FS={ge_iap['field_strength_label']}, "
          f"echo_norm={ge_iap['echo_time_norm']:.3f}")
    print(f"  Siemens→GE target IAP:  FS={siemens_iap['field_strength_label']}, "
          f"echo_norm={siemens_iap['echo_time_norm']:.3f}")

    # Load model
    print("\nLoading diffusion model...")
    editor = CounterfactualEditor(
        config_path=args.config_path,
        checkpoint_path=args.diffusion_checkpoint,
    )

    # Build generation jobs: [(source_row, target_mfg, target_iap, out_dir)]
    ge_rows      = df[df["manufacturer_label"] == GE_LABEL].sample(
        n=min(args.max_per_class, (df["manufacturer_label"] == GE_LABEL).sum()),
        random_state=args.seed,
    )
    siemens_rows = df[df["manufacturer_label"] == SIEMENS_LABEL].sample(
        n=min(args.max_per_class, (df["manufacturer_label"] == SIEMENS_LABEL).sum()),
        random_state=args.seed,
    )

    jobs = (
        [(row, SIEMENS_LABEL, ge_iap,      ge_dir)      for _, row in ge_rows.iterrows()] +
        [(row, GE_LABEL,      siemens_iap, siemens_dir) for _, row in siemens_rows.iterrows()]
    )

    print(f"\nTotal CFs to generate: {len(jobs)} "
          f"({len(ge_rows)} GE→Siemens + {len(siemens_rows)} Siemens→GE)")

    metadata_rows = []
    success = 0
    fail = 0
    t0 = time.time()

    for idx, (row, target_mfg, target_iap, out_dir) in enumerate(jobs):
        img_name = Path(row["image_path"]).name
        cf_path  = out_dir / img_name

        if cf_path.exists():
            # Already generated — add to metadata
            metadata_rows.append({
                "original_path":      row["image_path"],
                "cf_path":            str(cf_path.relative_to(PROJECT_ROOT)),
                "patient_id":         row["patient_id"],
                "source_manufacturer_label": int(row["manufacturer_label"]),
                "target_manufacturer_label": int(target_mfg),
                "field_strength_label":      int(row["field_strength_label"]),
                "echo_time_norm":            float(row["echo_time_norm"]),
            })
            success += 1
            continue

        try:
            img_01 = np.load(PROJECT_ROOT / row["image_path"]).astype(np.float32)
            img_01 = np.clip(img_01, 0.0, 1.0)

            result = editor.generate_counterfactual(
                image_01=img_01,
                target_manufacturer_label=int(target_mfg),
                target_field_strength_label=int(target_iap["field_strength_label"]),
                target_echo_time_norm=float(target_iap["echo_time_norm"]),
                edit_timestep=args.edit_timestep,
                guidance_scale=args.guidance_scale,
                seed=args.seed + idx,
            )

            np.save(cf_path, result["generated_01"].astype(np.float32))

            metadata_rows.append({
                "original_path":      row["image_path"],
                "cf_path":            str(cf_path.relative_to(PROJECT_ROOT)),
                "patient_id":         row["patient_id"],
                "source_manufacturer_label": int(row["manufacturer_label"]),
                "target_manufacturer_label": int(target_mfg),
                "field_strength_label":      int(row["field_strength_label"]),
                "echo_time_norm":            float(row["echo_time_norm"]),
            })
            success += 1

        except Exception as e:
            fail += 1
            print(f"  [FAIL] {img_name}: {e}")
            continue

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (len(jobs) - idx - 1) / rate
            print(f"  [{idx+1}/{len(jobs)}] ✓{success} ✗{fail} | "
                  f"{rate:.1f} img/s | ETA {remaining/60:.1f} min")

    # Save metadata CSV
    meta_path = output_dir / "augmented_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Success: {success}  |  Failed: {fail}")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Output dir: {output_dir}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
