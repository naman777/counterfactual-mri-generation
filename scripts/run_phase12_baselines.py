#!/usr/bin/env python3
"""
Phase 12 — Step 12.3: Baseline Comparison Runner.

Runs the same test examples through baseline transformations
and computes identical metrics for fair comparison.

Usage:
    python scripts/run_phase12_baselines.py --n_examples 100 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.engine import ExplainabilityEngine
from src.inference.baselines import (
    identity_baseline,
    contrast_shift_baseline,
    gamma_baseline,
    gaussian_noise_baseline,
    histogram_matching_baseline,
)

# Reuse Phase 10 infrastructure
from scripts.run_explainability_batch import (
    build_target_condition,
    select_diverse_examples,
)


# Baseline configurations
BASELINES = {
    "identity": {
        "fn": lambda img, ref, seed: identity_baseline(img),
        "description": "Identity (no change)",
        "needs_reference": False,
    },
    "contrast_shift": {
        "fn": lambda img, ref, seed: contrast_shift_baseline(img, alpha=1.10, beta=0.0),
        "description": "Contrast Shift (α=1.10)",
        "needs_reference": False,
    },
    "gamma": {
        "fn": lambda img, ref, seed: gamma_baseline(img, gamma=0.90),
        "description": "Gamma Correction (γ=0.90)",
        "needs_reference": False,
    },
    "gaussian_noise": {
        "fn": lambda img, ref, seed: gaussian_noise_baseline(img, sigma=0.03, seed=seed),
        "description": "Gaussian Noise (σ=0.03)",
        "needs_reference": False,
    },
    "histogram_matching": {
        "fn": lambda img, ref, seed: histogram_matching_baseline(img, ref),
        "description": "Histogram Matching",
        "needs_reference": True,
    },
}

# Controls to evaluate baselines against
CONTROLS = ["field_strength", "echo_time", "manufacturer"]

# CSV columns
CSV_COLUMNS = [
    "baseline",
    "control",
    "example_id",
    "patient_id",
    "global_ssim",
    "mean_abs_diff",
    "max_abs_diff",
    "hist_l1",
    "hist_intersection",
    "mfg_gain",
    "mfg_hit",
    "fs_gain",
    "fs_hit",
    "echo_improvement",
    "echo_moved_closer",
    "mfg_stable",
    "fs_stable",
]


def find_reference_image(
    test_df: pd.DataFrame,
    target_cond: dict,
    exclude_patient: str,
    control: str,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """
    Find a reference image from the test set that matches the target condition.
    Used for histogram matching baseline.
    """
    candidates = test_df.copy()

    # Filter by target condition
    if control == "field_strength":
        candidates = candidates[
            candidates["field_strength_label"] == target_cond["field_strength_label"]
        ]
    elif control == "manufacturer":
        candidates = candidates[
            candidates["manufacturer_label"] == target_cond["manufacturer_label"]
        ]
    elif control == "echo_time":
        # For echo time, find closest echo_time_norm
        target_echo = target_cond["echo_time_norm"]
        candidates = candidates.copy()
        candidates["echo_dist"] = (candidates["echo_time_norm"] - target_echo).abs()
        candidates = candidates.nsmallest(20, "echo_dist")

    # Exclude same patient
    candidates = candidates[candidates["patient_id"] != exclude_patient]

    if len(candidates) == 0:
        return None

    # Random selection
    selected = candidates.sample(1, random_state=int(rng.integers(0, 2**31)))
    ref_path = str(PROJECT_ROOT / selected.iloc[0]["image_path"])
    try:
        ref = np.load(ref_path).astype(np.float32)
        return np.clip(ref, 0.0, 1.0)
    except Exception:
        return None


def run_baselines(
    test_df: pd.DataFrame,
    engine: ExplainabilityEngine,
    output_dir: Path,
    n_examples: int = 100,
    seed: int = 42,
):
    """Run all baselines on all controls and save results."""

    rng = np.random.default_rng(seed)

    # Use the same example selection as the main evaluation runner
    examples = select_diverse_examples(test_df, n_examples, seed=seed)
    print(f"Selected {len(examples)} examples from {examples['patient_id'].nunique()} patients")
    print(f"(Same selection as main evaluation with seed={seed})")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_csv_rows = []
    all_summaries = {}

    for control in CONTROLS:
        print(f"\n{'='*70}")
        print(f"Control: {control}")
        print(f"{'='*70}")

        control_dir = output_dir / control
        control_dir.mkdir(parents=True, exist_ok=True)

        for baseline_name, baseline_cfg in BASELINES.items():
            print(f"\n  --- Baseline: {baseline_name} ({baseline_cfg['description']}) ---")

            baseline_metrics = []
            csv_rows = []
            success = 0
            fail = 0

            for idx, (_, row) in enumerate(examples.iterrows()):
                example_id = f"example_{idx:03d}"

                try:
                    # Load original image
                    image_path = str(PROJECT_ROOT / row["image_path"])
                    original = np.load(image_path).astype(np.float32)
                    original = np.clip(original, 0.0, 1.0)

                    # Build conditions
                    source_cond, target_cond = build_target_condition(row, control)

                    # Get reference image for histogram matching
                    ref_image = None
                    if baseline_cfg["needs_reference"]:
                        ref_image = find_reference_image(
                            test_df, target_cond, str(row["patient_id"]),
                            control, rng,
                        )
                        if ref_image is None:
                            # Fallback: use original (makes it identity-like)
                            ref_image = original.copy()

                    # Apply baseline
                    baseline_output = baseline_cfg["fn"](
                        original, ref_image, seed + idx
                    )

                    # Compute metrics
                    results = engine.compute_all(
                        original=original,
                        counterfactual=baseline_output,
                        source_condition=source_cond,
                        target_condition=target_cond,
                    )

                    # Build metrics entry
                    metrics_entry = engine.results_to_json_serializable(results)
                    metrics_entry["baseline"] = baseline_name
                    metrics_entry["control"] = control
                    metrics_entry["example_id"] = example_id
                    metrics_entry["patient_id"] = str(row["patient_id"])
                    baseline_metrics.append(metrics_entry)

                    csv_row = {
                        "baseline": baseline_name,
                        "control": control,
                        "example_id": example_id,
                        "patient_id": str(row["patient_id"]),
                        "global_ssim": results["global_ssim"],
                        "mean_abs_diff": results["mean_abs_diff"],
                        "max_abs_diff": results["max_abs_diff"],
                        "hist_l1": results["hist_l1"],
                        "hist_intersection": results["hist_intersection"],
                        "mfg_gain": results["predictor_gain"]["mfg_gain"],
                        "mfg_hit": results["target_hit"]["mfg_hit"],
                        "fs_gain": results["predictor_gain"]["fs_gain"],
                        "fs_hit": results["target_hit"]["fs_hit"],
                        "echo_improvement": results["predictor_gain"]["echo_improvement"],
                        "echo_moved_closer": results["target_hit"]["echo_moved_closer"],
                        "mfg_stable": results["stability"]["mfg_stable"],
                        "fs_stable": results["stability"]["fs_stable"],
                    }
                    csv_rows.append(csv_row)
                    all_csv_rows.append(csv_row)
                    success += 1

                except Exception as e:
                    print(f"    [{example_id}] FAILED: {e}")
                    fail += 1
                    continue

            # Aggregate baseline summary
            if baseline_metrics:
                ssims = [m["global_ssim"] for m in baseline_metrics]
                mads = [m["mean_abs_diff"] for m in baseline_metrics]

                # Extract control-specific gain
                if control == "field_strength":
                    gains = [m["predictor_gain"]["fs_gain"] for m in baseline_metrics]
                    hits = [m["target_hit"]["fs_hit"] for m in baseline_metrics]
                elif control == "echo_time":
                    gains = [m["predictor_gain"]["echo_improvement"] for m in baseline_metrics]
                    hits = [m["target_hit"]["echo_moved_closer"] for m in baseline_metrics]
                else:  # manufacturer
                    gains = [m["predictor_gain"]["mfg_gain"] for m in baseline_metrics]
                    hits = [m["target_hit"]["mfg_hit"] for m in baseline_metrics]

                summary_key = f"{baseline_name}__{control}"
                all_summaries[summary_key] = {
                    "baseline": baseline_name,
                    "control": control,
                    "n_examples": success,
                    "ssim_mean": float(np.mean(ssims)),
                    "ssim_std": float(np.std(ssims)),
                    "mad_mean": float(np.mean(mads)),
                    "mad_std": float(np.std(mads)),
                    "gain_mean": float(np.mean(gains)),
                    "gain_std": float(np.std(gains)),
                    "hit_rate": float(np.mean(hits)),
                }

                print(f"    Results: {success}/{success+fail} | "
                      f"SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f} | "
                      f"MAD={np.mean(mads):.4f} | "
                      f"Hit={np.mean(hits):.2%} | "
                      f"Gain={np.mean(gains):+.4f}")

    # ─── Save combined baseline CSV ──────────────────────────────────
    csv_path = output_dir / "baseline_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\nBaseline results CSV saved: {csv_path}")

    # ─── Save summary JSON ───────────────────────────────────────────
    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"Baseline summary saved: {summary_path}")

    # ─── Save baseline summary CSV ───────────────────────────────────
    summary_csv_path = output_dir / "baseline_summary.csv"
    summary_rows = []
    for key, s in all_summaries.items():
        summary_rows.append(s)

    if summary_rows:
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Baseline summary CSV saved: {summary_csv_path}")

    # Print final summary table
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\n{'Baseline':<22} {'Control':<18} {'SSIM':<14} {'MAD':<10} {'Hit Rate':<10} {'Gain':<10}")
    print("-" * 84)
    for key, s in all_summaries.items():
        print(f"{s['baseline']:<22} {s['control']:<18} "
              f"{s['ssim_mean']:.4f}±{s['ssim_std']:.3f} "
              f"{s['mad_mean']:.4f}     "
              f"{s['hit_rate']:.2%}     "
              f"{s['gain_mean']:+.4f}")

    return all_summaries


def main():
    parser = argparse.ArgumentParser(
        description="Phase 12 — Baseline Comparison Runner"
    )
    parser.add_argument("--test_csv", type=str,
                        default="data/processed/ax_t1_128/splits/test.csv")
    parser.add_argument("--predictor_checkpoint", type=str,
                        default="checkpoints/predictor/ax_t1_128/best.pt")
    parser.add_argument("--label_maps_dir", type=str,
                        default="data/processed/ax_t1_128/label_maps/")
    parser.add_argument("--output_dir", type=str,
                        default="reports/phase12/baselines/")
    parser.add_argument("--n_examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    test_csv = str(PROJECT_ROOT / args.test_csv)
    output_dir = Path(PROJECT_ROOT / args.output_dir)

    print("=" * 70)
    print("Phase 12 — Baseline Comparison Runner")
    print("=" * 70)
    print(f"Test CSV:   {test_csv}")
    print(f"Output dir: {output_dir}")
    print(f"N examples: {args.n_examples}")
    print(f"Seed:       {args.seed}")
    print()

    # Load test data
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} test samples from {df['patient_id'].nunique()} patients")

    # Build explainability engine (no diffusion model needed for baselines)
    print("\nLoading explainability engine...")
    engine = ExplainabilityEngine(
        predictor_checkpoint=args.predictor_checkpoint,
        label_maps_dir=args.label_maps_dir,
    )

    # Run baselines
    run_baselines(
        test_df=df,
        engine=engine,
        output_dir=output_dir,
        n_examples=args.n_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
