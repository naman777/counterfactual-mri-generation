#!/usr/bin/env python3
"""
Phase 12 — Step 12.2: Main Evaluation Runner.

Runs counterfactual editing on N test examples for a given control axis
and saves all metrics, panels, and counterfactual .npy files.

Usage:
    python scripts/run_phase12_evaluation.py \
        --control field_strength \
        --n_examples 100 \
        --seed 42

    Run three times (once per axis):
        --control field_strength  --edit_timestep 100 --guidance_scale 3.0
        --control echo_time       --edit_timestep 100 --guidance_scale 1.0
        --control manufacturer    --edit_timestep 300 --guidance_scale 5.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.engine import ExplainabilityEngine
from src.explainability.panel_generator import generate_publication_panel
from src.inference.counterfactual_editor import CounterfactualEditor

# Reuse existing infrastructure from Phase 10 batch runner
from scripts.run_explainability_batch import (
    CONTROL_CONFIGS,
    build_target_condition,
    select_diverse_examples,
    aggregate_metrics,
)


# ─── CSV summary columns ────────────────────────────────────────────

SUMMARY_COLUMNS = [
    "example_id",
    "patient_id",
    "image_path",
    "control",
    "edit_timestep",
    "guidance_scale",
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
    "generation_time_s",
]


def run_evaluation_for_control(
    control: str,
    test_df: pd.DataFrame,
    editor: CounterfactualEditor,
    engine: ExplainabilityEngine,
    output_dir: Path,
    n_examples: int = 100,
    edit_timestep: int | None = None,
    guidance_scale: float | None = None,
    seed: int = 42,
    save_panels: bool = True,
):
    """
    Run Phase 12 evaluation for a single control axis.

    Returns:
        summary: dict with aggregated metrics
    """
    config = CONTROL_CONFIGS[control]
    if edit_timestep is None:
        edit_timestep = config["edit_timestep"]
    if guidance_scale is None:
        guidance_scale = config["guidance_scale"]

    print(f"\n{'='*70}")
    print(f"Phase 12 Evaluation — {control}")
    print(f"Settings: edit_timestep={edit_timestep}, guidance_scale={guidance_scale}")
    print(f"N examples: {n_examples}, Seed: {seed}")
    print(f"{'='*70}\n")

    # Select diverse examples (same logic as Phase 10 batch runner)
    examples = select_diverse_examples(test_df, n_examples, seed=seed)
    print(f"Selected {len(examples)} examples from {examples['patient_id'].nunique()} patients")

    # Output directories
    control_dir = output_dir / control
    panels_dir = control_dir / "panels"
    metrics_dir = control_dir / "metrics"
    counterfactuals_dir = control_dir / "counterfactuals"

    for d in [panels_dir, metrics_dir, counterfactuals_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    csv_rows = []
    success_count = 0
    fail_count = 0
    total_time = 0.0

    for idx, (_, row) in enumerate(examples.iterrows()):
        example_id = f"example_{idx:03d}"
        print(f"\n[{idx+1}/{len(examples)}] {example_id} | patient={row['patient_id']}")

        try:
            # 1. Load original image
            image_path = str(PROJECT_ROOT / row["image_path"])
            original = np.load(image_path).astype(np.float32)
            original = np.clip(original, 0.0, 1.0)

            # 2. Build conditions
            source_cond, target_cond = build_target_condition(row, control)

            # 3. Generate counterfactual
            t_start = time.time()
            cf_result = editor.generate_counterfactual(
                image_01=original,
                target_manufacturer_label=target_cond["manufacturer_label"],
                target_field_strength_label=target_cond["field_strength_label"],
                target_echo_time_norm=target_cond["echo_time_norm"],
                edit_timestep=edit_timestep,
                guidance_scale=guidance_scale,
                seed=seed + idx,
            )
            counterfactual = cf_result["generated_01"]
            gen_time = time.time() - t_start
            total_time += gen_time

            # 4. Save counterfactual .npy (needed for baseline comparison)
            cf_npy_path = counterfactuals_dir / f"{example_id}_counterfactual.npy"
            np.save(str(cf_npy_path), counterfactual)

            # 5. Compute explainability
            results = engine.compute_all(
                original=original,
                counterfactual=counterfactual,
                source_condition=source_cond,
                target_condition=target_cond,
            )

            # 6. Save panel
            if save_panels:
                panel_path = panels_dir / f"{example_id}_panel"
                generate_publication_panel(
                    original=original,
                    counterfactual=counterfactual,
                    results=results,
                    source_condition=source_cond,
                    target_condition=target_cond,
                    save_path=str(panel_path),
                    title=f"Phase 12 — {config['description']} — {example_id}",
                )

            # 7. Save per-example metrics JSON
            metrics = engine.results_to_json_serializable(results)
            metrics["example_id"] = example_id
            metrics["image_path"] = str(row["image_path"])
            metrics["patient_id"] = str(row["patient_id"])
            metrics["generation_time_s"] = float(gen_time)
            metrics["edit_timestep"] = edit_timestep
            metrics["guidance_scale"] = guidance_scale
            metrics["control"] = control

            metrics_path = metrics_dir / f"{example_id}_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)

            all_metrics.append(metrics)

            # 8. Build CSV row
            csv_row = {
                "example_id": example_id,
                "patient_id": str(row["patient_id"]),
                "image_path": str(row["image_path"]),
                "control": control,
                "edit_timestep": edit_timestep,
                "guidance_scale": guidance_scale,
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
                "generation_time_s": float(gen_time),
            }
            csv_rows.append(csv_row)
            success_count += 1

            # Log progress
            print(f"  SSIM={results['global_ssim']:.4f} | "
                  f"MAD={results['mean_abs_diff']:.4f} | "
                  f"MfgGain={results['predictor_gain']['mfg_gain']:+.4f} | "
                  f"FSGain={results['predictor_gain']['fs_gain']:+.4f} | "
                  f"EchoImp={results['predictor_gain']['echo_improvement']:+.4f} | "
                  f"Time={gen_time:.1f}s")

        except Exception as e:
            print(f"  FAILED: {e}")
            fail_count += 1
            continue

    # ─── Save per-example CSV ────────────────────────────────────────
    csv_path = control_dir / f"{control}_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nPer-example CSV saved: {csv_path}")

    # ─── Aggregate summary ───────────────────────────────────────────
    summary = aggregate_metrics(all_metrics, control)
    summary["n_success"] = success_count
    summary["n_failed"] = fail_count
    summary["n_total"] = len(examples)
    summary["edit_timestep"] = edit_timestep
    summary["guidance_scale"] = guidance_scale
    summary["total_time_s"] = total_time
    summary["mean_time_per_example_s"] = total_time / max(success_count, 1)

    summary_path = control_dir / f"{control}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved: {summary_path}")

    # Print summary
    print(f"\n{'-'*50}")
    print(f"  {control} EVALUATION COMPLETE")
    print(f"  Success: {success_count}/{len(examples)}")
    print(f"  SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"  MAD:  {summary['mad_mean']:.4f} ± {summary['mad_std']:.4f}")
    if control == "field_strength":
        print(f"  FS Hit Rate: {summary['fs_hit_rate']:.2%}")
        print(f"  FS Gain: {summary['fs_gain_mean']:.4f} ± {summary['fs_gain_std']:.4f}")
    elif control == "echo_time":
        print(f"  Echo Closer Rate: {summary['echo_moved_closer_rate']:.2%}")
        print(f"  Echo Improvement: {summary['echo_improvement_mean']:.4f} ± {summary['echo_improvement_std']:.4f}")
    elif control == "manufacturer":
        print(f"  Mfg Hit Rate: {summary['mfg_hit_rate']:.2%}")
        print(f"  Mfg Gain: {summary['mfg_gain_mean']:.4f} ± {summary['mfg_gain_std']:.4f}")
    print(f"  Mfg Stability: {summary['mfg_stability_rate']:.2%}")
    print(f"  FS  Stability: {summary['fs_stability_rate']:.2%}")
    print(f"  Total time: {total_time:.0f}s ({summary['mean_time_per_example_s']:.1f}s/example)")
    print(f"{'-'*50}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Phase 12 — Main Evaluation Runner"
    )
    parser.add_argument(
        "--control", type=str, required=True,
        choices=["field_strength", "echo_time", "manufacturer"],
        help="Control axis to evaluate",
    )
    parser.add_argument("--test_csv", type=str,
                        default="data/processed/ax_t1_128/splits/test.csv")
    parser.add_argument("--diffusion_checkpoint", type=str,
                        default="checkpoints/diffusion/ax_t1_128/latest.pt")
    parser.add_argument("--predictor_checkpoint", type=str,
                        default="checkpoints/predictor/ax_t1_128/best.pt")
    parser.add_argument("--label_maps_dir", type=str,
                        default="data/processed/ax_t1_128/label_maps/")
    parser.add_argument("--config_path", type=str,
                        default="configs/diffusion.yaml")
    parser.add_argument("--output_dir", type=str,
                        default="reports/phase12/")
    parser.add_argument("--n_examples", type=int, default=100)
    parser.add_argument("--edit_timestep", type=int, default=None,
                        help="Override edit timestep (default: use best per control)")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Override guidance scale (default: use best per control)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_panels", action="store_true",
                        help="Skip panel generation (faster)")

    args = parser.parse_args()

    # Resolve paths
    test_csv = str(PROJECT_ROOT / args.test_csv)
    output_dir = Path(PROJECT_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 12 — Main Evaluation Runner")
    print("=" * 70)
    print(f"Control:    {args.control}")
    print(f"Test CSV:   {test_csv}")
    print(f"Output dir: {output_dir}")
    print(f"N examples: {args.n_examples}")
    print(f"Seed:       {args.seed}")
    print()

    # Load test data
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} test samples from {df['patient_id'].nunique()} patients")

    # Build models
    print("\nLoading diffusion model...")
    editor = CounterfactualEditor(
        config_path=args.config_path,
        checkpoint_path=args.diffusion_checkpoint,
    )

    print("\nLoading explainability engine...")
    engine = ExplainabilityEngine(
        predictor_checkpoint=args.predictor_checkpoint,
        label_maps_dir=args.label_maps_dir,
    )

    # Run evaluation
    summary = run_evaluation_for_control(
        control=args.control,
        test_df=df,
        editor=editor,
        engine=engine,
        output_dir=output_dir,
        n_examples=args.n_examples,
        edit_timestep=args.edit_timestep,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        save_panels=not args.no_panels,
    )

    print("\n" + "=" * 70)
    print("PHASE 12 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
