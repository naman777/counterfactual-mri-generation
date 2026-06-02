#!/usr/bin/env python3
"""
Phase 10 — Step 10.3: Batch Explainability Panel Runner.

Runs the ExplainabilityEngine on test examples for each control axis,
saves panels + JSON metrics per example, and produces aggregated summaries.

Usage:
    python scripts/run_explainability_batch.py \
        --test_csv data/processed/ax_t1_128/splits/test.csv \
        --diffusion_checkpoint checkpoints/diffusion/ax_t1_128/latest.pt \
        --predictor_checkpoint checkpoints/predictor/ax_t1_128/best.pt \
        --label_maps_dir data/processed/ax_t1_128/label_maps/ \
        --output_dir reports/explainability/batch/ \
        --n_examples 50 \
        --seed 42
"""

from __future__ import annotations

import argparse
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


# ─── Control axis configurations ──────────────────────────────────


# Best settings from Phase 9
CONTROL_CONFIGS = {
    "field_strength": {
        "edit_timestep": 100,
        "guidance_scale": 3.0,
        "description": "Field Strength Editing (1.5T ↔ 3.0T)",
    },
    "echo_time": {
        "edit_timestep": 100,
        "guidance_scale": 1.0,
        "description": "EchoTime Editing (continuous)",
    },
    "manufacturer": {
        "edit_timestep": 300,
        "guidance_scale": 5.0,
        "description": "Manufacturer Editing (GE ↔ SIEMENS)",
    },
}


def build_target_condition(row: pd.Series, control: str) -> dict:
    """
    Build target condition by flipping the specified control axis
    while keeping everything else the same.
    """
    source = {
        "manufacturer_label": int(row["manufacturer_label"]),
        "field_strength_label": int(row["field_strength_label"]),
        "echo_time_norm": float(row["echo_time_norm"]),
    }

    target = source.copy()

    if control == "field_strength":
        # Flip field strength
        if source["field_strength_label"] == 0:
            target["field_strength_label"] = 1  # 1.5T -> 3.0T
        else:
            target["field_strength_label"] = 0  # 3.0T -> 1.5T

    elif control == "echo_time":
        # Move to opposite end of echo time range
        if source["echo_time_norm"] > 0.5:
            target["echo_time_norm"] = 0.15
        else:
            target["echo_time_norm"] = 0.85

    elif control == "manufacturer":
        # Flip manufacturer
        if source["manufacturer_label"] == 0:
            target["manufacturer_label"] = 1  # GE -> SIEMENS
        else:
            target["manufacturer_label"] = 0  # SIEMENS -> GE

    return source, target


def select_diverse_examples(
    df: pd.DataFrame,
    n_examples: int,
    max_per_patient: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Select diverse examples ensuring patient-level diversity.
    """
    rng = np.random.default_rng(seed)

    # Shuffle
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    selected = []
    patient_counts = {}

    for _, row in df_shuffled.iterrows():
        pid = row["patient_id"]
        if patient_counts.get(pid, 0) >= max_per_patient:
            continue
        selected.append(row)
        patient_counts[pid] = patient_counts.get(pid, 0) + 1
        if len(selected) >= n_examples:
            break

    return pd.DataFrame(selected).reset_index(drop=True)


def run_batch_for_control(
    control: str,
    test_df: pd.DataFrame,
    editor: CounterfactualEditor,
    engine: ExplainabilityEngine,
    output_dir: Path,
    n_examples: int = 50,
    seed: int = 42,
):
    """
    Run batch explainability for a single control axis.
    """
    config = CONTROL_CONFIGS[control]
    edit_timestep = config["edit_timestep"]
    guidance_scale = config["guidance_scale"]

    print(f"\n{'='*70}")
    print(f"Control: {control}")
    print(f"Settings: edit_timestep={edit_timestep}, guidance_scale={guidance_scale}")
    print(f"{'='*70}\n")

    # Select examples
    examples = select_diverse_examples(test_df, n_examples, seed=seed)
    print(f"Selected {len(examples)} examples from {examples['patient_id'].nunique()} patients")

    # Output paths
    panels_dir = output_dir / "panels" / control
    panels_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    success_count = 0
    fail_count = 0

    for idx, (_, row) in enumerate(examples.iterrows()):
        example_id = f"example_{idx:03d}"
        print(f"\n[{idx+1}/{len(examples)}] Processing {example_id}...")

        try:
            # Load image
            image_path = str(PROJECT_ROOT / row["image_path"])
            original = np.load(image_path).astype(np.float32)
            original = np.clip(original, 0.0, 1.0)

            # Build conditions
            source_cond, target_cond = build_target_condition(row, control)

            # Generate counterfactual
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

            # Compute explainability
            results = engine.compute_all(
                original=original,
                counterfactual=counterfactual,
                source_condition=source_cond,
                target_condition=target_cond,
            )

            # Generate panel
            panel_path = panels_dir / f"{example_id}_panel"
            generate_publication_panel(
                original=original,
                counterfactual=counterfactual,
                results=results,
                source_condition=source_cond,
                target_condition=target_cond,
                save_path=str(panel_path),
                title=f"{config['description']} — {example_id}",
            )

            # Save metrics JSON
            metrics = engine.results_to_json_serializable(results)
            metrics["example_id"] = example_id
            metrics["image_path"] = str(row["image_path"])
            metrics["patient_id"] = str(row["patient_id"])
            metrics["generation_time_s"] = float(gen_time)
            metrics["edit_timestep"] = edit_timestep
            metrics["guidance_scale"] = guidance_scale
            metrics["control"] = control

            metrics_path = panels_dir / f"{example_id}_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)

            all_metrics.append(metrics)
            success_count += 1

            # Log progress
            print(f"  SSIM={results['global_ssim']:.4f} | "
                  f"MAD={results['mean_abs_diff']:.4f} | "
                  f"MfgHit={results['target_hit']['mfg_hit']} | "
                  f"FSHit={results['target_hit']['fs_hit']} | "
                  f"EchoCloser={results['target_hit']['echo_moved_closer']} | "
                  f"Time={gen_time:.1f}s")

        except Exception as e:
            print(f"  FAILED: {e}")
            fail_count += 1
            continue

    # Aggregate summary
    summary = aggregate_metrics(all_metrics, control)
    summary["n_success"] = success_count
    summary["n_failed"] = fail_count
    summary["n_total"] = len(examples)
    summary["edit_timestep"] = edit_timestep
    summary["guidance_scale"] = guidance_scale

    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{control}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    return summary


def aggregate_metrics(metrics_list: list, control: str) -> dict:
    """Aggregate metrics across all examples for a control axis."""
    if not metrics_list:
        return {"control": control, "n_examples": 0}

    # Collect scalar metrics
    ssims = [m["global_ssim"] for m in metrics_list]
    mads = [m["mean_abs_diff"] for m in metrics_list]
    max_diffs = [m["max_abs_diff"] for m in metrics_list]
    hist_l1s = [m["hist_l1"] for m in metrics_list]
    hist_ints = [m["hist_intersection"] for m in metrics_list]

    # Predictor gain metrics
    mfg_gains = [m["predictor_gain"]["mfg_gain"] for m in metrics_list]
    fs_gains = [m["predictor_gain"]["fs_gain"] for m in metrics_list]
    echo_improvements = [m["predictor_gain"]["echo_improvement"] for m in metrics_list]

    # Hit rates
    mfg_hits = [m["target_hit"]["mfg_hit"] for m in metrics_list]
    fs_hits = [m["target_hit"]["fs_hit"] for m in metrics_list]
    echo_closer = [m["target_hit"]["echo_moved_closer"] for m in metrics_list]

    # Stability
    mfg_stable = [m["stability"]["mfg_stable"] for m in metrics_list]
    fs_stable = [m["stability"]["fs_stable"] for m in metrics_list]

    summary = {
        "control": control,
        "n_examples": len(metrics_list),
        # SSIM
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "ssim_median": float(np.median(ssims)),
        "ssim_min": float(np.min(ssims)),
        "ssim_max": float(np.max(ssims)),
        # MAD
        "mad_mean": float(np.mean(mads)),
        "mad_std": float(np.std(mads)),
        # Max diff
        "max_diff_mean": float(np.mean(max_diffs)),
        # Histogram
        "hist_l1_mean": float(np.mean(hist_l1s)),
        "hist_l1_std": float(np.std(hist_l1s)),
        "hist_intersection_mean": float(np.mean(hist_ints)),
        # Manufacturer gain
        "mfg_gain_mean": float(np.mean(mfg_gains)),
        "mfg_gain_std": float(np.std(mfg_gains)),
        "mfg_hit_rate": float(np.mean(mfg_hits)),
        # Field strength gain
        "fs_gain_mean": float(np.mean(fs_gains)),
        "fs_gain_std": float(np.std(fs_gains)),
        "fs_hit_rate": float(np.mean(fs_hits)),
        # Echo improvement
        "echo_improvement_mean": float(np.mean(echo_improvements)),
        "echo_improvement_std": float(np.std(echo_improvements)),
        "echo_moved_closer_rate": float(np.mean(echo_closer)),
        # Stability
        "mfg_stability_rate": float(np.mean(mfg_stable)),
        "fs_stability_rate": float(np.mean(fs_stable)),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 10 — Batch Explainability Runner")
    parser.add_argument("--test_csv", type=str,
                        default="data/processed/ax_t1_128/splits/test.csv")
    parser.add_argument("--diffusion_checkpoint", type=str,
                        default="checkpoints/diffusion/ax_t1_128/latest.pt")
    parser.add_argument("--predictor_checkpoint", type=str,
                        default="checkpoints/predictor/ax_t1_128/best.pt")
    parser.add_argument("--label_maps_dir", type=str,
                        default="data/processed/ax_t1_128/label_maps/")
    parser.add_argument("--output_dir", type=str,
                        default="reports/explainability/batch/")
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--max_per_patient", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controls", nargs="+",
                        default=["field_strength", "echo_time", "manufacturer"],
                        choices=["field_strength", "echo_time", "manufacturer"])
    parser.add_argument("--config_path", type=str, default="configs/diffusion.yaml")

    args = parser.parse_args()

    # Resolve paths
    test_csv = str(PROJECT_ROOT / args.test_csv)
    output_dir = Path(PROJECT_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 10 — Batch Explainability Runner")
    print("=" * 70)
    print(f"Test CSV: {test_csv}")
    print(f"Diffusion checkpoint: {args.diffusion_checkpoint}")
    print(f"Predictor checkpoint: {args.predictor_checkpoint}")
    print(f"Output dir: {output_dir}")
    print(f"N examples per control: {args.n_examples}")
    print(f"Controls: {args.controls}")
    print(f"Seed: {args.seed}")
    print()

    # Load test data
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} test samples from {df['patient_id'].nunique()} patients")

    # Build counterfactual editor
    print("\nLoading diffusion model...")
    editor = CounterfactualEditor(
        config_path=args.config_path,
        checkpoint_path=args.diffusion_checkpoint,
    )

    # Build explainability engine
    print("\nLoading explainability engine...")
    engine = ExplainabilityEngine(
        predictor_checkpoint=args.predictor_checkpoint,
        label_maps_dir=args.label_maps_dir,
    )

    # Run for each control
    all_summaries = {}
    for control in args.controls:
        summary = run_batch_for_control(
            control=control,
            test_df=df,
            editor=editor,
            engine=engine,
            output_dir=output_dir,
            n_examples=args.n_examples,
            seed=args.seed,
        )
        all_summaries[control] = summary

    # Print final summary
    print("\n" + "=" * 70)
    print("BATCH EXPLAINABILITY COMPLETE")
    print("=" * 70)

    for control, summary in all_summaries.items():
        print(f"\n--- {control} ---")
        print(f"  Examples: {summary['n_success']}/{summary['n_total']}")
        print(f"  SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
        print(f"  MAD:  {summary['mad_mean']:.4f} ± {summary['mad_std']:.4f}")
        if control == "field_strength":
            print(f"  FS Hit Rate: {summary['fs_hit_rate']:.2%}")
            print(f"  FS Gain: {summary['fs_gain_mean']:.4f}")
        elif control == "echo_time":
            print(f"  Echo Moved Closer: {summary['echo_moved_closer_rate']:.2%}")
            print(f"  Echo Improvement: {summary['echo_improvement_mean']:.4f}")
        elif control == "manufacturer":
            print(f"  Mfg Hit Rate: {summary['mfg_hit_rate']:.2%}")
            print(f"  Mfg Gain: {summary['mfg_gain_mean']:.4f}")

    # Save combined summary
    combined_path = output_dir / "summaries" / "all_controls_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()
