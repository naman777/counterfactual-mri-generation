#!/usr/bin/env python3
"""
Phase 12 — Step 12.4: Ablation Table Runner.

Runs field strength editing (strongest axis) across a grid of
edit timesteps and guidance scales on a smaller subset (25 examples)
to produce the ablation table.

Usage:
    python scripts/run_phase12_ablation.py \
        --n_examples 25 \
        --seed 42 \
        --timestep_grid 100 200 300 400 \
        --guidance_grid 1.0 3.0 5.0
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
from src.inference.counterfactual_editor import CounterfactualEditor

# Reuse Phase 10 infrastructure
from scripts.run_explainability_batch import (
    build_target_condition,
    select_diverse_examples,
)


# Ablation CSV columns
ABLATION_COLUMNS = [
    "edit_timestep",
    "guidance_scale",
    "n_examples",
    "ssim_mean",
    "ssim_std",
    "mad_mean",
    "mad_std",
    "fs_gain_mean",
    "fs_gain_std",
    "fs_hit_rate",
    "mfg_stability_rate",
    "mean_time_per_example_s",
]


def run_ablation_config(
    examples: pd.DataFrame,
    editor: CounterfactualEditor,
    engine: ExplainabilityEngine,
    edit_timestep: int,
    guidance_scale: float,
    seed: int = 42,
) -> dict:
    """
    Run field_strength evaluation for a single ablation configuration.

    Returns:
        dict with aggregated metrics for this configuration.
    """
    control = "field_strength"

    ssims = []
    mads = []
    fs_gains = []
    fs_hits = []
    mfg_stables = []
    gen_times = []
    success = 0
    fail = 0

    for idx, (_, row) in enumerate(examples.iterrows()):
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
            gen_times.append(gen_time)

            # Compute metrics
            results = engine.compute_all(
                original=original,
                counterfactual=counterfactual,
                source_condition=source_cond,
                target_condition=target_cond,
            )

            ssims.append(results["global_ssim"])
            mads.append(results["mean_abs_diff"])
            fs_gains.append(results["predictor_gain"]["fs_gain"])
            fs_hits.append(results["target_hit"]["fs_hit"])
            mfg_stables.append(results["stability"]["mfg_stable"])
            success += 1

        except Exception as e:
            fail += 1
            continue

    if not ssims:
        return {
            "edit_timestep": edit_timestep,
            "guidance_scale": guidance_scale,
            "n_examples": 0,
            "ssim_mean": 0, "ssim_std": 0,
            "mad_mean": 0, "mad_std": 0,
            "fs_gain_mean": 0, "fs_gain_std": 0,
            "fs_hit_rate": 0,
            "mfg_stability_rate": 0,
            "mean_time_per_example_s": 0,
        }

    return {
        "edit_timestep": edit_timestep,
        "guidance_scale": guidance_scale,
        "n_examples": success,
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "mad_mean": float(np.mean(mads)),
        "mad_std": float(np.std(mads)),
        "fs_gain_mean": float(np.mean(fs_gains)),
        "fs_gain_std": float(np.std(fs_gains)),
        "fs_hit_rate": float(np.mean(fs_hits)),
        "mfg_stability_rate": float(np.mean(mfg_stables)),
        "mean_time_per_example_s": float(np.mean(gen_times)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 12 — Ablation Table Runner"
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
                        default="reports/phase12/ablation/")
    parser.add_argument("--n_examples", type=int, default=25)
    parser.add_argument("--control", type=str, default="field_strength",
                        choices=["field_strength", "echo_time", "manufacturer"])
    parser.add_argument("--timestep_grid", type=int, nargs="+",
                        default=[100, 200, 300, 400])
    parser.add_argument("--guidance_grid", type=float, nargs="+",
                        default=[1.0, 3.0, 5.0])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    test_csv = str(PROJECT_ROOT / args.test_csv)
    output_dir = Path(PROJECT_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_configs = len(args.timestep_grid) * len(args.guidance_grid)

    print("=" * 70)
    print("Phase 12 — Ablation Table Runner")
    print("=" * 70)
    print(f"Control:         {args.control}")
    print(f"Timestep grid:   {args.timestep_grid}")
    print(f"Guidance grid:   {args.guidance_grid}")
    print(f"Total configs:   {n_configs}")
    print(f"Examples/config: {args.n_examples}")
    print(f"Seed:            {args.seed}")
    print()

    # Load test data
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} test samples from {df['patient_id'].nunique()} patients")

    # Select examples once (same for all configurations)
    examples = select_diverse_examples(df, args.n_examples, seed=args.seed)
    print(f"Selected {len(examples)} examples from {examples['patient_id'].nunique()} patients")

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

    # Run ablation grid
    all_results = []
    config_idx = 0

    for timestep in args.timestep_grid:
        for guidance in args.guidance_grid:
            config_idx += 1
            print(f"\n{'─'*60}")
            print(f"Config {config_idx}/{n_configs}: timestep={timestep}, guidance={guidance}")
            print(f"{'─'*60}")

            result = run_ablation_config(
                examples=examples,
                editor=editor,
                engine=engine,
                edit_timestep=timestep,
                guidance_scale=guidance,
                seed=args.seed,
            )
            all_results.append(result)

            print(f"  SSIM={result['ssim_mean']:.4f}±{result['ssim_std']:.4f} | "
                  f"MAD={result['mad_mean']:.4f} | "
                  f"FS Gain={result['fs_gain_mean']:+.4f} | "
                  f"Hit={result['fs_hit_rate']:.2%} | "
                  f"MfgStab={result['mfg_stability_rate']:.2%}")

    # ─── Save ablation table CSV ─────────────────────────────────────
    csv_path = output_dir / "ablation_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ABLATION_COLUMNS)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nAblation table CSV saved: {csv_path}")

    # ─── Save ablation JSON ──────────────────────────────────────────
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Ablation results JSON saved: {json_path}")

    # Print final table
    print(f"\n{'='*70}")
    print("ABLATION TABLE")
    print(f"{'='*70}")
    print(f"\n{'Timestep':<10} {'Guidance':<10} {'SSIM↑':<16} {'MAD↓':<10} "
          f"{'FS Gain↑':<16} {'Hit Rate↑':<10} {'Mfg Stab':<10}")
    print("-" * 92)
    for r in all_results:
        print(f"{r['edit_timestep']:<10} {r['guidance_scale']:<10.1f} "
              f"{r['ssim_mean']:.4f}±{r['ssim_std']:.3f}  "
              f"{r['mad_mean']:.4f}     "
              f"{r['fs_gain_mean']:+.4f}±{r['fs_gain_std']:.3f} "
              f"{r['fs_hit_rate']:.2%}     "
              f"{r['mfg_stability_rate']:.2%}")

    print(f"\n{'='*70}")
    print("ABLATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
