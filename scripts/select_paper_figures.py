#!/usr/bin/env python3
"""
Phase 10 — Step 10.4: Paper Figure Selector.

Reads batch explainability results, ranks examples by quality,
and produces combined publication figures for the paper.

Usage:
    python scripts/select_paper_figures.py \
        --batch_dir reports/explainability/batch/ \
        --output_dir reports/paper_figures/ \
        --n_per_axis 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.engine import ExplainabilityEngine
from src.explainability.panel_generator import generate_combined_panel


def load_metrics_for_control(batch_dir: Path, control: str) -> list:
    """Load all per-example metrics JSONs for a control axis."""
    panels_dir = batch_dir / "panels" / control
    if not panels_dir.exists():
        print(f"Warning: {panels_dir} does not exist")
        return []

    metrics_files = sorted(panels_dir.glob("*_metrics.json"))
    all_metrics = []
    for mf in metrics_files:
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_metrics_file"] = str(mf)
            all_metrics.append(data)

    print(f"Loaded {len(all_metrics)} metrics files for {control}")
    return all_metrics


def rank_field_strength(metrics_list: list, n: int = 3) -> list:
    """
    Rank field strength examples by:
      1. target_field_gain DESC (want strong target alignment)
      2. global_ssim DESC (want good anatomy preservation)

    Returns top N.
    """
    scored = []
    for m in metrics_list:
        fs_gain = m.get("predictor_gain", {}).get("fs_gain", 0.0)
        ssim = m.get("global_ssim", 0.0)
        scored.append((fs_gain, ssim, m))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [s[2] for s in scored[:n]]


def rank_echo_time(metrics_list: list, n: int = 3) -> list:
    """
    Rank echo time examples by:
      1. echo_improvement DESC
      2. global_ssim DESC
    """
    scored = []
    for m in metrics_list:
        echo_imp = m.get("predictor_gain", {}).get("echo_improvement", 0.0)
        ssim = m.get("global_ssim", 0.0)
        scored.append((echo_imp, ssim, m))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [s[2] for s in scored[:n]]


def rank_manufacturer(metrics_list: list) -> list:
    """
    Select two representative manufacturer examples:
      1. "balanced" — moderate gain + moderate SSIM
      2. "high-gain" — highest gain regardless of SSIM
    """
    if not metrics_list:
        return []

    # High-gain: best mfg_gain
    by_gain = sorted(
        metrics_list,
        key=lambda m: m.get("predictor_gain", {}).get("mfg_gain", 0.0),
        reverse=True,
    )
    high_gain = by_gain[0]

    # Balanced: score = mfg_gain * ssim (compromise)
    scored = []
    for m in metrics_list:
        mfg_gain = m.get("predictor_gain", {}).get("mfg_gain", 0.0)
        ssim = m.get("global_ssim", 0.0)
        score = mfg_gain * ssim
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    balanced = scored[0][1]

    # Avoid selecting the same example twice
    if balanced.get("example_id") == high_gain.get("example_id"):
        if len(scored) > 1:
            balanced = scored[1][1]

    return [balanced, high_gain]


def build_panel_data_from_metrics(
    metrics: dict,
    batch_dir: Path,
    control: str,
    row_label: str,
) -> dict:
    """
    Build panel data dict from metrics for the combined panel generator.
    Loads the original and counterfactual images from the image paths.
    """
    # Load original image
    image_path = metrics.get("image_path", "")
    abs_image_path = PROJECT_ROOT / image_path
    if abs_image_path.exists():
        original = np.load(str(abs_image_path)).astype(np.float32)
        original = np.clip(original, 0.0, 1.0)
    else:
        print(f"Warning: cannot find {abs_image_path}, using zeros")
        original = np.zeros((128, 128), dtype=np.float32)

    # We need to regenerate the counterfactual or load from saved panel
    # For now, we'll generate a synthetic counterfactual from the metrics
    # In practice, the batch runner should also save the counterfactual .npy
    # For the paper figure selector, we re-run generation
    return {
        "original": original,
        "source_condition": metrics.get("source_condition", {}),
        "target_condition": metrics.get("target_condition", {}),
        "row_label": row_label,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 10 — Paper Figure Selector")
    parser.add_argument("--batch_dir", type=str,
                        default="reports/explainability/batch/")
    parser.add_argument("--output_dir", type=str,
                        default="reports/paper_figures/")
    parser.add_argument("--n_per_axis", type=int, default=3)
    parser.add_argument("--diffusion_checkpoint", type=str,
                        default="checkpoints/diffusion/ax_t1_128/latest.pt")
    parser.add_argument("--predictor_checkpoint", type=str,
                        default="checkpoints/predictor/ax_t1_128/best.pt")
    parser.add_argument("--label_maps_dir", type=str,
                        default="data/processed/ax_t1_128/label_maps/")
    parser.add_argument("--config_path", type=str,
                        default="configs/diffusion.yaml")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    batch_dir = Path(PROJECT_ROOT / args.batch_dir)
    output_dir = Path(PROJECT_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 10 — Paper Figure Selector")
    print("=" * 70)

    # Load metrics for each control
    fs_metrics = load_metrics_for_control(batch_dir, "field_strength")
    et_metrics = load_metrics_for_control(batch_dir, "echo_time")
    mfg_metrics = load_metrics_for_control(batch_dir, "manufacturer")

    # Rank and select
    print("\n--- Ranking examples ---")

    fs_selected = rank_field_strength(fs_metrics, n=args.n_per_axis)
    et_selected = rank_echo_time(et_metrics, n=args.n_per_axis)
    mfg_selected = rank_manufacturer(mfg_metrics)

    # Print selections
    print(f"\nField Strength — top {len(fs_selected)}:")
    for i, m in enumerate(fs_selected):
        print(f"  {i+1}. {m.get('example_id', '?')} | "
              f"SSIM={m.get('global_ssim', 0):.4f} | "
              f"FS Gain={m.get('predictor_gain', {}).get('fs_gain', 0):.4f}")

    print(f"\nEcho Time — top {len(et_selected)}:")
    for i, m in enumerate(et_selected):
        print(f"  {i+1}. {m.get('example_id', '?')} | "
              f"SSIM={m.get('global_ssim', 0):.4f} | "
              f"Echo Imp={m.get('predictor_gain', {}).get('echo_improvement', 0):.4f}")

    print(f"\nManufacturer — {len(mfg_selected)} selected:")
    for i, m in enumerate(mfg_selected):
        label = "balanced" if i == 0 else "high-gain"
        print(f"  {label}: {m.get('example_id', '?')} | "
              f"SSIM={m.get('global_ssim', 0):.4f} | "
              f"Mfg Gain={m.get('predictor_gain', {}).get('mfg_gain', 0):.4f}")

    # Save selection metadata
    selection = {
        "field_strength": [
            {
                "example_id": m.get("example_id"),
                "image_path": m.get("image_path"),
                "global_ssim": m.get("global_ssim"),
                "fs_gain": m.get("predictor_gain", {}).get("fs_gain"),
            }
            for m in fs_selected
        ],
        "echo_time": [
            {
                "example_id": m.get("example_id"),
                "image_path": m.get("image_path"),
                "global_ssim": m.get("global_ssim"),
                "echo_improvement": m.get("predictor_gain", {}).get("echo_improvement"),
            }
            for m in et_selected
        ],
        "manufacturer": [
            {
                "example_id": m.get("example_id"),
                "type": "balanced" if i == 0 else "high-gain",
                "image_path": m.get("image_path"),
                "global_ssim": m.get("global_ssim"),
                "mfg_gain": m.get("predictor_gain", {}).get("mfg_gain"),
            }
            for i, m in enumerate(mfg_selected)
        ],
    }

    selection_path = output_dir / "paper_figure_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, default=str)
    print(f"\nSelection saved: {selection_path}")

    # --- Generate combined figure ---
    # We need to regenerate counterfactuals for the selected examples
    print("\n--- Generating combined paper figure ---")
    print("Loading models for regeneration...")

    from src.inference.counterfactual_editor import CounterfactualEditor
    from src.explainability.engine import ExplainabilityEngine
    from src.explainability.panel_generator import generate_combined_panel

    editor = CounterfactualEditor(
        config_path=args.config_path,
        checkpoint_path=args.diffusion_checkpoint,
    )
    engine = ExplainabilityEngine(
        predictor_checkpoint=args.predictor_checkpoint,
        label_maps_dir=args.label_maps_dir,
    )

    # Build panel data for combined figure
    panels = []

    # Row 1: Best field strength
    if fs_selected:
        panel = _regenerate_panel(
            fs_selected[0], "field_strength", editor, engine,
            row_label="Field Strength (best)", seed=args.seed,
        )
        if panel:
            panels.append(panel)

    # Row 2: Best echo time
    if et_selected:
        panel = _regenerate_panel(
            et_selected[0], "echo_time", editor, engine,
            row_label="EchoTime (best)", seed=args.seed,
        )
        if panel:
            panels.append(panel)

    # Row 3: Manufacturer balanced
    if len(mfg_selected) >= 1:
        panel = _regenerate_panel(
            mfg_selected[0], "manufacturer", editor, engine,
            row_label="Manufacturer (balanced)", seed=args.seed,
        )
        if panel:
            panels.append(panel)

    # Row 4: Manufacturer high-gain
    if len(mfg_selected) >= 2:
        panel = _regenerate_panel(
            mfg_selected[1], "manufacturer", editor, engine,
            row_label="Manufacturer (high-gain)", seed=args.seed,
        )
        if panel:
            panels.append(panel)

    if panels:
        combined_path = output_dir / "combined_main_figure"
        generate_combined_panel(
            panels_data=panels,
            save_path=str(combined_path),
            title="Counterfactual MRI Generation — Explainability Overview",
        )
        print(f"\nCombined figure saved: {combined_path}")
    else:
        print("\nNo panels generated (no batch results found?)")

    # Also generate per-axis best figures
    for control, selected, n_label in [
        ("field_strength", fs_selected, "best"),
        ("echo_time", et_selected, "best"),
    ]:
        if not selected:
            continue

        per_axis_panels = []
        for i, m in enumerate(selected[:3]):
            panel = _regenerate_panel(
                m, control, editor, engine,
                row_label=f"#{i+1}", seed=args.seed,
            )
            if panel:
                per_axis_panels.append(panel)

        if per_axis_panels:
            axis_path = output_dir / f"fig_{control}_{n_label}3"
            generate_combined_panel(
                panels_data=per_axis_panels,
                save_path=str(axis_path),
                title=f"{control.replace('_', ' ').title()} — Top 3 Examples",
            )

    print("\n" + "=" * 70)
    print("Paper figure selection complete!")
    print("=" * 70)


def _regenerate_panel(
    metrics: dict,
    control: str,
    editor,
    engine,
    row_label: str,
    seed: int = 42,
) -> dict:
    """Regenerate a counterfactual and compute explainability for a single selected example."""
    from scripts.run_explainability_batch import CONTROL_CONFIGS

    config = CONTROL_CONFIGS[control]
    image_path = metrics.get("image_path", "")
    abs_path = PROJECT_ROOT / image_path

    if not abs_path.exists():
        print(f"  Warning: cannot find {abs_path}")
        return None

    try:
        original = np.load(str(abs_path)).astype(np.float32)
        original = np.clip(original, 0.0, 1.0)

        source_cond = metrics.get("source_condition", {})
        target_cond = metrics.get("target_condition", {})

        # Convert string keys if needed
        for cond in [source_cond, target_cond]:
            if "manufacturer_label" in cond:
                cond["manufacturer_label"] = int(cond["manufacturer_label"])
            if "field_strength_label" in cond:
                cond["field_strength_label"] = int(cond["field_strength_label"])
            if "echo_time_norm" in cond:
                cond["echo_time_norm"] = float(cond["echo_time_norm"])

        # Regenerate counterfactual
        example_idx = int(metrics.get("example_id", "example_000").split("_")[-1])
        cf_result = editor.generate_counterfactual(
            image_01=original,
            target_manufacturer_label=target_cond["manufacturer_label"],
            target_field_strength_label=target_cond["field_strength_label"],
            target_echo_time_norm=target_cond["echo_time_norm"],
            edit_timestep=config["edit_timestep"],
            guidance_scale=config["guidance_scale"],
            seed=seed + example_idx,
        )
        counterfactual = cf_result["generated_01"]

        # Compute explainability
        results = engine.compute_all(
            original=original,
            counterfactual=counterfactual,
            source_condition=source_cond,
            target_condition=target_cond,
        )

        return {
            "original": original,
            "counterfactual": counterfactual,
            "results": results,
            "source_condition": source_cond,
            "target_condition": target_cond,
            "row_label": row_label,
        }

    except Exception as e:
        print(f"  Error regenerating panel: {e}")
        return None


if __name__ == "__main__":
    main()
