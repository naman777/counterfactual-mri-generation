#!/usr/bin/env python3
"""
Phase 12 — Step 12.5: Results Aggregation.

Reads all Phase 12 outputs and produces final paper tables
in CSV and LaTeX format.

Usage:
    python scripts/aggregate_phase12_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_control_summary(phase12_dir: Path, control: str) -> dict | None:
    """Load the summary JSON for a given control axis."""
    summary_path = phase12_dir / control / f"{control}_summary.json"
    if not summary_path.exists():
        print(f"Warning: {summary_path} not found")
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_summary(phase12_dir: Path) -> dict | None:
    """Load baseline comparison summary."""
    summary_path = phase12_dir / "baselines" / "baseline_summary.json"
    if not summary_path.exists():
        print(f"Warning: {summary_path} not found")
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ablation_results(phase12_dir: Path) -> list | None:
    """Load ablation grid results."""
    json_path = phase12_dir / "ablation" / "ablation_results.json"
    if not json_path.exists():
        print(f"Warning: {json_path} not found")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format as mean ± std."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_pct(val: float) -> str:
    """Format as percentage."""
    return f"{val*100:.1f}\\%"


# ─── Table 1: Main Results ───────────────────────────────────────────


def build_table1(phase12_dir: Path, output_dir: Path):
    """Build Table 1: Main quantitative results across all 3 axes."""
    print("\n--- Building Table 1: Main Results ---")

    controls = ["field_strength", "echo_time", "manufacturer"]
    control_labels = {
        "field_strength": "Field Strength",
        "echo_time": "EchoTime",
        "manufacturer": "Manufacturer",
    }

    rows = []
    for control in controls:
        summary = load_control_summary(phase12_dir, control)
        if summary is None:
            print(f"  Skipping {control}: no summary found")
            continue

        # Select the primary gain metric for each control
        if control == "field_strength":
            gain_mean = summary.get("fs_gain_mean", 0)
            gain_std = summary.get("fs_gain_std", 0)
            hit_rate = summary.get("fs_hit_rate", 0)
        elif control == "echo_time":
            gain_mean = summary.get("echo_improvement_mean", 0)
            gain_std = summary.get("echo_improvement_std", 0)
            hit_rate = summary.get("echo_moved_closer_rate", 0)
        else:  # manufacturer
            gain_mean = summary.get("mfg_gain_mean", 0)
            gain_std = summary.get("mfg_gain_std", 0)
            hit_rate = summary.get("mfg_hit_rate", 0)

        row = {
            "Control": control_labels[control],
            "N": summary.get("n_examples", summary.get("n_success", 0)),
            "SSIM_mean": summary.get("ssim_mean", 0),
            "SSIM_std": summary.get("ssim_std", 0),
            "MAD_mean": summary.get("mad_mean", 0),
            "MAD_std": summary.get("mad_std", 0),
            "Gain_mean": gain_mean,
            "Gain_std": gain_std,
            "Hit_Rate": hit_rate,
            "Mfg_Stability": summary.get("mfg_stability_rate", 0),
            "FS_Stability": summary.get("fs_stability_rate", 0),
        }
        rows.append(row)
        print(f"  {control}: SSIM={row['SSIM_mean']:.4f} | Gain={gain_mean:.4f} | Hit={hit_rate:.2%}")

    # Save CSV
    csv_path = output_dir / "table1_main_results.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")

    # Generate LaTeX
    latex_path = output_dir / "table_latex" / "table1.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table1_latex(rows, latex_path)

    return rows


def _write_table1_latex(rows: list, path: Path):
    """Write Table 1 as LaTeX."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quantitative evaluation of counterfactual MRI generation across three acquisition-parameter control axes. Each row reports metrics averaged over test examples with fixed optimal hyperparameters.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Control & N & SSIM $\uparrow$ & MAD $\downarrow$ & Target Gain $\uparrow$ & Hit Rate $\uparrow$ \\",
        r"\midrule",
    ]

    for row in rows:
        ssim_str = format_mean_std(row["SSIM_mean"], row["SSIM_std"])
        mad_str = format_mean_std(row["MAD_mean"], row["MAD_std"], 4)
        gain_str = format_mean_std(row["Gain_mean"], row["Gain_std"])
        hit_str = format_pct(row["Hit_Rate"])

        lines.append(
            f"{row['Control']} & {row['N']} & {ssim_str} & {mad_str} & {gain_str} & {hit_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX saved: {path}")


# ─── Table 2: Baseline Comparison ────────────────────────────────────


def build_table2(phase12_dir: Path, output_dir: Path):
    """Build Table 2: Diffusion vs baselines comparison."""
    print("\n--- Building Table 2: Baseline Comparison ---")

    baseline_summary = load_baseline_summary(phase12_dir)

    # Also load diffusion results as "Ours"
    controls = ["field_strength", "echo_time", "manufacturer"]
    control_labels = {
        "field_strength": "Field Strength",
        "echo_time": "EchoTime",
        "manufacturer": "Manufacturer",
    }

    rows = []

    for control in controls:
        # Diffusion (ours) row
        diff_summary = load_control_summary(phase12_dir, control)
        if diff_summary:
            if control == "field_strength":
                gain = diff_summary.get("fs_gain_mean", 0)
                hit = diff_summary.get("fs_hit_rate", 0)
            elif control == "echo_time":
                gain = diff_summary.get("echo_improvement_mean", 0)
                hit = diff_summary.get("echo_moved_closer_rate", 0)
            else:
                gain = diff_summary.get("mfg_gain_mean", 0)
                hit = diff_summary.get("mfg_hit_rate", 0)

            rows.append({
                "Method": "Ours (Diffusion)",
                "Control": control_labels[control],
                "SSIM_mean": diff_summary.get("ssim_mean", 0),
                "SSIM_std": diff_summary.get("ssim_std", 0),
                "MAD_mean": diff_summary.get("mad_mean", 0),
                "MAD_std": diff_summary.get("mad_std", 0),
                "Gain_mean": gain,
                "Hit_Rate": hit,
            })

        # Baseline rows
        if baseline_summary:
            baseline_names = ["identity", "contrast_shift", "gamma", "gaussian_noise", "histogram_matching"]
            baseline_labels = {
                "identity": "Identity",
                "contrast_shift": "Contrast Shift",
                "gamma": "Gamma",
                "gaussian_noise": "Gaussian Noise",
                "histogram_matching": "Histogram Match",
            }

            for bl_name in baseline_names:
                key = f"{bl_name}__{control}"
                if key in baseline_summary:
                    bl = baseline_summary[key]
                    rows.append({
                        "Method": baseline_labels.get(bl_name, bl_name),
                        "Control": control_labels[control],
                        "SSIM_mean": bl.get("ssim_mean", 0),
                        "SSIM_std": bl.get("ssim_std", 0),
                        "MAD_mean": bl.get("mad_mean", 0),
                        "MAD_std": bl.get("mad_std", 0),
                        "Gain_mean": bl.get("gain_mean", 0),
                        "Hit_Rate": bl.get("hit_rate", 0),
                    })

    # Save CSV
    csv_path = output_dir / "table2_baseline_comparison.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")

    # Generate LaTeX (one table per control for clarity)
    latex_path = output_dir / "table_latex" / "table2.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table2_latex(rows, controls, control_labels, latex_path)

    return rows


def _write_table2_latex(rows: list, controls: list, control_labels: dict, path: Path):
    """Write Table 2 as LaTeX."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of our diffusion-based counterfactual editing against non-generative baselines. Only the diffusion model achieves consistent target alignment across all control axes.}",
        r"\label{tab:baseline_comparison}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Control & Method & SSIM $\uparrow$ & MAD $\downarrow$ & Gain $\uparrow$ & Hit Rate $\uparrow$ \\",
        r"\midrule",
    ]

    for i, control in enumerate(controls):
        control_rows = [r for r in rows if r["Control"] == control_labels[control]]
        for j, row in enumerate(control_rows):
            method = row["Method"]
            ssim_str = f"{row['SSIM_mean']:.4f}"
            mad_str = f"{row['MAD_mean']:.4f}"
            gain_str = f"{row['Gain_mean']:+.4f}"
            hit_str = format_pct(row["Hit_Rate"])

            ctrl_label = control_labels[control] if j == 0 else ""

            if method == "Ours (Diffusion)":
                method = r"\textbf{Ours (Diffusion)}"
                ssim_str = r"\textbf{" + ssim_str + "}"
                gain_str = r"\textbf{" + gain_str + "}"
                hit_str = r"\textbf{" + hit_str + "}"

            lines.append(f"{ctrl_label} & {method} & {ssim_str} & {mad_str} & {gain_str} & {hit_str} \\\\")

        if i < len(controls) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX saved: {path}")


# ─── Table 3: Ablation ──────────────────────────────────────────────


def build_table3(phase12_dir: Path, output_dir: Path):
    """Build Table 3: Ablation over edit timestep and guidance scale."""
    print("\n--- Building Table 3: Ablation ---")

    results = load_ablation_results(phase12_dir)
    if results is None:
        print("  No ablation results found")
        return []

    # Save CSV (already exists in ablation dir, copy to paper_tables)
    csv_path = output_dir / "table3_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys()) if results else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {csv_path}")

    # Generate LaTeX
    latex_path = output_dir / "table_latex" / "table3.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table3_latex(results, latex_path)

    return results


def _write_table3_latex(results: list, path: Path):
    """Write Table 3 as LaTeX."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on edit timestep and guidance scale for field strength editing. Higher timesteps improve target alignment at the cost of anatomy preservation (SSIM).}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"Edit Steps & Guidance & SSIM $\uparrow$ & MAD $\downarrow$ & FS Gain $\uparrow$ & Hit Rate $\uparrow$ \\",
        r"\midrule",
    ]

    # Find the best config (highest SSIM among those with hit_rate >= 0.9)
    best_idx = None
    best_ssim = -1
    for i, r in enumerate(results):
        if r.get("fs_hit_rate", 0) >= 0.9 and r.get("ssim_mean", 0) > best_ssim:
            best_ssim = r["ssim_mean"]
            best_idx = i

    for i, r in enumerate(results):
        ssim_str = format_mean_std(r["ssim_mean"], r["ssim_std"])
        mad_str = f"{r['mad_mean']:.4f}"
        gain_str = format_mean_std(r["fs_gain_mean"], r["fs_gain_std"])
        hit_str = format_pct(r["fs_hit_rate"])

        # Bold the best config
        if i == best_idx:
            ssim_str = r"\textbf{" + ssim_str + "}"
            hit_str = r"\textbf{" + hit_str + "}"

        lines.append(
            f"{r['edit_timestep']} & {r['guidance_scale']:.1f} & {ssim_str} & {mad_str} & {gain_str} & {hit_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX saved: {path}")


# ─── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 12 — Results Aggregation"
    )
    parser.add_argument("--phase12_dir", type=str,
                        default="reports/phase12/")
    parser.add_argument("--output_dir", type=str,
                        default="reports/paper_tables/")

    args = parser.parse_args()

    phase12_dir = Path(PROJECT_ROOT / args.phase12_dir)
    output_dir = Path(PROJECT_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table_latex").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 12 — Results Aggregation")
    print("=" * 70)
    print(f"Phase 12 dir: {phase12_dir}")
    print(f"Output dir:   {output_dir}")

    # Build all three tables
    table1 = build_table1(phase12_dir, output_dir)
    table2 = build_table2(phase12_dir, output_dir)
    table3 = build_table3(phase12_dir, output_dir)

    # Final summary
    print(f"\n{'='*70}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
