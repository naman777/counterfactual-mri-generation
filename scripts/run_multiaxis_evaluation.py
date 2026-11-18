#!/usr/bin/env python3
"""
P1.2 — Multi-Axis Simultaneous Editing Evaluation.

Edits TWO acquisition parameters simultaneously and compares results
against single-axis editing to measure disentanglement quality.

Three dual-axis combinations:
  1. Manufacturer + Field Strength  (e.g. GE/1.5T → Siemens/3.0T)
  2. Manufacturer + Echo Time       (e.g. GE/short ET → Siemens/long ET)
  3. Field Strength + Echo Time     (e.g. 1.5T/short ET → 3.0T/long ET)

For each combination we measure:
  - SSIM             (anatomy preservation)
  - MAD              (pixel distortion)
  - Per-axis gain    (did each target axis shift correctly?)
  - Per-axis hit     (did predictor switch to target for each axis?)
  - Disentanglement effect: additive vs interfering

Disentanglement metric:
  AdditiveDelta = |dual_axis_gain - (singleA_gain + singleB_gain)|
  Lower = more additive (ideal). Higher = interfering.

Usage:
    python scripts/run_multiaxis_evaluation.py --device cuda
    python scripts/run_multiaxis_evaluation.py --n_examples 50 --device cuda

Outputs:
    reports/multiaxis/multiaxis_results.csv
    reports/multiaxis/multiaxis_summary.json
    reports/multiaxis/multiaxis_table.tex
    reports/multiaxis/fig_multiaxis_comparison.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.engine import ExplainabilityEngine
from src.inference.counterfactual_editor import CounterfactualEditor
from scripts.run_explainability_batch import (
    select_diverse_examples,
    aggregate_metrics,
)


# ─── Dual-axis combination definitions ───────────────────────────────────────

DUAL_AXIS_COMBOS = {
    "mfg_plus_fs": {
        "label":       "Mfg + Field Strength",
        "axes":        ["manufacturer", "field_strength"],
        "edit_timestep": 300,      # manufacturer needs higher timestep
        "guidance_scale": 5.0,
        "description": "GE/1.5T → Siemens/3.0T (or reversed)",
    },
    "mfg_plus_et": {
        "label":       "Mfg + Echo Time",
        "axes":        ["manufacturer", "echo_time"],
        "edit_timestep": 300,
        "guidance_scale": 5.0,
        "description": "Manufacturer + Echo Time simultaneously",
    },
    "fs_plus_et": {
        "label":       "Field Strength + Echo Time",
        "axes":        ["field_strength", "echo_time"],
        "edit_timestep": 100,      # field strength uses lower timestep
        "guidance_scale": 3.0,
        "description": "Field Strength + Echo Time simultaneously",
    },
}

# Single-axis reference settings (from main eval)
SINGLE_AXIS_REF = {
    "field_strength": {"edit_timestep": 100, "guidance_scale": 3.0},
    "manufacturer":   {"edit_timestep": 300, "guidance_scale": 5.0},
    "echo_time":      {"edit_timestep": 100, "guidance_scale": 1.0},
}

CSV_COLUMNS = [
    "example_id", "patient_id", "combo",
    "edit_timestep", "guidance_scale",
    "global_ssim", "mean_abs_diff",
    "mfg_gain", "mfg_hit",
    "fs_gain",  "fs_hit",
    "echo_improvement", "echo_moved_closer",
    "mfg_stable", "fs_stable",
    "generation_time_s",
]


# ─── Target condition builder for dual-axis ───────────────────────────────────

def build_dual_target(row: pd.Series, combo: str) -> tuple[dict, dict]:
    """
    Build source + target conditions flipping BOTH axes in the combo.
    """
    source = {
        "manufacturer_label":  int(row["manufacturer_label"]),
        "field_strength_label": int(row["field_strength_label"]),
        "echo_time_norm":      float(row["echo_time_norm"]),
    }
    target = source.copy()

    axes = DUAL_AXIS_COMBOS[combo]["axes"]

    if "manufacturer" in axes:
        target["manufacturer_label"] = 1 - source["manufacturer_label"]  # flip

    if "field_strength" in axes:
        target["field_strength_label"] = 1 - source["field_strength_label"]  # flip

    if "echo_time" in axes:
        if source["echo_time_norm"] > 0.5:
            target["echo_time_norm"] = 0.15
        else:
            target["echo_time_norm"] = 0.85

    return source, target


# ─── Run one combo ────────────────────────────────────────────────────────────

def run_combo(
    combo: str,
    test_df: pd.DataFrame,
    editor: CounterfactualEditor,
    engine: ExplainabilityEngine,
    out_dir: Path,
    n_examples: int = 50,
    seed: int = 42,
) -> dict:
    cfg = DUAL_AXIS_COMBOS[combo]
    print(f"\n{'='*65}")
    print(f"Combo: {cfg['label']}")
    print(f"  Axes:     {cfg['axes']}")
    print(f"  Settings: t={cfg['edit_timestep']}, w={cfg['guidance_scale']}")
    print(f"  Desc:     {cfg['description']}")
    print(f"{'='*65}")

    examples = select_diverse_examples(test_df, n_examples, seed=seed)
    print(f"  Selected {len(examples)} examples from "
          f"{examples['patient_id'].nunique()} patients")

    combo_dir = out_dir / combo
    combo_dir.mkdir(parents=True, exist_ok=True)

    csv_rows, all_metrics = [], []
    success, fail = 0, 0
    total_time = 0.0

    for idx, (_, row) in enumerate(examples.iterrows()):
        ex_id = f"example_{idx:03d}"
        try:
            img_path = str(PROJECT_ROOT / row["image_path"])
            original = np.load(img_path).astype(np.float32)
            original = np.clip(original, 0.0, 1.0)

            source_cond, target_cond = build_dual_target(row, combo)

            t0 = time.time()
            cf = editor.generate_counterfactual(
                image_01=original,
                target_manufacturer_label=target_cond["manufacturer_label"],
                target_field_strength_label=target_cond["field_strength_label"],
                target_echo_time_norm=target_cond["echo_time_norm"],
                edit_timestep=cfg["edit_timestep"],
                guidance_scale=cfg["guidance_scale"],
                seed=seed + idx,
            )
            counterfactual = cf["generated_01"]
            gen_time = time.time() - t0
            total_time += gen_time

            results = engine.compute_all(
                original=original,
                counterfactual=counterfactual,
                source_condition=source_cond,
                target_condition=target_cond,
            )

            gain = results["predictor_gain"]
            hit  = results["target_hit"]
            stab = results["stability"]

            csv_rows.append({
                "example_id":       ex_id,
                "patient_id":       str(row["patient_id"]),
                "combo":            combo,
                "edit_timestep":    cfg["edit_timestep"],
                "guidance_scale":   cfg["guidance_scale"],
                "global_ssim":      results["global_ssim"],
                "mean_abs_diff":    results["mean_abs_diff"],
                "mfg_gain":         gain["mfg_gain"],
                "mfg_hit":          hit["mfg_hit"],
                "fs_gain":          gain["fs_gain"],
                "fs_hit":           hit["fs_hit"],
                "echo_improvement": gain["echo_improvement"],
                "echo_moved_closer":hit["echo_moved_closer"],
                "mfg_stable":       stab["mfg_stable"],
                "fs_stable":        stab["fs_stable"],
                "generation_time_s": gen_time,
            })
            all_metrics.append(engine.results_to_json_serializable(results))
            success += 1

            print(f"  [{idx+1:3d}/{n_examples}] SSIM={results['global_ssim']:.4f} "
                  f"MAD={results['mean_abs_diff']:.4f} "
                  f"MfgHit={hit['mfg_hit']} FSHit={hit['fs_hit']} "
                  f"EchoCloser={hit['echo_moved_closer']} t={gen_time:.1f}s")

        except Exception as e:
            print(f"  [{idx+1:3d}/{n_examples}] FAILED: {e}")
            fail += 1

    # Save per-combo CSV
    csv_path = combo_dir / f"{combo}_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Aggregate
    df_combo = pd.DataFrame(csv_rows)
    summary = {
        "combo":           combo,
        "label":           cfg["label"],
        "axes":            cfg["axes"],
        "n_success":       success,
        "n_failed":        fail,
        "n_total":         n_examples,
        "edit_timestep":   cfg["edit_timestep"],
        "guidance_scale":  cfg["guidance_scale"],
        "ssim_mean":       float(df_combo["global_ssim"].mean()),
        "ssim_std":        float(df_combo["global_ssim"].std()),
        "mad_mean":        float(df_combo["mean_abs_diff"].mean()),
        "mad_std":         float(df_combo["mean_abs_diff"].std()),
        "mfg_gain_mean":   float(df_combo["mfg_gain"].mean()),
        "mfg_hit_rate":    float(df_combo["mfg_hit"].astype(float).mean()),
        "fs_gain_mean":    float(df_combo["fs_gain"].mean()),
        "fs_hit_rate":     float(df_combo["fs_hit"].astype(float).mean()),
        "echo_improvement_mean": float(df_combo["echo_improvement"].mean()),
        "echo_moved_closer_rate": float(df_combo["echo_moved_closer"].astype(float).mean()),
        "mean_time_per_example_s": total_time / max(success, 1),
    }

    print(f"\n  ── {cfg['label']} Summary ──")
    print(f"  SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"  MAD:  {summary['mad_mean']:.4f} ± {summary['mad_std']:.4f}")
    print(f"  Mfg Hit: {summary['mfg_hit_rate']:.1%}  FS Hit: {summary['fs_hit_rate']:.1%}  "
          f"Echo Closer: {summary['echo_moved_closer_rate']:.1%}")

    return summary


# ─── Single-axis reference table (from existing CSVs) ─────────────────────────

SINGLE_AXIS_CSV = {
    "field_strength": "reports/phase12/field_strength/field_strength_results.csv",
    "manufacturer":   "reports/phase12/manufacturer/manufacturer_results.csv",
    "echo_time":      "reports/phase12/echo_time/echo_time_results.csv",
}

def load_single_axis_refs() -> dict:
    """Load existing single-axis summaries for comparison."""
    refs = {}
    for axis, csv_rel in SINGLE_AXIS_CSV.items():
        p = PROJECT_ROOT / csv_rel
        if p.exists():
            df = pd.read_csv(p)
            gain_col = {"field_strength": "fs_gain",
                        "manufacturer":   "mfg_gain",
                        "echo_time":      "echo_improvement"}[axis]
            hit_col  = {"field_strength": "fs_hit",
                        "manufacturer":   "mfg_hit",
                        "echo_time":      "echo_moved_closer"}[axis]
            refs[axis] = {
                "ssim_mean":  float(df["global_ssim"].mean()),
                "mad_mean":   float(df["mean_abs_diff"].mean()),
                "gain_mean":  float(df[gain_col].mean()),
                "hit_rate":   float(df[hit_col].astype(float).mean()),
            }
        else:
            print(f"  ⚠ Single-axis CSV not found: {p}")
    return refs


# ─── Disentanglement analysis ─────────────────────────────────────────────────

def compute_disentanglement(combo_summaries: list[dict], single_refs: dict) -> list[dict]:
    """
    For each dual-axis combo, compare each edited axis's gain against
    the corresponding single-axis result.

    Interference = |dual_gain - single_gain| for each edited axis.
    Lower interference → more additive/independent effects.
    """
    rows = []
    for s in combo_summaries:
        axes = s["axes"]
        row  = {"combo": s["combo"], "label": s["label"]}

        for axis in axes:
            if axis not in single_refs:
                continue
            ref = single_refs[axis]

            if axis == "manufacturer":
                dual_gain  = s["mfg_gain_mean"]
                dual_hit   = s["mfg_hit_rate"]
            elif axis == "field_strength":
                dual_gain  = s["fs_gain_mean"]
                dual_hit   = s["fs_hit_rate"]
            elif axis == "echo_time":
                dual_gain  = s["echo_improvement_mean"]
                dual_hit   = s["echo_moved_closer_rate"]
            else:
                continue

            interference = abs(dual_gain - ref["gain_mean"])
            hit_delta    = dual_hit - ref["hit_rate"]

            row[f"{axis}_single_gain"] = ref["gain_mean"]
            row[f"{axis}_dual_gain"]   = dual_gain
            row[f"{axis}_single_hit"]  = ref["hit_rate"]
            row[f"{axis}_dual_hit"]    = dual_hit
            row[f"{axis}_interference"] = interference
            row[f"{axis}_hit_delta"]   = hit_delta

        row["ssim_dual"]   = s["ssim_mean"]
        row["ssim_single_avg"] = float(np.mean(
            [single_refs[a]["ssim_mean"] for a in axes if a in single_refs]
        ))
        rows.append(row)
    return rows


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_comparison(combo_summaries: list[dict], single_refs: dict, out_path: Path):
    """Bar chart comparing dual-axis vs single-axis hit rates and SSIM."""
    plt.style.use("dark_background")
    combos = [s["label"] for s in combo_summaries]
    x = np.arange(len(combos))
    width = 0.28

    fig, axes_plots = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0D1117")
    for ax in axes_plots:
        ax.set_facecolor("#161B22")

    colors = {"single": "#4C9BE8", "dual": "#E8834C"}

    # ── Plot 1: Mfg Hit Rate ──────────────────────────────────────────
    ax = axes_plots[0]
    single_mfg = [single_refs.get("manufacturer", {}).get("hit_rate", 0)] * len(combos)
    dual_mfg   = [s["mfg_hit_rate"] for s in combo_summaries]
    ax.bar(x - width/2, single_mfg, width, label="Single-axis", color=colors["single"], alpha=0.8)
    ax.bar(x + width/2, dual_mfg,   width, label="Dual-axis",   color=colors["dual"],   alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(combos, rotation=15, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Manufacturer Hit Rate", color="white"); ax.set_ylim(0, 1)
    ax.set_title("Mfg Hit Rate:\nSingle vs Dual", color="white", fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1C2128", labelcolor="white", edgecolor="#30363D")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#30363D")

    # ── Plot 2: FS Hit Rate ───────────────────────────────────────────
    ax = axes_plots[1]
    single_fs = [single_refs.get("field_strength", {}).get("hit_rate", 0)] * len(combos)
    dual_fs   = [s["fs_hit_rate"] for s in combo_summaries]
    ax.bar(x - width/2, single_fs, width, label="Single-axis", color=colors["single"], alpha=0.8)
    ax.bar(x + width/2, dual_fs,   width, label="Dual-axis",   color=colors["dual"],   alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(combos, rotation=15, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Field Strength Hit Rate", color="white"); ax.set_ylim(0, 1)
    ax.set_title("FS Hit Rate:\nSingle vs Dual", color="white", fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1C2128", labelcolor="white", edgecolor="#30363D")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#30363D")

    # ── Plot 3: SSIM ─────────────────────────────────────────────────
    ax = axes_plots[2]
    single_ssim = [
        float(np.mean([single_refs.get(a, {}).get("ssim_mean", 0)
                        for a in DUAL_AXIS_COMBOS[s["combo"]]["axes"]
                        if a in single_refs]))
        for s in combo_summaries
    ]
    dual_ssim = [s["ssim_mean"] for s in combo_summaries]
    ax.bar(x - width/2, single_ssim, width, label="Single-axis avg", color=colors["single"], alpha=0.8)
    ax.bar(x + width/2, dual_ssim,   width, label="Dual-axis",       color=colors["dual"],   alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(combos, rotation=15, ha="right", color="white", fontsize=9)
    ax.set_ylabel("SSIM ↑", color="white"); ax.set_ylim(0, 1)
    ax.set_title("SSIM:\nSingle-axis avg vs Dual", color="white", fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1C2128", labelcolor="white", edgecolor="#30363D")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#30363D")

    plt.suptitle("Multi-Axis Simultaneous Editing vs Single-Axis Reference",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figure saved: {out_path.name}")


# ─── LaTeX table ──────────────────────────────────────────────────────────────

def generate_latex(combo_summaries: list[dict], single_refs: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-axis simultaneous editing results vs.\ single-axis baseline. "
        r"Hit rates and SSIM reported for each combination. "
        r"$\Delta$Hit = dual $-$ single for each axis.}",
        r"\label{tab:multiaxis}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Combination & SSIM $\uparrow$ & Mfg Hit $\uparrow$ & FS Hit $\uparrow$ "
        r"& Echo Closer $\uparrow$ & $\Delta$Mfg Hit \\",
        r"\midrule",
    ]
    single_mfg_hit = single_refs.get("manufacturer", {}).get("hit_rate", 0)
    for s in combo_summaries:
        delta = s["mfg_hit_rate"] - single_mfg_hit
        lines.append(
            f"  {s['label']} & {s['ssim_mean']:.3f} & "
            f"{s['mfg_hit_rate']:.2f} & {s['fs_hit_rate']:.2f} & "
            f"{s['echo_moved_closer_rate']:.2f} & "
            f"{delta:+.2f} \\\\"
        )
    # Single-axis reference row
    lines += [
        r"\midrule",
        r"\textit{Single-axis ref} & & & & & \\",
    ]
    for axis, ref in single_refs.items():
        gain_label = {"manufacturer": "Mfg", "field_strength": "FS",
                      "echo_time": "Echo"}[axis]
        lines.append(f"  \\quad {gain_label} & {ref['ssim_mean']:.3f} & "
                     f"{ref['hit_rate']:.2f} & --- & --- & --- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P1.2 — Multi-Axis Evaluation")
    parser.add_argument("--n_examples",  type=int,   default=50)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--combos",      nargs="+",
                        default=list(DUAL_AXIS_COMBOS.keys()),
                        choices=list(DUAL_AXIS_COMBOS.keys()))
    parser.add_argument("--test_csv",    default="data/processed/ax_t1_128/splits/test.csv")
    parser.add_argument("--diffusion_checkpoint",
                        default="checkpoints/diffusion/ax_t1_128/latest.pt")
    parser.add_argument("--predictor_checkpoint",
                        default="checkpoints/predictor/ax_t1_128/best.pt")
    parser.add_argument("--label_maps_dir",
                        default="data/processed/ax_t1_128/label_maps/")
    parser.add_argument("--config_path", default="configs/diffusion.yaml")
    parser.add_argument("--output_dir",  default="reports/multiaxis/")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("P1.2 — Multi-Axis Simultaneous Editing Evaluation")
    print("=" * 65)
    print(f"  Combos:     {args.combos}")
    print(f"  N examples: {args.n_examples} per combo")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {out_dir}")

    # Load data
    df = pd.read_csv(PROJECT_ROOT / args.test_csv)
    print(f"\nLoaded {len(df)} test samples from {df['patient_id'].nunique()} patients")

    # Load models
    print("\nLoading diffusion model...")
    editor = CounterfactualEditor(
        config_path=args.config_path,
        checkpoint_path=args.diffusion_checkpoint,
        device=args.device,
    )

    print("\nLoading explainability engine...")
    engine = ExplainabilityEngine(
        predictor_checkpoint=args.predictor_checkpoint,
        label_maps_dir=args.label_maps_dir,
        device=args.device,
    )

    # Load single-axis references
    print("\nLoading single-axis reference results...")
    single_refs = load_single_axis_refs()
    for ax, ref in single_refs.items():
        print(f"  {ax}: SSIM={ref['ssim_mean']:.4f}, "
              f"HitRate={ref['hit_rate']:.2%}, Gain={ref['gain_mean']:+.4f}")

    # Run each combo
    combo_summaries = []
    for combo in args.combos:
        summary = run_combo(
            combo=combo,
            test_df=df,
            editor=editor,
            engine=engine,
            out_dir=out_dir,
            n_examples=args.n_examples,
            seed=args.seed,
        )
        combo_summaries.append(summary)

    # Disentanglement analysis
    print("\n\nDISENTANGLEMENT ANALYSIS")
    print("=" * 65)
    disent_rows = compute_disentanglement(combo_summaries, single_refs)
    for row in disent_rows:
        print(f"\n  {row['label']}:")
        for axis in ["manufacturer", "field_strength", "echo_time"]:
            if f"{axis}_interference" in row:
                print(f"    {axis}: single_gain={row[f'{axis}_single_gain']:+.4f}  "
                      f"dual_gain={row[f'{axis}_dual_gain']:+.4f}  "
                      f"interference={row[f'{axis}_interference']:.4f}  "
                      f"hit_delta={row[f'{axis}_hit_delta']:+.2%}")
        print(f"    SSIM: single_avg={row['ssim_single_avg']:.4f}  "
              f"dual={row['ssim_dual']:.4f}")

    # Save summary JSON
    full_summary = {
        "combos":            combo_summaries,
        "single_axis_refs":  single_refs,
        "disentanglement":   disent_rows,
        "n_examples":        args.n_examples,
        "seed":              args.seed,
    }
    json_path = out_dir / "multiaxis_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, default=str)
    print(f"\nJSON saved: {json_path.name}")

    # Save combined CSV
    all_csvs = []
    for combo in args.combos:
        cp = out_dir / combo / f"{combo}_results.csv"
        if cp.exists():
            all_csvs.append(pd.read_csv(cp))
    if all_csvs:
        combined = pd.concat(all_csvs, ignore_index=True)
        combined.to_csv(out_dir / "multiaxis_results.csv", index=False)
        print(f"CSV saved: multiaxis_results.csv")

    # LaTeX table
    latex = generate_latex(combo_summaries, single_refs)
    tex_path = out_dir / "multiaxis_table.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"LaTeX saved: multiaxis_table.tex")

    # Figure
    print("\nGenerating figures...")
    plot_comparison(combo_summaries, single_refs, out_dir / "fig_multiaxis_comparison.png")

    # Final summary table
    print(f"\n{'='*65}")
    print("MULTI-AXIS FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"{'Combo':<28} {'SSIM':>8} {'MfgHit':>8} {'FSHit':>8} {'EchoCloser':>12}")
    print("-" * 65)
    for s in combo_summaries:
        print(f"{s['label']:<28} {s['ssim_mean']:>8.4f} "
              f"{s['mfg_hit_rate']:>8.2%} {s['fs_hit_rate']:>8.2%} "
              f"{s['echo_moved_closer_rate']:>12.2%}")
    print("\nSingle-axis reference:")
    for axis, ref in single_refs.items():
        print(f"  {axis:<24} SSIM={ref['ssim_mean']:.4f}  HitRate={ref['hit_rate']:.2%}")
    print(f"\n✅ Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
