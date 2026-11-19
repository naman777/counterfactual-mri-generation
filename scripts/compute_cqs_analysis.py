#!/usr/bin/env python3
"""
P1.1 — CQS & CADI Analysis Script.

Formalised Explainability Framework for Counterfactual MRI Generation.

Metrics
-------
CQS (Counterfactual Quality Score) — per-example quality:
    CQS = α·SSIM + β·NormGain + γ·Stability   (default α=β=γ=1/3)

CADI (Control Axis Difficulty Index) — per-axis controllability:
    CADI(a) = HitRate(a) × MeanGain(a) / MeanMAD(a)
    Higher = easier to control. Negative = unreliable.

Usage
-----
    python scripts/compute_cqs_analysis.py
    python scripts/compute_cqs_analysis.py --alpha 0.5 --beta 0.3 --gamma 0.2

Outputs
-------
    reports/cqs_cadi/cqs_per_example.csv      ← per-example CQS scores
    reports/cqs_cadi/cadi_summary.json        ← CADI per axis + summary stats
    reports/cqs_cadi/cadi_summary.csv         ← same as CSV
    reports/cqs_cadi/cqs_cadi_table.tex       ← LaTeX table (paper-ready)
    reports/cqs_cadi/fig_cqs_distribution.png ← CQS distribution figure
    reports/cqs_cadi/fig_cadi_ranking.png     ← CADI bar chart
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.engine import ExplainabilityEngine


# ─── Configuration ────────────────────────────────────────────────────────────

AXES = {
    "field_strength": {
        "csv":        "reports/phase12/field_strength/field_strength_results.csv",
        "gain_col":   "fs_gain",
        "hit_col":    "fs_hit",
        "label":      "Field Strength",
        "color":      "#4C9BE8",
    },
    "manufacturer": {
        "csv":        "reports/phase12/manufacturer/manufacturer_results.csv",
        "gain_col":   "mfg_gain",
        "hit_col":    "mfg_hit",
        "label":      "Manufacturer",
        "color":      "#E8834C",
    },
    "echo_time": {
        "csv":        "reports/phase12/echo_time/echo_time_results.csv",
        "gain_col":   "echo_improvement",
        "hit_col":    "echo_moved_closer",
        "label":      "Echo Time",
        "color":      "#4CE89B",
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_axis_df(control: str) -> pd.DataFrame:
    cfg = AXES[control]
    csv_path = PROJECT_ROOT / cfg["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["control"] = control
    return df


def compute_cqs_for_df(
    df: pd.DataFrame,
    control: str,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.DataFrame:
    """
    Compute CQS for every row of a results DataFrame using the formula:

        CQS = α·SSIM + β·NormGain + γ·Stability

    NormGain maps raw gain ∈ [-1, 1] → [0, 1] via (gain + 1) / 2.
    Stability = 1 if both unchanged axes stayed stable, else 0.
    """
    cfg = AXES[control]
    gain_col = cfg["gain_col"]

    # SSIM component (already [0, 1])
    ssim_comp = df["global_ssim"].clip(0.0, 1.0)

    # Gain component: normalise raw gain to [0, 1]
    raw_gain = df[gain_col].astype(float)
    gain_comp = ((raw_gain + 1.0) / 2.0).clip(0.0, 1.0)

    # Stability component
    if control == "field_strength":
        # mfg was NOT edited → check mfg_stable; fs was edited → skip
        stab = df["mfg_stable"].astype(float)
    elif control == "manufacturer":
        # fs was NOT edited → check fs_stable; mfg was edited → skip
        stab = df["fs_stable"].astype(float)
    elif control == "echo_time":
        # both mfg and fs were NOT edited
        stab = ((df["mfg_stable"].astype(float) + df["fs_stable"].astype(float)) / 2.0)
    else:
        stab = pd.Series(np.ones(len(df)), index=df.index)

    cqs = alpha * ssim_comp + beta * gain_comp + gamma * stab

    result = df[["example_id", "patient_id", "control", "global_ssim",
                 "mean_abs_diff"]].copy()
    result["raw_gain"]          = raw_gain.values
    result["ssim_component"]    = ssim_comp.values
    result["gain_component"]    = gain_comp.values
    result["stability_component"] = stab.values
    result["cqs"]               = cqs.values
    result["alpha"]             = alpha
    result["beta"]              = beta
    result["gamma"]             = gamma
    return result


def compute_cadi_for_axis(df: pd.DataFrame, control: str) -> dict:
    """Compute CADI for one axis from its results DataFrame."""
    cfg = AXES[control]
    hit_rate  = df[cfg["hit_col"]].astype(float).mean()
    mean_gain = df[cfg["gain_col"]].astype(float).mean()
    mean_mad  = df["mean_abs_diff"].mean()

    cadi = ExplainabilityEngine.compute_cadi(hit_rate, mean_gain, mean_mad)
    return {
        "control":    control,
        "label":      cfg["label"],
        "hit_rate":   float(hit_rate),
        "mean_gain":  float(mean_gain),
        "mean_mad":   float(mean_mad),
        "cadi":       float(cadi),
        "n":          int(len(df)),
    }


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_cqs_distribution(cqs_df: pd.DataFrame, out_path: Path):
    """KDE/violin plot of CQS distribution per control axis."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0D1117")
    for ax in axes:
        ax.set_facecolor("#161B22")

    # Left: violin / box per axis
    ax = axes[0]
    controls  = list(AXES.keys())
    data      = [cqs_df[cqs_df["control"] == c]["cqs"].values for c in controls]
    colors    = [AXES[c]["color"] for c in controls]
    labels    = [AXES[c]["label"] for c in controls]
    positions = range(len(controls))

    vparts = ax.violinplot(data, positions=positions, showmedians=True,
                           showextrema=True)
    for i, (pc, col) in enumerate(zip(vparts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        vparts[part].set_color("white")
        vparts[part].set_linewidth(1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylabel("CQS", color="white", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("CQS Distribution per Control Axis", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363D")

    # Right: scatter — SSIM vs Gain coloured by CQS
    ax = axes[1]
    for control in controls:
        sub = cqs_df[cqs_df["control"] == control]
        sc = ax.scatter(
            sub["ssim_component"],
            sub["gain_component"],
            c=sub["cqs"],
            cmap="plasma",
            alpha=0.6,
            s=30,
            label=AXES[control]["label"],
        )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("CQS", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("SSIM Component", color="white", fontsize=11)
    ax.set_ylabel("Gain Component", color="white", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("SSIM vs Gain (coloured by CQS)", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363D")
    patches = [mpatches.Patch(color=AXES[c]["color"], label=AXES[c]["label"])
               for c in controls]
    ax.legend(handles=patches, fontsize=9, facecolor="#1C2128",
              labelcolor="white", edgecolor="#30363D")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figure saved: {out_path.name}")


def plot_cadi_ranking(cadi_rows: list[dict], out_path: Path):
    """Horizontal bar chart ranking axes by CADI."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    labels = [r["label"] for r in cadi_rows]
    cadis  = [r["cadi"]  for r in cadi_rows]
    colors = [AXES[r["control"]]["color"] for r in cadi_rows]

    sorted_idx = np.argsort(cadis)[::-1]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_cadis  = [cadis[i]  for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(sorted_labels, sorted_cadis, color=sorted_colors,
                   alpha=0.8, edgecolor="#30363D", height=0.5)
    for bar, val in zip(bars, sorted_cadis):
        ax.text(max(val + 0.05, 0.05), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", color="white", fontsize=11,
                fontweight="bold")

    ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("CADI  (HitRate × MeanGain / MeanMAD)", color="white",
                  fontsize=11)
    ax.set_title("Control Axis Difficulty Index (CADI)\nHigher = Easier to Control",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363D")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figure saved: {out_path.name}")


# ─── LaTeX Table ──────────────────────────────────────────────────────────────

def generate_latex_table(
    cadi_rows: list[dict],
    cqs_df: pd.DataFrame,
    alpha: float, beta: float, gamma: float,
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Control Axis Difficulty Index (CADI) and mean Counterfactual Quality Score "
         r"(CQS, $\alpha=\beta=\gamma=" + f"{alpha:.2f}" + r"$) per control axis. "
         r"Higher CADI $\uparrow$ = easier to control. "
         r"CQS $\in [0,1]$, higher is better.}"),
        r"\label{tab:cqs_cadi}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        (r"Control Axis & Hit Rate $\uparrow$ & Mean Gain $\uparrow$ & "
         r"Mean MAD $\downarrow$ & CADI $\uparrow$ & Mean CQS $\uparrow$ \\"),
        r"\midrule",
    ]
    for row in sorted(cadi_rows, key=lambda r: r["cadi"], reverse=True):
        ctrl = row["control"]
        mean_cqs = cqs_df[cqs_df["control"] == ctrl]["cqs"].mean()
        lines.append(
            f"  {row['label']} & {row['hit_rate']:.2f} & "
            f"{row['mean_gain']:+.3f} & {row['mean_mad']:.4f} & "
            f"\\textbf{{{row['cadi']:.2f}}} & {mean_cqs:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P1.1 — CQS & CADI Analysis")
    parser.add_argument("--alpha", type=float, default=1 / 3,
                        help="CQS weight for SSIM (anatomy)")
    parser.add_argument("--beta",  type=float, default=1 / 3,
                        help="CQS weight for gain (target alignment)")
    parser.add_argument("--gamma", type=float, default=1 / 3,
                        help="CQS weight for stability (cross-axis)")
    parser.add_argument("--output_dir", default="reports/cqs_cadi/")
    args = parser.parse_args()

    # Normalise weights
    total = args.alpha + args.beta + args.gamma
    alpha, beta, gamma = args.alpha / total, args.beta / total, args.gamma / total
    print(f"\nCQS weights: α={alpha:.3f}  β={beta:.3f}  γ={gamma:.3f}")

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-example CQS ──────────────────────────────────────────────────────
    all_cqs_dfs, cadi_rows = [], []

    for control in AXES:
        print(f"\n{'─'*52}")
        print(f"Processing: {control.upper()}")
        print(f"{'─'*52}")
        try:
            df = load_axis_df(control)
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping: {e}")
            continue

        cqs_df_axis = compute_cqs_for_df(df, control, alpha, beta, gamma)
        cadi_row    = compute_cadi_for_axis(df, control)

        mean_cqs = cqs_df_axis["cqs"].mean()
        std_cqs  = cqs_df_axis["cqs"].std()

        print(f"  n = {len(cqs_df_axis)}")
        print(f"  Mean CQS  = {mean_cqs:.4f} ± {std_cqs:.4f}")
        print(f"  CADI      = {cadi_row['cadi']:.4f}")
        print(f"    (HitRate={cadi_row['hit_rate']:.3f}  "
              f"MeanGain={cadi_row['mean_gain']:+.4f}  "
              f"MeanMAD={cadi_row['mean_mad']:.5f})")

        all_cqs_dfs.append(cqs_df_axis)
        cadi_rows.append(cadi_row)

    if not all_cqs_dfs:
        print("\nERROR: No results loaded. Check that phase12 CSVs exist.")
        sys.exit(1)

    combined_cqs = pd.concat(all_cqs_dfs, ignore_index=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    cqs_csv  = out_dir / "cqs_per_example.csv"
    cadi_csv = out_dir / "cadi_summary.csv"
    combined_cqs.to_csv(cqs_csv, index=False)
    pd.DataFrame(cadi_rows).to_csv(cadi_csv, index=False)
    print(f"\nCSV saved: {cqs_csv.name}")
    print(f"CSV saved: {cadi_csv.name}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    cadi_json = out_dir / "cadi_summary.json"
    with open(cadi_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
                "axes": cadi_rows,
                "cqs_summary": [
                    {
                        "control": c,
                        "mean_cqs": float(combined_cqs[combined_cqs["control"] == c]["cqs"].mean()),
                        "std_cqs":  float(combined_cqs[combined_cqs["control"] == c]["cqs"].std()),
                        "min_cqs":  float(combined_cqs[combined_cqs["control"] == c]["cqs"].min()),
                        "max_cqs":  float(combined_cqs[combined_cqs["control"] == c]["cqs"].max()),
                    }
                    for c in AXES if c in combined_cqs["control"].values
                ],
            },
            f, indent=2,
        )
    print(f"JSON saved: {cadi_json.name}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    latex = generate_latex_table(cadi_rows, combined_cqs, alpha, beta, gamma)
    tex_path = out_dir / "cqs_cadi_table.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"LaTeX saved: {tex_path.name}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_cqs_distribution(combined_cqs, out_dir / "fig_cqs_distribution.png")
    plot_cadi_ranking(cadi_rows, out_dir / "fig_cadi_ranking.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CQS / CADI SUMMARY")
    print(f"{'='*60}")
    print(f"{'Axis':<20} {'CADI':>8} {'Hit%':>8} {'MeanGain':>10} {'MeanCQS':>10}")
    print("-" * 60)
    for row in sorted(cadi_rows, key=lambda r: r["cadi"], reverse=True):
        ctrl = row["control"]
        mean_cqs = combined_cqs[combined_cqs["control"] == ctrl]["cqs"].mean()
        print(f"{row['label']:<20} {row['cadi']:>8.3f} "
              f"{row['hit_rate']*100:>7.1f}% "
              f"{row['mean_gain']:>+10.4f} "
              f"{mean_cqs:>10.4f}")

    print(f"\n✅ Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
