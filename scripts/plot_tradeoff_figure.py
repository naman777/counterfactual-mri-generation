#!/usr/bin/env python3
"""
Phase 12 — Step 12.7: Tradeoff Scatter Plot.

Produces the SSIM vs. Predictor Gain scatter plot for each control axis.
This is a key paper figure showing the anatomy preservation vs.
target alignment tradeoff.

Usage:
    python scripts/plot_tradeoff_figure.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


CONTROL_LABELS = {
    "field_strength": "Field Strength",
    "echo_time": "EchoTime",
    "manufacturer": "Manufacturer",
}

# Visual styling
CONTROL_COLORS = {
    "field_strength": "#4285F4",  # Google Blue
    "echo_time": "#34A853",       # Google Green
    "manufacturer": "#EA4335",    # Google Red
}

CONTROL_MARKERS = {
    "field_strength": "o",
    "echo_time": "s",
    "manufacturer": "^",
}


def load_per_example_metrics(phase12_dir: Path, control: str) -> list:
    """Load all per-example metrics JSONs for a control axis."""
    metrics_dir = phase12_dir / control / "metrics"
    if not metrics_dir.exists():
        print(f"Warning: {metrics_dir} not found")
        return []

    metrics_files = sorted(metrics_dir.glob("*_metrics.json"))
    all_metrics = []
    for mf in metrics_files:
        with open(mf, "r", encoding="utf-8") as f:
            all_metrics.append(json.load(f))

    return all_metrics


def plot_single_axis_scatter(
    ax: plt.Axes,
    metrics_list: list,
    control: str,
    title: str,
):
    """Plot scatter for a single control axis."""
    if not metrics_list:
        ax.set_title(f"{title} (no data)")
        return

    ssims = []
    gains = []

    for m in metrics_list:
        ssim = m.get("global_ssim", 0)
        ssims.append(ssim)

        if control == "field_strength":
            gain = m.get("predictor_gain", {}).get("fs_gain", 0)
        elif control == "echo_time":
            gain = m.get("predictor_gain", {}).get("echo_improvement", 0)
        else:
            gain = m.get("predictor_gain", {}).get("mfg_gain", 0)
        gains.append(gain)

    color = CONTROL_COLORS[control]
    marker = CONTROL_MARKERS[control]

    ax.scatter(
        ssims, gains,
        c=color,
        marker=marker,
        s=40,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    # Add mean point
    mean_ssim = np.mean(ssims)
    mean_gain = np.mean(gains)
    ax.scatter(
        [mean_ssim], [mean_gain],
        c=color,
        marker="D",
        s=120,
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
        label=f"Mean ({mean_ssim:.3f}, {mean_gain:.3f})",
    )

    # Styling
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Global SSIM (anatomy preservation) →", fontsize=9)
    ax.set_ylabel("Target Gain (alignment) →", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Reference lines
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Stats annotation
    n = len(ssims)
    stats_text = (
        f"N={n}\n"
        f"SSIM: {mean_ssim:.3f}±{np.std(ssims):.3f}\n"
        f"Gain: {mean_gain:.3f}±{np.std(gains):.3f}"
    )
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def plot_combined_3panel(
    phase12_dir: Path,
    output_dir: Path,
    controls: list,
    dpi: int = 300,
):
    """Produce the combined 3-panel tradeoff scatter figure."""
    print("\n--- Generating tradeoff scatter plots ---")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=dpi)
    fig.suptitle(
        "Anatomy Preservation vs. Target Alignment Tradeoff",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, control in zip(axes, controls):
        metrics = load_per_example_metrics(phase12_dir, control)
        print(f"  {control}: loaded {len(metrics)} examples")

        plot_single_axis_scatter(
            ax=ax,
            metrics_list=metrics,
            control=control,
            title=CONTROL_LABELS[control],
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "fig_tradeoff_combined_3panel.png"
    pdf_path = output_dir / "fig_tradeoff_combined_3panel.pdf"

    fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(str(pdf_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def plot_individual_axis(
    phase12_dir: Path,
    output_dir: Path,
    control: str,
    dpi: int = 300,
):
    """Plot individual scatter for one control axis."""
    metrics = load_per_example_metrics(phase12_dir, control)
    if not metrics:
        print(f"  No data for {control}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5), dpi=dpi)

    plot_single_axis_scatter(
        ax=ax,
        metrics_list=metrics,
        control=control,
        title=f"{CONTROL_LABELS[control]} — SSIM vs. Target Gain",
    )

    fig.tight_layout()

    png_path = output_dir / f"fig_tradeoff_{control}.png"
    pdf_path = output_dir / f"fig_tradeoff_{control}.pdf"

    fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(str(pdf_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def plot_ablation_scatter(
    phase12_dir: Path,
    output_dir: Path,
    dpi: int = 300,
):
    """
    Plot ablation results as SSIM vs Gain with color=timestep, shape=guidance.
    """
    json_path = phase12_dir / "ablation" / "ablation_results.json"
    if not json_path.exists():
        print("  No ablation results found")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5), dpi=dpi)

    # Color by timestep, marker by guidance
    timesteps = sorted(set(r["edit_timestep"] for r in results))
    guidances = sorted(set(r["guidance_scale"] for r in results))

    cmap = plt.cm.viridis
    ts_colors = {ts: cmap(i / max(len(timesteps) - 1, 1)) for i, ts in enumerate(timesteps)}
    gs_markers = {g: m for g, m in zip(guidances, ["o", "s", "^", "D", "v", "P"])}

    for r in results:
        ax.scatter(
            r["ssim_mean"], r["fs_gain_mean"],
            c=[ts_colors[r["edit_timestep"]]],
            marker=gs_markers.get(r["guidance_scale"], "o"),
            s=100,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        # Error bars
        ax.errorbar(
            r["ssim_mean"], r["fs_gain_mean"],
            xerr=r["ssim_std"],
            yerr=r["fs_gain_std"],
            fmt="none",
            ecolor="gray",
            alpha=0.4,
            capsize=2,
        )

    # Legend
    ts_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ts_colors[ts],
               markersize=8, label=f"t={ts}")
        for ts in timesteps
    ]
    gs_handles = [
        Line2D([0], [0], marker=gs_markers[g], color="w", markerfacecolor="gray",
               markersize=8, label=f"γ={g:.1f}")
        for g in guidances
    ]

    legend1 = ax.legend(
        handles=ts_handles, title="Edit Timestep",
        loc="lower left", fontsize=7, title_fontsize=8,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=gs_handles, title="Guidance Scale",
        loc="lower right", fontsize=7, title_fontsize=8,
    )

    ax.set_title("Ablation: SSIM vs. Field Strength Gain", fontsize=11, fontweight="bold")
    ax.set_xlabel("Global SSIM (anatomy preservation) →", fontsize=9)
    ax.set_ylabel("FS Target Gain (alignment) →", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)

    fig.tight_layout()

    png_path = output_dir / "fig_ablation_scatter.png"
    pdf_path = output_dir / "fig_ablation_scatter.pdf"

    fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(str(pdf_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 12 — Tradeoff Scatter Plot Generator"
    )
    parser.add_argument("--phase12_dir", type=str,
                        default="reports/phase12/")
    parser.add_argument("--output_dir", type=str,
                        default="reports/paper_figures/")
    parser.add_argument("--controls", nargs="+",
                        default=["field_strength", "echo_time", "manufacturer"])
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    phase12_dir = Path(PROJECT_ROOT / args.phase12_dir)
    output_dir = Path(PROJECT_ROOT / args.output_dir)

    print("=" * 70)
    print("Phase 12 — Tradeoff Scatter Plot Generator")
    print("=" * 70)
    print(f"Phase 12 dir: {phase12_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Controls:     {args.controls}")

    # Combined 3-panel figure
    plot_combined_3panel(phase12_dir, output_dir, args.controls, dpi=args.dpi)

    # Individual per-axis figures
    for control in args.controls:
        plot_individual_axis(phase12_dir, output_dir, control, dpi=args.dpi)

    # Ablation scatter
    plot_ablation_scatter(phase12_dir, output_dir, dpi=args.dpi)

    print(f"\n{'='*70}")
    print("TRADEOFF PLOTS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
