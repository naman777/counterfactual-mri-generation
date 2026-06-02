#!/usr/bin/env python3
"""
Phase 10 — Publication-Quality Panel Generator.

Produces a 6-column explainability panel:
  [Original] [Counterfactual] [Diff Map] [SSIM Map] [Histogram] [Predictor Table]

All panels are rendered at 300 DPI for paper submission.
Saves as both PNG (for reports) and PDF (for paper).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch


def generate_publication_panel(
    original: np.ndarray,
    counterfactual: np.ndarray,
    results: Dict,
    source_condition: Dict,
    target_condition: Dict,
    save_path: str,
    title: Optional[str] = None,
    dpi: int = 300,
):
    """
    Generate a publication-quality 6-column explainability panel.

    Layout: [Original] [Counterfactual] [Diff Map] [SSIM Map] [Histogram] [Predictor Table]

    Args:
        original: numpy [H, W] in [0, 1]
        counterfactual: numpy [H, W] in [0, 1]
        results: output of ExplainabilityEngine.compute_all()
        source_condition: dict with manufacturer_label, field_strength_label, echo_time_norm
        target_condition: dict with manufacturer_label, field_strength_label, echo_time_norm
        save_path: base path (without extension). Will save .png and .pdf
        title: optional super-title for the panel
        dpi: figure DPI (default 300 for publication)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract data from results
    diff_map = results["diff_map"]
    ssim_map = results["ssim_map"]
    global_ssim = results["global_ssim"]
    mean_abs_diff = results["mean_abs_diff"]
    hist_data = results["histogram_data"]
    pred_before = results["predictor_before"]
    pred_after = results["predictor_after"]
    pred_gain = results["predictor_gain"]

    # Label decoding maps
    MFG_LABELS = {0: "GE", 1: "SIEMENS"}
    FS_LABELS = {0: "1.5T", 1: "3.0T"}

    # Decode conditions
    src_mfg = MFG_LABELS.get(source_condition.get("manufacturer_label", -1), "?")
    tgt_mfg = MFG_LABELS.get(target_condition.get("manufacturer_label", -1), "?")
    src_fs = FS_LABELS.get(source_condition.get("field_strength_label", -1), "?")
    tgt_fs = FS_LABELS.get(target_condition.get("field_strength_label", -1), "?")
    src_echo = source_condition.get("echo_time_norm", 0.0)
    tgt_echo = target_condition.get("echo_time_norm", 0.0)

    # --- Create figure ---
    fig = plt.figure(figsize=(18, 3.8), dpi=dpi)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.25, hspace=0.05,
                           left=0.03, right=0.97, top=0.88, bottom=0.08)

    # --- Panel 1: Original ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax1.set_title("Original", fontsize=9, fontweight="bold")
    _label_bottom(ax1, f"{src_mfg} | {src_fs} | ET={src_echo:.3f}")
    ax1.axis("off")

    # --- Panel 2: Counterfactual ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(counterfactual, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax2.set_title("Counterfactual", fontsize=9, fontweight="bold")
    _label_bottom(ax2, f"Target: {tgt_mfg} | {tgt_fs} | ET={tgt_echo:.3f}")
    ax2.axis("off")

    # --- Panel 3: Difference Map ---
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(diff_map, cmap="hot", vmin=0, vmax=1, aspect="equal")
    ax3.set_title("Difference Map", fontsize=9, fontweight="bold")
    _label_bottom(ax3, f"MAD={mean_abs_diff:.4f}")
    ax3.axis("off")
    # Small colorbar
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    cbar3.ax.tick_params(labelsize=6)

    # --- Panel 4: SSIM Map ---
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(ssim_map, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax4.set_title("SSIM Map", fontsize=9, fontweight="bold")
    _label_bottom(ax4, f"Global SSIM={global_ssim:.4f}")
    ax4.axis("off")
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    cbar4.ax.tick_params(labelsize=6)

    # --- Panel 5: Histogram ---
    ax5 = fig.add_subplot(gs[0, 4])
    bins = np.array(hist_data["bins"])
    orig_hist = np.array(hist_data["orig_hist"])
    cf_hist = np.array(hist_data["cf_hist"])
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    ax5.fill_between(bin_centers, orig_hist, alpha=0.5, color="#4285F4", label="Original")
    ax5.fill_between(bin_centers, cf_hist, alpha=0.5, color="#EA4335", label="Counterfactual")
    ax5.plot(bin_centers, orig_hist, color="#4285F4", linewidth=0.8)
    ax5.plot(bin_centers, cf_hist, color="#EA4335", linewidth=0.8)
    ax5.set_title("Histogram", fontsize=9, fontweight="bold")
    ax5.set_xlabel("Intensity", fontsize=7)
    ax5.set_ylabel("Density", fontsize=7)
    ax5.tick_params(labelsize=6)
    ax5.legend(fontsize=6, loc="upper right", framealpha=0.7)
    hist_l1 = results.get("hist_l1", 0)
    hist_int = results.get("hist_intersection", 0)
    _label_bottom(ax5, f"L1={hist_l1:.3f} | Int={hist_int:.3f}")

    # --- Panel 6: Predictor Table ---
    ax6 = fig.add_subplot(gs[0, 5])
    ax6.axis("off")
    ax6.set_title("Predictor Shift", fontsize=9, fontweight="bold")

    # Build table data
    table_data = _build_predictor_table(
        pred_before, pred_after, pred_gain,
        source_condition, target_condition,
        MFG_LABELS, FS_LABELS,
    )

    table = ax6.table(
        cellText=table_data["rows"],
        colLabels=table_data["headers"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)

    # Style table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#E8EAED")
            cell.set_text_props(fontweight="bold", fontsize=7)
        else:
            cell.set_facecolor("#FFFFFF")
            # Color gain column
            if col == 4 and row > 0:
                try:
                    gain_val = float(table_data["rows"][row - 1][4])
                    if gain_val > 0.01:
                        cell.set_facecolor("#E6F4EA")  # Light green
                    elif gain_val < -0.01:
                        cell.set_facecolor("#FCE8E6")  # Light red
                except (ValueError, IndexError):
                    pass

    # Save PNG and PDF
    png_path = str(save_path) + ".png"
    pdf_path = str(save_path) + ".pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Panel saved: {png_path}")
    print(f"Panel saved: {pdf_path}")

    return {"png_path": png_path, "pdf_path": pdf_path}


def generate_combined_panel(
    panels_data: list,
    save_path: str,
    title: Optional[str] = None,
    dpi: int = 300,
):
    """
    Generate a combined multi-row panel for the paper.

    Args:
        panels_data: list of dicts, each with:
            original, counterfactual, results, source_condition, target_condition, row_label
        save_path: base path (without extension)
        title: super-title
        dpi: figure DPI
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(panels_data)
    fig_height = 3.2 * n_rows + 0.8

    fig = plt.figure(figsize=(18, fig_height), dpi=dpi)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(
        n_rows, 6, figure=fig, wspace=0.25, hspace=0.35,
        left=0.03, right=0.97, top=0.94, bottom=0.03,
    )

    MFG_LABELS = {0: "GE", 1: "SIEMENS"}
    FS_LABELS = {0: "1.5T", 1: "3.0T"}

    for row_idx, panel in enumerate(panels_data):
        original = panel["original"]
        counterfactual = panel["counterfactual"]
        results = panel["results"]
        source_cond = panel["source_condition"]
        target_cond = panel["target_condition"]
        row_label = panel.get("row_label", f"Example {row_idx + 1}")

        diff_map = results["diff_map"]
        ssim_map = results["ssim_map"]
        global_ssim = results["global_ssim"]
        mean_abs_diff = results["mean_abs_diff"]
        hist_data = results["histogram_data"]
        pred_before = results["predictor_before"]
        pred_after = results["predictor_after"]
        pred_gain = results["predictor_gain"]

        src_mfg = MFG_LABELS.get(source_cond.get("manufacturer_label", -1), "?")
        tgt_mfg = MFG_LABELS.get(target_cond.get("manufacturer_label", -1), "?")
        src_fs = FS_LABELS.get(source_cond.get("field_strength_label", -1), "?")
        tgt_fs = FS_LABELS.get(target_cond.get("field_strength_label", -1), "?")
        src_echo = source_cond.get("echo_time_norm", 0.0)
        tgt_echo = target_cond.get("echo_time_norm", 0.0)

        # Column 0: Original
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(original, cmap="gray", vmin=0, vmax=1)
        if row_idx == 0:
            ax.set_title("Original", fontsize=9, fontweight="bold")
        ax.set_ylabel(row_label, fontsize=8, fontweight="bold", rotation=90, labelpad=10)
        _label_bottom(ax, f"{src_mfg}|{src_fs}|ET={src_echo:.2f}")
        ax.axis("off")

        # Column 1: Counterfactual
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(counterfactual, cmap="gray", vmin=0, vmax=1)
        if row_idx == 0:
            ax.set_title("Counterfactual", fontsize=9, fontweight="bold")
        _label_bottom(ax, f"{tgt_mfg}|{tgt_fs}|ET={tgt_echo:.2f}")
        ax.axis("off")

        # Column 2: Diff map
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.imshow(diff_map, cmap="hot", vmin=0, vmax=1)
        if row_idx == 0:
            ax.set_title("Diff Map", fontsize=9, fontweight="bold")
        _label_bottom(ax, f"MAD={mean_abs_diff:.4f}")
        ax.axis("off")

        # Column 3: SSIM map
        ax = fig.add_subplot(gs[row_idx, 3])
        ax.imshow(ssim_map, cmap="gray", vmin=0, vmax=1)
        if row_idx == 0:
            ax.set_title("SSIM Map", fontsize=9, fontweight="bold")
        _label_bottom(ax, f"SSIM={global_ssim:.4f}")
        ax.axis("off")

        # Column 4: Histogram
        ax = fig.add_subplot(gs[row_idx, 4])
        bins = np.array(hist_data["bins"])
        orig_hist = np.array(hist_data["orig_hist"])
        cf_hist = np.array(hist_data["cf_hist"])
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        ax.fill_between(bin_centers, orig_hist, alpha=0.5, color="#4285F4")
        ax.fill_between(bin_centers, cf_hist, alpha=0.5, color="#EA4335")
        ax.plot(bin_centers, orig_hist, color="#4285F4", linewidth=0.8)
        ax.plot(bin_centers, cf_hist, color="#EA4335", linewidth=0.8)
        if row_idx == 0:
            ax.set_title("Histogram", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=5)

        # Column 5: Predictor table (simplified for multi-row)
        ax = fig.add_subplot(gs[row_idx, 5])
        ax.axis("off")
        if row_idx == 0:
            ax.set_title("Predictor Shift", fontsize=9, fontweight="bold")

        table_data = _build_predictor_table(
            pred_before, pred_after, pred_gain,
            source_cond, target_cond,
            MFG_LABELS, FS_LABELS,
        )
        table = ax.table(
            cellText=table_data["rows"],
            colLabels=table_data["headers"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.0, 1.4)

        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_facecolor("#E8EAED")
                cell.set_text_props(fontweight="bold", fontsize=6)

    # Save
    png_path = str(save_path) + ".png"
    pdf_path = str(save_path) + ".pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Combined panel saved: {png_path}")
    print(f"Combined panel saved: {pdf_path}")

    return {"png_path": png_path, "pdf_path": pdf_path}


# ─── Helper functions ───────────────────────────────────────────────


def _label_bottom(ax, text: str, fontsize: int = 6):
    """Add a small label below an axis."""
    ax.text(
        0.5, -0.08, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="center",
        va="top",
        color="#555555",
    )


def _build_predictor_table(
    pred_before, pred_after, pred_gain,
    source_condition, target_condition,
    mfg_labels, fs_labels,
) -> Dict:
    """Build the predictor shift table data for matplotlib table."""
    # Decode predictions
    before_mfg = mfg_labels.get(pred_before["mfg_pred_label"], "?")
    after_mfg = mfg_labels.get(pred_after["mfg_pred_label"], "?")
    target_mfg = mfg_labels.get(target_condition.get("manufacturer_label", -1), "?")

    before_fs = fs_labels.get(pred_before["fs_pred_label"], "?")
    after_fs = fs_labels.get(pred_after["fs_pred_label"], "?")
    target_fs = fs_labels.get(target_condition.get("field_strength_label", -1), "?")

    before_echo = f"{pred_before['echo_norm_pred']:.3f}"
    after_echo = f"{pred_after['echo_norm_pred']:.3f}"
    target_echo = f"{target_condition.get('echo_time_norm', 0):.3f}"

    # Format gains
    mfg_gain_str = f"{pred_gain['mfg_gain']:+.4f}"
    fs_gain_str = f"{pred_gain['fs_gain']:+.4f}"
    echo_gain_str = f"{pred_gain['echo_improvement']:+.4f}"

    headers = ["Control", "Before", "After", "Target", "Gain"]
    rows = [
        ["Mfg", before_mfg, after_mfg, target_mfg, mfg_gain_str],
        ["Field", before_fs, after_fs, target_fs, fs_gain_str],
        ["Echo", before_echo, after_echo, target_echo, echo_gain_str],
    ]

    return {"headers": headers, "rows": rows}
