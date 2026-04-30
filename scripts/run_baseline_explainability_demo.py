#!/usr/bin/env python3
"""
Run baseline transformations and explanation maps.

Command:
python scripts/run_baseline_explainability_demo.py \
  --metadata_csv data/processed/ax_t1_128/processed_metadata.csv \
  --output_dir reports/baselines/ax_t1_128 \
  --num_examples 5 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.baselines import make_all_baselines
from src.explainability.difference_map import compute_absolute_difference, compute_difference_summary
from src.explainability.ssim_map import compute_local_ssim_map, compute_ssim_summary
from src.explainability.histogram import compute_histogram_comparison


def resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.exists():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    raise FileNotFoundError(f"Image not found: {path_value}")


def load_image(path_value: str | Path) -> np.ndarray:
    path = resolve_path(path_value)
    if path.suffix.lower() == ".npy":
        image = np.load(path).astype(np.float32)
    else:
        import imageio.v2 as imageio
        image = imageio.imread(path).astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
    if image.ndim == 3:
        image = image[..., 0]
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def find_reference_image(df: pd.DataFrame, source_row: pd.Series, seed: int):
    """Pick a reference image from a different manufacturer if possible."""
    candidates = df.copy()
    if "manufacturer" in df.columns:
        candidates = candidates[candidates["manufacturer"] != source_row["manufacturer"]]
    if len(candidates) == 0:
        return None
    ref_row = candidates.sample(n=1, random_state=seed).iloc[0]
    return load_image(ref_row["image_path"])


def save_image_grid(original: np.ndarray, outputs: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    names = ["original"] + list(outputs.keys())
    images = [original] + [outputs[n] for n in outputs.keys()]
    cols = len(images)

    plt.figure(figsize=(3 * cols, 3))
    for i, (name, img) in enumerate(zip(names, images), start=1):
        plt.subplot(1, cols, i)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title(name)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_explanation_grid(original: np.ndarray, generated: np.ndarray, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    diff = compute_absolute_difference(original, generated)
    global_ssim, ssim_map = compute_local_ssim_map(original, generated)

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap="gray", vmin=0, vmax=1)
    plt.title("original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(generated, cmap="gray", vmin=0, vmax=1)
    plt.title("baseline")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(diff)
    plt.title("abs diff")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(ssim_map, vmin=0, vmax=1)
    plt.title(f"SSIM map\nscore={global_ssim:.3f}")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_histogram_plot(original: np.ndarray, generated: np.ndarray, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    comparison = compute_histogram_comparison(original, generated, bins=64)
    bin_edges = comparison["bin_edges"]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(7, 4))
    plt.plot(centers, comparison["hist_original"], label="original")
    plt.plot(centers, comparison["hist_generated"], label="generated")
    plt.xlabel("Normalized intensity")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline explainability demo.")
    parser.add_argument("--metadata_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--num_examples", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata_csv)

    required = ["image_path", "patient_id", "manufacturer", "field_strength", "echo_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    sample_df = df.sample(n=min(args.num_examples, len(df)), random_state=args.seed).reset_index(drop=True)
    records = []

    for example_idx, row in sample_df.iterrows():
        original = load_image(row["image_path"])
        reference = find_reference_image(df, row, seed=args.seed + example_idx)
        outputs = make_all_baselines(original, reference_image=reference, seed=args.seed + example_idx)

        example_name = f"example_{example_idx:03d}"
        title = f"{example_name} | patient={row['patient_id']} | {row['manufacturer']} | {row['field_strength']} | TE={row['echo_time']}"

        save_image_grid(original, outputs, figures_dir / f"{example_name}_baseline_grid.png", title)

        for baseline_name, generated in outputs.items():
            diff_summary = compute_difference_summary(original, generated)
            ssim_summary = compute_ssim_summary(original, generated)
            hist_summary = compute_histogram_comparison(original, generated, bins=64)

            save_explanation_grid(
                original,
                generated,
                figures_dir / f"{example_name}_{baseline_name}_explanation.png",
                f"{example_name} | {baseline_name}",
            )
            save_histogram_plot(
                original,
                generated,
                figures_dir / f"{example_name}_{baseline_name}_histogram.png",
                f"{example_name} | {baseline_name} histogram",
            )

            records.append({
                "example": example_name,
                "baseline": baseline_name,
                "image_path": row["image_path"],
                "patient_id": row["patient_id"],
                "manufacturer": row["manufacturer"],
                "field_strength": row["field_strength"],
                "echo_time": float(row["echo_time"]),
                **diff_summary,
                **ssim_summary,
                "hist_l1_distance": float(hist_summary["l1_distance"]),
                "hist_l2_distance": float(hist_summary["l2_distance"]),
                "hist_intersection": float(hist_summary["intersection"]),
                "original_mean_intensity": float(hist_summary["original_mean_intensity"]),
                "generated_mean_intensity": float(hist_summary["generated_mean_intensity"]),
                "original_std_intensity": float(hist_summary["original_std_intensity"]),
                "generated_std_intensity": float(hist_summary["generated_std_intensity"]),
            })

    results_df = pd.DataFrame(records)
    results_csv = output_dir / "baseline_metrics.csv"
    results_df.to_csv(results_csv, index=False)

    summary = {
        "metadata_csv": args.metadata_csv,
        "num_examples": int(len(sample_df)),
        "num_baseline_records": int(len(results_df)),
        "baselines": sorted(results_df["baseline"].unique().tolist()),
        "mean_metrics_by_baseline": results_df.groupby("baseline")[
            ["mean_absolute_difference", "global_ssim", "hist_l1_distance", "hist_intersection"]
        ].mean().to_dict(),
    }
    summary_path = output_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Baseline Explainability Demo Complete ===")
    print(f"Saved metrics CSV: {results_csv}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved figures in: {figures_dir}")
    print("\nMean metrics by baseline:")
    print(results_df.groupby("baseline")[[
        "mean_absolute_difference",
        "global_ssim",
        "hist_l1_distance",
        "hist_intersection",
    ]].mean())


if __name__ == "__main__":
    main()
