#!/usr/bin/env python3
"""
Evaluate the acquisition-parameter predictor on a split CSV.

Command:
python scripts/evaluate_predictor.py \
  --checkpoint checkpoints/predictor/ax_t1_128/best.pt \
  --test_csv data/processed/ax_t1_128/splits/test.csv \
  --output_dir reports/predictor/ax_t1_128/test_eval

Outputs:
- metrics.json
- predictions.csv
- manufacturer_confusion_matrix.png
- field_strength_confusion_matrix.png
- echo_time_norm_predictions_vs_true.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import make_dataloader
from src.models.predictor import AcquisitionPredictor


def load_checkpoint(path: str | Path, device: torch.device) -> Dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return torch.load(checkpoint_path, map_location=device)


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()

    rows = []

    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)

        manufacturer_prob = torch.softmax(outputs["manufacturer_logits"], dim=1)
        field_prob = torch.softmax(outputs["field_strength_logits"], dim=1)

        manufacturer_pred = manufacturer_prob.argmax(dim=1).detach().cpu().numpy()
        field_pred = field_prob.argmax(dim=1).detach().cpu().numpy()
        echo_pred = outputs["echo_time_norm"].detach().cpu().numpy()

        manufacturer_true = batch["manufacturer"].detach().cpu().numpy()
        field_true = batch["field_strength"].detach().cpu().numpy()
        echo_true = batch["echo_time_norm"].detach().cpu().numpy()
        echo_raw = batch["echo_time"].detach().cpu().numpy()

        image_paths = batch["image_path"]
        patient_ids = batch["patient_id"]

        for i in range(len(image_paths)):
            rows.append(
                {
                    "image_path": image_paths[i],
                    "patient_id": patient_ids[i],
                    "manufacturer_true": int(manufacturer_true[i]),
                    "manufacturer_pred": int(manufacturer_pred[i]),
                    "manufacturer_pred_confidence": float(manufacturer_prob[i, manufacturer_pred[i]].item()),
                    "field_strength_true": int(field_true[i]),
                    "field_strength_pred": int(field_pred[i]),
                    "field_strength_pred_confidence": float(field_prob[i, field_pred[i]].item()),
                    "echo_time_true": float(echo_raw[i]),
                    "echo_time_norm_true": float(echo_true[i]),
                    "echo_time_norm_pred": float(echo_pred[i]),
                }
            )

    return pd.DataFrame(rows)


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_echo_scatter(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.scatter(df["echo_time_norm_true"], df["echo_time_norm_pred"], alpha=0.5)
    plt.xlabel("True EchoTime normalized")
    plt.ylabel("Predicted EchoTime normalized")
    plt.title("EchoTime Regression: Predicted vs True")

    min_val = min(df["echo_time_norm_true"].min(), df["echo_time_norm_pred"].min())
    max_val = max(df["echo_time_norm_true"].max(), df["echo_time_norm_pred"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate acquisition predictor.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--test_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]

    model = AcquisitionPredictor(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_manufacturers=int(config["model"]["num_manufacturers"]),
        num_field_strengths=int(config["model"]["num_field_strengths"]),
        feature_dim=int(config["model"].get("feature_dim", 256)),
        dropout=float(config["model"].get("dropout", 0.20)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    loader = make_dataloader(
        csv_path=args.test_csv,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augment=False,
    )

    pred_df = predict(model, loader, device)
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    manufacturer_cm = confusion_matrix(
        pred_df["manufacturer_true"],
        pred_df["manufacturer_pred"],
    )
    field_cm = confusion_matrix(
        pred_df["field_strength_true"],
        pred_df["field_strength_pred"],
    )

    echo_mae = mean_absolute_error(
        pred_df["echo_time_norm_true"],
        pred_df["echo_time_norm_pred"],
    )
    echo_mse = mean_squared_error(
        pred_df["echo_time_norm_true"],
        pred_df["echo_time_norm_pred"],
    )
    echo_rmse = np.sqrt(echo_mse)

    # Mean-baseline comparison for EchoTime normalized.
    mean_baseline_pred = np.full_like(
        pred_df["echo_time_norm_true"].values,
        fill_value=pred_df["echo_time_norm_true"].mean(),
        dtype=np.float32,
    )
    echo_mean_baseline_mae = mean_absolute_error(
        pred_df["echo_time_norm_true"],
        mean_baseline_pred,
    )

    metrics = {
        "num_samples": int(len(pred_df)),
        "manufacturer_accuracy": float(accuracy_score(pred_df["manufacturer_true"], pred_df["manufacturer_pred"])),
        "manufacturer_balanced_accuracy": float(balanced_accuracy_score(pred_df["manufacturer_true"], pred_df["manufacturer_pred"])),
        "field_strength_accuracy": float(accuracy_score(pred_df["field_strength_true"], pred_df["field_strength_pred"])),
        "field_strength_balanced_accuracy": float(balanced_accuracy_score(pred_df["field_strength_true"], pred_df["field_strength_pred"])),
        "echo_time_norm_mae": float(echo_mae),
        "echo_time_norm_rmse": float(echo_rmse),
        "echo_time_norm_mean_baseline_mae": float(echo_mean_baseline_mae),
        "echo_beats_mean_baseline": bool(echo_mae < echo_mean_baseline_mae),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        manufacturer_cm,
        "Manufacturer Confusion Matrix",
        output_dir / "manufacturer_confusion_matrix.png",
    )
    plot_confusion_matrix(
        field_cm,
        "Field Strength Confusion Matrix",
        output_dir / "field_strength_confusion_matrix.png",
    )
    plot_echo_scatter(
        pred_df,
        output_dir / "echo_time_norm_predictions_vs_true.png",
    )

    print("\n=== Predictor Evaluation ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved predictions: {pred_path}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    print(f"Saved figures in: {output_dir}")


if __name__ == "__main__":
    main()
