#!/usr/bin/env python3
"""
Train the acquisition-parameter predictor.

Command:
python src/training/train_predictor.py --config configs/predictor.yaml

Outputs:
- checkpoints/predictor/ax_t1_128/best.pt
- checkpoints/predictor/ax_t1_128/latest.pt
- reports/predictor/ax_t1_128/history.csv
- reports/predictor/ax_t1_128/training_summary.json
- reports/predictor/ax_t1_128/loss_curve.png
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error
from tqdm import tqdm

# Allow running from project root without installing as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import make_dataloader
from src.models.predictor import AcquisitionPredictor, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_device(config_device: str) -> torch.device:
    if config_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(config_device)


def compute_class_weights(csv_path: str | Path, label_col: str, num_classes: int) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    counts = df[label_col].value_counts().to_dict()

    weights = []
    total = len(df)
    for class_idx in range(num_classes):
        count = counts.get(class_idx, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    manufacturer_loss_fn: nn.Module,
    field_strength_loss_fn: nn.Module,
    echo_loss_fn: nn.Module,
    echo_loss_weight: float,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_manufacturer_loss = 0.0
    total_field_loss = 0.0
    total_echo_loss = 0.0
    n_samples = 0

    all_manufacturer_true = []
    all_manufacturer_pred = []
    all_field_true = []
    all_field_pred = []
    all_echo_true = []
    all_echo_pred = []

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        manufacturer_targets = batch["manufacturer"].to(device)
        field_targets = batch["field_strength"].to(device)
        echo_targets = batch["echo_time_norm"].to(device)

        outputs = model(images)

        manufacturer_loss = manufacturer_loss_fn(
            outputs["manufacturer_logits"],
            manufacturer_targets,
        )
        field_loss = field_strength_loss_fn(
            outputs["field_strength_logits"],
            field_targets,
        )
        echo_loss = echo_loss_fn(
            outputs["echo_time_norm"],
            echo_targets,
        )

        loss = manufacturer_loss + field_loss + echo_loss_weight * echo_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        n_samples += batch_size

        total_loss += loss.item() * batch_size
        total_manufacturer_loss += manufacturer_loss.item() * batch_size
        total_field_loss += field_loss.item() * batch_size
        total_echo_loss += echo_loss.item() * batch_size

        all_manufacturer_true.extend(manufacturer_targets.detach().cpu().numpy().tolist())
        all_manufacturer_pred.extend(outputs["manufacturer_logits"].argmax(dim=1).detach().cpu().numpy().tolist())

        all_field_true.extend(field_targets.detach().cpu().numpy().tolist())
        all_field_pred.extend(outputs["field_strength_logits"].argmax(dim=1).detach().cpu().numpy().tolist())

        all_echo_true.extend(echo_targets.detach().cpu().numpy().tolist())
        all_echo_pred.extend(outputs["echo_time_norm"].detach().cpu().numpy().tolist())

    metrics = {
        "loss": total_loss / n_samples,
        "manufacturer_loss": total_manufacturer_loss / n_samples,
        "field_strength_loss": total_field_loss / n_samples,
        "echo_loss": total_echo_loss / n_samples,
        "manufacturer_accuracy": accuracy_score(all_manufacturer_true, all_manufacturer_pred),
        "manufacturer_balanced_accuracy": balanced_accuracy_score(all_manufacturer_true, all_manufacturer_pred),
        "field_strength_accuracy": accuracy_score(all_field_true, all_field_pred),
        "field_strength_balanced_accuracy": balanced_accuracy_score(all_field_true, all_field_pred),
        "echo_time_norm_mae": mean_absolute_error(all_echo_true, all_echo_pred),
    }

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    manufacturer_loss_fn: nn.Module,
    field_strength_loss_fn: nn.Module,
    echo_loss_fn: nn.Module,
    echo_loss_weight: float,
    device: torch.device,
    desc: str = "val",
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_manufacturer_loss = 0.0
    total_field_loss = 0.0
    total_echo_loss = 0.0
    n_samples = 0

    all_manufacturer_true = []
    all_manufacturer_pred = []
    all_field_true = []
    all_field_pred = []
    all_echo_true = []
    all_echo_pred = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        manufacturer_targets = batch["manufacturer"].to(device)
        field_targets = batch["field_strength"].to(device)
        echo_targets = batch["echo_time_norm"].to(device)

        outputs = model(images)

        manufacturer_loss = manufacturer_loss_fn(
            outputs["manufacturer_logits"],
            manufacturer_targets,
        )
        field_loss = field_strength_loss_fn(
            outputs["field_strength_logits"],
            field_targets,
        )
        echo_loss = echo_loss_fn(
            outputs["echo_time_norm"],
            echo_targets,
        )

        loss = manufacturer_loss + field_loss + echo_loss_weight * echo_loss

        batch_size = images.size(0)
        n_samples += batch_size

        total_loss += loss.item() * batch_size
        total_manufacturer_loss += manufacturer_loss.item() * batch_size
        total_field_loss += field_loss.item() * batch_size
        total_echo_loss += echo_loss.item() * batch_size

        all_manufacturer_true.extend(manufacturer_targets.detach().cpu().numpy().tolist())
        all_manufacturer_pred.extend(outputs["manufacturer_logits"].argmax(dim=1).detach().cpu().numpy().tolist())

        all_field_true.extend(field_targets.detach().cpu().numpy().tolist())
        all_field_pred.extend(outputs["field_strength_logits"].argmax(dim=1).detach().cpu().numpy().tolist())

        all_echo_true.extend(echo_targets.detach().cpu().numpy().tolist())
        all_echo_pred.extend(outputs["echo_time_norm"].detach().cpu().numpy().tolist())

    metrics = {
        "loss": total_loss / n_samples,
        "manufacturer_loss": total_manufacturer_loss / n_samples,
        "field_strength_loss": total_field_loss / n_samples,
        "echo_loss": total_echo_loss / n_samples,
        "manufacturer_accuracy": accuracy_score(all_manufacturer_true, all_manufacturer_pred),
        "manufacturer_balanced_accuracy": balanced_accuracy_score(all_manufacturer_true, all_manufacturer_pred),
        "field_strength_accuracy": accuracy_score(all_field_true, all_field_pred),
        "field_strength_balanced_accuracy": balanced_accuracy_score(all_field_true, all_field_pred),
        "echo_time_norm_mae": mean_absolute_error(all_echo_true, all_echo_pred),
    }

    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def plot_loss_curve(history_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Acquisition Predictor Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train acquisition-parameter predictor.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["training"].get("seed", 42)))

    device = choose_device(config["training"].get("device", "cuda"))
    print(f"[INFO] Device: {device}")

    train_csv = config["data"]["train_csv"]
    val_csv = config["data"]["val_csv"]

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))

    train_loader = make_dataloader(
        csv_path=train_csv,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        augment=bool(config["training"].get("augment_train", True)),
    )
    val_loader = make_dataloader(
        csv_path=val_csv,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        augment=False,
    )

    model = AcquisitionPredictor(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_manufacturers=int(config["model"]["num_manufacturers"]),
        num_field_strengths=int(config["model"]["num_field_strengths"]),
        feature_dim=int(config["model"].get("feature_dim", 256)),
        dropout=float(config["model"].get("dropout", 0.20)),
    ).to(device)

    print(f"[INFO] Trainable parameters: {count_parameters(model):,}")

    use_class_weights = bool(config["training"].get("use_class_weights", True))

    if use_class_weights:
        manufacturer_weights = compute_class_weights(
            train_csv,
            "manufacturer_label",
            int(config["model"]["num_manufacturers"]),
        ).to(device)
        field_weights = compute_class_weights(
            train_csv,
            "field_strength_label",
            int(config["model"]["num_field_strengths"]),
        ).to(device)
        print(f"[INFO] Manufacturer class weights: {manufacturer_weights.detach().cpu().tolist()}")
        print(f"[INFO] Field-strength class weights: {field_weights.detach().cpu().tolist()}")
    else:
        manufacturer_weights = None
        field_weights = None

    manufacturer_loss_fn = nn.CrossEntropyLoss(weight=manufacturer_weights)
    field_strength_loss_fn = nn.CrossEntropyLoss(weight=field_weights)
    echo_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    checkpoint_dir = Path(config["outputs"]["checkpoint_dir"])
    report_dir = Path(config["outputs"]["report_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(config["training"]["epochs"])
    echo_loss_weight = float(config["training"].get("echo_loss_weight", 0.5))

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            manufacturer_loss_fn=manufacturer_loss_fn,
            field_strength_loss_fn=field_strength_loss_fn,
            echo_loss_fn=echo_loss_fn,
            echo_loss_weight=echo_loss_weight,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            manufacturer_loss_fn=manufacturer_loss_fn,
            field_strength_loss_fn=field_strength_loss_fn,
            echo_loss_fn=echo_loss_fn,
            echo_loss_weight=echo_loss_weight,
            device=device,
            desc="val",
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_manufacturer_accuracy": train_metrics["manufacturer_accuracy"],
            "val_manufacturer_accuracy": val_metrics["manufacturer_accuracy"],
            "train_manufacturer_balanced_accuracy": train_metrics["manufacturer_balanced_accuracy"],
            "val_manufacturer_balanced_accuracy": val_metrics["manufacturer_balanced_accuracy"],
            "train_field_strength_accuracy": train_metrics["field_strength_accuracy"],
            "val_field_strength_accuracy": val_metrics["field_strength_accuracy"],
            "train_field_strength_balanced_accuracy": train_metrics["field_strength_balanced_accuracy"],
            "val_field_strength_balanced_accuracy": val_metrics["field_strength_balanced_accuracy"],
            "train_echo_time_norm_mae": train_metrics["echo_time_norm_mae"],
            "val_echo_time_norm_mae": val_metrics["echo_time_norm_mae"],
        }
        history.append(row)

        print(
            "[INFO] "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_mfg_acc={row['val_manufacturer_accuracy']:.4f} | "
            f"val_field_acc={row['val_field_strength_accuracy']:.4f} | "
            f"val_echo_norm_mae={row['val_echo_time_norm_mae']:.4f}"
        )

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            best_val_loss,
            config,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                config,
            )
            print(f"[INFO] Saved new best checkpoint with val_loss={best_val_loss:.4f}")

    history_df = pd.DataFrame(history)
    history_csv = report_dir / "history.csv"
    history_df.to_csv(history_csv, index=False)

    plot_loss_curve(history_df, report_dir / "loss_curve.png")

    summary = {
        "best_val_loss": best_val_loss,
        "final_epoch": epochs,
        "checkpoint_dir": str(checkpoint_dir),
        "report_dir": str(report_dir),
        "num_parameters": count_parameters(model),
        "config": config,
    }

    with open(report_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[INFO] Training complete.")
    print(f"[INFO] Best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"[INFO] History CSV: {history_csv}")
    print(f"[INFO] Loss curve: {report_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
