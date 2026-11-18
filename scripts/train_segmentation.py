#!/usr/bin/env python3
"""
P0.1 — Step 3: Train segmentation U-Net in 4 settings.

Settings:
    1. GE-only:          Train on GE images → test on GE (ID) + Siemens (OOD)
    2. Siemens-only:     Train on Siemens images → test on Siemens (ID) + GE (OOD)
    3. GE + CF-augment:  Train on GE + CF→Siemens → test on both
    4. Siemens + CF-aug: Train on Siemens + CF→GE → test on both

Loss: Binary cross-entropy + Dice loss (combined)
Metric: Dice score (foreground vs background)
Early stopping on validation Dice.

Usage:
    python scripts/train_segmentation.py --setting ge_only
    python scripts/train_segmentation.py --setting siemens_only
    python scripts/train_segmentation.py --setting ge_augmented
    python scripts/train_segmentation.py --setting siemens_augmented
    python scripts/train_segmentation.py --all_settings
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.segmentation_unet import SegmentationUNet

GE_LABEL      = 0
SIEMENS_LABEL = 1


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    """
    Dataset of (image, mask) pairs for segmentation training.

    Records: list of dicts with 'image_path' and 'mask_path'.
    Skips slices where mask has no foreground (to focus on meaningful examples).
    """

    def __init__(self, records: list[dict], min_foreground: float = 0.001):
        self.records = []
        for r in records:
            mask = np.load(r["mask_path"])
            if mask.mean() >= min_foreground:
                self.records.append(r)
        print(f"  Dataset: {len(self.records)} slices WITH foreground "
              f"(from {len(records)} total)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img  = np.load(r["image_path"]).astype(np.float32)
        mask = np.load(r["mask_path"]).astype(np.float32)
        img  = np.clip(img,  0.0, 1.0)
        mask = np.clip(mask, 0.0, 1.0)
        # Add channel dim
        img  = torch.from_numpy(img).unsqueeze(0)   # [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        return img, mask


def build_records(
    csv_path: Path,
    masks_dir: Path,
    manufacturer_filter: Optional[int] = None,
    cf_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Build a list of {image_path, mask_path} records.

    Args:
        csv_path: train or val CSV
        masks_dir: directory containing mask .npy files
        manufacturer_filter: if set, only include slices from this manufacturer
        cf_dir: if set, also include counterfactual images from this directory
    """
    df = pd.read_csv(csv_path)

    if manufacturer_filter is not None:
        df = df[df["manufacturer_label"] == manufacturer_filter]

    records = []
    for _, row in df.iterrows():
        img_path  = PROJECT_ROOT / row["image_path"]
        mask_name = Path(row["image_path"]).name
        mask_path = masks_dir / mask_name

        if img_path.exists() and mask_path.exists():
            records.append({
                "image_path": str(img_path),
                "mask_path":  str(mask_path),
            })

    # Add counterfactual images if requested
    if cf_dir is not None and cf_dir.exists():
        cf_meta_path = cf_dir.parent / "augmented_metadata.csv"
        if cf_meta_path.exists():
            cf_df = pd.read_csv(cf_meta_path)
            # Only add CFs whose SOURCE matches the manufacturer filter
            if manufacturer_filter is not None:
                cf_df = cf_df[cf_df["source_manufacturer_label"] == manufacturer_filter]

            for _, row in cf_df.iterrows():
                cf_path   = PROJECT_ROOT / row["cf_path"]
                # Use the SAME mask as the original image (same anatomy)
                orig_name = Path(row["original_path"]).name
                mask_path = masks_dir / orig_name

                if cf_path.exists() and mask_path.exists():
                    records.append({
                        "image_path": str(cf_path),
                        "mask_path":  str(mask_path),
                    })

    return records


# ─── Loss ─────────────────────────────────────────────────────────────────────

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def combined_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(pred_logits, target)
    dl  = dice_loss(pred_logits, target)
    return 0.5 * bce + 0.5 * dl


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0
    return float(2.0 * intersection / union)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device) -> dict:
    model.train()
    losses, dices = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            dices.append(dice_score(logits, masks))
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices))}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    losses, dices = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        losses.append(combined_loss(logits, masks).item())
        dices.append(dice_score(logits, masks))
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices))}


# ─── Main ─────────────────────────────────────────────────────────────────────

SETTINGS = {
    "ge_only": {
        "train_manufacturer": GE_LABEL,
        "use_cf":             False,
        "cf_dir_key":         None,
        "description":        "Train on GE only",
    },
    "siemens_only": {
        "train_manufacturer": SIEMENS_LABEL,
        "use_cf":             False,
        "cf_dir_key":         None,
        "description":        "Train on Siemens only",
    },
    "ge_augmented": {
        "train_manufacturer": GE_LABEL,
        "use_cf":             True,
        "cf_dir_key":         "ge_to_siemens",
        "description":        "Train on GE + CF→Siemens",
    },
    "siemens_augmented": {
        "train_manufacturer": SIEMENS_LABEL,
        "use_cf":             True,
        "cf_dir_key":         "siemens_to_ge",
        "description":        "Train on Siemens + CF→GE",
    },
}


def run_setting(
    setting_name: str,
    train_csv: Path,
    val_csv: Path,
    masks_dir: Path,
    aug_dir: Path,
    ckpt_dir: Path,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    patience: int = 10,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    cfg = SETTINGS[setting_name]
    print(f"\n{'='*60}")
    print(f"Setting: {setting_name} — {cfg['description']}")
    print(f"{'='*60}")

    cf_dir = aug_dir / cfg["cf_dir_key"] if cfg["use_cf"] and cfg["cf_dir_key"] else None

    train_records = build_records(train_csv, masks_dir,
                                  manufacturer_filter=cfg["train_manufacturer"],
                                  cf_dir=cf_dir)
    val_records   = build_records(val_csv,   masks_dir,
                                  manufacturer_filter=cfg["train_manufacturer"])

    print(f"  Train records (before foreground filter): {len(train_records)}")
    print(f"  Val records (before foreground filter):   {len(val_records)}")

    train_ds = SegDataset(train_records)
    val_ds   = SegDataset(val_records)

    if len(train_ds) == 0:
        print(f"  ⚠ No training data with foreground for {setting_name}. Skipping.")
        return {"setting": setting_name, "error": "no_foreground_data"}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=device.type == "cuda")

    model = SegmentationUNet(base_channels=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_dice = -1.0
    best_epoch    = 0
    ckpt_path     = ckpt_dir / f"seg_{setting_name}_best.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics   = evaluate(model, val_loader, device)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss":   val_metrics["loss"],
            "val_dice":   val_metrics["dice"],
        })

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch    = epoch
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "val_dice":   best_val_dice,
                "setting":    setting_name,
            }, ckpt_path)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} | "
              f"Val: loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} | "
              f"{elapsed:.1f}s")

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
            break

    print(f"\n  Best val Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"  Checkpoint: {ckpt_path}")

    # Save history
    history_path = ckpt_dir / f"seg_{setting_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "setting":      setting_name,
        "description":  cfg["description"],
        "best_val_dice": best_val_dice,
        "best_epoch":   best_epoch,
        "checkpoint":   str(ckpt_path),
        "train_size":   len(train_ds),
        "val_size":     len(val_ds),
    }


def main():
    parser = argparse.ArgumentParser(description="P0.1 — Train segmentation models")
    parser.add_argument("--setting", type=str, choices=list(SETTINGS.keys()),
                        help="Which setting to train")
    parser.add_argument("--all_settings", action="store_true",
                        help="Train all 4 settings sequentially")
    parser.add_argument("--train_csv", type=str,
                        default="data/processed/ax_t1_128/splits/train.csv")
    parser.add_argument("--val_csv", type=str,
                        default="data/processed/ax_t1_128/splits/val.csv")
    parser.add_argument("--masks_dir", type=str,
                        default="data/processed/ax_t1_128/seg_masks/")
    parser.add_argument("--aug_dir", type=str,
                        default="data/augmented/")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoints/segmentation/")
    parser.add_argument("--output_dir", type=str,
                        default="reports/segmentation/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.all_settings and not args.setting:
        parser.error("Provide --setting <name> or --all_settings")

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    settings_to_run = list(SETTINGS.keys()) if args.all_settings else [args.setting]

    results = []
    for setting in settings_to_run:
        result = run_setting(
            setting_name=setting,
            train_csv=PROJECT_ROOT / args.train_csv,
            val_csv=PROJECT_ROOT / args.val_csv,
            masks_dir=PROJECT_ROOT / args.masks_dir,
            aug_dir=PROJECT_ROOT / args.aug_dir,
            ckpt_dir=PROJECT_ROOT / args.ckpt_dir,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
        )
        results.append(result)

    # Save summary
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining summary saved: {summary_path}")

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  {r['setting']}: ERROR — {r['error']}")
        else:
            print(f"  {r['setting']}: best_val_dice={r['best_val_dice']:.4f} "
                  f"(epoch {r['best_epoch']})")


if __name__ == "__main__":
    main()
