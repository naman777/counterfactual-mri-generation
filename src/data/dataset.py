#!/usr/bin/env python3
"""
Dataset utilities for the acquisition-parameter predictor.

Expected CSV columns:
- image_path
- manufacturer_label
- field_strength_label
- echo_time_norm
- echo_time

The preprocessing script creates these columns in:
data/processed/ax_t1_128/processed_metadata.csv
data/processed/ax_t1_128/splits/train.csv
data/processed/ax_t1_128/splits/val.csv
data/processed/ax_t1_128/splits/test.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MRIAcquisitionDataset(Dataset):
    """
    PyTorch dataset for normalized 2D MRI slices saved as .npy or .png.

    Each item returns:
    {
        "image": Tensor[1, H, W],
        "manufacturer": LongTensor scalar,
        "field_strength": LongTensor scalar,
        "echo_time_norm": FloatTensor scalar,
        "echo_time": FloatTensor scalar,
        "image_path": str,
        "patient_id": str
    }
    """

    def __init__(
        self,
        csv_path: str | Path,
        augment: bool = False,
        image_key: str = "image_path",
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.augment = augment
        self.image_key = image_key

        required_cols = [
            image_key,
            "manufacturer_label",
            "field_strength_label",
            "echo_time_norm",
            "echo_time",
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: {missing}. "
                f"Available columns: {list(self.df.columns)}"
            )

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, path_value: str) -> Path:
        """
        Resolve image paths robustly across Windows/Linux path separators.
        """
        path_str = str(path_value).replace("\\", "/")
        path = Path(path_str)

        if path.exists():
            return path

        # If path is relative, try resolving relative to current working directory.
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path

        # Try resolving relative to the CSV parent directory.
        csv_relative = self.csv_path.parent / path
        if csv_relative.exists():
            return csv_relative

        raise FileNotFoundError(f"Image file not found: {path_value}")

    def _load_image(self, image_path: Path) -> np.ndarray:
        if image_path.suffix.lower() == ".npy":
            image = np.load(image_path).astype(np.float32)
        elif image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            try:
                import imageio.v2 as imageio
            except Exception as exc:
                raise ImportError(
                    "Loading PNG/JPG images requires imageio. Install with: pip install imageio"
                ) from exc

            image = imageio.imread(image_path).astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0
        else:
            raise ValueError(f"Unsupported image format: {image_path}")

        if image.ndim == 3:
            # If RGB somehow appears, convert to grayscale by taking the first channel.
            image = image[..., 0]

        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        return image

    def _apply_light_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Very light augmentation only.

        We avoid strong intensity augmentation because scanner/acquisition intensity
        differences are part of the signal we want the predictor to learn.
        """
        if not self.augment:
            return image

        # Small random horizontal flip. This should not change acquisition labels.
        if torch.rand(1).item() < 0.5:
            image = np.flip(image, axis=1).copy()

        return image

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image_path = self._resolve_image_path(str(row[self.image_key]))
        image = self._load_image(image_path)
        image = self._apply_light_augmentation(image)

        # Convert H x W to 1 x H x W.
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)

        item = {
            "image": image_tensor,
            "manufacturer": torch.tensor(int(row["manufacturer_label"]), dtype=torch.long),
            "field_strength": torch.tensor(int(row["field_strength_label"]), dtype=torch.long),
            "echo_time_norm": torch.tensor(float(row["echo_time_norm"]), dtype=torch.float32),
            "echo_time": torch.tensor(float(row["echo_time"]), dtype=torch.float32),
            "image_path": str(row[self.image_key]),
            "patient_id": str(row["patient_id"]) if "patient_id" in row else "",
        }

        return item


def make_dataloader(
    csv_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    augment: bool = False,
) -> torch.utils.data.DataLoader:
    dataset = MRIAcquisitionDataset(csv_path=csv_path, augment=augment)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
