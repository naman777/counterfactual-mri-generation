import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MRIDiffusionDataset(Dataset):
    """
    Dataset for conditional DDPM training on preprocessed MRI slices.

    Expected CSV columns:
    - image_path
    - manufacturer_label
    - field_strength_label
    - echo_time_norm

    Image format:
    - .npy files saved as 128x128 arrays normalized to [0, 1]

    Diffusion model input:
    - image converted from [0, 1] to [-1, 1]
    - returned as tensor shape [1, H, W]
    """

    def __init__(
        self,
        csv_path: str,
        image_column: str = "image_path",
        manufacturer_column: str = "manufacturer_label",
        field_strength_column: str = "field_strength_label",
        echo_time_column: str = "echo_time_norm",
        image_size: int = 128,
    ):
        self.csv_path = Path(csv_path)
        self.image_column = image_column
        self.manufacturer_column = manufacturer_column
        self.field_strength_column = field_strength_column
        self.echo_time_column = echo_time_column
        self.image_size = image_size

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_columns = [
            self.image_column,
            self.manufacturer_column,
            self.field_strength_column,
            self.echo_time_column,
        ]

        missing_columns = [c for c in required_columns if c not in self.df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: {missing_columns}\n"
                f"Available columns: {list(self.df.columns)}"
            )

        self.df = self.df.dropna(subset=required_columns).reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No valid rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, raw_path: str) -> Path:
        """
        Handles both Windows-style paths and Unix-style paths.
        """
        path = Path(raw_path)

        if path.exists():
            return path

        alt_path = Path(str(raw_path).replace("\\", "/"))
        if alt_path.exists():
            return alt_path

        raise FileNotFoundError(f"Image file not found: {raw_path}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image_path = self._resolve_image_path(row[self.image_column])

        image = np.load(image_path).astype(np.float32)

        if image.ndim != 2:
            raise ValueError(
                f"Expected image with shape [H, W], got {image.shape} at {image_path}"
            )

        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}x{self.image_size}, "
                f"got {image.shape} at {image_path}"
            )

        # Safety clamp in case any values are slightly outside [0, 1]
        image = np.clip(image, 0.0, 1.0)

        # DDPM training usually uses [-1, 1]
        image = (image * 2.0) - 1.0

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        manufacturer_label = torch.tensor(
            int(row[self.manufacturer_column]), dtype=torch.long
        )

        field_strength_label = torch.tensor(
            int(row[self.field_strength_column]), dtype=torch.long
        )

        echo_time_norm = torch.tensor(
            float(row[self.echo_time_column]), dtype=torch.float32
        )

        return {
            "image": image_tensor,
            "manufacturer_label": manufacturer_label,
            "field_strength_label": field_strength_label,
            "echo_time_norm": echo_time_norm,
            "image_path": str(image_path),
        }