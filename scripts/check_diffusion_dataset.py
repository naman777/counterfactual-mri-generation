import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
from torch.utils.data import DataLoader

from src.data.diffusion_dataset import MRIDiffusionDataset


def main():
    config_path = "configs/diffusion.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    train_dataset = MRIDiffusionDataset(
        csv_path=data_cfg["train_csv"],
        image_column=data_cfg["image_column"],
        manufacturer_column=data_cfg["manufacturer_column"],
        field_strength_column=data_cfg["field_strength_column"],
        echo_time_column=data_cfg["echo_time_column"],
        image_size=data_cfg["image_size"],
    )

    print("Dataset loaded successfully.")
    print(f"Number of training samples: {len(train_dataset)}")

    sample = train_dataset[0]

    print("\nSingle sample:")
    print("image shape:", sample["image"].shape)
    print("image min:", float(sample["image"].min()))
    print("image max:", float(sample["image"].max()))
    print("manufacturer_label:", sample["manufacturer_label"])
    print("field_strength_label:", sample["field_strength_label"])
    print("echo_time_norm:", sample["echo_time_norm"])
    print("image_path:", sample["image_path"])

    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\nBatch:")
    print("image batch shape:", batch["image"].shape)
    print("image batch min:", float(batch["image"].min()))
    print("image batch max:", float(batch["image"].max()))
    print("manufacturer labels:", batch["manufacturer_label"])
    print("field strength labels:", batch["field_strength_label"])
    print("echo_time_norm:", batch["echo_time_norm"])

    assert batch["image"].shape == torch.Size([4, 1, 128, 128])
    assert batch["image"].min() >= -1.0
    assert batch["image"].max() <= 1.0

    print("\nDiffusion dataset check passed.")


if __name__ == "__main__":
    main()