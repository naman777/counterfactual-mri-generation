import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
from torch.utils.data import DataLoader

from src.data.diffusion_dataset import MRIDiffusionDataset
from src.models.conditioning import AcquisitionConditionEmbedding


def main():
    config_path = "configs/diffusion.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    cond_cfg = cfg["condition"]

    dataset = MRIDiffusionDataset(
        csv_path=data_cfg["train_csv"],
        image_column=data_cfg["image_column"],
        manufacturer_column=data_cfg["manufacturer_column"],
        field_strength_column=data_cfg["field_strength_column"],
        echo_time_column=data_cfg["echo_time_column"],
        image_size=data_cfg["image_size"],
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    model = AcquisitionConditionEmbedding(
        num_manufacturers=cond_cfg["num_manufacturers"],
        num_field_strengths=cond_cfg["num_field_strengths"],
        manufacturer_embedding_dim=cond_cfg["manufacturer_embedding_dim"],
        field_strength_embedding_dim=cond_cfg["field_strength_embedding_dim"],
        echo_embedding_dim=cond_cfg["echo_embedding_dim"],
        condition_dim=cond_cfg["condition_dim"],
        dropout_prob=cond_cfg["condition_dropout_prob"],
    )

    model.train()

    cond_emb = model(
        manufacturer_label=batch["manufacturer_label"],
        field_strength_label=batch["field_strength_label"],
        echo_time_norm=batch["echo_time_norm"],
    )

    uncond_emb = model(
        manufacturer_label=batch["manufacturer_label"],
        field_strength_label=batch["field_strength_label"],
        echo_time_norm=batch["echo_time_norm"],
        force_unconditional=True,
    )

    print("Conditioning module loaded successfully.")
    print("Batch size:", batch["image"].shape[0])
    print("manufacturer labels:", batch["manufacturer_label"])
    print("field strength labels:", batch["field_strength_label"])
    print("echo_time_norm:", batch["echo_time_norm"])

    print("\nConditional embedding:")
    print("shape:", cond_emb.shape)
    print("mean:", float(cond_emb.mean()))
    print("std:", float(cond_emb.std()))

    print("\nUnconditional embedding:")
    print("shape:", uncond_emb.shape)
    print("mean:", float(uncond_emb.mean()))
    print("std:", float(uncond_emb.std()))

    assert cond_emb.shape == torch.Size([8, cond_cfg["condition_dim"]])
    assert uncond_emb.shape == torch.Size([8, cond_cfg["condition_dim"]])

    model.eval()

    with torch.no_grad():
        cond_eval_1 = model(
            manufacturer_label=batch["manufacturer_label"],
            field_strength_label=batch["field_strength_label"],
            echo_time_norm=batch["echo_time_norm"],
        )

        cond_eval_2 = model(
            manufacturer_label=batch["manufacturer_label"],
            field_strength_label=batch["field_strength_label"],
            echo_time_norm=batch["echo_time_norm"],
        )

    assert torch.allclose(cond_eval_1, cond_eval_2), (
        "Eval embeddings should be deterministic."
    )

    print("\nEval determinism check passed.")
    print("Conditioning check passed.")


if __name__ == "__main__":
    main()