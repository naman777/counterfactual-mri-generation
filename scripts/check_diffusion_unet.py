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
from src.models.diffusion_unet import MRIConditionalUNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    config_path = "configs/diffusion.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    cond_cfg = cfg["condition"]
    model_cfg = cfg["model"]
    diffusion_cfg = cfg["diffusion"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"].to(device)
    manufacturer_label = batch["manufacturer_label"].to(device)
    field_strength_label = batch["field_strength_label"].to(device)
    echo_time_norm = batch["echo_time_norm"].to(device)

    conditioning = AcquisitionConditionEmbedding(
        num_manufacturers=cond_cfg["num_manufacturers"],
        num_field_strengths=cond_cfg["num_field_strengths"],
        manufacturer_embedding_dim=cond_cfg["manufacturer_embedding_dim"],
        field_strength_embedding_dim=cond_cfg["field_strength_embedding_dim"],
        echo_embedding_dim=cond_cfg["echo_embedding_dim"],
        condition_dim=cond_cfg["condition_dim"],
        dropout_prob=cond_cfg["condition_dropout_prob"],
    ).to(device)

    unet = MRIConditionalUNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
        channel_multipliers=model_cfg["channel_multipliers"],
        time_embedding_dim=model_cfg["time_embedding_dim"],
        condition_dim=cond_cfg["condition_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    conditioning.train()
    unet.train()

    batch_size = images.shape[0]

    timesteps = torch.randint(
        low=0,
        high=diffusion_cfg["num_train_timesteps"],
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    condition_embedding = conditioning(
        manufacturer_label=manufacturer_label,
        field_strength_label=field_strength_label,
        echo_time_norm=echo_time_norm,
    )

    noisy_images = torch.randn_like(images)

    predicted_noise = unet(
        noisy_image=noisy_images,
        timesteps=timesteps,
        condition_embedding=condition_embedding,
    )

    loss = torch.nn.functional.mse_loss(predicted_noise, torch.randn_like(predicted_noise))
    loss.backward()

    print("U-Net loaded successfully.")
    print("Number of U-Net parameters:", count_parameters(unet))
    print("Number of conditioning parameters:", count_parameters(conditioning))

    print("\nInput batch:")
    print("images shape:", images.shape)
    print("noisy_images shape:", noisy_images.shape)
    print("timesteps:", timesteps)
    print("condition_embedding shape:", condition_embedding.shape)

    print("\nOutput:")
    print("predicted_noise shape:", predicted_noise.shape)
    print("predicted_noise min:", float(predicted_noise.min()))
    print("predicted_noise max:", float(predicted_noise.max()))
    print("dummy loss:", float(loss.detach().cpu()))

    assert predicted_noise.shape == images.shape
    assert torch.isfinite(predicted_noise).all()
    assert torch.isfinite(loss).all()

    print("\nBackward pass successful.")
    print("Diffusion U-Net check passed.")


if __name__ == "__main__":
    main()