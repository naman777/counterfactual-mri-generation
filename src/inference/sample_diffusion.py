import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.conditioning import AcquisitionConditionEmbedding
from src.models.ddpm_scheduler import DDPMScheduler
from src.models.diffusion_unet import MRIConditionalUNet




def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def invert_label_map(label_map):
    return {int(v): k for k, v in label_map.items()}


def build_models(cfg, device):
    cond_cfg = cfg["condition"]
    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]

    conditioning = AcquisitionConditionEmbedding(
        num_manufacturers=cond_cfg["num_manufacturers"],
        num_field_strengths=cond_cfg["num_field_strengths"],
        manufacturer_embedding_dim=cond_cfg["manufacturer_embedding_dim"],
        field_strength_embedding_dim=cond_cfg["field_strength_embedding_dim"],
        echo_embedding_dim=cond_cfg["echo_embedding_dim"],
        condition_dim=cond_cfg["condition_dim"],
        dropout_prob=0.0,
    ).to(device)

    unet = MRIConditionalUNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
        channel_multipliers=model_cfg["channel_multipliers"],
        time_embedding_dim=model_cfg["time_embedding_dim"],
        condition_dim=cond_cfg["condition_dim"],
        dropout=0.0,
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        beta_schedule=diff_cfg["beta_schedule"],
        prediction_type=diff_cfg["prediction_type"],
    ).to(device)

    return conditioning, unet, scheduler


def load_checkpoint(checkpoint_path, conditioning, unet, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    conditioning.load_state_dict(checkpoint["conditioning_state_dict"])
    unet.load_state_dict(checkpoint["unet_state_dict"])

    epoch = checkpoint.get("epoch", None)
    global_step = checkpoint.get("global_step", None)
    best_val_loss = checkpoint.get("best_val_loss", None)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {epoch}")
    print(f"Checkpoint global_step: {global_step}")
    print(f"Checkpoint best_val_loss: {best_val_loss}")


@torch.no_grad()
def ddim_sample(
    conditioning,
    unet,
    scheduler,
    manufacturer_label,
    field_strength_label,
    echo_time_norm,
    image_size,
    image_channels,
    num_inference_steps,
    guidance_scale,
    device,
):
    """
    Deterministic DDIM-style sampling using a DDPM-trained epsilon model.

    This is suitable for our first sampling test because it allows fewer
    inference steps such as 50 instead of all 1000 training steps.
    """
    conditioning.eval()
    unet.eval()

    batch_size = manufacturer_label.shape[0]

    x = torch.randn(
        batch_size,
        image_channels,
        image_size,
        image_size,
        device=device,
    )

    train_timesteps = scheduler.num_train_timesteps

    timesteps = torch.linspace(
        train_timesteps - 1,
        0,
        num_inference_steps,
        device=device,
    ).long()

    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        t_batch = torch.full(
            (batch_size,),
            int(t.item()),
            device=device,
            dtype=torch.long,
        )

        cond_emb = conditioning(
            manufacturer_label=manufacturer_label,
            field_strength_label=field_strength_label,
            echo_time_norm=echo_time_norm,
            force_unconditional=False,
        )

        uncond_emb = conditioning(
            manufacturer_label=manufacturer_label,
            field_strength_label=field_strength_label,
            echo_time_norm=echo_time_norm,
            force_unconditional=True,
        )

        eps_cond = unet(
            noisy_image=x,
            timesteps=t_batch,
            condition_embedding=cond_emb,
        )

        eps_uncond = unet(
            noisy_image=x,
            timesteps=t_batch,
            condition_embedding=uncond_emb,
        )

        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        alpha_bar_t = scheduler.alphas_cumprod[t].view(1, 1, 1, 1)

        if i == len(timesteps) - 1:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        else:
            t_prev = timesteps[i + 1]
            alpha_bar_prev = scheduler.alphas_cumprod[t_prev].view(1, 1, 1, 1)

        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps

    x = torch.clamp(x, -1.0, 1.0)
    return x


def tensor_to_numpy_image(x):
    """
    Convert tensor from [-1, 1] to [0, 1].
    Input: [1, H, W]
    Output: [H, W]
    """
    x = x.detach().cpu()
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    return x.squeeze(0).numpy()


def save_grid(images, titles, output_path, ncols=4):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_images = len(images)
    nrows = int(np.ceil(n_images / ncols))

    plt.figure(figsize=(3 * ncols, 3 * nrows))

    for idx, image in enumerate(images):
        ax = plt.subplot(nrows, ncols, idx + 1)
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(titles[idx], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved sample grid: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--output_name", type=str, default="debug_samples.png")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device_name = cfg["training"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print("Device:", device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = cfg["paths"]["best_checkpoint"]

    num_steps = args.num_steps
    if num_steps is None:
        num_steps = int(cfg["sampling"]["num_inference_steps"])

    guidance_scale = args.guidance_scale
    if guidance_scale is None:
        guidance_scale = float(cfg["sampling"]["guidance_scale"])

    manufacturer_map = load_json(cfg["paths"]["manufacturer_label_map"])
    field_map = load_json(cfg["paths"]["field_strength_label_map"])

    id_to_manufacturer = invert_label_map(manufacturer_map)
    id_to_field = invert_label_map(field_map)

    print("Manufacturer map:", manufacturer_map)
    print("Field strength map:", field_map)
    print("Num samples:", args.num_samples)
    print("Sampling steps:", num_steps)
    print("Guidance scale:", guidance_scale)

    conditioning, unet, scheduler = build_models(cfg, device)
    load_checkpoint(checkpoint_path, conditioning, unet, device)

    # Build a repeated condition grid:
    # GE 1.5T low/mid/high TE
    # GE 3.0T low/mid/high TE
    # SIEMENS 1.5T low/mid/high TE
    # SIEMENS 3.0T low/mid/high TE
    condition_list = []

    echo_values = [0.15, 0.50, 0.85]

    for manufacturer_id in [0, 1]:
        for field_id in [0, 1]:
            for echo_norm in echo_values:
                condition_list.append((manufacturer_id, field_id, echo_norm))

    while len(condition_list) < args.num_samples:
        condition_list.extend(condition_list)

    condition_list = condition_list[: args.num_samples]

    manufacturer_label = torch.tensor(
        [c[0] for c in condition_list],
        dtype=torch.long,
        device=device,
    )

    field_strength_label = torch.tensor(
        [c[1] for c in condition_list],
        dtype=torch.long,
        device=device,
    )

    echo_time_norm = torch.tensor(
        [c[2] for c in condition_list],
        dtype=torch.float32,
        device=device,
    )

    samples = ddim_sample(
        conditioning=conditioning,
        unet=unet,
        scheduler=scheduler,
        manufacturer_label=manufacturer_label,
        field_strength_label=field_strength_label,
        echo_time_norm=echo_time_norm,
        image_size=int(cfg["data"]["image_size"]),
        image_channels=int(cfg["data"]["image_channels"]),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        device=device,
    )

    images = []
    titles = []

    for i in range(samples.shape[0]):
        image = tensor_to_numpy_image(samples[i])
        images.append(image)

        m_id = int(manufacturer_label[i].detach().cpu())
        f_id = int(field_strength_label[i].detach().cpu())
        te = float(echo_time_norm[i].detach().cpu())

        m_name = id_to_manufacturer.get(m_id, str(m_id))
        f_name = id_to_field.get(f_id, str(f_id))

        title = f"{m_name}\n{f_name}, TE_norm={te:.2f}"
        titles.append(title)

    sample_dir = Path(cfg["paths"]["sample_dir"])
    output_path = sample_dir / args.output_name

    save_grid(images, titles, output_path, ncols=4)

    # Save raw numpy samples too.
    raw_output_path = sample_dir / args.output_name.replace(".png", ".npy")
    np.save(raw_output_path, samples.detach().cpu().numpy())
    print(f"Saved raw sample tensor: {raw_output_path}")


if __name__ == "__main__":
    main()