import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.diffusion_dataset import MRIDiffusionDataset
from src.models.conditioning import AcquisitionConditionEmbedding
from src.models.ddpm_scheduler import DDPMScheduler
from src.models.diffusion_unet import MRIConditionalUNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_dataloader(csv_path, cfg, shuffle):
    data_cfg = cfg["data"]

    dataset = MRIDiffusionDataset(
        csv_path=csv_path,
        image_column=data_cfg["image_column"],
        manufacturer_column=data_cfg["manufacturer_column"],
        field_strength_column=data_cfg["field_strength_column"],
        echo_time_column=data_cfg["echo_time_column"],
        image_size=data_cfg["image_size"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=data_cfg["num_workers"],
        pin_memory=bool(data_cfg["pin_memory"] and torch.cuda.is_available()),
        drop_last=shuffle,
    )

    return dataset, loader


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

    scheduler = DDPMScheduler(
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        beta_schedule=diff_cfg["beta_schedule"],
        prediction_type=diff_cfg["prediction_type"],
    ).to(device)

    return conditioning, unet, scheduler


def save_checkpoint(
    path,
    epoch,
    global_step,
    conditioning,
    unet,
    optimizer,
    scaler,
    best_val_loss,
    cfg,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "conditioning_state_dict": conditioning.state_dict(),
        "unet_state_dict": unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": cfg,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def train_one_epoch(
    epoch,
    conditioning,
    unet,
    scheduler,
    train_loader,
    optimizer,
    scaler,
    cfg,
    device,
    global_step,
    max_steps=None,
):
    conditioning.train()
    unet.train()

    grad_clip_norm = cfg["training"]["grad_clip_norm"]
    use_amp = bool(cfg["training"]["mixed_precision"] and device.type == "cuda")

    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=True)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        manufacturer_label = batch["manufacturer_label"].to(device, non_blocking=True)
        field_strength_label = batch["field_strength_label"].to(device, non_blocking=True)
        echo_time_norm = batch["echo_time_norm"].to(device, non_blocking=True)

        batch_size = images.shape[0]

        timesteps = torch.randint(
            low=0,
            high=scheduler.num_train_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(images)
        noisy_images = scheduler.q_sample(
            x_start=images,
            timesteps=timesteps,
            noise=noise,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            condition_embedding = conditioning(
                manufacturer_label=manufacturer_label,
                field_strength_label=field_strength_label,
                echo_time_norm=echo_time_norm,
            )

            predicted_noise = unet(
                noisy_image=noisy_images,
                timesteps=timesteps,
                condition_embedding=condition_embedding,
            )

            loss = F.mse_loss(predicted_noise, noise)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(conditioning.parameters()) + list(unet.parameters()),
                    grad_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(conditioning.parameters()) + list(unet.parameters()),
                    grad_clip_norm,
                )

            optimizer.step()

        loss_value = float(loss.detach().cpu())
        running_loss += loss_value
        num_batches += 1
        global_step += 1

        pbar.set_postfix({"loss": f"{loss_value:.5f}", "step": global_step})

        if max_steps is not None and global_step >= max_steps:
            break

    mean_loss = running_loss / max(num_batches, 1)
    return mean_loss, global_step


@torch.no_grad()
def validate(
    epoch,
    conditioning,
    unet,
    scheduler,
    val_loader,
    cfg,
    device,
    max_val_batches=25,
):
    conditioning.eval()
    unet.eval()

    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False)

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        manufacturer_label = batch["manufacturer_label"].to(device, non_blocking=True)
        field_strength_label = batch["field_strength_label"].to(device, non_blocking=True)
        echo_time_norm = batch["echo_time_norm"].to(device, non_blocking=True)

        batch_size = images.shape[0]

        timesteps = torch.randint(
            low=0,
            high=scheduler.num_train_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(images)
        noisy_images = scheduler.q_sample(
            x_start=images,
            timesteps=timesteps,
            noise=noise,
        )

        condition_embedding = conditioning(
            manufacturer_label=manufacturer_label,
            field_strength_label=field_strength_label,
            echo_time_norm=echo_time_norm,
        )

        predicted_noise = unet(
            noisy_image=noisy_images,
            timesteps=timesteps,
            condition_embedding=condition_embedding,
        )

        loss = F.mse_loss(predicted_noise, noise)

        loss_value = float(loss.detach().cpu())
        running_loss += loss_value
        num_batches += 1

        pbar.set_postfix({"val_loss": f"{loss_value:.5f}"})

        if batch_idx + 1 >= max_val_batches:
            break

    mean_loss = running_loss / max(num_batches, 1)
    return mean_loss


def append_history(history_csv, row):
    history_csv = Path(history_csv)
    history_csv.parent.mkdir(parents=True, exist_ok=True)

    file_exists = history_csv.exists()

    with open(history_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "global_step", "train_loss", "val_loss"],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_summary(summary_path, summary):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diffusion.yaml",
        help="Path to diffusion config YAML.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run short debug training.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum global training steps for debug mode.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["debug"]["enabled"] = True

    max_steps = args.max_steps
    if max_steps is None and cfg["debug"]["enabled"]:
        max_steps = int(cfg["debug"]["max_steps"])

    set_seed(int(cfg["project"]["seed"]))

    requested_device = cfg["training"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    print("Device:", device)
    print("Debug mode:", cfg["debug"]["enabled"])
    print("Max steps:", max_steps)

    Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["sample_dir"]).mkdir(parents=True, exist_ok=True)

    train_dataset, train_loader = make_dataloader(
        cfg["data"]["train_csv"],
        cfg,
        shuffle=True,
    )

    val_dataset, val_loader = make_dataloader(
        cfg["data"]["val_csv"],
        cfg,
        shuffle=False,
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Batch size:", cfg["training"]["batch_size"])

    conditioning, unet, scheduler = build_models(cfg, device)

    print("U-Net parameters:", count_parameters(unet))
    print("Conditioning parameters:", count_parameters(conditioning))

    optimizer = torch.optim.AdamW(
        list(conditioning.parameters()) + list(unet.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    use_amp = bool(cfg["training"]["mixed_precision"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    global_step = 0

    history_csv = Path(cfg["paths"]["log_dir"]) / "history.csv"
    summary_path = Path(cfg["paths"]["log_dir"]) / "training_summary.json"

    total_epochs = int(cfg["training"]["epochs"])

    if cfg["debug"]["enabled"]:
        total_epochs = 999999

    for epoch in range(1, total_epochs + 1):
        train_loss, global_step = train_one_epoch(
            epoch=epoch,
            conditioning=conditioning,
            unet=unet,
            scheduler=scheduler,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            global_step=global_step,
            max_steps=max_steps,
        )

        val_loss = validate(
            epoch=epoch,
            conditioning=conditioning,
            unet=unet,
            scheduler=scheduler,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
        )

        append_history(
            history_csv,
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"global_step={global_step}"
        )

        save_checkpoint(
            path=cfg["paths"]["latest_checkpoint"],
            epoch=epoch,
            global_step=global_step,
            conditioning=conditioning,
            unet=unet,
            optimizer=optimizer,
            scaler=scaler,
            best_val_loss=best_val_loss,
            cfg=cfg,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                path=cfg["paths"]["best_checkpoint"],
                epoch=epoch,
                global_step=global_step,
                conditioning=conditioning,
                unet=unet,
                optimizer=optimizer,
                scaler=scaler,
                best_val_loss=best_val_loss,
                cfg=cfg,
            )

            print(f"Saved new best checkpoint with val_loss={best_val_loss:.6f}")

        summary = {
            "experiment_name": cfg["project"]["experiment_name"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "batch_size": cfg["training"]["batch_size"],
            "global_step": global_step,
            "latest_epoch": epoch,
            "latest_train_loss": train_loss,
            "latest_val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "latest_checkpoint": cfg["paths"]["latest_checkpoint"],
            "best_checkpoint": cfg["paths"]["best_checkpoint"],
            "debug_mode": cfg["debug"]["enabled"],
        }

        save_summary(summary_path, summary)

        if max_steps is not None and global_step >= max_steps:
            print("Reached max_steps. Stopping debug training.")
            break

    print("Training finished.")
    print("Best val loss:", best_val_loss)
    print("History CSV:", history_csv)
    print("Summary JSON:", summary_path)


if __name__ == "__main__":
    main()