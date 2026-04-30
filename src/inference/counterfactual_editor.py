import json
import inspect
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from src.models.conditioning import AcquisitionConditionEmbedding
from src.models.diffusion_unet import MRIConditionalUNet
from src.models.ddpm_scheduler import DDPMScheduler


class CounterfactualEditor:
    """
    Phase 9 counterfactual editor.

    Core idea:
      real image x0
      -> partially noise to x_t
      -> denoise from t_edit to 0 using target acquisition condition
      -> counterfactual image
    """

    def __init__(
        self,
        config_path: str = "configs/diffusion.yaml",
        checkpoint_path: str = "checkpoints/diffusion/ax_t1_128/latest.pt",
        device: Optional[str] = None,
    ):
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = self._resolve(config_path)
        self.checkpoint_path = self._resolve(checkpoint_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        if device is None:
            requested_device = self.cfg.get("training", {}).get("device", "cuda")
            if requested_device == "cuda" and not torch.cuda.is_available():
                requested_device = "cpu"
            device = requested_device

        self.device = torch.device(device)

        self.conditioner = self._build_conditioner().to(self.device)
        self.unet = self._build_unet().to(self.device)
        self.scheduler = self._build_scheduler().to(self.device)

        self._load_checkpoint()

        self.conditioner.eval()
        self.unet.eval()
        self.scheduler.eval()

    def _resolve(self, path_like: str) -> Path:
        p = Path(path_like)
        if p.is_absolute():
            return p
        return self.root / p

    def _build_conditioner(self) -> AcquisitionConditionEmbedding:
        c = self.cfg["condition"]
        return AcquisitionConditionEmbedding(
            num_manufacturers=int(c.get("num_manufacturers", 2)),
            num_field_strengths=int(c.get("num_field_strengths", 2)),
            manufacturer_embedding_dim=int(c.get("manufacturer_embedding_dim", 32)),
            field_strength_embedding_dim=int(c.get("field_strength_embedding_dim", 32)),
            echo_embedding_dim=int(c.get("echo_embedding_dim", 32)),
            condition_dim=int(c.get("condition_dim", 128)),
            dropout_prob=float(c.get("condition_dropout_prob", 0.1)),
        )

    def _build_unet(self) -> MRIConditionalUNet:
        m = self.cfg["model"]
        c = self.cfg["condition"]
        return MRIConditionalUNet(
            in_channels=int(m.get("in_channels", 1)),
            out_channels=int(m.get("out_channels", 1)),
            base_channels=int(m.get("base_channels", 64)),
            channel_multipliers=m.get("channel_multipliers", [1, 2, 4]),
            time_embedding_dim=int(m.get("time_embedding_dim", 128)),
            condition_dim=int(c.get("condition_dim", 128)),
            dropout=float(m.get("dropout", 0.1)),
        )

    def _build_scheduler(self) -> DDPMScheduler:
        d = self.cfg["diffusion"]
        return DDPMScheduler(
            num_train_timesteps=int(d.get("num_train_timesteps", 1000)),
            beta_start=float(d.get("beta_start", 0.0001)),
            beta_end=float(d.get("beta_end", 0.02)),
            beta_schedule=str(d.get("beta_schedule", "linear")),
            prediction_type=str(d.get("prediction_type", "epsilon")),
        )

    def _load_checkpoint(self):
        print(f"Loading diffusion checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        if "unet_state_dict" not in ckpt:
            raise KeyError("Checkpoint missing `unet_state_dict`.")
        if "conditioning_state_dict" not in ckpt:
            raise KeyError("Checkpoint missing `conditioning_state_dict`.")

        self.unet.load_state_dict(ckpt["unet_state_dict"])
        self.conditioner.load_state_dict(ckpt["conditioning_state_dict"])

        self.checkpoint_metadata = {
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "best_val_loss": ckpt.get("best_val_loss"),
        }

        print("Loaded checkpoint metadata:")
        print(json.dumps(self.checkpoint_metadata, indent=2))

    @staticmethod
    def load_image_npy(image_path: str) -> np.ndarray:
        arr = np.load(image_path).astype(np.float32)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image array, got shape {arr.shape}")

        arr = np.clip(arr, 0.0, 1.0)
        return arr

    @staticmethod
    def image_to_tensor_01_to_m11(image_01: np.ndarray, device: torch.device) -> torch.Tensor:
        x = torch.from_numpy(image_01).float()
        x = x.unsqueeze(0).unsqueeze(0)
        x = x * 2.0 - 1.0
        return x.to(device)

    @staticmethod
    def tensor_m11_to_image_01(x: torch.Tensor) -> np.ndarray:
        x = x.detach().float().cpu()
        x = x.squeeze(0).squeeze(0)
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        return x.numpy().astype(np.float32)

    @staticmethod
    def save_png(image_01: np.ndarray, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        img = np.clip(image_01, 0.0, 1.0)
        img_uint8 = (img * 255.0).round().astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(path)

    def make_condition_tensors(
        self,
        manufacturer_label: int,
        field_strength_label: int,
        echo_time_norm: float,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        manufacturer = torch.full(
            (batch_size,),
            int(manufacturer_label),
            dtype=torch.long,
            device=self.device,
        )
        field_strength = torch.full(
            (batch_size,),
            int(field_strength_label),
            dtype=torch.long,
            device=self.device,
        )
        echo_time = torch.full(
            (batch_size,),
            float(echo_time_norm),
            dtype=torch.float32,
            device=self.device,
        )
        return manufacturer, field_strength, echo_time

    def _condition_forward(
        self,
        manufacturer: torch.Tensor,
        field_strength: torch.Tensor,
        echo_time: torch.Tensor,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        """
        Adaptive call wrapper because conditioning forward signatures can vary.
        """

        sig = inspect.signature(self.conditioner.forward)
        params = sig.parameters

        kwargs = {}

        if "manufacturer_label" in params:
            kwargs["manufacturer_label"] = manufacturer
        elif "manufacturer_labels" in params:
            kwargs["manufacturer_labels"] = manufacturer
        elif "manufacturer" in params:
            kwargs["manufacturer"] = manufacturer

        if "field_strength_label" in params:
            kwargs["field_strength_label"] = field_strength
        elif "field_strength_labels" in params:
            kwargs["field_strength_labels"] = field_strength
        elif "field_strength" in params:
            kwargs["field_strength"] = field_strength

        if "echo_time_norm" in params:
            kwargs["echo_time_norm"] = echo_time
        elif "echo_time" in params:
            kwargs["echo_time"] = echo_time
        elif "echo" in params:
            kwargs["echo"] = echo_time

        if "force_unconditional" in params:
            kwargs["force_unconditional"] = force_unconditional
        elif "unconditional" in params:
            kwargs["unconditional"] = force_unconditional
        elif "drop_condition" in params:
            kwargs["drop_condition"] = force_unconditional

        try:
            return self.conditioner(**kwargs)
        except TypeError:
            pass

        # Fallback to common positional style.
        try:
            return self.conditioner(
                manufacturer,
                field_strength,
                echo_time,
                force_unconditional=force_unconditional,
            )
        except TypeError:
            pass

        # Final fallback: if unconditional is requested and module exposes embedding directly.
        if force_unconditional and hasattr(self.conditioner, "unconditional_embedding"):
            emb = self.conditioner.unconditional_embedding
            return emb.unsqueeze(0).repeat(manufacturer.shape[0], 1)

        return self.conditioner(manufacturer, field_strength, echo_time)

    def _unet_forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adaptive call wrapper for U-Net forward.
        """

        try:
            return self.unet(x, timestep, condition_embedding)
        except TypeError:
            pass

        try:
            return self.unet(
                x=x,
                timesteps=timestep,
                condition_embedding=condition_embedding,
            )
        except TypeError:
            pass

        try:
            return self.unet(
                sample=x,
                timestep=timestep,
                condition_embedding=condition_embedding,
            )
        except TypeError:
            pass

        return self.unet(
            noisy_image=x,
            timestep=timestep,
            condition_embedding=condition_embedding,
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Uses your existing scheduler q_sample.
        """

        try:
            return self.scheduler.q_sample(x0, t, noise)
        except TypeError:
            pass

        try:
            return self.scheduler.q_sample(x_start=x0, t=t, noise=noise)
        except TypeError:
            pass

        try:
            return self.scheduler.q_sample(x_0=x0, t=t, noise=noise)
        except TypeError:
            pass

        return self.scheduler.q_sample(original_samples=x0, timesteps=t, noise=noise)

    def _get_scheduler_buffer(self, names):
        for name in names:
            if hasattr(self.scheduler, name):
                return getattr(self.scheduler, name)
        raise AttributeError(f"Could not find scheduler buffer from candidates: {names}")

    @torch.no_grad()
    def denoise_from_timestep(
        self,
        x_t: torch.Tensor,
        target_manufacturer_label: int,
        target_field_strength_label: int,
        target_echo_time_norm: float,
        edit_timestep: int,
        guidance_scale: float = 3.0,
    ) -> torch.Tensor:
        """
        DDPM reverse process from edit_timestep down to 0.

        This uses the checkpoint's epsilon-prediction U-Net.
        """

        betas = self._get_scheduler_buffer(["betas"]).to(self.device)
        alphas = self._get_scheduler_buffer(["alphas"]).to(self.device)
        alpha_bars = self._get_scheduler_buffer(
            ["alpha_bars", "alphas_cumprod", "alpha_cumprod"]
        ).to(self.device)

        x = x_t

        manufacturer, field_strength, echo_time = self.make_condition_tensors(
            target_manufacturer_label,
            target_field_strength_label,
            target_echo_time_norm,
            batch_size=x.shape[0],
        )

        cond_emb = self._condition_forward(
            manufacturer,
            field_strength,
            echo_time,
            force_unconditional=False,
        )

        uncond_emb = self._condition_forward(
            manufacturer,
            field_strength,
            echo_time,
            force_unconditional=True,
        )

        for i in reversed(range(0, edit_timestep + 1)):
            t = torch.full((x.shape[0],), i, dtype=torch.long, device=self.device)

            eps_cond = self._unet_forward(x, t, cond_emb)
            eps_uncond = self._unet_forward(x, t, uncond_emb)

            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            beta_t = betas[i]
            alpha_t = alphas[i]
            alpha_bar_t = alpha_bars[i]

            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)

            mean = sqrt_recip_alpha_t * (
                x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps
            )

            if i > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean

        return torch.clamp(x, -1.0, 1.0)

    @torch.no_grad()
    def generate_counterfactual(
        self,
        image_01: np.ndarray,
        target_manufacturer_label: int,
        target_field_strength_label: int,
        target_echo_time_norm: float,
        edit_timestep: int = 100,
        guidance_scale: float = 1.0,
        seed: int = 42,
    ) -> Dict:
        if edit_timestep < 1:
            raise ValueError("edit_timestep must be >= 1")

        max_t = int(self.cfg["diffusion"]["num_train_timesteps"]) - 1
        if edit_timestep > max_t:
            raise ValueError(f"edit_timestep must be <= {max_t}")

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        x0 = self.image_to_tensor_01_to_m11(image_01, self.device)

        t = torch.full((x0.shape[0],), edit_timestep, dtype=torch.long, device=self.device)
        noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)

        generated_tensor = self.denoise_from_timestep(
            x_t=x_t,
            target_manufacturer_label=target_manufacturer_label,
            target_field_strength_label=target_field_strength_label,
            target_echo_time_norm=target_echo_time_norm,
            edit_timestep=edit_timestep,
            guidance_scale=guidance_scale,
        )

        generated_01 = self.tensor_m11_to_image_01(generated_tensor)

        return {
            "original_01": image_01.astype(np.float32),
            "generated_01": generated_01,
            "metadata": {
                "edit_timestep": int(edit_timestep),
                "guidance_scale": float(guidance_scale),
                "target_manufacturer_label": int(target_manufacturer_label),
                "target_field_strength_label": int(target_field_strength_label),
                "target_echo_time_norm": float(target_echo_time_norm),
                "seed": int(seed),
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_metadata": self.checkpoint_metadata,
            },
        }