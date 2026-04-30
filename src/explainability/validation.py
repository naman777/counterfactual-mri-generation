import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.models.predictor import AcquisitionPredictor


class PredictorValidator:
    """
    Runs the trained acquisition-parameter predictor on original and generated MRI images.

    Input images must be 2D numpy arrays in [0, 1].
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/predictor/ax_t1_128/best.pt",
        echo_stats_path: str = "data/processed/ax_t1_128/label_maps/echo_time_stats.json",
        device: Optional[str] = None,
    ):
        self.root = Path(__file__).resolve().parents[2]
        self.checkpoint_path = self._resolve(checkpoint_path)
        self.echo_stats_path = self._resolve(echo_stats_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.echo_stats = self._load_echo_stats()
        self.model = self._load_model()
        self.model.eval()

    def _resolve(self, path_like: str) -> Path:
        p = Path(path_like)
        if p.is_absolute():
            return p
        return self.root / p

    def _load_echo_stats(self) -> Dict:
        if not self.echo_stats_path.exists():
            return {}

        with open(self.echo_stats_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self) -> AcquisitionPredictor:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Predictor checkpoint not found: {self.checkpoint_path}")

        print(f"Loading predictor checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if "config" not in checkpoint:
            raise KeyError("Predictor checkpoint missing `config`.")
        if "model_state_dict" not in checkpoint:
            raise KeyError("Predictor checkpoint missing `model_state_dict`.")

        config = checkpoint["config"]

        model = AcquisitionPredictor(
            in_channels=int(config["model"].get("in_channels", 1)),
            num_manufacturers=int(config["model"]["num_manufacturers"]),
            num_field_strengths=int(config["model"]["num_field_strengths"]),
            feature_dim=int(config["model"].get("feature_dim", 256)),
            dropout=float(config["model"].get("dropout", 0.20)),
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])

        self.checkpoint_metadata = {
            "epoch": checkpoint.get("epoch"),
            "best_val_loss": checkpoint.get("best_val_loss"),
        }

        print("Loaded predictor checkpoint metadata:")
        print(json.dumps(self.checkpoint_metadata, indent=2))

        return model

    @staticmethod
    def _prepare_image(image_01: np.ndarray) -> torch.Tensor:
        if image_01.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {image_01.shape}")

        image_01 = np.clip(image_01.astype(np.float32), 0.0, 1.0)
        tensor = torch.from_numpy(image_01).float().unsqueeze(0).unsqueeze(0)
        return tensor

    def denormalize_echo_time(self, echo_norm: float) -> Optional[float]:
        if not self.echo_stats:
            return None

        if self.echo_stats.get("normalization") != "min_max":
            return None

        echo_min = float(self.echo_stats["min"])
        echo_max = float(self.echo_stats["max"])

        return float(echo_norm * (echo_max - echo_min) + echo_min)

    @torch.no_grad()
    def predict_image(self, image_01: np.ndarray) -> Dict:
        x = self._prepare_image(image_01).to(self.device)

        outputs = self.model(x)

        manufacturer_prob = F.softmax(outputs["manufacturer_logits"], dim=1)[0]
        field_prob = F.softmax(outputs["field_strength_logits"], dim=1)[0]

        manufacturer_pred = int(torch.argmax(manufacturer_prob).item())
        field_strength_pred = int(torch.argmax(field_prob).item())

        echo_time_norm_pred = outputs["echo_time_norm"].detach().cpu().view(-1)[0].item()
        echo_time_pred = self.denormalize_echo_time(echo_time_norm_pred)

        result = {
            "manufacturer_pred": manufacturer_pred,
            "manufacturer_prob_0_GE": float(manufacturer_prob[0].item()),
            "manufacturer_prob_1_SIEMENS": float(manufacturer_prob[1].item()),
            "manufacturer_pred_confidence": float(manufacturer_prob[manufacturer_pred].item()),
            "field_strength_pred": field_strength_pred,
            "field_strength_prob_0_1p5T": float(field_prob[0].item()),
            "field_strength_prob_1_3p0T": float(field_prob[1].item()),
            "field_strength_pred_confidence": float(field_prob[field_strength_pred].item()),
            "echo_time_norm_pred": float(echo_time_norm_pred),
            "echo_time_pred": None if echo_time_pred is None else float(echo_time_pred),
        }

        return result


def compute_predictor_shift(
    original_01: np.ndarray,
    generated_01: np.ndarray,
    target_manufacturer_label: int,
    target_field_strength_label: int,
    target_echo_time_norm: float,
    validator: PredictorValidator,
) -> Dict:
    before = validator.predict_image(original_01)
    after = validator.predict_image(generated_01)

    before_mfg_target_prob = (
        before["manufacturer_prob_0_GE"]
        if target_manufacturer_label == 0
        else before["manufacturer_prob_1_SIEMENS"]
    )
    after_mfg_target_prob = (
        after["manufacturer_prob_0_GE"]
        if target_manufacturer_label == 0
        else after["manufacturer_prob_1_SIEMENS"]
    )

    before_field_target_prob = (
        before["field_strength_prob_0_1p5T"]
        if target_field_strength_label == 0
        else before["field_strength_prob_1_3p0T"]
    )
    after_field_target_prob = (
        after["field_strength_prob_0_1p5T"]
        if target_field_strength_label == 0
        else after["field_strength_prob_1_3p0T"]
    )

    shift = {
        "before": before,
        "after": after,
        "target_manufacturer_label": int(target_manufacturer_label),
        "target_field_strength_label": int(target_field_strength_label),
        "target_echo_time_norm": float(target_echo_time_norm),
        "before_target_manufacturer_prob": float(before_mfg_target_prob),
        "after_target_manufacturer_prob": float(after_mfg_target_prob),
        "target_manufacturer_prob_gain": float(after_mfg_target_prob - before_mfg_target_prob),
        "before_target_field_strength_prob": float(before_field_target_prob),
        "after_target_field_strength_prob": float(after_field_target_prob),
        "target_field_strength_prob_gain": float(after_field_target_prob - before_field_target_prob),
        "before_echo_time_norm_pred": float(before["echo_time_norm_pred"]),
        "after_echo_time_norm_pred": float(after["echo_time_norm_pred"]),
        "before_echo_abs_error_to_target": float(abs(before["echo_time_norm_pred"] - target_echo_time_norm)),
        "after_echo_abs_error_to_target": float(abs(after["echo_time_norm_pred"] - target_echo_time_norm)),
        "echo_abs_error_improvement": float(
            abs(before["echo_time_norm_pred"] - target_echo_time_norm)
            - abs(after["echo_time_norm_pred"] - target_echo_time_norm)
        ),
    }

    return shift