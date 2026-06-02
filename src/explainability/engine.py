#!/usr/bin/env python3
"""
Phase 10 — Unified Explainability Engine.

Wraps all explainability functions into one class:
  - Absolute difference map
  - Local SSIM map
  - Histogram comparison
  - Predictor shift (before/after/gain)

Usage:
    engine = ExplainabilityEngine(
        predictor_checkpoint="checkpoints/predictor/ax_t1_128/best.pt",
        label_maps_dir="data/processed/ax_t1_128/label_maps/",
    )
    results = engine.compute_all(original, counterfactual, source_cond, target_cond)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.explainability.difference_map import (
    compute_absolute_difference,
    compute_difference_summary,
)
from src.explainability.ssim_map import (
    compute_local_ssim_map,
    compute_ssim_summary,
)
from src.explainability.histogram import (
    compute_histogram,
    compute_histogram_comparison,
    histogram_l1_distance,
    histogram_intersection,
)
from src.explainability.validation import PredictorValidator


class ExplainabilityEngine:
    """
    Unified explainability engine for counterfactual MRI evaluation.

    Loads the predictor once and provides a single `compute_all()` method
    that returns every metric needed for paper figures and evaluation.
    """

    # Label decoding maps (order matches label_maps JSONs)
    MANUFACTURER_LABELS = {0: "GE MEDICAL SYSTEMS", 1: "SIEMENS"}
    FIELD_STRENGTH_LABELS = {0: "1.5T", 1: "3.0T"}

    def __init__(
        self,
        predictor_checkpoint: str = "checkpoints/predictor/ax_t1_128/best.pt",
        label_maps_dir: str = "data/processed/ax_t1_128/label_maps/",
        device: Optional[str] = None,
    ):
        self.root = Path(__file__).resolve().parents[2]
        self.label_maps_dir = self._resolve(label_maps_dir)

        # Load label maps for decoding
        self._load_label_maps()

        # Build predictor validator
        echo_stats_path = str(self.label_maps_dir / "echo_time_stats.json")
        self.validator = PredictorValidator(
            checkpoint_path=predictor_checkpoint,
            echo_stats_path=echo_stats_path,
            device=device,
        )

        self.device = self.validator.device
        print(f"ExplainabilityEngine ready on device: {self.device}")

    def _resolve(self, path_like: str) -> Path:
        p = Path(path_like)
        if p.is_absolute():
            return p
        return self.root / p

    def _load_label_maps(self):
        """Load manufacturer and field strength label maps."""
        mfg_path = self.label_maps_dir / "manufacturer_to_id.json"
        fs_path = self.label_maps_dir / "field_strength_to_id.json"
        echo_path = self.label_maps_dir / "echo_time_stats.json"

        if mfg_path.exists():
            with open(mfg_path, "r", encoding="utf-8") as f:
                self._mfg_to_id = json.load(f)
                self._id_to_mfg = {v: k for k, v in self._mfg_to_id.items()}
        else:
            self._mfg_to_id = {"GE MEDICAL SYSTEMS": 0, "SIEMENS": 1}
            self._id_to_mfg = {0: "GE MEDICAL SYSTEMS", 1: "SIEMENS"}

        if fs_path.exists():
            with open(fs_path, "r", encoding="utf-8") as f:
                self._fs_to_id = json.load(f)
                self._id_to_fs = {v: k for k, v in self._fs_to_id.items()}
        else:
            self._fs_to_id = {"1.5T": 0, "3.0T": 1}
            self._id_to_fs = {0: "1.5T", 1: "3.0T"}

        if echo_path.exists():
            with open(echo_path, "r", encoding="utf-8") as f:
                self._echo_stats = json.load(f)
        else:
            self._echo_stats = {"min": 7.88, "max": 12.272}

    def decode_manufacturer(self, label: int) -> str:
        """Convert manufacturer label to string."""
        return self._id_to_mfg.get(label, f"unknown_{label}")

    def decode_field_strength(self, label: int) -> str:
        """Convert field strength label to string."""
        return self._id_to_fs.get(label, f"unknown_{label}")

    def encode_manufacturer(self, name: str) -> int:
        """Convert manufacturer name to label."""
        return self._mfg_to_id.get(name, -1)

    def encode_field_strength(self, name: str) -> int:
        """Convert field strength name to label."""
        return self._fs_to_id.get(name, -1)

    @staticmethod
    def _validate_image(image: np.ndarray, name: str = "image"):
        """Ensure image is 2D float32 in [0, 1]."""
        if image.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {image.shape}")
        if image.min() < -0.01 or image.max() > 1.01:
            raise ValueError(
                f"{name} must be in [0, 1], got range [{image.min():.4f}, {image.max():.4f}]"
            )

    def _run_predictor(self, image_01: np.ndarray) -> Dict:
        """
        Run predictor on a single image and return decoded predictions.

        Returns dict with:
            mfg_pred_label, mfg_pred_name, mfg_probs,
            fs_pred_label, fs_pred_name, fs_probs,
            echo_norm_pred, echo_pred_ms
        """
        raw = self.validator.predict_image(image_01)

        mfg_pred = raw["manufacturer_pred"]
        fs_pred = raw["field_strength_pred"]

        return {
            "mfg_pred_label": mfg_pred,
            "mfg_pred_name": self.decode_manufacturer(mfg_pred),
            "mfg_prob_GE": raw["manufacturer_prob_0_GE"],
            "mfg_prob_SIEMENS": raw["manufacturer_prob_1_SIEMENS"],
            "mfg_confidence": raw["manufacturer_pred_confidence"],
            "fs_pred_label": fs_pred,
            "fs_pred_name": self.decode_field_strength(fs_pred),
            "fs_prob_1p5T": raw["field_strength_prob_0_1p5T"],
            "fs_prob_3p0T": raw["field_strength_prob_1_3p0T"],
            "fs_confidence": raw["field_strength_pred_confidence"],
            "echo_norm_pred": raw["echo_time_norm_pred"],
            "echo_pred_ms": raw["echo_time_pred"],
        }

    def _compute_predictor_gain(
        self,
        before: Dict,
        after: Dict,
        target_condition: Dict,
    ) -> Dict:
        """
        Compute predictor gain metrics.

        For categorical (manufacturer, field_strength):
            gain = after_target_prob - before_target_prob
            hit = (after prediction == target)

        For continuous (echo_time):
            improvement = abs(before_error) - abs(after_error)
            moved_closer = improvement > 0
        """
        target_mfg_label = target_condition["manufacturer_label"]
        target_fs_label = target_condition["field_strength_label"]
        target_echo_norm = target_condition["echo_time_norm"]

        # --- Manufacturer gain ---
        if target_mfg_label == 0:
            before_mfg_target_prob = before["mfg_prob_GE"]
            after_mfg_target_prob = after["mfg_prob_GE"]
        else:
            before_mfg_target_prob = before["mfg_prob_SIEMENS"]
            after_mfg_target_prob = after["mfg_prob_SIEMENS"]

        mfg_gain = after_mfg_target_prob - before_mfg_target_prob
        mfg_hit = (after["mfg_pred_label"] == target_mfg_label)

        # --- Field strength gain ---
        if target_fs_label == 0:
            before_fs_target_prob = before["fs_prob_1p5T"]
            after_fs_target_prob = after["fs_prob_1p5T"]
        else:
            before_fs_target_prob = before["fs_prob_3p0T"]
            after_fs_target_prob = after["fs_prob_3p0T"]

        fs_gain = after_fs_target_prob - before_fs_target_prob
        fs_hit = (after["fs_pred_label"] == target_fs_label)

        # --- EchoTime improvement ---
        before_echo_error = abs(before["echo_norm_pred"] - target_echo_norm)
        after_echo_error = abs(after["echo_norm_pred"] - target_echo_norm)
        echo_improvement = before_echo_error - after_echo_error
        echo_moved_closer = echo_improvement > 0

        return {
            # Manufacturer
            "before_mfg_target_prob": float(before_mfg_target_prob),
            "after_mfg_target_prob": float(after_mfg_target_prob),
            "mfg_gain": float(mfg_gain),
            "mfg_hit": bool(mfg_hit),
            # Field strength
            "before_fs_target_prob": float(before_fs_target_prob),
            "after_fs_target_prob": float(after_fs_target_prob),
            "fs_gain": float(fs_gain),
            "fs_hit": bool(fs_hit),
            # EchoTime
            "before_echo_error": float(before_echo_error),
            "after_echo_error": float(after_echo_error),
            "echo_improvement": float(echo_improvement),
            "echo_moved_closer": bool(echo_moved_closer),
        }

    def compute_all(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray,
        source_condition: Dict,
        target_condition: Dict,
    ) -> Dict:
        """
        Compute all explainability metrics for one original-counterfactual pair.

        Args:
            original: numpy array [H, W] normalized [0, 1]
            counterfactual: numpy array [H, W] normalized [0, 1]
            source_condition: dict with keys
                manufacturer_label, field_strength_label, echo_time_norm
            target_condition: dict with keys
                manufacturer_label, field_strength_label, echo_time_norm

        Returns:
            dict with all explainability outputs:
                diff_map, diff_summary,
                ssim_map, global_ssim,
                histogram_data,
                predictor_before, predictor_after, predictor_gain,
                source_condition, target_condition
        """
        # Validate inputs
        self._validate_image(original, "original")
        self._validate_image(counterfactual, "counterfactual")

        original = np.clip(original.astype(np.float32), 0.0, 1.0)
        counterfactual = np.clip(counterfactual.astype(np.float32), 0.0, 1.0)

        # 1. Difference map
        diff_map = compute_absolute_difference(original, counterfactual, normalize=True)
        diff_summary = compute_difference_summary(original, counterfactual)

        # 2. SSIM map
        global_ssim, ssim_map = compute_local_ssim_map(original, counterfactual)

        # 3. Histogram comparison
        histogram_data = compute_histogram_comparison(original, counterfactual)
        hist_l1 = float(histogram_data["l1_distance"])
        hist_intersection_val = float(histogram_data["intersection"])

        # 4. Predictor before/after
        predictor_before = self._run_predictor(original)
        predictor_after = self._run_predictor(counterfactual)

        # 5. Predictor gain
        predictor_gain = self._compute_predictor_gain(
            predictor_before, predictor_after, target_condition
        )

        # 6. Stability checks (did unchanged controls stay stable?)
        source_mfg = source_condition["manufacturer_label"]
        source_fs = source_condition["field_strength_label"]
        target_mfg = target_condition["manufacturer_label"]
        target_fs = target_condition["field_strength_label"]

        mfg_was_edited = (source_mfg != target_mfg)
        fs_was_edited = (source_fs != target_fs)

        # Stability: if not edited, did prediction stay the same?
        mfg_stable = True if mfg_was_edited else (
            predictor_after["mfg_pred_label"] == predictor_before["mfg_pred_label"]
        )
        fs_stable = True if fs_was_edited else (
            predictor_after["fs_pred_label"] == predictor_before["fs_pred_label"]
        )

        results = {
            # Difference map
            "diff_map": diff_map,
            "mean_abs_diff": float(diff_summary["mean_absolute_difference"]),
            "max_abs_diff": float(diff_summary["max_absolute_difference"]),
            "diff_summary": diff_summary,
            # SSIM
            "ssim_map": ssim_map,
            "global_ssim": float(global_ssim),
            # Histogram
            "histogram_data": {
                "bins": histogram_data["bin_edges"].tolist(),
                "orig_hist": histogram_data["hist_original"].tolist(),
                "cf_hist": histogram_data["hist_generated"].tolist(),
            },
            "hist_l1": hist_l1,
            "hist_intersection": hist_intersection_val,
            # Predictor
            "predictor_before": predictor_before,
            "predictor_after": predictor_after,
            "predictor_gain": predictor_gain,
            # Target hit summary
            "target_hit": {
                "mfg_hit": predictor_gain["mfg_hit"],
                "fs_hit": predictor_gain["fs_hit"],
                "echo_moved_closer": predictor_gain["echo_moved_closer"],
            },
            # Stability
            "stability": {
                "mfg_was_edited": bool(mfg_was_edited),
                "fs_was_edited": bool(fs_was_edited),
                "mfg_stable": bool(mfg_stable),
                "fs_stable": bool(fs_stable),
            },
            # Conditions (for reference)
            "source_condition": {
                "manufacturer_label": int(source_condition["manufacturer_label"]),
                "manufacturer_name": self.decode_manufacturer(source_condition["manufacturer_label"]),
                "field_strength_label": int(source_condition["field_strength_label"]),
                "field_strength_name": self.decode_field_strength(source_condition["field_strength_label"]),
                "echo_time_norm": float(source_condition["echo_time_norm"]),
            },
            "target_condition": {
                "manufacturer_label": int(target_condition["manufacturer_label"]),
                "manufacturer_name": self.decode_manufacturer(target_condition["manufacturer_label"]),
                "field_strength_label": int(target_condition["field_strength_label"]),
                "field_strength_name": self.decode_field_strength(target_condition["field_strength_label"]),
                "echo_time_norm": float(target_condition["echo_time_norm"]),
            },
        }

        return results

    def results_to_json_serializable(self, results: Dict) -> Dict:
        """
        Convert results dict to fully JSON-serializable format
        (removes numpy arrays like diff_map, ssim_map).
        """
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                continue  # Skip large arrays
            elif isinstance(value, dict):
                json_results[key] = self._dict_to_serializable(value)
            elif isinstance(value, (np.floating, np.integer)):
                json_results[key] = float(value)
            elif isinstance(value, (np.bool_,)):
                json_results[key] = bool(value)
            else:
                json_results[key] = value
        return json_results

    @staticmethod
    def _dict_to_serializable(d: Dict) -> Dict:
        """Recursively convert dict values to JSON-serializable types."""
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                out[k] = bool(v)
            elif isinstance(v, dict):
                out[k] = ExplainabilityEngine._dict_to_serializable(v)
            else:
                out[k] = v
        return out
