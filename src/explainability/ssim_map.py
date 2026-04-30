#!/usr/bin/env python3
"""Local SSIM map utilities."""

from __future__ import annotations
import numpy as np


def compute_local_ssim_map(original: np.ndarray, generated: np.ndarray, win_size: int = 7) -> tuple[float, np.ndarray]:
    """
    Compute global SSIM and local SSIM map.
    Higher SSIM means stronger structural similarity.
    """
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:
        raise ImportError("Install scikit-image: pip install scikit-image") from exc

    original = np.clip(original.astype(np.float32), 0.0, 1.0)
    generated = np.clip(generated.astype(np.float32), 0.0, 1.0)

    score, ssim_map = structural_similarity(
        original,
        generated,
        data_range=1.0,
        win_size=win_size,
        full=True,
    )
    return float(score), ssim_map.astype(np.float32)


def compute_ssim_summary(original: np.ndarray, generated: np.ndarray) -> dict:
    """Return SSIM summary values."""
    score, ssim_map = compute_local_ssim_map(original, generated)
    return {
        "global_ssim": float(score),
        "local_ssim_mean": float(ssim_map.mean()),
        "local_ssim_min": float(ssim_map.min()),
        "low_ssim_fraction_lt_0_80": float((ssim_map < 0.80).mean()),
        "low_ssim_fraction_lt_0_90": float((ssim_map < 0.90).mean()),
    }
