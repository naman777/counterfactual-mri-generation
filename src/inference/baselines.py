#!/usr/bin/env python3
"""
Simple baseline transformations before diffusion.

All inputs/outputs are normalized 2D MRI slices in [0, 1].
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def ensure_float01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def identity_baseline(image: np.ndarray) -> np.ndarray:
    """Return image unchanged."""
    return ensure_float01(image).copy()


def contrast_shift_baseline(image: np.ndarray, alpha: float = 1.10, beta: float = 0.00) -> np.ndarray:
    """Simple linear intensity/contrast baseline: output = alpha * image + beta."""
    image = ensure_float01(image)
    return np.clip(alpha * image + beta, 0.0, 1.0).astype(np.float32)


def gamma_baseline(image: np.ndarray, gamma: float = 0.90) -> np.ndarray:
    """Gamma baseline. gamma < 1 brightens mid-intensities; gamma > 1 darkens."""
    image = ensure_float01(image)
    return np.clip(np.power(image, gamma), 0.0, 1.0).astype(np.float32)


def gaussian_noise_baseline(image: np.ndarray, sigma: float = 0.03, seed: Optional[int] = None) -> np.ndarray:
    """Add light Gaussian noise."""
    image = ensure_float01(image)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def histogram_matching_baseline(source_image: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
    """Match source histogram to reference histogram."""
    source_image = ensure_float01(source_image)
    reference_image = ensure_float01(reference_image)

    try:
        from skimage.exposure import match_histograms
    except Exception as exc:
        raise ImportError("Install scikit-image: pip install scikit-image") from exc

    matched = match_histograms(source_image, reference_image)
    return np.clip(matched, 0.0, 1.0).astype(np.float32)


def make_all_baselines(image: np.ndarray, reference_image: Optional[np.ndarray] = None, seed: int = 42) -> dict[str, np.ndarray]:
    """Create all simple baseline outputs."""
    outputs = {
        "identity": identity_baseline(image),
        "contrast_shift": contrast_shift_baseline(image, alpha=1.10, beta=0.00),
        "gamma": gamma_baseline(image, gamma=0.90),
        "gaussian_noise": gaussian_noise_baseline(image, sigma=0.03, seed=seed),
    }

    if reference_image is not None:
        outputs["histogram_matching"] = histogram_matching_baseline(image, reference_image)

    return outputs
