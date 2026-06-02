#!/usr/bin/env python3
"""Absolute difference map utilities."""

from __future__ import annotations
import numpy as np


def compute_absolute_difference(original: np.ndarray, generated: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute absolute pixel-wise difference map."""
    original = original.astype(np.float32)
    generated = generated.astype(np.float32)
    diff = np.abs(generated - original)

    if normalize and diff.max() > 0:
        diff = diff / float(diff.max())

    return diff.astype(np.float32)


def compute_difference_summary(original: np.ndarray, generated: np.ndarray) -> dict:
    """Return numerical summary of image change."""
    diff = np.abs(generated.astype(np.float32) - original.astype(np.float32))
    return {
        "mean_absolute_difference": float(diff.mean()),
        "median_absolute_difference": float(np.median(diff)),
        "max_absolute_difference": float(diff.max()),
        "changed_pixel_fraction_gt_0_05": float((diff > 0.05).mean()),
        "changed_pixel_fraction_gt_0_10": float((diff > 0.10).mean()),
    }


