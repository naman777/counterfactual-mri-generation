#!/usr/bin/env python3
"""Histogram comparison utilities."""

from __future__ import annotations
import numpy as np


def compute_histogram(image: np.ndarray, bins: int = 64, value_range: tuple[float, float] = (0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized image histogram."""
    image = np.clip(image.astype(np.float32), value_range[0], value_range[1])
    hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=value_range)
    hist = hist.astype(np.float64)

    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist.astype(np.float32), bin_edges.astype(np.float32)


def histogram_l1_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    return float(np.abs(hist_a - hist_b).sum())


def histogram_l2_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    return float(np.sqrt(((hist_a - hist_b) ** 2).sum()))


def histogram_intersection(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Higher is more similar. For normalized histograms, max is 1."""
    return float(np.minimum(hist_a, hist_b).sum())


def compute_histogram_comparison(original: np.ndarray, generated: np.ndarray, bins: int = 64) -> dict:
    """Compare original and generated histograms."""
    hist_original, bin_edges = compute_histogram(original, bins=bins)
    hist_generated, _ = compute_histogram(generated, bins=bins)

    return {
        "hist_original": hist_original,
        "hist_generated": hist_generated,
        "bin_edges": bin_edges,
        "l1_distance": histogram_l1_distance(hist_original, hist_generated),
        "l2_distance": histogram_l2_distance(hist_original, hist_generated),
        "intersection": histogram_intersection(hist_original, hist_generated),
        "original_mean_intensity": float(np.mean(original)),
        "generated_mean_intensity": float(np.mean(generated)),
        "original_std_intensity": float(np.std(original)),
        "generated_std_intensity": float(np.std(generated)),
    }
