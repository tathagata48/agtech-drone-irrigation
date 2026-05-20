"""
src/pipeline/fusion.py
======================
Top-level entry point for the precision irrigation sensor fusion pipeline.

Orchestrates four sequential stages:
    1. Spatial registration   (registration.py)
    2. Canopy segmentation    (segmentation.py)
    3. Thermal translation    (thermal.py)
    4. Contour alarm logic    (thermal.py)

This module is intentionally thin — it delegates all logic to the sub-modules
so that each stage can be unit-tested and swapped independently.
"""

from __future__ import annotations

import cv2
import numpy as np

from .registration import align_thermal_to_rgb
from .segmentation import segment_canopy
from .thermal import (
    translate_to_celsius,
    apply_canopy_mask_to_thermal,
    analyze_contours,
    build_canopy_heatmap,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def process_precision_irrigation(
    rgb_input: "str | np.ndarray",
    thermal_input: "str | np.ndarray",
    temp_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Fuse RGB + Thermal drone imagery and flag water-stressed crops.

    Parameters
    ----------
    rgb_input : str | np.ndarray
        File path OR BGR numpy array of the RGB drone capture.
    thermal_input : str | np.ndarray
        File path OR 8-bit greyscale numpy array of the thermal frame.
    temp_threshold : float
        Mean leaf temperature (°C) above which a plant is flagged STRESS.

    Returns
    -------
    overlay : np.ndarray
        BGR diagnostic image with coloured bounding boxes and text labels.
    heatmap : np.ndarray
        BGR canopy-only thermal heatmap (soil pixels are black).
    stress_count : int
        Number of plants whose mean leaf temperature exceeds temp_threshold.
    healthy_count : int
        Number of plants below temp_threshold.

    Raises
    ------
    ValueError
        If either input cannot be loaded or decoded.
    """
    # ── Load inputs ────────────────────────────────────────────────────────
    rgb_img = _load_rgb(rgb_input)
    thermal_raw = _load_thermal(thermal_input)

    h, w = rgb_img.shape[:2]

    # Resize thermal to match RGB resolution when sensors differ
    # (e.g., 12 MP RGB vs 320×240 thermal bolometer).
    if thermal_raw.shape[:2] != (h, w):
        logger.debug(
            "Resizing thermal %s → RGB %s", thermal_raw.shape[:2], (h, w)
        )
        thermal_raw = cv2.resize(
            thermal_raw, (w, h), interpolation=cv2.INTER_LINEAR
        )

    # ── Stage 1: Spatial registration ─────────────────────────────────────
    thermal_aligned = align_thermal_to_rgb(thermal_raw, (w, h))

    # ── Stage 2: Canopy segmentation ──────────────────────────────────────
    canopy_mask = segment_canopy(rgb_img)

    # ── Stage 3: Thermal → Celsius + masking ──────────────────────────────
    temp_celsius = translate_to_celsius(thermal_aligned)
    canopy_temp = apply_canopy_mask_to_thermal(temp_celsius, canopy_mask)

    # ── Stage 4: Per-plant contour analysis ───────────────────────────────
    overlay, stress_count, healthy_count = analyze_contours(
        rgb_img, temp_celsius, canopy_mask, temp_threshold
    )

    heatmap = build_canopy_heatmap(canopy_temp, canopy_mask)

    logger.info(
        "Pipeline complete — stress=%d, healthy=%d, threshold=%.1f°C",
        stress_count,
        healthy_count,
        temp_threshold,
    )
    return overlay, heatmap, stress_count, healthy_count


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_rgb(source: "str | np.ndarray") -> np.ndarray:
    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Could not load RGB image: {source!r}")
        return img
    if not isinstance(source, np.ndarray):
        raise TypeError(f"rgb_input must be str or ndarray, got {type(source)}")
    return source.copy()


def _load_thermal(source: "str | np.ndarray") -> np.ndarray:
    if isinstance(source, str):
        img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load thermal image: {source!r}")
        return img
    if not isinstance(source, np.ndarray):
        raise TypeError(
            f"thermal_input must be str or ndarray, got {type(source)}"
        )
    arr = source.copy()
    # Gradio and some loaders return HxWx3 even for greyscale images.
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr
