"""
src/pipeline/thermal.py
========================
Thermal translation, canopy masking, contour-based alarm logic, and
diagnostic heatmap rendering.

Temperature model
-----------------
The 8-bit sensor encodes temperature linearly:

    T_celsius = (pixel / 255) × (T_max − T_min) + T_min

Where T_min / T_max come from configs/default.yaml (or per-frame EXIF for
real FLIR / DJI payloads).  The module reads these from module-level
constants that the dashboard and CLI override via `set_temp_range()`.

Per-contour masking strategy
-----------------------------
For each detected plant contour we:

    1. Rasterise *only that contour* into a blank mask (single_mask).
    2. Compute the element-wise AND with the global canopy_mask.

Step 2 is the critical correctness step.  A contour's bounding box often
contains soil pixels (the corners of the rectangle around a circular plant).
Without the AND, those soil pixels would inflate the mean temperature by
several degrees — enough to trigger false stress alarms on healthy crops
during peak-sun hours when bare soil reaches 40°C+.

The AND ensures we average exclusively pixels that are *both* inside this
plant's outline *and* classified as leaf by the HSV segmenter.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Calibration constants (override via set_temp_range() or config loader) ──
_TEMP_MIN_C: float = 20.0
_TEMP_MAX_C: float = 45.0

# Contour filter
MIN_PLANT_AREA_PX: int = 200

# Visual style constants
BOX_STRESS_COLOR = (0, 0, 255)    # BGR: red
BOX_OK_COLOR     = (0, 255, 0)    # BGR: green
BOX_STRESS_THICK = 3
BOX_OK_THICK     = 2
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.55
FONT_THICK       = 2


def set_temp_range(temp_min_c: float, temp_max_c: float) -> None:
    """Override the global sensor calibration range at runtime."""
    global _TEMP_MIN_C, _TEMP_MAX_C
    if temp_min_c >= temp_max_c:
        raise ValueError("temp_min_c must be strictly less than temp_max_c")
    _TEMP_MIN_C = temp_min_c
    _TEMP_MAX_C = temp_max_c
    logger.info("Temp range updated: %.1f – %.1f °C", temp_min_c, temp_max_c)


def translate_to_celsius(thermal_u8: np.ndarray) -> np.ndarray:
    """
    Convert an 8-bit thermal image to a float32 Celsius field.

    Parameters
    ----------
    thermal_u8 : np.ndarray
        HxW uint8 array from the aligned thermal sensor.

    Returns
    -------
    np.ndarray
        HxW float32 array, values in [_TEMP_MIN_C, _TEMP_MAX_C].
    """
    span = _TEMP_MAX_C - _TEMP_MIN_C
    temp = (thermal_u8.astype(np.float32) / 255.0) * span + _TEMP_MIN_C
    return temp


def apply_canopy_mask_to_thermal(
    temp_celsius: np.ndarray,
    canopy_mask: np.ndarray,
) -> np.ndarray:
    """
    Zero-out all non-canopy pixels in the temperature field.

    This is the critical fusion step — bare soil routinely reaches 38–42°C
    in summer and would dominate the per-plant averages if not removed.

    Parameters
    ----------
    temp_celsius : np.ndarray   HxW float32 temperature field.
    canopy_mask  : np.ndarray   HxW uint8 binary mask (leaf=255, soil=0).

    Returns
    -------
    np.ndarray
        HxW float32 — leaf pixels retain their temperature, soil = 0.0.

    Notes
    -----
    np.where preserves dtype; the output is float32 identical to temp_celsius
    except where the mask is 0.
    """
    return np.where(canopy_mask > 0, temp_celsius, 0.0)


def analyze_contours(
    rgb_bgr: np.ndarray,
    temp_celsius: np.ndarray,
    canopy_mask: np.ndarray,
    temp_threshold: float,
) -> tuple[np.ndarray, int, int]:
    """
    Find individual plants, compute per-plant mean temperature, annotate.

    Parameters
    ----------
    rgb_bgr       : np.ndarray   Original BGR drone image (will be copied).
    temp_celsius  : np.ndarray   HxW float32 full-frame temperature field.
    canopy_mask   : np.ndarray   HxW uint8 canopy mask (leaf=255, soil=0).
    temp_threshold: float        Stress cutoff in °C.

    Returns
    -------
    overlay       : np.ndarray   BGR annotated image.
    stress_count  : int
    healthy_count : int
    """
    overlay = rgb_bgr.copy()
    stress_count = 0
    healthy_count = 0

    # Retrieve outermost contours only — RETR_EXTERNAL ignores internal holes
    # (e.g. a plant with a brown dead centre still yields one outer contour).
    contours, _ = cv2.findContours(
        canopy_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_PLANT_AREA_PX:
            # Skip dust, small debris, isolated pixels.
            continue

        # ── Per-plant mask: rasterise this contour into a blank canvas ────
        single_mask = np.zeros_like(canopy_mask)
        cv2.drawContours(
            single_mask, [contour], -1, color=255, thickness=-1  # filled
        )

        # ── AND with global canopy mask to exclude soil inside the bbox ───
        # The intersection selects pixels that are BOTH:
        #   • inside this plant's polygon (single_mask)
        #   • classified as leaf by the HSV segmenter (canopy_mask)
        leaf_pixels = temp_celsius[
            (single_mask > 0) & (canopy_mask > 0)
        ]

        # Edge case: contour derived from canopy_mask but zero overlap after AND.
        # Extremely rare (can occur at image boundaries), handled defensively.
        if leaf_pixels.size == 0:
            logger.debug("Empty leaf pixel set for contour, skipping.")
            continue

        mean_temp = float(np.mean(leaf_pixels))
        x, y, bw, bh = cv2.boundingRect(contour)

        if mean_temp > temp_threshold:
            color     = BOX_STRESS_COLOR
            thickness = BOX_STRESS_THICK
            label     = f"STRESS {mean_temp:.1f}C"
            stress_count += 1
        else:
            color     = BOX_OK_COLOR
            thickness = BOX_OK_THICK
            label     = f"OK {mean_temp:.1f}C"
            healthy_count += 1

        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, thickness)

        # Place label above the bounding box; clamp if the box hugs the top edge.
        label_y = y - 8 if y - 8 > 12 else y + bh + 18
        cv2.putText(
            overlay, label, (x, label_y),
            FONT, FONT_SCALE, color, FONT_THICK, cv2.LINE_AA,
        )

    return overlay, stress_count, healthy_count


def build_canopy_heatmap(
    canopy_temp: np.ndarray,
    canopy_mask: np.ndarray,
) -> np.ndarray:
    """
    Render a false-colour heatmap of canopy temperatures (soil = black).

    Parameters
    ----------
    canopy_temp  : np.ndarray   HxW float32 — masked thermal field (0 on soil).
    canopy_mask  : np.ndarray   HxW uint8 — leaf=255, soil=0.

    Returns
    -------
    np.ndarray  BGR uint8 heatmap, same H×W.

    Notes
    -----
    Normalise against the fixed calibration range (not np.min/np.max of the
    current frame) so that heatmap colours are comparable between captures.
    A cold field (all leaves at 22°C) and a hot one (all at 38°C) will show
    correctly as blue-vs-red rather than both appearing mid-range.
    """
    span = _TEMP_MAX_C - _TEMP_MIN_C
    # np.clip keeps values outside the calibrated range from wrapping around
    # the colormap (pixel artefacts from sensor saturation).
    norm = np.clip((canopy_temp - _TEMP_MIN_C) / span, 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)

    heatmap = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)

    # Re-zero soil: JET maps 0 to blue, which would make "no data" (soil)
    # look like "coldest leaf" — misleading.  Pure black is visually unambiguous.
    heatmap[canopy_mask == 0] = (0, 0, 0)

    return heatmap
