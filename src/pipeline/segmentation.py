"""
src/pipeline/segmentation.py
=============================
Canopy segmentation — isolating green vegetation from bare soil.

Why HSV instead of BGR / RGB?
------------------------------
Hue in HSV is a single channel that encodes perceived colour independently
of brightness.  RGB-based green detection (e.g. G > R and G > B) fails under:

    • Hard midday sunlight  → over-saturates channels differently per pixel
    • Canopy shadow         → darkens all channels uniformly, compressing ratios
    • Morning / evening     → warm colour cast shifts the "green" cluster in RGB

HSV Hue 35–85 tracks the spectral emission of chlorophyll across these
conditions far more robustly than any single RGB ratio.

Morphological cleanup sequence
--------------------------------
    CLOSE (dilate→erode): bridges intra-leaf holes caused by leaf-vein
        specular reflection and the sensor's Bayer demosaicing artefacts.
        If skipped, individual leaves fragment into dozens of tiny contours,
        multiplying false-alarm rates.

    OPEN  (erode→dilate): removes isolated soil blobs (algae patches, wet
        soil darkening, green plastic crop covers, tractor spray residue)
        that survived the HSV threshold.  Applied AFTER closing so it does
        not re-punch holes into bridged leaves.
"""

from __future__ import annotations

import cv2
import numpy as np


# ── Default HSV thresholds ────────────────────────────────────────────────────

# Hue 35–85 covers lemon-yellow-green through teal.
# Sat > 40: rejects pale sandy soil that can accidentally match low-saturation
#           green hues (Munsell 5GY4/2 type soil).
# Val > 40: rejects deep shadows — shadow pixels read near-zero across all
#           channels, so HSV places them in a low-value grey/black zone that
#           occasionally aliases into the green hue band.
HSV_LOWER_DEFAULT = np.array([35,  40,  40], dtype=np.uint8)
HSV_UPPER_DEFAULT = np.array([85, 255, 255], dtype=np.uint8)

# Morphological kernel — elliptical avoids the cardinal-axis bias of a
# rectangular kernel, which can elongate blobs along X/Y.
MORPH_KERNEL_SIZE: int = 5

# Minimum contour area in pixels — used by downstream stages, not here.
# Exposed from this module so callers only need to import from one place.
MIN_PLANT_AREA_PX: int = 200


def segment_canopy(
    bgr_image: np.ndarray,
    hsv_lower: np.ndarray | None = None,
    hsv_upper: np.ndarray | None = None,
    kernel_size: int = MORPH_KERNEL_SIZE,
) -> np.ndarray:
    """
    Produce a binary mask that is 255 on leaf pixels and 0 on bare soil.

    Parameters
    ----------
    bgr_image : np.ndarray
        Input RGB drone frame in **BGR** channel order (cv2 native).
    hsv_lower : np.ndarray | None
        Lower HSV bound (H, S, V) as uint8.  Defaults to HSV_LOWER_DEFAULT.
    hsv_upper : np.ndarray | None
        Upper HSV bound (H, S, V) as uint8.  Defaults to HSV_UPPER_DEFAULT.
    kernel_size : int
        Side length of the morphological structuring element in pixels.
        Larger values bridge wider intra-leaf gaps but may merge adjacent
        plants — tune with the interactive calibration notebook.

    Returns
    -------
    np.ndarray
        Single-channel uint8 mask, same H×W as the input.  Leaf = 255, soil = 0.
    """
    if hsv_lower is None:
        hsv_lower = HSV_LOWER_DEFAULT
    if hsv_upper is None:
        hsv_upper = HSV_UPPER_DEFAULT

    # Convert to HSV — cv2.COLOR_BGR2HSV expects BGR input.
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Threshold: pixels inside [hsv_lower, hsv_upper] → 255, rest → 0.
    raw_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Build structuring element once.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # CLOSE: fill intra-leaf holes.
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # OPEN: evict isolated soil noise.
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned


def get_calibration_bounds(
    bgr_image: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute adaptive HSV bounds from a user-supplied Region-Of-Interest
    containing known vegetation.

    Useful when lighting conditions deviate significantly from the defaults
    (e.g. overcast winter fields, early-morning frost, IR-cut filter removed).

    Parameters
    ----------
    bgr_image : np.ndarray
        Full-frame BGR drone image.
    roi : (x, y, w, h) | None
        Rectangle enclosing known-good leaf pixels.  If None, the entire
        image centre quarter is used as a rough vegetation sample.

    Returns
    -------
    (lower, upper) : tuple[np.ndarray, np.ndarray]
        HSV bounds with ±15 hue, ±40 saturation, ±40 value tolerance around
        the median of the ROI.
    """
    h, w = bgr_image.shape[:2]
    if roi is None:
        x, y, rw, rh = w // 4, h // 4, w // 2, h // 2
    else:
        x, y, rw, rh = roi

    sample = bgr_image[y : y + rh, x : x + rw]
    hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

    median = np.median(hsv_sample.reshape(-1, 3), axis=0).astype(np.uint8)

    lower = np.clip(median - np.array([15, 40, 40], dtype=np.int16), 0, 255).astype(
        np.uint8
    )
    upper = np.clip(median + np.array([15, 40, 40], dtype=np.int16), 0, 255).astype(
        np.uint8
    )
    return lower, upper
