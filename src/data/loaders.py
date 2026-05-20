"""
src/data/loaders.py
====================
Loaders and validators for real drone imagery.

Supported inputs
----------------
    • Local file paths (JPEG, PNG, TIFF)
    • NumPy arrays (BGR or RGB for colour; 2-D or 3-D for thermal)
    • FLIR radiometric JPEG (R-JPEG) — requires flirpy or flirimageextractor
    • DJI Zenmuse H20T / M3T thermal via DJI Thermal SDK wrapper

The module provides three main public functions:

    load_rgb(source)       → BGR np.ndarray
    load_thermal(source)   → 8-bit greyscale np.ndarray
    validate_pair(rgb, th) → raises if shapes are incompatible

For real R-JPEG / DJI formats, see docs/real_sensor_integration.md.
"""

from __future__ import annotations

import pathlib
from typing import Union

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Accepted file extensions for each modality
RGB_EXTENSIONS     = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
THERMAL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp",
                      ".rjpeg", ".seq"}  # FLIR native formats

ImageSource = Union[str, pathlib.Path, np.ndarray]


def load_rgb(source: ImageSource, colour_order: str = "bgr") -> np.ndarray:
    """
    Load an RGB drone frame as a BGR numpy array.

    Parameters
    ----------
    source : str | Path | np.ndarray
        File path or pre-loaded array.
    colour_order : str
        Channel order of a pre-loaded array — 'bgr' (cv2 native) or 'rgb'
        (Gradio / PIL / matplotlib convention).  Ignored for file paths.

    Returns
    -------
    np.ndarray  HxWx3 uint8 BGR image.
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"RGB array must be HxWx3, got shape {img.shape}"
            )
        if colour_order.lower() == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    path = pathlib.Path(source)
    _assert_extension(path, RGB_EXTENSIONS, "RGB")

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")

    logger.debug("Loaded RGB %s — shape %s", path.name, img.shape)
    return img


def load_thermal(
    source: ImageSource,
    force_8bit: bool = True,
) -> np.ndarray:
    """
    Load a thermal drone frame as an 8-bit greyscale numpy array.

    Parameters
    ----------
    source : str | Path | np.ndarray
        File path or pre-loaded array.
    force_8bit : bool
        If True (default), normalise 16-bit / float thermal data to uint8
        using the per-image min/max range.  For radiometric workflows where
        the 16-bit range carries calibrated temperature data, set False and
        handle scaling in thermal.py with a custom TEMP_MIN / TEMP_MAX.

    Returns
    -------
    np.ndarray  HxW uint8 greyscale image.
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
        # Collapse accidental 3-channel greyscale (from Gradio or PIL)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.dtype != np.uint8:
            img = _normalise_to_uint8(img)
        return img

    path = pathlib.Path(source)

    # ── FLIR R-JPEG: try optional flirpy / flirimageextractor ─────────────
    if path.suffix.lower() in {".rjpeg"}:
        return _load_flir_rjpeg(path, force_8bit)

    _assert_extension(path, THERMAL_EXTENSIONS, "Thermal")

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Some 16-bit TIFFs need IMREAD_UNCHANGED
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read thermal image: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8 and force_8bit:
        img = _normalise_to_uint8(img)

    logger.debug("Loaded thermal %s — shape %s, dtype %s",
                 path.name, img.shape, img.dtype)
    return img


def validate_pair(rgb: np.ndarray, thermal: np.ndarray) -> None:
    """
    Raise ValueError if the image pair has irreconcilable properties.

    The pipeline can handle resolution mismatches (it resizes thermal to RGB),
    but it cannot recover from corrupt or zero-size arrays.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB must be HxWx3, got {rgb.shape}")
    if thermal.ndim != 2:
        raise ValueError(f"Thermal must be HxW, got {thermal.shape}")
    if rgb.size == 0:
        raise ValueError("RGB image is empty (zero size).")
    if thermal.size == 0:
        raise ValueError("Thermal image is empty (zero size).")

    if rgb.shape[:2] != thermal.shape[:2]:
        logger.warning(
            "Resolution mismatch: RGB %s, Thermal %s — will auto-resize.",
            rgb.shape[:2], thermal.shape[:2],
        )


# ── Private helpers ──────────────────────────────────────────────────────────

def _assert_extension(
    path: pathlib.Path,
    allowed: set[str],
    label: str,
) -> None:
    if path.suffix.lower() not in allowed:
        raise ValueError(
            f"{label} file extension {path.suffix!r} not in {sorted(allowed)}"
        )


def _normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Stretch an arbitrary dtype array to uint8 using its own min/max."""
    arr_f = arr.astype(np.float32)
    lo, hi = arr_f.min(), arr_f.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    normalised = (arr_f - lo) / (hi - lo) * 255.0
    return normalised.astype(np.uint8)


def _load_flir_rjpeg(path: pathlib.Path, force_8bit: bool) -> np.ndarray:
    """
    Attempt to load a FLIR radiometric JPEG using flirpy.

    Returns a normalised uint8 array if flirpy is installed, otherwise
    falls back to OpenCV IMREAD_GRAYSCALE (loses radiometric data).
    """
    try:
        from flirpy.io.seq import SeqReader  # noqa: F401  (optional dep)
        import flirpy.image.rawconverter as rc
        raw = rc.RawConverter(str(path)).thermal_image
        if force_8bit:
            return _normalise_to_uint8(raw)
        return raw
    except ImportError:
        logger.warning(
            "flirpy not installed; loading FLIR R-JPEG as plain JPEG "
            "(radiometric data lost). Install with: pip install flirpy"
        )
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read FLIR file: {path}")
        return img
