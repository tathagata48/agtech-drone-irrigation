"""
src/utils/colormap.py
======================
Heatmap and colour utilities for visualisation.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_jet_heatmap(
    grey: np.ndarray,
    mask: np.ndarray | None = None,
    alpha: float = 0.6,
    base_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply the JET colormap to a greyscale image with optional masking
    and alpha-blending onto a base BGR image.

    Parameters
    ----------
    grey : np.ndarray
        HxW uint8 greyscale array to colourise.
    mask : np.ndarray | None
        HxW uint8 binary mask; pixels where mask==0 are set to black.
    alpha : float
        Blend weight for overlay onto base_bgr.  1.0 = full heatmap.
    base_bgr : np.ndarray | None
        Background BGR image for blending.  If None, returns pure heatmap.

    Returns
    -------
    np.ndarray  BGR uint8 heatmap or blended image.
    """
    coloured = cv2.applyColorMap(grey, cv2.COLORMAP_JET)

    if mask is not None:
        coloured[mask == 0] = (0, 0, 0)

    if base_bgr is not None:
        blended = cv2.addWeighted(base_bgr, 1 - alpha, coloured, alpha, 0)
        if mask is not None:
            blended[mask == 0] = base_bgr[mask == 0]
        return blended

    return coloured


def draw_temperature_legend(
    image: np.ndarray,
    temp_min: float,
    temp_max: float,
    pos: tuple[int, int] = (10, 10),
    height: int = 150,
    width: int = 20,
) -> np.ndarray:
    """
    Draw a vertical JET colorbar with temperature labels on an image in-place.

    Parameters
    ----------
    image    : np.ndarray   BGR image to annotate.
    temp_min : float        Temperature at the bottom of the bar (°C).
    temp_max : float        Temperature at the top of the bar (°C).
    pos      : (x, y)       Top-left corner of the bar.
    height   : int          Bar height in pixels.
    width    : int          Bar width in pixels.

    Returns
    -------
    np.ndarray  The annotated image (same object, modified in-place).
    """
    x, y = pos
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    bar = cv2.applyColorMap(
        np.tile(gradient, (1, width)), cv2.COLORMAP_JET
    )
    image[y : y + height, x : x + width] = bar

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"{temp_max:.0f}C", (x + width + 4, y + 10),
                font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"{temp_min:.0f}C", (x + width + 4, y + height),
                font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return image
