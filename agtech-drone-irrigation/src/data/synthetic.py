"""
src/data/synthetic.py
======================
Synthetic drone image pair generator for offline testing and Colab demos.

Generates two 500×500 images that mimic a nadir (straight-down) drone view:

    RGB   — Brown soil background with two rows of green circular "plants".
    Thermal — Matching temperature field: hot soil, cool Row-1 (hydrated),
              warm Row-2 (dehydrated).

Design choices
--------------
• Gaussian noise on both images prevents the segmenter from over-fitting to
  perfectly uniform colours that never occur in real fields.
• Gaussian blur on the thermal mimics the lower spatial resolution and
  thermal diffusion of a real uncooled microbolometer detector.
• Random seed is fixed (42) so outputs are bit-identical across runs,
  allowing deterministic test assertions.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Calibration constants (must match thermal.py) ─────────────────────────
TEMP_MIN_C: float = 20.0
TEMP_MAX_C: float = 45.0


def _temp_to_pixel(temp_c: float) -> int:
    """Convert a Celsius value to the corresponding 8-bit pixel intensity."""
    px = (temp_c - TEMP_MIN_C) / (TEMP_MAX_C - TEMP_MIN_C) * 255.0
    return int(np.clip(px, 0, 255))


# ── Public API ──────────────────────────────────────────────────────────────

def generate_synthetic_farm_data(
    out_rgb: str = "synthetic_rgb.jpg",
    out_thermal: str = "synthetic_thermal.jpg",
    image_size: int = 500,
    seed: int = 42,
) -> tuple[str, str]:
    """
    Create a paired mock RGB + thermal drone capture and write to disk.

    Layout
    ------
        Row 1 (y≈150): 5 plants — hydrated, mean leaf temp ≈ 26°C (cool)
        Row 2 (y≈350): 5 plants — dehydrated, mean leaf temp ≈ 34°C (hot)
        Background: bare soil at ≈ 40°C

    Parameters
    ----------
    out_rgb     : str   Output path for the RGB image.
    out_thermal : str   Output path for the thermal image.
    image_size  : int   Canvas size (square).  Default 500.
    seed        : int   NumPy RNG seed for reproducibility.

    Returns
    -------
    (out_rgb, out_thermal) : tuple[str, str]
        Paths to the saved images.
    """
    rng = np.random.default_rng(seed=seed)
    H = W = image_size

    rgb, thermal = _make_soil_background(H, W, rng)

    plant_xs = _spread_plants(W, n_plants=5, margin=60)
    radius = max(25, image_size // 14)

    # Row 1 — hydrated (cool leaves)
    _draw_plant_row(
        rgb, thermal,
        plant_xs=plant_xs,
        row_y=H // 3,
        radius=radius,
        leaf_color_bgr=(50, 180, 70),
        leaf_temp_c=26.0,
        temp_variation_c=1.5,
        rng=rng,
    )

    # Row 2 — dehydrated (warm leaves)
    _draw_plant_row(
        rgb, thermal,
        plant_xs=plant_xs,
        row_y=2 * H // 3,
        radius=radius,
        leaf_color_bgr=(40, 160, 55),
        leaf_temp_c=34.0,
        temp_variation_c=1.5,
        rng=rng,
    )

    # Thermal blur: simulates microbolometer spatial diffusion
    thermal = cv2.GaussianBlur(thermal, (7, 7), sigmaX=1.5)

    cv2.imwrite(out_rgb, rgb)
    cv2.imwrite(out_thermal, thermal)

    logger.info("Synthetic data written: %s, %s", out_rgb, out_thermal)
    return out_rgb, out_thermal


def generate_synthetic_arrays(
    image_size: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as generate_synthetic_farm_data but returns numpy arrays in memory.

    Returns
    -------
    (rgb_bgr, thermal_grey) : tuple[np.ndarray, np.ndarray]
        rgb_bgr    — HxWx3 uint8 BGR image.
        thermal_grey — HxW uint8 greyscale thermal image.
    """
    rng = np.random.default_rng(seed=seed)
    H = W = image_size

    rgb, thermal = _make_soil_background(H, W, rng)
    plant_xs = _spread_plants(W, n_plants=5, margin=60)
    radius = max(25, image_size // 14)

    _draw_plant_row(rgb, thermal, plant_xs, H // 3,      radius,
                    (50, 180, 70), 26.0, 1.5, rng)
    _draw_plant_row(rgb, thermal, plant_xs, 2 * H // 3,  radius,
                    (40, 160, 55), 34.0, 1.5, rng)

    thermal = cv2.GaussianBlur(thermal, (7, 7), sigmaX=1.5)
    return rgb, thermal


# ── Private helpers ──────────────────────────────────────────────────────────

def _make_soil_background(
    H: int, W: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy soil RGB + hot thermal background arrays."""
    # BGR (60, 80, 130) → dry earth brown.
    rgb = np.full((H, W, 3), [60, 80, 130], dtype=np.uint8)
    rgb_noise = rng.integers(-18, 18, size=(H, W, 3), dtype=np.int16)
    rgb = np.clip(rgb.astype(np.int16) + rgb_noise, 0, 255).astype(np.uint8)

    # Hot soil baseline: ~40°C
    soil_px = _temp_to_pixel(40.0)
    thermal = np.full((H, W), soil_px, dtype=np.uint8)
    t_noise = rng.integers(-10, 10, size=(H, W), dtype=np.int16)
    thermal = np.clip(thermal.astype(np.int16) + t_noise, 0, 255).astype(np.uint8)

    return rgb, thermal


def _spread_plants(width: int, n_plants: int, margin: int) -> list[int]:
    """Evenly space plant centres across the image width with side margins."""
    usable = width - 2 * margin
    step = usable // (n_plants - 1) if n_plants > 1 else usable // 2
    return [margin + i * step for i in range(n_plants)]


def _draw_plant_row(
    rgb: np.ndarray,
    thermal: np.ndarray,
    plant_xs: list[int],
    row_y: int,
    radius: int,
    leaf_color_bgr: tuple[int, int, int],
    leaf_temp_c: float,
    temp_variation_c: float,
    rng: np.random.Generator,
) -> None:
    """
    Draw one row of circular plants onto both the RGB and thermal canvases.

    Slight colour and temperature variation per plant prevents the pipeline
    from appearing artificially perfect on synthetic data.
    """
    base_px = _temp_to_pixel(leaf_temp_c)

    for cx in plant_xs:
        # RGB: add ±10 per-channel colour jitter
        color_jitter = rng.integers(-10, 10, size=3)
        r, g, b = leaf_color_bgr
        jittered = (
            int(np.clip(b + color_jitter[0], 0, 255)),
            int(np.clip(g + color_jitter[1], 0, 255)),
            int(np.clip(r + color_jitter[2], 0, 255)),
        )
        cv2.circle(rgb, (cx, row_y), radius, jittered, thickness=-1)

        # Thermal: slight per-plant temperature variation (±1.5°C)
        temp_variation_px = int(
            rng.uniform(-temp_variation_c, temp_variation_c)
            / (TEMP_MAX_C - TEMP_MIN_C) * 255
        )
        plant_px = int(np.clip(base_px + temp_variation_px, 0, 255))
        cv2.circle(thermal, (cx, row_y), radius, plant_px, thickness=-1)
