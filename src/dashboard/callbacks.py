"""
src/dashboard/callbacks.py
===========================
Gradio UI event handlers.

All BGR ↔ RGB conversions live here so the core pipeline (fusion.py and
friends) remains UI-agnostic and trivially testable without a running Gradio
session.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.pipeline.fusion import process_precision_irrigation
from src.data.synthetic import generate_synthetic_arrays
from src.utils.profiler import Profiler
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_inference(
    rgb_img: np.ndarray | None,
    thermal_img: np.ndarray | None,
    threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Gradio click handler for the Run Analysis button.

    Parameters
    ----------
    rgb_img     : np.ndarray | None   RGB array in RGB channel order (Gradio).
    thermal_img : np.ndarray | None   Greyscale or 3-ch array (Gradio).
    threshold   : float               Stress temperature cutoff in °C.

    Returns
    -------
    (heatmap_rgb, overlay_rgb, log_text)
        heatmap_rgb  — Canopy heatmap in RGB order for Gradio display.
        overlay_rgb  — Diagnostic overlay in RGB order.
        log_text     — Profiling and diagnostic text.
    """
    if rgb_img is None or thermal_img is None:
        return None, None, "⚠️  Please upload both an RGB and a Thermal image."

    # Gradio supplies RGB; cv2 / our pipeline expects BGR.
    rgb_bgr = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Thermal may arrive as HxWx3 (Gradio wraps greyscale in 3 channels).
    thermal_grey = thermal_img.astype(np.uint8)
    if thermal_grey.ndim == 3:
        thermal_grey = cv2.cvtColor(thermal_grey, cv2.COLOR_RGB2GRAY)

    profiler = Profiler()
    profiler.start()

    try:
        overlay_bgr, heatmap_bgr, n_stress, n_ok = process_precision_irrigation(
            rgb_bgr, thermal_grey, float(threshold)
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        return None, None, f"❌  Pipeline error: {exc}"

    elapsed_ms = profiler.stop()
    fps = (1000.0 / elapsed_ms) if elapsed_ms > 1e-9 else float("inf")

    # Back to RGB for Gradio
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    log = _format_log(elapsed_ms, fps, n_stress, n_ok, threshold,
                      rgb_bgr.shape[:2])

    return heatmap_rgb, overlay_rgb, log


def load_demo_images() -> tuple[np.ndarray, np.ndarray]:
    """
    Gradio click handler for the Load Synthetic Demo button.

    Returns
    -------
    (rgb_rgb, thermal_grey)
        rgb_rgb      — HxWx3 RGB array for display in the RGB Image widget.
        thermal_grey — HxW uint8 greyscale array for the Thermal widget.
    """
    rgb_bgr, thermal_grey = generate_synthetic_arrays(image_size=500)
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    return rgb_rgb, thermal_grey


# ── Private helpers ──────────────────────────────────────────────────────────

def _format_log(
    elapsed_ms: float,
    fps: float,
    n_stress: int,
    n_ok: int,
    threshold: float,
    frame_shape: tuple[int, int],
) -> str:
    total = n_stress + n_ok
    stress_pct = (n_stress / total * 100) if total > 0 else 0.0

    return (
        "═══════════ PRECISION IRRIGATION DIAGNOSTICS ══════════\n"
        f"  Frame size       : {frame_shape[1]} × {frame_shape[0]} px\n"
        f"  Latency          : {elapsed_ms:8.2f} ms\n"
        f"  Throughput       : {fps:8.2f} FPS\n"
        "────────────────────────────────────────────────────────\n"
        f"  Stress threshold : {threshold:.1f} °C\n"
        f"  Plants detected  : {total}\n"
        f"  Stressed zones   : {n_stress}  ({stress_pct:.0f}%)\n"
        f"  Healthy zones    : {n_ok}\n"
        "════════════════════════════════════════════════════════\n"
        + ("⚠️  IRRIGATION RECOMMENDED — stressed zones detected!\n"
           if n_stress > 0 else "✅  All zones within healthy temperature range.\n")
    )
