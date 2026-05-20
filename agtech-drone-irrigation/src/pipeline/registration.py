"""
src/pipeline/registration.py
============================
Spatial alignment of the thermal frame to the RGB frame.

Strategy
--------
In production multi-spectral drone payloads (DJI H20T, Parrot Sequoia,
MicaSense RedEdge) the thermal and visible cameras share a rigid mount but
are physically offset by a known, fixed distance. The lens-to-lens displacement
translates to a predictable pixel shift that is constant for a given flight
altitude and mounting geometry.

Two correction modes are provided:

    AFFINE_FIXED   — Hard-coded (dx, dy) shift from factory calibration.
                     Fastest; use when altitude is constant and the mount
                     has been characterised with a chessboard target.

    AFFINE_DYNAMIC — The caller supplies a custom 2×3 affine matrix,
                     e.g. one computed per-flight from a calibration target
                     visible in both bands.

For research / development the fixed mode is used by default with a
representative (−3, −2) pixel correction.  In production, derive the matrix
once per flight from cv2.estimateAffine2D on matched feature pairs.
"""

from __future__ import annotations

import cv2
import numpy as np

# Default registration correction — simulates a factory-calibrated payload
# with a 3 px horizontal / 2 px vertical lens offset.
# Sign convention: positive values shift the thermal image RIGHT/DOWN;
# negative values shift it LEFT/UP.
DEFAULT_CORRECTION_DX: float = -3.0
DEFAULT_CORRECTION_DY: float = -2.0


def make_correction_matrix(dx: float, dy: float) -> np.ndarray:
    """
    Build a 2×3 affine translation matrix for the given (dx, dy) pixel shift.

    Parameters
    ----------
    dx : float  Horizontal correction in pixels (negative = shift left).
    dy : float  Vertical correction in pixels   (negative = shift up).

    Returns
    -------
    np.ndarray  Shape (2, 3), dtype float32.
    """
    return np.array([[1.0, 0.0, dx],
                     [0.0, 1.0, dy]], dtype=np.float32)


# Module-level default matrix (used unless overridden in configs).
DEFAULT_MATRIX = make_correction_matrix(
    DEFAULT_CORRECTION_DX, DEFAULT_CORRECTION_DY
)


def align_thermal_to_rgb(
    thermal: np.ndarray,
    target_size: tuple[int, int],
    matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply an affine warp to bring the thermal image into pixel-perfect
    registration with the RGB frame.

    Parameters
    ----------
    thermal : np.ndarray
        8-bit greyscale thermal image (HxW).
    target_size : (width, height)
        Dimensions of the output (must match the RGB frame).
    matrix : np.ndarray | None
        2×3 affine matrix.  If None, DEFAULT_MATRIX is used.

    Returns
    -------
    np.ndarray  Aligned 8-bit greyscale image, same size as the RGB frame.

    Notes
    -----
    cv2.BORDER_REPLICATE fills the strip of pixels uncovered by the shift
    using the nearest edge value.  This avoids a black-border artefact that
    would otherwise contaminate the temperature readings near frame edges.

    An alternative, cv2.BORDER_REFLECT_101, is mathematically cleaner but
    introduces a mirror artefact visible in diagnostic visualisations.
    """
    if matrix is None:
        matrix = DEFAULT_MATRIX

    if matrix.shape != (2, 3):
        raise ValueError(
            f"Registration matrix must be shape (2,3), got {matrix.shape}"
        )

    aligned = cv2.warpAffine(
        thermal,
        matrix,
        target_size,                     # (width, height)
        flags=cv2.INTER_LINEAR,          # bilinear — smoother than NEAREST
        borderMode=cv2.BORDER_REPLICATE, # no black border artefacts
    )
    return aligned


def estimate_matrix_from_keypoints(
    rgb: np.ndarray,
    thermal_u8: np.ndarray,
) -> np.ndarray:
    """
    Estimate a registration matrix from matched ORB feature keypoints.

    This is a convenience function for the calibration workflow — run it on
    a capture that includes a visible calibration target (chessboard or
    ArUco marker) in both bands.

    Parameters
    ----------
    rgb : np.ndarray
        BGR colour frame from the RGB camera.
    thermal_u8 : np.ndarray
        8-bit greyscale thermal frame, already resized to RGB dimensions.

    Returns
    -------
    np.ndarray  2×3 affine matrix, or DEFAULT_MATRIX if not enough matches.
    """
    grey_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp_rgb, desc_rgb = orb.detectAndCompute(grey_rgb, None)
    kp_thr, desc_thr = orb.detectAndCompute(thermal_u8, None)

    if desc_rgb is None or desc_thr is None or len(kp_rgb) < 4 or len(kp_thr) < 4:
        return DEFAULT_MATRIX

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_rgb, desc_thr)
    matches = sorted(matches, key=lambda m: m.distance)[:50]

    if len(matches) < 4:
        return DEFAULT_MATRIX

    pts_rgb = np.float32([kp_rgb[m.queryIdx].pt for m in matches])
    pts_thr = np.float32([kp_thr[m.trainIdx].pt for m in matches])

    matrix, inliers = cv2.estimateAffine2D(
        pts_thr, pts_rgb,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )

    if matrix is None or (inliers is not None and inliers.sum() < 4):
        return DEFAULT_MATRIX

    return matrix.astype(np.float32)
