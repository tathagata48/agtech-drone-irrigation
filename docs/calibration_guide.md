# Camera Registration Calibration Guide

## Overview

Multi-spectral drone payloads have physically offset lenses.  The thermal
camera and RGB camera look at the same scene from slightly different angles,
producing a pixel-level misalignment in the captured images.

Our pipeline corrects this with an affine warp — a 2×3 matrix encoding a
2D translation (and optionally rotation / scale) that maps thermal pixels
onto RGB pixel coordinates.

## One-time calibration procedure

### Equipment
- A printed or mounted chessboard / ArUco marker grid (≥ 7×5 inner corners)
- A heat source (e.g. heating pad, hot-water bottle) placed on the board to
  make it visible in both the RGB and thermal bands simultaneously
- A drone flight at the target operating altitude (e.g. 50 m AGL)

### Steps

1. **Capture a calibration frame** with the warm chessboard visible in both cameras.

2. **Run the keypoint estimator:**

```python
import cv2
from src.pipeline.registration import estimate_matrix_from_keypoints

rgb     = cv2.imread("calib_rgb.jpg")
thermal = cv2.imread("calib_thermal.jpg", cv2.IMREAD_GRAYSCALE)
h, w    = rgb.shape[:2]
thermal = cv2.resize(thermal, (w, h))

M = estimate_matrix_from_keypoints(rgb, thermal)
print(M)
# [[1.   0.   -4.8]
#  [0.   1.   -3.1]]
```

3. **Record the translation values** (M[0,2] = dx, M[1,2] = dy) and update
   `configs/your_payload.yaml`:

```yaml
registration:
  correction_dx: -4.8
  correction_dy: -3.1
```

4. **Validate** by running the pipeline on a calibration capture and checking
   that leaf bounding boxes align correctly with visible plant positions.

## Altitude dependence

For tightly co-mounted cameras (< 5 cm separation), the pixel shift is
approximately constant across typical flight altitudes (30–120 m AGL) and
only needs re-calibration when the payload is physically adjusted.

For wider separations (e.g. gimbal-mounted sensors on separate arms), the
parallax varies with altitude and you should calibrate at your most common
operating height, or implement altitude-dependent look-up tables.

## Chessboard approach (alternative, higher accuracy)

```python
import cv2
import numpy as np

# Find chessboard corners in both images
GRID = (6, 4)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

grey_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
ret_rgb, corners_rgb = cv2.findChessboardCorners(grey_rgb, GRID, flags)
ret_thr, corners_thr = cv2.findChessboardCorners(thermal, GRID, flags)

if ret_rgb and ret_thr:
    M, _ = cv2.estimateAffine2D(
        corners_thr.reshape(-1, 2),
        corners_rgb.reshape(-1, 2),
        method=cv2.RANSAC,
    )
    print("Affine matrix:", M)
```
