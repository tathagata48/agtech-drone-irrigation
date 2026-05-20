# Real Sensor Integration Guide

## FLIR Radiometric JPEG (R-JPEG)

FLIR cameras with radiometric mode embed raw 16-bit temperature data inside
a standard JPEG.  Use `flirpy` to extract it:

```python
pip install flirpy

from flirpy.io.fff import FffReader
import numpy as np

reader = FffReader("capture_0001.rjpeg")
raw_temp = reader.thermal_image          # 2D float32 array in °C

# Find the per-frame range
temp_min = raw_temp.min()
temp_max = raw_temp.max()

# Normalise to 8-bit for our pipeline
norm = ((raw_temp - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)

# Update pipeline calibration constants to match this frame
from src.pipeline.thermal import set_temp_range
set_temp_range(temp_min, temp_max)

# Now run the pipeline
from src.pipeline.fusion import process_precision_irrigation
overlay, heatmap, n_stress, n_ok = process_precision_irrigation(
    rgb_bgr, norm, threshold=30.0
)
```

## DJI Zenmuse H20T / M3T

DJI Thermal SDK (TSDK) is available from the DJI developer portal.
The SDK exposes per-pixel temperature as float32 via `dirp_get_rjpeg_thermometry`:

```python
# Pseudocode — replace with actual TSDK Python binding calls
import dji_thermal_sdk as tsdk

handle = tsdk.dirp_create_from_rjpeg(rjpeg_bytes)
params = tsdk.dirp_get_measurement_params_total(handle)
temp_array = tsdk.dirp_measure_ex(handle, width, height)   # float32, °C

# Normalise as above, then call set_temp_range + process_precision_irrigation
```

## Per-frame vs static calibration

| Mode           | Accuracy   | When to use                              |
|----------------|------------|------------------------------------------|
| Static (YAML)  | ±2–5 °C    | Testing, synthetic data, fixed conditions |
| Per-frame EXIF | ±0.5–1 °C  | Production — always preferred            |

## Camera registration calibration

Run this once per new payload mount to derive your affine correction matrix:

```python
import cv2
import numpy as np
from src.pipeline.registration import estimate_matrix_from_keypoints

# Capture a scene with a visible chessboard calibration target
rgb = cv2.imread("calib_rgb.jpg")
thermal_u8 = cv2.imread("calib_thermal.jpg", cv2.IMREAD_GRAYSCALE)

# Resize thermal to RGB resolution
h, w = rgb.shape[:2]
thermal_u8 = cv2.resize(thermal_u8, (w, h))

# Estimate the matrix
M = estimate_matrix_from_keypoints(rgb, thermal_u8)
print("Correction matrix:")
print(M)
# → Update configs/your_payload.yaml with correction_dx / correction_dy
```

For rigidly-mounted co-axial cameras (< 5 cm lens separation), a pure
translation matrix suffices.  For wider separations or non-parallel optical
axes, use `cv2.findHomography` and apply with `cv2.warpPerspective` instead
of `cv2.warpAffine`.
