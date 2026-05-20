# HSV Canopy Segmentation Tuning Guide

## Why HSV?

RGB-based green detection fails when:
- **Midday sun** over-saturates the green channel differently per pixel
- **Canopy shadow** compresses all channels uniformly
- **Morning/evening** warm colour casts shift the "green" cluster

HSV Hue separates colour from brightness — Hue 35–85 tracks chlorophyll
across these conditions far more robustly.

## Default bounds

```yaml
hsv_lower: [35, 40, 40]   # Hue, Saturation, Value
hsv_upper: [85, 255, 255]
```

## Recommended adjustments by condition

| Condition                | Hue Lower | Sat Lower | Val Lower |
|--------------------------|-----------|-----------|-----------|
| Healthy crops (default)  | 35        | 40        | 40        |
| Early stress (yellowing) | 20        | 35        | 35        |
| Overcast / low light     | 35        | 30        | 30        |
| NDVI composites (false colour) | 130 | 40     | 40        |

## Interactive calibration with trackbars

Run this script to tune bounds live against a reference image:

```python
import cv2
import numpy as np

img = cv2.imread("reference_field.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def nothing(x): pass

cv2.namedWindow("Calibrate HSV")
for name, val in [("H_lo",35),("H_hi",85),("S_lo",40),("V_lo",40)]:
    cv2.createTrackbar(name, "Calibrate HSV", val, 255, nothing)

while True:
    h_lo = cv2.getTrackbarPos("H_lo", "Calibrate HSV")
    h_hi = cv2.getTrackbarPos("H_hi", "Calibrate HSV")
    s_lo = cv2.getTrackbarPos("S_lo", "Calibrate HSV")
    v_lo = cv2.getTrackbarPos("V_lo", "Calibrate HSV")

    mask = cv2.inRange(hsv,
                       np.array([h_lo, s_lo, v_lo]),
                       np.array([h_hi, 255,  255]))
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Calibrate HSV", result)

    if cv2.waitKey(1) == 27:
        print(f"Lower: [{h_lo}, {s_lo}, {v_lo}]")
        print(f"Upper: [{h_hi}, 255, 255]")
        break

cv2.destroyAllWindows()
```

## Morphological kernel size

The default kernel size (5 px) works for plants that are ≥ 15 px diameter.
For very small seedlings, reduce to 3. For large canopy cover with gaps,
increase to 7–9 and add a third CLOSE iteration.
