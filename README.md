<div align="center">

# 🚁 AgTech Drone — Precision Irrigation Sensor Fusion

**A production-grade computer vision pipeline that fuses aerial RGB and Thermal drone imagery to automatically detect water-stressed crops — in real time.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Google Colab](https://img.shields.io/badge/Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/tatha1234/agtech-drone-irrigation/blob/main/notebooks/AgTech_Drone_Colab.ipynb)
[![CI](https://img.shields.io/github/actions/workflow/status/tatha1234/agtech-drone-irrigation/ci.yml?label=CI&logo=githubactions&logoColor=white)](https://github.com/tatha1234/agtech-drone-irrigation/actions)

[**What it does**](#-what-it-does) · [**How it works**](#-how-it-works) · [**Quick Start**](#-quick-start) · [**Dashboard**](#-interactive-dashboard) · [**API**](#-api-reference) · [**Docs**](#-documentation)

</div>

---

## 🌱 What it does

Precision irrigation means watering **only the plants that need it** — not the whole field. This pipeline gives a drone the ability to answer that question automatically.

It takes two images captured simultaneously from a drone flying overhead:

| Input | What it shows |
|---|---|
| 📷 **RGB image** | Normal colour photo — used to locate green plants and ignore bare soil |
| 🌡️ **Thermal image** | Heat map of the ground — stressed plants run hotter because they've lost their ability to cool themselves through transpiration |

It outputs a **diagnostic overlay** with every plant labelled:

- 🟩 **OK** — plant is cool and hydrated, no action needed
- 🟥 **STRESS** — plant is overheating, irrigation required

> A healthy plant transpires water through its leaves, acting like a natural air conditioner. A water-stressed plant closes its stomata to conserve water — but as a result its temperature rises by 3–8°C above a healthy neighbour. This thermal signature is exactly what this pipeline detects.

---

## 🔬 How it works

The pipeline runs four sequential stages on every frame:

```
RGB Image ──────────────────────────────────────┐
                                                 ▼
                                    ┌────────────────────────┐
                                    │  ① Spatial Registration │
Thermal Image ──────────────────────│  Align thermal pixels   │
                                    │  to RGB pixel grid      │
                                    └────────────┬───────────┘
                                                 │
                                    ┌────────────▼───────────┐
                                    │  ② Canopy Segmentation  │
                                    │  HSV colour filter      │
                                    │  isolates green leaves  │
                                    │  Soil = ignored (black) │
                                    └────────────┬───────────┘
                                                 │
                                    ┌────────────▼───────────┐
                                    │  ③ Thermal Translation  │
                                    │  Pixel 0   →  20 °C    │
                                    │  Pixel 255 →  45 °C    │
                                    │  Soil pixels → zeroed  │
                                    └────────────┬───────────┘
                                                 │
                                    ┌────────────▼───────────┐
                                    │  ④ Contour Alarm Logic  │
                                    │  Per-plant mean temp    │
                                    │  > threshold → STRESS   │
                                    │  ≤ threshold → OK       │
                                    └────────────┬───────────┘
                                                 │
                              ┌──────────────────▼──────────────────┐
                              │           OUTPUT                      │
                              │  🗺️  Heatmap   📊  Overlay   📋  Log │
                              └──────────────────────────────────────┘
```

### Stage 1 — Spatial Registration
Drone cameras are physically offset from each other. A thermal lens and an RGB lens on the same payload see the same scene from slightly different angles. We correct this with an affine transformation matrix (`cv2.warpAffine`) that shifts the thermal frame to align pixel-for-pixel with the RGB frame.

### Stage 2 — Canopy Segmentation
We convert the RGB image to HSV colour space and threshold on **Hue 35–85** — the spectral range of chlorophyll. This is far more robust than RGB-based green detection because HSV Hue is independent of lighting intensity, so it works equally well in harsh midday sun, morning shadow, and overcast conditions. Morphological **Close→Open** operations bridge intra-leaf gaps and remove soil noise.

### Stage 3 — Thermal Translation & Masking
The 8-bit thermal sensor (pixel 0–255) is mapped linearly to real-world temperatures (20–45 °C). Then every pixel outside the canopy mask is **zeroed out** — bare soil can reach 40°C in summer and would completely swamp leaf temperatures if left in the analysis.

### Stage 4 — Contour Alarm Logic
`cv2.findContours` finds each individual plant. For each contour we compute the **mean temperature of leaf pixels only** (using a boolean AND mask to exclude any soil pixels inside the bounding box). If the mean exceeds the threshold → red STRESS box. Below → green OK box.

---

## ⚡ Quick Start

### Option A — Google Colab (recommended, zero setup)

No installation needed. Click below, then run all cells:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatha1234/agtech-drone-irrigation/blob/main/notebooks/AgTech_Drone_Colab.ipynb)

The notebook will install dependencies, generate synthetic farm data, run the pipeline, show a matplotlib preview, and launch the interactive Gradio dashboard — all in about 60 seconds.

---

### Option B — Run locally

**1. Clone and install**
```bash
git clone https://github.com/tatha1234/agtech-drone-irrigation.git
cd agtech-drone-irrigation
pip install -r requirements.txt
```

**2. Run the demo (generates synthetic data + saves results)**
```bash
python scripts/run_pipeline.py --demo
```
This creates `results/overlay.jpg` and `results/heatmap.jpg` — open them to see the output.

**3. Launch the interactive dashboard**
```bash
python scripts/run_pipeline.py --dashboard
# Open http://localhost:7860
```

**4. Process your own images**
```bash
python scripts/run_pipeline.py \
    --rgb    path/to/your_rgb.jpg \
    --thermal path/to/your_thermal.jpg \
    --threshold 30.0 \
    --out-dir results/
```

---

### Option C — Docker

```bash
docker build -t agtech-irrigation .
docker run -p 7860:7860 agtech-irrigation
# Open http://localhost:7860
```

---

## 🖥️ Interactive Dashboard

The Gradio dashboard gives you a live web UI with three panels:

```
┌─────────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
│   📥 INPUTS             │   │  🌡️ Canopy Heatmap    │   │  📊 Diagnostic     │
│                         │   │                      │   │     Overlay        │
│  [Upload RGB Image]     │   │  Soil = pure black   │   │                    │
│  [Upload Thermal Image] │──►│  Leaves = JET colors │   │  🟩 OK 26.1°C      │
│                         │   │  Blue = cool/healthy │   │  🟥 STRESS 34.2°C  │
│  🌡 Threshold slider    │   │  Red  = hot/stressed │   │                    │
│  ────────────────       │   └──────────────────────┘   └────────────────────┘
│  [🔍 Run Analysis]      │
│  [🎲 Load Demo]         │   ┌────────────────────────────────────────────────┐
└─────────────────────────┘   │  📋 Execution Log                              │
                              │  Latency: 8.2 ms  |  FPS: 121  |  Stress: 5   │
                              └────────────────────────────────────────────────┘
```

Click **🎲 Load Demo** to instantly populate both image slots with synthetic farm data — no files to upload. Drag the threshold slider and re-run to see how classification changes in real time.

---

## 🌡️ Temperature & Calibration

The default sensor model maps 8-bit pixel values linearly to temperature:

| Pixel | Temperature | Typical scene |
|-------|-------------|---------------|
| 0     | 20.0 °C     | Night-time / cool shade |
| 61    | 26.0 °C     | Hydrated leaf (healthy) |
| 143   | 34.0 °C     | Dehydrated leaf (stressed) |
| 204   | 40.0 °C     | Bare dry soil, summer |
| 255   | 45.0 °C     | Sensor saturation |

**For real FLIR / DJI payloads** the range varies per frame. Edit `configs/default.yaml`:
```yaml
thermal:
  temp_min_c: 20.0   # replace with your camera's min from EXIF
  temp_max_c: 45.0   # replace with your camera's max from EXIF
```

A ready-made profile for the **FLIR Duo Pro R** is included at `configs/flir_duo_pro.yaml`.
See [`docs/real_sensor_integration.md`](docs/real_sensor_integration.md) for FLIR SDK and DJI Thermal SDK code snippets.

---

## 🔧 HSV Tuning Guide

The default Hue range (35–85) targets healthy chlorophyll. Adjust for field conditions:

| Crop condition | Recommended Hue range | Why |
|---|---|---|
| Healthy crops (default) | 35 – 85 | Green → cyan |
| Early stress (yellowing) | 20 – 85 | Expands into yellow |
| Late stress (browning) | 10 – 85 | Expands further |
| Overcast / low light | 35 – 85, Sat > 30 | Lower sat threshold |

Run the interactive trackbar calibrator (see [`docs/hsv_tuning.md`](docs/hsv_tuning.md)) to dial in your exact field conditions visually.

---

## 📊 Performance

Benchmarks on a 500×500 frame (typical thermal sensor resolution):

| Hardware | Avg Latency | FPS |
|---|---|---|
| M2 MacBook Pro | ~8 ms | ~120 |
| Google Colab T4 | ~22 ms | ~45 |
| Raspberry Pi 4 | ~180 ms | ~6 |

For 4K RGB + upscaled thermal (4000×3000): ~210 ms / ~5 FPS on M2.

---

## 🧪 Testing

The project ships **106 tests** across unit, integration, and edge-case layers:

```bash
# Run everything
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
# open htmlcov/index.html in your browser

# Run a specific module
pytest tests/test_fusion.py -v
```

Test coverage breakdown:

| Module | Tests | Covers |
|---|---|---|
| `test_fusion.py` | 13 | End-to-end pipeline, input formats, edge cases |
| `test_integration.py` | 8 | Full-stack, immutability, determinism, FPS floor |
| `test_thermal.py` | 12 | Temperature mapping, masking, contour alarm logic |
| `test_segmentation.py` | 10 | HSV thresholds, morphology, binary mask correctness |
| `test_registration.py` | 6 | Affine matrix, border behaviour, keypoint fallback |
| `test_synthetic.py` | 11 | Data generator, pixel mapping, row temperature ordering |
| `test_loaders.py` | 11 | File I/O, format conversion, validation |
| `test_profiler.py` | 7 | Latency measurement, FPS, context manager |
| `test_config.py` | 7 | YAML loading, deep-merge, sensor profiles |
| `test_colormap.py` | 6 | JET heatmap, masking, legend rendering |

---

## 📖 API Reference

### `process_precision_irrigation()`

The main pipeline function — accepts file paths or numpy arrays:

```python
from src.pipeline.fusion import process_precision_irrigation

overlay, heatmap, n_stress, n_ok = process_precision_irrigation(
    rgb_input="drone_rgb.jpg",       # str path OR np.ndarray (BGR)
    thermal_input="drone_thermal.jpg", # str path OR np.ndarray (8-bit grey)
    temp_threshold=30.0,             # °C — plants above this → STRESS
)

# overlay  → BGR image with coloured bounding boxes and labels
# heatmap  → BGR JET colourmap of canopy temperatures (soil = black)
# n_stress → int, number of stressed plants detected
# n_ok     → int, number of healthy plants detected
```

### `generate_synthetic_farm_data()`

Create a paired RGB + Thermal test image instantly without any drone:

```python
from src.data.synthetic import generate_synthetic_farm_data

rgb_path, thermal_path = generate_synthetic_farm_data(
    out_rgb="test_rgb.jpg",
    out_thermal="test_thermal.jpg",
)
# Generates a 500×500 scene:
# Row 1 — 5 plants at ~26°C (hydrated, should be OK at threshold 30°C)
# Row 2 — 5 plants at ~34°C (dehydrated, should be STRESS at threshold 30°C)
```

### `load_config()` + `apply_config_to_pipeline()`

Override pipeline parameters from YAML without touching source code:

```python
from src.utils.config import load_config, apply_config_to_pipeline

cfg = load_config("configs/flir_duo_pro.yaml")  # deep-merges over default
apply_config_to_pipeline(cfg)                    # updates all modules
```

---

## 📁 Project Structure

```
agtech-drone-irrigation/
│
├── src/
│   ├── pipeline/
│   │   ├── fusion.py          # Main entry point — orchestrates all stages
│   │   ├── registration.py    # Affine thermal-to-RGB alignment
│   │   ├── segmentation.py    # HSV canopy segmentation + morphology
│   │   └── thermal.py         # Celsius mapping, masking, contour alarms
│   │
│   ├── data/
│   │   ├── synthetic.py       # Synthetic farm image generator
│   │   └── loaders.py         # Real image loaders (JPEG/PNG/TIFF/FLIR)
│   │
│   ├── dashboard/
│   │   ├── app.py             # Gradio Blocks UI layout
│   │   └── callbacks.py       # Button handlers, BGR↔RGB conversion
│   │
│   └── utils/
│       ├── config.py          # YAML config loader with deep-merge
│       ├── colormap.py        # JET heatmap + temperature legend
│       ├── profiler.py        # Wall-clock latency / FPS measurement
│       └── logger.py          # Structured logging factory
│
├── tests/                     # 106 pytest tests (unit + integration)
├── configs/
│   ├── default.yaml           # Default pipeline parameters
│   └── flir_duo_pro.yaml      # FLIR Duo Pro R sensor profile
├── docs/
│   ├── architecture.md        # Module dependency graph + data flow
│   ├── calibration_guide.md   # How to calibrate camera registration
│   ├── hsv_tuning.md          # Interactive HSV calibration guide
│   └── real_sensor_integration.md  # FLIR SDK + DJI Thermal SDK snippets
├── scripts/
│   ├── run_pipeline.py        # CLI: --demo, --dashboard, --rgb/--thermal
│   └── batch_process.py       # Multi-threaded batch processing
├── notebooks/
│   └── AgTech_Drone_Colab.ipynb  # Self-contained Colab notebook
├── Dockerfile                 # Multi-stage production container
└── pyproject.toml             # Package metadata + tool config
```

---

## 🤝 Contributing

Contributions are welcome — bug fixes, new sensor profiles, documentation improvements, or new features.

```bash
# Set up dev environment
git clone https://github.com/tatha1234/agtech-drone-irrigation.git
cd agtech-drone-irrigation
pip install -e ".[dev]"
pre-commit install

# Make your changes, then:
pytest tests/ -v          # must pass
ruff check src/ tests/    # must pass
black src/ tests/         # auto-formats

# Submit a PR against the `develop` branch
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow.

---

## 📄 License

MIT © 2024 — free to use, modify, and distribute. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Built with OpenCV · NumPy · Gradio · Python

*If this project helped you, consider giving it a ⭐*

</div>

