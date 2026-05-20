# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2024-06-01

### Added
- Core sensor fusion pipeline (`src/pipeline/fusion.py`)
- Affine spatial registration with keypoint-based estimation fallback
- HSV canopy segmentation with Close→Open morphological cleanup
- Linear 8-bit to Celsius thermal translation
- Per-contour leaf-pixel mean temperature with STRESS / OK alarm logic
- JET colormap canopy heatmap (soil rendered as pure black)
- Synthetic farm data generator (500×500, two rows, seed-reproducible)
- Gradio 4.x interactive dashboard with Load Demo button
- Batch processing CLI with multi-threaded worker pool
- Configuration system (YAML, per-sensor profiles)
- Full pytest test suite with 80%+ coverage requirement
- GitHub Actions CI (lint + test matrix Python 3.9–3.12 + smoke test)
- Documentation: architecture, HSV tuning, calibration, real sensor integration

---

## [Unreleased]

### Planned
- NDVI channel support (requires NIR-band camera)
- Altitude-dependent registration LUT
- ONNX export for edge inference (Jetson Orin / Raspberry Pi)
- REST API wrapper (FastAPI)
- DJI Thermal SDK direct integration
