# Architecture

## Module dependency graph

```
scripts/run_pipeline.py
    └── src/pipeline/fusion.py
            ├── src/pipeline/registration.py
            ├── src/pipeline/segmentation.py
            └── src/pipeline/thermal.py

src/dashboard/app.py
    └── src/dashboard/callbacks.py
            ├── src/pipeline/fusion.py
            ├── src/data/synthetic.py
            └── src/utils/profiler.py

src/data/
    ├── synthetic.py   (no internal deps)
    └── loaders.py     (no internal deps)

src/utils/
    ├── logger.py      (stdlib only)
    ├── profiler.py    (stdlib only)
    └── colormap.py    (opencv + numpy)
```

## Data flow

```
RGB (HxWx3 BGR uint8)          Thermal (HxW uint8)
        │                              │
        ▼                              ▼
[registration.align_thermal_to_rgb()]
        │                              │
        │               ┌─────────────┘
        │               ▼ aligned HxW uint8
        ▼
[segmentation.segment_canopy()]
→ canopy_mask HxW uint8 (leaf=255, soil=0)
        │
        ▼
[thermal.translate_to_celsius()]
→ temp_celsius HxW float32

[thermal.apply_canopy_mask_to_thermal()]
→ canopy_temp HxW float32 (soil=0.0)
        │
        ├──► [thermal.analyze_contours()]
        │      → overlay BGR + stress/ok counts
        │
        └──► [thermal.build_canopy_heatmap()]
               → heatmap BGR
```

## Thread safety

`process_precision_irrigation()` is stateless — all intermediate arrays are
local to each call.  It is safe to call from multiple threads concurrently
(as `batch_process.py` does).

The only shared mutable state is the `_TEMP_MIN_C` / `_TEMP_MAX_C` module-
level variables in `thermal.py`, modified by `set_temp_range()`.  In
multi-threaded batch processing, call `set_temp_range()` once before
spawning workers and do not call it again during execution.
