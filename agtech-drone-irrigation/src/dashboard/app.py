"""
src/dashboard/app.py
=====================
Gradio 4.x web dashboard for the precision irrigation pipeline.

Layout
------
    Left column  → inputs: RGB upload, Thermal upload, threshold slider,
                           action buttons (Run, Load Demo)
    Right column → outputs: canopy heatmap, diagnostic overlay, execution log

The app is designed to work both:
    • Locally via `python -m src.dashboard.app`
    • Inside Google Colab via `app.launch(share=True, inline=True)`

Gradio hands us numpy arrays in RGB channel order (not BGR).
The adapter in callbacks.py handles all BGR ↔ RGB conversions so the core
pipeline stays unaware of any UI concerns.
"""

from __future__ import annotations

import gradio as gr

from .callbacks import run_inference, load_demo_images


def build_dashboard() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="amber",
            font=gr.themes.GoogleFont("Inter"),
        ),
        title="AgTech Drone — Precision Irrigation",
        css=_CUSTOM_CSS,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🚁 Precision Irrigation — RGB × Thermal Sensor Fusion
            Upload paired drone captures (or click **Load Synthetic Demo**) to
            flag water-stressed crop zones in real time.

            > **Tip:** Stress threshold 30°C works well for most temperate
            > field conditions. Raise to 33–35°C for arid climates.
            """
        )

        # ── Main layout: inputs left, outputs right ─────────────────────────
        with gr.Row():

            # ── Inputs column ───────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📥 Inputs")

                rgb_in = gr.Image(
                    label="RGB Drone Image",
                    type="numpy",
                    height=260,
                    sources=["upload", "clipboard"],
                )
                thermal_in = gr.Image(
                    label="Thermal Image (8-bit greyscale)",
                    type="numpy",
                    image_mode="L",
                    height=260,
                    sources=["upload", "clipboard"],
                )
                thresh_in = gr.Slider(
                    minimum=20.0,
                    maximum=45.0,
                    value=30.0,
                    step=0.5,
                    label="🌡 Stress Temperature Threshold (°C)",
                    info="Plants whose mean leaf temp exceeds this are flagged STRESS.",
                )

                with gr.Row():
                    run_btn  = gr.Button("🔍  Run Analysis", variant="primary",  scale=2)
                    demo_btn = gr.Button("🎲  Load Demo",    variant="secondary", scale=1)

                gr.Markdown(
                    "_Supports JPEG, PNG, TIFF.  "
                    "Thermal must be single-channel 8-bit._",
                    elem_classes=["hint-text"],
                )

            # ── Outputs column ───────────────────────────────────────────────
            with gr.Column(scale=2, min_width=420):
                gr.Markdown("### 📊 Results")

                with gr.Row():
                    heatmap_out = gr.Image(
                        label="① Canopy Thermal Heatmap  (soil hidden)",
                        type="numpy",
                        height=300,
                    )
                    overlay_out = gr.Image(
                        label="② Diagnostic Overlay",
                        type="numpy",
                        height=300,
                    )

                log_out = gr.Textbox(
                    label="③ Execution Log & Profiling Metrics",
                    lines=10,
                    max_lines=14,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="Run the pipeline to see metrics here…",
                )

        # ── About accordion ─────────────────────────────────────────────────
        with gr.Accordion("ℹ️  About this pipeline", open=False):
            gr.Markdown(_ABOUT_TEXT)

        # ── Wire up callbacks ────────────────────────────────────────────────
        run_btn.click(
            fn=run_inference,
            inputs=[rgb_in, thermal_in, thresh_in],
            outputs=[heatmap_out, overlay_out, log_out],
        )

        demo_btn.click(
            fn=load_demo_images,
            inputs=[],
            outputs=[rgb_in, thermal_in],
        )

    return demo


# ── Custom CSS ───────────────────────────────────────────────────────────────

_CUSTOM_CSS = """
.hint-text { color: #888; font-size: 0.82em; margin-top: 4px; }
"""

# ── About text ───────────────────────────────────────────────────────────────

_ABOUT_TEXT = """
### Pipeline stages

1. **Spatial registration** — Affine warp corrects the fixed pixel offset
   between the RGB and thermal lenses (`cv2.warpAffine`).

2. **Canopy segmentation** — HSV inRange (Hue 35–85) isolates chlorophyll;
   morphological Close→Open removes holes and soil noise.

3. **Thermal translation** — 8-bit pixel values are mapped linearly to
   20–45 °C; soil pixels are zeroed out.

4. **Contour alarm logic** — Per-plant mean temperature (leaf pixels only)
   is compared against the threshold; STRESS / OK labels and bounding boxes
   are drawn on the overlay.

### Temperature scale

| Pixel | °C   |
|-------|------|
| 0     | 20.0 |
| 128   | 32.5 |
| 255   | 45.0 |

### Source code

[github.com/your-org/agtech-drone-irrigation](https://github.com/your-org/agtech-drone-irrigation)
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = build_dashboard()
    app.launch(
        share=True,      # Required for Google Colab
        debug=False,
        inline=True,     # Render inside Jupyter / Colab cell
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
