# ============================================================
# Dockerfile — AgTech Drone Precision Irrigation Pipeline
# ============================================================
# Multi-stage build:
#   builder  → installs Python deps into a venv
#   runtime  → copies only the venv + source; no build tools
#
# Build:
#   docker build -t agtech-irrigation .
#
# Run (Gradio dashboard on port 7860):
#   docker run -p 7860:7860 agtech-irrigation
#
# Run (headless demo, save results to host):
#   docker run --rm -v $(pwd)/results:/app/results \
#       agtech-irrigation python scripts/run_pipeline.py --demo --out-dir /app/results
# ============================================================

# ── Stage 1: builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AgTech Drone Irrigation Pipeline"
LABEL org.opencontainers.image.description="RGB+Thermal sensor fusion for precision irrigation"
LABEL org.opencontainers.image.source="https://github.com/your-org/agtech-drone-irrigation"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Minimal runtime system libs (same as builder, no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source tree
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY configs/   ./configs/
COPY notebooks/ ./notebooks/
COPY README.md  ./

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Results directory (mount a volume here in production)
RUN mkdir -p /app/results

# Gradio listens on 7860 by default
EXPOSE 7860

# Health check — verify the pipeline imports cleanly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.pipeline.fusion import process_precision_irrigation; print('OK')"

# Default: launch the interactive dashboard
CMD ["python", "-m", "src.dashboard.app"]
