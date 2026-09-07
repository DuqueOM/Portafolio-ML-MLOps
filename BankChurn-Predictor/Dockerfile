# Multi-stage Dockerfile for BankChurn Predictor
# Optimized for minimal image size (~600MB target)
# Stage 1: Builder
FROM python:3.13.15-slim-bookworm AS builder

LABEL maintainer="DuqueOM <DuqueOrtegaMutis@gmail.com>" \
      version="3.0.0" \
      description="BankChurn Predictor — Bank customer churn prediction API"

# Build environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy production requirements (no jupyter, dev tools, torch/CUDA)
COPY requirements-prod.txt ./

# Install deps in venv; xgboost/lightgbm --no-deps avoids nvidia-nccl-cu12 (~290MB).
# narwhals: required transitive dep of lightgbm>=4.7, excluded by --no-deps above.
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --no-compile -r requirements-prod.txt && \
    pip install --no-cache-dir --no-deps "xgboost>=1.5.0,<3.0" && \
    pip install --no-cache-dir --no-deps "lightgbm>=4.0.0,<5.0" && \
    pip install --no-cache-dir "narwhals>=1.15" && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name 'tests' -not -path '*/numpy/*' -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type f -name '*.pyc' -delete 2>/dev/null; \
    find /opt/venv -type f -name '*.pyo' -delete 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/scipy/datasets 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/pandas/tests 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/pip 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/setuptools 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/wheel 2>/dev/null; \
    /opt/venv/bin/python -c "import fastapi, uvicorn, xgboost, lightgbm"

# Stage 2: Runtime — Lightweight final image
FROM python:3.13.15-slim-bookworm AS runtime

# Runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PATH="/opt/venv/bin:$PATH"
ENV MODEL_PATH=models/model.joblib

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/app appuser

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy configs first (change less frequently, better cache)
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser pyproject.toml ./

# Copy source code (changes more frequently)
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser monitoring/ ./monitoring/

# Create required directories
RUN mkdir -p logs data/raw data/processed models results && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (API)
CMD ["uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
