# Multi-stage Dockerfile for NLPInsight Analyzer
# Optimized for transformer model inference (CPU)
# Stage 1: Builder
FROM python:3.13.15-slim-bookworm AS builder

LABEL maintainer="Duque Ortega Mutis (DuqueOM) <DuqueOrtegaMutis@gmail.com>"
LABEL project="NLPInsight-Analyzer"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements.in* ./

# Install dependencies in venv (CPU-only torch to keep image small).
# torch via pip/urllib3 has been observed to fail the TLS handshake against
# download.pytorch.org's CDN edge from GitHub-hosted runners (reproduced
# consistently, not a one-off blip — curl against the identical URL from an
# unrelated network succeeds every time). The two clients build a different
# ClientHello (cipher order, ALPN), and some CDN/WAF edges are known to be
# pickier about Python's urllib3 fingerprint than curl's. So: try pip twice
# with backoff first (cheap, usually enough), then fall back to resolving
# the exact wheel from PyTorch's index and fetching it with curl instead.
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    ( pip install --no-cache-dir --no-compile torch --index-url https://download.pytorch.org/whl/cpu \
      || (sleep 10 && pip install --no-cache-dir --no-compile torch --index-url https://download.pytorch.org/whl/cpu) \
      || ( echo "pip install of torch failed twice via download.pytorch.org - falling back to curl" && \
           PYTAG=$(python -c "import sys;print(f'cp{sys.version_info[0]}{sys.version_info[1]}')") && \
           WHL_PATH=$(curl -fsSL --retry 3 --retry-delay 3 https://download.pytorch.org/whl/cpu/torch/ \
                      | grep -oE "href=\"/whl/cpu/torch-[0-9][^\"]*-${PYTAG}-${PYTAG}-manylinux[^\"]*x86_64\.whl" \
                      | sed 's/^href="//' | sort -V | tail -1) && \
           test -n "$WHL_PATH" && \
           curl -fsSL --retry 3 --retry-delay 3 -o /tmp/torch.whl "https://download.pytorch.org${WHL_PATH}" && \
           pip install --no-cache-dir --no-compile /tmp/torch.whl && \
           rm -f /tmp/torch.whl \
         ) \
    ) && \
    if [ -f requirements.in ]; then \
        sed -e '/--hash=/d' -e 's/ \\$//' -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' -e '/^torch/d' requirements.in > requirements_clean.txt; \
    else \
        sed -e '/--hash=/d' -e 's/ \\$//' -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' -e '/^torch/d' requirements.txt > requirements_clean.txt; \
    fi && \
    pip install --no-cache-dir --no-compile -r requirements_clean.txt && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name 'tests' -not -path '*/numpy/*' -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name 'test' -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type f -name '*.pyc' -delete 2>/dev/null; \
    find /opt/venv -type f -name '*.pyo' -delete 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/torch/test 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/torch/include 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/torch/share 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/torch/_inductor 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/torch/_dynamo 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/caffe2 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/pip 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/setuptools 2>/dev/null; \
    rm -rf /opt/venv/lib/python*/site-packages/wheel 2>/dev/null; \
    /opt/venv/bin/python -c "import fastapi, uvicorn, torch"

# Stage 2: Runtime
FROM python:3.13.15-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=models \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# Install runtime dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/app appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy configs first (change less frequently, better cache)
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser pyproject.toml ./

# Copy source code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app/ ./app/

# Create directories
RUN mkdir -p logs models artifacts data/raw .cache/huggingface && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (API)
CMD ["uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
