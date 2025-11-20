FROM python:3.11-slim
# Minimal Dockerfile for training and serving API

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies early for caching
COPY requirements.txt ./
# Remover hashes de requirements.txt para evitar conflictos con dependencias transitivas
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v '^\s*--hash' requirements.txt | grep -v '^\s*#' | grep -v '^\s*$' > requirements_no_hash.txt && \
    pip install --no-cache-dir -r requirements_no_hash.txt && \
    pip cache purge

# Copy code
COPY . .

# Ensure app directory is owned by non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Default command: serve API
CMD ["uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1
