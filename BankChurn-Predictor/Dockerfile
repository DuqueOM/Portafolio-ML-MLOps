# Multi-stage Dockerfile para BankChurn Predictor
# Stage 1: Builder - Compila dependencias
FROM python:3.12-slim AS builder

# Metadatos
LABEL maintainer="Daniel Duque <daniel.duque@example.com>"
LABEL version="2.0.0"
LABEL description="BankChurn Predictor - Sistema de predicción de abandono de clientes bancarios"

# Variables de entorno de build
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Instalar build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar y procesar requirements
COPY requirements.txt requirements.in* ./

# Instalar dependencias en un virtualenv
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    if [ -f requirements.in ]; then \
        sed -e '/--hash=/d' -e 's/ \\$//' -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' requirements.in > requirements_clean.txt; \
    else \
        sed -e '/--hash=/d' -e 's/ \\$//' -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' requirements.txt > requirements_clean.txt; \
    fi && \
    pip install --no-cache-dir -r requirements_clean.txt

# Stage 2: Runtime - Imagen final ligera
FROM python:3.12-slim AS runtime

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PATH="/opt/venv/bin:$PATH"

# Instalar solo runtime dependencies mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear usuario no-root para seguridad
RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/app appuser

WORKDIR /app

# Copiar virtualenv desde builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copiar código fuente
COPY --chown=appuser:appuser . .

# Crear directorios necesarios con permisos correctos
RUN mkdir -p logs data/raw data/processed models results && \
    chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 8000

# Health check mejorado
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto (API)
CMD ["uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
