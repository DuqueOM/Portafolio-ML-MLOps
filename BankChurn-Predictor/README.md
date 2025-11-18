# 🏦 BankChurn Predictor

**Sistema de predicción de abandono de clientes bancarios con machine learning avanzado y manejo de clases desbalanceadas**
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)](https://scikit-learn.org)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.637-green.svg)](EXECUTIVE_SUMMARY.md)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.867-brightgreen.svg)](EXECUTIVE_SUMMARY.md)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Título + 1 línea elevator (problema y valor).
BankChurn Predictor — Clasificador de churn bancario que prioriza clientes en riesgo para campañas de retención, con pipeline reproducible y API lista para demo.

## TL;DR — Cómo ejecutar demo en 3 pasos (comandos concretos).
1. `make install` 
2. `make train`   # entrena y guarda modelo en `models/` + métricas en `results/` 
3. `make api-start` y `curl -s http://localhost:8000/health | jq`  # verifica API

## Instalación (dependencias core + cómo usar Docker demo).
- Local (core):
  - `python -m venv .venv && source .venv/bin/activate` 
  - `pip install -r requirements-core.txt` 
- Desarrollo / full (tests, MLflow, Evidently):
  - `pip install -r requirements.txt`  # incluye dev + monitorización opcional
- Docker:
  - `docker build -t bankchurn-predictor .` 
  - `docker run -p 8000:8000 bankchurn-predictor` 

## Quickstart (ej: make demo o python -m main --mode demo) — entradas y salidas esperadas.
- Entrenamiento:
  - `python main.py --mode train --config configs/config.yaml --input data/raw/Churn.csv` 
  - Entrada: CSV con columnas estándar de Beta Bank en `data/raw/Churn.csv`.  
  - Salida: `models/best_model.pkl`, `models/model_v1.0.0.pkl`, `results/training_results.json`.
- Evaluación:
  - `python main.py --mode eval --config configs/config.yaml --input data/raw/Churn.csv` 
  - Salida: métricas F1/ROC-AUC + matriz de confusión en consola.
- Predicción batch:
  - `python main.py --mode predict --config configs/config.yaml --input data/new_customers.csv --output predictions.csv` 
  - Salida: `predictions.csv` con `churn_prediction`, `churn_probability`, `risk_level`.
- API docs:
  - Tras `make api-start`, abre `http://localhost:8000/docs` para la documentación interactiva de FastAPI.

## Versión actual (v1) — alcance real

- **Implementado en v1:**
  - CLI `train | eval | predict` vía `main.py`, parametrizada por `configs/config.yaml` y datasets en `data/`.
  - Modelo ensemble (LogReg + RandomForest) con manejo explícito de desbalance (resampling + class weights) y calibración de probabilidades.
  - Artefactos reproducibles: modelos en `models/`, métricas y reports en `results/`, logs en `logs/`.
  - API FastAPI (`app/fastapi_app.py`) para inferencia online y scripts de monitoreo de drift en `monitoring/`.
  - Tests de datos/modelo/API/fairness en `tests/` y soporte para MLflow (modo local `file:./mlruns`).
- **Roadmap / contenido conceptual:**
  - Extensiones de interpretabilidad avanzada (p.ej. SHAP global/local) y workflows de retraining continuo se consideran trabajo futuro, apoyado por la estructura de `docs/` y las model/data cards.

## 🚀 Demo rápida

Desde el directorio `BankChurn-Predictor/`:

```bash
# Instalar dependencias
make install

# Entrenar modelo y generar artifacts (models/, results/)
make train

# Levantar API de inferencia
make api-start

# Healthcheck
curl -s http://localhost:8000/health | jq

# Predicción de ejemplo
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @app/example_payload.json | jq
```

### Demo con Docker

```bash
docker build -t bankchurn-predictor .
docker run -p 8000:8000 bankchurn-predictor
```

## Model card summary (objetivo, datos, métricas clave, limitaciones).

- Objetivo: predecir `Exited` y priorizar clientes de alto riesgo.
- Datos: dataset sintético Beta Bank (≈10k clientes, fuerte desbalance 80/20), almacenado en `data/raw/Churn.csv`.
- Métricas típicas: F1≈0.64, ROC-AUC≈0.87 (ejemplo educativo; ver `EXECUTIVE_SUMMARY.md` y `results/training_results.json`).
- Limitaciones: datos sintéticos, riesgo de sesgo por `Geography` y `Age`; no usar en producción real sin recalibrar.

## 🛠️ Stack tecnológico

- **Lenguaje:** Python 3.8+.
- **ML:** scikit-learn (LogisticRegression, RandomForest, VotingClassifier), Optuna para hyperopt (modo avanzado).
- **MLOps / tracking:** MLflow (opcional, backend `file:./mlruns`), DVC para datos/pipelines (`dvc.yaml`).
- **API:** FastAPI + Uvicorn para servir el modelo empaquetado (`models/model_v1.0.0.pkl`).
- **Monitoreo:** scripts KS/PSI en `monitoring/` para evaluar drift de distribución.
- **Infraestructura:** Docker + `docker-compose.yml`, GitHub Actions para CI (pytest+cov, mypy, flake8).

## 📚 Documentación técnica

Para detalles de arquitectura, pipeline y decisiones de diseño, ver:

- `docs/architecture.md` — componentes principales (BankChurnPredictor, ResampleClassifier, API FastAPI, monitoring, MLflow).
- `docs/training_pipeline.md` — flujo completo de entrenamiento/evaluación, estructura de `results/training_results.json` y criterios de métricas.
- `model_card.md` — ficha del modelo (uso previsto, datos, performance, ética/fairness, SLOs).
- `data_card.md` — ficha del dataset (origen, distribución de clases, limitaciones y sesgos potenciales).
- `EXECUTIVE_SUMMARY.md` — resumen ejecutivo orientado a negocio/portafolio.

## Tests y CI (cómo correr tests).

- Local:
  - Instalar dependencias completas: `pip install -r requirements.txt` o `make install-dev`.
  - Ejecutar tests: `pytest --cov=. --cov-report=term-missing`.
- CI:
  - Workflow `.github/workflows/ci.yml` instala `requirements.txt` por proyecto y ejecuta `pytest --cov`, `mypy` y `flake8`.
  - Para este proyecto se ejecuta además un smoke-train: `python main.py --mode train --config configs/config.yaml --seed 42 --input data/raw/Churn.csv`.

## Monitorización y retraining (qué existe y qué no).

- Drift:
  - Script `monitoring/check_drift.py` con KS/PSI y reporte Evidently opcional.
  - Ejemplo: `python monitoring/check_drift.py --ref data/raw/Churn.csv --cur data/raw/Churn.csv --out-json results/drift.json --report-html results/drift_report.html` o `make check-drift`.
- MLflow (opcional):
  - Soporte local `file:./mlruns` a través de `scripts/run_mlflow.py`.
  - Ejemplo: `make mlflow-demo` (requiere dependencias de `requirements.txt`).
- Retraining:
  - Manual vía `python main.py --mode train ...` o pipeline DVC (`dvc repro`).
  - No hay scheduler ni retraining automático incluido (roadmap integrarlo con cron/CI/CD).
- Uso responsable:
  - Dataset sintético con posibles sesgos (`Geography`, `Age`); revisar `model_card.md`.
  - No usar el modelo como única fuente de decisión en contextos reales.

## Estructura del repo (breve).

- `main.py`: CLI `train|eval|predict|hyperopt`.
- `app/fastapi_app.py`: API de inferencia (`/health`, `/predict`, `/predict_batch`, `/docs`).
- `configs/config.yaml`: esquema de datos, hiperparámetros, rutas.
- `data/`: scripts de preprocesamiento y datasets (`data/raw/Churn.csv`, `data/processed/churn_processed.csv`).
- `monitoring/`: chequeo de drift KS/PSI + reporte Evidently opcional (`check_drift.py`).
- `tests/`: tests de datos, modelo, API y fairness.
- `docs/`, `model_card.md`, `data_card.md`, `EXECUTIVE_SUMMARY.md`: documentación técnica y de negocio.

```text
BankChurn-Predictor/
├── app/                  # API FastAPI y ejemplos de carga de modelo
├── configs/              # Configuración YAML (paths, split, hiperparámetros)
├── data/                 # Datos de entrada (p.ej. Churn.csv) y derivados
├── docs/                 # Documentación técnica detallada (arquitectura, pipeline)
├── monitoring/           # Scripts de chequeo de drift (KS/PSI, reports)
├── notebooks/            # Notebooks de EDA, demo y presentación
├── scripts/              # Scripts auxiliares (entrenamiento, MLflow, etc.)
├── tests/                # Tests de datos, modelo, API y fairness
├── main.py               # CLI principal (train | eval | predict | hyperopt)
├── model_card.md         # Ficha del modelo (uso, métricas, ética/fairness, SLOs)
├── data_card.md          # Ficha del dataset (origen, sesgos, gobernanza)
├── EXECUTIVE_SUMMARY.md  # Resumen ejecutivo orientado a negocio/portafolio
├── requirements*.txt     # Dependencias core/avanzadas
└── Dockerfile            # Imagen mínima para API de inferencia
```

## Contacto / autor / licencia.

- Autor: Duque Ortega Mutis (DuqueOM) — ver más contexto de negocio y métricas en `EXECUTIVE_SUMMARY.md`.
- Licencia: MIT (ver `LICENSE` en el monorepo).
- Datos: ver `DATA_LICENSE` y `data_card.md`.
