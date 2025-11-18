# GoldRecovery Process Optimizer

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests)

## Título + 1 línea elevator (problema y valor).
GoldRecovery Process Optimizer — Ensemble de modelos que predice y optimiza la recuperación final de oro en un circuito metalúrgico, con CLI, API y dashboard listos para demo.

## TL;DR — Cómo ejecutar demo en 3 pasos (comandos concretos).
1. `make install` 
2. `make train` 
3. `python -m app.example_load` o `uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000` y probar `/health` y `/predict`.

## Instalación (dependencias core + cómo usar Docker demo).
- Local (runtime/API v1):
  - `python -m venv .venv && source .venv/bin/activate` 
  - `pip install -r requirements-core.txt`  # CLI + ensemble XGB/LGBM/RF + API
- Full industrial stack (optimización avanzada, control de procesos, dashboards, MLflow/Evidently, tests):
  - `pip install -r requirements.txt` 
- Docker:
  - `docker build -t goldrecovery .` 
  - `docker run -p 8000:8000 -e MODEL_PATH=models/metallurgical_model.pkl goldrecovery` 

## Quickstart — entradas y salidas esperadas.
- Entrenamiento:
  - `python main.py --mode train --config configs/config.yaml --input gold_recovery_train.csv --model models/metallurgical_model.pkl` 
  - Salida: ensemble entrenado (`models/metallurgical_model.pkl`), `results/cv_results.json` con métricas de CV y métricas de test logueadas en consola.
- Evaluación:
  - `python main.py --mode eval --config configs/config.yaml --input gold_recovery_test.csv --model models/metallurgical_model.pkl` 
  - Salida: `results/metrics.json` y `results/metrics.csv` con sMAPE, MAE, RMSE y R².
- Predicción (CLI):
  - `python main.py --mode predict --config configs/config.yaml --input gold_recovery_test.csv --model models/metallurgical_model.pkl --output results/predictions.csv` 
  - Salida: CSV con `predicted_recovery` (y features limpias asociadas).
- API FastAPI:
  - `uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000` 
  - Healthcheck: `curl -s http://localhost:8000/health | jq`
  - Predicción:
    ```bash
    curl -s -X POST http://localhost:8000/predict \
      -H 'Content-Type: application/json' \
      -d '{"instances":[{"features":{"rougher.output.concentrate_au":2.1,"primary_cleaner.output.concentrate_au":3.4}}]}' | jq
    ```

## Versión actual (v1) — alcance real vs roadmap.

- **Implementado en v1:**
  - Pipeline de entrenamiento/evaluación definido en `main.py`, `evaluate.py` y `data/preprocess.py` con ensemble XGBoost + LightGBM + RandomForest (pesos en `configs/config.yaml`).
  - CLI `train|eval|predict|optimize|monitor` usando `configs/config.yaml`, export de modelo combinado (`models/model_v1.0.0.pkl`) y demo de carga en `app/example_load.py`.
  - API FastAPI (`app/fastapi_app.py`), script de monitoreo de drift, notebooks de EDA/demo y tests básicos en `tests/`.
- **Roadmap / funcionalidades futuras:**
  - Interpretabilidad avanzada (SHAP), monitoreo automatizado, integración con control de procesos y optimización económica adicional descritos en README/model_card como trabajo futuro.

## Estructura del repo (breve).
- `main.py`: CLI `train|eval|predict|optimize|monitor`.
- `app/fastapi_app.py`: API `/health`, `/predict`.
- `configs/config.yaml`: modelos, pesos del ensemble, paths.
- `data/preprocess.py`: carga y features (ratios, temporales, imputación).
- `monitoring/check_drift.py`: drift sobre columnas clave como `final.output.recovery`.
- `notebooks/`: EDA, demo, presentación.
- `tests/`: datos y modelo.

## Model card summary (objetivo, datos, métricas clave, limitaciones).
- Objetivo: predecir y estabilizar `final.output.recovery` para mejorar operación y reducir variabilidad del proceso.
- Datos: `gold_recovery_train/test/full.csv`, datos horarios de planta con variables de proceso, concentraciones y estados.
- Métricas: sMAPE/MAE/RMSE/R2 vs baseline promedio (ver `results/cv_results.json`, `results/metrics.json`).
- Limitaciones: single-site, drift mineralógico, falta de costos energéticos/insumos en el modelo base; se requiere calibración por planta.

## Tests y CI (cómo correr tests).
- Local: `pytest` en `tests/` (por ejemplo `pytest -q` o `pytest --cov=. --cov-report=term-missing`).
- CI: el workflow global `.github/workflows/ci.yml` instala `requirements.txt` para este proyecto y ejecuta `pytest --cov=.`, `mypy` y `flake8`.

## Monitorización y retraining (qué existe y qué no).
- Drift:
  - `make check-drift` → usa `monitoring/check_drift.py` para comparar distribuciones entre un CSV de referencia y uno actual usando KS/PSI.
  - La salida JSON incluye `max_psi`, `max_ks` y `recommended_action` ∈ {`ok`, `review`, `retrain`} (ver criterios en el script).
- Retraining: manual vía CLI (`train`); no hay scheduler ni retraining automático basado en drift (roadmap integrarlo con CI/CD y monitorización continua).
- MLflow: `MLFLOW_TRACKING_URI=file:./mlruns MLFLOW_EXPERIMENT=GoldRecovery make mlflow-demo` para registrar runs (opcional).

## Contacto / autor / licencia.
- Autor: Duque Ortega Mutis (DuqueOM).
- Licencias: `LICENSE`, `DATA_LICENSE`.
- Documentación ampliada: `model_card.md`, `data_card.md`.

---

## Motivación y objetivo

- Optimizar la recuperación metálica (Au/Ag/Pb) y estabilizar la variabilidad del proceso.
- Proveer un workflow repetible: entrenamiento, evaluación, inferencia y monitoreo.
- Facilitar integración con sistemas operacionales mediante una API REST.

## Dataset

- Archivos: `gold_recovery_train.csv`, `gold_recovery_test.csv`, `gold_recovery_full.csv`.
- Licencia: ver `DATA_LICENSE`.
- Tamaño: datos horarios, múltiples variables de proceso, concentraciones y estados.
- Splits: train/test provistos; el pipeline permite `train_test_split` adicional.
- Principales features: concentraciones por etapa, estados de celdas, dosificación, tamaño de partícula, timestamp.
- Problemas conocidos: valores faltantes, outliers y potencial drift entre periodos.

## Preprocesamiento

- El preprocesamiento se implementa en `data/preprocess.py` y sigue, a grandes rasgos, los siguientes pasos:
  - **Carga y combinación de CSVs** (`preprocess.load_csvs`): concatena `gold_recovery_train.csv`, `gold_recovery_test.csv` y/o otros archivos de proceso.
  - **Cálculo opcional de recovery intermedio** (`compute_recovery`): si falta `rougher.output.recovery` pero existen las columnas de feed/concentrado/relave, se calcula la recuperación según la fórmula estándar del proyecto.
  - **Limpieza básica** (`basic_clean`):
    - Convierte `date` a tipo datetime y ordena cronológicamente.
    - Filtra filas con `final.output.recovery` fuera del rango [0, 100].
    - Elimina columnas con más de 60% de valores nulos.
  - **Features derivadas** (`create_features`):
    - Ratios de recuperación Au/Ag entre etapas (p. ej. `primary_cleaner.output.concentrate_au` / `rougher.output.concentrate_au`).
    - Features temporales a partir de `date` (`hour`, `day_of_week`, `month`) para capturar patrones de turno/día/mes.
  - **Imputación de valores faltantes** (`fill_missing_with_median`):
    - Imputación por mediana en todas las columnas numéricas restantes.

El resultado es un DataFrame limpio y enriquecido, listo para entrenar el ensemble de regresores definido en `configs/config.yaml` y consumido por `main.py`.

## Baselines

Se utilizan dos baselines para evaluar el rendimiento del modelo:

- Baseline 1: modelo de regresión lineal
- Baseline 2: modelo de regresión lineal con selección de características

## Modelos probados

Se prueban varios modelos de machine learning para predecir la recuperación metálica:

- Modelo 1: Random Forest
- Modelo 2: Gradient Boosting
- Modelo 3: Support Vector Machine

## Entrenamiento

El entrenamiento se realiza mediante el uso de la biblioteca Scikit-learn. Se utiliza la función `train_test_split` para dividir los datos en conjuntos de entrenamiento y prueba.

## Validación y métricas

Se utiliza la función `cross_val_score` para evaluar el rendimiento del modelo mediante validación cruzada. Se calculan las métricas de rendimiento siguientes:

- sMAPE (Symmetric Mean Absolute Percentage Error)
- MAE (Mean Absolute Error)

## Resultados

Los resultados muestran que el modelo de Gradient Boosting es el que mejor se desempeña en la predicción de la recuperación metálica.

## Interpretabilidad y análisis de errores

Se utiliza la biblioteca SHAP para analizar la contribución de cada característica en la predicción del modelo.

## Robustez y tests

Se realizan pruebas de robustez para evaluar la estabilidad del modelo ante cambios en los datos.

## Reproducibilidad

Comandos exactos:

```bash
# 1) Instalar
make install

# 2) Entrenar
make train

# 3) Evaluar (usa test del YAML)
make eval

# 4) Predecir sobre CSV arbitrario
python main.py --mode predict --model models/metallurgical_model.pkl \
 --input gold_recovery_test.csv --output results/predictions.csv
```

Ejecución directa con configuración explícita:

```bash
python main.py --mode train --config configs/config.yaml --seed 42 --input gold_recovery_train.csv
python main.py --mode evaluate --config configs/config.yaml --input gold_recovery_test.csv
```

## Estructura de carpetas

```
GoldRecovery-Process-Optimizer/
├─ app/
│  ├─ fastapi_app.py
│  ├─ streamlit_dashboard.py
│  └─ example_load.py
├─ configs/
│  └─ config.yaml
├─ data/
│  └─ preprocess.py
├─ monitoring/
│  └─ check_drift.py
├─ notebooks/
│  ├─ demo.ipynb
│  ├─ exploratory.ipynb
│  └─ presentation.ipynb
├─ scripts/
│  ├─ recovery_simulation.py
│  └─ run_mlflow.py
├─ tests/
│  ├─ test_data.py
│  └─ test_model.py
├─ gold_recovery_train.csv
├─ gold_recovery_test.csv
├─ gold_recovery_full.csv
├─ evaluate.py
├─ main.py
├─ Makefile
├─ Dockerfile
├─ docker-compose.yml
├─ model_card.md
├─ data_card.md
├─ DATA_LICENSE
├─ LICENSE
└─ README.md
```

## Despliegue

- FastAPI: `app/fastapi_app.py`.
- Levantar localmente:

```bash
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

- Docker:

```bash
docker build -t goldrecovery:latest .
docker run -p 8000:8000 -e MODEL_PATH=models/metallurgical_model.pkl goldrecovery:latest
```

- Docker Compose:

```bash
docker compose up --build
```

- Endpoints:

```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/predict \
 -H 'Content-Type: application/json' \
 -d '{
   "instances": [
     {"features": {"rougher.output.concentrate_au": 2.1, "primary_cleaner.output.concentrate_au": 3.4,
                     "rougher.output.concentrate_ag": 1.2, "primary_cleaner.output.concentrate_ag": 2.0}}
   ]
 }'
```

## Costos y limitaciones

- Costo computacional moderado; GBMs escalan sublinealmente con features/filas.
- Limitación por drift y rango operacional; se recomienda monitoreo continuo.
- Aspectos éticos: no usar predicciones para operar fuera de límites de seguridad.

## Próximos pasos

- Añadir SHAP y monitoreo de drift (PSI) automatizado.
- Integrar MLflow/W&B para tracking y versionado.
- Optimización de hiperparámetros con Optuna.

## Créditos y referencias

- Autor: Daniel Duque — MIT License (ver `LICENSE`).
- Basado en prácticas de MLOps y control de procesos.

## FAQ (reclutador)

  - Semillas fijas, `configs/config.yaml`, `Makefile`, Dockerfile y artefactos versionados en `results/`.
- ¿Qué baseline usaste y por qué?
  - Dummy promedio para acotar MAE mínimo esperable; compara contra ensemble para cuantificar ganancia.
- ¿Cómo manejas datos faltantes y outliers?
  - Imputación por mediana y filtros de validez en `preprocess.py`; análisis IQR/KS en notebooks.
- ¿Cómo previenes leakage?
  - Separación temporal/split explícito, y features derivadas sin look-ahead.
- ¿Qué harías para producción?
  - Monitoreo de drift, validación en línea, retraining programado, límites operacionales y auditoría de versiones.

Proyecto profesional, listo para demo y despliegue local. Ver `model_card.md` y `DATA_LICENSE`.
