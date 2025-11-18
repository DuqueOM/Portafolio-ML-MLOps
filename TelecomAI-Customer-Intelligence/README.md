# TelecomAI Customer Intelligence — Producción

![python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![license](https://img.shields.io/badge/License-MIT-yellow.svg)
![ci](https://github.com/DuqueOM/Projects_Data_Scientist/actions/workflows/ci.yml/badge.svg)

## Título + 1 línea elevator (problema y valor).
TelecomAI Customer Intelligence — Clasificador reproducible que recomienda plan Ultra vs Smart en función del uso, con API y Docker listos para demo en 5 minutos.

## TL;DR — Cómo ejecutar demo en 3 pasos (comandos concretos).
1. `make install` 
2. `make start-demo`  # entrena y levanta API en Docker
3. `curl -s http://localhost:8000/health | jq` y luego un `POST` `/predict` de ejemplo.

## Instalación (dependencias core + cómo usar Docker demo).
- Local:
  - `python -m venv .venv && source .venv/bin/activate` 
  - `pip install -r requirements-core.txt`  # runtime mínimo (CLI + API)
- Full (MLflow, SHAP, Evidently, tests, notebooks):
  - `pip install -r requirements.txt` 
- Docker:
  - `docker compose up --build -d`  # usando `docker-compose.yml`, levanta API en 8000
  - o `docker build -t telecomai . && docker run -p 8000:8000 telecomai` 

## Quickstart — entradas y salidas esperadas.
- Entrenamiento:
  - `python main.py --mode train --config configs/config.yaml` 
  - Entrada: `users_behavior.csv` con columnas numéricas (`calls`, `minutes`, `messages`, `mb_used`).  
  - Salida: `artifacts/model.joblib`, `artifacts/preprocessor.joblib`, `artifacts/metrics.json` + gráficas en `artifacts/` (`roc_curve.png`, `confusion_matrix.png`).
- Evaluación:
  - `python main.py --mode eval --config configs/config.yaml` 
  - Salida: métricas actualizadas en `artifacts/metrics.json` y gráficas de ROC y matriz de confusión.
- Predicción batch:
  - `python main.py --mode predict --config configs/config.yaml --input_csv users_behavior.csv --output_path artifacts/preds.csv` 
  - Salida: CSV con columnas originales + `pred_is_ultra` y `proba_is_ultra`.
- API FastAPI:
  - `uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000`
  - Healthcheck: `curl -s http://localhost:8000/health | jq`
  - Predicción ejemplo:
    ```bash
    curl -s -X POST http://localhost:8000/predict \
      -H 'Content-Type: application/json' \
      -d '{"calls":85,"minutes":516.7,"messages":56,"mb_used":22696.96}' | jq
    ```

## Estructura del repo (breve).
- `main.py`: CLI `train|eval|predict`.
- `app/fastapi_app.py`: API (`/health`, `/predict`) usando pipeline sklearn.
- `configs/config.yaml`: paths, features, split, parámetros de Logistic Regression y configuración MLflow opcional.
- `data/preprocess.py`: carga y preprocesamiento numérico.
- `tests/`: datos, modelo, API E2E.
- `monitoring/check_drift.py`: script KS/PSI con Evidently opcional.
- `scripts/run_mlflow.py`: ejemplo de tracking MLflow.
- `model_card.md`, `data_card.md`: documentación extendida de modelo y datos.

## Model card summary (objetivo, datos, métricas clave, limitaciones).
- Objetivo: predecir `is_ultra` para recomendar plan (Ultra vs Smart).
- Datos: ~3.2k filas sintéticas de uso mensual de voz/SMS/datos (`users_behavior.csv`).
- Métricas: accuracy, F1 y ROC-AUC en `artifacts/metrics.json`.
- Limitaciones: dataset educativo, sin atributos demográficos; drift probable en escenarios reales.
- Fairness y atributos sensibles: el dataset no contiene variables sensibles (edad, género, región, etc.), por lo que **no se incluyen fairness tests** por segmento. En un entorno real se mitigaría añadiendo atributos sensibles anonimizados, diseñando tests de paridad (TPR/FPR) por grupo, estableciendo umbrales de aceptación y documentando resultados y decisiones en la model card antes de desplegar.

## Tests y CI (cómo correr tests).
- Local: desde la carpeta `TelecomAI-Customer-Intelligence/`, con el entorno virtual activado y las dependencias instaladas (`pip install -r requirements.txt`), `make test` ejecuta `pytest` sobre datos, modelo y API E2E (`tests/`).
- CI: el workflow global `.github/workflows/ci.yml` ejecuta este proyecto dentro de una matriz, añade el directorio del proyecto al `PYTHONPATH` y, desde la raíz del proyecto, lanza `pytest --cov=.`, `mypy .` y `flake8 .`.

## Monitorización y retraining (qué existe y qué no).
- Drift: `python monitoring/check_drift.py --ref users_behavior.csv --cur users_behavior.csv --features calls minutes messages mb_used --out artifacts/drift_report.json` (opcionalmente genera HTML de Evidently si está instalado).
- MLflow: `make mlflow-demo` para registrar métricas y artefactos si MLflow está activo.
- Retraining: manual vía CLI (`python main.py --mode train ...`); no hay scheduler ni job de retraining automático (roadmap integrarlo con cron/CI/CD).

## Contacto / autor / licencia.
- Autor: Duque Ortega Mutis (DuqueOM).
- Licencias: `LICENSE`, `DATA_LICENSE`.
- Documentación ampliada: ver `model_card.md` y `data_card.md`.

## Resumen Ejecutivo
Clasificador binario para recomendar plan Ultra (1) vs Smart (0) a partir del uso: `calls`, `minutes`, `messages`, `mb_used`. Proyecto listo para portafolio y producción: CLI reproducible por YAML, evaluación automatizada, API con FastAPI, Docker, tests y notebooks.

---

## Motivación y Objetivo
- **Motivación:** Optimizar la asignación de planes móviles con base en el comportamiento de uso para mejorar ARPU y satisfacción.
- **Objetivo:** Predecir `is_ultra` y entregar una recomendación de plan con métricas auditables y pipeline reproducible.

## Estructura de Carpetas
```
TelecomAI-Customer-Intelligence/
├── app/
│   ├── fastapi_app.py
│   └── example_load.py
├── configs/
│   └── config.yaml
├── data/
│   └── preprocess.py
├── notebooks/
│   ├── demo.ipynb
│   ├── exploratory.ipynb
│   └── presentation.ipynb
├── monitoring/
│   └── check_drift.py
├── scripts/
│   └── run_mlflow.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api_e2e.py
├── artifacts/                # generado en runtime
├── models/
├── main.py
├── evaluate.py
├── requirements.txt           # dependencias mínimas del proyecto
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── model_card.md
├── data_card.md
├── LICENSE
├── DATA_LICENSE
└── users_behavior.csv
```

## Dataset
- **Archivo:** `users_behavior.csv` (~3.2K filas)
- **Features:** `calls`, `minutes`, `messages`, `mb_used`
- **Target:** `is_ultra` (0/1)
- **Licencia:** Educativa/portafolio (ver `DATA_LICENSE`). Sin PII.
- **Splits:** 80/20 estratificado (configurable en `configs/config.yaml`).

## Preprocesamiento
- Imputación mediana y estandarización (sklearn `ColumnTransformer`).
- Pipeline definido en `data/preprocess.py`. Sólo variables numéricas.

## Baselines
- **Regla mayoritaria:** Predecir siempre la clase más frecuente.
- **Modelo simple:** Regresión Logística v1 (`liblinear`, `class_weight=balanced`).

## Modelos Probados y Configuración
- v1: `logreg` — parámetros en `configs/config.yaml` (`C=1.0`, `penalty=l2`, `solver=liblinear`, `class_weight=balanced`).
- Justificación: patrón tabular con relaciones aproximadamente lineales; baseline fuerte, interpretable, rápido.

## Entrenamiento
- Split 80/20 estratificado, `seed=42` (configurable).
- Artefactos guardados en `artifacts/`:
  - `model.joblib`, `preprocessor.joblib`
  - `metrics.json`, `confusion_matrix.png`, `roc_curve.png`

## Validación y Métricas
- Métricas primarias: `f1`, `roc_auc`.
- Secundarias: `accuracy`, `precision`, `recall`.
- Curvas: ROC (`roc_curve.png`) y matriz de confusión (`confusion_matrix.png`).
- Comparativa con baseline (regla mayoritaria). Los valores se generan al entrenar.

## Reproducibilidad — Comandos
```bash
# Instalación
make install

# Entrenar
make train

# Evaluar
make eval

# Inferencia por lote (salida en artifacts/predictions.csv)
make predict

# Servir API local
make serve

# Docker API
docker compose up --build -d
```

## Demo en 5 minutos
```bash
make start-demo             # instala deps, entrena y levanta API en Docker
curl -s http://localhost:8000/health | jq
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"calls":85,"minutes":516.7,"messages":56,"mb_used":22696.96}' | jq
```

## How to run demo locally
```bash
make install
make train
make serve                  # uvicorn local
# en otra terminal
python app/example_load.py  # ejemplo de carga de modelo exportado
```

## MLflow Tracking
- Habilitado opcionalmente vía `configs/config.yaml` → `mlflow`.
- Ejecutar ejemplo de logging:
```bash
make mlflow-demo
```
- Al entrenar se exporta el pipeline a `models/model_v1.0.0.pkl` y se registran métricas/artefactos (si MLflow está activo).

## Monitoring (drift)
Ejemplo de chequeo KS/PSI y reporte Evidently (opcional):
```bash
python monitoring/check_drift.py \
  --ref users_behavior.csv \
  --cur users_behavior.csv \
  --features calls minutes messages mb_used \
  --out artifacts/drift_report.json
```

## CLI (main.py)
```bash
python main.py --mode train   --config configs/config.yaml
python main.py --mode eval    --config configs/config.yaml
python main.py --mode predict --config configs/config.yaml --input_csv users_behavior.csv --output_path artifacts/preds.csv
```

## Despliegue (FastAPI)
- Endpoint: `POST /predict`
- Payload de ejemplo:
```json
{
  "calls": 85,
  "minutes": 516.7,
  "messages": 56,
  "mb_used": 22696.96
}
```
- Respuesta:
```json
{"prediction": 1, "probability_is_ultra": 0.83}
```

## Tests
```bash
make test
```
Incluyen: validación de esquema del dataset y smoke test de entrenamiento/inferencia (`tests/`).
Además: test E2E del endpoint `/predict`.

## Interpretabilidad y Análisis de Errores
- Importancias (coeficientes de LR) y revisión de falsos positivos/negativos.
- EDA y análisis por segmentos en `notebooks/exploratory.ipynb`.

## Robustez y Buenas Prácticas
- Desbalanceo: `class_weight=balanced`.
- Fijación de semillas (`random_seed` en config).
- Tests unitarios básicos (`pytest`).
- Evitar fuga de información: preprocesamiento dentro de `Pipeline`.

## Costos y Limitaciones
- Entrenamiento en CPU < 1 min (dataset actual).
- Limitaciones: posible drift; dataset educativo (riesgo de sesgos / falta de cobertura).

## Ejemplo numérico de impacto (ilustrativo)

- Supón una base de ~3,200 clientes, de los cuales ~40% serían elegibles para Ultra (`is_ultra=1`).
- Si el modelo alcanza, por ejemplo, **recall=0.80** y **precision=0.75** sobre la clase `is_ultra=1`:
  - Clientes realmente elegibles: ≈1,280.
  - Clientes elegibles correctamente identificados (TP): ≈1,024 (80% de 1,280).
  - Clientes contactados por recomendación (TP+FP): ≈1,365.
- Asumiendo un **uplift medio de ARPU de +5 USD/mes** para quienes migran correctamente a Ultra y **coste de contacto de 0.5 USD** por cliente:
  - Ingreso bruto mensual estimado por recomendaciones correctas: 1,024 × 5 ≈ **5,120 USD/mes**.
  - Coste de contactos: 1,365 × 0.5 ≈ **682 USD/mes**.
  - **ROI neto mensual aproximado**: ≈ 4,438 USD (ejemplo puramente ilustrativo basado en datos sintéticos).

## Lógica de elección de threshold

- La implementación actual utiliza el **threshold por defecto de scikit-learn (0.5)** a partir de `predict_proba` → `predict`.
- Para producción, la recomendación es seleccionar el threshold en función de objetivos de negocio:
  - Maximizar F1 si se busca equilibrio entre precisión y recall.
  - Minimizar coste total de errores ponderando **FP** (contactos innecesarios) vs **FN** (clientes elegibles no migrados).
  - Ajustar threshold por segmento (p. ej., heavy data users) si la sensibilidad deseada varía.
- El campo `threshold` en `configs/config.yaml` actúa como **hook** de configuración para futuras extensiones (p. ej., scripts de scoring offline que apliquen un umbral distinto de 0.5 según curvas precision–recall analizadas en notebooks).

## Próximos Pasos
- HPO con Optuna (espacio sobre `C`, `penalty`).
- Tracking con MLflow/W&B (opcional ya presente en `requirements.txt`).
- Monitoreo de drift y reentrenos programados.

## Créditos y Referencias
- Autor: Daniel Duque.
- Librerías: scikit-learn, FastAPI, matplotlib, seaborn.

---

## Preguntas Frecuentes (FAQ)
- ¿Por qué Regresión Logística? Interpretabilidad, rapidez y baseline competitivo en tabulares lineales.
- ¿Cómo manejo desbalanceo? `class_weight=balanced` y evaluación con ROC-AUC/F1; ajustar `threshold` si aplica.
- ¿Cómo replico todo? `make install && make train && make eval`.
- ¿Cómo sirvo el modelo? `make serve` (local) o `docker compose up`.
- ¿Cómo cambio features/modelo? Editar `configs/config.yaml` y `data/preprocess.py`.

## Resumen Ejecutivo para Portafolio
Clasificador de recomendación de plan móvil (Ultra vs Smart) con pipeline reproducible, API de inferencia, Docker y tests. Resultados y artefactos auditables en `artifacts/`. Preparado para iteración con HPO y monitoreo.
