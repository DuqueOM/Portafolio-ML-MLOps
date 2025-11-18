# PR — CarVision Market Intelligence (Mejoras de explicabilidad, validación temporal y drift)

## Scope del cambio

- Añadido notebook `notebooks/explainability_shap.ipynb` para análisis SHAP global/local del modelo de pricing.
- Extendida la evaluación en `evaluate.py` con backtesting temporal basado en `model_year` y generación de:
  - `artifacts/metrics_temporal.json` (métricas en tramo reciente).
  - `artifacts/error_by_segment.csv` (errores por segmento: condition, type, model_year).
- Actualizados `README.md`, `model_card.md` y `data_card.md` para reflejar interpretabilidad, backtesting temporal y supuestos de datos.
- Refuerzo de `monitoring/check_drift.py` para producir un resumen con umbrales simples (PSI/KS) y una recomendación de reentreno.

## Cómo correr la demo (revisor)

### 1) Entrenamiento y evaluación base

```bash
cd CarVision-Market-Intelligence
make install         # o pip install -r requirements.txt
make train           # entrena el modelo y guarda artifacts/model.joblib
make eval            # ejecuta evaluate.py y genera artifacts/metrics*.json
```

Artefactos relevantes tras `make eval`:
- `artifacts/metrics.json` — métricas del modelo en test aleatorio.
- `artifacts/metrics_baseline.json` — baseline DummyRegressor.
- `artifacts/metrics_bootstrap.json` (si bootstrap activado en config).
- `artifacts/metrics_temporal.json` — métricas en backtest temporal (segmento reciente por `model_year`).
- `artifacts/error_by_segment.csv` — errores por segmento (condition/type/model_year) en el tramo temporal.

### 2) Dashboard Streamlit

```bash
make start-demo      # instala deps, entrena y lanza Streamlit
# Abrir en navegador
# http://localhost:8501
```

Esperado en la demo:
- Dashboard con vistas de pricing y distribución de vehículos.
- Posibilidad de explorar segmentos de inventario y precios.

### 3) API FastAPI de inferencia

```bash
make install
make train
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000

# Healthcheck
curl -s http://localhost:8000/health | jq

# Predicción de ejemplo
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @example_payload.json | jq
```

### 4) Interpretabilidad SHAP (opcional)

Abrir el notebook:

- `notebooks/explainability_shap.ipynb`

Ejecutar tras `make train` para:
- Ver un summary plot global de importancia de features.
- Ver ejemplos locales (force plot) para vehículos concretos.

### 5) Drift check y recomendación de reentreno

```bash
python monitoring/check_drift.py \
  --ref vehicles_us.csv \
  --cur vehicles_us.csv \
  --features price model_year odometer \
  --out artifacts/drift_report.json
```

El JSON `artifacts/drift_report.json` incluye:
- KS y PSI por feature.
- Campo `summary.recommend_retrain` con una recomendación simple basada en umbrales (PSI y p-value KS).

---

Este PR no modifica la API pública del proyecto; añade capacidad analítica para explicabilidad, evaluación temporal y monitoreo de drift sin romper flujos existentes (`make train|eval`, Streamlit y FastAPI).
