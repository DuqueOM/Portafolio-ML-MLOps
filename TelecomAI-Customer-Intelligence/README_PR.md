# PR — TelecomAI Customer Intelligence (impacto de negocio, API robusta y campaña simulada)

## Scope del cambio

- Extensión de `README.md` y `model_card.md` con un ejemplo numérico de impacto de negocio y explicación de la lógica de threshold.
- Ampliación de `tests/test_api_e2e.py` con casos de payload inválido (campo faltante y tipo incorrecto) y asserts de códigos de error (422).
- Actualización de `data_card.md` con una sección de privacidad y ética de targeting (ausencia de PII, riesgos de upsell agresivo).
- Refinamiento de `scripts/run_mlflow.py` para registrar una métrica de negocio derivada: número aproximado de clientes elegibles correctamente clasificados.
- Creación de `notebooks/campaign_simulation.ipynb` para simular una campaña con KPIs (contactados, TP/FP, ROI ilustrativo) y cohorts simples por uso de datos.

## Cómo correr la demo (revisor)

### 1) CLI (train / eval / predict)

```bash
cd TelecomAI-Customer-Intelligence
make install

# Entrenar y generar artifacts (artifacts/model.joblib, metrics.json, etc.)
make train

# Evaluar modelo (recalcula métricas y gráficos en artifacts/)
make eval

# Predicción batch sobre users_behavior.csv
a python main.py --mode predict \
  --config configs/config.yaml \
  --input_csv users_behavior.csv \
  --output_path artifacts/preds.csv
```

### 2) API FastAPI

```bash
# Levantar API local (sin Docker)
make install
make train
make serve   # uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000

# Healthcheck
curl -s http://localhost:8000/health | jq

# Predicción de ejemplo
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"calls": 85, "minutes": 516.7, "messages": 56, "mb_used": 22696.96}' | jq
```

Opcionalmente, vía Docker:

```bash
docker compose up --build -d
# API disponible en http://localhost:8000
```

### 3) Simulación de campaña y KPIs

Abrir el notebook:

- `notebooks/campaign_simulation.ipynb`

Ejecutar después de `make train` para:
- Calcular métricas de campaña (TP, FP, precisión, recall, F1) con threshold=0.5.
- Estimar un ROI ilustrativo usando supuestos de ARPU adicional por migración y coste por contacto.
- Ver cohortes simples por uso de datos (`mb_used`) y cómo cambia la performance por segmento.

### 4) MLflow demo (opcional)

```bash
make mlflow-demo
```

- `scripts/run_mlflow.py`:
  - Lee `artifacts/metrics.json`.
  - Estima el número de clientes elegibles correctamente clasificados a partir del recall y `is_ultra` en `users_behavior.csv`.
  - Loguea métricas técnicas + métrica de negocio (`biz_*`) en el experimento `TelecomAI`.

---

Este PR no cambia la API pública, sólo añade:
- Mejor contexto de negocio (impacto numérico y threshold).
- Robustece tests de API.
- Añade trazabilidad de impacto (métricas de negocio en MLflow y notebook de campaña).
