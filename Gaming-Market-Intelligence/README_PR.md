# PR — Gaming Market Intelligence (export estadístico, KPIs de inversión y API)

## Scope del cambio

- Nuevo script `scripts/export_stats.py` que ejecuta pruebas de hipótesis sobre el dataset (`games.csv`) y genera `artifacts/hypothesis_tests_summary.json`.
- Nuevo módulo `evaluate_business.py` que calcula KPIs de inversión a partir de las predicciones del clasificador (reducción de fallos de inversión, tasas de éxito/fallo y fallos evitados).
- Actualización de `data_card.md` y `model_card.md` para detallar riesgos éticos de sesgo por **plataforma**, **región** y **género (de juego)**.
- Test adicional en `tests/test_model.py` para validar la coherencia básica de los KPIs de negocio.

## Cómo correr la demo (revisor)

### 1) Entrenamiento del modelo

```bash
cd Gaming-Market-Intelligence
make install-deps
make train              # entrena el modelo y guarda artifacts/model/model.joblib
```

Artefactos relevantes tras el entrenamiento:
- `artifacts/model/model.joblib` — pipeline entrenado.
- `artifacts/metrics/metrics_eval.json` (después de `make eval`, ver abajo).

### 2) Export de estadísticas de hipótesis

```bash
python scripts/export_stats.py
```

Esto genera:
- `artifacts/hypothesis_tests_summary.json` con resumen de:
  - Diferencias de ventas entre plataformas (ANOVA).
  - Diferencias por género de juego (ANOVA).
  - Correlación critic_score vs user_score.
  - Tendencia temporal de ventas medias por año (Spearman).

### 3) KPIs de inversión (evaluate_business)

```bash
python evaluate_business.py
```

Esto:
- Carga el modelo entrenado desde `artifacts/model/model.joblib`.
- Evalúa en el conjunto de test y calcula KPIs de inversión:
  - `total_projects`, `invested_all`, `invested_model`.
  - `failure_rate_all` vs `failure_rate_model`.
  - `relative_failure_reduction` y `estimated_failures_avoided`.
- Escribe los resultados en `artifacts/metrics/metrics_business.json`.

### 4) Evaluación estándar y API /predict

```bash
# Evaluación técnica (métricas habituales)
make eval      # llama a evaluate.py y guarda artifacts/metrics/metrics_eval.json

# Levantar API de inferencia
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000

# Healthcheck
curl -s http://localhost:8000/health | jq

# Llamada a /predict (ejemplo)
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"platform":"PS4","year_of_release":2015,"genre":"Action","critic_score":82,"user_score":8.1,"rating":"M"}' | jq
```

### 5) Notas para el revisor

- `scripts/export_stats.py` y `evaluate_business.py` no cambian la API pública; añaden visibilidad cuantitativa a:
  - cómo se distribuye el éxito por plataforma/género/año,
  - y cómo el clasificador puede reducir fallos de inversión frente a la estrategia de “invertir en todos los títulos”.
- Los tests en `tests/test_model.py` validan que los KPIs calculados son coherentes (tasas en [0,1], fallos evitados ≥ 0, etc.).
- La demo completa combina:
  1. Entrenamiento (`make train`).
  2. Export estadístico (`python scripts/export_stats.py`).
  3. KPIs de inversión (`python evaluate_business.py`).
  4. Inferencia online vía `/predict` para casos individuales.
