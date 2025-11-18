# PR — GoldRecovery Process Optimizer (preprocesamiento claro, drift thresholds, HPO)

## Scope del cambio

- Actualización de `README.md` para describir fielmente el preprocesamiento real de `data/preprocess.py` (cálculo opcional de recovery, filtros de rango, drop de columnas muy nulas, features derivadas y temporales, imputación por mediana).
- Model card extendida con un ejemplo ilustrativo de impacto operativo (toneladas recuperadas / menor variabilidad) ligado a mejoras en sMAPE/MAE.
- `monitoring/check_drift.py` ahora genera un resumen con `max_psi`, `max_ks` y una `recommended_action` (`ok`, `review`, `retrain`) según umbrales simples; `README.md` documenta estos thresholds y acciones sugirió.
- Nuevo script `scripts/run_optuna.py` para HPO de un modelo XGBoost usando Optuna, guardando resultados en `results/optuna_xgb_results.json`.

## Cómo correr la demo (revisor)

### 1) Entrenamiento y evaluación base

```bash
cd GoldRecovery-Process-Optimizer
make install          # instala dependencias (incluye optuna de forma opcional)
make train            # entrena ensemble XGB/LGBM/RF y guarda modelos en models/
make eval             # evalúa en test y genera métricas en results/
```

Artefactos relevantes:
- `models/metallurgical_model.pkl` — modelo/ensemble principal para FastAPI.
- `results/metrics.json` — métricas de evaluación (sMAPE, MAE, etc.).

### 2) Ejemplo de carga de modelo

```bash
python -m app.example_load
```

- Verifica que el paquete de modelo se carga correctamente y que se puede ejecutar una predicción simple.

### 3) API FastAPI y /predict

```bash
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000

# Healthcheck
curl -s http://localhost:8000/health | jq

# Predicción de ejemplo
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "instances": [
      {
        "features": {
          "rougher.output.concentrate_au": 2.1,
          "primary_cleaner.output.concentrate_au": 3.4,
          "rougher.output.concentrate_ag": 1.2,
          "primary_cleaner.output.concentrate_ag": 2.0
        }
      }
    ]
  }' | jq
```

Outputs a validar:
- Respuesta JSON con `prediction` o estructura equivalente según implementación de `fastapi_app.py`.
- `status_code` 200 y `health` ok.

### 4) Drift check con thresholds documentados

```bash
python monitoring/check_drift.py \
  --ref gold_recovery_train.csv \
  --cur gold_recovery_test.csv \
  --cols final.output.recovery \
  --out-json results/drift.json
```

Revisar `results/drift.json`:
- `drift` por columna (KS, PSI).
- `summary.max_psi`, `summary.max_ks`.
- `summary.recommended_action` (`ok`, `review`, `retrain`) según thresholds descritos en README.

### 5) HPO con Optuna (avanzado, opcional)

```bash
python scripts/run_optuna.py
```

Esto ejecuta una búsqueda de hiperparámetros para un modelo XGBoost sencillo y guarda:
- `results/optuna_xgb_results.json` con `best_value` (RMSE) y `best_params`.

---

Este PR no rompe la API ni el flujo de entrenamiento/evaluación existentes; mejora la transparencia del preprocesamiento, hace explícitos los thresholds de drift y añade un script de HPO opcional para exploración de hiperparámetros.
