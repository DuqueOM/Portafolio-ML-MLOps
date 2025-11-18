# README_PR — BankChurn-Predictor

Pequeño README para acompañar cada PR en este subproyecto.

## 1. Resumen del PR

- _[Completar]_ Describir en 2–3 bullets el objetivo principal del cambio
  (p.ej. refactor de pipeline, nuevos tests, mejora de métricas, fixes de API).

## 2. Cambios clave

- _[Completar]_ Lista breve de cambios técnicos relevantes:
  - [ ] Código (nuevas funciones/módulos)
  - [ ] Configuración (`configs/config.yaml`, Makefile, CI)
  - [ ] Tests (`tests/`)
  - [ ] Docs (README, model_card, data_card, etc.)

## 3. Cómo ejecutar la demo de este PR

Desde `BankChurn-Predictor/`:

```bash
make install
make train
make api-start
# Opcional: smoke scripts, mlflow, drift
# make mlflow-demo
# make check-drift
```

Si cambiaste la CLI directamente:

```bash
python main.py --mode train --config configs/config.yaml --input data/raw/Churn.csv
python main.py --mode eval  --config configs/config.yaml --input data/raw/Churn.csv
python main.py --mode predict --config configs/config.yaml --input data/new_customers.csv --output predictions.csv
```

## 4. Artefactos / métricas relevantes

- _[Completar]_ enlaces o paths a:
  - Artefactos de modelo (`models/`, `results/`)
  - Métricas (`results/training_results.json`, `artifacts/` adicionales)
  - Capturas/GIFs si aplica (FastAPI docs, dashboards, etc.)

## 5. Notas de backward compatibility / riesgos

- _[Completar]_ ¿Rompe algo en la CLI, API o contratos de datos?
- Señalar cualquier migración manual necesaria.
