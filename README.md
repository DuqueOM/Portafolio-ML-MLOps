# Data Science Portfolio — Production-Ready Demos

## TL;DR

Portafolio con 7 proyectos listos para demo y reproducibilidad. Cada proyecto incluye: Makefile, API FastAPI/Streamlit (cuando aplica), export de modelo `model_v1.0.0.pkl`, notebook de demo, script de MLflow, monitoreo de drift y Dockerfile con `HEALTHCHECK`. La CI, mediante el workflow global `.github/workflows/ci.yml`, corre `pytest`, `mypy` y `flake8` por proyecto.

Para una vista más detallada (stack, ownership, CI y comandos), ver `docs/portfolio_landing.md`.

## Proyectos

- **BankChurn Predictor [Python 3.8+]** — Clasificación de churn, ensemble calibrado, fairness tests.
  - Problema: priorizar clientes bancarios en riesgo de churn.
  - Demo 3 pasos: `make install && make train && make api-start`.
  - Stack: Python, scikit-learn, FastAPI, DVC, MLflow, Evidently (opcional).
  - Rol objetivo: Senior Data Scientist — Customer Intelligence / MLOps-aware.
- **CarVision Market Intelligence [Python 3.10+]** — Pricing de vehículos, dashboard Streamlit, backtesting temporal.
  - Problema: estimar precio de vehículos usados y explorar mercado.
  - Demo 3 pasos: `make start-demo` (instala, entrena y lanza dashboard/API).
  - Stack: Python, scikit-learn, FastAPI, Streamlit, MLflow.
  - Rol objetivo: Senior Data Scientist — Pricing & Product Analytics.
- **GoldRecovery Process Optimizer [Python 3.11]** — Recuperación metalúrgica, ensemble y simulación de escenarios.
  - Problema: predecir y optimizar `final.output.recovery` en planta metalúrgica.
  - Demo 3 pasos: `make install && make train && python -m app.example_load`.
  - Stack: Python, XGBoost, LightGBM, RandomForest, FastAPI, Streamlit, MLflow.
  - Rol objetivo: ML/DS Engineer — Industrial Analytics / Process Optimization.
- **Chicago Mobility Analytics [Python 3.10+]** — Predicción de duración de viajes, geospatial sample (GeoParquet/PostGIS).
  - Problema: estimar duración de viajes de taxi usando tiempo y clima.
  - Demo 3 pasos: `pip install -r requirements-core.txt && python main.py --mode train --config configs/default.yaml && python -m app.example_load`.
  - Stack: Python, scikit-learn, FastAPI, geospatial tooling (GeoPandas/Parquet), MLflow (demo).
  - Rol objetivo: Data Scientist — Mobility / Time Series.
- **Gaming Market Intelligence [Python 3.10+]** — Éxito de videojuegos, notebooks de retención (Kaplan–Meier) y LTV.
  - Problema: estimar probabilidad de éxito comercial de un videojuego antes del lanzamiento.
  - Demo 3 pasos: `pip install -r requirements-core.txt && python main.py --mode train --config configs/config.yaml && python -m app.example_load`.
  - Stack: Python, scikit-learn, statsmodels, FastAPI, Streamlit, MLflow (demo).
  - Rol objetivo: Applied Data Scientist — Gaming / Market Analytics.
- **OilWell Location Optimizer [Python 3.10+]** — Selección de pozos por región, bootstrap risk y sensibilidad de escenarios.
  - Problema: seleccionar región/subconjunto de pozos maximizando retorno ajustado por riesgo.
  - Demo 3 pasos: `make install && make train && make api`.
  - Stack: Python, scikit-learn, bootstrap personalizado, FastAPI, MLflow.
  - Rol objetivo: Data Scientist — Quant Risk / Energy.
- **TelecomAI Customer Intelligence [Python 3.10+]** — Predicción de churn telco con MLflow y monitoreo Evidently.
  - Problema: recomendar plan Ultra vs Smart en función del uso de cliente.
  - Demo 3 pasos: `make install && make start-demo && curl -s http://localhost:8000/health | jq`.
  - Stack: Python, scikit-learn, FastAPI, Docker, MLflow, Evidently.
  - Rol objetivo: ML Engineer — Customer Analytics / Telco.

## Cómo ejecutar demos localmente

Ejecutar desde el directorio del proyecto respectivo.

- BankChurn-Predictor
  - `make install && make train && make api-start`
  - MLflow: `make mlflow-demo`
  - Drift: `make check-drift`
- CarVision-Market-Intelligence
  - `make start-demo` (entrena y lanza Streamlit/API)
  - MLflow: `make mlflow-demo`
- GoldRecovery-Process-Optimizer
  - `make start-demo`
  - MLflow: `make mlflow-demo`
  - Simulación: `python scripts/recovery_simulation.py --model models/model_v1.0.0.pkl --csv gold_recovery_test.csv`
- Chicago-Mobility-Analytics
  - `make install && make train && python -m app.example_load`
  - MLflow: `python scripts/run_mlflow.py`
- Gaming-Market-Intelligence
  - `make start-demo`
  - Retención: abrir `notebooks/retention_survival.ipynb`
- OilWell-Location-Optimizer
  - `make start-demo`
  - Optimización: `python scripts/optimize_selection.py --csv geo_data_1.csv --n-select 200`
- TelecomAI-Customer-Intelligence
  - `make start-demo`
  - MLflow: `make mlflow-demo`

## CLI unificada (por proyecto)

Los comandos base de CLI siguen el patrón:

```bash
python main.py --mode train   --config configs/config.yaml
python main.py --mode eval    --config configs/config.yaml
python main.py --mode predict --config configs/config.yaml [...]
```

Ejemplos por proyecto (ejecutar dentro de cada carpeta):

- BankChurn-Predictor  
  - `python main.py --mode train --config configs/config.yaml --input data/raw/Churn.csv`
  - `python main.py --mode eval --config configs/config.yaml --input data/raw/Churn.csv`
  - `python main.py --mode predict --config configs/config.yaml --input data/new_customers.csv --output predictions.csv`
- CarVision-Market-Intelligence  
  - `python main.py --mode train --config configs/config.yaml`
  - `python main.py --mode eval --config configs/config.yaml`
  - `python main.py --mode predict --config configs/config.yaml --input_json example_payload.json`
- GoldRecovery-Process-Optimizer  
  - `python main.py --mode train --config configs/config.yaml --input gold_recovery_train.csv`
  - `python main.py --mode eval --config configs/config.yaml --input gold_recovery_test.csv`
  - `python main.py --mode predict --config configs/config.yaml --input gold_recovery_test.csv --output results/predictions.csv`
- Chicago-Mobility-Analytics  
  - `python main.py --mode train --config configs/default.yaml --seed 42`
  - `python main.py --mode eval --config configs/default.yaml --seed 42`
  - `python main.py --mode predict --config configs/default.yaml --start_ts "2017-11-11 10:00:00" --weather_conditions Good`
- Gaming-Market-Intelligence  
  - `python main.py --mode train --config configs/config.yaml`
  - `python main.py --mode eval --config configs/config.yaml`
  - `python main.py --mode predict --config configs/config.yaml --payload '{"platform":"PS4","genre":"Action","year_of_release":2015,"critic_score":85,"user_score":8.2,"rating":"M"}'`
- OilWell-Location-Optimizer  
  - `python main.py --mode train --config configs/default.yaml`
  - `python main.py --mode eval --config configs/default.yaml`
  - `python main.py --mode predict --config configs/default.yaml --region 1 --payload '{"records":[{"f0":1.0,"f1":-2.0,"f2":3.0}]}'`
- TelecomAI-Customer-Intelligence  
  - `python main.py --mode train --config configs/config.yaml`
  - `python main.py --mode eval --config configs/config.yaml`
  - `python main.py --mode predict --config configs/config.yaml --input_csv users_behavior.csv --output_path artifacts/preds.csv`

## Roadmap / Limitaciones / Datos sintéticos vs reales

- Datos mayormente sintéticos o educativos; no representan producción real.
- Monitoreo de drift (KS/PSI) incluido; Evidently es opcional.
- Registrar modelos en MLflow Registry requiere backend soportado (file-store no registra).
- DVC incluido en BankChurn para pipeline reproducible.

## Contacto / Licencia

- Autor: Daniel Duque — MIT License
- Contacto: ver cada README del proyecto
