# Data Science Portfolio — Production-Ready Demos (Landing)

## 1. Resumen Ejecutivo

Portafolio con 7 proyectos listos para demo técnica y conversación de MLOps. Cada proyecto incluye:

- **Pipeline reproducible** vía `Makefile` o CLI (`main.py` con `--mode` y `--config`).
- **API de inferencia** (FastAPI/Streamlit) cuando aplica.
- **Artefactos versionados**: modelos (`model_v1.0.0.pkl` o `model.joblib`), métricas JSON y notebooks de demo.
- **Monitoreo básico**: scripts de drift (KS/PSI) y, en proyectos customer-facing, integración opcional con Evidently.
- **Soporte para MLflow** para tracking de runs en modo `file:./mlruns`.
- **Dockerfile** y, en varios casos, `docker-compose.yml` con `HEALTHCHECK`.

Esta landing sintetiza el valor de cada proyecto, el stack técnico y qué parte del trabajo fue realizada específicamente por el autor.

---

## 2. Resumen por Proyecto

| Proyecto | Dominio | Valor principal | Stack clave | Demo rápida | Rol objetivo |
|---------|---------|-----------------|------------|------------|--------------|
| **BankChurn-Predictor** | Churn bancario | Clasificador con manejo de desbalance, fairness y explicación SHAP | Python, scikit-learn, FastAPI, MLflow, DVC | `make install && make train && make api-start` | Senior Data Scientist — Customer Intelligence / MLOps-aware |
| **CarVision-Market-Intelligence** | Pricing autos usados | Modelo de precio + dashboard Streamlit + API | Python, scikit-learn, FastAPI, Streamlit, Optuna, MLflow | `make start-demo` | Senior Data Scientist — Pricing & Product Analytics |
| **GoldRecovery-Process-Optimizer** | Minería / metalurgia | Predicción de recuperación y simulación de escenarios | Python, XGBoost/LightGBM/RF, FastAPI, MLflow | `make install && make train && python -m app.example_load` | ML/DS Engineer — Industrial Analytics / Process Optimization |
| **Chicago-Mobility-Analytics** | Movilidad urbana | Predicción de duración de viajes + demo de streaming y backtest spatiotemporal | Python, scikit-learn, geospatial libs, OR-Tools (streaming demo) | `make install && make train && python -m app.example_load` | Data Scientist — Mobility / Time Series |
| **Gaming-Market-Intelligence** | Mercado gaming | Clasificador de éxito + notebooks de retención y ROI | Python, scikit-learn, statsmodels, notebooks causal/ROI | `make install-deps && make train && python -m app.example_load` | Applied Data Scientist — Gaming / Market Analytics |
| **OilWell-Location-Optimizer** | Energía / upstream | Selección de pozos con riesgo (bootstrap/VaR) y restricciones realistas | Python, scikit-learn, NumPy, Monte Carlo, PuLP | `make start-demo` | Data Scientist — Quant Risk / Energy |
| **TelecomAI-Customer-Intelligence** | Telecom | Clasificador Ultra vs Smart con API, Docker y drift | Python, scikit-learn, FastAPI, Docker, MLflow, Evidently (opcional) | `make start-demo` | ML Engineer — Customer Analytics / Telco |

## 2.1 Comparativa técnica (modelo, métricas, nivel producción)

| Proyecto | Problema | Modelo principal | Métricas clave (v1) | Stack ML/MLOps | Nivel de producción |
|----------|----------|------------------|----------------------|-----------------|---------------------|
| **BankChurn-Predictor** | Predicción de churn bancario (`Exited`) | Ensemble Voting (LogReg + RandomForest, con resampling + calibración) | F1, ROC-AUC, precision, recall, accuracy | scikit-learn, Optuna (hyperopt), MLflow, DVC, FastAPI | CLI + tests (incl. fairness) + API + Docker + CI + model/data cards |
| **CarVision-Market-Intelligence** | Pricing de vehículos usados (`price`) | RandomForestRegressor en `Pipeline` sklearn | RMSE, MAE, MAPE, R² | scikit-learn, Optuna (HPO script), MLflow, FastAPI, Streamlit | CLI train/eval/predict + dashboard + API + tests + Docker + CI |
| **GoldRecovery-Process-Optimizer** | Predicción de `final.output.recovery` (0–100%) | Ensemble XGBoost + LightGBM + RandomForest | sMAPE, MAE, RMSE, R² | XGBoost, LightGBM, scikit-learn, FastAPI, MLflow | CLI + tests (incl. fairness smoke) + API + Docker + CI + model/data cards |
| **Chicago-Mobility-Analytics** | Duración de viajes urbanos (`duration_seconds`) | RandomForestRegressor | MAE, RMSE, R² | scikit-learn, geospatial tooling (DVC assets), FastAPI | CLI train/eval/predict + API + tests (incl. fairness por clima) + CI |
| **Gaming-Market-Intelligence** | Éxito comercial de videojuegos (`is_successful`) | RandomForestClassifier en `Pipeline` | F1, accuracy, ROC-AUC, PR-AUC | scikit-learn, statsmodels, MLflow (demo), notebooks de negocio | CLI train/eval/predict + tests (incl. fairness por plataforma) + API demo + CI |
| **OilWell-Location-Optimizer** | Selección de pozos petroleros por región (`product`) | LinearRegression por región + bootstrap de riesgo | RMSE regional, métricas de utilidad esperada, pérdida prob. | scikit-learn, NumPy, bootstrap personalizado, FastAPI, MLflow | CLI train/eval/predict + API + tests (incl. API E2E) + Docker + CI |
| **TelecomAI-Customer-Intelligence** | Recomendación de plan móvil (`is_ultra`) | LogisticRegression en `Pipeline` | F1, ROC-AUC, accuracy, precision, recall | scikit-learn, MLflow, FastAPI, Evidently (drift demo) | CLI train/eval/predict + tests (incl. API contrato) + API + Docker + CI + model/data cards |

---

## 3. Stack Técnico Global

- **Lenguaje**: Python 3.8–3.11.
- **ML / Estadística**: scikit-learn, XGBoost, statsmodels, RandomForest, regresión logística.
- **MLOps / Tracking**: MLflow (modo local `file:./mlruns`), DVC (BankChurn, Chicago geospatial assets).
- **APIs y Frontends**: FastAPI, Streamlit.
- **Monitoreo de datos**: scripts KS/PSI, integración opcional con Evidently.
- **Optimización**: PuLP/CVXPY (OilWell y Chicago streaming demo), OR-Tools cuando aplica.
- **Infraestructura**: Docker, `docker-compose`, GitHub Actions (`.github/workflows/ci.yml`).

---

## 4. Ownership (¿Qué hizo el autor?)

Por proyecto, el trabajo realizado incluye (además del desarrollo original del curso):

- **BankChurn-Predictor**
  - Integración de MLflow demo (`make mlflow-demo`).
  - Monitoreo de drift (`monitoring/check_drift.py`).
  - API FastAPI y tests de preprocesamiento.
  - `model_card.md` y `data_card.md` con documentación estructurada.

- **CarVision-Market-Intelligence**
  - API FastAPI de inferencia y demo de carga de modelo (`app/example_load.py`).
  - Backtesting temporal y espacial (en rama de proyecto) y script de HPO con Optuna.
  - Makefile con comandos `train`, `eval`, `serve` y demo integrada.
  - `model_card.md` y `data_card.md` con supuestos y limitaciones.

- **GoldRecovery-Process-Optimizer**
  - Pipeline reproducible de entrenamiento/evaluación y demo de carga del modelo.
  - Scripts de simulación (`scripts/recovery_simulation.py`) y MLflow demo.
  - Monitoreo de drift y documentación de modelo en `model_card.md` / `data_card.md`.

- **Chicago-Mobility-Analytics**
  - API/CLI de entrenamiento y ejemplo de inferencia.
  - Demo de streaming (asignación de conductores) y backtest spatiotemporal (en rama de proyecto).
  - DVC para assets geoespaciales.
  - `model_card.md` y `data_card.md` de duración de viajes.

- **Gaming-Market-Intelligence**
  - Pipeline de clasificación tabular y demo rápida.
  - Notebooks de retención (Kaplan–Meier) y ROI de portafolio.
  - Monitoreo de drift y documentación en `model_card.md` / `data_card.md`.

- **OilWell-Location-Optimizer**
  - Scripts de optimización de pozos con constraints de CAPEX/permits (en rama de proyecto).
  - Monte Carlo VaR de portafolio y documentación de supuestos en `docs/assumptions.md`.
  - `model_card.md` y `data_card.md` alineados con el riesgo financiero.

- **TelecomAI-Customer-Intelligence**
  - CLI reproducible, API FastAPI + Docker, tests de contrato (`tests/`).
  - Monitoreo de drift mediante script KS/PSI y documentación de política de recomendación (en rama de proyecto).
  - `model_card.md` y `data_card.md` para dataset y modelo.

---

## 5. Calidad, CI y Smoke Tests

- Workflow CI (`.github/workflows/ci.yml`):
  - Matriz sobre los 7 subproyectos.
  - Para cada uno: instala dependencias, corre `pytest --cov=. --cov-report=term-missing`, `mypy .` y `flake8 .`.
  - Para **BankChurn**: paso adicional de smoke-train (`SMOKE=1`) para validar el pipeline de entrenamiento end-to-end.

Esto hace que cada PR tenga una verificación mínima de:

- Estilo (flake8).
- Tests unitarios de datos/modelo (pytest con cobertura).
- Type-checking (mypy) por proyecto.
- Smoke-train en al menos un proyecto representativo.

---

## 6. Demos y Screenshots

- Cada proyecto incluye README con comandos de demo (`make start-demo`, `make api-start`, etc.).
- Para enriquecer el portafolio visualmente se pueden añadir más adelante:
  - GIFs de dashboards (Streamlit, etc.).
  - Capturas de los endpoints en uso (FastAPI docs, curl + jq).

---

## 7. Cómo Navegar este Monorepo

- Ver **README.md** en la raíz para una vista rápida y comandos de demo.
- Ver cada subdirectorio de proyecto para detalles y documentación técnica.
- Esta `portfolio_landing` sirve como índice central para reclutadores y revisores técnicos.
