# Arquitectura — BankChurn Predictor

## Visión general

- **Objetivo:** predecir la probabilidad de churn (`Exited`) y exponer el modelo vía CLI y API REST.
- **Tipo de problema:** clasificación binaria en datos tabulares con clases desbalanceadas.
- **Componentes clave:**
  - `main.py` — CLI para train/evaluate/predict/hyperopt.
  - `BankChurnPredictor` — clase principal de entrenamiento/evaluación.
  - `ResampleClassifier` — estimador custom para oversampling/undersampling.
  - `app/fastapi_app.py` — API FastAPI de inferencia.
  - `monitoring/check_drift.py` — chequeos de drift.
  - `scripts/run_mlflow.py` — logging de métricas/artefactos en MLflow.

## Flujo de datos

1. **Entrada:** CSV `Churn.csv` en formato Beta Bank.
2. **Carga y validación:** `BankChurnPredictor.load_data()` valida presencia de `Exited` y reporta distribución de clases.
3. **Preprocesamiento:** `BankChurnPredictor.preprocess_data()`
   - Elimina columnas irrelevantes (`RowNumber`, `CustomerId`, `Surname`).
   - Separa features/target.
   - Construye `ColumnTransformer` con:
     - numéricas: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` → `StandardScaler`.
     - categóricas: `Geography`, `Gender` → `OneHotEncoder(drop='first', sparse_output=False)`.
4. **Split:** `train_test_split` estratificado según `training.test_size` en `configs/config.yaml`.
5. **Entrenamiento:** `BankChurnPredictor.train()` ejecuta CV estratificada (5 folds) y devuelve métricas agregadas.
6. **Evaluación:** `BankChurnPredictor.evaluate()` calcula métricas de test y objetos auxiliares (matriz de confusión, classification report).
7. **Persistencia:**
   - `models/best_model.pkl` y `preprocessor_*.pkl` vía `save_model()`.
   - `results/training_results.json` con `cv_results` y `test_results`.
   - Paquete combinado `models/model_v1.0.0.pkl` para demos/serving.

## Arquitectura de modelo

### Preprocesamiento

- `ColumnTransformer`:
  - Pipeline numérico: `StandardScaler`.
  - Pipeline categórico: `OneHotEncoder(drop='first', handle_unknown="ignore")`.
- Preprocesamiento se ajusta en train y se reutiliza en evaluate/predict vía artefactos.

### Modelo base

- Ensemble principal en `BankChurnPredictor.create_model()`:
  - `LogisticRegression` con `class_weight="balanced"`.
  - `RandomForestClassifier` con `class_weight="balanced_subsample"`.
  - Ambos envueltos en `VotingClassifier(voting="soft", weights=[0.4, 0.6])`.
- Opcionalmente envueltos en `ResampleClassifier` para oversampling/undersampling.
- Calibración Platt (`CalibratedClassifierCV(method="sigmoid", cv="prefit")`) sobre el modelo final.

### Manejo de clases desbalanceadas

- Estrategias combinadas:
  - `class_weight` en modelos base.
  - `ResampleClassifier` con `strategy` configurable (`oversample`, `undersample`, `none`).
  - Métricas centradas en F1/ROC-AUC y uso de CV estratificada.

## API de inferencia

- `app/fastapi_app.py` expone endpoints:
  - `GET /health` — health check básico.
  - `POST /predict` — predicción individual.
  - `POST /predict_batch` — predicciones en lote.
  - `GET /metrics`, `GET /model_info` — metadatos de modelo.
- Carga el modelo vía `BankChurnPredictor` y mantiene métricas simples de uso (conteo de requests, tiempos).

## MLOps y monitoreo

- **Tracking:** `scripts/run_mlflow.py` lee `results/training_results.json` y registra:
  - métricas de CV (`cv_*`), métricas de test (`test_*`), artefactos de resultados y config.
  - opcionalmente registra un `Pipeline` sklearn en el Model Registry (si el backend lo soporta).
- **Drift:** `monitoring/check_drift.py` (no detallado aquí) soporta KS/PSI y generación de reportes Evidently (si está instalado).
- **Versionado:**
  - Checkpoints y preprocesador versionados con timestamps.
  - `dvc.yaml` define un stage de entrenamiento que produce `models/` y `results/training_results.json`.

## Estructura de carpetas (vista lógica)

- `main.py` — entrypoint de CLI y orquestador de pipeline.
- `app/` — servicio de inferencia FastAPI.
- `configs/config.yaml` — configuración de datos/entrenamiento/modelo/evaluación/logging/deployment.
- `data/` — preprocesamiento y utilidades de datos.
- `scripts/` — integración con MLflow y utilidades.
- `monitoring/` — chequeos de drift.
- `tests/` — tests de datos, modelo, API y fairness.
- `docs/` — esta documentación de arquitectura y pipeline.
