# Model Card — Chicago Mobility Duration Model

## 1. Overview

- **Task**: Regression — predict trip duration (seconds) for Saturday taxi rides in Chicago.
- **Model**: RandomForestRegressor (scikit-learn).
- **Inputs**:
  - `start_ts` (timestamp of trip start).
  - `weather_conditions` (categorical: `Good` or `Bad`).
- **Output**:
  - `duration_seconds` (expected trip duration in seconds).

## 2. Intended Use

- **Primary use**: Support operations and planning for urban mobility (e.g. expected travel time under different weather conditions).
- **Intended users**: Data scientists, operations analysts, and backend services consuming the FastAPI endpoint.
- **Out-of-scope**: Real-time safety-critical decisions, legal or contractual guarantees of arrival time.

## 3. Training & Evaluation Data

- **Source**: Educational subset of Chicago taxi trip data (Saturday samples).
- **Main file**: `moved_project_sql_result_07.csv`.
- **Variables**:
  - `start_ts` — start time of the trip.
  - `weather_conditions` — `Good` or `Bad`.
  - `duration_seconds` — observed trip duration.
- **Splits**:
  - Train / Validation / Test defined in `configs/default.yaml`.

## 4. Preprocessing

- Parse `start_ts` to datetime.
- Derive features: `hour`, `day_of_week`, `is_weekend`, `weather_is_bad`.
- Remove rows with non-positive `duration_seconds`.

## 5. Metrics

Typical metrics on the test set (example values; recompute with `python main.py --mode train` and `python evaluate.py`):

- MAE (Mean Absolute Error) in seconds.
- RMSE (Root Mean Squared Error) in seconds.
- R² coefficient of determination.

Baselines used for comparison:

- Global mean duration.
- Mean duration by `weather_conditions`.

## 6. Ethical Considerations & Risks

- **Bias and fairness**:
  - The dataset covers a specific city and time window (Saturdays), possibly missing patterns from weekdays or other areas.
  - If extended with spatial features, the model could inadvertently encode geographic or socio-economic biases.
- **Misuse risks**:
  - Using predicted durations as hard guarantees to customers without reflecting uncertainty.
  - Using the model to justify unequal service quality across neighborhoods.
- **Mitigations**:
  - Communicate predictions as estimates with error bounds.
  - Periodically evaluate performance across subgroups (time ranges, weather conditions, neighborhoods if added).

## 7. Limitations

- Limited feature set (no detailed route, traffic or event data).
- Small dataset → risk of overfitting and unstable estimates if used beyond the training distribution.
- Designed for Saturday trips; generalisation to other days is not guaranteed.

### Current Scope Limitations (v1)
- El modelo actual sólo aborda **predicción de duración de viajes** en función de tiempo/clima; no implementa forecasting de demanda multi-zona, optimización de rutas ni algoritmos de reinforcement learning descritos en el README como roadmap.
- Las métricas de ahorro operativo e impacto urbano mencionadas a nivel de plataforma son ejemplos ilustrativos; cualquier despliegue real requeriría modelos adicionales (demanda, asignación de flota, pricing) y validación con datos operativos.

## 8. Privacy & Data Governance

- El subset usado en este proyecto se basa en datos abiertos anonimizados; no incluye identificadores directos de pasajeros.
- En un despliegue real, se deben anonimizar ubicaciones de alta resolución y aplicar agregaciones espaciales/temporales para reducir riesgo de reidentificación.
- Evitar el logging de coordenadas crudas y timestamps exactos en servicios de inferencia; preferir métricas agregadas y ventanas de tiempo.

## 9. Maintenance & Monitoring

For production use, recommended practices include:

- Monitoring input data distributions (weather patterns, demand levels).
- Tracking prediction error over time and retraining if it degrades.
- Versioning datasets and models (e.g. with DVC, MLflow, or similar).

## 10. Contact

- **Author**: Daniel Duque
- **Use case**: Portfolio project — Chicago Mobility Analytics.
