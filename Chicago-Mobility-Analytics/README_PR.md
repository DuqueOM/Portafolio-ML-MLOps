# PR — Chicago-Mobility-Analytics (clarificación v1, impacto operativo y geospatial demo)

## Scope del cambio

- Reorganización de `README.md` para distinguir la **versión implementada (v1)** —modelo de duración de viajes + API FastAPI— del **roadmap avanzado** (forecasting de demanda, optimización de rutas, RL, IoT).
- Creación del notebook `notebooks/operational_impact.ipynb` para traducir las métricas de error del modelo (MAE/RMSE) a KPIs de tiempo de viaje/espera y ahorro diario ilustrativo.
- Creación de `notebooks/geospatial_demo.ipynb` como demo mínima de hotspots geoespaciales usando `moved_project_sql_result_07.csv` y Folium.
- Actualización de `model_card.md` para añadir una sección explícita de **limitaciones de alcance (v1)**.

## Cómo correr el flujo v1 completo (revisor)

### 1) Entrenamiento y ejemplo de carga

```bash
cd Chicago-Mobility-Analytics
make install
make train           # entrena RandomForestRegressor de duración y guarda models/duration_model.pkl
python -m app.example_load   # muestra una predicción de ejemplo usando el modelo exportado
```

Artefactos relevantes tras `make train`:
- `models/duration_model.pkl` — modelo de duración de viajes.
- `artifacts/metrics.json` — métricas de validación/test (MAE, RMSE, R²) generadas por el entrenamiento.

### 2) API FastAPI de duración de viajes

```bash
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000

# Healthcheck
curl -s http://localhost:8000/health | jq

# Predicción de ejemplo (duración esperada en segundos)
curl -s -X GET "http://localhost:8000/predict?start_ts=2017-11-11T10:00:00&weather_conditions=Good" | jq
```

Esperado:
- `/health` devuelve un JSON simple con `status` OK.
- `/predict` devuelve `{"duration_seconds": ...}` con una duración consistente con la hora y el clima.

### 3) Notebook de impacto operativo

Abrir:

- `notebooks/operational_impact.ipynb`

Ejecutar tras `make train` para:
- Leer `artifacts/metrics.json`.
- Interpretar el MAE/RMSE de test en términos de segundos/minutos por viaje.
- Estimar, con supuestos ilustrativos (viajes/día), el ahorro total de horas de viaje y una fracción de ahorro en tiempo de espera.

### 4) Demo geoespacial mínima

Abrir:

- `notebooks/geospatial_demo.ipynb`

Ejecutar para:
- Leer `moved_project_sql_result_07.csv`.
- Inferir columnas de lat/lon (o ajustarlas manualmente si es necesario).
- Renderizar un mapa de calor (Folium HeatMap) de hotspots de actividad en Chicago.

### 5) Notas para el revisor

- El alcance real de v1 se centra en **predicción de duración de viajes** + API REST. Las secciones del README sobre forecasting avanzado, optimización geoespacial y RL son roadmap conceptual.
- `operational_impact.ipynb` y `geospatial_demo.ipynb` no modifican el pipeline de producción; proporcionan narrativas adicionales sobre impacto operativo y contexto geoespacial.
- La demo completa recomendada es:
  1. `make train` + `python -m app.example_load`.
  2. API `/predict` con un par de consultas de ejemplo.
  3. Explorar el notebook de impacto operativo.
  4. Ejecutar la demo geoespacial mínima para visualizar hotspots.
