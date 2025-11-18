# PR — OilWell Location Optimizer (aclaración de alcance, deps core/advanced y pipeline de riesgo end-to-end)

## Scope del cambio

- Refactor de `README.md` para separar claramente:
  - **Funcionalidad implementada (v1)**: CLI `main.py` + config `configs/default.yaml`, API FastAPI y scripts de drift/MLflow.
  - **Metodologías avanzadas (conceptual)**: secciones de bootstrap avanzado, optimización con CVXPY y Monte Carlo marcadas como roadmap, no parte del pipeline v1.
- División de dependencias en:
  - `requirements-core.txt` — stack mínimo para CLI + API (pandas/numpy/sklearn/fastapi/etc.).
  - `requirements-advanced.txt` — librerías de riesgo/optimización/notebooks (PyMC, CVXPY, QuantLib, Streamlit, etc.).
- Dockerfile actualizado para instalar sólo `requirements-core.txt` por defecto.
- Model card enriquecida con sección de **runtime** (tiempos aproximados de train/eval/inferencia en entorno de referencia).
- Nuevo notebook `notebooks/risk_pipeline_end_to_end.ipynb` que ejecuta entrenamiento, evaluación de riesgo (bootstrap) y recomendación de región de forma encadenada.

## Cómo correr la demo (revisor)

### 1) Demo rápida con Makefile

```bash
cd OilWell-Location-Optimizer
make install           # instala requirements-core.txt
make train             # entrena modelos lineales por región (usa configs/default.yaml)
make eval              # ejecuta bootstrap por región y guarda artifacts/risk_results.json
make api               # levanta API FastAPI en http://localhost:8000

# Healthcheck
curl -s http://localhost:8000/health | jq

# Predicción de ejemplo
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"region":1, "records":[{"f0":1.0,"f1":-2.0,"f2":3.0}]}' | jq
```

Artefactos relevantes tras `make train` + `make eval`:
- `artifacts/metrics.json` — métricas de RMSE vs baseline por región.
- `artifacts/risk_results.json` — beneficio esperado, intervalos de confianza y probabilidad de pérdida por región.

### 2) Imagen Docker (runtime mínimo)

```bash
cd OilWell-Location-Optimizer
docker build -t oilwell-optimizer .
docker run -p 8000:8000 oilwell-optimizer

# Healthcheck dentro del contenedor
curl -s http://localhost:8000/health | jq
```

La imagen sólo instala `requirements-core.txt` por defecto, lo que reduce tamaño y complejidad. Para utilizar PyMC/CVXPY/Streamlit u otras capacidades avanzadas se recomienda instalar `requirements-advanced.txt` en un entorno separado.

### 3) Pipeline de riesgo end-to-end (notebook)

Abrir el notebook:

- `notebooks/risk_pipeline_end_to_end.ipynb`

Ejecutar tras `make install` (y opcionalmente `make train`/`make eval`, aunque el notebook vuelve a lanzar el pipeline) para:
- Llamar programáticamente a `cmd_train` y `cmd_eval` (definidos en `main.py`) usando `configs/default.yaml`.
- Leer `artifacts/metrics.json` y `artifacts/risk_results.json`.
- Mostrar métricas por región y sugerir una región recomendada según beneficio esperado y probabilidad de pérdida.

### 4) Dependencias avanzadas (trade-offs)

- `requirements-advanced.txt` incluye librerías pesadas (PyMC, CVXPY, QuantLib, Streamlit, etc.) pensadas para:
  - análisis de riesgo avanzado (Monte Carlo bayesiano),
  - optimización de portafolios con constraints complejos,
  - dashboards interactivos y notebooks ricos.
- Estas dependencias **no** son necesarias para correr la demo estándar de CLI/API y se han separado para:
  - reducir el tamaño de la imagen Docker,
  - minimizar tiempos de instalación en CI/demos rápidas.
- Para explorar las ideas de roadmap en el README (optimización avanzada, simulaciones Monte Carlo detalladas), instalar explícitamente:

```bash
pip install -r requirements-advanced.txt
```

---

Este PR no cambia la firma de la API ni del CLI; aclara el alcance real del proyecto, separa dependencias core/advanced y añade un pipeline reproducible de riesgo end-to-end orientado a revisores técnicos.
