# 🚀 Comandos de Reproducibilidad - BankChurn Predictor

## Setup Inicial

```bash
# 1. Crear entorno virtual
python -m venv bankchurn-env
source bankchurn-env/bin/activate  # Linux/Mac
# bankchurn-env\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Crear estructura de directorios
mkdir -p {data/raw,data/processed,models,results,logs,configs}
```

## Entrenamiento Completo

```bash
# Entrenamiento desde cero con configuración por defecto
python main.py --mode train --input data/raw/Churn.csv --seed 42

# Entrenamiento con configuración custom
python main.py --mode train --config configs/config.yaml --seed 42

# Entrenamiento con logging detallado
python main.py --mode train --input data/raw/Churn.csv --seed 42 2>&1 | tee logs/training.log
```

## Evaluación

```bash
# Evaluar modelo entrenado
python main.py --mode evaluate --model models/best_model.pkl --input data/raw/Churn.csv

# Evaluar con datos de test específicos
python main.py --mode evaluate --model models/best_model.pkl --input data/test_set.csv
```

## Predicción

```bash
# Predicciones en nuevos datos
python main.py --mode predict --model models/best_model.pkl --input data/new_customers.csv --output predictions.csv

# Predicciones con modelo específico
python main.py --mode predict --model models/best_model_20241116.pkl --input data/batch_customers.csv --output batch_predictions.csv
```

## Optimización de Hiperparámetros

```bash
# Optimización básica (100 trials)
python main.py --mode hyperopt --input data/raw/Churn.csv --n_trials 100

# Optimización extendida (500 trials, 2 horas timeout)
python main.py --mode hyperopt --input data/raw/Churn.csv --n_trials 500 --timeout 7200

# Optimización con configuración específica
python main.py --mode hyperopt --config configs/hyperopt_config.yaml --n_trials 200
```

## Pipeline Completo de Reproducibilidad

```bash
#!/bin/bash
# run_complete_pipeline.sh

echo "=== PIPELINE COMPLETO BANKCHURN PREDICTOR ==="

# 1. Setup
echo "1. Configurando entorno..."
python -m venv bankchurn-env
source bankchurn-env/bin/activate
pip install -r requirements.txt

# 2. Entrenamiento
echo "2. Entrenando modelo..."
python main.py --mode train --input data/raw/Churn.csv --seed 42

# 3. Evaluación
echo "3. Evaluando modelo..."
python main.py --mode evaluate --model models/best_model.pkl --input data/raw/Churn.csv

# 4. Predicciones de ejemplo
echo "4. Generando predicciones de ejemplo..."
python main.py --mode predict --model models/best_model.pkl --input data/raw/Churn.csv --output example_predictions.csv

echo "=== PIPELINE COMPLETADO ==="
echo "Resultados disponibles en:"
echo "- Modelo: models/best_model.pkl"
echo "- Métricas: results/training_results.json"
echo "- Predicciones: example_predictions.csv"
```

## Comandos de Desarrollo

```bash
# Ejecutar tests
python -m pytest tests/ -v

# Ejecutar tests con cobertura
python -m pytest tests/ --cov=src --cov-report=html

# Linting y formato de código
black src/ main.py
flake8 src/ main.py
mypy src/ main.py

# Generar documentación
sphinx-build -b html docs/ docs/_build/
```

## Docker

```bash
# Build de imagen
docker build -t bankchurn-predictor:latest .

# Ejecutar entrenamiento en container
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models bankchurn-predictor:latest python main.py --mode train

# Ejecutar API en container
docker run -p 8000:8000 bankchurn-predictor:latest uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000

# Docker Compose (completo)
docker-compose up -d
```

## Comandos de Monitoreo

```bash
# Monitorear logs en tiempo real
tail -f logs/bankchurn.log

# Verificar performance del modelo
python evaluate.py --model models/best_model.pkl --metrics f1_score roc_auc

# Generar reporte de drift
python scripts/check_data_drift.py --reference data/raw/Churn.csv --current data/new_batch.csv
```

## Comandos de Backup y Versionado

```bash
# Backup de modelo con timestamp
cp models/best_model.pkl models/backup/best_model_$(date +%Y%m%d_%H%M%S).pkl

# Versionado con git
git add models/ results/
git commit -m "Model v1.0.0 - F1: 0.637, AUC: 0.867"
git tag v1.0.0

# Exportar modelo para producción
python scripts/export_model.py --model models/best_model.pkl --format onnx --output models/production/
```

## Troubleshooting

```bash
# Verificar instalación
python -c "import sklearn, pandas, numpy; print('Dependencias OK')"

# Verificar datos
python -c "import pandas as pd; df = pd.read_csv('data/raw/Churn.csv'); print(f'Datos: {df.shape}')"

# Verificar modelo
python -c "import joblib; model = joblib.load('models/best_model.pkl'); print('Modelo cargado OK')"

# Debug mode
python main.py --mode train --input data/raw/Churn.csv --seed 42 --debug

# Verificar memoria y CPU
python scripts/profile_training.py --input data/raw/Churn.csv
```

## Comandos de Producción

```bash
# Healthcheck de API
curl -X GET "http://localhost:8000/health"

# Test de predicción
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"CreditScore": 650, "Geography": "Germany", "Gender": "Female", "Age": 45, "Tenure": 5, "Balance": 120000, "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 75000}'

# Batch prediction via API
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d @data/batch_request.json

# Monitoreo de performance
curl -X GET "http://localhost:8000/metrics"
```
