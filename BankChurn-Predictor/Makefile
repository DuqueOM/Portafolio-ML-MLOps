# Makefile para BankChurn Predictor
# Uso: make <target>

.PHONY: help install test train evaluate predict clean docker-build docker-run lint format

# Variables
PYTHON := python
PIP := pip
DOCKER_IMAGE := bankchurn-predictor
DOCKER_TAG := latest

# Colores para output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Mostrar ayuda
	@echo "$(GREEN)BankChurn Predictor - Comandos disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Instalar dependencias
	@echo "$(GREEN)Instalando dependencias...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencias instaladas correctamente$(NC)"

install-dev: ## Instalar dependencias de desarrollo
	@echo "$(GREEN)Instalando dependencias de desarrollo...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy
	@echo "$(GREEN)Dependencias de desarrollo instaladas$(NC)"

setup: ## Configurar entorno inicial
	@echo "$(GREEN)Configurando entorno...$(NC)"
	mkdir -p data/raw data/processed models results logs configs
	@if [ ! -f configs/config.yaml ]; then \
		echo "$(YELLOW)Copiando configuración por defecto...$(NC)"; \
		cp configs/config.yaml.example configs/config.yaml 2>/dev/null || true; \
	fi
	@echo "$(GREEN)Entorno configurado$(NC)"

test: ## Ejecutar tests
	@echo "$(GREEN)Ejecutando tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short

test-coverage: ## Ejecutar tests con cobertura
	@echo "$(GREEN)Ejecutando tests con cobertura...$(NC)"
	$(PYTHON) -m pytest tests/ --cov=src --cov=main --cov-report=html --cov-report=term

lint: ## Verificar calidad de código
	@echo "$(GREEN)Verificando calidad de código...$(NC)"
	flake8 src/ main.py app/ --max-line-length=100 --ignore=E203,W503
	mypy src/ main.py --ignore-missing-imports

format: ## Formatear código
	@echo "$(GREEN)Formateando código...$(NC)"
	black src/ main.py app/ tests/ --line-length=100
	@echo "$(GREEN)Código formateado$(NC)"

train: ## Entrenar modelo
	@echo "$(GREEN)Entrenando modelo...$(NC)"
	$(PYTHON) main.py --mode train --config configs/config.yaml --seed 42 --input data/raw/Churn.csv

train-hyperopt: ## Entrenar con optimización de hiperparámetros
	@echo "$(GREEN)Entrenando con optimización de hiperparámetros...$(NC)"
	$(PYTHON) main.py --mode hyperopt --n_trials 100 --timeout 3600

evaluate: ## Evaluar modelo
	@echo "$(GREEN)Evaluando modelo...$(NC)"
	$(PYTHON) main.py --mode evaluate --model models/best_model.pkl --config configs/config.yaml --input data/raw/Churn.csv

predict: ## Hacer predicciones
	@echo "$(GREEN)Realizando predicciones...$(NC)"
	$(PYTHON) main.py --mode predict --input data/raw/Churn.csv --output predictions.csv

# quick demo: install, train and serve
start-demo: ## Demo rápida en 5 minutos
	$(MAKE) install
	$(MAKE) train
	$(MAKE) api-start

mlflow-demo: ## Registrar un run de ejemplo en MLflow (opcional)
	$(PYTHON) scripts/run_mlflow.py

check-drift: ## Chequeo de drift (KS/PSI) y reporte Evidently opcional
	$(PYTHON) monitoring/check_drift.py --ref data/raw/Churn.csv --cur data/raw/Churn.csv --out-json results/drift.json --report-html results/drift_report.html

test-fairness: ## Ejecutar tests de fairness
	$(PYTHON) -m pytest -q tests/test_fairness.py

api-start: ## Iniciar API
	@echo "$(GREEN)Iniciando API...$(NC)"
	cd app && uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

# alias
serve: ## Alias para iniciar API
	$(MAKE) api-start

api-test: ## Probar API
	@echo "$(GREEN)Probando API...$(NC)"
	curl -X GET "http://localhost:8000/health"
	@echo ""
	curl -X GET "http://localhost:8000/model_info"

docker-build: ## Construir imagen Docker
	@echo "$(GREEN)Construyendo imagen Docker...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)Imagen Docker construida: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-run: ## Ejecutar container Docker
	@echo "$(GREEN)Ejecutando container Docker...$(NC)"
	docker run -d \
		--name bankchurn-container \
		-p 8000:8000 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)Container ejecutándose en http://localhost:8000$(NC)"

docker-stop: ## Detener container Docker
	@echo "$(YELLOW)Deteniendo container Docker...$(NC)"
	docker stop bankchurn-container || true
	docker rm bankchurn-container || true

docker-compose-up: ## Iniciar con docker-compose
	@echo "$(GREEN)Iniciando servicios con docker-compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Servicios iniciados$(NC)"

docker-compose-down: ## Detener docker-compose
	@echo "$(YELLOW)Deteniendo servicios docker-compose...$(NC)"
	docker-compose down

pipeline: ## Ejecutar pipeline completo
	@echo "$(GREEN)Ejecutando pipeline completo...$(NC)"
	$(MAKE) setup
	$(MAKE) install
	$(MAKE) test
	$(MAKE) train
	$(MAKE) evaluate
	@echo "$(GREEN)Pipeline completado exitosamente$(NC)"

clean: ## Limpiar archivos temporales
	@echo "$(YELLOW)Limpiando archivos temporales...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "$(GREEN)Archivos temporales eliminados$(NC)"

clean-all: clean ## Limpiar todo (incluyendo modelos y resultados)
	@echo "$(RED)Limpiando modelos y resultados...$(NC)"
	rm -rf models/*.pkl
	rm -rf results/*
	rm -rf logs/*.log
	@echo "$(GREEN)Limpieza completa realizada$(NC)"

backup: ## Crear backup de modelos y resultados
	@echo "$(GREEN)Creando backup...$(NC)"
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz models/ results/ configs/
	@echo "$(GREEN)Backup creado$(NC)"

profile: ## Perfilar rendimiento del entrenamiento
	@echo "$(GREEN)Perfilando rendimiento...$(NC)"
	$(PYTHON) -m cProfile -o profile_stats.prof main.py --mode train
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile_stats.prof'); p.sort_stats('cumulative').print_stats(20)"

benchmark: ## Benchmark de predicciones
	@echo "$(GREEN)Ejecutando benchmark...$(NC)"
	$(PYTHON) -c "
import time
import pandas as pd
from main import BankChurnPredictor
import numpy as np

# Cargar modelo
predictor = BankChurnPredictor()
predictor.load_model('models/best_model.pkl', 'models/preprocessor.pkl')

# Crear datos de prueba
n_samples = 1000
test_data = pd.DataFrame({
    'CreditScore': np.random.randint(300, 851, n_samples),
    'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Age': np.random.randint(18, 93, n_samples),
    'Tenure': np.random.randint(0, 11, n_samples),
    'Balance': np.random.uniform(0, 250000, n_samples),
    'NumOfProducts': np.random.randint(1, 5, n_samples),
    'HasCrCard': np.random.choice([0, 1], n_samples),
    'IsActiveMember': np.random.choice([0, 1], n_samples),
    'EstimatedSalary': np.random.uniform(11, 200000, n_samples)
})

# Benchmark
start_time = time.time()
predictions, probabilities = predictor.predict(test_data)
end_time = time.time()

total_time = end_time - start_time
avg_time_per_prediction = total_time / n_samples * 1000

print(f'Benchmark Results:')
print(f'Total samples: {n_samples}')
print(f'Total time: {total_time:.4f} seconds')
print(f'Average time per prediction: {avg_time_per_prediction:.4f} ms')
print(f'Predictions per second: {n_samples / total_time:.2f}')
"

docs: ## Generar documentación
	@echo "$(GREEN)Generando documentación...$(NC)"
	@echo "Documentación disponible en:"
	@echo "- README.md: Documentación principal"
	@echo "- API_EXAMPLES.md: Ejemplos de API"
	@echo "- COMMANDS.md: Comandos de reproducibilidad"
	@echo "- EXECUTIVE_SUMMARY.md: Resumen ejecutivo"

check-deps: ## Verificar dependencias
	@echo "$(GREEN)Verificando dependencias...$(NC)"
	$(PYTHON) -c "
import sys
required_packages = ['pandas', 'numpy', 'scikit-learn', 'fastapi', 'uvicorn', 'pydantic', 'joblib', 'optuna']
missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        missing.append(package)
        print(f'✗ {package}')

if missing:
    print(f'\nPaquetes faltantes: {missing}')
    print('Ejecutar: make install')
    sys.exit(1)
else:
    print('\n$(GREEN)Todas las dependencias están instaladas$(NC)')
"

# Target por defecto
.DEFAULT_GOAL := help
