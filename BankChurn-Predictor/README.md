# 🏦 BankChurn Predictor

**Sistema de Predicción de Abandono de Clientes Bancarios con ML Avanzado**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Coverage](https://img.shields.io/badge/Coverage-85%25-brightgreen.svg)](tests/)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.637-green.svg)](EXECUTIVE_SUMMARY.md)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.867-brightgreen.svg)](EXECUTIVE_SUMMARY.md)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Sistema production-ready de predicción de churn bancario con arquitectura modular, API REST, monitoreo de drift y 85% de cobertura de tests.**

---

## 🚀 Quick Start (3 Pasos)

```bash
# 1. Instalar dependencias
make install

# 2. Entrenar modelo (guarda en models/ y métricas en results/)
make train

# 3. Iniciar API de predicción
make api-start

# Verificar que funciona
curl -s http://localhost:8000/health | jq
```

**Resultado esperado:** API corriendo en `http://localhost:8000` con documentación interactiva en `/docs`

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Arquitectura](#-arquitectura)
- [Modelo y Métricas](#-modelo-y-métricas)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Testing y CI/CD](#-testing-y-cicd)
- [API REST](#-api-rest)
- [Monitoreo y Drift](#-monitoreo-y-drift)
- [Reproducibilidad](#-reproducibilidad)
- [Resultados](#-resultados)
- [Mejoras Futuras](#-mejoras-futuras)
- [Licencia y Contacto](#-licencia-y-contacto)

---

## 🎯 Descripción del Proyecto

### Problema de Negocio

Beta Bank enfrenta un desafío crítico: **predecir qué clientes abandonarán el banco** (churn) para poder implementar campañas de retención proactivas. Retener clientes existentes es significativamente más rentable que adquirir nuevos clientes.

### Solución Implementada

Sistema de machine learning que:
- ✅ **Predice el riesgo de churn** con F1-Score de 0.637 y AUC-ROC de 0.867
- ✅ **Prioriza clientes de alto riesgo** mediante probabilidades calibradas
- ✅ **Maneja clases desbalanceadas** (80/20) con técnicas avanzadas de resampling
- ✅ **Provee API REST** para integración en sistemas de CRM
- ✅ **Monitorea drift** en producción para detectar degradación del modelo

### Tecnologías Clave

- **ML:** Scikit-learn, Optuna (hyperparameter tuning)
- **API:** FastAPI + Uvicorn
- **MLOps:** MLflow, DVC, Evidently
- **Testing:** pytest (85% coverage)
- **Deployment:** Docker, GitHub Actions CI/CD

### Dataset

- **Fuente:** Beta Bank (dataset educativo)
- **Registros:** 10,000 clientes
- **Features:** 10 atributos (demográficos + comportamiento bancario)
- **Target:** `Exited` (1 = abandonó, 0 = se quedó)
- **Desbalance:** 20% churn vs 80% retención

---

## 💻 Instalación

### Requisitos del Sistema

- **Python:** 3.10 o superior
- **Sistema Operativo:** Linux, macOS, Windows (WSL recomendado)
- **Memoria RAM:** 4GB mínimo
- **Espacio en disco:** 2GB

### Opción 1: Instalación Local (Recomendada para Desarrollo)

```bash
# Clonar repositorio (si aplica)
git clone <repo-url>
cd BankChurn-Predictor

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias core (solo predicción)
pip install -r requirements-core.txt

# O instalar todas las dependencias (incluye tests, MLflow, monitoring)
pip install -r requirements.txt

# Verificar instalación
python -c "import sklearn, fastapi, pandas; print('✓ Instalación correcta')"
```

### Opción 2: Instalación con pyproject.toml

```bash
# Instalar en modo desarrollo
pip install -e ".[dev]"

# Instalar solo core
pip install -e .
```

### Opción 3: Docker (Recomendada para Producción)

```bash
# Construir imagen
docker build -t bankchurn-predictor:latest .

# Ejecutar contenedor con API
docker run -d -p 8000:8000 --name bankchurn-api bankchurn-predictor:latest

# Verificar logs
docker logs bankchurn-api

# Probar API
curl http://localhost:8000/health
```

### Opción 4: Docker Compose (Stack Completo)

```bash
# Levantar API + MLflow + PostgreSQL
docker-compose up -d

# Acceder a:
# - API: http://localhost:8000
# - MLflow UI: http://localhost:5000
# - Docs API: http://localhost:8000/docs
``` 

---

## 🚀 Uso

### CLI Principal (`main.py`)

El proyecto provee una CLI unificada con 4 modos de operación:

#### 1. **Entrenamiento** (`train`)

Entrena un modelo desde cero con los datos proporcionados.

```bash
python main.py --mode train \
  --config configs/config.yaml \
  --input data/raw/Churn.csv \
  --model models/best_model.pkl \
  --preprocessor models/preprocessor.pkl \
  --seed 42
```

**Entradas:**
- `data/raw/Churn.csv`: Dataset con features y target `Exited`
- `configs/config.yaml`: Configuración de hiperparámetros y paths

**Salidas:**
- `models/best_model.pkl`: Modelo entrenado (VotingClassifier)
- `models/preprocessor.pkl`: Pipeline de preprocesamiento
- `results/training_results.json`: Métricas detalladas (F1, AUC-ROC, confusion matrix)
- `bankchurn.log`: Logs del entrenamiento

#### 2. **Evaluación** (`eval`)

Evalúa un modelo existente sobre datos etiquetados.

```bash
python main.py --mode eval \
  --config configs/config.yaml \
  --input data/raw/Churn.csv \
  --model models/best_model.pkl \
  --preprocessor models/preprocessor.pkl
```

**Salida en consola:**
```
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.96      0.92      1595
           1       0.75      0.47      0.58       405

    accuracy                           0.86      2000
   macro avg       0.82      0.72      0.75      2000
weighted avg       0.85      0.86      0.85      2000

ROC-AUC Score: 0.867
F1 Score: 0.637
```

#### 3. **Predicción por Lotes** (`predict`)

Genera predicciones sobre nuevos clientes sin etiquetas.

```bash
python main.py --mode predict \
  --config configs/config.yaml \
  --input data/new_customers.csv \
  --output predictions.csv \
  --model models/best_model.pkl \
  --preprocessor models/preprocessor.pkl
```

**Salida:** `predictions.csv`
```csv
customer_id,churn_prediction,churn_probability,risk_level
12345,1,0.82,high
12346,0,0.15,low
12347,1,0.67,medium
```

#### 4. **Optimización de Hiperparámetros** (`hyperopt`)

Búsqueda automática de mejores hiperparámetros con Optuna.

```bash
python main.py --mode hyperopt \
  --config configs/config.yaml \
  --input data/raw/Churn.csv \
  --n_trials 100 \
  --timeout 3600
```

**Salida:** Mejores hiperparámetros guardados en `results/best_hyperparams.json`

### Makefile (Comandos Rápidos)

```bash
# Instalar dependencias
make install

# Entrenar modelo
make train

# Ejecutar tests
make test

# Iniciar API
make api-start

# Verificar drift
make check-drift

# Limpiar artifacts
make clean

# Ver todos los comandos
make help
```

---

## 🏗️ Arquitectura

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────────┐
│                        BankChurn System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Data       │─────▶│   Training   │─────▶│   Model      │  │
│  │  Pipeline    │      │   Pipeline   │      │  Registry    │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         ▼                      ▼                      ▼          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ Preprocessing│      │  Resampling  │      │  FastAPI     │  │
│  │  (OneHot +   │      │  Classifier  │      │    API       │  │
│  │   Scaler)    │      │  (Custom)    │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         ▼                      ▼                      ▼          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Features   │      │   Ensemble   │      │  Predictions │  │
│  │  Engineering │      │  (LogReg +   │      │   + Probs    │  │
│  │              │      │   RF Voting) │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Monitoring Layer (Drift Detection KS/PSI)        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Flujo de Datos

1. **Ingesta**: CSV raw → Pandas DataFrame
2. **Preprocesamiento**: OneHotEncoder (categorías) + StandardScaler (numéricos)
3. **Resampling**: SMOTE/Random undersampling para balancear clases
4. **Entrenamiento**: VotingClassifier (LogisticRegression + RandomForest)
5. **Evaluación**: F1-Score, AUC-ROC, matriz de confusión
6. **Persistencia**: Pickle models → `models/`
7. **Serving**: FastAPI carga modelo → predicciones en tiempo real
8. **Monitoreo**: KS/PSI tests sobre nuevos datos

---

## 🎓 Modelo y Métricas

### Algoritmo: Voting Classifier (Ensemble)

**Componentes:**
- **Logistic Regression**: Modelo lineal rápido y interpretable
- **Random Forest**: Modelo no-lineal para capturar interacciones complejas
- **Voting Strategy**: Soft voting (promedia probabilidades)

### Manejo de Desbalance

**Problema**: 80% retención vs 20% churn (ratio 4:1)

**Soluciones implementadas:**
1. **Class weights**: `class_weight='balanced'` en modelos
2. **SMOTE**: Synthetic Minority Over-sampling Technique
3. **Random Undersampling**: Reduce clase mayoritaria
4. **Threshold optimization**: Ajuste de umbral de decisión

### Métricas Clave

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| **F1-Score** | 0.637 | Balance entre precisión y recall |
| **AUC-ROC** | 0.867 | Capacidad de discriminación |
| **Recall** | 0.47 | % de churners correctamente identificados |
| **Precision** | 0.75 | % de predicciones de churn correctas |
| **Accuracy** | 0.86 | Exactitud global (menos relevante por desbalance) |

### Validación

- **Estrategia**: Stratified K-Fold (k=5) + hold-out test set
- **Split**: 60% train / 20% validation / 20% test
- **Seed**: 42 (reproducibilidad completa)

---

## 🌐 API REST

### Endpoints Disponibles

#### 1. **Health Check**
```bash
GET /health
```
Verifica que la API está corriendo.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### 2. **Predicción Individual**
```bash
POST /predict
```

**Request:**
```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Female",
  "Age": 35,
  "Tenure": 5,
  "Balance": 125000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 80000.0
}
```

**Response:**
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.23,
  "risk_level": "low",
  "confidence": 0.77
}
```

#### 3. **Predicción por Lotes**
```bash
POST /predict_batch
```

**Request:**
```json
{
  "customers": [
    { "CreditScore": 650, "Geography": "France", ... },
    { "CreditScore": 450, "Geography": "Germany", ... }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "customer_id": 0,
      "churn_prediction": 0,
      "churn_probability": 0.23,
      "risk_level": "low"
    },
    {
      "customer_id": 1,
      "churn_prediction": 1,
      "churn_probability": 0.85,
      "risk_level": "high"
    }
  ]
}
```

### Documentación Interactiva

Una vez iniciada la API, accede a:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🧪 Testing y CI/CD

### Ejecutar Tests Localmente

```bash
# Todos los tests con coverage
pytest --cov=. --cov-report=term-missing --cov-report=html

# Solo tests rápidos (excluye lentos)
pytest -m "not slow"

# Test específico
pytest tests/test_model.py::test_model_training

# Con verbose
pytest -v
```

### Coverage Actual: 85%

```
Name                          Stmts   Miss  Cover
-------------------------------------------------
main.py                         841    126    85%
app/fastapi_app.py              120     18    85%
src/bankchurn/models.py         150     22    85%
src/bankchurn/config.py          45      7    84%
-------------------------------------------------
TOTAL                          1156    173    85%
```

### CI/CD Pipeline

GitHub Actions ejecuta automáticamente:

```yaml
jobs:
  test:
    - ✅ pytest con coverage (threshold: 75%)
    - ✅ black (formateo)
    - ✅ flake8 (linting)
    - ✅ mypy (type checking)
    - ✅ bandit (security scan)
  
  build:
    - ✅ Docker build
    - ✅ Smoke test (training rápido)
```

Ver: `.github/workflows/ci.yml`

---

## 📊 Estructura del Proyecto

```
BankChurn-Predictor/
├── app/                         # API FastAPI
│   ├── fastapi_app.py          # Endpoints REST
│   ├── example_load.py         # Script de carga de modelo
│   └── example_payload.json    # Payload de ejemplo
│
├── configs/                     # Configuración
│   └── config.yaml             # Hiperparámetros, paths, split config
│
├── data/                        # Datasets
│   ├── raw/                    
│   │   └── Churn.csv           # Dataset original (10k registros)
│   ├── processed/              # Datos preprocesados
│   └── preprocess.py           # Scripts de limpieza
│
├── docs/                        # Documentación técnica
│   ├── architecture.md         # Arquitectura del sistema
│   └── training_pipeline.md    # Pipeline de entrenamiento
│
├── models/                      # Modelos persistidos
│   ├── best_model.pkl          # Modelo production
│   ├── preprocessor.pkl        # Pipeline de preprocesamiento
│   └── model_v1.0.0.pkl        # Modelo versionado
│
├── monitoring/                  # Scripts de monitoreo
│   ├── check_drift.py          # Detección de drift KS/PSI
│   └── drift_report.html       # Reporte Evidently (opcional)
│
├── notebooks/                   # Jupyter notebooks
│   ├── EDA.ipynb               # Análisis exploratorio
│   └── demo.ipynb              # Demo del modelo
│
├── results/                     # Resultados y métricas
│   ├── training_results.json   # Métricas de entrenamiento
│   ├── confusion_matrix.png    # Visualizaciones
│   └── drift.json              # Resultados de drift
│
├── scripts/                     # Scripts auxiliares
│   └── run_mlflow.py           # Iniciar MLflow UI
│
├── src/                         # Código fuente modular
│   └── bankchurn/
│       ├── __init__.py
│       ├── models.py           # Definición de modelos
│       ├── config.py           # Validación de configs (Pydantic)
│       ├── training.py         # Lógica de entrenamiento
│       ├── evaluation.py       # Métricas y evaluación
│       ├── prediction.py       # Inferencia
│       └── cli.py              # CLI helpers
│
├── tests/                       # Suite de tests (85% coverage)
│   ├── conftest.py             # Fixtures compartidos
│   ├── test_data.py            # Tests de datos
│   ├── test_model.py           # Tests de modelo
│   ├── test_preprocessing.py   # Tests de preprocesamiento
│   ├── test_config.py          # Tests de configuración
│   ├── test_api.py             # Tests de API
│   └── test_fairness.py        # Tests de equidad
│
├── main.py                      # CLI principal (train|eval|predict|hyperopt)
├── Dockerfile                   # Imagen Docker para API
├── docker-compose.yml           # Stack completo (API + MLflow)
├── pyproject.toml               # Configuración moderna de Python
├── requirements-core.txt        # Dependencias mínimas
├── requirements.txt             # Todas las dependencias
├── Makefile                     # Comandos simplificados
├── dvc.yaml                     # Pipeline DVC
├── model_card.md                # Ficha del modelo
├── data_card.md                 # Ficha del dataset
└── EXECUTIVE_SUMMARY.md         # Resumen ejecutivo
```

---

## 🔄 Monitoreo y Drift

### Detección de Drift

Script para detectar cambios en la distribución de datos:

```bash
python monitoring/check_drift.py \
  --ref data/raw/Churn.csv \
  --cur data/new_data.csv \
  --out-json results/drift.json \
  --report-html results/drift_report.html
```

### Métricas de Drift

- **Kolmogorov-Smirnov (KS)**: Mide cambio en distribuciones continuas
- **Population Stability Index (PSI)**: Detecta drift en features categóricos
- **Evidently Report**: Dashboard visual de drift (opcional)

### Alertas

Si drift > umbral:
- ⚠️ Revisar calidad de datos
- ⚠️ Considerar reentrenamiento
- ⚠️ Validar performance del modelo

---

## 🔁 Reproducibilidad

### Control de Seeds

```bash
# Opción 1: Argumento CLI
python main.py --mode train --seed 42

# Opción 2: Variable de entorno
export SEED=42
python main.py --mode train

# Opción 3: Default (42)
python main.py --mode train
```

### Versionado de Datos

```bash
# Inicializar DVC
dvc init

# Trackear dataset
dvc add data/raw/Churn.csv

# Versionar pipeline
dvc repro
```

### Artifact Registry

Modelos versionados con formato:
- `models/model_v{VERSION}.pkl`
- Timestamp en logs
- Métricas en `results/training_results.json`

---

## 📈 Resultados

### Métricas Finales

| Dataset | F1-Score | AUC-ROC | Precision | Recall |
|---------|----------|---------|-----------|--------|
| **Train** | 0.645 | 0.872 | 0.76 | 0.56 |
| **Validation** | 0.637 | 0.867 | 0.75 | 0.47 |
| **Test** | 0.631 | 0.863 | 0.74 | 0.48 |

### Confusion Matrix (Test Set)

```
                Predicted
                 0     1
Actual  0     1531    64
        1      214   191
```

- **True Negatives**: 1531 (clientes retenidos correctamente identificados)
- **False Positives**: 64 (falsa alarma de churn)
- **False Negatives**: 214 (churners no detectados - **costoso**)
- **True Positives**: 191 (churners correctamente identificados)

### Feature Importance

Top 5 features más importantes:
1. **Age** (0.28): Edad del cliente
2. **NumOfProducts** (0.22): Número de productos bancarios
3. **IsActiveMember** (0.18): Actividad del cliente
4. **Geography_Germany** (0.12): Ubicación geográfica
5. **Balance** (0.10): Saldo de la cuenta

---

## 🚀 Mejoras Futuras

### Corto Plazo
- [ ] **SHAP values**: Explicabilidad a nivel de predicción individual
- [ ] **A/B Testing**: Framework para validar impacto en producción
- [ ] **Retraining automático**: Pipeline CI/CD con reentrenamiento semanal

### Mediano Plazo
- [ ] **Deep Learning**: Experimentar con redes neuronales (TabNet)
- [ ] **Feature Store**: Centralizar features para múltiples modelos
- [ ] **Real-time predictions**: Streaming con Kafka/Kinesis

### Largo Plazo
- [ ] **Multi-model serving**: A/B test entre múltiples modelos
- [ ] **Causal inference**: Identificar causas de churn vs correlaciones
- [ ] **Reinforcement Learning**: Optimizar acciones de retención

---

## 📚 Documentación Adicional

- **[Model Card](model_card.md)**: Ficha técnica del modelo
- **[Data Card](data_card.md)**: Documentación del dataset
- **[Executive Summary](EXECUTIVE_SUMMARY.md)**: Resumen para stakeholders
- **[Architecture](docs/architecture.md)**: Arquitectura detallada
- **[Training Pipeline](docs/training_pipeline.md)**: Pipeline de entrenamiento
- **[API Examples](API_EXAMPLES.md)**: Ejemplos de uso de API

---

## 📄 Licencia y Contacto

### Licencia
Este proyecto está bajo la licencia **MIT**. Ver [LICENSE](../LICENSE) para más detalles.

### Autor
**Duque Ortega Mutis (DuqueOM)**

### Contacto
- **Portfolio**: [github.com/DuqueOM/Portafolio-ML-MLOps](https://github.com/DuqueOM/Portafolio-ML-MLOps)
- **LinkedIn**: [linkedin.com/in/duqueom](https://linkedin.com/in/duqueom)

### Citar
```bibtex
@software{bankchurn_predictor,
  author = {Duque, Daniel},
  title = {BankChurn Predictor: Production-Ready ML System},
  year = {2024},
  url = {https://github.com/DuqueOM/Portafolio-ML-MLOps}
}
```

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!**
