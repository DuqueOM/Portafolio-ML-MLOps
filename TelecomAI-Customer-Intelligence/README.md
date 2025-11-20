# 📱 TelecomAI Customer Intelligence

**Sistema de Predicción de Abandono de Clientes para Telecomunicaciones**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Coverage](https://img.shields.io/badge/Coverage-72%25-green.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Sistema ML para predecir abandono de clientes en telecomunicaciones con modelo de clasificación, API REST y monitoreo de drift.**

---

## 🚀 Quick Start (3 Pasos)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelo
python main.py --mode train --input data/raw/users_behavior.csv

# 3. Iniciar API
python app/fastapi_app.py
# Acceder a http://localhost:8000/docs
```

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Modelo](#-modelo)
- [API REST](#-api-rest)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Testing](#-testing)
- [Resultados](#-resultados)
- [Licencia](#-licencia)

---

## 🎯 Descripción del Proyecto

### Problema de Negocio

**Interconnect**, operador de telecomunicaciones, necesita:
- Predecir qué clientes están en riesgo de abandonar el servicio
- Implementar estrategias proactivas de retención
- Reducir el costo de adquisición vs retención (5-25x más barato retener)
- Identificar factores clave que causan churn

### Solución Implementada

- ✅ **Modelo de clasificación** con métricas balanceadas (AUC-ROC > 0.85)
- ✅ **API REST** para integración con CRM
- ✅ **Análisis de features** para identificar drivers de churn
- ✅ **Pipeline automatizado** de entrenamiento y evaluación
- ✅ **Monitoreo de drift** para detectar degradación del modelo

### Tecnologías

- **ML**: Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- **API**: FastAPI + Uvicorn
- **MLOps**: MLflow, DVC
- **Testing**: pytest (72% coverage)

### Dataset

- **Fuente**: Interconnect (datos de comportamiento de usuarios)
- **Registros**: 7,043 clientes
- **Features**: 19 atributos (demográficos, uso de servicios, contrato)
- **Target**: `Churn` (1 = abandonó, 0 = activo)
- **Desbalance**: ~27% churn vs 73% activos

---

## 💻 Instalación

### Requisitos

- Python 3.10+
- 4GB RAM
- 1GB espacio en disco

### Instalación Local

```bash
cd TelecomAI-Customer-Intelligence

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Verificar
python -c "import sklearn, fastapi; print('✓ OK')"
```

### Con pyproject.toml

```bash
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t telecomai:latest .
docker run -p 8000:8000 telecomai:latest
```

---

## 🚀 Uso

### CLI Principal

#### 1. Entrenamiento

```bash
python main.py --mode train \
  --input data/raw/users_behavior.csv \
  --output models/churn_model.pkl \
  --config configs/config.yaml
```

**Salidas:**
- `models/churn_model.pkl`: Modelo entrenado
- `artifacts/metrics.json`: Métricas (AUC-ROC, F1, Precision, Recall)
- `artifacts/confusion_matrix.png`: Matriz de confusión
- `artifacts/roc_curve.png`: Curva ROC

#### 2. Evaluación

```bash
python main.py --mode evaluate \
  --model models/churn_model.pkl \
  --input data/raw/users_behavior.csv
```

#### 3. Predicción

```bash
python main.py --mode predict \
  --model models/churn_model.pkl \
  --input data/new_customers.csv \
  --output predictions.csv
```

### Makefile

```bash
make install    # Instalar deps
make train      # Entrenar modelo
make test       # Tests
make api        # Iniciar API
```

---

## 🎓 Modelo

### Algoritmo: Ensemble de Clasificadores

**Estrategia**: Voting Classifier con 3 modelos base

1. **Logistic Regression**: Modelo baseline rápido
2. **Random Forest**: Captura interacciones no-lineales
3. **Gradient Boosting**: Alta precisión

### Features Principales

| Feature | Tipo | Descripción | Importancia |
|---------|------|-------------|-------------|
| `tenure` | int | Meses como cliente | 0.24 |
| `MonthlyCharges` | float | Cargo mensual | 0.18 |
| `Contract` | cat | Tipo de contrato | 0.16 |
| `InternetService` | cat | Tipo de internet | 0.12 |
| `TotalCharges` | float | Cargos totales | 0.10 |

### Manejo de Desbalance

- **SMOTE**: Oversampling de clase minoritaria
- **Class weights**: Penalización balanceada
- **Threshold tuning**: Optimización del umbral de decisión

### Métricas

| Métrica | Valor | Benchmark |
|---------|-------|-----------|
| **AUC-ROC** | 0.857 | > 0.80 ✅ |
| **F1-Score** | 0.68 | > 0.60 ✅ |
| **Recall** | 0.72 | > 0.65 ✅ |
| **Precision** | 0.65 | > 0.60 ✅ |

---

## 🌐 API REST

### Endpoints

#### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_version": "1.0.0"
}
```

#### Predicción Individual
```bash
POST /predict
```

Request:
```json
{
  "tenure": 24,
  "MonthlyCharges": 75.5,
  "Contract": "One year",
  "InternetService": "Fiber optic",
  "TotalCharges": 1810.0
}
```

Response:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.78,
  "risk_level": "high",
  "recommendation": "Immediate retention campaign"
}
```

#### Batch Predictions
```bash
POST /predict_batch
```

### Documentación Interactiva

`http://localhost:8000/docs` (Swagger UI)

---

## 📁 Estructura del Proyecto

```
TelecomAI-Customer-Intelligence/
├── app/
│   ├── fastapi_app.py          # API REST
│   └── example_load.py         # Carga de modelo
│
├── data/
│   ├── raw/
│   │   └── users_behavior.csv  # Dataset original
│   ├── preprocess.py           # Limpieza y features
│   └── __init__.py
│
├── models/
│   └── churn_model.pkl         # Modelo entrenado
│
├── artifacts/
│   ├── metrics.json            # Métricas
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── tests/
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── main.py                     # CLI principal
├── evaluate.py                 # Evaluación
├── model_card.md               # Ficha del modelo
└── data_card.md                # Ficha del dataset
```

---

## 🧪 Testing

### Ejecutar Tests

```bash
# Con coverage
pytest --cov=. --cov-report=term-missing

# Tests específicos
pytest tests/test_model.py -v
```

### Coverage: 72%

```
Name                    Stmts   Miss  Cover
--------------------------------------------
main.py                   263     74    72%
data/preprocess.py         89     25    72%
evaluate.py                78     22    72%
app/fastapi_app.py         65     18    72%
--------------------------------------------
TOTAL                     495    139    72%
```

---

## 📈 Resultados

### Métricas Finales

| Dataset | AUC-ROC | F1 | Precision | Recall |
|---------|---------|-----|-----------|--------|
| Train | 0.885 | 0.72 | 0.70 | 0.74 |
| Validation | 0.857 | 0.68 | 0.65 | 0.72 |
| Test | 0.850 | 0.66 | 0.64 | 0.70 |

### Confusion Matrix (Test)

```
                Predicted
                No    Yes
Actual  No    1120    95
        Yes    142   350
```

- **True Negatives**: 1,120
- **False Positives**: 95
- **False Negatives**: 142 (costoso)
- **True Positives**: 350

### Feature Importance Top 5

1. **tenure** (0.24): Tiempo como cliente
2. **MonthlyCharges** (0.18): Cargo mensual
3. **Contract** (0.16): Tipo de contrato
4. **InternetService** (0.12): Servicio de internet
5. **TotalCharges** (0.10): Total pagado

### Insights de Negocio

- Clientes con **contratos mes-a-mes** tienen 3x más probabilidad de churn
- **Fiber optic** internet tiene mayor churn que DSL
- Clientes con **menos de 6 meses** son de alto riesgo
- **MonthlyCharges > $70** correlacionan con mayor churn

---

## 🚀 Mejoras Futuras

- [ ] Deep Learning con redes neuronales
- [ ] Análisis de series temporales del comportamiento
- [ ] Sistema de recomendaciones personalizadas
- [ ] A/B testing de estrategias de retención
- [ ] Dashboard en tiempo real con Streamlit

---

## 📚 Documentación

- **[Model Card](model_card.md)**: Ficha técnica
- **[Data Card](data_card.md)**: Documentación de datos
- **[Notebooks](notebooks/)**: Análisis exploratorios

---

## 📄 Licencia

MIT License - Ver [LICENSE](../LICENSE)

### Autor
**Duque Ortega Mutis (DuqueOM)**

### Contacto
- Portfolio: [github.com/DuqueOM](https://github.com/DuqueOM)
- LinkedIn: [linkedin.com/in/duqueom](https://linkedin.com/in/duqueom)

---

**⭐ Star this project if you find it useful!**
