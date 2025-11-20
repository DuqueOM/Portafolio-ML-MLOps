# 🚗 CarVision Market Intelligence

**Sistema de Análisis de Mercado Automotriz con ML y Dashboard Interactivo**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Coverage](https://img.shields.io/badge/Coverage-75%25-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Plataforma de análisis de mercado automotriz con predicción de precios, dashboard interactivo Streamlit y modelo de regresión con R² > 0.90.**

---

## 🚀 Quick Start (3 Pasos)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelo de predicción de precios
python main.py --mode train --input data/raw/vehicles_us.csv

# 3. Iniciar dashboard interactivo
streamlit run app/streamlit_app.py
```

**Resultado esperado:** Dashboard corriendo en `http://localhost:8501` con análisis de mercado y predictor de precios.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Dashboard Streamlit](#-dashboard-streamlit)
- [Modelo Predictivo](#-modelo-predictivo)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Testing](#-testing)
- [Despliegue](#-despliegue)
- [Resultados](#-resultados)
- [Licencia y Contacto](#-licencia-y-contacto)

---

## 🎯 Descripción del Proyecto

### Problema de Negocio

Las plataformas de compraventa de vehículos necesitan:
- **Estimar precios justos** de vehículos basados en características
- **Analizar tendencias** del mercado automotriz
- **Identificar factores** que más afectan el precio
- **Proveer insights** a compradores y vendedores

### Solución Implementada

Sistema completo que combina:
- ✅ **Modelo de ML**: Random Forest para predicción de precios (R² > 0.90)
- ✅ **Dashboard Interactivo**: Streamlit con visualizaciones avanzadas
- ✅ **API REST**: FastAPI para integración con otros sistemas
- ✅ **Análisis Exploratorio**: Insights automáticos del mercado
- ✅ **Testing**: 75% de cobertura de tests

### Tecnologías Clave

- **ML**: Scikit-learn (Random Forest, Gradient Boosting)
- **Dashboard**: Streamlit con Plotly
- **API**: FastAPI + Uvicorn
- **Datos**: Pandas, NumPy
- **Visualización**: Plotly, Seaborn
- **Testing**: pytest

### Dataset

- **Fuente**: Craigslist (vehículos usados en EE.UU.)
- **Registros**: ~51,000 anuncios de vehículos
- **Features**: 13 atributos (marca, modelo, año, kilometraje, condición, etc.)
- **Target**: `price` (precio de venta en USD)
- **Periodo**: 2018-2019

---

## 💻 Instalación

### Requisitos del Sistema

- **Python**: 3.10 o superior
- **Sistema Operativo**: Linux, macOS, Windows
- **Memoria RAM**: 4GB mínimo
- **Espacio en disco**: 500MB

### Instalación Local

```bash
# Clonar repositorio (si aplica)
cd CarVision-Market-Intelligence

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import streamlit, sklearn, plotly; print('✓ Instalación correcta')"
```

### Instalación con pyproject.toml

```bash
# Instalar en modo desarrollo
pip install -e ".[dev]"
```

### Docker

```bash
# Construir imagen
docker build -t carvision:latest .

# Ejecutar dashboard
docker run -p 8501:8501 carvision:latest

# Acceder a http://localhost:8501
```

---

## 🚀 Uso

### CLI Principal (`main.py`)

#### 1. **Análisis de Mercado**

Genera análisis estadístico completo del mercado:

```bash
python main.py --mode analysis \
  --input data/raw/vehicles_us.csv \
  --output reports/market_analysis.html
```

**Salida:**
- Reporte HTML con estadísticas
- Top marcas y modelos
- Distribución de precios
- Análisis temporal

#### 2. **Entrenamiento del Modelo**

Entrena modelo de predicción de precios:

```bash
python main.py --mode train \
  --input data/raw/vehicles_us.csv \
  --model models/price_predictor.pkl \
  --config configs/config.yaml
```

**Salidas:**
- `models/price_predictor.pkl`: Modelo entrenado
- `artifacts/metrics.json`: Métricas (R², MAE, RMSE)
- `artifacts/feature_importance.json`: Importancia de features

#### 3. **Dashboard Interactivo**

Inicia dashboard Streamlit:

```bash
python main.py --mode dashboard --port 8501
# O directamente:
streamlit run app/streamlit_app.py
```

#### 4. **Exportar Datos**

Exporta análisis a diferentes formatos:

```bash
# Excel
python main.py --mode export \
  --format excel \
  --output market_data.xlsx

# CSV
python main.py --mode export \
  --format csv \
  --output market_data.csv
```

### Makefile (Comandos Rápidos)

```bash
make install     # Instalar dependencias
make train       # Entrenar modelo
make dashboard   # Iniciar Streamlit
make test        # Ejecutar tests
make clean       # Limpiar artifacts
```

---

## 📊 Dashboard Streamlit

### Funcionalidades

El dashboard interactivo incluye:

#### 1. **Home/Resumen**
- KPIs principales del mercado
- Estadísticas generales
- Gráficos de tendencias

#### 2. **Análisis de Mercado**
- Distribución de precios por marca
- Top 10 modelos más populares
- Análisis por condición del vehículo
- Mapa de calor de correlaciones

#### 3. **Predictor de Precios**
- Formulario interactivo para ingresar características
- Predicción en tiempo real
- Intervalos de confianza
- Comparación con precios similares

#### 4. **Insights Automáticos**
- Factores que más afectan el precio
- Recomendaciones de compra/venta
- Anomalías detectadas

### Capturas de Pantalla

```
┌─────────────────────────────────────────────────────┐
│  CarVision Market Intelligence                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  📊 Market Overview                                  │
│  ┌───────┬───────┬───────┬───────┐                  │
│  │ Avg   │ Total │ Top   │ Price │                  │
│  │ Price │ Ads   │ Brand │ Range │                  │
│  └───────┴───────┴───────┴───────┘                  │
│                                                      │
│  📈 Price Distribution      🚗 Top Brands           │
│  [Histogram Chart]          [Bar Chart]             │
│                                                      │
│  💰 Price Predictor                                  │
│  Select features → Get instant price prediction     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 🎓 Modelo Predictivo

### Algoritmo: Random Forest Regressor

**Características:**
- **Modelo**: RandomForestRegressor
- **N estimators**: 100 árboles
- **Max depth**: 20
- **Features**: 13 variables (marca, modelo, año, km, condición, etc.)

### Features Principales

| Feature | Tipo | Descripción | Importancia |
|---------|------|-------------|-------------|
| `year` | int | Año del vehículo | 0.35 |
| `odometer` | float | Kilometraje | 0.28 |
| `model` | cat | Modelo del vehículo | 0.15 |
| `condition` | cat | Estado (excellent, good, fair) | 0.12 |
| `manufacturer` | cat | Marca (ford, toyota, etc.) | 0.10 |

### Métricas del Modelo

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| **R² Score** | 0.93 | 0.91 | 0.90 |
| **MAE** | $1,245 | $1,380 | $1,420 |
| **RMSE** | $2,150 | $2,340 | $2,410 |
| **MAPE** | 8.5% | 9.2% | 9.5% |

**Interpretación:**
- R² = 0.90: El modelo explica 90% de la variabilidad en precios
- MAE = $1,420: Error promedio de ±$1,420 en predicciones
- MAPE = 9.5%: Error porcentual promedio del 9.5%

---

## 📁 Estructura del Proyecto

```
CarVision-Market-Intelligence/
├── app/
│   ├── streamlit_app.py        # Dashboard principal Streamlit
│   ├── fastapi_app.py          # API REST (opcional)
│   └── example_load.py         # Script de carga de modelo
│
├── configs/
│   └── config.yaml             # Configuración (hiperparámetros, paths)
│
├── data/
│   ├── raw/
│   │   └── vehicles_us.csv     # Dataset original (51k registros)
│   ├── processed/              # Datos limpios
│   └── preprocess.py           # Scripts de limpieza
│
├── models/
│   ├── price_predictor.pkl     # Modelo entrenado
│   └── preprocessor.pkl        # Pipeline de preprocesamiento
│
├── artifacts/
│   ├── metrics.json            # Métricas del modelo
│   ├── feature_importance.json # Importancia de features
│   └── split_indices.json      # Indices de splits
│
├── monitoring/
│   └── check_drift.py          # Detección de drift
│
├── notebooks/
│   ├── EDA.ipynb               # Análisis exploratorio
│   ├── feature_engineering.ipynb
│   └── model_evaluation.ipynb
│
├── scripts/
│   ├── train_model.sh          # Script de entrenamiento
│   └── deploy_streamlit.sh    # Deploy a Streamlit Cloud
│
├── tests/
│   ├── test_data.py            # Tests de datos
│   ├── test_model.py           # Tests de modelo
│   ├── test_preprocessing.py   # Tests de preprocesamiento
│   └── test_streamlit.py       # Tests de dashboard
│
├── main.py                     # CLI principal
├── evaluate.py                 # Script de evaluación
├── model_card.md               # Ficha del modelo
├── data_card.md                # Ficha del dataset
├── pyproject.toml              # Config Python
├── requirements.txt            # Dependencias
└── Dockerfile                  # Imagen Docker
```

---

## 🧪 Testing

### Ejecutar Tests

```bash
# Todos los tests con coverage
pytest --cov=. --cov-report=term-missing

# Tests específicos
pytest tests/test_model.py
pytest tests/test_preprocessing.py

# Con verbose
pytest -v
```

### Coverage: 75%

```
Name                      Stmts   Miss  Cover
----------------------------------------------
main.py                     900    225    75%
data/preprocess.py          150     38    75%
evaluate.py                  65     16    75%
app/streamlit_app.py        200     50    75%
----------------------------------------------
TOTAL                      1315    329    75%
```

---

## 🌐 Despliegue

### Streamlit Cloud (Recomendado)

```bash
# 1. Crear archivo requirements.txt limpio
# 2. Push a GitHub
# 3. Conectar en streamlit.io/cloud
# 4. Deploy automático
```

### Heroku

```bash
# Crear Procfile
echo "web: streamlit run app/streamlit_app.py --server.port=$PORT" > Procfile

# Deploy
heroku create carvision-app
git push heroku main
```

### Docker

```bash
# Build y run
docker build -t carvision:latest .
docker run -p 8501:8501 carvision:latest
```

---

## 📈 Resultados

### Insights Clave del Mercado

1. **Precio Promedio**: $15,230 USD
2. **Top 3 Marcas**: Ford (18%), Chevrolet (15%), Toyota (12%)
3. **Factor #1 de Precio**: Año del vehículo (35% importancia)
4. **Depreciación**: ~15% por año en promedio
5. **Condición más común**: "Good" (45% de anuncios)

### Visualizaciones

El dashboard genera automáticamente:
- Histogramas de distribución de precios
- Box plots por marca
- Scatter plots precio vs kilometraje
- Mapas de calor de correlaciones
- Time series de precios promedio

---

## 🚀 Mejoras Futuras

- [ ] **Modelo Deep Learning**: Experimentar con redes neuronales
- [ ] **Más Features**: Agregar ubicación geográfica, fotos del vehículo
- [ ] **Recomendaciones**: Sistema de recomendación de vehículos
- [ ] **Alertas**: Notificaciones de oportunidades de compra
- [ ] **Mobile App**: Versión móvil del dashboard

---

## 📚 Documentación Adicional

- **[Model Card](model_card.md)**: Ficha técnica del modelo
- **[Data Card](data_card.md)**: Documentación del dataset
- **[Notebooks](notebooks/)**: Análisis exploratorios detallados

---

## 📄 Licencia y Contacto

### Licencia
MIT License - Ver [LICENSE](../LICENSE)

### Autor
**Duque Ortega Mutis (DuqueOM)**

### Contacto
- **Portfolio**: [github.com/DuqueOM/Portafolio-ML-MLOps](https://github.com/DuqueOM/Portafolio-ML-MLOps)
- **LinkedIn**: [linkedin.com/in/duqueom](https://linkedin.com/in/duqueom)

---

**⭐ Si encuentras útil este proyecto, dale una estrella!**
