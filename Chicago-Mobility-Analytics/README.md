# 🚕 Chicago Mobility Analytics

**Sistema de Análisis y Predicción de Demanda de Taxis en Chicago**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Coverage](https://img.shields.io/badge/Coverage-50%25-yellow.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Análisis temporal y predicción de demanda de taxis con modelos de series temporales y regresión.**

---

## 🚀 Quick Start

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelo
python main.py --mode train --input data/raw/taxi_data.csv

# 3. Predecir demanda
python main.py --mode predict --date "2018-03-15" --hour 18
```

---

## 🎯 Descripción

### Problema
Sweet Lift Taxi en Chicago necesita predecir la demanda de taxis para optimizar la asignación de conductores durante picos de demanda (especialmente en aeropuertos).

### Solución
- ✅ Modelo de regresión para predecir número de viajes por hora
- ✅ Análisis de patrones temporales (día, hora, día de semana)
- ✅ Feature engineering con lags y rolling statistics
- ✅ RMSE < 50 viajes (precisión del 85%)

### Tecnologías
- **ML**: Scikit-learn, LightGBM
- **Análisis**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn
- **Testing**: pytest (50% coverage)

### Dataset
- **Fuente**: Sweet Lift Taxi - Chicago
- **Registros**: ~26,000 observaciones horarias
- **Periodo**: Verano 2017
- **Target**: Número de viajes por hora

---

## 💻 Instalación

```bash
cd Chicago-Mobility-Analytics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Con pyproject.toml
```bash
pip install -e ".[dev]"
```

---

## 🚀 Uso

### CLI

#### Entrenamiento
```bash
python main.py --mode train \
  --input data/raw/taxi_data.csv \
  --output models/demand_predictor.pkl
```

#### Predicción
```bash
python main.py --mode predict \
  --model models/demand_predictor.pkl \
  --date "2018-03-15" \
  --hour 18
```

Output:
```
Predicted demand: 42 trips
Confidence interval: [38, 46]
```

#### Evaluación
```bash
python main.py --mode evaluate \
  --model models/demand_predictor.pkl \
  --test-data data/processed/test.csv
```

---

## 🎓 Modelo

### Algoritmo: Gradient Boosting (LightGBM)

**Features**:
- `hour`: Hora del día (0-23)
- `day_of_week`: Día de la semana (0-6)
- `is_weekend`: Indicador de fin de semana
- `lag_1h`, `lag_24h`: Demanda en horas anteriores
- `rolling_mean_3h`: Promedio móvil 3 horas

### Métricas

| Métrica | Valor | Benchmark |
|---------|-------|-----------|
| **RMSE** | 48.2 | < 50 ✅ |
| **MAE** | 35.1 | < 40 ✅ |
| **R²** | 0.82 | > 0.75 ✅ |

---

## 📁 Estructura

```
Chicago-Mobility-Analytics/
├── data/
│   ├── raw/taxi_data.csv
│   └── preprocess.py
├── models/
│   └── demand_predictor.pkl
├── artifacts/
│   └── metrics.json
├── tests/
│   └── test_preprocessing.py
├── main.py
└── evaluate.py
```

---

## 🧪 Testing

```bash
pytest --cov=. --cov-report=term-missing
```

Coverage: 50%

---

## 📈 Resultados

### Insights Clave
- **Pico de demanda**: 18:00-20:00 horas (+35%)
- **Día más ocupado**: Viernes (+28% vs promedio)
- **Aeropuertos**: 40% de viajes en horas pico
- **Predicción**: Error promedio de ±35 viajes

---

## 📄 Licencia

MIT License - Ver [LICENSE](../LICENSE)

**Autor**: Duque Ortega Mutis (DuqueOM)

---

**⭐ Star if useful!**
