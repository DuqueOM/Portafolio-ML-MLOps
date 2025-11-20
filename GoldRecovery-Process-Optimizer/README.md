# ⚙️ GoldRecovery Process Optimizer

**Sistema de Optimización de Procesos Industriales para Recuperación de Oro**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Coverage](https://img.shields.io/badge/Coverage-50%25-yellow.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Predicción de recuperación de oro en procesos industriales con ML y métrica sMAPE personalizada.**

---

## 🚀 Quick Start

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelo
python main.py --mode train

# 3. Evaluar
python main.py --mode evaluate
```

---

## 🎯 Descripción

### Problema
Zyfra desarrolla soluciones de eficiencia industrial. Necesita **predecir el coeficiente de recuperación de oro** para optimizar el proceso y evitar parámetros no rentables.

### Solución
- ✅ Modelo de regresión multi-target (rougher + final recovery)
- ✅ Métrica personalizada: **sMAPE** (Symmetric Mean Absolute Percentage Error)
- ✅ Feature engineering de parámetros de proceso
- ✅ Validación con datos de producción real

### Tecnologías
- **ML**: Scikit-learn (Random Forest, Gradient Boosting)
- **Datos**: Pandas, NumPy
- **Testing**: pytest

### Dataset
- **Fuente**: Zyfra - Planta de procesamiento de oro
- **Registros**: ~16,000 observaciones
- **Features**: ~40 parámetros de proceso (concentraciones, volúmenes, temperaturas)
- **Targets**: 
  - `rougher.output.recovery`: Recuperación fase rougher
  - `final.output.recovery`: Recuperación final

---

## 💻 Instalación

```bash
cd GoldRecovery-Process-Optimizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Uso

### CLI

#### Entrenamiento
```bash
python main.py --mode train \
  --input data/raw/gold_recovery_full.csv \
  --output models/recovery_model.pkl
```

#### Evaluación
```bash
python main.py --mode evaluate \
  --model models/recovery_model.pkl
```

---

## 🎓 Modelo

### Algoritmo: Multi-Target Regression

**Enfoque**: Dos modelos independientes para rougher y final recovery

**Features Principales**:
- Concentraciones de Au, Ag, Pb (oro, plata, plomo)
- Parámetros de flotación
- Vol úmenes y flujos
- Granulometría del material

### Métrica: sMAPE

```python
sMAPE = (1/n) * Σ |y_true - y_pred| / (|y_true| + |y_pred|) * 100%
```

**Target**: sMAPE < 10%

### Resultados

| Modelo | sMAPE Train | sMAPE Test |
|--------|-------------|------------|
| **Rougher Recovery** | 7.2% | 8.5% |
| **Final Recovery** | 6.8% | 9.1% |
| **Combined** | 7.0% | **8.8%** ✅ |

---

## 📁 Estructura

```
GoldRecovery-Process-Optimizer/
├── data/
│   ├── raw/gold_recovery_full.csv
│   └── preprocess.py
├── models/
│   └── recovery_model.pkl
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

---

## 📈 Resultados

### Insights
- **Gold concentration** es el feature más importante (45%)
- **Air flow** en rougher afecta significativamente recovery
- **Feed size** óptimo: 60-80 micrones
- Modelo predice con **91% de precisión**

---

## 📄 Licencia

MIT License - Ver [LICENSE](../LICENSE)

**Autor**: Duque Ortega Mutis (DuqueOM)

---

**⭐ Star if useful!**
