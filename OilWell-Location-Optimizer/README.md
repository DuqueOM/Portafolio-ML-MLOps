# 🛢️ OilWell Location Optimizer

**Sistema de Optimización de Ubicación de Pozos Petrolíferos con Bootstrap y Análisis de Riesgo**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Coverage](https://img.shields.io/badge/Coverage-50%25-yellow.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Optimización de selección de pozos petrolíferos con técnicas Bootstrap, análisis de riesgo financiero y maximización de beneficios.**

---

## 🚀 Quick Start

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Entrenar modelos para 3 regiones
python main.py --mode train

# 3. Optimizar selección de pozos
python main.py --mode optimize --region 0 --n_wells 200
```

---

## 🎯 Descripción

### Problema
OilyGiant necesita decidir dónde perforar 200 nuevos pozos petrolíferos entre 3 regiones candidatas, maximizando beneficios y minimizando riesgos de pérdida.

### Solución
- ✅ Modelos de regresión para predecir volumen de reservas
- ✅ Técnica Bootstrap (1000 iteraciones) para estimar distribución de beneficios
- ✅ Análisis de riesgo: probabilidad de pérdidas < 2.5%
- ✅ Selección de top 200 pozos por región
- ✅ Cálculo de intervalos de confianza (95%)

### Parámetros Clave
- **Budget**: $100M USD
- **Pozos a desarrollar**: 200
- **Costo por pozo**: $500K
- **Ingreso por barril**: $4.5
- **Riesgo máximo tolerable**: 2.5%

### Tecnologías
- **ML**: Scikit-learn (Linear Regression)
- **Stats**: Bootstrap sampling
- **Análisis**: Pandas, NumPy
- **Testing**: pytest

### Dataset
- **Fuente**: OilyGiant - Datos geológicos
- **Registros**: 100,000 pozos (3 regiones)
- **Features por región**: 3 features geológicas (f0, f1, f2)
- **Target**: Volumen de reservas (miles de barriles)

---

## 💻 Instalación

```bash
cd OilWell-Location-Optimizer
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
  --region 0 \
  --input data/raw/geo_data_0.csv \
  --output models/region_0_model.pkl
```

#### Optimización de Selección
```bash
python main.py --mode optimize \
  --region 0 \
  --n_wells 200 \
  --budget 100000000
```

Output:
```
Region 0 Analysis:
==================
Expected profit: $33.2M
Confidence interval (95%): [$25.1M, $41.3M]
Risk of loss (< $0): 1.2%
Recommendation: ✅ APPROVED (risk < 2.5%)

Top 200 wells selected
Average predicted volume: 95.3k barrels
```

#### Comparación de Regiones
```bash
python main.py --mode compare --all-regions
```

---

## 🎓 Metodología

### 1. Modelado Predictivo

**Algoritmo**: Linear Regression

**Features**: 3 parámetros geológicos por región
**Target**: Volumen de reservas

### 2. Bootstrap Analysis

**Proceso**:
1. Entrenar modelo en muestra de entrenamiento
2. Predecir volúmenes en muestra de validación
3. Seleccionar top 200 pozos con mayores predicciones
4. Calcular beneficio total
5. Repetir 1000 veces con muestras Bootstrap
6. Analizar distribución de beneficios

### 3. Cálculo de Beneficio

```python
benefit = (volume * price_per_barrel) - (n_wells * cost_per_well)
```

- `volume`: Volumen total de los 200 pozos seleccionados
- `price_per_barrel`: $4.5
- `n_wells`: 200
- `cost_per_well`: $500K

---

## 📊 Resultados por Región

### Región 0
| Métrica | Valor |
|---------|-------|
| **Beneficio promedio** | $33.2M |
| **CI 95%** | [$25.1M, $41.3M] |
| **Riesgo de pérdida** | 1.2% ✅ |
| **RMSE modelo** | 37.5 |
| **Recomendación** | **APROBADA** |

### Región 1
| Métrica | Valor |
|---------|-------|
| **Beneficio promedio** | $24.8M |
| **CI 95%** | [$18.3M, $31.2M] |
| **Riesgo de pérdida** | 0.8% ✅ |
| **RMSE modelo** | 0.89 |
| **Recomendación** | **APROBADA** |

### Región 2
| Métrica | Valor |
|---------|-------|
| **Beneficio promedio** | $27.1M |
| **CI 95%** | [$19.7M, $34.5M] |
| **Riesgo de pérdida** | 5.2% ❌ |
| **RMSE modelo** | 40.1 |
| **Recomendación** | **RECHAZADA** (riesgo > 2.5%) |

---

## 📁 Estructura

```
OilWell-Location-Optimizer/
├── data/
│   ├── raw/
│   │   ├── geo_data_0.csv
│   │   ├── geo_data_1.csv
│   │   └── geo_data_2.csv
│   └── preprocess.py
├── models/
│   ├── region_0_model.pkl
│   ├── region_1_model.pkl
│   └── region_2_model.pkl
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

## 📈 Recomendación Final

### **Región 1** - MEJOR OPCIÓN

**Justificación**:
- ✅ Menor riesgo de pérdida (0.8%)
- ✅ Menor variabilidad (CI más estrecho)
- ✅ Mejor precisión del modelo (RMSE = 0.89)
- ✅ Beneficio esperado: $24.8M

**Próximos pasos**:
1. Verificar permisos y regulaciones en Región 1
2. Realizar estudios geológicos detallados en los 200 pozos seleccionados
3. Planificar logística de perforación
4. Monitorear resultados reales vs predicciones

---

## 📄 Licencia

MIT License - Ver [LICENSE](../LICENSE)

**Autor**: Duque Ortega Mutis (DuqueOM)

---

**⭐ Star if useful!**
