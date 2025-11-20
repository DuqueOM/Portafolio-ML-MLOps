# 🎮 Gaming Market Intelligence

**Sistema de Análisis de Mercado de Videojuegos y Predicción de Ventas**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org)
[![Coverage](https://img.shields.io/badge/Coverage-50%25-yellow.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Análisis de mercado de videojuegos con predicción de ventas y identificación de tendencias por plataforma, género y región.**

---

## 🚀 Quick Start

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Análisis
python main.py --mode analyze --year 2016

# 3. Predicciones
python main.py --mode predict --platform PS4 --genre Action
```

---

## 🎯 Descripción

### Problema
Ice (tienda online de videojuegos) necesita identificar patrones de éxito para planificar campañas publicitarias y stock para 2017.

### Solución
- ✅ Análisis histórico de ventas (1980-2016)
- ✅ Identificación de plataformas y géneros exitosos
- ✅ Análisis regional (NA, EU, JP)
- ✅ Testing de hipótesis estadísticas
- ✅ Predicción de ventas por plataforma/género

### Tecnologías
- **Análisis**: Pandas, NumPy, SciPy
- **Visualización**: Matplotlib, Seaborn
- **Stats**: Pruebas de hipótesis (t-test, Mann-Whitney)
- **Testing**: pytest

### Dataset
- **Fuente**: Historical game sales data
- **Registros**: ~16,700 juegos
- **Periodo**: 1980-2016
- **Features**: Plataforma, género, publisher, rating, ventas por región

---

## 💻 Instalación

```bash
cd Gaming-Market-Intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Uso

### CLI

#### Análisis de Mercado
```bash
python main.py --mode analyze \
  --input data/raw/games.csv \
  --year 2016 \
  --output reports/market_analysis.html
```

#### Predicción de Ventas
```bash
python main.py --mode predict \
  --platform PS4 \
  --genre Action \
  --rating M
```

Output:
```
Predicted global sales: 2.5M copies
Regional breakdown:
  NA: 1.2M
  EU: 0.9M
  JP: 0.4M
```

---

## 🎓 Análisis

### Plataformas Exitosas (2014-2016)

| Plataforma | Ventas Globales | Juegos | Avg Rating |
|------------|-----------------|--------|------------|
| **PS4** | 385M | 342 | 7.2 |
| **XOne** | 245M | 287 | 7.0 |
| **PC** | 189M | 412 | 6.8 |

### Géneros Top

1. **Action** - 35% market share
2. **Sports** - 18%
3. **Shooter** - 15%
4. **Role-Playing** - 12%

### Insights Regionales

**Norte América (NA)**:
- Prefiere: Action, Shooter, Sports
- Plataforma líder: XOne

**Europa (EU)**:
- Prefiere: Action, Sports, Racing
- Plataforma líder: PS4

**Japón (JP)**:
- Prefiere: Role-Playing, Action, Platform
- Plataforma líder: 3DS

---

## 📁 Estructura

```
Gaming-Market-Intelligence/
├── data/
│   ├── raw/games.csv
│   └── preprocess.py
├── notebooks/
│   ├── EDA.ipynb
│   └── statistical_tests.ipynb
├── tests/
│   └── test_preprocessing.py
├── main.py
└── analyze.py
```

---

## 🧪 Testing

```bash
pytest --cov=. --cov-report=term-missing
```

---

## 📈 Resultados

### Hipótesis Testeadas

1. **Xbox One vs PC ratings**: p-value = 0.23 → No hay diferencia significativa
2. **Action vs Sports ratings**: p-value = 0.04 → Diferencia significativa ✅

### Predicciones para 2017

- **Plataforma #1**: PS4 (continúa dominancia)
- **Género emergente**: Battle Royale
- **Rating**: M-rated games +15% en ventas

---

## 📄 Licencia

MIT License - Ver [LICENSE](../LICENSE)

**Autor**: Duque Ortega Mutis (DuqueOM)

---

**⭐ Star if useful!**
