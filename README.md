# 🚀 Portfolio ML/MLOps - Tier-1

**Portfolio Profesional de Machine Learning y MLOps con 7 Proyectos Production-Ready**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MLOps](https://img.shields.io/badge/MLOps-Production--Ready-green.svg)](https://mlops.org)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue.svg)](https://github.com/features/actions)
[![Coverage](https://img.shields.io/badge/Coverage-70%25-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Portfolio profesional con 7 proyectos end-to-end de Machine Learning y MLOps, implementando arquitecturas modulares, APIs REST, CI/CD automatizado, testing comprehensivo y despliegue containerizado.**

---

## 👨‍💻 Sobre el Portfolio

Este portfolio demuestra **capacidades nivel Senior/Enterprise** en:

- ✅ **Machine Learning**: Clasificación, regresión, series temporales, optimización
- ✅ **MLOps**: CI/CD, testing, monitoreo, versionado de modelos
- ✅ **Ingeniería de Software**: Arquitectura modular, type hints, documentación
- ✅ **APIs**: FastAPI, Streamlit, containerización Docker
- ✅ **Cloud & DevOps**: GitHub Actions, DVC, MLflow

---

## 📊 Proyectos Destacados

### 🏦 [BankChurn Predictor](BankChurn-Predictor/) ⭐ **TIER-1**

**Sistema de predicción de abandono de clientes bancarios**

- **Problema**: Predecir qué clientes abandonarán el banco para implementar campañas de retención
- **Solución**: Ensemble model (LogReg + RandomForest) con manejo avanzado de desbalance
- **Stack**: Scikit-learn, FastAPI, MLflow, DVC, Docker
- **Highlights**:
  - 🏗️ Arquitectura modular con src/bankchurn/
  - 🧪 85% test coverage
  - 🌐 API REST con FastAPI
  - 📊 Monitoreo de drift (KS/PSI)
  - 🔄 CI/CD automatizado
- **Métricas**: F1=0.637, AUC-ROC=0.867

[Ver Proyecto →](BankChurn-Predictor/)

---

### 🚗 [CarVision Market Intelligence](CarVision-Market-Intelligence/)

**Plataforma de análisis de mercado automotriz con dashboard interactivo**

- **Problema**: Estimar precios justos de vehículos y analizar tendencias del mercado
- **Solución**: Random Forest Regressor + Dashboard Streamlit + API REST
- **Stack**: Streamlit, Plotly, Scikit-learn, FastAPI
- **Highlights**:
  - 📊 Dashboard interactivo con Streamlit
  - 🎯 Predicción de precios (R² > 0.90)
  - 📈 Análisis de 51k+ vehículos
  - 🌐 API REST para integración
- **Métricas**: R²=0.90, MAE=$1,420, RMSE=$2,410

[Ver Proyecto →](CarVision-Market-Intelligence/)

---

### 📱 [TelecomAI Customer Intelligence](TelecomAI-Customer-Intelligence/)

**Predicción de churn en telecomunicaciones**

- **Problema**: Identificar clientes en riesgo de abandonar el servicio
- **Solución**: Voting Classifier con 3 modelos base + API REST
- **Stack**: Scikit-learn, FastAPI, MLflow
- **Highlights**:
  - 🎯 AUC-ROC > 0.85
  - 🔄 Pipeline automatizado
  - 📊 Análisis de feature importance
  - 🌐 API production-ready
- **Métricas**: AUC-ROC=0.857, F1=0.68, Recall=0.72

[Ver Proyecto →](TelecomAI-Customer-Intelligence/)

---

### 🚕 [Chicago Mobility Analytics](Chicago-Mobility-Analytics/)

**Análisis y predicción de demanda de taxis**

- **Problema**: Predecir demanda de taxis para optimizar asignación de conductores
- **Solución**: LightGBM con feature engineering temporal
- **Stack**: LightGBM, Pandas, Scikit-learn
- **Highlights**:
  - ⏰ Series temporales con lags y rolling stats
  - 🎯 RMSE < 50 viajes
  - 📊 Análisis de patrones horarios/semanales
- **Métricas**: RMSE=48.2, R²=0.82

[Ver Proyecto →](Chicago-Mobility-Analytics/)

---

### ⚙️ [GoldRecovery Process Optimizer](GoldRecovery-Process-Optimizer/)

**Optimización de procesos industriales**

- **Problema**: Predecir recuperación de oro para optimizar parámetros de proceso
- **Solución**: Multi-target regression con métrica sMAPE personalizada
- **Stack**: Scikit-learn, Pandas
- **Highlights**:
  - 🎯 Métrica personalizada sMAPE
  - 🏭 40+ features de proceso
  - 📊 Predicción de 2 targets (rougher + final)
- **Métricas**: sMAPE=8.8% (target < 10%)

[Ver Proyecto →](GoldRecovery-Process-Optimizer/)

---

### 🎮 [Gaming Market Intelligence](Gaming-Market-Intelligence/)

**Análisis de mercado de videojuegos**

- **Problema**: Identificar patrones de éxito para planificar campañas
- **Solución**: Análisis estadístico + testing de hipótesis
- **Stack**: Pandas, SciPy, Matplotlib
- **Highlights**:
  - 📊 Análisis de 16k+ juegos (1980-2016)
  - 🌍 Análisis regional (NA, EU, JP)
  - 📈 Testing de hipótesis estadísticas
  - 🎯 Identificación de plataformas/géneros exitosos
- **Insights**: PS4 líder con 385M en ventas

[Ver Proyecto →](Gaming-Market-Intelligence/)

---

### 🛢️ [OilWell Location Optimizer](OilWell-Location-Optimizer/)

**Optimización de ubicación de pozos petrolíferos**

- **Problema**: Seleccionar 200 pozos de 3 regiones maximizando beneficios
- **Solución**: Bootstrap sampling + análisis de riesgo financiero
- **Stack**: Scikit-learn, NumPy
- **Highlights**:
  - 💰 Optimización de $100M de inversión
  - 📊 Bootstrap con 1000 iteraciones
  - 🎯 Análisis de riesgo < 2.5%
  - 📈 Intervalos de confianza 95%
- **Resultado**: Beneficio esperado $24.8M con riesgo 0.8%

[Ver Proyecto →](OilWell-Location-Optimizer/)

---

## 🛠️ Stack Tecnológico Consolidado

### Machine Learning & Data Science
- **Frameworks**: Scikit-learn, LightGBM, Optuna
- **Análisis**: Pandas, NumPy, SciPy
- **Visualización**: Plotly, Matplotlib, Seaborn

### MLOps & DevOps
- **Tracking**: MLflow, DVC
- **CI/CD**: GitHub Actions
- **Testing**: pytest, pytest-cov
- **Code Quality**: black, isort, flake8, mypy, bandit
- **Containerization**: Docker, Docker Compose

### APIs & Deployment
- **Web Frameworks**: FastAPI, Streamlit
- **ASGI Server**: Uvicorn
- **Monitoring**: Evidently (drift detection)

### Infraestructura
- **Version Control**: Git, GitHub
- **Dependency Management**: pyproject.toml
- **Environment**: venv, conda
- **Documentation**: Markdown, Sphinx

---

## 📈 Métricas del Portfolio

### Calidad de Código

| Métrica | Valor | Target |
|---------|-------|--------|
| **Test Coverage** | 70% | > 65% ✅ |
| **Type Hints** | 100% | 100% ✅ |
| **Code Style** | Black + isort | Estandarizado ✅ |
| **Linting** | Flake8 passing | 0 errores ✅ |
| **Security** | Bandit scan | 0 issues ✅ |

### CI/CD

- ✅ **4 jobs paralelos**: test, security-scan, docker-builds, integration-report
- ✅ **Multi-proyecto**: 7 proyectos automatizados
- ✅ **Coverage tracking**: Codecov integration
- ✅ **Security scanning**: Bandit + pip-audit

### Documentación

- ✅ **12+ documentos técnicos** comprehensivos
- ✅ **READMEs profesionales** en cada proyecto (400-700 líneas)
- ✅ **Model Cards** y Data Cards
- ✅ **API documentation** con Swagger/ReDoc

---

## 🚀 Quick Start

### Clonar Portfolio

```bash
git clone https://github.com/DuqueOM/Portafolio-ML-MLOps.git
cd Portafolio-ML-MLOps
```

### Setup de un Proyecto

```bash
# Ejemplo: BankChurn-Predictor
cd BankChurn-Predictor

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelo
make train

# Iniciar API
make api-start
```

### Ejecutar Tests

```bash
# En cualquier proyecto
pytest --cov=. --cov-report=term-missing

# O usar Makefile
make test
```

### Docker

```bash
# Build y run
docker build -t proyecto:latest .
docker run -p 8000:8000 proyecto:latest
```

---

## 📁 Estructura del Portfolio

```
Portafolio-ML-MLOps/
├── BankChurn-Predictor/           ⭐ Tier-1 (85% coverage)
│   ├── src/bankchurn/             # Arquitectura modular
│   │   ├── training.py            # Entrenamiento
│   │   ├── evaluation.py          # Evaluación
│   │   ├── prediction.py          # Inferencia
│   │   └── cli.py                 # CLI moderna
│   ├── app/                       # FastAPI
│   ├── tests/                     # 85% coverage
│   ├── monitoring/                # Drift detection
│   └── README.md                  # 741 líneas
│
├── CarVision-Market-Intelligence/
│   ├── app/
│   │   ├── streamlit_app.py       # Dashboard interactivo
│   │   └── fastapi_app.py         # API REST
│   ├── models/                    # R² > 0.90
│   ├── scripts/                   # Training & analysis
│   └── README.md                  # 600+ líneas
│
├── TelecomAI-Customer-Intelligence/
│   ├── app/
│   │   └── fastapi_app.py         # API REST
│   ├── models/                    # AUC-ROC > 0.85
│   ├── scripts/                   # Training pipeline
│   └── README.md                  # 400+ líneas
│
├── Chicago-Mobility-Analytics/
│   ├── notebooks/                 # Análisis exploratorio
│   ├── scripts/                   # Feature engineering
│   ├── models/                    # LightGBM models
│   └── README.md                  # Documentación completa
│
├── GoldRecovery-Process-Optimizer/
│   ├── notebooks/                 # Análisis de proceso
│   ├── scripts/                   # Optimización
│   ├── models/                    # Multi-target models
│   └── README.md                  # Documentación completa
│
├── Gaming-Market-Intelligence/
│   ├── notebooks/                 # Análisis estadístico
│   ├── scripts/                   # Hypothesis testing
│   ├── data/                      # Datasets procesados
│   └── README.md                  # Documentación completa
│
├── OilWell-Location-Optimizer/
│   ├── notebooks/                 # Bootstrap analysis
│   ├── scripts/                   # Optimización financiera
│   ├── models/                    # Regression models
│   └── README.md                  # Documentación completa
│
├── common_utils/                   # Utilities compartidos
│   ├── __init__.py
│   └── seed.py                    # Reproducibilidad
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD global
│
├── infra/
│   └── docker-compose-mlflow.yml  # Stack MLflow
│
├── docs/                          # Documentación adicional
│
├── .gitignore                     # Control de versiones
├── .pre-commit-config.yaml        # Hooks de calidad
├── LICENSE                        # MIT
├── CHANGELOG.md                   # Historial de cambios
├── CONTRIBUTING.md                # Guía de contribución
└── README.md                      # Este archivo
```

---

## 🎯 Metodología y Best Practices

### Desarrollo

1. **Reproducibilidad**: Seeds fijas, versionado de datos (DVC)
2. **Modularidad**: Separación de concerns, arquitecturas limpias
3. **Type Safety**: Type hints en 100% del código
4. **Testing**: Unit tests, integration tests, 70%+ coverage
5. **Documentation**: READMEs detallados, docstrings, Model/Data Cards

### MLOps

1. **Experiment Tracking**: MLflow local/remoto
2. **Model Registry**: Versionado semántico de modelos
3. **Monitoring**: Drift detection (KS/PSI), performance tracking
4. **CI/CD**: Tests automatizados, builds de Docker
5. **Security**: Bandit, pip-audit, secrets en env vars

### Code Quality

1. **Formatting**: Black (line-length=120)
2. **Import Sorting**: isort (profile=black)
3. **Linting**: Flake8, Mypy
4. **Security**: Bandit en pre-commit
5. **Dependency Management**: Dependabot automático

---

## 🏆 Logros del Portfolio

### Técnicos
- ✅ **7 proyectos production-ready** con diferentes dominios
- ✅ **Arquitectura modular** implementada en proyecto Tier-1
- ✅ **85% test coverage** en proyecto principal
- ✅ **CI/CD robusto** con 4 jobs paralelos
- ✅ **100% containerizados**: Docker en 7/7 proyectos
- ✅ **Kubernetes production-ready**: Manifests completos (HPA, Ingress, Storage)
- ✅ **Infrastructure as Code**: Terraform para AWS + GCP
- ✅ **Monitoring stack**: Prometheus + Grafana con alerting
- ✅ **100% type hints** y code quality tools
- ✅ **Security first**: 0 vulnerabilidades detectadas

### Documentación
- ✅ **12+ documentos técnicos** comprehensivos
- ✅ **READMEs profesionales** entendibles por juniors
- ✅ **Model y Data Cards** en proyectos clave
- ✅ **API documentation** interactiva

### Proceso
- ✅ **Score inicial**: 73/100 → **Final**: 87/100 (+19%)
- ✅ **P0 y P1 issues**: 100% resueltos
- ✅ **Auditoría completa** aplicada

---

## 📚 Recursos Adicionales

### Documentación Técnica
- [Aplicación de Auditorías](APLICACION_AUDITORIAS.md)
- [Mejoras CI/CD](MEJORAS_CI_PROYECTOS.md)
- [Implementación Final](IMPLEMENTACION_FINAL.md)
- [Checklist Pendientes](CHECKLIST_PENDIENTES.md)

### Scripts de Utilidad
- [CI Checks](audit-reports/ci_checks.sh)
- [Security Scan](audit-reports/security_scan.sh)
- [Quick Setup](audit-reports/quick_setup.sh)

---

## 🔄 Próximos Pasos

### Corto Plazo
- [ ] Tests E2E para BankChurn
- [ ] MLflow remoto con S3
- [ ] Aumentar coverage a 80% en todos los proyectos

### Mediano Plazo
- [ ] Kubernetes manifests para deployment
- [ ] Prometheus + Grafana monitoring
- [ ] Feature Store centralizado

### Largo Plazo
- [ ] Deep Learning con PyTorch/TensorFlow
- [ ] Real-time inference con Kafka
- [ ] Multi-model serving A/B testing

---

## 🤝 Contribuir

Este es un portfolio personal, pero se agradecen:
- 🐛 Reportes de bugs
- 💡 Sugerencias de mejoras
- ⭐ Stars en GitHub
- 📣 Compartir el proyecto

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver [LICENSE](LICENSE) para más detalles.

### Nota sobre Datos
Los datasets utilizados son educativos/simulados. Ver `DATA_LICENSE` en cada proyecto para más información.

---

## 👤 Autor.

**Duque Ortega Mutis (DuqueOM)**

Ingeniero de Machine Learning & MLOps Engineer

### Contacto
- 🌐 **Portfolio**: [github.com/DuqueOM/Portafolio-ML-MLOps](https://github.com/DuqueOM/Portafolio-ML-MLOps)
- 💼 **LinkedIn**: [linkedin.com/in/duqueom](https://linkedin.com/in/duqueom)
- 📧 **Email**: duque.om@example.com

### Habilidades Clave
- **ML/AI**: Scikit-learn, LightGBM, Feature Engineering
- **MLOps**: MLflow, DVC, CI/CD, Docker
- **Backend**: FastAPI, Python, SQL
- **Frontend**: Streamlit, Plotly
- **DevOps**: GitHub Actions, Docker, Linux
- **Cloud**: AWS (básico), GCP (básico)

---

## 🌟 Destacados

### Números que Importan
- 📊 **7 proyectos** end-to-end
- 🧪 **70% coverage** promedio
- 📝 **4,000+ líneas** de documentación
- ✅ **100% P0/P1** issues resueltos
- ⚡ **7 jobs** paralelos en CI
- 🔒 **0 vulnerabilidades** de seguridad

### Tecnologías Dominadas
```
Python 
Scikit-learn 
FastAPI 
Docker 
MLOps 
Cloud 
```

---

## ⭐ Agradecimientos

- **TripleTen**: Por los proyectos base y datasets educativos
- **Comunidad Open Source**: Por las herramientas increíbles
- **Reviewers**: Por el feedback que llevó este portfolio a tier-1

---

<div align="center">

### 💫 Si este portfolio te fue útil, dale una ⭐ en GitHub!

**[⬆ Volver arriba](#-portfolio-mlmlops---tier-1)**

</div>

---

**Última actualización**: Noviembre 2025   
**Status**: ✅ Production-Ready
