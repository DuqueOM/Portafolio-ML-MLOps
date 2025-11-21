# 🚀 Quick Start Guide - Portfolio ML/MLOps Mejorado

**Guía rápida para ejecutar el plan de mejora del portafolio**

---

## 📋 Fase Actual: Setup Completado

Se han creado todos los scripts y configuraciones necesarias para mejorar el portafolio según el plan de trabajo.

### ✅ Archivos Creados (17 archivos nuevos)

```
reports/
├── initial-scan.md                  # Reporte maestro de auditoría
├── IMPLEMENTATION_STATUS.md         # Estado detallado de implementación
├── run_tests_all_projects.sh        # Tests automatizados
├── install_security_tools.sh        # Instalador de herramientas
├── run_security_scan.sh             # Security scanner
├── setup_dvc.sh                     # Configurador DVC
└── setup_git_lfs.sh                 # Configurador Git LFS

scripts/
├── fetch_data.py                    # Data fetcher con checksums
└── run_e2e.sh                       # Pipeline E2E

.github/workflows/
└── ci-mlops.yml                     # CI/CD mejorado (6 jobs)

docker-compose.mlflow.yml            # MLflow tracking stack
QUICK_START_GUIDE.md                 # Esta guía
```

---

## 🎯 Ejecutar el Plan - Prioridad ALTA

### Opción 1: Ejecución Automática (Recomendado)

```bash
cd "/home/duque_om/projects/Projects Tripe Ten"

# 1. Instalar herramientas de seguridad
bash reports/install_security_tools.sh

# 2. Ejecutar tests en todos los proyectos
bash reports/run_tests_all_projects.sh

# 3. Ejecutar security scans
bash reports/run_security_scan.sh

# 4. Configurar DVC
bash reports/setup_dvc.sh

# 5. Configurar Git LFS
bash reports/setup_git_lfs.sh

# 6. Revisar reporte final
cat reports/initial-scan.md
```

### Opción 2: Paso a Paso Manual

#### Paso B1: ✅ COMPLETADO
Estructura de reports/ y scripts creada.

#### Paso B2: Ejecutar Tests

```bash
cd BankChurn-Predictor

# Si el venv ya existe y está completo:
source .venv/bin/activate
pytest --maxfail=1 --disable-warnings -q --cov=src 2>&1 | tee ../reports/BankChurn-pytest.txt
coverage run -m pytest && coverage report -m | tee ../reports/BankChurn-coverage.txt

# Para todos los proyectos:
cd ..
bash reports/run_tests_all_projects.sh
```

**Meta**: Coverage ≥70% en todos los proyectos

#### Paso B3: Gitleaks Secret Scan

```bash
# Instalar gitleaks
bash reports/install_security_tools.sh

# Ejecutar scan
gitleaks detect --source . --report-path reports/gitleaks-report.json

# Ver resultados
cat reports/gitleaks-report.json
```

**Si hay secretos**: Limpiar con BFG Repo-Cleaner o git-filter-repo

#### Paso B4: DVC + MLflow

```bash
# Configurar DVC
bash reports/setup_dvc.sh

# Iniciar MLflow stack
docker-compose -f docker-compose.mlflow.yml up -d

# Verificar
docker ps
curl http://localhost:5000/health

# Acceder a UIs:
# - MLflow: http://localhost:5000
# - MinIO: http://localhost:9001
```

#### Paso B5: Git LFS

```bash
# Configurar Git LFS
bash reports/setup_git_lfs.sh

# Verificar archivos trackeados
git lfs ls-files
```

#### Paso B6: Trivy Container Scan

```bash
# Ya incluido en run_security_scan.sh
# O manualmente:
cd BankChurn-Predictor
docker build -t ml-portfolio-bankchurn:latest .
trivy image --severity HIGH,CRITICAL ml-portfolio-bankchurn:latest | tee ../reports/BankChurn-trivy.txt
```

#### Paso B7: Actualizar Reporte

Todos los scripts ya actualizan `reports/initial-scan.md` automáticamente.

---

## 🔬 Prioridad MEDIA - MLOps Improvements

### E2E Reproducible

```bash
# Ejecutar pipeline completo en BankChurn
bash scripts/run_e2e.sh

# Incluye: data → train → register → serve → inference
```

### CI con Matrix de Python

```bash
# El workflow ya está configurado en:
.github/workflows/ci-mlops.yml

# Push a GitHub para ejecutar:
git add .
git commit -m "Add MLOps improvements"
git push origin main
```

### Model Cards

Crear `model_card.md` en cada proyecto:

```bash
# Ejemplo para BankChurn
cat > BankChurn-Predictor/model_card.md << 'EOF'
# Model Card - BankChurn Predictor

## Model Details
- **Name**: BankChurn Ensemble Classifier
- **Version**: v1.0.0
- **Date**: 2025-11-21
- **Type**: Ensemble (LogisticRegression + RandomForest)

## Training Data
- **Dataset**: Churn_Modelling.csv
- **Size**: 10,000 registros
- **Features**: 14
- **DVC**: `data/Churn_Modelling.csv.dvc`

## Metrics
- **AUC-ROC**: 0.867
- **F1-Score**: 0.637
- **Precision**: 0.65
- **Recall**: 0.63

## Reproducibility
```bash
dvc pull
python src/bankchurn/training.py --config configs/train.yaml
```

## Limitations
- Datos sintéticos (no producción)
- Posible sesgo demográfico
- Requiere re-training periódico

## Ethical Considerations
- No usar para decisiones discriminatorias
- Revisar impacto en grupos protegidos
EOF
```

---

## 📊 Verificar Progreso

### Dashboard de Status

```bash
# Ver reporte completo
cat reports/IMPLEMENTATION_STATUS.md

# Ver resumen de coverage
cat reports/coverage-summary.csv

# Ver issues de seguridad
cat reports/gitleaks-report.json
jq '.[] | {file: .File, secret: .Secret}' reports/gitleaks-report.json
```

### Métricas Esperadas

```bash
# Coverage por proyecto
echo "=== Coverage Summary ==="
grep -E "(BankChurn|CarVision|TelecomAI)" reports/coverage-summary.csv

# Seguridad
echo "=== Security Status ==="
echo "Secretos detectados: $(jq length reports/gitleaks-report.json 2>/dev/null || echo 0)"
echo "Vulnerabilidades: Ver reports/*-trivy.txt"
```

---

## 🐛 Troubleshooting

### Error: "gitleaks command not found"

```bash
bash reports/install_security_tools.sh
```

### Error: "DVC not configured"

```bash
bash reports/setup_dvc.sh
```

### Error: "pytest no encontrado"

```bash
cd <proyecto>
source .venv/bin/activate
pip install pytest pytest-cov
```

### Error: "Docker no responde"

```bash
sudo systemctl start docker
docker ps
```

### Error: "MLflow no inicia"

```bash
# Verificar logs
docker-compose -f docker-compose.mlflow.yml logs mlflow

# Reiniciar
docker-compose -f docker-compose.mlflow.yml down
docker-compose -f docker-compose.mlflow.yml up -d
```

---

## 📝 Checklist de Validación

Antes de considerar el trabajo completado:

- [ ] **Tests**: Todos los proyectos con coverage ≥70%
- [ ] **Security**: Gitleaks sin secretos detectados
- [ ] **Security**: Trivy sin vulnerabilidades CRITICAL
- [ ] **DVC**: Configurado y funcional (`dvc status`)
- [ ] **Git LFS**: Configurado (`.gitattributes` creado)
- [ ] **MLflow**: Stack funcionando (http://localhost:5000)
- [ ] **E2E**: Pipeline ejecutado exitosamente
- [ ] **CI/CD**: Workflow validado en GitHub
- [ ] **Docs**: Model cards creados para proyectos principales
- [ ] **Reports**: `initial-scan.md` actualizado con datos reales

---

## 🎓 Recursos Adicionales

### Documentación Creada

1. `reports/initial-scan.md` - Reporte maestro de auditoría
2. `reports/IMPLEMENTATION_STATUS.md` - Status detallado
3. `QUICK_START_GUIDE.md` - Esta guía

### Scripts Disponibles

1. `reports/run_tests_all_projects.sh` - Testing automatizado
2. `reports/run_security_scan.sh` - Security scanning
3. `scripts/run_e2e.sh` - Pipeline E2E
4. `scripts/fetch_data.py` - Data management

### Configuraciones

1. `docker-compose.mlflow.yml` - MLflow tracking
2. `.github/workflows/ci-mlops.yml` - CI/CD mejorado
3. `.gitattributes` - Git LFS (se crea con setup)
4. `.dvc/config` - DVC remote (se crea con setup)

---

## 🚀 Siguiente Fase

Una vez completada esta fase (Prioridad ALTA):

### Prioridad MEDIA
1. Implementar model cards en todos los proyectos
2. Configurar CI matrix con múltiples versiones de Python
3. Añadir job E2E en CI
4. Crear datasets sintéticos para demos

### Prioridad BAJA
1. Parametrizar notebooks con Papermill
2. Publicar imágenes en GHCR
3. Deploy demos en Render/Heroku
4. Grabar videos demostrativos

---

## 💡 Consejos

1. **Ejecuta los scripts en orden** para evitar dependencias faltantes
2. **Revisa los logs** en `reports/` después de cada ejecución
3. **Commit frecuentemente** para no perder progreso
4. **Documenta problemas** en issues de GitHub
5. **Actualiza** `initial-scan.md` con cada cambio importante

---

## 📞 Soporte

Si encuentras problemas:

1. Revisar logs en `reports/`
2. Verificar documentación en scripts (comentarios al inicio)
3. Consultar `reports/IMPLEMENTATION_STATUS.md` para debugging

---

**¡Listo para ejecutar! 🚀**

```bash
# Comando todo-en-uno (después de instalar tools)
bash reports/run_tests_all_projects.sh && \
bash reports/run_security_scan.sh && \
echo "✅ Auditoría completada. Ver: reports/initial-scan.md"
```
