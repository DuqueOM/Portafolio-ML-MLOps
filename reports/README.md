# 📋 Reports Directory - Portfolio ML/MLOps

Directorio central para todos los reportes de auditoría, tests, seguridad y análisis del portafolio.

---

## 📁 Estructura de Archivos

### Documentación Principal

| Archivo | Descripción |
|---------|-------------|
| `initial-scan.md` | **Reporte maestro** de auditoría inicial |
| `IMPLEMENTATION_STATUS.md` | Estado detallado de implementación |
| `README.md` | Este archivo |

### Scripts de Automatización

| Script | Propósito | Comando |
|--------|-----------|---------|
| `run_tests_all_projects.sh` | Ejecutar tests en todos los proyectos | `bash run_tests_all_projects.sh` |
| `install_security_tools.sh` | Instalar gitleaks, trivy, DVC, Git LFS | `bash install_security_tools.sh` |
| `run_security_scan.sh` | Ejecutar scans de seguridad completos | `bash run_security_scan.sh` |
| `setup_dvc.sh` | Configurar DVC y remote storage | `bash setup_dvc.sh` |
| `setup_git_lfs.sh` | Configurar Git LFS para modelos | `bash setup_git_lfs.sh` |

### Reportes Generados (automáticos)

Estos archivos se generan automáticamente al ejecutar los scripts:

#### Tests & Coverage
- `<proyecto>-pytest.txt` - Resultados de pytest
- `<proyecto>-coverage.txt` - Reporte de coverage detallado
- `coverage-summary.csv` - Resumen de coverage de todos los proyectos

#### Security
- `gitleaks-report.json` - Secretos detectados
- `gitleaks-output.txt` - Output legible de gitleaks
- `<proyecto>-trivy.txt` - Vulnerabilidades de contenedores
- `<proyecto>-trivy.json` - Vulnerabilidades en formato JSON

#### Installation Logs
- `<proyecto>-install.log` - Logs de instalación de dependencias
- `BankChurn-install-log.txt` - Log específico de BankChurn
- `BankChurn-install-clean.log` - Log de instalación limpia

#### Docker
- `<proyecto>-docker-build.log` - Logs de Docker builds

---

## 🚀 Quick Start

### 1. Setup Inicial

```bash
# Instalar herramientas necesarias
bash install_security_tools.sh

# Configurar DVC
bash setup_dvc.sh

# Configurar Git LFS
bash setup_git_lfs.sh
```

### 2. Ejecutar Auditoría Completa

```bash
# Tests en todos los proyectos
bash run_tests_all_projects.sh

# Security scans
bash run_security_scan.sh

# Ver reporte final
cat initial-scan.md
```

### 3. Ver Resultados

```bash
# Coverage summary
cat coverage-summary.csv

# Security issues
cat gitleaks-report.json

# Status general
cat IMPLEMENTATION_STATUS.md
```

---

## 📊 Métricas Objetivo

### Coverage Target
- **Global**: ≥70% en todos los proyectos
- **Tier-1 (BankChurn)**: ≥85%

### Security Target
- **Secretos**: 0 detectados
- **Vulnerabilidades CRITICAL**: 0
- **Vulnerabilidades HIGH**: < 5

### Quality Target
- **Linting**: 0 errores
- **Type hints**: 100% en código core
- **Tests**: 100% passing

---

## 📈 Estado Actual

Ver archivos:
- `initial-scan.md` - Reporte completo
- `IMPLEMENTATION_STATUS.md` - Status detallado
- `../QUICK_START_GUIDE.md` - Guía de inicio

---

## 🔄 Workflow Recomendado

### Para Desarrolladores

```bash
# 1. Antes de empezar a trabajar
cd <proyecto>
dvc pull
git lfs pull

# 2. Desarrollar features
# ... código ...

# 3. Ejecutar tests localmente
pytest --cov=. --cov-report=term-missing

# 4. Antes de commit
cd ..
bash reports/run_security_scan.sh

# 5. Commit si todo está OK
git add .
git commit -m "..."
git push
```

### Para CI/CD

El workflow `.github/workflows/ci-mlops.yml` ejecuta automáticamente:
1. Tests & coverage
2. Security scans
3. Docker builds & Trivy
4. E2E tests
5. Integration reports

---

## 📦 Dependencias de Scripts

### `run_tests_all_projects.sh`
**Requiere**:
- Python 3.12+
- pytest, pytest-cov
- requirements.txt en cada proyecto

**Genera**:
- `<proyecto>-pytest.txt`
- `<proyecto>-coverage.txt`
- `coverage-summary.csv`

### `run_security_scan.sh`
**Requiere**:
- gitleaks
- trivy
- Docker (para container scans)

**Genera**:
- `gitleaks-report.json`
- `<proyecto>-trivy.txt`
- `<proyecto>-trivy.json`

### `install_security_tools.sh`
**Instala**:
- gitleaks v8.18.0+
- trivy (latest)
- dvc[s3]
- git-lfs

### `setup_dvc.sh`
**Configura**:
- DVC init
- Remote storage (S3, local, GDrive, Azure)
- Track large datasets

### `setup_git_lfs.sh`
**Configura**:
- Git LFS hooks
- `.gitattributes` para modelos
- Migración de archivos existentes

---

## 🐛 Troubleshooting

### Error: "command not found"
```bash
# Instalar herramientas
bash install_security_tools.sh
```

### Error: "No module named pytest"
```bash
cd <proyecto>
source .venv/bin/activate
pip install pytest pytest-cov
```

### Error: "DVC not initialized"
```bash
bash setup_dvc.sh
```

### Ver logs detallados
```bash
# Logs de instalación
cat <proyecto>-install.log

# Logs de Docker
cat <proyecto>-docker-build.log

# Output de gitleaks
cat gitleaks-output.txt
```

---

## 📝 Checklist de Validación

Antes de considerar la auditoría completa:

- [ ] Todos los scripts ejecutados exitosamente
- [ ] Coverage ≥70% en todos los proyectos
- [ ] Gitleaks sin secretos detectados
- [ ] Trivy sin vulnerabilidades CRITICAL
- [ ] DVC configurado y funcional
- [ ] Git LFS configurado
- [ ] initial-scan.md actualizado con datos reales
- [ ] CI/CD validado en GitHub Actions

---

## 🎯 Próximos Pasos

Después de completar la auditoría inicial:

1. **Prioridad MEDIA**:
   - Crear model cards
   - Implementar pipeline E2E
   - Configurar MLflow tracking

2. **Prioridad BAJA**:
   - Parametrizar notebooks
   - Publicar imágenes en GHCR
   - Deploy demos

---

## 📚 Recursos Adicionales

- [DVC Documentation](https://dvc.org/doc)
- [Gitleaks Documentation](https://github.com/gitleaks/gitleaks)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

**Última actualización**: 2025-11-21  
**Mantenedor**: DuqueOM
