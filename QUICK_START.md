# 🚀 Quick Start - Portfolio TOP-3

**Portfolio ML/MLOps Tier-1** - Listo para presentar

---

## 📊 Portfolio Summary

### TOP-3 Projects

| # | Proyecto | Sector | Coverage | Tests |
|---|----------|--------|----------|-------|
| 1 | **TelecomAI-Customer-Intelligence** | Telecom | 87% | 54 |
| 2 | **CarVision-Market-Intelligence** | Automotive | 81% | 13 |
| 3 | **BankChurn-Predictor** | Banking | 68% | 107 |

**Promedio: 78.7%** 🚀

---

## ⚡ Quick Commands

### Run All Tests
```bash
bash scripts/run_tests_top3.sh
```

### Start MLflow Stack
```bash
docker compose -f docker-compose.mlflow.yml up -d
```

### Security Scan
```bash
gitleaks detect --source . --report-path reports/gitleaks-report.json
```

### View Coverage
```bash
cat reports/coverage-summary-TOP3.csv
```

---

## 📁 Key Files

### Reports
- `reports/initial-scan-COMPLETE.md` - Audit completo
- `reports/PORTFOLIO_TIER1_FINAL.md` - Reporte ejecutivo
- `WORKFLOW_COMPLETE.md` - Resumen del workflow
- `ARCHIVED_PROJECTS.md` - Proyectos archivados

### Configuration
- `.github/workflows/ci-portfolio-top3.yml` - CI/CD
- `docker-compose.mlflow.yml` - MLflow stack
- `.gitattributes` - Git LFS config
- `.gitleaksignore` - Security exceptions

---

## 🎯 Next Steps

### Development
```bash
# 1. Iniciar MLflow
docker compose -f docker-compose.mlflow.yml up -d

# 2. Trackear datasets con DVC
cd BankChurn-Predictor
dvc add data/*.csv
git add data/*.dvc .gitignore
git commit -m "chore(dvc): track datasets"
dvc push

# 3. Run training con MLflow tracking
python -m src.bankchurn.cli train --config configs/config.yaml
```

### Production
```bash
# Build Docker images
docker build -t bankchurn-api:latest BankChurn-Predictor/
docker build -t carvision-api:latest CarVision-Market-Intelligence/
docker build -t telecomai-api:latest TelecomAI-Customer-Intelligence/

# Run security scans
trivy image bankchurn-api:latest
```

---

## ✅ Status

- ✅ Tests: 174 totales, todos pasando
- ✅ Coverage: 78.7% promedio
- ✅ Security: Gitleaks clean
- ✅ DVC: Configurado
- ✅ MLflow: Ready
- ✅ Git LFS: 5 modelos tracked
- ✅ CI/CD: GitHub Actions ready

---

**Portfolio Status**: 🏆 **TIER-1 PRODUCTION-READY**
