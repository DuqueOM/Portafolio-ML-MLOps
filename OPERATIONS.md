# 🔧 Operations Runbook - ML-MLOps Portfolio

## Purpose

This runbook provides operational procedures, troubleshooting guides, and emergency response protocols for maintaining and operating the ML-MLOps Portfolio services.

## Table of Contents
- [Quick Reference](#quick-reference)
- [Service Management](#service-management)
- [Deployment Procedures](#deployment-procedures)
- [Model Retraining](#model-retraining)
- [Rollback Procedures](#rollback-procedures)
- [Troubleshooting](#troubleshooting)
- [Monitoring & Alerts](#monitoring--alerts)
- [Emergency Procedures](#emergency-procedures)
- [Maintenance Windows](#maintenance-windows)

---

## Quick Reference

### Service Endpoints

| Service | URL | Health Check |
|---------|-----|--------------|
| BankChurn API | `http://localhost:8001` | `/health` |
| CarVision API | `http://localhost:8002` | `/health` |
| Telecom API | `http://localhost:8003` | `/health` |
| MLflow UI | `http://localhost:5000` | `/health` |
| Prometheus | `http://localhost:9090` | `/-/healthy` |
| Grafana | `http://localhost:3000` | `/api/health` |

### Common Commands

```bash
# Start all services
make docker-demo

# Check service health
make health-check

# View logs
docker-compose -f docker-compose.demo.yml logs -f [service_name]

# Restart service
docker-compose -f docker-compose.demo.yml restart [service_name]

# Stop all services
docker-compose -f docker-compose.demo.yml down
```

---

## Service Management

### Starting Services

#### Full Stack (Recommended)
```bash
# Using make
make docker-demo-up

# Using docker-compose directly
docker-compose -f docker-compose.demo.yml up -d

# With monitoring (Prometheus + Grafana)
docker-compose -f docker-compose.demo.yml --profile monitoring up -d
```

#### Individual Service
```bash
# BankChurn only
cd BankChurn-Predictor
docker build -t bankchurn:latest .
docker run -d -p 8001:8000 --name bankchurn bankchurn:latest
```

### Stopping Services

```bash
# Stop all services
docker-compose -f docker-compose.demo.yml down

# Stop specific service
docker-compose -f docker-compose.demo.yml stop bankchurn

# Stop and remove volumes
docker-compose -f docker-compose.demo.yml down -v
```

### Restarting Services

```bash
# Restart all
docker-compose -f docker-compose.demo.yml restart

# Restart specific service
docker-compose -f docker-compose.demo.yml restart bankchurn

# Hard restart (rebuild)
docker-compose -f docker-compose.demo.yml up -d --build bankchurn
```

---

## Deployment Procedures

### Pre-Deployment Checklist

- [ ] All tests pass in CI/CD
- [ ] Code review approved
- [ ] Security scans passed (Trivy, Bandit)
- [ ] Documentation updated
- [ ] Backup current model version
- [ ] Alert team of upcoming deployment
- [ ] Schedule deployment window
- [ ] Prepare rollback plan

### Standard Deployment

**Step 1: Pre-deployment validation**
```bash
# Run local CI checks
make ci-local

# Test Docker build
docker build -t bankchurn:test .

# Run smoke tests
pytest tests/ -m smoke
```

**Step 2: Build and tag images**
```bash
# Get git SHA for tagging
GIT_SHA=$(git rev-parse --short HEAD)

# Build image
docker build -t ghcr.io/duqueom/bankchurn:${GIT_SHA} .
docker tag ghcr.io/duqueom/bankchurn:${GIT_SHA} ghcr.io/duqueom/bankchurn:latest
```

**Step 3: Deploy**
```bash
# Push to registry
docker push ghcr.io/duqueom/bankchurn:${GIT_SHA}
docker push ghcr.io/duqueom/bankchurn:latest

# Update Kubernetes (if using K8s)
kubectl set image deployment/bankchurn bankchurn=ghcr.io/duqueom/bankchurn:${GIT_SHA}

# Or restart docker-compose
docker-compose -f docker-compose.demo.yml up -d --no-deps --build bankchurn
```

**Step 4: Verification**
```bash
# Check health
curl http://localhost:8001/health

# Check model info
curl http://localhost:8001/model_info

# Run smoke prediction
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_request.json

# Monitor logs for errors
docker-compose -f docker-compose.demo.yml logs -f bankchurn --tail=100
```

**Step 5: Post-deployment monitoring**
- Monitor error rates for 15 minutes
- Check Prometheus metrics
- Verify prediction latencies
- Review Grafana dashboards

---

## Model Retraining

### When to Retrain

**Triggers:**
- Data drift detected (>50% features drifted)
- Performance degradation (>5% drop in metrics)
- New data available (monthly batch)
- Feature engineering updates
- Manual trigger for experiments

### Retraining Procedure

**Step 1: Prepare data**
```bash
cd BankChurn-Predictor

# Pull latest data
dvc pull

# Or download new data
python scripts/fetch_data.py --output data/raw/
```

**Step 2: Run training pipeline**
```bash
# Using DVC
dvc repro

# Or manual training
python main.py --mode train \
  --config configs/config.yaml \
  --seed 42 \
  --input data/raw/Churn.csv
```

**Step 3: Evaluate new model**
```bash
# Run evaluation
python main.py --mode evaluate \
  --model models/best_model.pkl \
  --config configs/config.yaml

# Compare with baseline
python scripts/compare_models.py \
  --baseline models/baseline_model.pkl \
  --candidate models/best_model.pkl
```

**Step 4: Register model in MLflow**
```bash
# Model is auto-registered during training
# Verify in MLflow UI: http://localhost:5000

# Or manually register
mlflow models register \
  --model-uri runs:/<run_id>/model \
  --name bankchurn-model
```

**Step 5: Promote to production**
```bash
# Tag model as production
mlflow models update \
  --name bankchurn-model \
  --version <version> \
  --stage Production

# Update model reference in config
vim configs/config.yaml  # Update model_version

# Redeploy service
make docker-demo-up
```

### Automated Retraining

**GitHub Actions workflow:**
```yaml
# Triggered on schedule or manual dispatch
name: Retrain Model
on:
  schedule:
    - cron: '0 0 1 * *'  # Monthly
  workflow_dispatch:
```

---

## Rollback Procedures

### Quick Rollback (Emergency)

**Docker Compose:**
```bash
# Rollback to previous image
docker-compose -f docker-compose.demo.yml down
docker pull ghcr.io/duqueom/bankchurn:<previous-sha>
docker tag ghcr.io/duqueom/bankchurn:<previous-sha> bankchurn:latest
docker-compose -f docker-compose.demo.yml up -d
```

**Kubernetes:**
```bash
# Rollback deployment
kubectl rollout undo deployment/bankchurn

# Rollback to specific revision
kubectl rollout undo deployment/bankchurn --to-revision=2

# Check rollout status
kubectl rollout status deployment/bankchurn
```

### Model Rollback

**Rollback to previous model version:**
```bash
# List model versions
mlflow models list --name bankchurn-model

# Promote previous version
mlflow models update \
  --name bankchurn-model \
  --version <previous-version> \
  --stage Production

# Update config
vim configs/config.yaml

# Restart service
docker-compose -f docker-compose.demo.yml restart bankchurn
```

### Data Rollback

**DVC rollback:**
```bash
# List data versions
git log --oneline -- data/raw/Churn.csv.dvc

# Checkout previous version
git checkout <commit-sha> -- data/raw/Churn.csv.dvc
dvc checkout

# Pull data
dvc pull
```

---

## Troubleshooting

### Service Won't Start

**Symptoms:** Container exits immediately

**Diagnosis:**
```bash
# Check logs
docker-compose -f docker-compose.demo.yml logs bankchurn

# Check container status
docker ps -a

# Inspect container
docker inspect <container-id>
```

**Common Causes:**
1. **Port already in use**
   ```bash
   # Check port usage
   lsof -i :8001
   
   # Kill process or change port
   docker-compose -f docker-compose.demo.yml up -d --force-recreate
   ```

2. **Missing dependencies**
   ```bash
   # Rebuild with no cache
   docker-compose -f docker-compose.demo.yml build --no-cache bankchurn
   ```

3. **Model file missing**
   ```bash
   # Check model exists
   ls -lh BankChurn-Predictor/models/
   
   # Retrain if missing
   cd BankChurn-Predictor && python main.py --mode train
   ```

### High Latency

**Symptoms:** Predictions taking >2 seconds

**Diagnosis:**
```bash
# Check resource usage
docker stats

# Check API metrics
curl http://localhost:8001/metrics | grep latency

# Profile endpoint
time curl -X POST "http://localhost:8001/predict" -d @sample.json
```

**Solutions:**
1. **Scale horizontally**
   ```bash
   docker-compose -f docker-compose.demo.yml up -d --scale bankchurn=3
   ```

2. **Optimize model**
   - Use smaller model
   - Quantize model
   - Cache predictions

3. **Add caching**
   ```python
   # Already implemented in app with LRU cache
   from functools import lru_cache
   ```

### Memory Leak

**Symptoms:** Memory usage continuously increasing

**Diagnosis:**
```bash
# Monitor memory over time
watch -n 5 'docker stats --no-stream bankchurn-api'

# Profile memory usage
python -m memory_profiler app/fastapi_app.py
```

**Solutions:**
1. Restart service periodically
2. Review model loading (should load once)
3. Check for unclosed connections
4. Increase memory limits

### MLflow Connection Issues

**Symptoms:** "Connection refused" errors

**Diagnosis:**
```bash
# Check MLflow is running
curl http://localhost:5000/health

# Check service connectivity
docker-compose -f docker-compose.demo.yml exec bankchurn ping mlflow
```

**Solutions:**
```bash
# Restart MLflow
docker-compose -f docker-compose.demo.yml restart mlflow

# Check network
docker network ls
docker network inspect ml-mlops-network

# Verify environment variable
docker-compose -f docker-compose.demo.yml exec bankchurn env | grep MLFLOW
```

### Data Drift Alert

**Symptoms:** GitHub issue created by drift detection workflow

**Response:**
1. **Review drift report**
   - Download artifact from GitHub Actions
   - Open `drift_report.html`
   - Identify drifted features

2. **Investigate root cause**
   - Data pipeline changes?
   - User behavior shift?
   - Data quality issues?

3. **Take action**
   - If valid drift: Schedule retraining
   - If data issue: Fix pipeline
   - If false positive: Adjust thresholds

---

## Monitoring & Alerts

### Key Metrics to Monitor

**Service Health:**
- Request rate (RPS)
- Error rate (%)
- Latency (p50, p95, p99)
- CPU/Memory usage

**ML Performance:**
- Prediction latency
- Model version
- Feature drift score
- Prediction distribution

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Error rate | >1% | >5% | Check logs, consider rollback |
| Latency p99 | >2s | >5s | Scale up or optimize |
| CPU usage | >70% | >90% | Scale horizontally |
| Memory usage | >80% | >95% | Restart or scale |
| Drift score | >30% | >50% | Review data, schedule retrain |

### Grafana Dashboards

**Available dashboards:**
1. **Service Overview** - Request rate, errors, latency
2. **ML Metrics** - Predictions per minute, model versions
3. **Infrastructure** - CPU, memory, disk usage
4. **Business KPIs** - Daily predictions, API usage

Access: `http://localhost:3000` (admin/admin)

---

## Emergency Procedures

### Service Down

**Priority: P0 - Critical**

**Immediate Actions:**
1. Check health endpoints
2. Review recent deployments
3. Check for infrastructure issues
4. Notify team

**Response:**
```bash
# Quick health check
make health-check

# Emergency rollback
kubectl rollout undo deployment/bankchurn

# Or docker-compose rollback
docker-compose -f docker-compose.demo.yml down
docker-compose -f docker-compose.demo.yml up -d
```

### Data Corruption

**Priority: P1 - High**

**Immediate Actions:**
1. Stop affected service
2. Assess scope of corruption
3. Restore from backup

**Response:**
```bash
# Stop service
docker-compose -f docker-compose.demo.yml stop bankchurn

# Restore data from DVC
git checkout <last-good-commit> -- data/
dvc checkout
dvc pull

# Verify data integrity
python scripts/validate_data.py

# Restart service
docker-compose -f docker-compose.demo.yml start bankchurn
```

### Model Performance Degradation

**Priority: P2 - Medium**

**Response:**
1. Verify with A/B test
2. Check for data drift
3. Review recent data changes
4. Schedule emergency retraining

---

## Maintenance Windows

### Scheduled Maintenance

**Frequency:** Monthly (first Sunday, 2-4 AM UTC)

**Activities:**
- Apply security patches
- Update dependencies
- Database maintenance
- Clean up old artifacts
- Review and update documentation

**Procedure:**
```bash
# 1. Notify users (24h advance)
# 2. Take snapshot/backup
# 3. Update services
# 4. Run health checks
# 5. Monitor for 1 hour
# 6. Send completion notice
```

### Backup Schedule

- **Models**: Continuous (MLflow + S3 versioning)
- **Data**: Daily via DVC
- **Database**: Daily automated backups
- **Config**: Git versioned
- **Logs**: 30-day retention

---

## Contacts & Escalation

### On-Call Rotation

| Severity | Response Time | Escalation Path |
|----------|---------------|-----------------|
| P0 - Critical | 15 minutes | Team Lead → Director |
| P1 - High | 1 hour | Team Lead |
| P2 - Medium | 4 hours | Primary Engineer |
| P3 - Low | 1 business day | Assignee |

### Key Contacts

- **Team Lead**: [Contact Info]
- **ML Engineer**: [Contact Info]
- **DevOps Engineer**: [Contact Info]
- **Data Engineer**: [Contact Info]

---

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2024-11 | 1.0.0 | Initial runbook | DuqueOM |

---

**Document Owner:** DuqueOM  
**Review Cycle:** Quarterly  
**Last Reviewed:** November 2024
