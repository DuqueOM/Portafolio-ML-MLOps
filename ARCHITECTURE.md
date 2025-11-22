# 🏗️ System Architecture - ML-MLOps Portfolio

## Overview

This document describes the technical architecture of the ML-MLOps Portfolio, a production-ready machine learning platform demonstrating enterprise-grade ML engineering practices.

## Table of Contents
- [High-Level Architecture](#high-level-architecture)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Infrastructure](#infrastructure)
- [CI/CD Pipeline](#cicd-pipeline)
- [Security Architecture](#security-architecture)
- [Monitoring & Observability](#monitoring--observability)
- [Technology Stack](#technology-stack)

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Layer"
        UI[Web UI/Dashboard]
        API_CLIENT[API Clients]
    end
    
    subgraph "API Gateway Layer"
        INGRESS[Ingress Controller]
    end
    
    subgraph "ML Services"
        BANK[BankChurn API<br/>:8001]
        CAR[CarVision API<br/>:8002]
        TELECOM[Telecom API<br/>:8003]
    end
    
    subgraph "ML Platform"
        MLFLOW[MLflow Server<br/>:5000]
        DVC[DVC Storage]
    end
    
    subgraph "Monitoring"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
        EVIDENTLY[Evidently<br/>Drift Detection]
    end
    
    subgraph "Storage"
        S3[S3/GCS<br/>Model Artifacts]
        RDS[PostgreSQL<br/>MLflow Backend]
    end
    
    UI --> INGRESS
    API_CLIENT --> INGRESS
    INGRESS --> BANK
    INGRESS --> CAR
    INGRESS --> TELECOM
    
    BANK --> MLFLOW
    CAR --> MLFLOW
    TELECOM --> MLFLOW
    
    MLFLOW --> S3
    MLFLOW --> RDS
    
    BANK --> PROMETHEUS
    CAR --> PROMETHEUS
    TELECOM --> PROMETHEUS
    
    PROMETHEUS --> GRAFANA
    
    EVIDENTLY --> S3
```

---

## Component Architecture

### ML Services Architecture

Each ML service follows a consistent pattern:

```mermaid
graph LR
    subgraph "ML Service Pod"
        API[FastAPI Application]
        MODEL[Model Inference Engine]
        CACHE[In-Memory Cache]
        PREPROCESS[Preprocessing Pipeline]
    end
    
    subgraph "External Dependencies"
        MLFLOW_API[MLflow API]
        STORAGE[Object Storage]
        METRICS[Metrics Exporter]
    end
    
    REQUEST[Client Request] --> API
    API --> PREPROCESS
    PREPROCESS --> MODEL
    MODEL --> CACHE
    MODEL --> MLFLOW_API
    MLFLOW_API --> STORAGE
    API --> METRICS
```

**Components:**
- **FastAPI Application**: REST API endpoints with Pydantic validation
- **Model Inference Engine**: scikit-learn/custom models loaded from MLflow
- **Preprocessing Pipeline**: Feature engineering and transformations
- **In-Memory Cache**: LRU cache for repeated predictions
- **Metrics Exporter**: Prometheus metrics for monitoring

---

## Data Flow

### Training Pipeline

```mermaid
sequenceDiagram
    participant DEV as Developer
    participant GIT as GitHub
    participant CI as GitHub Actions
    participant DVC as DVC Remote
    participant MLFLOW as MLflow Server
    participant S3 as S3/GCS Storage
    
    DEV->>GIT: Push code changes
    GIT->>CI: Trigger CI/CD
    CI->>CI: Run tests & linting
    CI->>DVC: Pull data (dvc pull)
    CI->>CI: Run training (dvc repro)
    CI->>MLFLOW: Log metrics & params
    CI->>MLFLOW: Register model
    MLFLOW->>S3: Store model artifacts
    CI->>DVC: Push new data/model versions
    CI->>GIT: Post CML report
```

### Inference Pipeline

```mermaid
sequenceDiagram
    participant CLIENT as Client
    participant API as FastAPI Service
    participant CACHE as Cache Layer
    participant MODEL as Model
    participant MLFLOW as MLflow
    participant METRICS as Prometheus
    
    CLIENT->>API: POST /predict
    API->>API: Validate input (Pydantic)
    API->>CACHE: Check cache
    alt Cache Hit
        CACHE->>API: Return cached result
    else Cache Miss
        API->>MODEL: Preprocess & predict
        MODEL->>MODEL: Run inference
        MODEL->>CACHE: Store result
        MODEL->>API: Return prediction
    end
    API->>METRICS: Record metrics
    API->>MLFLOW: Log inference (async)
    API->>CLIENT: Return JSON response
```

---

## Infrastructure

### Kubernetes Deployment

```yaml
# Simplified K8s architecture
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bankchurn-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bankchurn
  template:
    spec:
      containers:
      - name: bankchurn-api
        image: ghcr.io/duqueom/bankchurn:latest
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
```

### Terraform Infrastructure

**AWS Resources:**
- EKS Cluster (Kubernetes)
- S3 Buckets (model artifacts, MLflow)
- RDS PostgreSQL (MLflow backend)
- ECR Repositories (Docker images)
- CloudWatch (logging)

**GCP Resources:**
- GKE Cluster
- Cloud Storage (artifacts)
- Cloud SQL (PostgreSQL)
- Artifact Registry
- Cloud Monitoring

---

## CI/CD Pipeline

### Pipeline Stages

```mermaid
graph LR
    TRIGGER[Git Push/PR] --> LINT[Lint & Format]
    LINT --> TYPE[Type Check]
    TYPE --> TEST[Unit Tests]
    TEST --> INTEGRATION[Integration Tests]
    INTEGRATION --> SECURITY[Security Scan]
    SECURITY --> DOCKER[Build Docker]
    DOCKER --> SCAN[Container Scan]
    SCAN --> DEPLOY[Deploy]
    
    style TRIGGER fill:#e1f5ff
    style DEPLOY fill:#c8e6c9
```

**Pipeline Jobs:**

1. **Code Quality**
   - Black formatting
   - isort imports
   - flake8 linting
   - mypy type checking
   - bandit security

2. **Testing**
   - Unit tests (pytest)
   - Integration tests
   - Coverage report (>65% threshold)
   - E2E tests

3. **Container Security**
   - Docker build (multi-stage)
   - Trivy vulnerability scan
   - Image size optimization
   - Non-root user enforcement

4. **Deployment**
   - Push to GHCR/ECR
   - Update K8s manifests
   - Rolling update
   - Health check verification

---

## Security Architecture

### Security Layers

```mermaid
graph TB
    subgraph "Application Security"
        PYDANTIC[Input Validation<br/>Pydantic]
        CORS[CORS Policies]
        RATE[Rate Limiting]
    end
    
    subgraph "Container Security"
        NONROOT[Non-Root User]
        SCAN[Trivy Scanning]
        MINIMAL[Minimal Base Image]
    end
    
    subgraph "Infrastructure Security"
        SECRETS[Secret Management]
        IAM[IAM Roles]
        ENCRYPTION[Encryption at Rest]
        TLS[TLS/HTTPS]
    end
    
    subgraph "Network Security"
        FIREWALL[Firewall Rules]
        VPC[VPC Isolation]
        SG[Security Groups]
    end
```

**Security Practices:**

1. **Secrets Management**
   - GitHub Secrets for CI/CD
   - AWS Secrets Manager / GCP Secret Manager
   - No hardcoded credentials

2. **Container Security**
   - Multi-stage builds
   - Non-root user (UID 1000)
   - Read-only root filesystem
   - Trivy scanning in CI

3. **API Security**
   - Input validation (Pydantic)
   - Rate limiting
   - CORS configuration
   - API key authentication (optional)

4. **Infrastructure Security**
   - VPC isolation
   - Private subnets for databases
   - Security groups
   - Encryption at rest (S3, RDS)

---

## Monitoring & Observability

### Metrics Collection

```mermaid
graph LR
    subgraph "Services"
        API1[BankChurn API]
        API2[CarVision API]
        API3[Telecom API]
    end
    
    subgraph "Metrics Pipeline"
        EXPORTER[Prometheus Exporter]
        PROM[Prometheus]
        GRAFANA[Grafana]
    end
    
    subgraph "Alerting"
        ALERT[Alert Manager]
        SLACK[Slack/Email]
    end
    
    API1 --> EXPORTER
    API2 --> EXPORTER
    API3 --> EXPORTER
    EXPORTER --> PROM
    PROM --> GRAFANA
    PROM --> ALERT
    ALERT --> SLACK
```

**Monitored Metrics:**

1. **System Metrics**
   - CPU/Memory usage
   - Request latency (p50, p95, p99)
   - Request rate (RPS)
   - Error rate

2. **ML Metrics**
   - Prediction latency
   - Model version in use
   - Feature statistics
   - Prediction distribution

3. **Business Metrics**
   - Predictions per day
   - API usage by endpoint
   - User patterns

### Drift Detection

**Evidently Integration:**
- Scheduled daily checks (2 AM UTC)
- Compares reference vs current data
- Generates HTML reports
- Creates GitHub issues on drift alert
- Tracks feature drift over time

---

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.12 | Main development language |
| **ML Framework** | scikit-learn, imbalanced-learn | Model training |
| **API Framework** | FastAPI, Uvicorn | REST API services |
| **Validation** | Pydantic | Data validation |
| **Containerization** | Docker (multi-stage) | Application packaging |
| **Orchestration** | Kubernetes | Container orchestration |
| **ML Tracking** | MLflow | Experiment tracking |
| **Data Versioning** | DVC | Data/model versioning |
| **IaC** | Terraform | Infrastructure as code |
| **CI/CD** | GitHub Actions | Automation pipeline |
| **Monitoring** | Prometheus + Grafana | Observability |
| **Drift Detection** | Evidently | Data drift monitoring |

### Development Tools

- **Linting**: black, isort, flake8
- **Type Checking**: mypy
- **Testing**: pytest, pytest-cov
- **Security**: bandit, trivy
- **Pre-commit**: Automated checks

---

## Deployment Patterns

### Blue-Green Deployment

```mermaid
graph LR
    LB[Load Balancer] --> BLUE[Blue Environment<br/>v1.0.0]
    LB -.-> GREEN[Green Environment<br/>v1.1.0]
    
    style BLUE fill:#4fc3f7
    style GREEN fill:#81c784
```

**Process:**
1. Deploy new version (green) alongside old (blue)
2. Run smoke tests on green
3. Gradually shift traffic: 10% → 50% → 100%
4. Monitor metrics during transition
5. Rollback to blue if issues detected
6. Decommission blue after validation

---

## Scaling Strategy

### Horizontal Scaling

```mermaid
graph TB
    subgraph "Auto-Scaling"
        HPA[Horizontal Pod Autoscaler]
        METRICS[Metrics Server]
    end
    
    subgraph "Pods"
        POD1[Pod 1]
        POD2[Pod 2]
        POD3[Pod 3]
        PODN[Pod N]
    end
    
    METRICS --> HPA
    HPA --> POD1
    HPA --> POD2
    HPA --> POD3
    HPA -.-> PODN
```

**Scaling Rules:**
- CPU > 70%: Scale up
- Memory > 80%: Scale up
- Request latency > 500ms: Scale up
- Min replicas: 2
- Max replicas: 10

---

## Disaster Recovery

**Backup Strategy:**
- **Models**: Versioned in S3/GCS with lifecycle policies
- **Data**: DVC remote storage with versioning
- **Database**: Daily automated backups (7-day retention)
- **Config**: GitOps (version controlled)

**Recovery Time Objectives (RTO):**
- Service restoration: < 15 minutes
- Data restoration: < 1 hour
- Full system restoration: < 4 hours

---

## Future Enhancements

1. **Feature Store** (Feast)
   - Centralized feature management
   - Online/offline serving
   - Feature freshness monitoring

2. **A/B Testing Framework**
   - Multi-armed bandit algorithms
   - Traffic splitting
   - Statistical significance testing

3. **Model Explainability**
   - SHAP values
   - LIME explanations
   - Feature importance dashboard

4. **Advanced Monitoring**
   - Distributed tracing (Jaeger)
   - APM (Application Performance Monitoring)
   - Log aggregation (ELK stack)

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

---

**Document Version:** 1.0.0  
**Last Updated:** November 2024  
**Maintained by:** DuqueOM
