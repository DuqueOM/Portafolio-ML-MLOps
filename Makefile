# Makefile - ML-MLOps Portfolio Root
# Unified commands for managing the entire portfolio

.PHONY: help install test lint format docker-build docker-demo clean

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Projects
PROJECTS := BankChurn-Predictor CarVision-Market-Intelligence TelecomAI-Customer-Intelligence

help: ## Show this help message
	@echo "$(GREEN)ML-MLOps Portfolio - Available Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'

# ═══════════════════════════════════════════════
# Development Commands
# ═══════════════════════════════════════════════

install: ## Install all project dependencies
	@echo "$(GREEN)Installing dependencies for all projects...$(NC)"
	@for project in $(PROJECTS); do \
		echo "$(BLUE)► Installing $$project...$(NC)"; \
		cd $$project && $(MAKE) install && cd ..; \
	done
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing dev dependencies...$(NC)"
	pip install pre-commit
	pre-commit install
	@for project in $(PROJECTS); do \
		echo "$(BLUE)► Installing dev deps for $$project...$(NC)"; \
		cd $$project && $(MAKE) install-dev 2>/dev/null || pip install -r requirements.txt && cd ..; \
	done
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

# ═══════════════════════════════════════════════
# Testing & Quality
# ═══════════════════════════════════════════════

test: ## Run tests for all projects
	@echo "$(GREEN)Running tests for all projects...$(NC)"
	@for project in $(PROJECTS); do \
		echo "$(BLUE)► Testing $$project...$(NC)"; \
		cd $$project && pytest tests/ -q || echo "$(RED)Tests failed for $$project$(NC)" && cd ..; \
	done

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@for project in $(PROJECTS); do \
		echo "$(BLUE)► Coverage for $$project...$(NC)"; \
		cd $$project && pytest --cov=. --cov-report=term-missing && cd ..; \
	done

lint: ## Run linting on all projects
	@echo "$(GREEN)Running linting...$(NC)"
	pre-commit run --all-files || echo "$(YELLOW)Some linting issues found$(NC)"

format: ## Format code for all projects
	@echo "$(GREEN)Formatting code...$(NC)"
	black . --exclude '/(\.git|\.venv|venv|data|artifacts|mlruns)/'
	isort . --skip-gitignore
	@echo "$(GREEN)✓ Code formatted$(NC)"

typecheck: ## Run mypy type checking
	@echo "$(GREEN)Running type checking...$(NC)"
	@for project in $(PROJECTS); do \
		echo "$(BLUE)► Type checking $$project...$(NC)"; \
		cd $$project && mypy src/ app/ --ignore-missing-imports || echo "$(YELLOW)Type issues in $$project$(NC)" && cd ..; \
	done

# ═══════════════════════════════════════════════
# Docker Commands
# ═══════════════════════════════════════════════

docker-build: ## Build Docker images for all projects
	@echo "$(GREEN)Building Docker images...$(NC)"
	@for project in $(PROJECTS); do \
		echo "$(BLUE)► Building $$project...$(NC)"; \
		cd $$project && docker build -t $$(echo $$project | tr '[:upper:]' '[:lower:]'):latest . && cd ..; \
	done
	@echo "$(GREEN)✓ All images built$(NC)"

docker-demo: ## Start full demo stack
	@echo "$(GREEN)Starting demo stack...$(NC)"
	bash scripts/demo.sh

docker-demo-up: ## Start demo without tests (docker-compose up)
	@echo "$(GREEN)Starting demo services...$(NC)"
	docker-compose -f docker-compose.demo.yml up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(YELLOW)MLflow UI:$(NC)      http://localhost:5000"
	@echo "$(YELLOW)BankChurn API:$(NC)  http://localhost:8001"
	@echo "$(YELLOW)CarVision API:$(NC)  http://localhost:8002"
	@echo "$(YELLOW)Telecom API:$(NC)    http://localhost:8003"

docker-demo-down: ## Stop demo stack
	@echo "$(YELLOW)Stopping demo stack...$(NC)"
	docker-compose -f docker-compose.demo.yml down
	@echo "$(GREEN)✓ Demo stopped$(NC)"

docker-logs: ## View logs from demo stack
	docker-compose -f docker-compose.demo.yml logs -f

docker-scan: ## Run Trivy security scan on images
	@echo "$(GREEN)Scanning Docker images...$(NC)"
	@for project in $(PROJECTS); do \
		IMAGE=$$(echo $$project | tr '[:upper:]' '[:lower:]'); \
		echo "$(BLUE)► Scanning $$IMAGE...$(NC)"; \
		trivy image --severity HIGH,CRITICAL $$IMAGE:latest || true; \
	done

# ═══════════════════════════════════════════════
# DVC & MLflow
# ═══════════════════════════════════════════════

dvc-init: ## Initialize DVC
	@echo "$(GREEN)Initializing DVC...$(NC)"
	dvc init || echo "$(YELLOW)DVC already initialized$(NC)"
	@echo "$(GREEN)✓ DVC ready$(NC)"

dvc-repro: ## Reproduce DVC pipelines for all projects
	@echo "$(GREEN)Reproducing DVC pipelines...$(NC)"
	@for project in $(PROJECTS); do \
		if [ -f $$project/dvc.yaml ]; then \
			echo "$(BLUE)► Running DVC pipeline for $$project...$(NC)"; \
			cd $$project && dvc repro && cd ..; \
		else \
			echo "$(YELLOW)No dvc.yaml in $$project$(NC)"; \
		fi \
	done

mlflow-ui: ## Start MLflow UI
	@echo "$(GREEN)Starting MLflow UI...$(NC)"
	mlflow ui --port 5000 --backend-store-uri ./mlruns

# ═══════════════════════════════════════════════
# CI/CD Simulation
# ═══════════════════════════════════════════════

ci-local: ## Simulate CI pipeline locally
	@echo "$(GREEN)Running CI pipeline locally...$(NC)"
	@echo "$(BLUE)[1/4] Linting...$(NC)"
	$(MAKE) lint
	@echo "$(BLUE)[2/4] Type checking...$(NC)"
	$(MAKE) typecheck
	@echo "$(BLUE)[3/4] Tests...$(NC)"
	$(MAKE) test
	@echo "$(BLUE)[4/4] Docker build...$(NC)"
	$(MAKE) docker-build
	@echo "$(GREEN)✓ CI pipeline completed$(NC)"

security-scan: ## Run security scans (bandit + trivy)
	@echo "$(GREEN)Running security scans...$(NC)"
	bandit -r . -f json -o reports/bandit-report.json || true
	bandit -r . -ll || true
	$(MAKE) docker-scan

# ═══════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════

clean: ## Clean temporary files
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-docker: ## Remove all Docker containers and images
	@echo "$(RED)Removing Docker containers and images...$(NC)"
	docker-compose -f docker-compose.demo.yml down -v --remove-orphans
	@for project in $(PROJECTS); do \
		IMAGE=$$(echo $$project | tr '[:upper:]' '[:lower:]'); \
		docker rmi $$IMAGE:latest 2>/dev/null || true; \
	done
	@echo "$(GREEN)✓ Docker cleanup complete$(NC)"

# ═══════════════════════════════════════════════
# Documentation
# ═══════════════════════════════════════════════

docs: ## Generate documentation
	@echo "$(GREEN)Documentation available at:$(NC)"
	@echo "  - README.md (main)"
	@echo "  - docs/portfolio_landing.md"
	@echo "  - Each project has its own README"

architecture-diagram: ## View architecture info
	@echo "$(GREEN)Architecture documentation:$(NC)"
	@echo "  - ARCHITECTURE.md (see file for system diagram)"
	@echo "  - Use 'make docker-demo' to see services in action"

# ═══════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════

health-check: ## Check health of all demo services
	@echo "$(GREEN)Checking service health...$(NC)"
	@curl -s http://localhost:5000/health 2>/dev/null && echo "$(GREEN)✓ MLflow$(NC)" || echo "$(RED)✗ MLflow$(NC)"
	@curl -s http://localhost:8001/health 2>/dev/null && echo "$(GREEN)✓ BankChurn$(NC)" || echo "$(RED)✗ BankChurn$(NC)"
	@curl -s http://localhost:8002/health 2>/dev/null && echo "$(GREEN)✓ CarVision$(NC)" || echo "$(RED)✗ CarVision$(NC)"
	@curl -s http://localhost:8003/health 2>/dev/null && echo "$(GREEN)✓ Telecom$(NC)" || echo "$(RED)✗ Telecom$(NC)"

stats: ## Show repository statistics
	@echo "$(GREEN)Repository Statistics:$(NC)"
	@echo "$(BLUE)Projects:$(NC) $$(ls -d */ | wc -l)"
	@echo "$(BLUE)Python files:$(NC) $$(find . -name '*.py' -not -path '*/\.*' -not -path '*/venv/*' | wc -l)"
	@echo "$(BLUE)Test files:$(NC) $$(find . -name 'test_*.py' | wc -l)"
	@echo "$(BLUE)Docker images:$(NC) $$(find . -name 'Dockerfile' | wc -l)"

.DEFAULT_GOAL := help
