# Nexus Forge Makefile
# Production-ready build and development commands

.PHONY: help install install-backend install-frontend install-pre-commit \
        lint lint-backend lint-frontend format format-backend format-frontend \
        type-check test test-backend test-frontend test-e2e coverage \
        coverage-backend coverage-frontend clean build run-backend run-frontend \
        run docker-build docker-up docker-down migrate security-check

# Default target
.DEFAULT_GOAL := help

# Python executable
PYTHON := python3
PIP := $(PYTHON) -m pip

# Node executable
NPM := npm

# Project directories
BACKEND_DIR := .
FRONTEND_DIR := frontend
VENV_DIR := venv

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo '$(CYAN)Nexus Forge - Available Commands$(NC)'
	@echo ''
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make $(CYAN)<target>$(NC)\n\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation

install: install-backend install-frontend install-pre-commit ## Install all dependencies
	@echo "$(GREEN)✓ All dependencies installed successfully!$(NC)"

install-backend: ## Install Python dependencies
	@echo "$(CYAN)Installing Python dependencies...$(NC)"
	$(PIP) install -e ".[dev,test,docs]"
	@echo "$(GREEN)✓ Python dependencies installed$(NC)"

install-frontend: ## Install Node dependencies
	@echo "$(CYAN)Installing Node dependencies...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) install
	@echo "$(GREEN)✓ Node dependencies installed$(NC)"

install-pre-commit: ## Install and setup pre-commit hooks
	@echo "$(CYAN)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

##@ Code Quality

lint: lint-backend lint-frontend ## Run all linters

lint-backend: ## Run Python linters (ruff, mypy, bandit)
	@echo "$(CYAN)Running Python linters...$(NC)"
	ruff check nexus_forge tests
	mypy nexus_forge --strict
	bandit -r nexus_forge -ll
	@echo "$(GREEN)✓ Python linting passed$(NC)"

lint-frontend: ## Run TypeScript/JavaScript linters
	@echo "$(CYAN)Running frontend linters...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) run lint
	cd $(FRONTEND_DIR) && $(NPM) run type-check
	@echo "$(GREEN)✓ Frontend linting passed$(NC)"

format: format-backend format-frontend ## Format all code

format-backend: ## Format Python code
	@echo "$(CYAN)Formatting Python code...$(NC)"
	black nexus_forge tests
	isort nexus_forge tests
	ruff check --fix nexus_forge tests
	@echo "$(GREEN)✓ Python code formatted$(NC)"

format-frontend: ## Format TypeScript/JavaScript code
	@echo "$(CYAN)Formatting frontend code...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) run format
	@echo "$(GREEN)✓ Frontend code formatted$(NC)"

type-check: ## Run type checking for both backend and frontend
	@echo "$(CYAN)Running type checks...$(NC)"
	mypy nexus_forge --strict
	cd $(FRONTEND_DIR) && $(NPM) run type-check
	@echo "$(GREEN)✓ Type checking passed$(NC)"

##@ Testing

test: test-backend test-frontend ## Run all tests

test-backend: ## Run Python tests
	@echo "$(CYAN)Running Python tests...$(NC)"
	pytest tests/ -v --tb=short

test-frontend: ## Run frontend tests
	@echo "$(CYAN)Running frontend tests...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) test -- --watchAll=false

test-e2e: ## Run end-to-end tests
	@echo "$(CYAN)Running E2E tests...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) run test:e2e

coverage: coverage-backend coverage-frontend ## Generate coverage reports

coverage-backend: ## Generate Python coverage report
	@echo "$(CYAN)Generating Python coverage report...$(NC)"
	pytest tests/ --cov=nexus_forge --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated at htmlcov/index.html$(NC)"

coverage-frontend: ## Generate frontend coverage report
	@echo "$(CYAN)Generating frontend coverage report...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) run test:coverage
	@echo "$(GREEN)✓ Frontend coverage report generated$(NC)"

##@ Build & Run

build: ## Build production assets
	@echo "$(CYAN)Building production assets...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) run build
	@echo "$(GREEN)✓ Production build complete$(NC)"

run-backend: ## Run backend development server
	@echo "$(CYAN)Starting backend server...$(NC)"
	uvicorn nexus_forge.main:app --reload --host 0.0.0.0 --port 8000

run-frontend: ## Run frontend development server
	@echo "$(CYAN)Starting frontend server...$(NC)"
	cd $(FRONTEND_DIR) && $(NPM) start

run: ## Run both backend and frontend (requires 2 terminals)
	@echo "$(YELLOW)Please run the following commands in separate terminals:$(NC)"
	@echo "  1. make run-backend"
	@echo "  2. make run-frontend"

##@ Database

migrate: ## Run database migrations
	@echo "$(CYAN)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations applied$(NC)"

migrate-create: ## Create a new migration (usage: make migrate-create name="migration_name")
	@echo "$(CYAN)Creating new migration...$(NC)"
	alembic revision --autogenerate -m "$(name)"
	@echo "$(GREEN)✓ Migration created$(NC)"

##@ Docker

docker-build: ## Build Docker images
	@echo "$(CYAN)Building Docker images...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-up: ## Start Docker containers
	@echo "$(CYAN)Starting Docker containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Docker containers started$(NC)"

docker-down: ## Stop Docker containers
	@echo "$(CYAN)Stopping Docker containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Docker containers stopped$(NC)"

##@ Security & Maintenance

security-check: ## Run security checks
	@echo "$(CYAN)Running security checks...$(NC)"
	bandit -r nexus_forge -ll
	safety check || true
	cd $(FRONTEND_DIR) && npm audit || true
	@echo "$(GREEN)✓ Security check complete$(NC)"

clean: ## Clean build artifacts and caches
	@echo "$(CYAN)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf $(FRONTEND_DIR)/build
	rm -rf $(FRONTEND_DIR)/coverage
	@echo "$(GREEN)✓ Clean complete$(NC)"

update-deps: ## Update all dependencies to latest versions
	@echo "$(CYAN)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,test,docs]"
	cd $(FRONTEND_DIR) && $(NPM) update
	pre-commit autoupdate
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

##@ Pre-commit

pre-commit-run: ## Run pre-commit on all files
	@echo "$(CYAN)Running pre-commit on all files...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks passed$(NC)"

pre-commit-update: ## Update pre-commit hooks
	@echo "$(CYAN)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate
	@echo "$(GREEN)✓ Pre-commit hooks updated$(NC)"

##@ Production

prod-check: lint type-check test security-check ## Run all production checks
	@echo "$(GREEN)✓ All production checks passed!$(NC)"

prod-build: prod-check build ## Build for production after all checks
	@echo "$(GREEN)✓ Production build complete!$(NC)"

##@ Development Shortcuts

dev: ## Quick development setup
	@make install
	@make migrate
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo "$(YELLOW)Run 'make run' to start the servers$(NC)"

refresh: clean install migrate ## Clean and reinstall everything
	@echo "$(GREEN)✓ Environment refreshed!$(NC)"

##@ Advanced Testing

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(NC)"
	pytest tests/unit/ -v --tb=short
	@echo "$(GREEN)✓ Unit tests passed$(NC)"

test-integration: ## Run integration tests only
	@echo "$(CYAN)Running integration tests...$(NC)"
	pytest tests/integration/ -v --tb=short
	@echo "$(GREEN)✓ Integration tests passed$(NC)"

test-performance: ## Run performance tests
	@echo "$(CYAN)Running performance tests...$(NC)"
	pytest tests/performance/ -v --benchmark-only
	@echo "$(GREEN)✓ Performance tests completed$(NC)"

test-security: security-check ## Run security tests
	@echo "$(CYAN)Running security tests...$(NC)"
	pytest tests/security/ -v
	@echo "$(GREEN)✓ Security tests passed$(NC)"

##@ Deployment Commands

setup-dev: install migrate ## Setup development environment
	@echo "$(CYAN)Setting up development environment...$(NC)"
	cp .env.example .env
	@echo "$(YELLOW)Please update .env with your API keys$(NC)"
	@echo "$(GREEN)✓ Development setup complete$(NC)"

run-local: ## Run locally with development settings
	@echo "$(CYAN)Starting local development servers...$(NC)"
	honcho start

build-docker: ## Build Docker image for production
	@echo "$(CYAN)Building production Docker image...$(NC)"
	docker build -t nexus-forge:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

run-docker: ## Run Docker container
	@echo "$(CYAN)Running Docker container...$(NC)"
	docker run -p 8000:8000 --env-file .env nexus-forge:latest

deploy-gcr: ## Deploy to Google Cloud Run
	@echo "$(CYAN)Deploying to Google Cloud Run...$(NC)"
	gcloud run deploy nexus-forge \
		--source . \
		--platform managed \
		--region us-central1 \
		--allow-unauthenticated
	@echo "$(GREEN)✓ Deployed to Google Cloud Run$(NC)"

validate-deployment: ## Validate production deployment
	@echo "$(CYAN)Validating deployment...$(NC)"
	python scripts/validate_deployment.py
	@echo "$(GREEN)✓ Deployment validated$(NC)"

deploy-k8s: ## Deploy to Kubernetes
	@echo "$(CYAN)Deploying to Kubernetes...$(NC)"
	kubectl apply -f k8s/
	@echo "$(GREEN)✓ Deployed to Kubernetes$(NC)"

monitor-k8s: ## Monitor Kubernetes deployment
	@echo "$(CYAN)Monitoring Kubernetes deployment...$(NC)"
	kubectl get pods -w

##@ Debugging & Monitoring

check-agent-health: ## Check health of all agents
	@echo "$(CYAN)Checking agent health...$(NC)"
	python -m nexus_forge.tools.health_check
	@echo "$(GREEN)✓ Agent health check complete$(NC)"

debug-agent: ## Debug a specific agent (usage: make debug-agent AGENT=starri)
	@echo "$(CYAN)Debugging agent: $(AGENT)...$(NC)"
	python -m nexus_forge.tools.debug_agent --agent $(AGENT)

profile-generation: ## Profile app generation performance
	@echo "$(CYAN)Profiling generation performance...$(NC)"
	python -m nexus_forge.tools.profiler --task generation
	@echo "$(GREEN)✓ Profiling complete$(NC)"

analyze-logs: ## Analyze application logs
	@echo "$(CYAN)Analyzing logs...$(NC)"
	python -m nexus_forge.tools.log_analyzer
	@echo "$(GREEN)✓ Log analysis complete$(NC)"

health-check: ## Run comprehensive health check
	@echo "$(CYAN)Running health check...$(NC)"
	curl -f http://localhost:8000/health || exit 1
	@echo "$(GREEN)✓ Health check passed$(NC)"

test-db-connection: ## Test database connection
	@echo "$(CYAN)Testing database connection...$(NC)"
	python -m nexus_forge.tools.test_db
	@echo "$(GREEN)✓ Database connection successful$(NC)"