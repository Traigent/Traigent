# Makefile for Traigent SDK Development
# Run 'make help' to see available commands

.PHONY: help install install-dev test test-unit test-integration test-coverage lint format security clean analyze test-validation test-validation-unit test-validation-failures test-validation-traced jaeger-start jaeger-stop analyze-traces sonar-scan sonar-local-start sonar-local-stop sonar-local-down sonar-local-clean sonar-local sonar-local-issues

# Variables
PYTHON ?= .venv/bin/python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
BANDIT := $(PYTHON) -m bandit
COVERAGE := $(PYTHON) -m coverage

# Directories
SRC_DIR := traigent
TEST_DIR := tests
# NOTE: examples, playground, paper_experiments moved to TraigentDemo

help:  ## Show this help message
	@echo "Traigent SDK Development Commands"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	$(PIP) install -e .

install-dev:  ## Install package with all development dependencies
	$(PIP) install -e ".[all,dev,dspy,docs]"
	$(PYTHON) -m pre_commit install

test:  ## Run all tests
	$(PYTEST) $(TEST_DIR) -v

test-unit:  ## Run unit tests only
	$(PYTEST) $(TEST_DIR)/unit -v

test-integration:  ## Run integration tests only
	$(PYTEST) $(TEST_DIR)/integration -v

test-validation:  ## Run optimizer validation tests
	TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true $(PYTEST) $(TEST_DIR)/optimizer_validation -v

test-validation-unit:  ## Run optimizer validation unit tests only
	TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true $(PYTEST) $(TEST_DIR)/optimizer_validation -v -m "unit"

test-validation-failures:  ## Run optimizer validation failure tests
	TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true $(PYTEST) $(TEST_DIR)/optimizer_validation/failures -v

test-validation-traced:  ## Run validation tests with OpenTelemetry tracing (requires Jaeger)
	TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true \
	OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
	TRAIGENT_TRACE_ENABLED=true \
	$(PYTEST) $(TEST_DIR)/optimizer_validation -v

jaeger-start:  ## Start Jaeger for trace visualization (requires Docker)
	@echo "Starting Jaeger all-in-one container..."
	@docker run -d --name traigent-jaeger \
		-p 16686:16686 \
		-p 4317:4317 \
		-p 4318:4318 \
		jaegertracing/all-in-one:latest || echo "Jaeger may already be running"
	@echo "Jaeger UI available at: http://localhost:16686"

jaeger-stop:  ## Stop Jaeger container
	@echo "Stopping Jaeger container..."
	@docker stop traigent-jaeger && docker rm traigent-jaeger || echo "Jaeger not running"

analyze-traces:  ## Analyze trace files from test runs
	$(PYTHON) scripts/analyze_traces.py $(TEST_DIR)/optimizer_validation/traces

test-coverage:  ## Run tests with coverage report
	$(COVERAGE) run -m pytest $(TEST_DIR) -v
	$(COVERAGE) report -m
	$(COVERAGE) html
	@echo "Coverage report generated in htmlcov/index.html"

lint:  ## Run all linters (ruff, mypy, bandit)
	@echo "Running Ruff..."
	$(RUFF) check $(SRC_DIR) --fix
	@echo "Running MyPy..."
	$(MYPY) $(SRC_DIR) --install-types --non-interactive
	@echo "Running Bandit..."
	$(BANDIT) -r $(SRC_DIR) -ll --skip B101,B601

format:  ## Format code with black and isort
	@echo "Formatting with Black..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	@echo "Sorting imports with isort..."
	$(PYTHON) -m isort $(SRC_DIR) $(TEST_DIR) --profile black

security:  ## Run security checks
	@echo "Running Bandit security scan..."
	$(BANDIT) -r $(SRC_DIR) -f json -o security_report.json
	@echo "Checking for hardcoded secrets..."
	@grep -r "sk-proj\|sk-ant\|api_key\|secret\|password" --include="*.py" $(SRC_DIR) || echo "No hardcoded secrets found"

clean:  ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

optuna-benchmarks:  ## Run Optuna vs baseline benchmark suite
	$(PYTHON) scripts/benchmarks/run_optuna_benchmarks.py

quality-check:  ## Run all quality checks (lint, format check, tests)
	@echo "Running quality checks..."
	$(MAKE) lint
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(MAKE) test-coverage
	$(MAKE) security

quick-fix:  ## Quick fix common issues (format, simple lint fixes)
	@echo "Applying quick fixes..."
	$(MAKE) format
	$(RUFF) check $(SRC_DIR) --fix --unsafe-fixes
	@echo "Quick fixes applied!"

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

install-hooks:  ## Install git pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

update-deps:  ## Update all dependencies to latest versions
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,integrations,bayesian,docs]"

dev-server:  ## Run the Streamlit playground UI (moved to TraigentDemo)
	@echo "NOTE: Playground has been moved to TraigentDemo project"
	@echo "Run: cd ../TraigentDemo && streamlit run playground/traigent_control_center.py"

docs:  ## Build documentation
	cd docs && $(MAKE) html
	@echo "Documentation built in docs/_build/html/"

# Development workflow shortcuts
dev: install-dev install-hooks  ## Complete development setup

check: quality-check  ## Alias for quality-check

fix: quick-fix  ## Alias for quick-fix

# SonarQube scanning
sonar-scan:  ## Run SonarQube scan using local config file
	@if ! command -v sonar-scanner >/dev/null 2>&1; then \
		echo "Error: sonar-scanner not found. Install from https://docs.sonarqube.org/latest/analyzing-source-code/scanners/sonarscanner/"; \
		exit 1; \
	fi
	@if [ ! -f "sonar-project.local.properties" ]; then \
		echo "Error: sonar-project.local.properties not found"; \
		exit 1; \
	fi
	@if [ -z "$$SONAR_LOCAL_TOKEN" ]; then \
		echo "Error: SONAR_LOCAL_TOKEN not set."; \
		echo "Create a token at http://localhost:9000 -> My Account -> Security -> Generate Tokens"; \
		exit 1; \
	fi
	sonar-scanner -Dproject.settings=sonar-project.local.properties -Dsonar.token=$$SONAR_LOCAL_TOKEN
	@echo "Scan complete. View results at: http://localhost:9000/dashboard?id=TraigentSDK"

# Local SonarQube (avoids consuming cloud tokens)
sonar-local-start:  ## Start local SonarQube server (Docker)
	@echo "Starting local SonarQube..."
	cd scripts/sonarqube-local && docker compose up -d
	@echo "Waiting for SonarQube to be ready (this may take 1-2 minutes)..."
	@until curl -s http://localhost:9000/api/system/status 2>/dev/null | grep -q '"status":"UP"'; do \
		echo "  Still starting..."; \
		sleep 5; \
	done
	@echo "SonarQube is ready at http://localhost:9000 (login: admin/admin)"

sonar-local-stop:  ## Stop local SonarQube server
	cd scripts/sonarqube-local && docker compose stop

sonar-local-down:  ## Stop and remove local SonarQube (keeps data volumes)
	cd scripts/sonarqube-local && docker compose down

sonar-local-clean:  ## Remove local SonarQube and all data
	cd scripts/sonarqube-local && docker compose down -v

sonar-local:  ## Run SonarQube analysis locally (requires sonar-local-start first)
	@if ! curl -s http://localhost:9000/api/system/status 2>/dev/null | grep -q '"status":"UP"'; then \
		echo "Error: Local SonarQube not running. Start it with: make sonar-local-start"; \
		exit 1; \
	fi
	@if [ -z "$$SONAR_LOCAL_TOKEN" ]; then \
		echo "Error: SONAR_LOCAL_TOKEN not set. Source .env.local or set it manually."; \
		echo "Create a token at http://localhost:9000 -> My Account -> Security -> Generate Tokens"; \
		exit 1; \
	fi
	$$HOME/sonar-scanner/bin/sonar-scanner \
		-Dproject.settings=scripts/sonarqube-local/sonar-project-local.properties \
		-Dsonar.login=$$SONAR_LOCAL_TOKEN
	@echo ""
	@echo "View results at: http://localhost:9000/dashboard?id=traigent-local"

sonar-local-issues:  ## Show issues from local SonarQube
	@if [ -z "$$SONAR_LOCAL_TOKEN" ]; then \
		echo "Error: SONAR_LOCAL_TOKEN not set"; \
		exit 1; \
	fi
	@curl -s -u "$$SONAR_LOCAL_TOKEN:" "http://localhost:9000/api/issues/search?componentKeys=traigent-local&statuses=OPEN&ps=50" \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Open Issues: {d[\"total\"]}'); [print(f'  [{i[\"severity\"]}] {i[\"component\"].split(\":\")[-1]}:{i.get(\"line\",\"?\")} - {i[\"message\"][:60]}...') for i in d.get('issues',[])]"
