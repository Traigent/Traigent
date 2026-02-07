# TraiGent Maintenance Guide

This document catalogs all maintenance systems, validation tools, and developer utilities in the TraiGent project.

## Quick Reference

| System | Location | Command |
|--------|----------|---------|
| Quality Check | Makefile | `make quality-check` |
| Code Review | Moved to TraigentDemo | `tools-and-utilities/code-review/` |
| Architecture Analysis | `tools/architecture/` | `./run_analysis.sh` |
| Test Evidence Validation | `tests/optimizer_validation/tools/` | `python -m tests.optimizer_validation.tools.validate_evidence` |
| Quality Manager | `scripts/maintenance/` | `python traigent_quality_manager.py` |
| Cleanup Manager | `scripts/maintenance/` | `python safe_cleanup_manager.py` |
| Documentation Health | `scripts/documentation/` | `python doc_dashboard.py` |

---

## 1. Code Quality Management

### 1.1 Unified Quality Manager

**Location**: `scripts/maintenance/traigent_quality_manager.py`

Unified tool for code quality reporting and automated fixing with rollback support.

```bash
# Generate quality report
python scripts/maintenance/traigent_quality_manager.py --report

# Preview fixes (dry run)
python scripts/maintenance/traigent_quality_manager.py --fix --dry-run

# Apply fixes with backup
python scripts/maintenance/traigent_quality_manager.py --fix

# Apply fixes without prompts
python scripts/maintenance/traigent_quality_manager.py --fix --auto-yes

# Rollback to previous state
python scripts/maintenance/traigent_quality_manager.py --list-backups
python scripts/maintenance/traigent_quality_manager.py --rollback <backup_id>
```

### 1.2 Makefile Targets

```bash
make format          # Format with Black + isort
make lint            # Run Ruff, MyPy, Bandit
make security        # Security scans (Bandit, secrets)
make quality-check   # Full quality pipeline
make quick-fix       # Format + unsafe Ruff fixes
```

### 1.3 Pre-commit Hooks

**Configuration**: `.pre-commit-config.yaml`

Hooks run automatically on commit:
- Black (formatting)
- Ruff (linting with auto-fix)
- detect-secrets (secret detection)
- Bandit (security)
- MyPy (type checking)
- Standard hooks (whitespace, EOF, YAML/JSON validation)

```bash
make install-hooks   # Install hooks
make pre-commit      # Run on all files
```

---

## 2. Code Review System

> **Moved**: The code review system has been moved to the TraigentDemo repository
> at `tools-and-utilities/code-review/`. See that repo for usage instructions.

### 2.1 Review Tracks

| Track | Focus |
|-------|-------|
| Code Quality | Maintainability, readability |
| Soundness | Logic correctness, edge cases |
| Performance | Efficiency, optimization |
| Security | Vulnerabilities, best practices |

### 2.2 Usage

```bash
# From the TraigentDemo repository:
cd ../TraigentDemo/tools-and-utilities/code-review/
python automation/run_all_validations.py --module traigent/core/cache_policy.py
```

### 2.3 Key Components

- `automation/run_all_validations.py` - Main orchestrator
- `automation/_shared/validator.py` - Report validation
- `automation/_shared/function_inventory.py` - Code enumeration
- `viewer/` - React + TypeScript web interface

---

## 3. Architecture Analysis

**Location**: `tools/architecture/`

7-step architecture analysis pipeline with drift detection.

### 3.1 Running Analysis

```bash
# Full analysis
./tools/architecture/run_analysis.sh

# Quick analysis (skip expensive operations)
./tools/architecture/run_analysis.sh --quick

# With coverage metrics
./tools/architecture/run_analysis.sh --with-coverage

# Update baseline after improvements
./tools/architecture/run_analysis.sh --update-baseline
```

### 3.2 Analysis Steps

1. **Main Analysis** - Diagrams, complexity, class hierarchy
2. **Call Graph** - Function call relationships
3. **Priority Issues** - High-priority architectural issues
4. **Focused Diagrams** - Top classes, hub modules
5. **SVG Rendering** - Visual diagrams
6. **Threshold Checking** - Complexity/size violations
7. **Baseline Comparison** - Drift detection

### 3.3 Output Artifacts

| File | Description |
|------|-------------|
| `output/architecture_report.md` | Full architecture report |
| `output/priority_issues.md` | High-priority issues |
| `output/drift_report.md` | Changes from baseline |
| `output/threshold_report.md` | Threshold violations |
| `output/svg/` | Visual diagrams |
| `baseline_metrics.json` | Comparison baseline |

---

## 4. Test Evidence Validation

**Location**: `tests/optimizer_validation/`

Comprehensive test evidence capture and validation system for optimizer tests.

### 4.1 Evidence Schema

**Schema File**: `specs/evidence_schema.json`

Required sections in test evidence:

| Section | Required Fields |
|---------|-----------------|
| `scenario` | `name`, `config_space`, `injection_mode`, `max_trials` |
| `expected` | `outcome` |
| `actual` | `type` |
| `validation_checks` | Array of checks with `passed` boolean |

### 4.2 Validation Tool

```bash
# Generate a JSON report
TRAIGENT_MOCK_LLM=true pytest tests/optimizer_validation/ \
    --json-report --json-report-file=report.json

# Validate the report
python -m tests.optimizer_validation.tools.validate_evidence report.json

# Show warnings too
python -m tests.optimizer_validation.tools.validate_evidence report.json -v

# Output as JSON
python -m tests.optimizer_validation.tools.validate_evidence report.json --json
```

### 4.3 Interactive Viewer

```bash
cd tests/optimizer_validation/viewer
python -m http.server 8765
# Open http://127.0.0.1:8765/
```

Features:
- Test browsing by dimension category
- Evidence display with schema validation
- Coverage analysis heatmap
- Gap identification

### 4.4 Knowledge Graph

Semantic coverage analysis for test dimensions.

```bash
cd tests/optimizer_validation/viewer
python knowledge_graph.py
# Open coverage_analysis.html
```

Tracks 12 dimensions:
- InjectionMode, ExecutionMode, Algorithm, ConfigSpaceType
- ObjectiveConfig, StopCondition, ParallelMode, ConstraintUsage
- FailureMode, Reproducibility, EvaluatorType, ExpectedOutcome

---

## 5. Documentation Health

**Location**: `scripts/documentation/`

### 5.1 Documentation Dashboard

```bash
python scripts/documentation/doc_dashboard.py
```

Metrics tracked:
- Coverage (documented vs undocumented)
- Freshness (recently updated)
- Quality (completeness scores)
- Consistency (format adherence)

### 5.2 Documentation Validation

```bash
python scripts/validation/validate_docs.py
```

---

## 6. Project Cleanup

**Location**: `scripts/maintenance/safe_cleanup_manager.py`

Safe cleanup with automatic backups and undo capability.

```bash
# Analyze what can be cleaned
python scripts/maintenance/safe_cleanup_manager.py --analyze

# Preview cleanup (dry run)
python scripts/maintenance/safe_cleanup_manager.py --cleanup --dry-run

# Interactive cleanup
python scripts/maintenance/safe_cleanup_manager.py --cleanup --interactive

# Automatic cleanup
python scripts/maintenance/safe_cleanup_manager.py --cleanup --auto

# Undo cleanup
python scripts/maintenance/safe_cleanup_manager.py --list-backups
python scripts/maintenance/safe_cleanup_manager.py --undo <backup_id>
```

---

## 7. Performance Monitoring

### 7.1 Performance Check Hook

**Location**: `scripts/hooks/performance_check.py`

Pre-push hook for regression detection.

```bash
python scripts/hooks/performance_check.py
```

### 7.2 Baseline Management

```bash
# Update baseline after verified improvements
python scripts/hooks/update_baseline.py
```

### 7.3 Tracing

```bash
# Start Jaeger
make jaeger-start

# Run tests with tracing
make test-validation-traced

# View traces
open http://localhost:16686

# Stop Jaeger
make jaeger-stop
```

---

## 8. CI/CD Workflows

**Location**: `.github/workflows/`

| Workflow | Purpose | Trigger |
|----------|---------|---------|
| `tests.yml` | Test suite with coverage | Push/PR to main/develop |
| `quality.yml` | Code quality pipeline | Push/PR |
| `architecture-analysis.yml` | Drift detection | PR to main |
| `traigent-ci-gates.yml` | Performance regression gates | PR |
| `documentation.yml` | Doc build validation | Push/PR |
| `test-examples.yml` | Example validation | Push/PR |
| `publish.yml` | Release publishing | Tags |
| `sonarcloud.yml` | SonarCloud analysis | Push/PR |

---

## 9. Testing Infrastructure

### 9.1 Test Categories

| Directory | Purpose | Command |
|-----------|---------|---------|
| `tests/unit/` | Unit tests | `make test-unit` |
| `tests/integration/` | Integration tests | `make test-integration` |
| `tests/optimizer_validation/` | Optimizer validation | `make test-validation` |
| `tests/security/` | Security tests | `pytest tests/security/` |
| `tests/performance/` | Performance tests | `pytest tests/performance/` |
| `tests/e2e/` | End-to-end tests | `pytest tests/e2e/` |

### 9.2 Coverage

```bash
make test-coverage   # Generate HTML coverage report
# Output: htmlcov/index.html
```

Target: >85% coverage

---

## 10. Analysis Scripts

**Location**: `scripts/analysis/`

| Script | Purpose |
|--------|---------|
| `find_deprecated_patterns.py` | Find deprecated patterns after refactoring |
| `find_old_files.py` | Identify unused files |
| `find_n_plus_one.py` | Detect N+1 query patterns |
| `duplicate_detector.py` | Code duplication detection |
| `deprecation_scanner.py` | Scan for deprecated usage |
| `generate_file_inventory.py` | Generate file inventory (CSV) |

---

## 11. Developer Workflows

### Daily Development

```bash
make install-dev     # One-time setup
make format          # Before committing
make lint            # Check issues
make test            # Run tests
```

### Weekly Maintenance

```bash
# Quality review
python scripts/maintenance/traigent_quality_manager.py --report

# Cleanup
python scripts/maintenance/safe_cleanup_manager.py --cleanup --interactive

# Architecture check
./tools/architecture/run_analysis.sh --quick
```

### Pre-Release

```bash
# Full analysis
./tools/architecture/run_analysis.sh --with-coverage

# Documentation health
python scripts/documentation/doc_dashboard.py

# Security scan
make security

# Full test suite
make test-coverage
```

### Code Review

> Moved to TraigentDemo repository. See `tools-and-utilities/code-review/`.

---

## 12. Traceability Tools

> **Moved**: Traceability and trace flow tools have been moved to the TraigentDemo
> repository at `tools-and-utilities/traceability/` and `tools-and-utilities/trace_flows/`.

---

## 13. Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, tool configs (mypy, pytest, coverage, ruff, black, isort) |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `scripts/config/mkdocs.yml` | Documentation site config |
| `scripts/config/tox.ini` | Tox automation config |

---

## Version History

| Date | Change |
|------|--------|
| 2025-12-27 | Added test evidence validation system |
| 2025-12-27 | Created this maintenance guide |
