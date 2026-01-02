# Traigent SDK Claude Instructions

## 🧠 Project Context
Traigent is a Python SDK for zero-code LLM optimization using decorators (`@traigent.optimize`). It intercepts calls, injects parameters, and optimizes against objectives (accuracy, cost, latency).

## 🏗️ Architecture & Patterns
- **Entry Point**: `traigent/api/decorators.py` (`@optimize`) wraps user functions in `OptimizedFunction` (`traigent/core/optimized_function.py`).
- **Orchestration**: `OptimizationOrchestrator` (`traigent/core/orchestrator.py`) manages the optimization loop and backend sync.
- **Integrations**: Adapters in `traigent/integrations/` (e.g., `LangChainAdapter`) handle framework-specific logic. **Do not inline framework logic**; use adapters.
- **Execution Modes**: `edge_analytics` (local + analytics), `mock` (testing), `cloud` (production).
- **Security**: `traigent/security/` handles JWT validation and encryption. **Never bypass auth in production code.**

## 🛠️ Development Workflow
- **Setup**: `make install-dev` (installs with `[dev,integrations,analytics,security]`).
- **Testing**: `TRAIGENT_MOCK_MODE=true pytest tests/` (Critical: Use mock mode to avoid API costs).
  - **Markers**: `pytest -m "unit"`, `pytest -m "integration"`, `pytest -m "security"`.
- **Linting**: `make lint` (runs Ruff, MyPy, Bandit). `make format` (Black, Isort).
- **Mock Mode**: Always use `TRAIGENT_MOCK_MODE=true` for local development and tests.

## 📐 Code Formatting & Linting (CRITICAL)

**Before committing any Python code changes, ALWAYS run:**

```bash
make format && make lint
```

Or manually:

```bash
# Format code
isort traigent/ tests/ examples/
black traigent/ tests/ examples/

# Lint code (use these exact commands - CI flags differ)
ruff check traigent/ --fix
black --check traigent/
isort --check-only traigent/
```

### Formatting Rules
- **Black**: Line length 88, Python 3.8+ target
- **isort**: Profile "black", line length 88 (must match black)
- **Ruff**: For linting only, not formatting

### ⚠️ CI vs Local Commands
The CI workflow uses different flags than local development:
- **CI**: `ruff check traigent/ --output-format=github` (for GitHub annotations)
- **Local**: `ruff check traigent/ --fix` (to auto-fix issues)

**Never use these flag combinations** (they are incompatible):
- `--output-format=github --statistics` ❌ (ruff doesn't support this)

### Pre-commit Checklist
1. Run `make format` (runs isort + black)
2. Run `make lint` (runs ruff + mypy + bandit)
3. Verify: `black --check traigent/ && isort --check-only traigent/`

## 🚨 Critical Rules
1. **Format Before Commit**: Always run `make format` before committing Python changes.
2. **Cross-Repo Policy**: Do not modify external systems. File issues for backend/upstream changes.
3. **Async-First**: Cloud operations (`traigent/cloud/`) must be async. Avoid blocking sync wrappers.
4. **Secrets**: NEVER hardcode secrets. Use `traigent.utils.env_config` or `os.environ`.
5. **Type Hints**: Required for all public APIs. Run `make lint` to verify.
6. **Dependencies**: Add new deps to `pyproject.toml` under the appropriate optional group (e.g., `integrations`).

## 🧩 Common Tasks
- **New Integration**: Create adapter in `traigent/integrations/`, register in `traigent/integrations/registry.py`.
- **New Metric**: Implement `BaseMetric` in `traigent/metrics/`, register in `metric_registry.py`.
- **Debug**: Set `TRAIGENT_LOG_LEVEL=DEBUG`.

## 🧪 Test Quality Guidelines (CRITICAL)

When writing tests, avoid these anti-patterns identified in our meta-analysis:

### Anti-Patterns to Avoid

1. **IT-VRO (Validator Reliance Only)**: Never rely solely on a validator abstraction without explicit assertions.

   ```python
   # BAD - relies only on validator
   validation = result_validator(scenario, result)
   assert validation.passed

   # GOOD - explicit assertions THEN validator
   assert len(result.trials) >= 1, "Should have trials"
   for trial in result.trials:
       assert trial.config, "Trial should have config"
   validation = result_validator(scenario, result)
   assert validation.passed
   ```

2. **IT-VTA (Vacuous Truth Assertions)**: Avoid assertions that are always true.

   ```python
   # BAD - always true
   assert len(result.trials) >= 0
   assert not isinstance(result, Exception)

   # GOOD - meaningful bounds
   assert len(result.trials) >= expected_min
   assert result.stop_reason == "max_trials"
   ```

3. **IT-CBM (Condition-Behavior Mismatch)**: Ensure test setup actually triggers the behavior under test.

   ```python
   # BAD - config space (2 items) smaller than max_trials (10)
   # max_trials never triggers; config_exhaustion triggers first
   scenario = TestScenario(max_trials=10, config_space={"model": ["a", "b"]})

   # GOOD - config space larger than max_trials
   scenario = TestScenario(max_trials=3, config_space={"model": ["a", "b", "c", "d", "e"]})
   ```

### Test Structure Best Practices

1. **Explicit over Implicit**: Always add explicit assertions for the specific behavior under test.
2. **Specific Values**: Assert exact expected values, not just types.
3. **Document Validator Contract**: Know what validators check and don't check.
4. **Unit Tests**: Prefer direct function calls with specific assertions over complex scenario runners.

### Reference
See `tests/optimizer_validation/META_ANALYSIS_REPORT.md` and `tests/optimizer_validation/INDEPENDENT_META_ANALYSIS.md` for detailed analysis of test quality issues.
