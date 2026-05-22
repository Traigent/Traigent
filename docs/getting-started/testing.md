# Testing Guide

This guide covers running tests in the Traigent SDK project.

## Quick Start

### Install Test Dependencies

```bash
# Using pip
pip install -e ".[test]"

# Or install dev dependencies (includes test + linting tools)
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests through the legacy shell fixture path
# (may emit DeprecationWarning; prefer enable_mock_mode_for_quickstart() in code)
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest

# Run with coverage through the same legacy shell fixture path
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest --cov=traigent --cov-report=html

# Run all unit tests with repo-default ignores
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest tests/unit -q --tb=short

# Run tests matching a pattern
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest -k "test_optimization" -v
```

## Test Dependencies

The `[test]` extra includes:

- **pytest>=7.0.0** - Testing framework
- **pytest-asyncio>=0.21.0** - Async test support
- **pytest-cov>=4.0.0** - Coverage reporting
- **pytest-mock>=3.10.0** - Mocking utilities
- **coverage>=7.0.0** - Code coverage tracking
- **ragas>=0.3.6** - RAG evaluation metrics
- **rapidfuzz>=3.14.0** - Fuzzy string matching

## Test Categories

### Unit Tests

Located in `tests/unit/`, these test individual components in isolation:

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific module tests
pytest tests/unit/core/ -v
pytest tests/unit/api/ -v
pytest tests/unit/security/ -v
```

### Integration Tests

Located in `tests/integration/`, these test cross-component functionality:

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run optimizer integration tests
pytest tests/integration/optimizers/ -v
```

### Security Tests

Located in `tests/security/`, these validate security features:

```bash
# Run security tests
pytest tests/security/ -v
```

## Coverage Reports

### Generate Coverage Report

```bash
# HTML report (opens in browser)
pytest --cov=traigent --cov-report=html
python -m http.server -d htmlcov 8080  # View at http://localhost:8080

# Terminal report
pytest --cov=traigent --cov-report=term-missing

# XML report (for CI/CD)
pytest --cov=traigent --cov-report=xml
```

### Coverage Targets

Use the generated report (`coverage.xml` or `htmlcov/`) to track current coverage. Follow any active sprint or team-specific targets in the repository tracking docs.

## Mock Mode Testing

Traigent includes a mock mode for testing without real API calls. In local tutorial or test code, use the in-code helper:

```python
from traigent.testing import enable_mock_mode_for_quickstart

enable_mock_mode_for_quickstart()
```

For shell-only fixtures and older scripts, the legacy env var still works outside production, but direct user-set activation emits `DeprecationWarning`:

```bash
TRAIGENT_MOCK_LLM=true pytest tests/
TRAIGENT_MOCK_LLM=true python examples/core/rag-optimization/run.py
```

**Mock Mode Features:**
- No API keys required.
- Fast execution — LLM calls are intercepted and replaced with canned
  responses so trials don't hit the network.
- Evaluator scoring is unchanged — your `metric_functions`, custom
  evaluator, or the built-in `LocalEvaluator` accuracy calculator runs
  the same scoring logic in both mock and real modes. There is no
  fabricated random-score path.

> Note: The legacy `MockModeOptions` fields (`enabled`,
> `override_evaluator`, `base_accuracy`, `variance`) are kept on the
> schema for backwards compatibility but are inert — mock mode is
> activated by `traigent.testing.enable_mock_mode_for_quickstart()`
> in local code, not by passing `mock=...`. The legacy
> `TRAIGENT_MOCK_LLM=true` env var remains for shell fixtures and
> backwards compatibility, but emits `DeprecationWarning` when users set
> it directly. See issue #874.

## Development Testing Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dev dependencies (includes test)
pip install -e ".[dev]"
```

### 2. Run Tests Before Committing

```bash
# Run all tests through the legacy shell fixture path
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest

# Run with coverage through the same legacy shell fixture path
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest --cov=traigent --cov-report=term-missing

# Run linters
ruff check traigent/
black traigent/ --check
mypy traigent/
```

### 3. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Testing Best Practices

### Writing Tests

1. **Organize by module**: Match test structure to source structure
2. **Use fixtures**: Leverage pytest fixtures for setup/teardown
3. **Mock external calls**: Use `pytest-mock` for API calls
4. **Test edge cases**: Include boundary conditions and error cases
5. **Use parametrize**: Test multiple inputs with `@pytest.mark.parametrize`

### Example Test

```python
import pytest
from traigent.api.decorators import optimize

@pytest.mark.asyncio
async def test_optimization_basic():
    """Test basic optimization workflow"""

    @optimize(
        configuration_space={"x": [1, 2, 3]},
        objectives=["accuracy"],
        eval_dataset="test.jsonl"
    )
    def test_func(x: int) -> str:
        return f"result-{x}"

    # Run optimization
    results = await test_func.optimize()

    # Assertions
    assert results.best_config is not None
    assert "x" in results.best_config
    assert results.best_config["x"] in [1, 2, 3]
```

## Troubleshooting

### Import Errors

```bash
# Make sure package is installed in editable mode
pip install -e ".[recommended]"

# Check installation
python -c "import traigent; print(traigent.__version__)"
```

### Test Discovery Issues

```bash
# Run from project root
cd /path/to/Traigent
pytest

# Explicitly specify test directory
pytest tests/ -v
```

### Async Test Issues

```bash
# Make sure pytest-asyncio is installed
pip install pytest-asyncio

# Check pytest configuration in pyproject.toml
grep -A 5 'tool.pytest' pyproject.toml
```

### Mock Mode Not Working

```bash
# Legacy shell fixture path only
echo $TRAIGENT_MOCK_LLM
```

Preferred explicit activation in local tutorial or test code:

```python
from traigent.testing import enable_mock_mode_for_quickstart

enable_mock_mode_for_quickstart()
```

## 📈 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[test]"

    - name: Run tests
      run: pytest --cov=traigent --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 🔗 Related Documentation

- [Installation Guide](installation.md)
- [Contributing Guide](../contributing/CONTRIBUTING.md)
- [Development Setup](../architecture/project-structure.md)

---

**Questions?** Check our [GitHub Issues](https://github.com/Traigent/Traigent/issues) or [Discussions](https://github.com/Traigent/Traigent/discussions).
