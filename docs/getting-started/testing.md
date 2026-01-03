# Testing Guide

This guide covers running tests in the Traigent SDK project.

## Quick Start

### Install Test Dependencies

```bash
# Using pip
pip install -e ".[test]"

# Using uv (faster)
uv pip install -e ".[test]"

# Or install dev dependencies (includes test + linting tools)
uv pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests (mock mode to avoid API costs)
TRAIGENT_MOCK_LLM=true pytest

# Run with coverage
TRAIGENT_MOCK_LLM=true pytest --cov=traigent --cov-report=html

# Run specific test file
TRAIGENT_MOCK_LLM=true pytest tests/unit/core/test_orchestrator.py

# Run specific test
TRAIGENT_MOCK_LLM=true pytest tests/unit/core/test_orchestrator.py::test_orchestrator_initialization -v

# Run tests matching a pattern
TRAIGENT_MOCK_LLM=true pytest -k "test_optimization" -v
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

Traigent includes a mock mode for testing without real API calls:

```bash
# Enable mock mode
export TRAIGENT_MOCK_LLM=true

# Run tests with mock mode
TRAIGENT_MOCK_LLM=true pytest tests/

# Run specific example in mock mode
TRAIGENT_MOCK_LLM=true python examples/core/hello-world/run.py
```

**Mock Mode Features:**
- No API keys required
- Realistic accuracy scores (0.75 +/- 0.25)
- Fast execution
- Deterministic results when you set `mock={"random_seed": 123}`

## Development Testing Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/traigent/traigent-sdk.git
cd Traigent

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dev dependencies (includes test)
uv pip install -e ".[dev]"
```

### 2. Run Tests Before Committing

```bash
# Run all tests
TRAIGENT_MOCK_LLM=true pytest

# Run with coverage
TRAIGENT_MOCK_LLM=true pytest --cov=traigent --cov-report=term-missing

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
pip install -e .

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

# Check pytest.ini configuration
cat pytest.ini
```

### Mock Mode Not Working

```bash
# Verify environment variable
echo $TRAIGENT_MOCK_LLM

# Set explicitly in test
import os
os.environ["TRAIGENT_MOCK_LLM"] = "true"
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
        pip install uv
        uv pip install -e ".[test]"

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
