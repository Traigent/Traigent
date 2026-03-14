# Traigent SDK Test Suite

This directory contains the main test suite for the Traigent SDK.

## Test Organization

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for component interactions
- `e2e/` - End-to-end tests for complete workflows
- `shared/` - Shared test utilities and mocks

## Running Tests

To run the full test suite:
```bash
pytest
```

To run specific test categories:
```bash
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/e2e/          # End-to-end tests only
```

## Excluded Directories

Certain directories (legacy tests, manual validation, bridges) are automatically excluded
by the `addopts` in `pyproject.toml` (`[tool.pytest.ini_options]`).

## Test Standards

See [TEST_NAMING_STANDARDS.md](./TEST_NAMING_STANDARDS.md) for test file naming conventions.
