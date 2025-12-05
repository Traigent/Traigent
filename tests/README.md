# TraiGent SDK Test Suite

This directory contains the main test suite for the TraiGent SDK.

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

The following directories contain demo and example code with their own test requirements and are **not** part of the main SDK test suite:

- `demos/` - Demo applications (may require TensorFlow, specific databases, etc.)
- `playground/` - Experimental code and LangChain examples

These directories are automatically excluded by pytest.ini configuration.

## Test Standards

See [TEST_NAMING_STANDARDS.md](./TEST_NAMING_STANDARDS.md) for test file naming conventions.
