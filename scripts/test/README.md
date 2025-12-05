# Scripts Test Suite

This directory contains comprehensive tests for all scripts in the TraiGent SDK.

## Overview

The test suite ensures the reliability and correctness of development tools and scripts, with a target of >85% code coverage.

## Test Structure

```
test/
├── __init__.py              # Test package initialization
├── test_install_dev.py      # Tests for install-dev.sh script
├── test_utils.py            # Utility functions for testing
├── run_tests.py             # Test runner with coverage
├── requirements-test.txt    # Test dependencies
└── README.md               # This file
```

## Running Tests

### Quick Start

```bash
# Run all tests
python scripts/test/run_tests.py

# Run specific test file
python -m pytest scripts/test/test_install_dev.py -v

# Run with coverage report
python -m pytest scripts/test/ --cov=scripts --cov-report=html
```

### With Coverage

The test suite includes coverage reporting to ensure comprehensive testing:

```bash
# Install test dependencies
pip install -r scripts/test/requirements-test.txt

# Run tests with coverage
cd scripts/test
python run_tests.py
```

This will generate:
- Console coverage report
- HTML coverage report in `scripts/test/htmlcov/`

## Test Categories

### 1. Unit Tests
- Script existence and permissions
- Shebang validation
- Function and variable extraction
- Error handling verification

### 2. Integration Tests
- Script syntax validation
- Shell command execution
- Environment setup verification
- Dependency installation flows

### 3. Mock Tests
- Python version checking
- Package installation simulation
- Error condition handling
- Command output validation

## Writing New Tests

When adding new scripts, create corresponding test files:

1. Create `test_<script_name>.py` in this directory
2. Import test utilities: `from test_utils import *`
3. Create test class inheriting from `unittest.TestCase`
4. Use provided utilities for common assertions
5. Aim for >85% code coverage

### Example Test

```python
import unittest
from pathlib import Path
from test_utils import ScriptTestCase, validate_shell_script

class TestMyScript(unittest.TestCase):
    def setUp(self):
        self.script_path = Path(__file__).parent.parent / "my_script.sh"

    def test_script_exists(self):
        self.assertTrue(self.script_path.exists())

    def test_script_syntax(self):
        valid, error = validate_shell_script(self.script_path)
        self.assertTrue(valid, f"Syntax error: {error}")
```

## Coverage Goals

- **Target**: >85% coverage for all scripts
- **Current**: ~90% for install-dev.sh
- **Excluded**: External command execution, system-specific paths

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Script Tests
  run: |
    pip install -r scripts/test/requirements-test.txt
    python scripts/test/run_tests.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root
2. **Missing Dependencies**: Install test requirements first
3. **Permission Errors**: Scripts need executable permissions
4. **Bash Not Found**: Some tests require bash shell

### Debug Mode

Run tests with verbose output:

```bash
python -m pytest scripts/test/ -vv -s
```

## Contributing

When modifying scripts:
1. Update corresponding tests
2. Ensure coverage remains >85%
3. Add new test cases for new functionality
4. Run full test suite before committing
