# Testing Guidelines for TraiGent SDK

## Overview

This document provides guidelines for writing robust, isolated tests in the TraiGent SDK test suite. Following these guidelines ensures tests are reliable, maintainable, and free from flakiness.

## Key Principle: Test Isolation

**Every test must be completely isolated from other tests.** Tests should:
- Start with a clean state
- Not depend on execution order
- Not leak state to other tests
- Pass consistently whether run individually or in a batch

## Global State Management

### The Problem

TraiGent SDK uses several global singletons that can cause test interference:

1. **`_API_KEY_MANAGER`** (in `traigent.config.api_keys`)
   - Stores API keys globally
   - Maintains a `_warned` flag that persists across tests

2. **`_GLOBAL_CONFIG`** (in `traigent.api.functions`)
   - Stores SDK configuration globally
   - Accumulates changes across tests

3. **Warning State**
   - Python's warning system caches warnings
   - Can cause warnings to not trigger in subsequent tests

### The Solution

Our test suite implements automatic global state reset through:

1. **Automatic Reset Fixture** (`conftest.py`)
   - The `reset_global_state` fixture runs before and after EVERY test
   - Automatically resets all known global state
   - No manual intervention needed for basic tests

2. **Test Isolation Utilities** (`tests/utils/isolation.py`)
   - Provides additional isolation tools for complex scenarios
   - Includes state verification and leak detection

## Writing Isolated Tests

### Basic Test Class

For most test classes, inherit from `TestIsolationMixin`:

```python
from tests.utils.isolation import TestIsolationMixin

class TestMyFeature(TestIsolationMixin):
    """Test class with automatic isolation."""

    def test_something(self):
        # This test automatically runs with clean state
        pass
```

### Testing with API Keys

When testing API key functionality:

```python
class TestAPIKeys(TestIsolationMixin):
    def setup_method(self, method):
        super().setup_method(method)  # Important: call parent setup
        self.manager = APIKeyManager()
        # Manager starts clean due to global reset

    def test_api_key_warning(self):
        # Test warnings without worrying about state
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.manager.set_api_key("test", "key", "code")
            assert len(w) == 1  # Will always work
```

### Testing with Global Configuration

When testing global configuration:

```python
from traigent.api.functions import configure

def test_configuration():
    # Global config is automatically reset before this test
    configure(parallel_workers=4)
    # No need to manually reset - handled automatically
```

### Using Context Managers

For fine-grained control:

```python
from tests.utils.isolation import isolated_test_context

def test_with_isolation():
    with isolated_test_context():
        # Code here runs with guaranteed clean state
        perform_test_operations()
```

### Detecting State Leaks

To ensure your test doesn't leak state:

```python
from tests.utils.isolation import detect_state_leaks

@detect_state_leaks
def test_no_leaks():
    # This test will fail if it modifies global state
    perform_operations()
```

## Common Pitfalls and Solutions

### Pitfall 1: Manual State Management

❌ **Don't do this:**
```python
def test_something():
    _API_KEY_MANAGER._keys.clear()  # Manual reset
    _API_KEY_MANAGER._warned = False
    # Test code...
```

✅ **Do this instead:**
```python
def test_something():
    # Automatic reset handles this
    # Test code...
```

### Pitfall 2: Assuming Clean Import State

❌ **Don't do this:**
```python
def test_first_warning():
    # Assumes this is the first test to trigger warning
    manager.set_api_key("test", "key", "code")
    # May fail if another test ran first
```

✅ **Do this instead:**
```python
def test_first_warning():
    # State is always clean due to automatic reset
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        manager.set_api_key("test", "key", "code")
        assert len(w) == 1  # Always works
```

### Pitfall 3: Test Order Dependencies

❌ **Don't do this:**
```python
def test_step_1():
    configure(api_keys={"openai": "key"})

def test_step_2():
    # Assumes test_step_1 ran first
    key = get_api_key("openai")
    assert key == "key"  # Fails if run independently
```

✅ **Do this instead:**
```python
def test_api_key_flow():
    # Complete flow in one test
    configure(api_keys={"openai": "key"})
    key = get_api_key("openai")
    assert key == "key"
```

## Testing Patterns

### Pattern: Testing Warnings

```python
def test_warning_behavior():
    """Test that warnings are triggered correctly."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Trigger the warning
        trigger_warning_condition()

        # Verify warning
        assert len(w) == 1
        assert "expected message" in str(w[0].message)
```

### Pattern: Testing Singleton Behavior

```python
def test_singleton_modification():
    """Test modifications to singleton objects."""
    # Get reference to singleton
    from traigent.config.api_keys import _API_KEY_MANAGER

    # Modify it (automatically reset after test)
    _API_KEY_MANAGER.set_api_key("test", "value", "env")

    # Test behavior
    assert _API_KEY_MANAGER.get_api_key("test") == "value"
    # No cleanup needed - automatic reset handles it
```

### Pattern: Mock Environment Variables

```python
def test_env_vars():
    """Test environment variable handling."""
    with patch.dict("os.environ", {"TEST_API_KEY": "env-value"}):
        # Test with environment variable
        result = get_api_key_from_env("test")
        assert result == "env-value"
```

## Debugging Test Issues

### Symptom: Tests Pass Individually but Fail in Batch

**Likely Cause:** Global state pollution

**Diagnosis:**
1. Check if test modifies global singletons
2. Look for warning state dependencies
3. Check for module-level variables

**Solution:**
- Ensure test class inherits from `TestIsolationMixin`
- Verify `conftest.py` has the `reset_global_state` fixture
- Use `detect_state_leaks` decorator to identify the issue

### Symptom: Warnings Not Triggering

**Likely Cause:** Warning already triggered in previous test

**Solution:**
- Global reset fixture should handle this automatically
- If still issues, explicitly use `warnings.resetwarnings()`
- Use `warnings.catch_warnings()` context manager

### Symptom: Configuration Not Reset

**Likely Cause:** Test modifies configuration outside standard APIs

**Solution:**
- Use standard `configure()` function
- Don't directly modify `_GLOBAL_CONFIG`
- Rely on automatic reset fixture

## CI/CD Considerations

### Running Tests in CI

The test suite is designed to work consistently in CI environments:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=traigent --cov-report=term-missing

# Run specific test file
pytest tests/unit/config/test_api_keys.py

# Run with verbose output
pytest tests/ -v
```

### Parallel Test Execution

Our isolation ensures tests can run in parallel:

```bash
# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

## Best Practices Summary

1. **Always inherit from `TestIsolationMixin`** for test classes that touch global state
2. **Don't manually reset global state** - let the automatic fixture handle it
3. **Test warnings with `catch_warnings()`** context manager
4. **Write self-contained tests** that don't depend on execution order
5. **Use the isolation utilities** when you need fine-grained control
6. **Verify no state leaks** with the `detect_state_leaks` decorator
7. **Document any special isolation needs** in test docstrings

## Adding New Global State

If you add new global state to the SDK:

1. Update `tests/conftest.py:reset_global_state()` to reset it
2. Update `tests/utils/isolation.py:GlobalStateManager` to handle it
3. Add tests to verify isolation works correctly
4. Document the new global state in this guide

## Conclusion

Following these guidelines ensures our test suite remains:
- **Reliable**: Tests pass consistently
- **Fast**: No unnecessary setup/teardown
- **Maintainable**: Clear patterns and utilities
- **Scalable**: Easy to add new tests

Remember: **Isolated tests are happy tests!**
