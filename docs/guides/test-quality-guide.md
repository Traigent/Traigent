# Test Quality Guide

This guide documents test quality best practices and anti-patterns identified through meta-analysis of the Traigent test suite. It provides actionable guidance for writing high-quality tests.

## Quick Reference

### Critical Anti-Patterns to Avoid

| Code | Name | Description | Severity |
|------|------|-------------|----------|
| IT-VRO | Validator Reliance Only | Relying solely on a validator abstraction without explicit assertions | HIGH |
| IT-VTA | Vacuous Truth Assertions | Assertions that are always true (e.g., `len(x) >= 0`) | MEDIUM |
| IT-CBM | Condition-Behavior Mismatch | Test setup doesn't trigger the intended behavior | HIGH |
| IT-NTV | No Trial Verification | Tests that don't verify trial content | MEDIUM |
| IT-NSR | No Stop Reason Check | Tests named for stop conditions that don't verify stop_reason | MEDIUM |

## Anti-Pattern Examples

### IT-VRO: Validator Reliance Only

**Problem**: Relying solely on a validator abstraction hides what's being tested and can mask bugs.

```python
# BAD - Only checks validator
async def test_optimization_completes(scenario_runner, result_validator):
    scenario = TestScenario(...)
    func, result = await scenario_runner(scenario)
    validation = result_validator(scenario, result)
    assert validation.passed  # What is actually being tested?
```

```python
# GOOD - Explicit assertions THEN validator
async def test_optimization_completes(scenario_runner, result_validator):
    scenario = TestScenario(...)
    func, result = await scenario_runner(scenario)

    # EXPLICIT ASSERTIONS - clearly document what we're testing
    assert not isinstance(result, Exception), f"Unexpected error: {result}"
    assert hasattr(result, "trials"), "Result should have trials attribute"
    assert len(result.trials) >= 1, "Should complete at least one trial"
    for trial in result.trials:
        assert trial.config, "Each trial should have a config"

    # Validator AFTER explicit checks for additional coverage
    validation = result_validator(scenario, result)
    assert validation.passed, validation.summary()
```

### IT-VTA: Vacuous Truth Assertions

**Problem**: Assertions that are mathematically always true provide no test value.

```python
# BAD - These assertions can never fail
assert len(result.trials) >= 0  # len() is always >= 0
assert True
assert not isinstance(result, Exception)  # Weak - any non-exception passes
assert result is not None or result is None  # Tautology
```

```python
# GOOD - Meaningful assertions with specific bounds
assert len(result.trials) >= 1, "Should have at least one trial"
assert len(result.trials) == expected_count, f"Expected {expected_count} trials"
assert result.stop_reason == "max_trials", f"Wrong stop reason: {result.stop_reason}"
assert isinstance(error, ValueError), f"Expected ValueError, got {type(error)}"
```

### IT-CBM: Condition-Behavior Mismatch

**Problem**: Test setup doesn't actually trigger the behavior being tested.

```python
# BAD - max_trials can never trigger (config space exhausts first)
scenario = TestScenario(
    name="test_max_trials_stops",
    max_trials=10,
    config_space={"model": ["a", "b"]}  # Only 2 combinations!
)
# Grid search will exhaust after 2 trials, not 10
# Test name suggests max_trials but it never triggers
```

```python
# GOOD - Config space larger than max_trials
scenario = TestScenario(
    name="test_max_trials_stops",
    max_trials=3,
    config_space={"model": ["a", "b", "c", "d", "e"]}  # 5 > 3
)
# Now max_trials will actually trigger at trial 3
# Also verify the stop reason!
assert result.stop_reason == "max_trials"
```

## Test Writing Checklist

Before writing any test, answer these questions:

1. **What specific behavior am I testing?**
   - Can you state it in one sentence?

2. **What explicit assertions verify this behavior?**
   - List the specific assertions, not just "it passes"

3. **Does my test setup actually trigger this behavior?**
   - For stop conditions: is the config space large enough?
   - For failures: does the input actually cause the failure?

4. **Would this test fail if the behavior was broken?**
   - If you removed the feature, would the test catch it?

## Validation Tools

### 1. Static Analysis - Test Weakness Analyzer

Detects anti-patterns through AST analysis:

```bash
# Analyze optimizer validation tests
python -m tests.optimizer_validation.tools.test_weakness_analyzer

# Analyze specific directory
python -m tests.optimizer_validation.tools.test_weakness_analyzer -d tests/unit/

# JSON output for CI integration
python -m tests.optimizer_validation.tools.test_weakness_analyzer --output json
```

### 2. Assertion Lint - CI Integration

Lightweight linter suitable for CI pipelines:

```bash
# Run lint checks
python -m tests.optimizer_validation.tools.lint_test_assertions

# Strict mode (fail on warnings)
python -m tests.optimizer_validation.tools.lint_test_assertions --strict

# GitHub Actions annotation format
python -m tests.optimizer_validation.tools.lint_test_assertions --format github
```

**Lint Rules:**
- `T001`: result_validator called without asserting validation.passed (ERROR)
- `T002`: No explicit behavior assertions (WARNING)
- `T003`: Vacuous assertion detected (ERROR/WARNING)

### 3. Mutation Testing - Oracle Strength

Tests whether your test suite catches semantic bugs:

```bash
# Run all mutation operators
python -m tests.optimizer_validation.tools.mutation_oracle

# Run specific mutant
python -m tests.optimizer_validation.tools.mutation_oracle --mutant MUT-SKIP-TRIALS

# Save detailed report
python -m tests.optimizer_validation.tools.mutation_oracle --report mutation_report.json
```

**Mutation Operators:**
| Operator | Description | Catches |
|----------|-------------|---------|
| MUT-SKIP-TRIALS | Returns empty trials | Tests without trial verification |
| MUT-IGNORE-MAX | Ignores max_trials | Tests without stop condition checks |
| MUT-SWAP-DIRECTION | Swaps maximize/minimize | Tests without score validation |
| MUT-DROP-METRICS | Removes metrics | Tests that don't validate metrics |
| MUT-FORCE-VALIDATOR | Validator always passes | Validator-only tests |
| MUT-WRONG-STOP-REASON | Returns wrong stop_reason | Tests without stop_reason checks |

## Test Structure Templates

### Unit Test Template

```python
"""Unit tests for {module_name}.

Tests cover:
- Basic functionality
- Edge cases
- Error handling
- Type validation
"""

import pytest
from {module} import {ClassName}


class Test{ClassName}:
    """Tests for {ClassName}."""

    def test_basic_functionality(self) -> None:
        """Test {specific behavior being tested}."""
        # ARRANGE - set up test data
        obj = ClassName(param="value")
        expected = "expected_result"

        # ACT - perform the action
        result = obj.method()

        # ASSERT - verify explicit outcomes
        assert result == expected, f"Expected {expected}, got {result}"
        assert isinstance(result, str), f"Expected str, got {type(result)}"

    def test_edge_case_empty_input(self) -> None:
        """Test behavior with empty input."""
        obj = ClassName()

        result = obj.method("")

        # Don't just check it doesn't error - verify specific behavior
        assert result == "", "Empty input should return empty output"

    def test_error_on_invalid_input(self) -> None:
        """Test that invalid input raises appropriate error."""
        obj = ClassName()

        with pytest.raises(ValueError, match="must be positive"):
            obj.method(-1)
```

### Async Integration Test Template

```python
"""Integration tests for {feature}.

Tests cover:
- End-to-end workflow
- Component interaction
- State transitions
"""

import pytest


class TestIntegration{Feature}:
    """Integration tests for {feature}."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self) -> None:
        """Test complete workflow from input to output."""
        # ARRANGE
        config = {"param": "value"}
        expected_trials = 3

        # ACT
        result = await run_workflow(config)

        # ASSERT - verify all important aspects
        assert result.status == "completed", f"Wrong status: {result.status}"
        assert len(result.trials) == expected_trials
        assert result.best_config is not None
        assert result.stop_reason == "max_trials"

        # Verify trial structure
        for trial in result.trials:
            assert trial.config, "Trial missing config"
            assert "accuracy" in trial.metrics, "Trial missing accuracy metric"
```

## Running Tests

Always use mock mode for local development:

```bash
# Run all tests with mock mode
TRAIGENT_MOCK_MODE=true pytest tests/

# Run with coverage report
TRAIGENT_MOCK_MODE=true pytest tests/ --cov=traigent --cov-report=term-missing

# Run specific test file with verbose output
TRAIGENT_MOCK_MODE=true pytest tests/unit/api/test_decorators.py -v

# Run tests matching pattern
TRAIGENT_MOCK_MODE=true pytest -k "test_constraint" -v
```

## Common Mistakes

1. **Trusting mock mode alone**
   - Mock mode is lenient; add explicit assertions

2. **Skipping validators entirely**
   - Use validators, but add explicit assertions BEFORE them

3. **Testing only happy path**
   - Include edge cases and error conditions

4. **Ignoring test names**
   - If test is `test_timeout_stops`, verify `stop_reason == "timeout"`

5. **Using vacuous checks**
   - Every assertion should be capable of failing

6. **Forgetting type annotations**
   - Add `-> None` return type to test methods

## Reference Documentation

For detailed analysis and methodology:

- **Meta-Analysis Report**: `tests/optimizer_validation/META_ANALYSIS_REPORT.md`
- **Independent Analysis**: `tests/optimizer_validation/INDEPENDENT_META_ANALYSIS.md`
- **Tools README**: `tests/optimizer_validation/tools/README.md`
- **CLAUDE.md Guidelines**: See "Test Quality Guidelines" section

## Summary

Write tests that:
- Have **explicit assertions** for the specific behavior under test
- Use **meaningful bounds** (not `>= 0`)
- **Match setup to intent** (stop conditions should be triggerable)
- **Validate oracle strength** using mutation testing
- Follow the **AAA pattern**: Arrange, Act, Assert
