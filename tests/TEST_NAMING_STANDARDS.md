# Test Naming Standards for TraiGent SDK

## Overview
This document defines the standardized naming conventions for test files across the TraiGent SDK to improve organization, discoverability, and maintainability.

## Naming Convention

### Format: `test_{component}_{functionality}.py`

- `component`: The specific module/class being tested
- `functionality`: The specific aspect or feature being tested

### Examples

#### Good Names
- `test_traigent_cloud_client.py` (tests TraiGentCloudClient)
- `test_bayesian_optimizer.py` (tests BayesianOptimizer)
- `test_decorator_framework_override.py` (tests decorator framework override feature)
- `test_retry_circuit_breaker.py` (tests circuit breaker in retry system)

#### Bad Names (to be renamed)
- `test_client.py` → `test_traigent_cloud_client.py`
- `test_models.py` → `test_cloud_models.py`
- `test_auth.py` → `test_cloud_authentication.py`
- `test_service.py` → `test_cloud_service.py`

## Directory Structure Standards

### Unit Tests: `tests/unit/{module}/{component}/`
```
tests/unit/
├── api/
│   ├── test_traigent_decorator.py
│   ├── test_api_functions.py
│   └── test_api_types.py
├── cloud/
│   ├── test_traigent_cloud_client.py
│   ├── test_cloud_authentication.py
│   ├── test_cloud_models.py
│   └── test_cloud_sessions.py
├── optimizers/
│   ├── test_bayesian_optimizer.py
│   ├── test_grid_optimizer.py
│   └── test_random_optimizer.py
```

### Integration Tests: `tests/integration/`
```
tests/integration/
├── test_decorator_cloud_integration.py
├── test_optimizer_evaluator_integration.py
└── test_end_to_end_workflows.py
```

### E2E Tests: `tests/e2e/`
```
tests/e2e/
├── test_user_optimization_workflow.py
├── test_privacy_preservation_workflow.py
└── test_hybrid_optimization_workflow.py
```

## Test Class Naming

### Format: `Test{Component}{Functionality}`

Examples:
- `TestTraiGentCloudClient`
- `TestBayesianOptimizer`
- `TestDecoratorFrameworkOverride`
- `TestCloudAuthentication`

## Test Method Naming

### Format: `test_{action}_{scenario}_{expected_result}`

Examples:
- `test_suggest_next_trial_with_history_returns_config`
- `test_authenticate_with_invalid_key_raises_error`
- `test_optimize_function_with_timeout_completes_successfully`

## Implementation Plan

### Phase 1: High-Impact Renames (Priority)
1. `test_client.py` → `test_traigent_cloud_client.py`
2. `test_models.py` → `test_cloud_models.py`
3. `test_auth.py` → `test_cloud_authentication.py`
4. `test_service.py` → `test_cloud_service.py`

### Phase 2: Comprehensive Standardization
1. Review all test files for naming consistency
2. Update imports in test files after renames
3. Update CI/CD configuration if needed
4. Update documentation references

## Benefits

1. **Improved Discoverability**: Developers can quickly find tests for specific components
2. **Better Organization**: Clear mapping between source code and test files
3. **Reduced Confusion**: No more ambiguous names like `test_client.py`
4. **Easier Maintenance**: Consistent patterns make adding new tests straightforward
5. **Better IDE Support**: Better autocomplete and navigation

## Automation

Consider adding a linter rule to enforce these naming conventions for new test files.
