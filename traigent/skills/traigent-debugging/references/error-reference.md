# Traigent Exception Reference

## Exception Hierarchy

```
Exception
  +-- TraigentError (base for all Traigent exceptions)
  |     +-- ConfigurationError
  |     |     +-- ProviderValidationError
  |     |     +-- ConfigurationSpaceError
  |     +-- ValidationError
  |     |     +-- TraigentValidationError
  |     |     +-- TVLValidationError
  |     |     |     +-- TVLConstraintError
  |     |     +-- DatasetValidationError
  |     |     +-- ObjectiveValidationError
  |     |     +-- JWTValidationError
  |     +-- InvocationError
  |     +-- EvaluationError
  |     +-- OptimizationError
  |     +-- OptimizationStateError
  |     +-- CostLimitExceeded
  |     +-- FeatureNotAvailableError
  |     +-- DataIntegrityError
  |     |     +-- MetricExtractionError
  |     |     +-- DTOSerializationError
  |     +-- ClientError
  |     |     +-- StandardizedClientError
  |     +-- AuthenticationError
  |     +-- ServiceError
  |     |     +-- TraigentConnectionError
  |     |     +-- ServiceUnavailableError
  |     |     +-- QuotaExceededError
  |     +-- SessionError
  |     +-- AgentExecutionError
  |     +-- PlatformCapabilityError
  |     +-- StorageError
  |     |     +-- TraigentStorageError
  |     +-- PluginError
  |     |     +-- PluginVersionError
  |     +-- RetryableError
  |     |     +-- RateLimitError
  |     |     +-- NetworkError
  |     +-- NonRetryableError
  |     +-- RetryError
  |     +-- TraigentTimeoutError
  |     +-- TrialPrunedError
  +-- UserWarning
        +-- TraigentWarning
              +-- ConfigAccessWarning
              +-- TraigentDeprecationWarning
```

## Exceptions Users Commonly Encounter

### TraigentError

**Base exception for all Traigent errors.**

| Attribute | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable error message. |
| `details` | `dict[str, Any]` | Additional error context. Default: `{}`. |

```python
from traigent.utils.exceptions import TraigentError

try:
    results = func.optimize(dataset="data.jsonl")
except TraigentError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
```

### ConfigurationError

**Invalid or malformed configuration.**

Raised when configuration values are invalid, features are unsupported, or required configuration is missing. Set `TRAIGENT_DEBUG=1` for full tracebacks.

| Inherited from | TraigentError |
|---|---|
| `message` | Error description |
| `details` | Additional context |

**Common triggers**:
- Non-list values in `configuration_space`
- Empty configuration space
- Invalid parameter names
- Unsupported execution mode

**Resolution**: Check the error message and fix the configuration. Use `TRAIGENT_DEBUG=1` for the full traceback.

### ProviderValidationError

**API key validation failure.**

Raised before optimization starts when provider API keys are invalid or missing.

| Attribute | Type | Description |
|---|---|---|
| `message` | `str` | Error description with all failed providers. |
| `failed_providers` | `list[tuple[str, str]]` | List of `(provider_name, error_type)` tuples. |
| `details` | `dict[str, Any]` | Additional context. |

**Resolution**:
1. Set correct API keys in environment variables
2. Or skip validation: `validate_providers=False` in decorator
3. Or set `TRAIGENT_SKIP_PROVIDER_VALIDATION=true`

### CostLimitExceeded

**Budget exceeded during optimization.**

| Attribute | Type | Description |
|---|---|---|
| `accumulated` | `float` | Total cost spent so far (USD). |
| `limit` | `float` | The configured cost limit (USD). |

```python
from traigent.utils.exceptions import CostLimitExceeded

try:
    results = func.optimize(dataset="data.jsonl")
except CostLimitExceeded as e:
    print(f"Spent ${e.accumulated:.2f} of ${e.limit:.2f} budget")
```

**Resolution**: Increase `cost_limit`, use cheaper models, or reduce `max_trials`.

### OptimizationStateError

**Invalid lifecycle state for the requested operation.**

| Attribute | Type | Description |
|---|---|---|
| `message` | `str` | What went wrong and what state is needed. |
| `current_state` | `str \| None` | The current lifecycle state. |
| `expected_states` | `list[str]` | States that would make this operation valid. |

**Common triggers**:
- Calling `traigent.get_config()` with no active trial and no applied config
- Calling `get_trial_config()` outside of an optimization trial
- Operations that require a specific lifecycle state

**Resolution**: Either run optimization first, or call `apply_best_config()` before using the function.

### InvocationError

**Function raised an exception during a trial.**

| Attribute | Type | Description |
|---|---|---|
| `message` | `str` | Error description. |
| `config` | `dict[str, Any] \| None` | The configuration that caused the failure. |
| `input_data` | `dict[str, Any] \| None` | The input data that was passed. |
| `original_error` | `Exception \| None` | The underlying exception from the user function. |

**Resolution**: Check `original_error` to see what actually failed inside your function.

### EvaluationError

**Evaluator function failed when scoring output.**

| Attribute | Type | Description |
|---|---|---|
| `message` | `str` | Error description. |
| `config` | `dict[str, Any] \| None` | The configuration being evaluated. |
| `original_error` | `Exception \| None` | The underlying evaluator exception. |

**Resolution**: Ensure your evaluator handles edge cases (None, empty strings, unexpected output formats).

### ValidationError

**Input validation failure.**

Base class for validation-related errors. Subclasses include:
- `DatasetValidationError` - invalid dataset format
- `ObjectiveValidationError` - invalid objective specification
- `TraigentValidationError` - general validation with suggestions

### FeatureNotAvailableError

**Missing plugin or optional dependency.**

| Attribute | Type | Description |
|---|---|---|
| `feature_name` | `str` | Name of the requested feature. |
| `plugin_name` | `str \| None` | Required plugin name. |
| `install_hint` | `str \| None` | pip install command to fix. |

**Resolution**: Run the `install_hint` command (e.g., `pip install traigent[integrations]`).

### DataIntegrityError

**Data corruption or invalid conversion detected.**

| Attribute | Type | Description |
|---|---|---|
| `message` | `str` | Error description. |
| `field` | `str \| None` | Field that failed validation. |
| `value` | `Any` | The invalid value. |

Subclass `MetricExtractionError` adds `trial_id`, `example_id`, and `original_error`.

### ServiceError

**Remote service communication failure.**

| Attribute | Type | Description |
|---|---|---|
| `service_name` | `str \| None` | Name of the service. |
| `endpoint` | `str \| None` | API endpoint that failed. |
| `status_code` | `int \| None` | HTTP status code. |

Subclasses: `TraigentConnectionError`, `ServiceUnavailableError`, `QuotaExceededError`.

### RateLimitError

**Provider rate limit exceeded.**

| Attribute | Type | Description |
|---|---|---|
| `retry_after` | `float \| None` | Suggested wait time in seconds. |

Traigent handles retries automatically for retryable errors. This surfaces when retries are exhausted.

## Catching Exceptions

### Catch all Traigent errors

```python
from traigent.utils.exceptions import TraigentError

try:
    results = func.optimize(dataset="data.jsonl")
except TraigentError as e:
    print(f"Traigent error: {e.message}")
```

### Catch specific errors

```python
from traigent.utils.exceptions import (
    CostLimitExceeded,
    ConfigurationError,
    ProviderValidationError,
    OptimizationStateError,
)

try:
    results = func.optimize(dataset="data.jsonl")
except CostLimitExceeded as e:
    print(f"Over budget: ${e.accumulated:.2f}")
except ProviderValidationError as e:
    print(f"Bad API keys: {e.failed_providers}")
except ConfigurationError as e:
    print(f"Config problem: {e.message}")
except OptimizationStateError as e:
    print(f"Wrong state: {e.current_state}, need: {e.expected_states}")
```

## Warnings

Traigent also issues warnings (non-fatal) for deprecated patterns:

| Warning | Meaning |
|---|---|
| `ConfigAccessWarning` | Using deprecated config access method. Use `traigent.get_config()` instead. |
| `TraigentDeprecationWarning` | A deprecated feature is being used. |
