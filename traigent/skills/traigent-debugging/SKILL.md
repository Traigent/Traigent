---
name: traigent-debugging
description: "Debug and troubleshoot Traigent optimization issues. Use when encountering CostLimitExceeded, ConfigurationError, OptimizationStateError, ModuleNotFoundError, or when optimization produces unexpected results. Covers mock mode, logging configuration, and common error resolution."
license: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
metadata:
  author: Traigent
  version: "1.0"
---

# Debugging and Troubleshooting Traigent

## When to Use

Use this skill when:

- An optimization run fails or produces unexpected results
- You encounter a Traigent exception (CostLimitExceeded, ConfigurationError, etc.)
- You need to test without real API keys (mock mode)
- You want to increase logging verbosity for diagnosis
- Your function works standalone but fails during optimization
- Import errors occur due to missing optional dependencies

## Quick Diagnostic

Enable detailed logging to see what Traigent is doing at each step:

```bash
# Full debug logging
export TRAIGENT_LOG_LEVEL=DEBUG

# Show full tracebacks for ConfigurationError (normally shows clean message only)
export TRAIGENT_DEBUG=1
```

Then run your optimization. Debug output includes:
- Configuration sampling decisions
- Trial execution start/stop/status
- Metric extraction and scoring
- Cost tracking per trial
- Backend communication (if using hybrid portal tracking)

## Common Errors

### ConfigurationError

**When raised**: Invalid or malformed configuration values, unsupported features, missing required configuration.

```
traigent.utils.exceptions.ConfigurationError: Invalid configuration_space: 'model' values must be a list
```

**Common causes and fixes**:

```python
# WRONG: configuration_space values must be lists
@traigent.optimize(
    configuration_space={"model": "gpt-4o-mini"},  # String, not list
)

# CORRECT
@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini"]},  # List
)
```

```python
# WRONG: empty configuration space
@traigent.optimize(
    configuration_space={},
)

# CORRECT: at least one parameter to optimize
@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.5, 1.0]},
)
```

Set `TRAIGENT_DEBUG=1` to see the full traceback instead of the clean error message.

### CostLimitExceeded

**When raised**: Cost approval is declined before the first trial (pre-run only). Has `accumulated` and `limit` attributes. `CostLimitExceeded` is a subclass of `OptimizationError`.

> **Mid-run budget overruns do not raise this exception.** When accumulated spend hits the limit during a run, the run stops gracefully and returns a partial `OptimizationResult` — check `result.stop_reason == "cost_limit"`.

```
traigent.utils.exceptions.CostLimitExceeded: Cost limit exceeded: $0.52 >= $0.50 USD
```

**Fixes**:

```python
# Option 1: Increase the cost limit
@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
    cost_limit=2.0,  # $2.00 USD
)

# Option 2: Use cheaper models
@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini"]},  # Cheaper than gpt-4o
)

# Option 3: Reduce trials
@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
    max_trials=5,  # Fewer trials = lower cost
)

# Option 4: Pre-approve costs (suppresses the pre-run prompt)
# export TRAIGENT_COST_APPROVED=true  or  cost_approved=True in code
```

**Handling the pre-run exception**:

```python
from traigent.utils.exceptions import CostLimitExceeded

try:
    results = await func.optimize()
except CostLimitExceeded as e:
    # Raised only when approval is declined before the run starts
    print(f"Cost approval declined: ${e.accumulated:.2f} of ${e.limit:.2f} limit")
```

**Handling a mid-run budget stop** (no exception — check `stop_reason`):

```python
results = await func.optimize()
if results.stop_reason == "cost_limit":
    print(f"Run stopped at cost limit: ${results.total_cost:.2f} spent")
    # results.best_config and results.trials hold what was collected
```

### OptimizationStateError

**When raised**: Accessing configuration in an invalid lifecycle state.

```
traigent.utils.exceptions.OptimizationStateError: Cannot access config outside of optimization trial
```

**Common causes**:

```python
# WRONG: calling get_config() with no active trial or applied config
@traigent.optimize(...)
def my_func(text):
    config = traigent.get_config()
    return config["model"]

# Calling the function directly without optimize() or apply_best_config()
my_func("hello")  # OptimizationStateError!

# CORRECT: run optimization first, then apply
results = my_func.optimize_sync()
my_func.apply_best_config(results)
my_func("hello")  # Works - get_config() returns applied config
```

Has `current_state` and `expected_states` attributes for diagnostics.

### ProviderValidationError

**When raised**: `@traigent.optimize` never raises this automatically -- there
is no `validate_providers=` decorator parameter. It is only raised if your
own code calls the standalone `traigent.providers.validate_providers()`
helper as a pre-run check and chooses to raise on a failed result:

```python
from traigent.providers import get_failed_providers, validate_providers
from traigent.utils.exceptions import ProviderValidationError

results = validate_providers(["gpt-4o-mini", "claude-3-haiku-20240307"])
failed = get_failed_providers(results)
if failed:
    raise ProviderValidationError(f"Provider validation failed: {failed}", failed_providers=failed)
```

```
traigent.utils.exceptions.ProviderValidationError: Provider validation failed:
  - openai: InvalidAPIKey
  - anthropic: MissingAPIKey
```

**Fixes**:

```bash
# Set the correct API keys
export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
export ANTHROPIC_API_KEY="sk-ant-..."  # pragma: allowlist secret
```

If your own pre-run check should be skippable, gate it on
`traigent.utils.env_config.skip_provider_validation()` (true when
`TRAIGENT_SKIP_PROVIDER_VALIDATION=true` or mock mode is on) rather than
building a bespoke skip flag -- `@traigent.optimize` itself has no
provider-validation skip parameter.

The `failed_providers` attribute contains a list of `(provider, error_type)` tuples.

### InvocationError

**When raised**: The decorated function raised an exception during a trial.

```
traigent.utils.exceptions.InvocationError: Function 'classify' failed with config {'model': 'gpt-4o'}
```

Has `config`, `input_data`, and `original_error` attributes. Check the original error:

```python
from traigent.utils.exceptions import InvocationError

try:
    results = await func.optimize()
except InvocationError as e:
    print(f"Config that caused failure: {e.config}")
    print(f"Original error: {e.original_error}")
```

### EvaluationError

**When raised**: The evaluator function failed when scoring a trial's output.

```
traigent.utils.exceptions.EvaluationError: Evaluator raised exception for config {'model': 'gpt-4o'}
```

Check your evaluator function handles edge cases (empty output, None, unexpected formats).

### FeatureNotAvailableError

**When raised**: A feature requires an uninstalled plugin or optional dependency.

```
traigent.utils.exceptions.FeatureNotAvailableError: Feature 'LangChain integration' is not available.
  Requires the 'traigent-langchain' plugin. Install with: pip install traigent[integrations]
```

**Fix**: Install the indicated package.

### ModuleNotFoundError (Python)

**When raised**: Missing optional dependencies.

```
ModuleNotFoundError: No module named 'langchain_openai'
```

**Fix**:

```bash
# Install with specific extras
pip install traigent[integrations]    # LangChain, LiteLLM support
pip install traigent[all]             # Everything
pip install traigent[dev]             # Development tools
```

## Mock Mode

Test your optimization setup without making real API calls or connecting to the backend:

```bash
# Mock LLM responses (no API keys needed)
export TRAIGENT_MOCK_LLM=true

# Skip backend connection (fully local/offline run)
export TRAIGENT_OFFLINE_MODE=true
```

```python
import os
os.environ["TRAIGENT_MOCK_LLM"] = "true"
os.environ["TRAIGENT_OFFLINE_MODE"] = "true"

import traigent

@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"], "temperature": [0.0, 0.5]},
    objectives=["accuracy"],
    eval_dataset="test_data.jsonl",
    max_trials=5,
)
def my_func(text):
    config = traigent.get_config()
    # LLM calls return mock responses
    return "mock response"

# Runs without API keys or backend
results = await my_func.optimize()
```

Mock mode is essential for:
- CI/CD pipelines
- Unit testing
- Validating configuration space setup
- Testing evaluator logic

See [Mock Mode reference](references/mock-mode.md) for details.

## Troubleshooting Decision Tree

### No trials ran

1. Check dataset: does the file exist and contain valid JSONL?
2. Check configuration space: is it non-empty with valid lists?
3. Check for ConfigurationError in output
4. Enable `TRAIGENT_LOG_LEVEL=DEBUG` and check for early failures

### All trials failed

1. Check API keys: are they set and valid?
2. Check `results.failed_trials` for error messages:
   ```python
   for trial in results.failed_trials:
       print(f"{trial.trial_id}: {trial.error_message}")
   ```
3. Test your function standalone (outside optimization) with a sample config
4. Check provider status pages for outages

### Wrong results (low scores)

1. Check your evaluator: does it correctly score good vs bad outputs?
2. Check your dataset: are expected outputs correct?
3. Check configuration space: does it include good model/parameter combinations?
4. Check `results.best_metrics` and compare with manual testing
5. Look at individual trial scores:
   ```python
   for trial in results.successful_trials:
       print(f"{trial.config} -> {trial.metrics}")
   ```

### Cost too high

1. Reduce `max_trials` to limit total API calls
2. Set a `cost_limit` on the decorator
3. Use cheaper models in the configuration space (e.g., `gpt-4o-mini` instead of `gpt-4o`)
4. Reduce dataset size for initial exploration
5. Check `results.experiment_stats.cost_per_configuration` to identify expensive configs

### Optimization is slow

1. Check trial durations: `results.experiment_stats.average_trial_duration`
2. Reduce dataset size for faster feedback
3. Set a `timeout` on the decorator
4. Use smaller models for initial exploration
5. Reduce `max_trials` or configuration space size

## Environment Verification

Verify your Traigent installation and environment:

```bash
# Check Traigent version and configuration
traigent info

# Validate a dataset file
traigent validate dataset.jsonl
```

From Python:

```python
import traigent
print(traigent.__version__)

# Check if mock mode is active
import os
print(f"Mock LLM: {os.getenv('TRAIGENT_MOCK_LLM', 'false')}")
print(f"Offline mode: {os.getenv('TRAIGENT_OFFLINE_MODE', 'false')}")
print(f"Log level: {os.getenv('TRAIGENT_LOG_LEVEL', 'INFO')}")
```

## Graceful Fallback Pattern

When optimization might fail, use a try/except pattern with a known-good default:

```python
import traigent
from traigent.utils.exceptions import TraigentError, CostLimitExceeded

DEFAULT_CONFIG = {"model": "gpt-4o-mini", "temperature": 0.0}

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7],
    },
    objectives=["accuracy"],
    eval_dataset="eval_data.jsonl",
    max_trials=10,
)
def classify(text):
    config = traigent.get_config()
    # ... LLM call ...
    return result

# Attempt optimization with fallback
try:
    results = await classify.optimize()

    if results.best_score is not None and results.best_score >= 0.7:
        classify.apply_best_config(results)
        print(f"Applied optimized config: {results.best_config}")
    else:
        print(f"Score {results.best_score} too low, using default config")

except CostLimitExceeded as e:
    print(f"Budget exceeded (${e.accumulated:.2f}/${e.limit:.2f}), using default config")

except TraigentError as e:
    print(f"Optimization failed: {e.message}, using default config")

# The function still works with either applied or default config
```

This pattern ensures your application remains functional even when optimization encounters problems.

## Reference Files

- [Complete Error Reference](references/error-reference.md)
- [Mock Mode Details](references/mock-mode.md)
- [Logging Configuration](references/logging-config.md)
