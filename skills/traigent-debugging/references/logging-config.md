# Logging Configuration Reference

## Overview

Traigent uses Python's standard `logging` module with two primary environment variables for controlling verbosity:

| Variable | Purpose | Values |
|---|---|---|
| `TRAIGENT_LOG_LEVEL` | Set the logging level for all Traigent loggers | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `TRAIGENT_DEBUG` | Enable full tracebacks for ConfigurationError | `1` (enabled), unset (disabled) |

## TRAIGENT_LOG_LEVEL

Controls the verbosity of Traigent's internal logging.

### Setting

```bash
# Command line
export TRAIGENT_LOG_LEVEL=DEBUG

# Or inline
TRAIGENT_LOG_LEVEL=DEBUG python my_script.py
```

```python
# In Python (set before importing traigent)
import os
os.environ["TRAIGENT_LOG_LEVEL"] = "DEBUG"
import traigent
```

### Log Levels

| Level | What It Shows |
|---|---|
| `DEBUG` | Everything: config sampling, trial start/stop, metric extraction, cost tracking, backend communication, internal state changes. Very verbose. |
| `INFO` | Optimization lifecycle events: run start, trial completion, best config updates, run completion. The default level. |
| `WARNING` | Non-fatal issues: deprecated API usage, non-numeric metric values, retry attempts, fallback behavior. |
| `ERROR` | Trial failures, evaluation errors, provider errors, unrecoverable issues within a trial. |
| `CRITICAL` | Fatal errors that prevent the optimization from continuing at all. Rare. |

### What Each Level Shows

#### DEBUG

```
DEBUG traigent.core.orchestrator: Sampling configuration: {'model': 'gpt-4o-mini', 'temperature': 0.3}
DEBUG traigent.core.trial_lifecycle: Starting trial trial-001 with config {'model': 'gpt-4o-mini', 'temperature': 0.3}
DEBUG traigent.core.trial_lifecycle: Trial trial-001 function returned in 2.3s
DEBUG traigent.core.trial_lifecycle: Trial trial-001 metrics: {'accuracy': 0.85}
DEBUG traigent.core.cost_enforcement: Trial trial-001 cost: $0.0012, total: $0.0012
DEBUG traigent.core.orchestrator: Trial trial-001 completed, score=0.85 (best so far)
```

Use DEBUG when:
- Investigating why a specific trial failed
- Understanding configuration sampling behavior
- Debugging metric extraction issues
- Tracing cost tracking calculations
- Diagnosing backend communication problems

#### INFO

```
INFO traigent.core.orchestrator: Starting optimization with max_trials=10
INFO traigent.core.orchestrator: Trial 1/10 completed: score=0.85
INFO traigent.core.orchestrator: Trial 2/10 completed: score=0.72
INFO traigent.core.orchestrator: New best score: 0.92 with config {'model': 'gpt-4o', 'temperature': 0.0}
INFO traigent.core.orchestrator: Optimization completed: best_score=0.92, duration=45.3s
```

Use INFO for:
- Normal operation monitoring
- Tracking optimization progress
- Production logging

#### WARNING

```
WARNING traigent.core.trial_lifecycle: Non-numeric metric value for 'model_name': 'gpt-4o' (use metadata instead)
WARNING traigent.utils.exceptions: ConfigAccessWarning: Use traigent.get_config() instead of get_current_config()
WARNING traigent.core.cost_enforcement: Cost approaching limit: $0.45/$0.50
```

#### ERROR

```
ERROR traigent.core.trial_lifecycle: Trial trial-003 failed: APIError: Rate limit exceeded
ERROR traigent.core.orchestrator: All trials failed, no valid results
```

## TRAIGENT_DEBUG

This variable controls traceback display for `ConfigurationError` specifically.

### Default Behavior (TRAIGENT_DEBUG not set)

ConfigurationError shows a clean, single-line message:

```
traigent.utils.exceptions.ConfigurationError: Invalid configuration_space: 'model' values must be a list
```

### With TRAIGENT_DEBUG=1

Full Python traceback is shown:

```
Traceback (most recent call last):
  File "my_script.py", line 15, in <module>
    results = func.optimize(dataset="data.jsonl")
  File "/path/to/traigent/core/optimized_function.py", line 234, in optimize
    self._validate_config_space(config_space)
  File "/path/to/traigent/core/optimized_function.py", line 178, in _validate_config_space
    raise ConfigurationError(f"Invalid configuration_space: '{key}' values must be a list")
traigent.utils.exceptions.ConfigurationError: Invalid configuration_space: 'model' values must be a list
```

```bash
# Enable
export TRAIGENT_DEBUG=1

# Disable
unset TRAIGENT_DEBUG
```

This only affects `ConfigurationError` and its subclasses. All other exceptions always show full tracebacks.

## Combining Logging Options

For maximum diagnostic information:

```bash
export TRAIGENT_LOG_LEVEL=DEBUG
export TRAIGENT_DEBUG=1
python my_script.py
```

For production with minimal noise:

```bash
export TRAIGENT_LOG_LEVEL=WARNING
# TRAIGENT_DEBUG unset
python my_script.py
```

## Programmatic Logging Configuration

You can also configure Traigent's logger directly:

```python
import logging

# Get the Traigent root logger
traigent_logger = logging.getLogger("traigent")
traigent_logger.setLevel(logging.DEBUG)

# Add a file handler for Traigent logs
handler = logging.FileHandler("traigent_debug.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
traigent_logger.addHandler(handler)
```

## Logging in Tests

For pytest, control logging with standard pytest options:

```bash
# Show Traigent debug logs during test runs
TRAIGENT_LOG_LEVEL=DEBUG pytest tests/ -s --log-cli-level=DEBUG

# Capture logs in test output
pytest tests/ --log-level=DEBUG
```

Or in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
log_cli = true
log_cli_level = "INFO"
```
