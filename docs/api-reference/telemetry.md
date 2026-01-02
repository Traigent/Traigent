# Traigent Telemetry Documentation

This document describes what telemetry data Traigent SDK collects, how it's used, retention policies, and how to opt-out.

## Overview

Traigent SDK collects telemetry data to improve the optimization experience and help diagnose issues. In the open-source build, telemetry stays local unless you opt into managed services. Telemetry can be completely disabled.

## What Data is Collected

### Optimization Metrics

During optimization runs, Traigent collects:

**Trial Lifecycle Events**:
- Trial suggested/intermediate/completed timestamps
- Trial status (completed, failed, pruned)
- Trial duration
- Trial configuration (parameter values being tested, with internal keys stripped)
- Trial metrics (accuracy, cost, latency, etc.)

**Optimization Run Metadata**:
- Optimization run ID
- Algorithm used (grid, random, bayesian, optuna)
- Number of trials executed
- Total optimization duration
- Execution mode (edge_analytics, cloud, etc.)
- Stop conditions triggered

**Performance Metrics**:
- LLM API response times
- Token usage per trial
- Cost per trial (when available)
- Dataset size and evaluation metrics

### What is NOT Collected

Traigent does **not** collect:

- **User prompts or inputs**
- **LLM responses or outputs**
- **Evaluation dataset contents**
- **Personal identifiable information (PII)**
- **API keys or credentials**
- **Source code or function implementations**

### Privacy Mode

When `privacy_enabled=True` is set in ExecutionOptions:

```python
@traigent.optimize(
    execution={
        "privacy_enabled": True,
    },
    ...
)
```

Traigent will:
- Redact prompts/responses from stored evaluation artifacts when possible
- Minimize logged content while keeping metrics and configuration metadata
- Keep results local in open-source builds

Privacy mode does not disable telemetry; use `TRAIGENT_DISABLE_TELEMETRY=true` for a full opt-out.

## How Telemetry is Used

Telemetry data is used for:

1. **Optimization Improvements**: Understanding which optimization strategies work best
2. **Bug Detection**: Identifying errors and failures to improve reliability
3. **Performance Analysis**: Measuring and improving SDK performance
4. **Usage Analytics**: Understanding how features are used to prioritize development

## Data Retention

**Edge Analytics Mode** (default):
- All data is stored **locally** on your machine
- Data is kept in `~/.traigent/` or your specified `local_storage_path`
- You control retention - delete files as needed
- No data is sent to external servers

**Cloud/Hybrid Mode** (managed service only):
- Metadata can be sent to Traigent backend for optimization coordination
- Retention policies depend on your managed-service agreement
- You can request data deletion at any time

## Opting Out of Telemetry

### Complete Opt-Out

To completely disable all telemetry collection, set the environment variable:

```bash
export TRAIGENT_DISABLE_TELEMETRY=true
```

Or in Python before importing Traigent:

```python
import os
os.environ["TRAIGENT_DISABLE_TELEMETRY"] = "true"

import traigent
```

### Accepted Values

The following values are recognized as "opt-out" (case-insensitive):
- `"true"`
- `"1"`
- `"yes"`

Any other value (including unset) means telemetry is enabled.

### What Happens When Disabled

When `TRAIGENT_DISABLE_TELEMETRY` is set:

1. **No telemetry events are emitted** - The `OptunaMetricsEmitter` immediately returns without recording
2. **No metrics are sent to collectors** - MetricsCollector calls are skipped
3. **Optimization still works normally** - Only telemetry is disabled, not functionality
4. **Local results are still saved** - Optimization results are still persisted locally for your use

## Telemetry Implementation

### OptunaMetricsEmitter

The main telemetry component is `OptunaMetricsEmitter` in `traigent/telemetry/optuna_metrics.py`:

```python
from traigent.telemetry.optuna_metrics import OptunaMetricsEmitter

# Telemetry is automatically disabled when env var is set
emitter = OptunaMetricsEmitter(
    metrics_collector=collector,
    listeners=[my_listener],
)

# This call does nothing if TRAIGENT_DISABLE_TELEMETRY is set
emitter.emit_trial_update(
    event="trial_completed",
    trial_id=123,
    study_name="my_optimization",
    payload={"metrics": {"accuracy": 0.95}},
)
```

### Implementation Details

The `OptunaMetricsEmitter` checks the environment variable on initialization:

```python
class OptunaMetricsEmitter:
    def __init__(self, ...):
        self._disabled = os.getenv("TRAIGENT_DISABLE_TELEMETRY", "").lower() in (
            "1",
            "true",
            "yes",
        )

    def emit_trial_update(self, ...):
        if self._disabled:
            return {}  # No-op when disabled

        # Normal telemetry emission
        ...
```

### Event Types

When telemetry is enabled, the following events may be emitted (not exhaustive):

- `trial_suggested` - Optuna proposed a new trial configuration
- `trial_intermediate` - Intermediate metric reported (pruning signal)
- `trial_completed` - A trial finished successfully
- `trial_failed` - A trial encountered an error
- `trial_pruned` - A trial was pruned early
- `trial_call_started` / `trial_call_completed` - Seamless injection wrapper lifecycle

Each event includes:
- `event`: Event type string
- `trial_id`: Trial identifier (if applicable)
- `study_name`: Optimization study name
- `timestamp`: ISO 8601 timestamp
- `payload`: Additional event-specific data

## Local Storage Structure

When using edge_analytics mode, data is stored locally:

```
~/.traigent/
├── sessions/
│   ├── 20260101_120000_my_function.json  # Optimization session record (trials + metadata)
│   └── ...
├── cache/
│   ├── model_responses/
│   └── ...
└── .locks/
```

You can customize the storage location:

```python
@traigent.optimize(
    execution={
        "local_storage_path": "./my_optimizations",
    },
    ...
)
```

## Telemetry Listeners

You can subscribe to telemetry events for your own monitoring:

```python
from traigent.telemetry.optuna_metrics import OptunaMetricsEmitter

def my_listener(event_data: dict):
    print(f"Trial event: {event_data['event']}")
    print(f"Trial ID: {event_data['trial_id']}")
    print(f"Metrics: {event_data.get('payload', {})}")

emitter = OptunaMetricsEmitter()
emitter.subscribe(my_listener)

# Your listener will be called for each telemetry event
# (unless TRAIGENT_DISABLE_TELEMETRY is set)
```

## Security and Privacy

### Data Sanitization

Configuration data is automatically sanitized before telemetry:

```python
from traigent.telemetry.optuna_metrics import sanitize_config

config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "_internal_key": "should_not_be_logged",
}

safe_config = sanitize_config(config)
# Returns: {"model": "gpt-4", "temperature": 0.7"}
```

Private/internal keys (starting with `_`) are automatically removed.

### Thread Safety

The `OptunaMetricsEmitter` is thread-safe and uses `threading.RLock` to protect listener subscription operations.

### Error Handling

Telemetry failures never crash your optimization:

- If a listener raises an exception, it's logged but doesn't affect optimization
- If metrics collection fails, it's logged but optimization continues
- Telemetry is defensive and isolated from core functionality

## Compliance

### GDPR

Traigent SDK is designed to be GDPR-compliant:

- **Right to Access**: All data is stored locally by default
- **Right to Deletion**: Delete files in `~/.traigent/` at any time
- **Right to Opt-Out**: Set `TRAIGENT_DISABLE_TELEMETRY=true`
- **Data Minimization**: Only essential optimization metadata is collected
- **No PII Collection**: Prompts, responses, and PII are not collected

### HIPAA and Sensitive Data

For HIPAA compliance or handling sensitive data:

1. **Enable Privacy Mode**:
   ```python
   @traigent.optimize(
       execution={"privacy_enabled": True},
       ...
   )
   ```

2. **Disable Telemetry**:
   ```bash
   export TRAIGENT_DISABLE_TELEMETRY=true
   ```

3. **Use Local Storage Only**:
   ```python
   @traigent.optimize(
       execution={
           "execution_mode": "edge_analytics",
           "local_storage_path": "/secure/location",
       },
       ...
   )
   ```

4. **Review Evaluation Dataset**: Ensure your evaluation datasets don't contain PII

## FAQ

### Q: Is telemetry enabled by default?

**A**: Yes, but only for local optimization metadata. In the default `edge_analytics` mode, all data stays on your machine.

### Q: How do I verify telemetry is disabled?

**A**: Check the environment variable:
```bash
echo $TRAIGENT_DISABLE_TELEMETRY
```

Or in Python:
```python
import os
print(os.getenv("TRAIGENT_DISABLE_TELEMETRY"))
```

### Q: Does disabling telemetry affect optimization performance?

**A**: No. Disabling telemetry has minimal performance impact. Optimization runs normally without telemetry.

### Q: Can I enable telemetry for some optimizations but not others?

**A**: The `TRAIGENT_DISABLE_TELEMETRY` environment variable is process-wide. `privacy_enabled` reduces stored content but does not disable telemetry.

### Q: Where can I see what telemetry data was collected?

**A**: In edge_analytics mode, check the JSON files in `~/.traigent/sessions/`. They contain the same trial metadata and metrics emitted to telemetry listeners.

### Q: Can I contribute telemetry data to improve Traigent?

**A**: Currently, telemetry is local-only in the open-source version. Future versions may offer optional anonymous telemetry reporting with explicit opt-in.

## Related Documentation

- [Decorator Reference](./decorator-reference.md) - Configuration options
- [Execution Modes Guide](../guides/execution-modes.md) - Edge Analytics vs Cloud
- [Privacy & Security](../contributing/SECURITY.md) - Security practices
- [API Reference](./complete-function-specification.md) - Full API documentation
