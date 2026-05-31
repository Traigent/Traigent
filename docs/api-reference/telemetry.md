# Traigent Telemetry Documentation

This document describes what telemetry data the Traigent SDK collects, how it is
used, retention policies, and how to opt out.

## Overview

Traigent collects local optimization metadata to help diagnose runs and improve
the optimization experience. In the open-source build, telemetry stays local
unless you opt into managed services. Telemetry can be completely disabled.

## What Data is Collected

### Optimization Metrics

During optimization runs, Traigent records:

**Trial Lifecycle Events**
- Trial suggested/intermediate/completed timestamps
- Trial status (completed, failed, pruned)
- Trial duration
- Trial configuration, with internal keys stripped
- Trial metrics (accuracy, cost, latency, and custom metrics)

**Optimization Run Metadata**
- Optimization run ID
- Algorithm used (`grid`, `random`, or a backend-routed smart strategy id when enabled)
- Number of trials executed
- Total optimization duration
- Execution mode (`edge_analytics`, `hybrid`, and related aliases)
- Stop conditions triggered

**Performance Metrics**
- LLM API response times
- Token usage per trial
- Cost per trial when available
- Dataset size and evaluation metrics

### What is Not Collected

Traigent does not collect:

- User prompts or inputs
- LLM responses or outputs
- Evaluation dataset contents
- Personal identifiable information (PII)
- API keys or credentials
- Source code or function implementations

### Privacy Mode

When `privacy_enabled=True` is set in `ExecutionOptions`:

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

Privacy mode does not disable telemetry. Use `TRAIGENT_DISABLE_TELEMETRY=true`
for a full opt-out.

## How Telemetry is Used

Telemetry data is used for:

1. Optimization improvements: understanding which strategies work best
2. Bug detection: identifying errors and failures to improve reliability
3. Performance analysis: measuring and improving SDK performance
4. Usage analytics: prioritizing feature work in managed deployments

## Data Retention

**Edge Analytics Mode** (default):

- All data is stored locally on your machine
- Data is kept in `~/.traigent/` or your specified `local_storage_path`
- You control retention by deleting files as needed
- No data is sent to external servers

**Hybrid Mode** (managed service only):

- Metadata can be sent to the Traigent backend for session tracking and managed coordination
- Retention policies depend on your managed-service agreement
- You can request data deletion at any time

## Opting Out of Telemetry

### Complete Opt-Out

To completely disable telemetry collection, set:

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

The following values are recognized as opt-out values, case-insensitive:

- `"true"`
- `"1"`
- `"yes"`

Any other value, including unset, means telemetry is enabled.

### What Happens When Disabled

When `TRAIGENT_DISABLE_TELEMETRY` is set:

1. No telemetry events are emitted
2. No metrics are sent to collectors
3. Optimization still works normally
4. Local results are still saved for your use

## Event Types

When telemetry is enabled, these events may be emitted:

- `trial_suggested` - An optimizer proposed a new trial configuration
- `trial_intermediate` - Intermediate metric reported
- `trial_completed` - A trial finished successfully
- `trial_failed` - A trial encountered an error
- `trial_pruned` - A trial was stopped early
- `trial_call_started` / `trial_call_completed` - Seamless injection wrapper lifecycle

Each event includes:

- `event`: Event type string
- `trial_id`: Trial identifier, if applicable
- `study_name`: Optimization study name
- `timestamp`: ISO 8601 timestamp
- `payload`: Additional event-specific data

## Local Storage Structure

When using `edge_analytics` mode, data is stored locally:

```text
~/.traigent/
|- sessions/
|  |- 20260101_120000_my_function.json  # Optimization session record
|  `- ...
|- cache/
|  |- model_responses/
|  `- ...
`- .locks/
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

SDK telemetry supports listener-style integrations for internal and managed
monitoring paths. Listener failures never crash optimization:

- If a listener raises an exception, it is logged but does not affect optimization
- If metrics collection fails, it is logged but optimization continues
- Telemetry is defensive and isolated from core functionality

## Security and Privacy

### Data Sanitization

Configuration data is sanitized before telemetry. Private or internal keys that
start with `_` are removed from exported event payloads.

Example input:

```python
config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "_internal_key": "should_not_be_logged",
}
```

Sanitized output:

```python
{"model": "gpt-4", "temperature": 0.7}
```

### Thread Safety

Telemetry listener subscription and emission paths are protected for concurrent
optimization runs.

## Compliance

### GDPR

Traigent minimizes personal data collection by default. In local mode, data
remains on your infrastructure. In managed mode, retention and deletion are
covered by your service agreement.
