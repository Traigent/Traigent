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
- Trial configuration. On the default portal-backed path, this includes tuned
  config-space key/value pairs being tested, with internal keys stripped. If you
  tune prompt variants or other free-text strings as configuration values, those
  tuned values are sent on the default path so the portal can display winning
  configs.
- Trial metrics (accuracy, cost, latency, etc.)

**Optimization Run Metadata**:
- Optimization run ID
- Algorithm used (grid, random, bayesian, optuna)
- Number of trials executed
- Total optimization duration
- Result source (`cloud_brain`, `local_fallback`, `explicit_local`, or `offline`)
- Stop conditions triggered
- Content-free tuned-variable observations can include knob names, enum/scalar values, numeric metrics, and aggregate effectuation events for backend optimization. Set `TRAIGENT_TVAR_OBSERVATION=off` to disable the `tvar_observation_v1` metadata sub-field, or use `TRAIGENT_TVAR_OBSERVATION=hashed` (default) to hash free-form string values in that sub-field. Only `off` and `hashed` are supported; unsupported values fall back to `hashed`. This setting does not control the trial `config` field.

**Performance Metrics**:
- LLM API response times
- Token usage per trial
- Cost per trial (when available)
- Dataset size and evaluation metrics

### What is NOT Collected

Outside tuned configuration values explicitly sent on the default portal-backed
path, and outside observability content you explicitly opt into recording,
Traigent does **not** collect:

- **User prompts or inputs**
- **LLM responses or outputs**
- **Evaluation dataset contents**
- **Personal identifiable information (PII)**
- **API keys or credentials**
- **Source code or function implementations**

### Data Boundary and No-Egress Runs

The default portal-backed path sends tuned config-space values and numeric
metrics. This is deliberate: the portal uses those values to display and compare
winning configs. If a prompt variant, persona instruction, model name, or other
string is part of the tuned configuration space, the default path sends that
string value. It does not send dataset example inputs, expected outputs, model
responses, or example metadata unless you put that content into the tuned
configuration itself.

Use privacy-mode redaction when you need backend coordination without sending
tuned string values. Privacy submissions preserve which keys were tuned, but
redact sensitive-key values and all string/free-text config values. Numeric,
boolean, and `None` config values still pass through. This redaction is gated by
the run's effective `TraigentConfig.privacy_enabled` setting, including
configuration loaded from `TRAIGENT_PRIVACY_MODE=true` for compatibility. Do not
use `@traigent.optimize(..., privacy_enabled=True)` for new code; that decorator
keyword is deprecated and emits a warning. Use `offline=True` instead when the
requirement is no Traigent backend egress.

Use `offline=True` when your policy requires no Traigent backend egress at all:

```python
@traigent.optimize(
    algorithm="grid",
    offline=True,
    ...
)
```

No-egress runs keep Traigent optimization metadata local while still allowing your
own function to call LLM providers or other services.

Use `TRAIGENT_DISABLE_TELEMETRY=true` for SDK telemetry opt-out, effective
`TraigentConfig.privacy_enabled` privacy mode to redact tuned string config
values on privacy-mode submissions, and `offline=True` for zero Traigent backend
egress.

### `@observe` Content Egress

The `@observe` decorator and `ObserveContext` are metadata-only by default. They
can send trace names, observation names, status, timing, tags, environment,
release, IDs, explicit metadata, and token/cost fields you pass, but they omit
function arguments, explicit context `input_data`, return values, and output data
unless you opt in.

To send prompt/output content for observability, opt in explicitly:

```python
@observe("llm-call", content_mode="record")
def call_model(prompt: str) -> str:
    ...
```

Alternatively set `TRAIGENT_OBSERVABILITY_CONTENT=record`. The supported content
modes are:

- `metadata` (default): omit input and output content
- `redacted`: send `{"redacted": true}` placeholders for input and output
- `record`: send captured input and output content

`redact_input=True` and `redact_output=True` force redaction placeholders for the
corresponding side even when `content_mode="record"`. The transport still applies
pattern-based secret scrubbing before send, but that scrubber is a final safety
net, not the content-egress policy.

`@observe(observation_type=GENERATION)` does not estimate LangChain or provider
token usage from prompts. Pass measured `input_tokens`, `output_tokens`,
`total_tokens`, or `cost_usd` from the provider response when recording
generation spans; otherwise usage is reported as unknown, not zero.

### Content-Free Execution Lineage

`ExecutionContextDTO` carries only versioned lineage identifiers. Client and
environment defaults are merged into each trace; fields omitted from a
per-trace DTO inherit those defaults. An explicit `None` clears a default and
is emitted as JSON `null`:

```python
from traigent.observability import ExecutionContextDTO

context = ExecutionContextDTO(toolset_id=None)
client.start_trace("agent-run", execution_context=context)
```

Dictionary inputs have the same explicit-null behavior. Free-form metadata and
content fields are not accepted in execution context.

### OpenTelemetry Tracing

OpenTelemetry tracing is opt-in and uses a separate flag from telemetry
opt-out:

```bash
export TRAIGENT_TRACE_ENABLED=true
```

`TRAIGENT_TRACE_ENABLED` is the canonical tracing flag and defaults to
`false`. It controls SDK span emission and workflow trace tracker creation.
The older plural spelling `TRAIGENT_TRACES_ENABLED` is **no longer supported**
as of 0.13.0 and is silently ignored. If you were using `TRAIGENT_TRACES_ENABLED`,
migrate to `TRAIGENT_TRACE_ENABLED`.

### Agent Workflow Spans

0.12.0 exposes a public helper for adding sanitized agent/node spans to the
active optimization workflow trace:

```python
from collections.abc import Mapping
from typing import Any

from traigent.observability import add_agent_span

def add_agent_span(
    node_id: str,
    *,
    span_type: str = "agent",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    latency_ms: float | None = None,
    model: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None
```

The helper is safe to call from user code. If no active optimization trial or
workflow trace manager exists, it logs at debug level and returns. Metadata is
limited to safe numeric values, sensitive keys such as prompt/response/output
are dropped, and model identifiers are validated before emission.

## How Telemetry is Used

Telemetry data is used for:

1. **Optimization Improvements**: Understanding which optimization strategies work best
2. **Bug Detection**: Identifying errors and failures to improve reliability
3. **Performance Analysis**: Measuring and improving SDK performance
4. **Usage Analytics**: Understanding how features are used to prioritize development

## Data Retention

**No-egress runs**:
- All data is stored **locally** on your machine
- Data is kept in `~/.traigent/` or your specified `local_storage_path`
- You control retention - delete files as needed
- No data is sent to the Traigent backend

**Portal-backed runs**:
- Tuned config-space values, configuration IDs/schema, and numeric metrics can be sent to Traigent backend for optimization coordination
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
- `"on"`

Any other value (including unset) means telemetry is enabled.

### What Happens When Disabled

When `TRAIGENT_DISABLE_TELEMETRY` is set:

1. **No telemetry events are emitted** - SDK telemetry collectors return without recording
2. **No metrics are sent to collectors** - MetricsCollector calls are skipped
3. **Optimization still works normally** - Only telemetry is disabled, not functionality
4. **Local results are still saved** - Optimization results are still persisted locally for your use

## Telemetry Implementation

Telemetry is routed through optimization storage, observability clients, and
workflow tracing. The previous SDK-local Optuna telemetry emitter has been
removed with the local Optuna dependency; smart optimizer telemetry is now a
backend concern.

### Event Types

When telemetry is enabled, the following events may be emitted (not exhaustive):

- `trial_suggested` - An optimizer proposed a new trial configuration
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

When using `offline=True`, data is stored locally:

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
    local_storage_path="./my_optimizations",
    ...
)
```

### Per-example content in optimization logs

By default, optimization runs also write per-trial logs under
`./.traigent/optimization_logs/` (the **project working directory**, unless
`TRAIGENT_OPTIMIZATION_LOG_DIR` / `TRAIGENT_RESULTS_FOLDER` relocate it). Each
example record includes the input **query**, the model **response**, and the
**expected** output as free text, alongside ids and metrics.

> ⚠️ On-disk redaction covers **structured PII only** (emails, Luhn-checked
> cards, API keys / bearer tokens, etc.). Free-text content in prompts and
> responses (names, addresses, proprietary data) is stored **verbatim**.

To keep ids and metrics while omitting that content, disable content logging:

```bash
export TRAIGENT_LOG_EXAMPLE_CONTENT=false   # also accepts 0 / no / off
```

or per-logger via `OptimizationLogger(..., log_example_content=False)`. With it
disabled, trial jsonl retains `example_id`, `accuracy`, `cost_usd`, `latency_ms`
and `error`, but `query` / `response` / `expected` are `null`.

Traigent writes a `.gitignore` (`*`) into the log root so this content is not
accidentally committed, but you should still keep `.traigent/` out of version
control and CI artifact collection in sensitive projects.

## Security and Sensitive Data

### Data Sanitization

Configuration data is sanitized before telemetry. Private/internal keys
(starting with `_`) are automatically removed. On default portal-backed
submissions, tuned config values are otherwise sent unchanged so the portal can
display winning configs. On privacy-mode submissions, sensitive-key values and
all string/free-text config values are redacted while preserving the tuned keys.

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

1. **Use `offline=True` for Zero Traigent Egress**:
   ```python
   @traigent.optimize(
       algorithm="grid",
       offline=True,
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
       algorithm="grid",
       offline=True,
       local_storage_path="/secure/location",
       ...
   )
   ```

4. **Review Evaluation Dataset**: Ensure your evaluation datasets don't contain PII

## Observability transport health

`ObservabilityClient.get_stats()` returns a process-local snapshot of buffered
trace delivery. Use it to alert on local loss before a trace reaches the
backend. The snapshot includes `dropped_items`, `dropped_by_reason`,
`queue_depth`, `inflight_items`, `oldest_inflight_age_seconds`, and
`retry_attempts`.

```python
from traigent.observability import ObservabilityClient, ObservabilityConfig


def report_transport_health(event_type: str, payload: dict) -> None:
    # Every local drop is reported. Delivery failures are one event per batch:
    # payloads include item_count and up to 20 trace_ids alongside drop_reason,
    # cumulative dropped_items, queue_depth, and event-specific details.
    metrics_sink.emit(event_type, payload)


client = ObservabilityClient(
    ObservabilityConfig(health_callback=report_transport_health)
)
stats = client.get_stats()
if stats["dropped_items"]:
    metrics_sink.emit("observability_transport_snapshot", stats)
```

`queue_depth` counts payloads still buffered; `inflight_items` counts payloads
already handed to a sender but not yet completed. A non-null
`oldest_inflight_age_seconds` makes a custom sender that does not return
diagnosable; custom senders should enforce their own request timeout.
`dropped_by_reason` is bounded to the transport's in-process lifetime and
distinguishes reasons such as
`queue_full`, `payload_too_large`, `payload_not_json_serializable`,
`transport_closed`, and `batch_delivery_failed`. These are local SDK metrics;
they are deliberately not added to the ingest wire payload until the strict
Schema and Backend contract is updated together.

`health_callback` is invoked only after SDK transport locks are released. It
may run on a background delivery thread, and callback exceptions are swallowed.
Keep it fast and safe for concurrent invocation. Batch-delivery failures emit a
single batch-shaped event with `item_count`, up to 20 `trace_ids`, and
`trace_ids_truncated`; queue, oversized-payload, serialization, and closed-
transport drops each emit their own event.

`flush(timeout=0)` and `close(timeout=0)` are immediate poll-style calls: they
return without starting a sender and do not emit a deadline-exceeded warning.
Previously, zero was treated like the configured flush timeout. An explicit
positive timeout bounds the wait; no-argument flush/close preserves synchronous
delivery. The atexit close path explicitly uses the default 30-second
`flush_timeout`, so shutdown can trade undelivered tail items for a bounded
interpreter exit.

## FAQ

### Q: Is telemetry enabled by default?

**A**: Yes, for local optimization metadata. The default portal-backed path can
send tuned config-space values, configuration IDs/schema, and numeric metrics;
use privacy-mode redaction when backend coordination is needed without tuned
string values, or `offline=True` for zero Traigent backend egress.

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

**A**: The `TRAIGENT_DISABLE_TELEMETRY` environment variable is process-wide.
Use `offline=True` when you need to disable Traigent backend egress for a run.

### Q: Where can I see what telemetry data was collected?

**A**: With `offline=True`, check the JSON files in `~/.traigent/sessions/`. They
contain the same trial metadata and metrics emitted to telemetry listeners.

### Q: Can I contribute telemetry data to improve Traigent?

**A**: Currently, telemetry is local-only in the open-source version. Future versions may offer optional anonymous telemetry reporting with explicit opt-in.

## Related Documentation

- [Decorator Reference](./decorator-reference.md) - Configuration options
- [Choosing the Right Optimization Model](../user-guide/choosing_optimization_model.md) - defaults, local search, no-egress runs, and migration guidance
- [Security](../contributing/SECURITY.md) - Security practices
- [API Reference](./complete-function-specification.md) - Full API documentation
