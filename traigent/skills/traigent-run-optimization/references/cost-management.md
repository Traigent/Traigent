# Cost Management Reference

Traigent provides real-time cost tracking and enforcement to prevent runaway LLM API spending during optimization runs.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRAIGENT_RUN_COST_LIMIT` | `2.0` | Maximum USD spending per optimization run. |
| `TRAIGENT_COST_APPROVED` | `false` | Set to `"true"` to skip the interactive cost approval handshake. |
| `TRAIGENT_STRICT_COST_ACCOUNTING` | `false` | Fail fast if cost extraction from LLM responses fails. |
| `TRAIGENT_REQUIRE_COST_TRACKING` | `false` | Raise exception if cost tracking cannot extract costs. |
| `TRAIGENT_COST_WARNING_THRESHOLD` | `0.5` | Warn when this fraction of the limit is consumed (0.0-1.0). |
| `TRAIGENT_COST_DIVERGENCE_THRESHOLD` | `2.0` | Log warning if actual/estimated cost ratio exceeds this. |
| `TRAIGENT_MOCK_LLM` | `false` | Bypass all cost tracking (no real LLM costs in mock mode). |

## Setting a Cost Limit

```bash
# Set via environment variable
export TRAIGENT_RUN_COST_LIMIT=5.00

# Or in Python before running optimization
import os
os.environ["TRAIGENT_RUN_COST_LIMIT"] = "5.00"
```

The default limit is $2.00 per optimization run. This applies per call to `optimize()` or `optimize_sync()`.

## CostLimitExceeded Exception

When accumulated costs reach or exceed the limit, Traigent raises `CostLimitExceeded`.

```python
from traigent.utils.exceptions import CostLimitExceeded
```

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `accumulated` | `float` | Total cost in USD accumulated before the limit was hit. |
| `limit` | `float` | The configured cost limit in USD. |

### Handling

```python
from traigent.utils.exceptions import CostLimitExceeded

try:
    results = await func.optimize(max_trials=100, algorithm="bayesian")
except CostLimitExceeded as e:
    print(f"Cost limit exceeded: ${e.accumulated:.2f} of ${e.limit:.2f} budget")
    # The optimization stopped gracefully.
    # Partial results may be available via the orchestrator.
```

### Stop Reason

When optimization stops due to cost (without raising an exception), the result will have:

```python
results = await func.optimize(max_trials=50, algorithm="random")
if results.stop_reason == "cost_limit":
    print(f"Stopped at cost limit. Total cost: ${results.total_cost:.2f}")
```

## Cost Tracking via LiteLLM

Traigent uses LiteLLM's cost tracking to extract per-call costs from LLM API responses. This works automatically with all LiteLLM-supported providers (OpenAI, Anthropic, Cohere, etc.).

Cost tracking happens transparently:

1. Before each trial, the `CostEnforcer` issues a `Permit` (reserving estimated cost).
2. The trial executes and makes LLM API calls.
3. After the trial, actual costs are extracted from responses and recorded.
4. If the accumulated cost exceeds the limit, the enforcer stops further trials.

## Adaptive Cost Estimation

The cost enforcer uses adaptive estimation based on observed trial costs:

- **Initial estimates** use a configured `estimated_cost_per_trial` value.
- **After trials complete**, estimates update via exponential moving average (EMA).
- **Confidence levels** (0.0-1.0) indicate reliability of estimates.
- **Divergence warnings** are logged when actual costs differ significantly from estimates (controlled by `TRAIGENT_COST_DIVERGENCE_THRESHOLD`).

This means the enforcer gets better at predicting costs as the optimization progresses, and can pre-emptively stop before exceeding the budget.

## Pre-Approving Costs

By default, Traigent may prompt for cost approval before running optimization (especially for expensive configurations). To skip this:

```bash
export TRAIGENT_COST_APPROVED=true
```

This is useful in CI/CD pipelines, automated scripts, or when you have already verified the budget is acceptable.

## Strict Cost Accounting

Enable strict mode to fail fast when cost information is unavailable:

```bash
export TRAIGENT_STRICT_COST_ACCOUNTING=true
```

In strict mode, if Traigent cannot extract the cost from an LLM response (e.g., the provider does not report usage), it raises `CostTrackingRequiredError` instead of logging a warning and continuing.

This is recommended for production environments where accurate cost tracking is critical.

## Cost Warning Threshold

Traigent logs a warning when the accumulated cost crosses a fraction of the limit:

```bash
export TRAIGENT_COST_WARNING_THRESHOLD=0.5  # Warn at 50% of budget
```

With a $5.00 limit and a 0.5 threshold, a warning is logged when costs reach $2.50.

## Thread Safety

The `CostEnforcer` is designed for parallel trial execution:

- Uses `threading.RLock()` for all shared state mutations.
- Each trial acquires a `Permit` before execution and releases it after.
- Permits have single-use semantics (cannot be released twice).
- Cost tracking is atomic and consistent across concurrent trials.

```
Trial 1: acquire_permit() -> execute -> track_cost(permit, $0.03)
Trial 2: acquire_permit() -> execute -> track_cost(permit, $0.05)
                                        ^-- lock ensures atomic update
```

## Mock Mode

When `TRAIGENT_MOCK_LLM=true`, all cost tracking is bypassed. No permits are issued, no costs are recorded, and `CostLimitExceeded` is never raised. Use this for local development and testing.

```bash
export TRAIGENT_MOCK_LLM=true
export TRAIGENT_OFFLINE_MODE=true
```

## Practical Examples

### CI/CD Pipeline

```bash
export TRAIGENT_RUN_COST_LIMIT=10.00
export TRAIGENT_COST_APPROVED=true
export TRAIGENT_STRICT_COST_ACCOUNTING=true

python run_optimization.py
```

### Development / Testing

```bash
export TRAIGENT_MOCK_LLM=true
export TRAIGENT_OFFLINE_MODE=true

pytest tests/
```

### Conservative Production Run

```bash
export TRAIGENT_RUN_COST_LIMIT=1.00
export TRAIGENT_COST_WARNING_THRESHOLD=0.3
export TRAIGENT_STRICT_COST_ACCOUNTING=true

python optimize_production.py
```

## Cost in Results

After optimization, cost information is available on the result object:

```python
results = await func.optimize(max_trials=10, algorithm="grid")

print(f"Total cost: ${results.total_cost:.4f}")
print(f"Total tokens: {results.total_tokens}")

# Per-trial costs (if available on trial objects)
for trial in results.trials:
    print(f"Trial {trial.trial_id}: score={trial.score}, cost=${trial.cost:.4f}")
```
