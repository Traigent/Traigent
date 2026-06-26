# Cost Management Reference

Traigent provides real-time cost tracking and enforcement to prevent runaway LLM API spending during optimization runs.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRAIGENT_RUN_COST_LIMIT` | `2.0` | Maximum USD spending per optimization run. |
| `TRAIGENT_COST_APPROVED` | `false` | Exact value `true` pre-approves both the cost-limit prompt and unpriced-model preflight. `1`, `yes`, and `on` do not approve. |
| `TRAIGENT_STRICT_COST_ACCOUNTING` | `false` | Exact value `true` fails fast before trial 1 on unpriced models and when runtime cost extraction is missing or unknown. |
| `TRAIGENT_REQUIRE_COST_TRACKING` | `false` | Raise exception if cost tracking cannot extract costs. |
| `TRAIGENT_COST_WARNING_THRESHOLD` | `0.5` | Warn when this fraction of the limit is consumed (0.0-1.0). |
| `TRAIGENT_COST_DIVERGENCE_THRESHOLD` | `2.0` | Log warning if actual/estimated cost ratio exceeds this. |
| `TRAIGENT_MOCK_LLM` | `false` | Mock LiteLLM/LangChain provider calls and skip optimized-function pricing preflight; not a global `CostEnforcer` bypass. |

## Setting a Cost Limit

```bash
# Set via environment variable
export TRAIGENT_RUN_COST_LIMIT=5.00

# Or in Python before running optimization
import os
os.environ["TRAIGENT_RUN_COST_LIMIT"] = "5.00"
```

The default limit is $2.00 per optimization run. This applies per call to `optimize()` or `optimize_sync()`.

## Cost Limit Behavior

Traigent enforces cost limits through two distinct surfaces:

### Pre-Run: CostLimitExceeded (raised)

`CostLimitExceeded` is raised when cost approval is **declined before the first trial** — for example when a non-interactive shell has no `TRAIGENT_COST_APPROVED=true`, or when the interactive cost-approval prompt is rejected. `CostLimitExceeded` is a subclass of `OptimizationError`, so `except OptimizationError` also catches it.

```python
from traigent.utils.exceptions import CostLimitExceeded, OptimizationError

try:
    results = await func.optimize(max_trials=100, algorithm="random")
except CostLimitExceeded as e:
    print(f"Cost approval declined: ${e.accumulated:.2f} of ${e.limit:.2f} limit")
```

#### CostLimitExceeded Attributes

| Attribute | Type | Description |
|---|---|---|
| `accumulated` | `float` | Total cost in USD accumulated when approval was declined. |
| `limit` | `float` | The configured cost limit in USD. |

### Mid-Run: Graceful stop (check stop_reason)

When accumulated spend reaches the budget limit *during* a run, Traigent stops gracefully after the current trial and **returns** a partial `OptimizationResult`. No exception is raised. Inspect `result.stop_reason` to detect this:

```python
results = await func.optimize(max_trials=50, algorithm="random")
if results.stop_reason == "cost_limit":
    print(f"Run stopped at cost limit. Total cost: ${results.total_cost:.2f}")
    # results.best_config and results.trials hold what was collected so far
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

By default, Traigent may prompt for cost approval before running optimization
(especially for expensive or unpriced configurations). To pre-approve
cost-sensitive execution:

```bash
export TRAIGENT_COST_APPROVED=true
```

The environment value must be exactly `true`; values such as `1`, `yes`, and
`on` do not approve. In code, runtime `cost_approved=True` must be a real
boolean; string values are ignored and logged as warnings.

Pre-approval covers both the cost-limit prompt and the unpriced-model preflight
gate. It proceeds with a warning for unpriced models; it does not add pricing
coverage. Without pre-approval, unpriced models block before trial 1: interactive
terminals prompt `y/N`, and non-interactive shells fail closed.

## Strict Cost Accounting

Enable strict mode to fail fast before trial 1 on unpriced models, and fail if
runtime cost information is missing or unknown:

```bash
export TRAIGENT_STRICT_COST_ACCOUNTING=true
```

The environment value must be exactly `true`. In strict mode, unpriced models
fail fast before trial 1, and missing or unknown runtime cost extraction raises
`CostTrackingRequiredError` instead of logging a warning and continuing.

Budget limits are separate: `TRAIGENT_RUN_COST_LIMIT` controls the run budget.
Mid-run overruns stop the run gracefully (check `result.stop_reason == "cost_limit"`);
pre-run approval failures raise `CostLimitExceeded`.

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

When `TRAIGENT_MOCK_LLM=true`, LiteLLM and LangChain provider calls are mocked
and the optimized-function pricing preflight is skipped because no provider
spend occurs through those paths. Raw SDK calls such as
`openai.chat.completions.create(...)` and `anthropic.messages.create(...)` are
not intercepted by mock mode. Mock mode does not globally bypass `CostEnforcer`
permits or accounting, so include `cost_approved=True` in code or
`TRAIGENT_COST_APPROVED=true` in shell dry runs.

```bash
export TRAIGENT_MOCK_LLM=true
export TRAIGENT_COST_APPROVED=true
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
export TRAIGENT_COST_APPROVED=true
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
