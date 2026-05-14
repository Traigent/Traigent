# Parallel Configuration Guide

Traigent uses `parallel_config` to control concurrency at the trial level and
within each trial's evaluation loop. You can set it globally, on the decorator,
or per optimization run.

## Where to Set `parallel_config`

1. **Global default**: `traigent.configure(parallel_config=...)`
2. **Decorator override**: `@traigent.optimize(parallel_config=...)`
3. **Runtime override**: `await fn.optimize(parallel_config=...)`

Later scopes override earlier ones field-by-field.

## ParallelConfig Fields

| Field | Meaning | Notes |
| --- | --- | --- |
| `mode` | `"auto"`, `"sequential"`, or `"parallel"` | `"auto"` infers from concurrency values. |
| `trial_concurrency` | Number of trials to run concurrently | `>1` enables parallel mode. |
| `example_concurrency` | Number of examples to evaluate concurrently per trial | Useful for large datasets. |
| `thread_workers` | Worker pool size for local evaluation/execution | Controls local thread pool usage. |

## Examples

### Global default (ParallelConfig)

```python
import traigent
from traigent.config.parallel import ParallelConfig

traigent.configure(
    parallel_config=ParallelConfig(
        mode="parallel",
        trial_concurrency=2,
        example_concurrency=4,
        thread_workers=4,
    )
)
```

### Decorator override (dict)

```python
@traigent.optimize(
    objectives=["accuracy"],
    configuration_space={"temperature": [0.0, 0.7]},
    parallel_config={"trial_concurrency": 2},
)
def answer(question: str) -> str:
    ...
```

### Runtime override (dict)

```python
result = await answer.optimize(
    max_trials=8,
    parallel_config={"trial_concurrency": 4, "thread_workers": 4},
)
```

## Resolution and Tuning

- `mode="auto"` switches to parallel execution when `trial_concurrency` or
  `example_concurrency` is greater than 1.
- In parallel mode, missing values are inferred from the worker pool and
  configuration space size.
- Keep `trial_concurrency * example_concurrency` near your available worker
  capacity to avoid oversubscription or provider throttling.

## Validation

- Unknown keys in a `parallel_config` dict raise `ValueError`.
- Concurrency values must be `>= 1`.
- If `mode="parallel"`, `trial_concurrency` must be greater than 1.

## Budget Guards in Parallel

Parallel execution uses shared guardrails to keep cost and sample budgets bounded:

- **Cost permits (shared state)**: Each parallel trial must acquire a cost permit
  before it runs. Permits reserve estimated cost (EMA) up front; trials that
  cannot reserve budget are cancelled before execution. After the trial finishes,
  actual cost is tracked and the reservation is released. Because reservations
  use estimates, total cost can exceed the limit by the estimate delta, but the
  stop condition prevents new trials once the limit is reached.

```python
# High-level flow (see CostEnforcer + ParallelExecutionManager)
permit = await cost_enforcer.acquire_permit_async()
if not permit.is_granted:
    return None  # trial cancelled
result = await coro
await cost_enforcer.track_cost_async(cost, permit=permit)
```

- **Sample ceilings (`max_total_examples`)**: When a global sample budget is
  configured, the orchestrator divides the remaining budget across the current
  parallel batch so each trial can consume at most its share. Each trial receives
  a `SampleBudgetLease`, and evaluation stops as soon as the lease is exhausted.

These guards are enforced during execution; stop conditions are checked between
parallel batches to decide whether to continue.

## Injection Mode Compatibility

Not all injection modes are safe for parallel trial execution:

| Injection Mode | Parallel Safe | Notes |
| --- | --- | --- |
| `context` | ✅ Yes | Uses contextvars (task-local); see [Thread Pool Examples](../api-reference/thread-pool-examples.md) for thread pools |
| `parameter` | ✅ Yes | Config passed explicitly per call |
| `seamless` | ✅ Yes | Uses context under the hood |

### Attribute Mode Removed

The function-attribute-based injection mode was **removed in v2.x** — it cannot
be made thread-safe under parallel trials, and we are no longer willing to ship
a mode that silently corrupts data when concurrency is enabled. Passing the
removed value as `injection_mode` now raises `ConfigurationError` at decoration
time with migration guidance.

**Why it was removed.** Attribute mode relied on a shared mutable function
attribute (`my_func.current_config`):

- Concurrent trials overwrote the same attribute simultaneously.
- There was no isolation between trials — all shared the same function object.
- Races caused silent data corruption with no surfaced error.

**Use these thread-safe alternatives:**

```python
# ✅ Context mode (recommended for parallel)
@traigent.optimize(
    injection_mode="context",
    parallel_config={"trial_concurrency": 4},
    ...
)
def my_func(query: str) -> str:
    config = traigent.get_config()  # Each trial gets its own config
    return process(query, config)

# ✅ Parameter mode (explicit, also parallel-safe)
@traigent.optimize(
    injection_mode="parameter",
    config_param="config",
    parallel_config={"trial_concurrency": 4},
    ...
)
def my_func(query: str, config: dict) -> str:
    return process(query, config)
```

> If you previously relied on attribute mode for external observation
> (e.g., reading `my_func.current_config` from a monitoring sidecar), migrate
> to `context` mode and read the active configuration from inside the optimized
> function — see the [Attribute Mode migration guide](../user-guide/injection_modes.md#4-attribute-mode-removed-in-v2x) for a use-case → replacement table covering sidecar monitoring, stateful buffers, type-safe access, and basic config reads.

## Related Documentation

- [Thread Pool Examples](../api-reference/thread-pool-examples.md) - Context propagation with custom thread pools
- [Evaluation Guide](./evaluation.md) - Parallel evaluation tips and troubleshooting
