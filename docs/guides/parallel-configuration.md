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

## Related Documentation

- [Thread Pool Examples](../api-reference/thread-pool-examples.md) - Context propagation with custom thread pools
- [Evaluation Guide](./evaluation.md) - Parallel evaluation tips and troubleshooting
