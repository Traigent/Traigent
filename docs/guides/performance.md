# Performance Tuning Guide

This guide covers parallel execution and performance optimization for TraiGent.

## Parallel Execution

TraiGent supports two levels of parallelism to speed up optimization runs:

### Concurrency Levels

| Level | Parameter | Description |
|-------|-----------|-------------|
| **Example-level** | `example_concurrency` | Run multiple dataset examples in parallel within a single trial |
| **Trial-level** | `trial_concurrency` | Evaluate multiple configurations simultaneously |

### Configuration

Set parallel configuration in the decorator or override at runtime:

```python
import traigent
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.api.decorators import ExecutionOptions

# Create a dataset
ds = Dataset(examples=[
    EvaluationExample(input_data={"x": i}, expected_output=f"val-{i}")
    for i in range(8)
])

@traigent.optimize(
    configuration_space={"p": [1, 2, 3, 4]},
    eval_dataset=ds,
    objectives=["accuracy"],
    execution=ExecutionOptions(
        execution_mode="edge_analytics",
        parallel_config={"example_concurrency": 4, "trial_concurrency": 2},
    ),
)
def fn(x: int) -> str:
    import time
    time.sleep(0.1)  # Simulate work
    return f"val-{x}"

# Override at call-site if needed
# results = await fn.optimize(
#     parallel_config={"example_concurrency": 8, "trial_concurrency": 4}
# )
```

### Guidelines

| Scenario | Recommended Settings |
|----------|---------------------|
| **CPU-bound tasks** | Higher `example_concurrency`, lower `trial_concurrency` |
| **I/O-bound (API calls)** | Moderate both, watch rate limits |
| **Memory-constrained** | Lower both values |
| **Large datasets** | Higher `example_concurrency` |
| **Many configurations** | Higher `trial_concurrency` |

### Rate Limiting Considerations

When making API calls (OpenAI, Anthropic, etc.):

1. Check your provider's rate limits
2. Start conservative: `example_concurrency=4`, `trial_concurrency=2`
3. Increase gradually while monitoring for rate limit errors
4. Consider adding retry logic with exponential backoff

## Privacy-Enabled Execution

For sensitive data, enable privacy mode:

```python
from traigent.api.decorators import ExecutionOptions

@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
    eval_dataset=ds,
    objectives=["accuracy"],
    execution=ExecutionOptions(
        execution_mode="hybrid",
        privacy_enabled=True,  # Never transmit input/output/prompts
    ),
)
def agent(x: int) -> str:
    return f"val-{x}"
```

With `privacy_enabled=True`:
- Input data stays local
- Output data stays local
- Only anonymized metrics are transmitted (if using cloud features)

## See Also

- [Execution Modes](execution-modes.md) - Local, hybrid, and cloud modes
- [Advanced Optimization](advanced-optimization.md) - Budget constraints and strategies
