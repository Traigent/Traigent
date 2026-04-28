# ParallelConfig Reference

`ParallelConfig` controls concurrent execution of trials and examples during optimization.

```python
from traigent.config.parallel import ParallelConfig
```

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `"auto" \| "sequential" \| "parallel" \| None` | `None` | Execution mode. `None` means inherit from parent scope. `"auto"` lets Traigent decide. |
| `trial_concurrency` | `int \| None` | `None` | Maximum number of trials to run concurrently. `None` means inherit. |
| `example_concurrency` | `int \| None` | `None` | Maximum number of examples to evaluate concurrently within a single trial. `None` means inherit. |
| `thread_workers` | `int \| None` | `None` | Size of the thread pool for parallel execution. `None` means inherit. |

## Construction

### Direct Construction

```python
config = ParallelConfig(
    mode="parallel",
    trial_concurrency=2,
    example_concurrency=4,
    thread_workers=8,
)
```

### From Dictionary

```python
config = ParallelConfig.from_dict({
    "mode": "parallel",
    "trial_concurrency": 2,
    "example_concurrency": 4,
})
```

Unknown keys raise `ValueError`:

```python
# Raises ValueError: Unknown parallel_config key(s): ['batch_size']
ParallelConfig.from_dict({"batch_size": 4})
```

### From Legacy Parameters

For backward compatibility with older Traigent versions:

```python
config = ParallelConfig.from_legacy(
    parallel_trials=2,     # Maps to trial_concurrency
    batch_size=4,          # Maps to example_concurrency
    parallel_workers=8,    # Maps to thread_workers
)
```

## Scoping: Where to Set ParallelConfig

ParallelConfig can be set at three levels. More specific scopes override broader ones.

### 1. Global (via traigent.configure)

Applies to all optimization runs in the process:

```python
import traigent

traigent.configure(
    parallel_config=ParallelConfig(
        mode="parallel",
        trial_concurrency=2,
    )
)
```

### 2. Decorator Level

Applies to a specific decorated function:

```python
from traigent.api.decorators import ExecutionOptions

@traigent.optimize(
    execution=ExecutionOptions(
        parallel_config=ParallelConfig(
            mode="parallel",
            trial_concurrency=3,
            example_concurrency=8,
        ),
    ),
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], prompt=query)
```

### 3. Runtime (at optimize() call)

Highest priority. Overrides both global and decorator settings:

```python
# This is passed via ExecutionOptions at decoration time,
# but you can also set configuration_space and other params at runtime
results = await my_func.optimize(max_trials=10, algorithm="grid")
```

## Merge Behavior

When multiple scopes define `ParallelConfig`, they are merged with more specific values winning:

```python
# Global: trial_concurrency=2, example_concurrency=4
# Decorator: trial_concurrency=3 (overrides global)
# Result: trial_concurrency=3, example_concurrency=4 (inherited from global)
```

Fields set to `None` mean "inherit from parent scope". Only explicitly set values override.

## Thread Safety

Parallel execution uses `ThreadPoolExecutor` internally. Traigent ensures thread safety through:

- **contextvars**: Each trial gets its own context. `traigent.get_config()` is thread-safe and returns the correct trial config.
- **Cost enforcement locks**: The `CostEnforcer` uses `threading.RLock()` to prevent race conditions when tracking costs across parallel trials.
- **Immutable config snapshots**: Each trial receives an immutable copy of its configuration.

### Custom Threads Inside Optimized Functions

If you spawn threads inside your optimized function, the Traigent context does not propagate automatically. Use `copy_context_to_thread()`:

```python
from traigent.config.context import copy_context_to_thread
import concurrent.futures

@traigent.optimize(
    execution=ExecutionOptions(
        parallel_config=ParallelConfig(trial_concurrency=2),
    ),
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def batch_process(items: list) -> list:
    cfg = traigent.get_config()
    snapshot = copy_context_to_thread()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(snapshot.run, process_one, item, cfg)
            for item in items
        ]
        return [f.result() for f in futures]
```

## Modes

| Mode | Behavior |
|---|---|
| `"sequential"` | Trials run one at a time. Examples within a trial also run sequentially. Simplest, no concurrency concerns. |
| `"parallel"` | Trials and/or examples run concurrently up to the configured concurrency limits. |
| `"auto"` | Traigent decides based on the config space size, dataset size, and available resources. |

## Practical Guidelines

| Scenario | Recommended Settings |
|---|---|
| Small dataset (< 20 examples) | `example_concurrency=1` (overhead not worth it) |
| Large dataset (100+ examples) | `example_concurrency=4-8` |
| Expensive trials (GPT-4) | `trial_concurrency=1-2` (control costs) |
| Cheap trials (GPT-3.5) | `trial_concurrency=4-8` (maximize throughput) |
| Rate-limited API | Keep total concurrency (`trial * example`) under your rate limit |
