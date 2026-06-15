# Algorithm Reference

Traigent supports two optimization algorithms locally: `"grid"` and `"random"`. Smart algorithms (Bayesian, Optuna TPE, CMA-ES, NSGA-II, etc.) run in the **Traigent cloud** and are not available in the local SDK.

Pass the algorithm name as a string to `optimize()` or `optimize_sync()`.

```python
results = await func.optimize(max_trials=10, algorithm="grid")
```

## Algorithm Comparison

### Local algorithms (available in the SDK)

| Algorithm | Strategy | Config Space Size | Trial Budget | Deterministic | Best For |
|---|---|---|---|---|---|
| `"grid"` | Exhaustive enumeration | Small (< 50 combos) | Must cover full space | Yes | Complete coverage, reproducibility |
| `"random"` | Uniform random sampling | Any | Limited (10-50) | No | Large spaces, quick exploration |

### Cloud algorithms (Traigent cloud only)

Smart algorithms — Bayesian (GP surrogate), Optuna TPE, CMA-ES, NSGA-II, and others — run on the Traigent cloud. Calling `get_optimizer("bayesian")` or passing `algorithm="bayesian"` (or `"optuna"`, `"tpe"`, `"nsga2"`, `"cmaes"`) in a local run raises an error directing you to the cloud. To use these algorithms, connect to [Traigent Portal](https://portal.traigent.ai) and use `execution_mode="hybrid"` or the cloud path when it is available.

## Grid Search

Enumerates every combination in the configuration space. Guaranteed to find the global optimum within the space.

```python
results = await func.optimize(algorithm="grid")
```

### Parameter Order

Control which parameters vary fastest vs slowest:

```python
results = await func.optimize(
    algorithm="grid",
    parameter_order={"model": 0, "temperature": 1, "max_tokens": 2},
)
```

Lower values vary slowest (outer loop), higher values vary fastest (inner loop). This is useful when you want to group trials by model to minimize repeated initial setup costs.

### When to Use

- Configuration space has fewer than 50 total combinations
- You need deterministic, reproducible results
- You want to guarantee the best config in the space is found
- Budget allows testing every combination

### Behavior

- Stops with `stop_reason="optimizer"` when all combinations are exhausted
- If `max_trials` is smaller than the config space, only a prefix is tested
- Iteration order is lexicographic by default (or controlled by `parameter_order`)

## Random Search

Samples configurations uniformly at random from the config space. Each trial is independent.

```python
results = await func.optimize(max_trials=20, algorithm="random")
```

### When to Use

- Large configuration spaces where exhaustive search is impractical
- You have a fixed trial budget and want broad coverage
- Starting point before using cloud-based smart algorithms
- Parameters have similar importance (no strong interactions)

### Behavior

- May sample the same configuration twice (with replacement)
- Stops when `max_trials` is reached
- Provides good coverage of high-dimensional spaces with fewer trials than grid

## Choosing an Algorithm

### Decision Guide

1. **Config space has < 50 combinations?** Use `"grid"` for complete coverage.
2. **Limited budget, large space?** Use `"random"` for broad exploration.
3. **Need adaptive or multi-objective optimization?** Use the [Traigent cloud](https://portal.traigent.ai) — smart algorithms (Bayesian, TPE, CMA-ES, NSGA-II) run there.

### Budget Guidelines (local algorithms)

| Config Space Size | Recommended Algorithm | Suggested max_trials |
|---|---|---|
| 1-10 | `"grid"` | Match space size |
| 10-50 | `"grid"` or `"random"` | Match space size or 20-30 |
| 50+ | `"random"` | 20-50; consider cloud for smarter search |

### Runtime Override

You can set the algorithm at decoration time and override it at runtime:

```python
@traigent.optimize(
    algorithm="grid",  # Default algorithm
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], prompt=query)

# Override at runtime
results = await my_func.optimize(algorithm="random", max_trials=20)
```
