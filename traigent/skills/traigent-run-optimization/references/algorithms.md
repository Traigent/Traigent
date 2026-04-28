# Algorithm Reference

Traigent supports four optimization algorithms. Pass the algorithm name as a string to `optimize()` or `optimize_sync()`.

```python
results = await func.optimize(max_trials=10, algorithm="grid")
```

## Algorithm Comparison

| Algorithm | Strategy | Config Space Size | Trial Budget | Deterministic | Best For |
|---|---|---|---|---|---|
| `"grid"` | Exhaustive enumeration | Small (< 50 combos) | Must cover full space | Yes | Complete coverage, reproducibility |
| `"random"` | Uniform random sampling | Any | Limited (10-50) | No | Large spaces, quick exploration |
| `"bayesian"` | Surrogate model guided | Medium-Large | 15-100 | No | Expensive trials, continuous params |
| `"optuna"` | Advanced TPE sampling | Large | 30+ | No | Advanced users, multi-objective |

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

Lower values vary slowest (outer loop), higher values vary fastest (inner loop). This is useful when you want to group trials by model to minimize cold-start costs.

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
- Early exploration phase before switching to Bayesian
- Parameters have similar importance (no strong interactions)

### Behavior

- May sample the same configuration twice (with replacement)
- Stops when `max_trials` is reached
- Provides good coverage of high-dimensional spaces with fewer trials than grid

## Bayesian Optimization

Uses a probabilistic surrogate model to predict which configurations are likely to perform well, then focuses trials on the most promising regions. Backed by Optuna's Tree-structured Parzen Estimator (TPE).

```python
results = await func.optimize(max_trials=30, algorithm="bayesian")
```

### When to Use

- Trials are expensive (each trial costs real money via LLM API calls)
- Configuration space has continuous parameters (temperature, top_p)
- You want to find good configs with fewer trials than random/grid
- Medium to large config spaces (50-10,000 combinations)

### Behavior

- First few trials are random (exploration phase)
- Subsequent trials are guided by the surrogate model (exploitation)
- Balances exploration vs exploitation automatically
- More sample-efficient than random search for smooth objective landscapes

### Limitations

- Requires several initial random trials before the model is useful (typically 5-10)
- Less effective for purely categorical spaces with no ordinal structure
- Non-deterministic: different runs may find different optima

## Optuna (Advanced)

Direct integration with the Optuna optimization framework. Provides access to Optuna's full feature set including advanced pruning, multi-objective optimization, and custom samplers.

```python
results = await func.optimize(max_trials=50, algorithm="optuna")
```

### When to Use

- You need advanced Optuna features (pruning, custom samplers)
- Very large or complex search spaces
- Multi-objective optimization with Pareto frontier analysis
- You are already familiar with Optuna and want fine-grained control

### Behavior

- Uses TPE sampler by default
- Supports early stopping of unpromising trials (pruning)
- Can handle mixed categorical/continuous/integer parameter spaces

## Choosing an Algorithm

### Decision Guide

1. **Config space has < 50 combinations?** Use `"grid"` for complete coverage.
2. **Limited budget, large space?** Use `"random"` for broad exploration.
3. **Trials are expensive, want efficiency?** Use `"bayesian"` for guided search.
4. **Need advanced features?** Use `"optuna"` for full Optuna access.

### Budget Guidelines

| Config Space Size | Recommended Algorithm | Suggested max_trials |
|---|---|---|
| 1-10 | `"grid"` | Match space size |
| 10-50 | `"grid"` or `"random"` | Match space size or 20-30 |
| 50-500 | `"random"` or `"bayesian"` | 20-50 |
| 500+ | `"bayesian"` or `"optuna"` | 30-100 |

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
results = await my_func.optimize(algorithm="bayesian", max_trials=20)
```
