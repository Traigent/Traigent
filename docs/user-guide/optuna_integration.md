# Optuna Integration in Traigent

> **Cloud-only feature.** Smart algorithms such as TPE, CMA-ES, and NSGA-II are
> **not available in the local SDK**. Passing `algorithm="tpe"`,
> `"optuna_tpe"`, `"optuna_random"`, `"optuna_grid"`, `"optuna_cmaes"`,
> `"optuna_nsga2"`, `"bayesian"`, `"nsga2"`, `"nsgaii"`, `"nsga_ii"`,
> `"cmaes"`, or `"cma_es"` without a reachable Traigent backend raises an
> `OptimizationError`.
>
> For local runs, use `algorithm="grid"` (exhaustive, small spaces) or `algorithm="random"` (large spaces, quick exploration).

## When to use Optuna (via the Traigent cloud)

- **Categorical-heavy search spaces** – Optuna's tree-structured Parzen
  estimator (TPE) provides significantly better exploration than random search.
- **Multi-objective optimisation** – Optuna keeps track of the full Pareto
  frontier instead of collapsing everything into a single scalar.
- **Large or continuous parameter spaces** – CMA-ES and TPE efficiently explore
  spaces where exhaustive grid search is impractical.

## Available Optuna optimisers (cloud)

| Optimiser name    | Registry id       | Recommended use case                         |
|-------------------|-------------------|----------------------------------------------|
| `OptunaTPEOptimizer` | `optuna_tpe`    | General-purpose categorical/continuous search |
| `OptunaRandomOptimizer` | `optuna_random` | Reproducible random baselines              |
| `OptunaCMAESOptimizer` | `optuna_cmaes` | High-dimensional continuous parameters     |
| `OptunaNSGAIIOptimizer` | `optuna_nsga2` | Multi-objective evolutionary strategies   |
| `OptunaGridOptimizer` | `optuna_grid`   | Exhaustive grid search via Optuna          |

These optimisers are available through the Traigent cloud. They are **not importable from the local SDK** — their implementation files have been relocated to the cloud service. Attempting to import them locally raises `ModuleNotFoundError`.

## Smart optimization via the cloud

Connect to the Traigent backend and pass a smart algorithm name through the
standard optimize flow. The cloud service handles sampling, pruning, and
coordination:

```python
# Requires a Traigent backend connection
result = optimized_fn.optimize(algorithm="tpe", max_trials=100)
```

Local calls with smart algorithm names raise a clear error:

```python
from traigent.optimizers.registry import get_optimizer
from traigent.utils.exceptions import OptimizationError

try:
    get_optimizer("tpe", config_space, objectives)
except OptimizationError as e:
    # "Smart optimization ('tpe') runs in the Traigent cloud..."
    print(e)
```

`algorithm="auto"` is different: it is cloud-first, but on connectivity
failures it warns once and degrades to a local grid/random search unless
`TRAIGENT_REQUIRE_CLOUD=1` is set.

## Conditional Parameters

Configuration dictionaries support conditional logic without leaving the
standard Traigent format:

```python
config_space = {
    "model": ["gpt-4", "gpt-3.5"],
    "max_tokens": {
        "type": "int",
        "low": 200,
        "high": 4000,
        "conditions": {"model": "gpt-4"},
        "default": 512,
    },
}
```

When the condition is not met the parameter reuses the provided `default`. The
same schema works for `float` and `categorical` definitions, including optional
`step` and `log` hints.

> **Note:** Conditional support currently covers simple equality checks as shown
> above. For more complex dependencies, supply the appropriate logic in your
> objective function or wrap the optimiser with custom pre-processing.

## Availability

Optuna optimisers require a Traigent cloud connection. They are not part of
the local SDK's registered algorithm set, and they do not silently downgrade to
local search when the backend is unavailable. For local optimization, use
`"grid"` or `"random"` via `func.optimize(algorithm="grid")` or
`func.optimize(algorithm="random")`.
