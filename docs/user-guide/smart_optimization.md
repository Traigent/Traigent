# Smart Optimization in Traigent

Traigent's Python SDK has two optimizer surfaces:

- Local optimization: `grid` and `random`.
- Backend-routed smart optimization: `bayesian`, `tpe`, `hyperband`, and `frontier_scout`.

The SDK does not expose Optuna as a Python dependency, install extra, registry
alias, optimizer class, adapter, or coordinator API. Smart strategies are
managed capabilities advertised by the backend.

## When to use smart strategies

Use local `grid` or `random` when you need fully offline runs, CI smoke tests,
or small search spaces. Use backend-routed smart strategies when evaluations are
expensive enough that adaptive search, multi-fidelity pruning, or Pareto
frontier exploration is worth managed orchestration.

| Strategy | Route | Best for |
| --- | --- | --- |
| `grid` | Local | Small finite spaces where exhaustive coverage matters |
| `random` | Local | Fast local exploration and reproducible baselines |
| `bayesian` | Backend | Sample-efficient managed search |
| `tpe` | Backend | Mixed categorical/continuous search with backend state |
| `hyperband` | Backend | Multi-fidelity runs that can stop weak candidates early |
| `frontier_scout` | Backend | Pareto-frontier search across quality, cost, and latency |

## Discover backend-routed strategies

Backend-routed strategies are hidden from the default local surface. Enable
backend smart strategy discovery only when a managed backend is part of the run:

```python
import traigent

traigent.configure(
    feature_flags={"optimizers": {"backend_smart": {"enabled": True}}}
)

strategies = traigent.get_available_strategies()
print(strategies.keys())
# dict_keys(["grid", "random", "bayesian", "frontier_scout", "hyperband", "tpe"])
```

Each smart strategy entry includes `backend_routed=True` and
`local_execution=False`. Direct local construction still fails with guidance to
use `grid` or `random` unless a backend route is available.

## Local examples

```python
results = await function.optimize(
    algorithm="random",
    max_trials=25,
    random_seed=42,
)
```

```python
results = await function.optimize(
    algorithm="grid",
    max_trials=50,
    parameter_order={"model": 0, "temperature": 1},
)
```

## Backend smart examples

Use smart strategy names only with a managed backend that advertises support:

```python
results = await function.optimize(
    algorithm="hyperband",
    max_resource=27,
    min_resource=3,
    reduction_factor=3,
)
```

```python
results = await function.optimize(
    algorithm="frontier_scout",
    candidate_bank_size=128,
    budget_cap_usd=10.0,
)
```

If the backend route is not available, the SDK raises a local-only error rather
than registering a placeholder optimizer.

## Unsupported local Optuna surface

These older local Optuna entrypoints are intentionally not part of the public
Python SDK surface:

- `pip install "traigent[bayesian]"`
- `algorithm="optuna"` or `algorithm="optuna_tpe"`
- `TRAIGENT_OPTUNA_ENABLED`
- `from traigent.optimizers import OptunaTPEOptimizer`
- `OptunaAdapter`, `OptunaCoordinator`, or local Optuna registry aliases

Keep local runs on `grid`/`random`. Use the backend-routed smart strategy names
above when managed orchestration is enabled.
