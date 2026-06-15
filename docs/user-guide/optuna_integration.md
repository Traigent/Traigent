# Optuna Integration in Traigent

> **Cloud-only feature.** Optuna-based optimisers (TPE, CMA-ES, NSGA-II, and others) are **not available in the local SDK**. Calling `get_optimizer("optuna_tpe")` or passing `algorithm="optuna"` / `"tpe"` / `"nsga2"` / `"cmaes"` in a local run raises an error. To use these algorithms, connect to [Traigent Portal](https://portal.traigent.ai).
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

These optimisers are available through the Traigent cloud. The code examples below illustrate the API shape for reference; they require a cloud-connected run, not a local SDK call.

## Ask/Tell coordination for distributed workloads

Edge deployments typically evaluate trials outside the coordinator process.
Traigent now exposes the `OptunaCoordinator` helper to manage that workflow:

```python
from traigent.optimizers.optuna_coordinator import OptunaCoordinator

coordinator = OptunaCoordinator(
    config_space=config_space,
    objectives=["accuracy", "cost"],
)

# Ask for a batch of configurations, dispatch them to remote workers
batch = coordinator.ask_batch(n_suggestions=4)

for trial_config in batch:
    trial_id = trial_config["_trial_id"]
    # Run the trial remotely and stream intermediate metrics
    should_prune = coordinator.report_intermediate(trial_id, step=3, value=0.55)
    if should_prune:
        coordinator.tell_pruned(trial_id, step=3)
    else:
        coordinator.tell_result(trial_id, values=[0.82, 0.11])
```

Additional utilities:

- `EdgeExecutor` – reference executor that demonstrates intermittent
  connectivity handling.
- `BatchOptimizer` – helper that batches requests and feeds results back into the
  coordinator, ideal for homogeneous fleets of workers.
- `RateLimitedOptimizer` – wraps a coordinator and throttles how quickly new
  trials are requested, useful for constrained edge deployments.

Implementation details live alongside the coordinator and batching utilities in
`traigent/optimizers/optuna_coordinator.py` and
`traigent/optimizers/batch_optimizers.py`.

### Conditional Parameters

Configuration dictionaries now support conditional logic without leaving the
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

### Study Visualizations

Call `generate_optuna_visualizations(study)` from
`traigent.optimizers.optuna_utils` to obtain Optuna's built-in optimization
history, parameter importance, parallel coordinate, and contour plots. Each
entry in the returned dictionary is a Plotly figure ready for display in a
notebook or dashboard.

### Storage Backends

Pass any Optuna storage URL (e.g. `sqlite:///traigent_optuna.db`,
`postgresql://user:pass@host:5432/optuna`, `redis://localhost:6379/0`) via the
optimizer constructor or coordinator `coordinator_kwargs`. Traigent forwards the
string directly to `optuna.create_study`; consult Optuna's documentation for
backend-specific prerequisites (migrations, credentials, network access).

## Adapter for existing decorators (cloud reference)

Some code paths call Traigent's optimiser decorator directly instead of the
registry. The `OptunaAdapter` translates those calls without requiring refactors.
**This runs via the Traigent cloud — it is not available in the local SDK.**

```python
# Cloud-only: requires a cloud-connected Traigent session
from traigent.optimizers.optuna_adapter import OptunaAdapter


def objective(**config):
    return {
        "accuracy": run_accuracy_benchmark(config),
        "cost": estimate_cost(config),
    }

result = OptunaAdapter.optimize(
    objective,
    config_space,
    ["accuracy", "cost"],
    algorithm="nsga2",   # cloud-only algorithm
    n_trials=75,
)

print("Best parameters", result["best_params"])
```

## Availability

Optuna optimisers require a Traigent cloud connection. They are not part of
the local SDK's registered algorithm set. For local optimization, use `"grid"`
or `"random"` via `func.optimize(algorithm="grid")` or
`func.optimize(algorithm="random")`.
