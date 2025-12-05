# Optuna Integration in TraiGent

The Optuna integration augments TraiGent's optimisation toolkit with a modern
ask/tell capable backend. Optuna-based optimisers ship **in addition** to the
existing random, grid, and Bayesian implementations—change the algorithm name
to opt into the new behaviour.

## When to use Optuna

- **Categorical-heavy search spaces** – Optuna's tree-structured Parzen
  estimator (TPE) provides significantly better exploration than our previous
  Gaussian-process implementation.
- **Multi-objective optimisation** – Optuna keeps track of the full Pareto
  frontier instead of collapsing everything into a single scalar.
- **Edge and distributed execution** – the ask/tell flow allows you to run
  trials on remote devices (or slow environments) and report metrics back to a
  central coordinator.

## New optimisers

| Optimiser name    | Registry id       | Recommended use case                         |
|-------------------|-------------------|----------------------------------------------|
| `OptunaTPEOptimizer` | `optuna_tpe`    | General-purpose categorical/continuous search |
| `OptunaRandomOptimizer` | `optuna_random` | Reproducible random baselines              |
| `OptunaCMAESOptimizer` | `optuna_cmaes` | High-dimensional continuous parameters     |
| `OptunaNSGAIIOptimizer` | `optuna_nsga2` | Multi-objective evolutionary strategies   |
| `OptunaGridOptimizer` | `optuna_grid`   | Exhaustive grid search via Optuna          |

All optimisers reuse the existing `BaseOptimizer` interface. Switching from the
legacy Bayesian optimiser is as simple as updating the `algorithm` attribute:

```python
from traigent.optimizers.registry import get_optimizer

config_space = {
    "model": ["gpt-4", "gpt-3.5"],
    "temperature": (0.0, 1.0),
    "max_tokens": (256, 1024),
}

optimizer = get_optimizer(
    "optuna_tpe",
    config_space=config_space,
    objectives=["accuracy", "cost"],
    max_trials=50,
)

config = optimizer.suggest_next_trial([])
# ... evaluate configuration on your workload ...
optimizer.report_trial_result(config["_optuna_trial_id"], [0.82, 0.12])
```

## Ask/Tell coordination for distributed workloads

Edge deployments typically evaluate trials outside the coordinator process.
TraiGent now exposes the `OptunaCoordinator` helper to manage that workflow:

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

Refer to the [Optuna Integration Plan](../plans/optuna_integration_plan.md) for
the detailed architecture and design decisions behind the coordinator and
batching layers.

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

## Adapter for existing decorators

Some code paths call TraiGent's optimiser decorator directly instead of the
registry. The `OptunaAdapter` translates those calls without requiring refactors:

```python
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
    algorithm="nsga2",
    n_trials=75,
)

print("Best parameters", result["best_params"])
```

## Dependency

The Optuna features require the `optuna` package. It has been added to the
`requirements-bayesian.txt` optional extras—install via `pip install -r
requirements/requirements-bayesian.txt` or include `optuna>=4.5.0` in your
project dependencies.
