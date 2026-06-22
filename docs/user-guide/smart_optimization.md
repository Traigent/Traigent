# Smart Optimization

Traigent SDKs expose a small local optimizer surface by default:

- `grid`
- `random`

Smart optimization is backend-routed. Use a TraigentBackend typed session or
hybrid SDK mode when you need guided search beyond the local defaults.

## Public Smart Strategies

| Strategy         | Runs In         | Use Case                                                 |
| ---------------- | --------------- | -------------------------------------------------------- |
| `bayesian`       | TraigentBackend | Public alias for guided TPE-style search                 |
| `tpe`            | TraigentBackend | Guided search over categorical and continuous spaces     |
| `optuna`         | TraigentBackend | Managed Optuna-family routing                            |
| `optuna_tpe`     | TraigentBackend | Explicit TPE routing                                     |
| `optuna_random`  | TraigentBackend | Managed random-search routing                            |
| `optuna_grid`    | TraigentBackend | Managed grid-search routing                              |
| `optuna_cmaes`   | TraigentBackend | Managed CMA-ES routing                                   |
| `optuna_nsga2`   | TraigentBackend | Managed NSGA-II routing                                  |
| `nsga2`          | TraigentBackend | NSGA-II alias                                            |
| `nsgaii`         | TraigentBackend | NSGA-II alias                                            |
| `nsga_ii`        | TraigentBackend | NSGA-II alias                                            |
| `cmaes`          | TraigentBackend | CMA-ES alias                                             |
| `cma_es`         | TraigentBackend | CMA-ES alias                                             |

Local SDK runs accept `grid` and `random`. Smart strategies listed above are
validated by the SDK and routed to TraigentBackend.

## Python Hybrid Session Example

```python
import traigent
from traigent import Choices, Range


@traigent.optimize(
    eval_dataset=[
        {"input": "What is the capital of France?", "expected": "Paris"},
        {"input": "What is the capital of Germany?", "expected": "Berlin"},
    ],
    objectives=["accuracy", "cost"],
    model=Choices(["fast", "accurate"]),
    temperature=Range(0.0, 1.0),
)
def answer_quality(question: str, model: str, temperature: float) -> str:
    return call_model(question, model=model, temperature=temperature)


result = await answer_quality.optimize(
    max_trials=12,
    algorithm="tpe",
)
```

Use `bayesian`, `tpe`, or one of the accepted Optuna-family aliases when you
want managed smart search through TraigentBackend.

## Local Defaults

For offline/local SDK runs, keep using `grid` or `random`. These require no
backend service and remain deterministic/reproducible under the existing local
SDK execution path.
