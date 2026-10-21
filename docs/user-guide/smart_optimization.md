# Smart Optimization

Traigent SDKs expose a small local optimizer surface by default:

- `grid`
- `random`

Smart optimization is backend-routed. Use a TraigentBackend typed session or
cloud SDK mode when you need guided search beyond the local defaults. See
[Execution Modes](execution-modes.md) for the full `local` / `cloud` /
`hybrid_api` picture.

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

## Python Cloud Session Example

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

## Recipe: Vary One Setting at a Time (OFAT)

Traigent does not have a built-in one-setting-at-a-time mode yet; it is tracked in
[#1710](https://github.com/Traigent/Traigent/issues/1710). You can get the same
comparison today with the local `grid` optimizer: hold every setting at a baseline,
and run one small grid per setting in which only that setting has more than one value.

```python
import asyncio

import traigent
from traigent import Choices

EVAL_DATASET = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is the capital of Germany?", "expected": "Berlin"},
]

# Baseline: every setting held at this value unless it is being swept.
BASELINE = {"model": "gpt-5.4-mini", "temperature": 0.2, "max_tokens": 200}

# The values to try for each setting, one setting at a time.
SWEEPS = {
    "model": ["gpt-5.4-mini", "claude-haiku-5-5"],
    "temperature": [0.0, 0.2, 0.7],
    "max_tokens": [100, 200, 400],
}


@traigent.optimize(
    eval_dataset=EVAL_DATASET,
    objectives=["accuracy", "cost"],
    configuration_space={name: Choices(values) for name, values in SWEEPS.items()},
    default_config=BASELINE,
    injection_mode="context",
)
def answer(question: str) -> str:
    cfg = traigent.get_config()  # the configuration of the trial being run
    return call_model(
        question,
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )


def one_at_a_time_space(knob: str) -> dict[str, Choices]:
    """Hold every setting at its baseline value, except `knob`, which takes all its values."""
    space = {name: Choices([value]) for name, value in BASELINE.items()}
    space[knob] = Choices(SWEEPS[knob])
    return space


async def run_ofat() -> dict[str, traigent.OptimizationResult]:
    results = {}
    for knob in SWEEPS:
        results[knob] = await answer.optimize(
            algorithm="grid",
            configuration_space=one_at_a_time_space(knob),
        )
    return results


results = asyncio.run(run_ofat())

# Compare: each setting's rows show how its values move accuracy and cost,
# with every other setting held at the baseline.
for knob, result in results.items():
    print(knob)
    for trial in result.trials:
        print(
            f"  {trial.config[knob]!r:<20} "
            f"accuracy={trial.metrics.get('accuracy')} cost={trial.metrics.get('cost')}"
        )
```

Notes:

- Each run also evaluates the `default_config` baseline as its own trial, so the
  baseline appears once as the seed and once as its grid value. Total trials are
  the sum over the sweeps of (values in the sweep + 1), not their product.
- Read each setting's effect from its own rows. Comparing `best_score` across the
  per-setting runs answers a different question, because each run has a different
  search space.
- One-at-a-time runs miss interactions between settings. Once you have a better
  baseline, re-run the recipe around it.
- Every model id in a sweep must have pricing the SDK can see. If it does not, the run stops
  at the cost preflight before any trial; supply per-token pricing with
  `TRAIGENT_CUSTOM_MODEL_PRICING_FILE` first.
- This recipe uses only the local `grid` optimizer, so it needs no backend service.
  Changing `BASELINE` to a new best config and re-running is the "OFAT around the
  peak" pattern described in #1710.
