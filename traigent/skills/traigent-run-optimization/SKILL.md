---
name: traigent-run-optimization
description: "Run Traigent optimization: async/sync execution, algorithm selection, cost limits, stop conditions, and parallel trials. Use when calling func.optimize() or optimize_sync(), choosing algorithms (grid/random — locally; bayesian/optuna/tpe run in the Traigent cloud), setting max_trials or cost_limit, configuring parallel execution, or handling CostLimitExceeded."
license: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
metadata:
  author: Traigent
  version: "1.0"
---

# Running Traigent Optimization

## When to Use

Use this skill after you have decorated a function with `@traigent.optimize()` and need to:

- Run optimization (async or sync)
- Choose an algorithm (grid or random locally; smart algorithms run in the Traigent cloud)
- Set trial limits, timeouts, or cost budgets
- Configure parallel trial execution
- Handle cost limit exceptions
- Interpret stop reasons and results

## Async Execution

The primary way to run optimization. Returns an `OptimizationResult`.

```python
import traigent

@traigent.optimize(
    eval_dataset="qa_test.jsonl",
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": [0.1, 0.5, 0.9],
    },
)
def answer(question: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], temperature=cfg["temperature"], prompt=question)

# Run optimization
results = await answer.optimize(max_trials=10, algorithm="grid")
```

### optimize() Parameters

| Parameter | Type | Description |
|---|---|---|
| `algorithm` | `str \| None` | Algorithm: `"grid"` or `"random"` (local). Smart algorithms (`"bayesian"`, `"optuna"`, etc.) run in the Traigent cloud. Falls back to decorator setting. |
| `max_trials` | `int \| None` | Maximum number of trials to run. |
| `timeout` | `float \| None` | Maximum wall-clock time in seconds. |
| `save_to` | `str \| None` | Path to save results to disk. |
| `custom_evaluator` | `Callable \| None` | Override evaluator for this run. |
| `callbacks` | `list[Callable] \| None` | Progress tracking callbacks. |
| `configuration_space` | `dict \| None` | Override config space for this run. |
| `objectives` | `list[str] \| ObjectiveSchema \| None` | Override objectives for this run. |
| `**algorithm_kwargs` | `Any` | Algorithm-specific parameters (e.g., `parameter_order` for grid). |

## Sync Execution

For scripts or notebooks where you do not want to manage an async event loop.

```python
# Option 1: optimize_sync() convenience method
results = answer.optimize_sync(max_trials=10, algorithm="grid")

# Option 2: asyncio.run() wrapper
import asyncio
results = asyncio.run(answer.optimize(max_trials=10, algorithm="grid"))
```

`optimize_sync()` accepts the same parameters as `optimize()`. It creates and manages the event loop internally.

## Algorithm Selection

### Grid Search

Exhaustive search over all configurations in the config space. Deterministic and complete.

```python
results = await func.optimize(max_trials=24, algorithm="grid")

# Control iteration order with parameter_order
results = await func.optimize(
    algorithm="grid",
    parameter_order={"model": 0, "temperature": 1},  # model varies slowest
)
```

**Best for**: Small config spaces (under 50 combinations) where you want to test everything.

### Random Search

Samples configurations randomly from the config space. Good for large spaces where exhaustive search is impractical.

```python
results = await func.optimize(max_trials=20, algorithm="random")
```

**Best for**: Large config spaces, quick exploration, when you have a limited trial budget.

### Smart Algorithms (Traigent Cloud)

Bayesian, Optuna TPE, CMA-ES, NSGA-II, and other smart algorithms are **not available locally**. Passing `algorithm="bayesian"`, `"optuna"`, `"tpe"`, `"nsga2"`, or `"cmaes"` in a local run raises an error. These algorithms run in the Traigent cloud — connect to [Traigent Portal](https://portal.traigent.ai) to use them.

### Quick Comparison

| Algorithm | Availability | Strategy | Config Space Size | Trial Budget |
|---|---|---|---|---|
| `"grid"` | Local SDK | Exhaustive | Small (< 50) | Matches space size |
| `"random"` | Local SDK | Sampling | Any | Limited |
| `"bayesian"` | **Cloud only** | Model-guided | Medium-Large | 15-100 |
| `"optuna"` / TPE / CMA-ES / NSGA-II | **Cloud only** | Advanced sampling | Large | 30+ |

## Cost Controls

Traigent tracks LLM API costs in real time and enforces budgets to prevent runaway spending.

### Setting a Cost Limit

Set the `TRAIGENT_RUN_COST_LIMIT` environment variable (in USD):

```bash
export TRAIGENT_RUN_COST_LIMIT=5.00  # $5 max per optimization run
```

The default limit is $2.00 per run.

### Cost Limit Behavior

Traigent enforces cost limits through **two distinct surfaces**:

**Pre-run (raises `CostLimitExceeded`)**: If cost approval is declined before the first trial — e.g. a non-interactive shell without `TRAIGENT_COST_APPROVED=true`, or the interactive prompt is rejected — Traigent raises `CostLimitExceeded`. `CostLimitExceeded` is a subclass of `OptimizationError`, so `except OptimizationError` also catches it.

```python
from traigent.utils.exceptions import CostLimitExceeded

try:
    results = await func.optimize(max_trials=100, algorithm="random")
except CostLimitExceeded as e:
    print(f"Cost approval declined: ${e.accumulated:.2f} / ${e.limit:.2f}")
```

The exception has two attributes:
- `e.accumulated` (float) - Total cost accumulated when approval was declined.
- `e.limit` (float) - The configured cost limit.

**Mid-run (graceful stop, no exception)**: If accumulated spend reaches the budget *during* a run, Traigent stops gracefully after the current trial and returns a partial `OptimizationResult`. No exception is raised. Check `result.stop_reason`:

```python
results = await func.optimize(max_trials=100, algorithm="random")
if results.stop_reason == "cost_limit":
    print(f"Run stopped at cost limit: ${results.total_cost:.2f} spent")
    # results.best_config and results.trials hold what was collected
```

### Pre-Approving Costs

Skip the interactive cost approval handshake:

```bash
export TRAIGENT_COST_APPROVED=true
```

`TRAIGENT_COST_APPROVED=true` must be the exact environment value `true`; values
such as `1`, `yes`, or `on` do not approve cost-sensitive execution. The runtime
`cost_approved=True` option must be a real boolean; strings are ignored and
logged as warnings.

Pre-approval covers both the cost-limit prompt and the unpriced-model preflight
gate. It proceeds with a warning for unpriced models; it does not add pricing
coverage. Without pre-approval, unpriced models block before trial 1: interactive
shells show a `y/N` prompt, and non-interactive shells fail closed.

Mock LLM mode skips the optimized-function pricing preflight because no provider
spend occurs, but it is not a global `CostEnforcer` bypass.

### Strict Cost Accounting

Fail fast before trial 1 on unpriced models, and fail if runtime cost tracking
cannot extract costs from LLM responses:

```bash
export TRAIGENT_STRICT_COST_ACCOUNTING=true
```

`TRAIGENT_STRICT_COST_ACCOUNTING=true` must be the exact value `true`. Budget
limits are separate: they are controlled by `TRAIGENT_RUN_COST_LIMIT`. A mid-run
overrun stops the run gracefully and returns a partial result
(`result.stop_reason == "cost_limit"`); only a pre-run approval failure raises
`CostLimitExceeded`.

## Stop Conditions

Optimization can stop for several reasons. Check `results.stop_reason`:

| Stop Reason | Trigger |
|---|---|
| `"max_trials_reached"` | Hit the configured `max_trials` limit. |
| `"max_samples_reached"` | Hit the `max_total_examples` limit across all trials. |
| `"timeout"` | Exceeded the `timeout` duration. |
| `"cost_limit"` | Hit the `TRAIGENT_RUN_COST_LIMIT` budget. |
| `"optimizer"` | Algorithm exhausted the search space (e.g., grid search finished). |
| `"plateau"` | No improvement detected over recent trials. |
| `"user_cancelled"` | User cancelled or declined cost approval. |
| `"condition"` | A generic stop condition was triggered. |
| `"error"` | Optimization failed due to an exception. |

```python
results = await func.optimize(max_trials=20, algorithm="grid")

print(f"Stop reason: {results.stop_reason}")
print(f"Trials completed: {len(results.trials)}")
print(f"Best score: {results.best_score}")
```

## Parallel Execution

Run trials and/or examples concurrently using `ParallelConfig`.

```python
from traigent.config.parallel import ParallelConfig
from traigent.api.decorators import ExecutionOptions

@traigent.optimize(
    execution=ExecutionOptions(
        parallel_config=ParallelConfig(
            mode="parallel",
            trial_concurrency=2,    # Run 2 trials at the same time
            example_concurrency=4,  # Evaluate 4 examples concurrently per trial
        ),
    ),
    eval_dataset="large_dataset.jsonl",
    objectives=["accuracy"],
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], prompt=query)

results = await my_func.optimize(max_trials=10, algorithm="random")
```

### ParallelConfig Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `"auto" \| "sequential" \| "parallel"` | `None` | Execution mode. `None` inherits from global config. |
| `trial_concurrency` | `int \| None` | `None` | Max concurrent trials. |
| `example_concurrency` | `int \| None` | `None` | Max concurrent examples per trial. |
| `thread_workers` | `int \| None` | `None` | Thread pool size. |

## Working with Results

`OptimizationResult` contains everything from the optimization run:

```python
results = await func.optimize(max_trials=10, algorithm="grid")

# Best configuration and score
print(results.best_config)     # {"model": "gpt-4", "temperature": 0.5}
print(results.best_score)      # 0.92

# Run metadata
print(results.algorithm)       # "grid"
print(results.duration)        # 45.2 (seconds)
print(results.stop_reason)     # "max_trials_reached"
print(results.total_cost)      # 0.34 (USD, if tracked)
print(results.optimization_id) # "opt_abc123"

# Trial details
for trial in results.trials:
    print(f"Config: {trial.config}, Score: {trial.score}")

# Derived properties
print(results.success_rate)       # 0.9 (fraction of successful trials)
print(len(results.successful_trials))  # 9
print(len(results.failed_trials))      # 1
```

### Applying the Best Config

After optimization, lock in the best configuration for production use:

```python
results = await func.optimize(max_trials=10, algorithm="grid")
func.apply_best_config(results)

# Now calling func uses the best config automatically
answer = func("What is Python?")

# The config is accessible via get_config() inside the function
# and via func.current_config from outside
print(func.current_config)  # {"model": "gpt-4", "temperature": 0.5}
```

## Complete Example

End-to-end optimization from import to results:

```python
import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions
from traigent.config.parallel import ParallelConfig
from traigent.utils.exceptions import CostLimitExceeded

def exact_match(prediction: str, expected: str) -> float:
    return 1.0 if prediction.strip() == expected.strip() else 0.0

@traigent.optimize(
    evaluation=EvaluationOptions(
        eval_dataset="qa_test.jsonl",
        scoring_function=exact_match,
    ),
    execution=ExecutionOptions(
        execution_mode="local",
        parallel_config=ParallelConfig(
            mode="parallel",
            trial_concurrency=2,
            example_concurrency=4,
        ),
        # NOTE: reps_per_trial / reps_aggregation are enterprise-only in this
        # SDK release. Passing non-default values raises a ValidationError at
        # ExecutionOptions construction. Defaults (1 / "mean") are required.
    ),
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": [0.0, 0.3, 0.7],
    },
)
def answer_question(question: str) -> str:
    cfg = traigent.get_config()
    return call_llm(
        model=cfg["model"],
        temperature=cfg["temperature"],
        prompt=question,
    )

async def main():
    try:
        results = await answer_question.optimize(
            max_trials=6,
            algorithm="grid",
            timeout=300.0,
        )
    except CostLimitExceeded as e:
        print(f"Budget exceeded: ${e.accumulated:.2f} / ${e.limit:.2f}")
        return

    print(f"Best config: {results.best_config}")
    print(f"Best score:  {results.best_score}")
    print(f"Stop reason: {results.stop_reason}")
    print(f"Duration:    {results.duration:.1f}s")
    print(f"Total cost:  ${results.total_cost:.2f}" if results.total_cost else "")

    # Apply and use in production
    answer_question.apply_best_config(results)
    answer = answer_question("What is the capital of France?")
    print(f"Answer: {answer}")

import asyncio
asyncio.run(main())
```

## See Also

- `references/algorithms.md` - Detailed algorithm comparison
- `references/parallel-config.md` - Full ParallelConfig reference
- `references/cost-management.md` - Cost enforcement details
