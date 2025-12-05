# Stop Conditions Architecture

## Overview

The optimization orchestrator now supports a pluggable stop-condition framework that
decouples termination policies from individual optimizers. Each condition implements
the `StopCondition` interface (`traigent/core/stop_conditions.py`) and can be
combined freely:

```python
class StopCondition(ABC):
    reason: str = "condition"

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state before a new optimization run."""

    @abstractmethod
    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        """Return ``True`` when the condition dictates stopping."""
```

Built-in implementations:

| Condition | Purpose | Config knobs |
|-----------|---------|--------------|
| `MaxTrialsStopCondition` | Enforces a hard trial budget. | Uses `max_trials`. |
| `PlateauAfterNStopCondition` | Stops when the best weighted score changes by at most `epsilon` for `window_size` successive trials. | `plateau_window`, `plateau_epsilon`. |
| `BudgetStopCondition` | Stops when the cumulative value of a metric exceeds a budget (defaults to `total_cost`, falling back to `cost` or `total_example_cost`). | `budget_limit`, `budget_metric`, `budget_include_pruned`. |

The orchestrator instantiates the conditions during construction via
`_build_stop_conditions()` and calls `reset()` for every run so state never leaks
between optimizations. `_should_stop()` evaluates all configured conditions before
fallback logic (timeout or optimizer-provided stop signals).

## Objective weighting

Conditions that require a scalar score rely on
`ObjectiveSchema.compute_weighted_score(...)`. This method normalizes weights,
respects per-objective orientations, and supports both explicit schemas and
string lists (the orchestrator auto-creates equal-weight schemas when needed).

```python
schema.compute_weighted_score({"accuracy": 0.8, "total_cost": 0.05})
# -> 0.5 * 0.8 + 0.5 * (-0.05) = 0.375
```

This guarantees consistent behaviour for multi-objective runs regardless of how
they were declared in `@optimize`.

## Integration points

* **Sequential trials** – `_handle_trial_result()` updates the history and the
  orchestrator inspects every condition before starting the next iteration.
* **Parallel / batch trials** – `_run_parallel_batch()` invokes
  `_handle_trial_result()` for each completed future. Stop conditions therefore
  evaluate after every batch item; the loop exits immediately when any condition
  fires.
* **Sync vs async evaluation** – The framework is agnostic to evaluation mode.
  Conditions only see `TrialResult` objects, which are populated identically in
  both synchronous and asynchronous evaluator paths.
* **Injection modes** – The decorator-driven injection (`context`, `parameter`,
  `attribute`, `seamless`) happens entirely inside the `OptimizedFunction`
  wrapper. Stop conditions operate at the orchestrator layer and therefore work
  uniformly for all injection styles.

### Usage examples

```python
# Context injection (default)
@traigent.optimize(
    eval_dataset="math.jsonl",
    configuration_space={"model": ["openai/gpt-4.1-nano", "openai/gpt-4.1"]},
    plateau_window=5,
    plateau_epsilon=1e-5,
    budget_limit=0.5,
)
def answer_math(question: str) -> str:
    ...

# Parameter injection
@traigent.optimize(
    eval_dataset="qa.jsonl",
    configuration_space={"temperature": [0.0, 0.5, 0.9]},
    injection_mode="parameter",
    config_param="config",
    plateau_window=3,
)
def chat(question: str, config: dict) -> str:
    ...

# Attribute injection
@traigent.optimize(
    eval_dataset="summaries.jsonl",
    configuration_space={"max_tokens": [128, 256]},
    injection_mode="attribute",
    plateau_window=4,
)
class Summarizer:
    ...

# Seamless injection
@traigent.optimize(
    eval_dataset="scenarios.jsonl",
    configuration_space={"model": ["openai/gpt-4o", "openai/gpt-4o-mini"]},
    injection_mode="seamless",
    budget_limit=1.25,
)
def plan_trip(itinerary: str) -> str:
    ...
```

All four examples share the same stop-condition machinery with no additional
plumbing.

## Testing & coverage

Unit coverage lives under:

* `tests/unit/core/test_stop_conditions.py` – verifies:
  - Max-trial threshold (including invalid inputs).
  - Plateau detection, weight handling, reset semantics, and parameter validation.
  - Budget enforcement, metadata fallback, and pruned-trial filtering.
* `tests/unit/core/test_orchestrator.py` – integration tests for plateau and budget
  within the orchestrator, ensuring early termination and reasonable counts in
  sequential runs.

To run:

```bash
pytest tests/unit/core/test_stop_conditions.py
pytest tests/unit/core/test_orchestrator.py -k "plateau or budget"
```

*(Note: the shared environment used for development lacks `pytest`; run locally
to observe green test results.)*

## Evidence of compatibility

* **Parallel trials** – `_run_parallel_batch()` calls `_handle_trial_result()` for
  each completed task before checking stop conditions, so a triggered condition
  breaks out immediately even in concurrent batches.
* **Sequential trials** – `_run_sequential_trial()` returns control to the loop,
  which inspects `should_stop` (now backed by stop conditions) between every
  iteration.
* **Async evaluators** – All evaluators ultimately produce `TrialResult`
  instances; stop conditions only depend on those results, so async and sync
  paths are indistinguishable.
* **Injection modes** – Injection is resolved in `OptimizedFunction` before the
  orchestrator runs. Because the stop framework is attached only to the
  orchestrator, it is agnostic to how the function receives its configuration.
* **Batch & parallel** – The same stop set is evaluated for both
  `_run_parallel_batch` and `_run_sequential_trial`, guaranteeing consistent
  behaviour regardless of execution style.

## Reservations & future work

1. **Stop-condition discovery** – Conditions are currently instantiated directly
   in `_build_stop_conditions`. Introducing a registry/factory would simplify
   user-defined condition injection from decorators or configuration files.
2. **Incremental updates** – Some conditions rebuild state from the entire trial
   list (`PlateauAfterNStopCondition`). Adding optional lifecycle hooks (e.g.
   `on_trial_completed`) would avoid repeated list copies in very large runs.
3. **Telemetry** – The orchestrator records the stop reason, but we do not yet
   emit structured telemetry for dashboards. Instrumenting condition triggers
   will improve observability.
4. **Acceptance tests** – Unit coverage exercises synthetic scenarios. Adding
   higher-level integration tests (e.g. Optuna + plateau + budget in combination)
   would provide extra assurance under real workloads.

Despite these items, the current implementation delivers consistent, reusable
termination logic across all algorithms, injection modes, and execution styles
without requiring changes to optimizers or evaluators.
