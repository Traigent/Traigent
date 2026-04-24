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
| `MetricLimitStopCondition` | Soft stop when the cumulative value of a named completed-trial metric reaches a limit. | `metric_limit`, `metric_name`, `metric_include_pruned`. |
| `CostLimitStopCondition` | Hard money-spend guard backed by `CostEnforcer` permits before execution starts. Parallel-safe. | `cost_limit`, `cost_approved`. |
| `HypervolumeConvergenceStopCondition` | Stops when hypervolume improvement remains below a threshold across a sliding window. | `convergence_metric="hypervolume_improvement"`, `convergence_window`, `convergence_threshold`. |

`budget_limit` remains as a deprecated alias for `metric_limit` for compatibility.
New code should not use it.

## Cost limit vs metric limit

Use `cost_limit` for user-facing spend control. It reserves cost before each trial
through `CostEnforcer`, works with parallel execution, and prevents launching new
work when remaining approved spend is insufficient.

Use `metric_limit` for non-cost cumulative metrics where post-hoc summation is the
right semantic: total tokens, cumulative latency, number of failed cases, or a
custom evaluator counter. Because it observes completed `TrialResult` metrics, it
can stop only before the next trial or batch item. In parallel mode, already
running trials may finish after the limit is reached.

`metric_name` is required with `metric_limit`:

```python
@traigent.optimize(
    eval_dataset="qa.jsonl",
    configuration_space={"temperature": [0.0, 0.5, 0.9]},
    metric_limit=50_000,
    metric_name="total_tokens",
)
def chat(question: str) -> str:
    ...
```

Legacy migration:

```python
# Deprecated compatibility path. Emits a DeprecationWarning.
@traigent.optimize(
    eval_dataset="qa.jsonl",
    configuration_space={"temperature": [0.0, 0.5, 0.9]},
    budget_limit=50_000,
    budget_metric="total_tokens",
)
def chat(question: str) -> str:
    ...
```

If a legacy `budget_limit` call omits the metric name, Traigent defaults to
`total_cost` for compatibility and warns that money spend control should use
`cost_limit` instead.

The orchestrator instantiates the conditions during construction via
`_configure_stop_conditions()` and calls `reset()` for every run so state never leaks
between optimizations. `_should_stop()` evaluates all configured conditions before
fallback logic (timeout or optimizer-provided stop signals).

## Vendor-error pause semantics

When every trial in a parallel batch fails with a vendor-classifiable error
(rate limit, quota, service unavailable, insufficient funds), the orchestrator
prompts the configured `PausePromptAdapter` (if any) before deciding whether to
stop with `stop_reason = "vendor_error"`. In non-interactive runs, or when no
adapter is configured, the run stops — matching sequential mode.

"Resume" currently means **continue after recording the failed batch** — the
failed `TrialResult`s are committed to the history before the vendor check runs,
so the optimizer moves on to the next set of configs rather than retrying the
same batch. Callers who need retry-the-same-batch semantics would need to move
the vendor check ahead of result commit, or add explicit retry bookkeeping.

When a batch mixes categories (e.g. some rate_limit, some insufficient_funds),
the most severe category is surfaced to the adapter via category precedence:
`INSUFFICIENT_FUNDS` < `QUOTA_EXHAUSTED` < `SERVICE_UNAVAILABLE` < `RATE_LIMIT`.
This prevents offering "resume" when at least one trial failed for a
non-recoverable reason.

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
    cost_limit=0.5,
    cost_approved=True,
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
    metric_limit=100_000,
    metric_name="total_tokens",
)
def plan_trip(itinerary: str) -> str:
    ...
```

All four examples share the same stop-condition machinery with no additional
plumbing.

Built-in convergence stops report `OptimizationResult.stop_reason = "convergence"`.
Custom stop-condition instances still report the generic `"condition"` reason
unless they are explicitly mapped into the public `StopReason` contract.

## Testing & coverage

Unit coverage lives under:

* `tests/unit/core/test_stop_conditions.py` – verifies:
  - Max-trial threshold (including invalid inputs).
  - Plateau detection, weight handling, reset semantics, and parameter validation.
  - Metric-limit enforcement, compatibility fallback, and pruned-trial filtering.
* `tests/unit/core/test_orchestrator.py` – integration tests for plateau and metric limit
  within the orchestrator, ensuring early termination and reasonable counts in
  sequential runs.

To run:

```bash
pytest tests/unit/core/test_stop_conditions.py
pytest tests/unit/core/test_orchestrator.py -k "plateau or metric_limit"
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
