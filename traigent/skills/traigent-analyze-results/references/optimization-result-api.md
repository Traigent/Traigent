# OptimizationResult and TrialResult API Reference

## OptimizationResult

The `OptimizationResult` dataclass is returned by `func.optimize()` and contains all data from an optimization run.

### Fields

| Field | Type | Description |
|---|---|---|
| `trials` | `list[TrialResult]` | All trial results from the optimization run, in execution order. |
| `best_config` | `dict[str, Any]` | The configuration that achieved the best objective score. Empty dict if no trial succeeded. |
| `best_score` | `float \| None` | The best objective score achieved. `None` when no trial produced a valid, rankable score. |
| `optimization_id` | `str` | Unique identifier for this optimization run. |
| `duration` | `float` | Total wall-clock time in seconds for the entire optimization. |
| `convergence_info` | `dict[str, Any]` | Dictionary with convergence statistics (see convergence-patterns.md for fields). |
| `status` | `OptimizationStatus` | Final status of the optimization. One of: `NOT_STARTED`, `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`. |
| `objectives` | `list[str]` | List of objective metric names being optimized (e.g., `["accuracy"]`). |
| `algorithm` | `str` | Name of the optimization algorithm used. |
| `timestamp` | `datetime` | When the optimization completed. |
| `metadata` | `dict[str, Any]` | Additional metadata from the optimization run. Default: `{}`. |
| `total_cost` | `float \| None` | Total API cost incurred in USD. `None` if cost tracking is not available. |
| `total_tokens` | `int \| None` | Total tokens consumed across all trials. `None` if token tracking is not available. |
| `metrics` | `dict[str, Any]` | Aggregated metrics across all trials. Default: `{}`. |
| `stop_reason` | `StopReason \| None` | Why the optimization stopped. See StopReason values below. |

### Properties

| Property | Return Type | Description |
|---|---|---|
| `experiment_stats` | `ExperimentStats` | Lazy-loaded, memoized aggregate statistics for the full experiment. Includes `total_duration`, `total_cost`, `unique_configurations`, `trial_counts`, `average_trial_duration`, `cost_per_configuration`, `success_rate`. |
| `successful_trials` | `list[TrialResult]` | Filtered list of trials with `status == COMPLETED`. |
| `failed_trials` | `list[TrialResult]` | Filtered list of trials with `status == FAILED`. |
| `success_rate` | `float` | Ratio of successful trials to total trials. Returns `0.0` if no trials exist. |
| `best_metrics` | `dict[str, float]` | Full metrics dict from the best-scoring trial. Returns `{}` if no trials exist. |

### Usage

```python
results = func.optimize(dataset="data.jsonl")

# Direct field access
print(results.best_config)       # {"model": "gpt-4o", "temperature": 0.0}
print(results.best_score)        # 0.92
print(results.optimization_id)   # "opt-a1b2c3d4"
print(results.duration)          # 45.3
print(results.status)            # OptimizationStatus.COMPLETED
print(results.stop_reason)       # "max_trials_reached"
print(results.total_cost)        # 0.0234
print(results.total_tokens)      # 15420

# Computed properties
print(results.success_rate)      # 0.8
print(results.best_metrics)      # {"accuracy": 0.92, "latency": 0.8}
print(len(results.successful_trials))  # 8
print(len(results.failed_trials))      # 2

# Experiment stats (lazy-loaded)
stats = results.experiment_stats
print(stats.total_duration)           # 45.3
print(stats.total_cost)              # 0.0234
print(stats.unique_configurations)    # 6
print(stats.trial_counts)            # {"total": 10, "completed": 8, "failed": 2, ...}
print(stats.average_trial_duration)   # 4.53
print(stats.cost_per_configuration)   # 0.0039
print(stats.success_rate)            # 0.8
```

---

## TrialResult

The `TrialResult` dataclass represents the outcome of a single optimization trial.

### Fields

| Field | Type | Description |
|---|---|---|
| `trial_id` | `str` | Unique identifier for this trial. |
| `config` | `dict[str, Any]` | The configuration used for this trial (e.g., `{"model": "gpt-4o", "temperature": 0.5}`). |
| `metrics` | `dict[str, float]` | Metric values produced by this trial (e.g., `{"accuracy": 0.85, "latency": 1.2}`). |
| `status` | `TrialStatus` | Status of this trial. One of: `NOT_STARTED`, `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`, `PRUNED`. |
| `duration` | `float` | Wall-clock execution time for this trial in seconds. |
| `timestamp` | `datetime` | When this trial was executed. |
| `error_message` | `str \| None` | Error message if the trial failed. `None` for successful trials. |
| `metadata` | `dict[str, Any]` | Additional metadata (e.g., example-level results, provider info). Default: `{}`. |

### Properties

| Property | Return Type | Description |
|---|---|---|
| `is_successful` | `bool` | `True` if `status == TrialStatus.COMPLETED`. |

### Methods

| Method | Signature | Description |
|---|---|---|
| `get_metric` | `get_metric(name: str, default: float \| None = None) -> float \| None` | Safely retrieve a metric value by name, returning `default` if the metric is not present. |

### Usage

```python
for trial in results.trials:
    # Field access
    print(trial.trial_id)       # "trial-001"
    print(trial.config)         # {"model": "gpt-4o-mini", "temperature": 0.3}
    print(trial.metrics)        # {"accuracy": 0.85, "latency": 1.2}
    print(trial.status)         # TrialStatus.COMPLETED
    print(trial.duration)       # 3.7
    print(trial.timestamp)      # datetime(2024, 1, 15, 10, 30, 0)
    print(trial.error_message)  # None (or error string for failed trials)
    print(trial.metadata)       # {}

    # Property
    if trial.is_successful:
        # Method
        accuracy = trial.get_metric("accuracy")          # 0.85
        latency = trial.get_metric("latency_ms", 0.0)    # 0.0 (not present, returns default)
```

---

## StopReason Values

`StopReason` is a string literal type. Possible values:

| Value | Description |
|---|---|
| `"max_trials_reached"` | The configured `max_trials` limit was reached. |
| `"max_samples_reached"` | The maximum number of samples/examples was reached. |
| `"timeout"` | The optimization exceeded its timeout duration. |
| `"cost_limit"` | The accumulated cost exceeded the configured budget. |
| `"optimizer"` | The optimizer decided to stop (e.g., configuration space exhausted). |
| `"plateau"` | A plateau was detected (no improvement over recent trials). |
| `"user_cancelled"` | The user cancelled the optimization or declined a cost approval prompt. |
| `"condition"` | A generic stop condition was triggered. |
| `"error"` | The optimization failed due to an unrecoverable exception. |
| `"vendor_error"` | A provider-side error occurred (rate limit, quota exceeded, service unavailable). |
| `"network_error"` | A network connectivity failure prevented further trials. |

---

## OptimizationStatus Enum

```python
from traigent.api.types import OptimizationStatus

OptimizationStatus.NOT_STARTED   # "not_started"
OptimizationStatus.PENDING       # "pending"
OptimizationStatus.RUNNING       # "running"
OptimizationStatus.COMPLETED     # "completed"
OptimizationStatus.FAILED        # "failed"
OptimizationStatus.CANCELLED     # "cancelled"
```

## TrialStatus Enum

```python
from traigent.api.types import TrialStatus

TrialStatus.NOT_STARTED  # "not_started"
TrialStatus.PENDING      # "pending"
TrialStatus.RUNNING      # "running"
TrialStatus.COMPLETED    # "completed"
TrialStatus.FAILED       # "failed"
TrialStatus.CANCELLED    # "cancelled"
TrialStatus.PRUNED       # "pruned"
```

---

## ExperimentStats

The `ExperimentStats` dataclass is accessible via `results.experiment_stats` and provides aggregate statistics.

### Fields

| Field | Type | Description |
|---|---|---|
| `total_duration` | `float` | Total wall-clock time in seconds. |
| `total_cost` | `float` | Total API cost in USD. |
| `unique_configurations` | `int` | Number of distinct configurations tested. |
| `trial_counts` | `dict[str, int]` | Breakdown by status: `total`, `completed`, `failed`, `cancelled`, `running`, `pending`, `not_started`, `exceptions`. |
| `average_trial_duration` | `float \| None` | Mean trial duration in seconds. `None` if no valid durations. |
| `cost_per_configuration` | `float \| None` | Average cost per unique configuration. `None` if no configurations. |
| `success_rate` | `float \| None` | Ratio of successful trials to total. `None` if no trials. |
| `error_message` | `str \| None` | Error message if stats computation failed. |

### to_dict()

```python
stats_dict = results.experiment_stats.to_dict()
# Returns all fields as a plain dictionary for logging or serialization.
```
