# TraiGent SDK API Reference

Accurate documentation for TraiGent SDK v1.0.0.

This document reflects the current implementation of TraiGent SDK.

## Core Decorator

### `@traigent.optimize()`

**Main decorator for making functions optimizable**

```python
def optimize(
    eval_dataset: str | list[str] | Dataset | None = None,
    objectives: Sequence[str] | ObjectiveSchema | None = None,
    configuration_space: dict[str, Any] | None = None,
    default_config: dict[str, Any] | None = None,
    constraints: list[Callable] | None = None,
    injection_mode: str | InjectionMode = InjectionMode.CONTEXT,
    config_param: str | None = None,
    auto_override_frameworks: bool = True,
    framework_targets: list[str] | None = None,
    execution_mode: str = "edge_analytics",
    local_storage_path: str | None = None,
    minimal_logging: bool = True,
    parallel_config: ParallelConfig | dict | None = None,
    privacy_enabled: bool | None = None,
    mock_mode_config: dict[str, Any] | None = None,
    custom_evaluator: Callable | None = None,
    scoring_function: Callable | None = None,
    metric_functions: dict[str, Callable] | None = None,
    evaluation: EvaluationOptions | dict[str, Any] | None = None,
    injection: InjectionOptions | dict[str, Any] | None = None,
    execution: ExecutionOptions | dict[str, Any] | None = None,
    mock: MockModeOptions | dict[str, Any] | None = None,
    **kwargs: Any,
) -> OptimizedFunction
```

> **New in 1.1** – supply grouped `EvaluationOptions`, `InjectionOptions`, `ExecutionOptions`, and `MockModeOptions` (or plain dicts) to keep the decorator concise. Legacy keyword arguments remain supported; conflicting values between a bundle and a direct keyword raise `TypeError`.

**Parameters**

| Parameter | Type | Default | Description | Notes |
| --- | --- | --- | --- | --- |
| `eval_dataset` | `str \| list[str] \| Dataset \| None` | `None` | Evaluation data used for scoring trials. Accepts a JSONL path, list of paths, or a pre-built `Dataset`. | Files must expose `input` and `expected_output` fields. Empty datasets raise validation errors. |
| `objectives` | `Sequence[str] \| ObjectiveSchema \| None` | `["accuracy"]` | Objectives to optimize. Lists are converted to `ObjectiveSchema`. | Provide an `ObjectiveSchema` when you need explicit orientations/weights. |
| `configuration_space` | `dict[str, Any] \| None` | `{}` | Search space describing tunable parameters. | Required for optimization. Lists denote discrete choices; `(min, max)` tuples denote ranges. |
| `default_config` | `dict[str, Any] \| None` | `{}` | Configuration applied before optimization runs. | Missing keys fall back to values detected in the decorated function. |
| `constraints` | `list[Callable] \| None` | `None` | Hard constraints evaluated before running a trial. | Callables should accept `(config, metrics=None)` and return `True/False`. |
| `injection_mode` | `str \| InjectionMode` | `"context"` | How optimized params reach your function. Supports `"context"`, `"parameter"`, `"attribute"`, `"seamless"`. | `"decorator"` is deprecated and maps to `"attribute"`. |
| `config_param` | `str \| None` | `None` | Parameter name used when `injection_mode="parameter"`. | Ignored for other injection modes. |
| `auto_override_frameworks` | `bool` | `True` | Enables automatic overrides for supported LLM clients (OpenAI, Anthropic, LangChain, etc.). | Disable when you manage client wiring manually. |
| `framework_targets` | `list[str] \| None` | `None` | Explicit list of framework classes to override. | Only applied when `auto_override_frameworks` is `True`. |
| `execution_mode` | `str` | `"edge_analytics"` | Determines where orchestration occurs. Supports `"edge_analytics"` (local-first) and `"cloud"` today. | `"hybrid"`/`"standard"` are **not yet supported** in OSS builds; `"privacy"` maps to `"hybrid"` + `privacy_enabled=True` but remains experimental. |
| `local_storage_path` | `str \| None` | `None` | Custom directory for persisted results in edge analytics mode. | Must be writable when `execution_mode="edge_analytics"`. Ignored for cloud runs. |
| `minimal_logging` | `bool` | `True` | Suppresses verbose logs in privacy-sensitive modes. | Only meaningful when running locally; cloud mode ignores this hint. |
| `parallel_config` | `ParallelConfig \| dict \| None` | `None` | Unified concurrency configuration for sequential/parallel runs. | Runtime overrides from `optimize()` take precedence over decorator/global settings. Bundled automatically when you pass `execution=ExecutionOptions(...)`. |
| `privacy_enabled` | `bool \| None` | `None` | Redacts prompts/responses from telemetry and logs. | Effective only for `"edge_analytics"` today. For `"cloud"` it is ignored. |
| `mock_mode_config` | `dict[str, Any] \| None` | `None` | Overrides behaviour when `TRAIGENT_MOCK_MODE` is enabled. | Keys include `"enabled"`, `"override_evaluator"`, `"base_accuracy"`, `"variance"`. |
| `custom_evaluator` | `Callable \| None` | `None` | Fully custom evaluation routine returning `ExampleResult`. | Takes precedence over `scoring_function` unless mock mode forces override. |
| `scoring_function` | `Callable \| None` | `None` | Simple scorer returning a numeric value or dict per example. | Used when no custom evaluator is supplied. |
| `metric_functions` | `dict[str, Callable] \| None` | `None` | Map of metric names to scoring callables. | Merges with `scoring_function`; useful for multiple metrics. |
| `evaluation` | `EvaluationOptions \| dict \| None` | `None` | Bundle `eval_dataset`, `custom_evaluator`, `scoring_function`, and `metric_functions`. | Preferred for new code; legacy keywords continue to work. |
| `injection` | `InjectionOptions \| dict \| None` | `None` | Bundle `injection_mode`, `config_param`, `auto_override_frameworks`, and `framework_targets`. | Conflicting values with direct keywords raise `TypeError`. |
| `execution` | `ExecutionOptions \| dict \| None` | `None` | Bundle execution-mode, storage, logging, privacy, and parallelism knobs. | A clean alternative to managing half a dozen keywords. |
| `mock` | `MockModeOptions \| dict \| None` | `None` | Bundle mock-mode behaviour tweaks such as `enabled`, `base_accuracy`, `variance`. | Converts to `mock_mode_config` internally. |
| `**kwargs` | `Any` | – | Additional optimizer settings routed to downstream components. | Recognised keys include `algorithm`, `max_trials`, `timeout`, `cache_policy`, and stop-condition parameters. Unknown keys are forwarded untouched. |

> **Removed parameter**: `commercial_mode` has been fully retired. To opt into managed cloud orchestration, set `execution_mode="cloud"` (or `execution=ExecutionOptions(execution_mode="cloud")`).

**Usage Notes**

- The default execution mode is now `"edge_analytics"` to keep evaluation data and model calls local unless you explicitly request `execution_mode="cloud"` (or `ExecutionOptions(execution_mode="cloud")`).
- Prefer the grouped option classes (`EvaluationOptions`, `InjectionOptions`, `ExecutionOptions`, `MockModeOptions`) when you need to adjust several related knobs. Import them from `traigent.api.decorators` and pass either instances or plain dicts.
- `parallel_config=ParallelConfig(...)` remains the primary way to control concurrency. Set global defaults via `traigent.configure(parallel_config=...)`, override them in the decorator, and fine-tune per `.optimize()` call. Later scopes override earlier ones field-by-field.
- `ParallelConfig` lives in `traigent.config.parallel`. You can pass either an instance or a simple `dict` with the same keys.
- Setting `privacy_enabled=True` while running in `"cloud"` mode does not redact prompts; the flag is honoured only in local/edge execution.
- `config_param` is required whenever you choose `injection_mode="parameter"`; forgetting it leaves your function without injected configs.
- Provide plain lists for quick starts; TraiGent infers orientations (maximize for accuracy-like metrics, minimize for cost/latency) and assigns equal weights. Use an `ObjectiveSchema` when you need explicit control over orientations, weights, or metric metadata.
- Removed decorator kwargs `auto_optimize`, `trigger`, `batch_size`, and `parallel_trials`. Use the grouped options or `parallel_config` instead.

#### Parallel configuration precedence

1. **Global default**: `traigent.configure(parallel_config=...)` defines the baseline.
2. **Decorator override**: `@traigent.optimize(parallel_config=...)` customises a single optimized function.
3. **Runtime override**: `await answer.optimize(parallel_config=...)` applies to one optimization call.

Later scopes override earlier ones field-by-field. The runtime resolution logs the active values and warns when concurrency is likely to trigger provider throttling or when async functions contain blocking calls.

**Global Settings That Interact With These Parameters**

| Setting | Configuration API | Description | Notes |
| --- | --- | --- | --- |
| `parallel_workers` | `traigent.configure(parallel_workers=...)` | Provides the default worker pool size when neither the decorator nor `.optimize()` specifies a `parallel_config.thread_workers`. | Internally used to set `LocalEvaluator.max_workers`; defaults to `1` if unset. Large values can overwhelm CPU-bound local functions. |

**Returns:** OptimizedFunction wrapper with optimization capabilities

**Example:**
```python
@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4"],
        "temperature": (0.0, 1.0)
    }
)
def my_agent(query: str) -> str:
    return process_query(query)
```

## OptimizedFunction Methods

When you decorate a function with `@traigent.optimize()`, it returns an OptimizedFunction object with these methods:

### Optimization Control

#### `.optimize()`
Run the optimization process.

```python
async def optimize(
    algorithm: str | None = None,
    max_trials: int | None = None,
    timeout: float | None = None,
    save_to: str | None = None,
    custom_evaluator: Callable | None = None,
    callbacks: list | None = None,
    configuration_space: dict[str, Any] | None = None,
    **algorithm_kwargs: Any,
) -> OptimizationResult
```

**Arguments**

| Parameter | Type | Default | Description | Notes |
| --- | --- | --- | --- | --- |
| `algorithm` | `str \| None` | Decorator default | Optimizer to use (`"grid"`, `"random"`, `"optuna"` variants). | Falls back to the decorator’s configured algorithm. |
| `max_trials` | `int \| None` | Decorator default | Maximum number of trials to execute. | `None` means unlimited; stop conditions may still end the run. |
| `timeout` | `float \| None` | Decorator default | Wall-clock budget in seconds. | Applies across the entire optimization loop. |
| `save_to` | `str \| None` | `None` | Optional path to persist results after completion. | Uses `.save_optimization_results()` under the hood. |
| `custom_evaluator` | `Callable \| None` | `None` | Overrides the evaluator for this run only. | Same contract as the decorator-level parameter. |
| `callbacks` | `list \| None` | `None` | Sequence of callback objects for progress reporting. | Must implement the callback protocol defined in `traigent.utils.callbacks`. |
| `configuration_space` | `dict[str, Any] \| None` | Decorator default | Overrides the decorator-defined search space. | Validated with the same rules as the decorator argument. |
| `**algorithm_kwargs` | `Any` | – | Extra tuning knobs for the orchestrator/evaluator. | See recognised keys below. |

**Recognised `algorithm_kwargs`**

| Key | Description |
| --- | --- |
| `parallel_config` | Provides a one-shot concurrency override (same structure as the decorator/global setting). |
| `max_examples` | Caps the number of dataset examples processed per trial. |
| `cache_policy` | One of `"allow_repeats"` (default) or other evaluator cache policies. |
| `budget_limit` / `budget_metric` / `budget_include_pruned` | Configure budget-based early stopping. |
| `plateau_window` / `plateau_epsilon` | Configure plateau detection stop conditions. |

Unknown keys are forwarded to the optimizer and may raise errors when unsupported.

**Example:**
```python
# Run optimization
result = await my_agent.optimize(algorithm="grid", max_trials=50)

# Save results
await my_agent.save_optimization_results("results.json")
```

### Configuration Management

#### `.get_best_config()`
**Get the best configuration found**

```python
def get_best_config() -> dict[str, Any] | None
```

#### `.current_config`
**Get/set current configuration**

```python
@property
def current_config() -> dict[str, Any]
```

#### `.set_config()`
**Set current configuration**

```python
def set_config(config: dict[str, Any]) -> None
```

#### `.apply_best_config()`
**Apply best configuration from optimization**

```python
def apply_best_config(results: OptimizationResult | None = None) -> bool
```

### Results & Analysis

#### `.get_optimization_results()`
**Get complete optimization results**

```python
def get_optimization_results() -> OptimizationResult | None
```

#### `.get_optimization_history()`
**Get history of all optimization runs**

```python
def get_optimization_history() -> list[OptimizationResult]
```

#### `.is_optimization_complete()`
**Check if optimization has been completed**

```python
def is_optimization_complete() -> bool
```

### Persistence

#### `.save_optimization_results()`
**Save optimization results to file**

```python
def save_optimization_results(path: str) -> None
```

#### `.load_optimization_results()`
**Load optimization results from file**

```python
def load_optimization_results(path: str) -> None
```

#### `.reset_optimization()`
**Clear optimization state**

```python
def reset_optimization() -> None
```

### Function Execution

#### `.run()`
**Execute the function with current configuration**

```python
def run(*args: Any, **kwargs: Any) -> Any
```

## Global Configuration Functions

### `traigent.configure()`

```python
def configure(
    default_storage_backend: str = None,
    parallel_workers: int = None,
    cache_policy: str = None,
    logging_level: str = None,
    api_keys: dict[str, str] = None,
    parallel_config: ParallelConfig | dict | None = None,
    objectives: Sequence[str] | ObjectiveSchema | None = None,
    feature_flags: dict[str, Any] = None,
) -> bool
```

- `objectives`: Optional global default applied when decorators omit the parameter. Accepts a list of objective names or an `ObjectiveSchema`.

### `traigent.get_version_info()`

```python
def get_version_info() -> dict[str, Any]
```

### `traigent.initialize()`

```python
def initialize() -> bool
```

### `traigent.override_config()`

```python
def override_config(
    objectives: Sequence[str] | ObjectiveSchema | None = None,
    configuration_space: dict[str, Any] = None,
    **kwargs: Any
) -> dict[str, Any]
```

- Supply an `ObjectiveSchema` here if you need explicit orientations or weights for a single optimization run.

### `traigent.set_strategy()`

```python
def set_strategy(
    algorithm: str = "bayesian",
    **kwargs: Any
) -> StrategyConfig
```

### `traigent.get_available_strategies()`

```python
def get_available_strategies() -> dict[str, Any]
```

### `traigent.get_optimization_insights()`

```python
def get_optimization_insights(
    function_name: str = None,
    **kwargs: Any
) -> dict[str, Any]
```

## Data Types

### OptimizationResult

```python
@dataclass
class OptimizationResult:
    trials: list[TrialResult]
    best_config: dict[str, Any] | None
    best_score: float | None
    optimization_id: str
    duration: float
    convergence_info: dict[str, Any]
    metadata: dict[str, Any]
```

### TrialResult

```python
@dataclass
class TrialResult:
    trial_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    status: TrialStatus
    duration: float
    timestamp: datetime
    error_message: str | None = None
```

### StrategyConfig

```python
@dataclass
class StrategyConfig:
    algorithm: str
    parameters: dict[str, Any]
```

## Utility Classes

### Constraints
- `temperature_constraint(min_val, max_val)`
- `model_cost_constraint(max_cost)`
- `max_tokens_constraint(max_tokens)`
- `ConstraintManager`

### Callbacks
- `ProgressBarCallback`
- `LoggingCallback`
- `StatisticsCallback`
- `get_default_callbacks()`
- `get_verbose_callbacks()`

### Analysis
- `ParetoFrontCalculator`
- `MultiObjectiveMetrics`
- `ParameterImportanceAnalyzer`
- `PlotGenerator`
- `create_quick_plot()`

### Validation
- `OptimizationValidator`
- `ValidationResult`
- `validate_and_suggest()`

### Persistence
- `PersistenceManager`

### Retry
- `RetryConfig`
- `retry()`

## Configuration Types

### TraigentConfig
Configuration class for TraiGent settings

### InjectionMode
- `CONTEXT`: Use configuration context (default)
- `PARAMETER`: Pass as function parameter
- `ATTRIBUTE`: Store configuration on a function attribute ("decorator" alias is deprecated)
- `SEAMLESS`: AST-based override of simple variable assignments

## Examples

### Basic Usage

```python
import traigent

@traigent.optimize(
    eval_dataset="qa_dataset.jsonl",
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4"],
        "temperature": [0.1, 0.3, 0.7]
    }
)
def answer_question(question: str) -> str:
    # Your existing code
    return llm.complete(question)

# Run optimization
results = await answer_question.optimize()

# Get best configuration
best_config = answer_question.get_best_config()
```

### Multi-Objective Optimization

```python
@traigent.optimize(
    eval_dataset="support_tickets.jsonl",
    objectives=["accuracy", "cost", "latency"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4", "claude-3-haiku"],
        "temperature": (0.0, 1.0),
        "max_tokens": [100, 500, 1000]
    }
)
def process_ticket(ticket: str) -> str:
    return support_chain.process(ticket)
```

### Custom Evaluation

```python
def custom_eval(func, config, example):
    # Custom evaluation logic
    result = func(example.input_data)
    accuracy = compute_accuracy(result, example.expected_output)
    return ExampleResult(
        example_id=example.example_id,
        metrics={"accuracy": accuracy, "custom_metric": compute_custom(result)}
    )

@traigent.optimize(
    eval_dataset="dataset.jsonl",
    custom_evaluator=custom_eval,
    objectives=["accuracy", "custom_metric"]
)
def my_function(input_text: str) -> str:
    return process(input_text)
```

---

This documentation reflects TraiGent SDK v1.0.0.
