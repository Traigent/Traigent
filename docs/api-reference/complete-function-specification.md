# Traigent SDK API Reference

Authoritative reference for Traigent SDK **v0.10.0 (Beta)**.

## Quick Navigation

- **[Decorator Reference](./decorator-reference.md)** - Detailed `@traigent.optimize()` decorator documentation with all parameters and examples
- **[Interactive Optimizer](./interactive_optimizer.md)** - Hybrid optimization API surface
- **[Telemetry Documentation](./telemetry.md)** - What data is collected, retention policies, and how to opt-out
- **[Thread Pool Examples](./thread-pool-examples.md)** - Using thread pools with proper context propagation
- **[Complete API Specification](#core-decorator)** - Full API reference (this document)

## Core Decorator

### `@traigent.optimize()`

**Main decorator for making functions optimizable**

```python
def optimize(
    *,
    objectives: list[str] | ObjectiveSchema | None = None,
    configuration_space: dict[str, Any] | ConfigSpace | None = None,
    default_config: dict[str, Any] | None = None,
    constraints: list[Constraint | Callable[..., Any]] | None = None,
    # TVL integration
    tvl_spec: str | Path | None = None,
    tvl_environment: str | None = None,
    tvl: TVLOptions | dict[str, Any] | None = None,
    # Grouped options (preferred)
    evaluation: EvaluationOptions | dict[str, Any] | None = None,
    injection: InjectionOptions | dict[str, Any] | None = None,
    execution: ExecutionOptions | dict[str, Any] | None = None,
    mock: MockModeOptions | dict[str, Any] | None = None,
    # Legacy compatibility
    legacy: LegacyOptimizeArgs | dict[str, Any] | None = None,
    **runtime_overrides: Any,
) -> OptimizedFunction
```

> **New in 0.8.0** – The decorator now uses keyword-only arguments with grouped option bundles. Legacy arguments are supported via the `legacy` parameter or directly in `**runtime_overrides`. Conflicting values between a bundle and a direct keyword raise `TypeError`.

**Core Parameters**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `objectives` | `list[str] \| ObjectiveSchema \| None` | `["accuracy"]` | Target metrics to optimize. Lists are converted to `ObjectiveSchema` with sensible defaults. Provide an `ObjectiveSchema` for explicit weights/orientations. |
| `configuration_space` | `dict[str, Any] \| ConfigSpace \| None` | `None` | Search space describing tunable parameters. Lists denote discrete choices; `(min, max)` tuples denote ranges. `Range`, `IntRange`, `LogRange`, `Choices`, and `ConfigSpace` are supported directly. Required for optimization. |
| `default_config` | `dict[str, Any] \| None` | `None` | Baseline configuration applied before the first trial. Missing keys remain unset unless you provide defaults via `default_config` or parameter defaults (for example `Range(..., default=...)`). |
| `constraints` | `list[Constraint \| Callable[..., Any]] \| None` | `None` | Hard constraints evaluated before running a trial. Accepts SE-friendly `Constraint` objects and/or callables that take `(config, metrics=None)` and return `True/False`. |

**SE-friendly tuned variables (first-class)**

Traigent treats the SE-friendly tuned variable objects as first-class citizens.
You can pass them directly in `configuration_space`, or supply a `ConfigSpace`
object with structured constraints.

```python
from traigent import Choices, ConfigSpace, Range, implies

temp = Range(0.0, 2.0, name="temperature")
model = Choices(["gpt-4", "gpt-3.5"], name="model")
space = ConfigSpace(
    tvars={"temperature": temp, "model": model},
    constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
)

@traigent.optimize(configuration_space=space)
def my_agent(question: str) -> str:
    return answer(question)
```

**TVL Integration** (Tuned Variable Language)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `tvl_spec` | `str \| Path \| None` | `None` | Path to a TVL specification file. When provided, becomes the authoritative source for configuration space, objectives, and constraints. |
| `tvl_environment` | `str \| None` | `None` | Named environment overlay from the TVL spec (e.g., "development", "production"). |
| `tvl` | `TVLOptions \| dict \| None` | `None` | Structured TVL options controlling how the spec is applied (`apply_evaluation_set`, `apply_configuration_space`, `apply_objectives`, `apply_constraints`, `apply_budget`, `registry_resolver`). |

**Grouped Option Bundles** (Preferred for new code)

| Parameter | Type | Description |
| --- | --- | --- |
| `evaluation` | `EvaluationOptions \| dict \| None` | Bundle for `eval_dataset`, `custom_evaluator`, `scoring_function`, and `metric_functions`. |
| `injection` | `InjectionOptions \| dict \| None` | Bundle for `injection_mode`, `config_param`, `auto_override_frameworks`, and `framework_targets`. |
| `execution` | `ExecutionOptions \| dict \| None` | Bundle for execution settings including `execution_mode`, `local_storage_path`, `parallel_config`, `privacy_enabled`, and `max_total_examples`. |
| `mock` | `MockModeOptions \| dict \| None` | Bundle for mock-mode settings: `enabled`, `override_evaluator`, `base_accuracy`, `variance`. |

**ExecutionOptions Fields** (open-source builds run in `edge_analytics` only; other modes are roadmap-compatible but not currently provisioned)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `execution_mode` | `str` | `"edge_analytics"` | `"edge_analytics"` is supported today; `"cloud"`, `"hybrid"`, `"standard"` are reserved for managed backends. `"privacy"` is accepted as a legacy alias for `"hybrid"` and sets `privacy_enabled=True`. |
| `local_storage_path` | `str \| None` | `None` | Custom directory for persisted results. Falls back to `TRAIGENT_RESULTS_FOLDER` or `~/.traigent/`. |
| `minimal_logging` | `bool` | `True` | Suppresses verbose logs in privacy-sensitive modes. |
| `parallel_config` | `ParallelConfig \| dict \| None` | `None` | Unified concurrency configuration. |
| `privacy_enabled` | `bool \| None` | `None` | Redacts prompts/responses from telemetry and logs. |
| `max_total_examples` | `int \| None` | `None` | Global sample budget across all trials (budget guardrail). |
| `samples_include_pruned` | `bool` | `True` | Whether pruned trials count toward the sample budget. |

**Legacy Compatibility**

| Parameter | Type | Description |
| --- | --- | --- |
| `legacy` | `LegacyOptimizeArgs \| dict \| None` | Adapter for the previous decorator signature. Pass a dict with historic keyword arguments; values merge with explicit parameters. |
| `**runtime_overrides` | `Any` | Legacy keyword compatibility and inline tuned-variable definitions. Unknown keys raise `TypeError`; use `.optimize()` for run controls like `algorithm`, `timeout`, `cache_policy`, `cost_limit`, and stop conditions. |

**Usage Notes**

- The default execution mode is `"edge_analytics"` and is the only supported mode in the open-source build. Other modes are present for forward compatibility but require a managed backend.
- Prefer the grouped option classes (`EvaluationOptions`, `InjectionOptions`, `ExecutionOptions`, `MockModeOptions`) when you need to adjust several related knobs. Import them from `traigent.api.decorators` and pass either instances or plain dicts.
- `parallel_config=ParallelConfig(...)` remains the primary way to control concurrency. Set global defaults via `traigent.configure(parallel_config=...)`, override them in the decorator, and fine-tune per `.optimize()` call. Later scopes override earlier ones field-by-field.
- `ParallelConfig` lives in `traigent.config.parallel`. You can pass either an instance or a simple `dict` with the same keys.
- `privacy_enabled=True` applies in local/edge contexts; cloud/hybrid execution is not available yet in OSS builds.
- `config_param` is required whenever you choose `injection_mode="parameter"`; forgetting it leaves your function without injected configs.
- Provide plain lists for quick starts; Traigent infers orientations (maximize for accuracy-like metrics, minimize for cost/latency) and assigns equal weights. Use an `ObjectiveSchema` when you need explicit control over orientations, weights, or metric metadata.
- Inline tuned-variable definitions accept `Range`, `IntRange`, `LogRange`, `Choices`, or numeric `(low, high)` tuples. Inline lists are not recognized; use `Choices([...])` instead.
- If you pass a `ConfigSpace` with constraints, omit `constraints=`. Supplying both raises `TypeError`.

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
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4"],
        "temperature": (0.0, 1.0)
    },
    evaluation={"eval_dataset": "evals.jsonl"},  # Grouped option bundle
    execution={"execution_mode": "edge_analytics"},
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
    callbacks: list[Callable] | None = None,
    configuration_space: dict[str, Any] | None = None,
    objectives: ObjectiveSchema | Sequence[str] | None = None,
    tvl_spec: str | Path | None = None,
    tvl_environment: str | None = None,
    tvl: TVLOptions | dict[str, Any] | None = None,
    **algorithm_kwargs: Any,
) -> OptimizationResult
```

**Arguments**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `algorithm` | `str \| None` | Decorator default | Optimizer to use: `"grid"`, `"random"`, `"bayesian"`, `"optuna"`. |
| `max_trials` | `int \| None` | Decorator default | Maximum number of trials to execute. `None` means unlimited. |
| `timeout` | `float \| None` | Decorator default | Wall-clock budget in seconds. |
| `save_to` | `str \| None` | `None` | Optional path to persist results after completion. |
| `custom_evaluator` | `Callable \| None` | `None` | Overrides the evaluator for this run only. |
| `callbacks` | `list[Callable] \| None` | `None` | Callback objects for progress reporting. |
| `configuration_space` | `dict[str, Any] \| None` | Decorator default | Overrides the decorator-defined search space. |
| `objectives` | `ObjectiveSchema \| Sequence[str] \| None` | Decorator default | Override objectives for this run. |
| `tvl_spec` | `str \| Path \| None` | `None` | Load TVL spec at runtime. |
| `tvl_environment` | `str \| None` | `None` | Environment overlay from the TVL spec. |
| `tvl` | `TVLOptions \| dict \| None` | `None` | Structured TVL options for runtime overrides. |
| `**algorithm_kwargs` | `Any` | – | Extra tuning knobs. See recognised keys below. |

**Recognised `algorithm_kwargs`**

| Key | Description |
| --- | --- |
| `parallel_config` | One-shot concurrency override (same structure as decorator/global setting). |
| `max_examples` | Caps the number of dataset examples processed per trial. |
| `max_total_examples` | Global sample budget across all trials. |
| `cache_policy` | One of `"allow_repeats"` (default) or other cache policies. |
| `cost_limit` | Maximum USD spending for this run. |
| `cost_approved` | Skip cost approval prompt. |
| `budget_limit` / `budget_metric` / `budget_include_pruned` | Configure budget-based early stopping. |
| `plateau_window` / `plateau_epsilon` | Configure plateau detection stop conditions. |
| `parameter_order` | Grid search only: mapping of parameter names to numeric priorities. Lower values vary slowest; higher values vary fastest. |
| `order` | Alias for `parameter_order` (grid search only). |

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

### `traigent.get_config()`

**Get the active configuration in any lifecycle context**

```python
def get_config() -> dict[str, Any]
```

This unified accessor works both during optimization trials and after applying the best configuration to your function. It is the recommended way to access configuration inside optimized functions.

**Returns:** Dictionary with the currently active configuration.

**Raises:** `OptimizationStateError` if no configuration is available (e.g., called outside an optimized function without `apply_best_config()`).

**Example:**
```python
@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4"], "temperature": (0.0, 1.0)}
)
def my_agent(query: str) -> str:
    config = traigent.get_config()  # Works during optimization AND after apply_best_config()
    return call_llm(query, model=config["model"], temperature=config["temperature"])
```

> **Note:** `traigent.get_trial_config()` is deprecated. Use `traigent.get_config()` instead, which works in all contexts.

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
    best_config: dict[str, Any]
    best_score: float
    optimization_id: str
    duration: float
    convergence_info: dict[str, Any]
    status: OptimizationStatus
    objectives: list[str]
    algorithm: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    # Cost and token tracking
    total_cost: float | None = None
    total_tokens: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
```

**Properties:**
- `total_trials`: Total number of trials executed
- `successful_trials`: Number of trials that completed successfully
- `success_rate`: Ratio of successful trials to total trials

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
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Properties:**
- `is_successful`: Whether the trial completed successfully (`status == TrialStatus.COMPLETED`)

**Methods:**
- `get_metric(name: str, default: float | None = None) -> float | None`: Get a specific metric value

### TrialStatus

```python
class TrialStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"
```

### OptimizationStatus

```python
class OptimizationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

## Utility Classes

### Constraints
- `temperature_constraint(min_temp, max_temp)`
- `model_cost_constraint(max_cost_per_1k_tokens)`
- `max_tokens_constraint(min_tokens, max_tokens)`
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
Configuration class for Traigent settings

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
    evaluation={"eval_dataset": "qa_dataset.jsonl"},
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
    evaluation={"eval_dataset": "support_tickets.jsonl"},
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

### Constraints (Utility Helper)

```python
from traigent import temperature_constraint

temp_guard = temperature_constraint(0.1, 0.7)

@traigent.optimize(
    evaluation={"eval_dataset": "support_tickets.jsonl"},
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4"],
        "temperature": (0.0, 1.0),
        "max_tokens": [100, 500, 1000],
    },
    constraints=[temp_guard.validate],
)
def process_ticket_with_constraints(ticket: str) -> str:
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
    evaluation={"eval_dataset": "dataset.jsonl", "custom_evaluator": custom_eval},
    objectives=["accuracy", "custom_metric"]
)
def my_function(input_text: str) -> str:
    return process(input_text)
```

---

This documentation reflects Traigent SDK v0.10.0.
