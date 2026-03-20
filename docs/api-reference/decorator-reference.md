# Traigent Decorator Reference

This document provides a comprehensive reference for the `@traigent.optimize()` decorator and `@optimize()` (when imported directly).

## Overview

The `@traigent.optimize()` decorator is the main entry point for Traigent SDK. It enables zero-code-change optimization for any function containing LLM invocations by automatically detecting and optimizing LLM parameters.

## Import Signatures

```python
# Primary import pattern (recommended)
import traigent

@traigent.optimize(...)
def my_function(...):
    ...

# Alternative: direct import
from traigent.api.decorators import optimize

@optimize(...)
def my_function(...):
    ...
```

Both forms are functionally identical. The `traigent.optimize` form is recommended for clarity.

## Complete Decorator Signature

```python
def optimize(
    *,
    # Core optimization parameters
    objectives: list[str] | ObjectiveSchema | None = None,
    configuration_space: dict[str, Any] | ConfigSpace | None = None,
    default_config: dict[str, Any] | None = None,
    constraints: list[Constraint | BoolExpr | Callable[..., Any]] | None = None,

    # TVL integration
    tvl_spec: str | Path | None = None,
    tvl_environment: str | None = None,
    tvl: TVLOptions | dict[str, Any] | None = None,

    # Grouped option bundles (preferred)
    evaluation: EvaluationOptions | dict[str, Any] | None = None,
    injection: InjectionOptions | dict[str, Any] | None = None,
    execution: ExecutionOptions | dict[str, Any] | None = None,
    mock: MockModeOptions | dict[str, Any] | None = None,

    # Legacy compatibility
    legacy: LegacyOptimizeArgs | dict[str, Any] | None = None,

    # Runtime overrides
    **runtime_overrides: Any,
) -> Callable[[Callable[..., Any]], Any]
```

## Parameter Groups

### Core Optimization Parameters

#### `objectives`

- **Type**: `list[str] | ObjectiveSchema | None`
- **Default**: `["accuracy"]`
- **Description**: Target metrics to optimize

> **Understanding Accuracy**: The meaning of "accuracy" depends on your use case:
> - **Classification**: Exact match percentage (e.g., sentiment classification)
> - **Semantic Similarity**: Embedding-based comparison (default, requires `OPENAI_API_KEY`)
> - **Custom Metrics**: ROUGE, BLEU, or business-specific calculations
>
> See the [Evaluation Guide](../user-guide/evaluation_guide.md#understanding-accuracy) for detailed explanations and examples.

**String List Form** (Simple):

```python
@traigent.optimize(
    objectives=["accuracy", "cost", "latency"],
    ...
)
```

Traigent automatically infers:

- **Orientations**: Maximize for accuracy-like metrics, minimize for cost/latency
- **Weights**: Equal weights for all objectives

**ObjectiveSchema Form** (Advanced):

```python
from traigent.core.objectives import ObjectiveSchema, ObjectiveDefinition

@traigent.optimize(
    objectives=ObjectiveSchema(
        definitions=[
            ObjectiveDefinition(name="accuracy", weight=2.0, orientation="maximize"),
            ObjectiveDefinition(name="cost", weight=1.0, orientation="minimize"),
            ObjectiveDefinition(name="latency", weight=0.5, orientation="minimize"),
        ]
    ),
    ...
)
```

Explicit control over:

- **Weights**: Relative importance of each objective
- **Orientations**: "maximize", "minimize", or "band" (TVL 0.9)
- **Metadata**: Additional objective configuration

**Banded Objectives** (TVL 0.9):

For objectives where you want to stay within a target range rather than maximize/minimize:

```python
from traigent.core.objectives import ObjectiveSchema, ObjectiveDefinition
from traigent.tvl.models import BandTarget

@traigent.optimize(
    objectives=ObjectiveSchema(
        definitions=[
            ObjectiveDefinition(name="accuracy", weight=2.0, orientation="maximize"),
            # Banded objective: target response length between 100-200 tokens
            ObjectiveDefinition(
                name="response_length",
                orientation="band",
                band=BandTarget(low=100, high=200),
                band_test="TOST",    # Two One-Sided Tests for equivalence
                band_alpha=0.05,     # Significance level
                weight=1.0,
            ),
            # Alternative: center ± tolerance
            ObjectiveDefinition(
                name="cost",
                orientation="band",
                band=BandTarget(center=0.01, tol=0.005),  # Band is [0.005, 0.015]
                band_test="TOST",
                band_alpha=0.05,
                weight=1.0,
            ),
        ]
    ),
    ...
)
```

Banded objectives use TOST (Two One-Sided Tests) to statistically verify that the metric falls within the target band.

> **Current Status**: Banded objectives are fully parsed and available in `ObjectiveDefinition`. During optimization, they are currently treated as minimize objectives. For full TOST-based statistical testing, use `PromotionGate.evaluate()` with collected sample data after optimization.

#### `configuration_space`

- **Type**: `dict[str, Any] | ConfigSpace | None`
- **Default**: `None`
- **Description**: Search space describing tunable parameters

> ⚠️ **Deprecation Notice (TVL 0.9)**: When using TVL spec files, the `configuration_space` key inside the spec is deprecated in favor of `tvars` (typed variables). The decorator parameter itself remains valid. Define your search space using `tvars` for full TVL 0.9 support including type safety, units, and registry domains.

**Discrete Choices** (List):

```python
configuration_space={
    "model": ["gpt-4o-mini", "gpt-4", "claude-3-haiku"],
    "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
}
```

**Continuous Ranges** (Tuple):

```python
configuration_space={
    "temperature": (0.0, 1.0),  # min, max
    "top_p": (0.0, 1.0),
}
```

**Mixed Configuration**:

```python
configuration_space={
    "model": ["gpt-4o-mini", "gpt-4"],
    "temperature": (0.0, 1.0),
    "max_tokens": [100, 500, 1000, 2000],
}
```

**SE-friendly tuned variables (first-class)**:

```python
from traigent import Choices, Range

@traigent.optimize(
    configuration_space={
        "model": Choices(["gpt-4", "gpt-3.5"]),
        "temperature": Range(0.0, 2.0),
    },
)
def my_agent(prompt: str) -> str:
    return run(prompt)
```

You can also pass a `ConfigSpace` object when you want to bundle `tvars` and structured constraints together.

#### `default_config`

- **Type**: `dict[str, Any] | None`
- **Default**: `None`
- **Description**: Baseline configuration for first trial

```python
@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4"], "temperature": (0.0, 1.0)},
    default_config={"model": "gpt-4o-mini", "temperature": 0.7},
    ...
)
```

#### `constraints`

- **Type**: `list[Constraint | Callable[..., Any]] | None`
- **Default**: `None`
- **Description**: Validation functions that return True/False

Constraint objects and callables can be mixed; Traigent normalizes them to a common callable interface internally.

**Configuration-only Constraints**:

```python
def max_tokens_constraint(config):
    if config["model"] == "gpt-4" and config["max_tokens"] > 2000:
        return False
    return True

@traigent.optimize(
    constraints=[max_tokens_constraint],
    ...
)
```

**Metrics-based Constraints**:

```python
def cost_constraint(config, metrics=None):
    if metrics and metrics.get("cost", 0) > 0.10:
        return False
    return True

@traigent.optimize(
    constraints=[cost_constraint],
    ...
)
```

**Structured Constraints (SE-friendly)**:

```python
from traigent import Choices, Range, implies

model = Choices(["gpt-4", "gpt-3.5"], name="model")
temp = Range(0.0, 2.0, name="temperature")

@traigent.optimize(
    configuration_space={"model": model, "temperature": temp},
    constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
)
def my_agent(prompt: str) -> str:
    return run(prompt)
```

### TVL Integration

Traigent supports the TVL (Tuned Variables Language) 0.9 specification for declarative optimization configuration. TVL specs provide:

- **Typed Variables (tvars)**: Parameters with explicit types (`bool`, `int`, `float`, `enum[str]`, `tuple[...]`). Integer ranges preserve `int` type for proper sampling.
- **Structural Constraints**: Boolean formulas over tvars (compiled to DNF)
- **Derived Constraints**: Linear arithmetic over environment symbols
- **Banded Objectives**: TOST equivalence testing with target bands (parsed; use `PromotionGate` for evaluation)
- **Promotion Policy**: Epsilon-Pareto dominance with configurable error rates (use `PromotionGate.from_spec_artifact()` for statistical decisions)
- **Exploration Settings**: Strategy (supports `{type: ...}` dict format), convergence criteria, parallelism, and budgets

When a TVL spec is loaded, the `TVLSpecArtifact` provides access to all parsed sections including `tvars`, `constraints` (compiled structural constraints as callables), `derived_constraints`, `promotion_policy`, `convergence`, `exploration_budgets`, and `exploration_parallelism`.

**Runtime Wiring (TVL 0.9 `exploration` section)**:

- `exploration.budgets.max_trials` → `max_trials`
- `exploration.budgets.max_spend_usd` → `cost_limit`
- `exploration.budgets.max_wallclock_s` → `timeout`
- `exploration.strategy.type` → `algorithm`
- `exploration.parallelism.max_parallel_trials` → `parallel_config.trial_concurrency`

> **Note**: If both `exploration` (TVL 0.9) and `optimization` (legacy) sections are present in a spec, an error is raised. Use only one format.

#### `tvl_spec`

- **Type**: `str | Path | None`
- **Default**: `None`
- **Description**: Path to TVL specification file

```python
@traigent.optimize(
    tvl_spec="specs/my_optimization.tvl",
    tvl_environment="production",
    ...
)
```

#### `tvl_environment`

- **Type**: `str | None`
- **Default**: `None`
- **Description**: Named environment overlay from the TVL spec

#### `tvl`

- **Type**: `TVLOptions | dict[str, Any] | None`
- **Default**: `None`
- **Description**: Structured TVL options controlling how the spec is applied (configuration space, objectives, constraints, budgets) and how to resolve registry domains

```python
from traigent.tvl.options import TVLOptions

@traigent.optimize(
    tvl=TVLOptions(
        spec_path="specs/my_optimization.tvl",
        environment="production",
        validate_constraints=True,        # Compile and validate structural constraints
        registry_resolver=my_resolver,    # Required if spec uses registry domains
        apply_evaluation_set=True,        # Wire evaluation_set.uri to evaluation settings
        apply_configuration_space=True,
        apply_objectives=True,
        apply_constraints=True,
        apply_budget=True,
    ),
    ...
)
```

**TVLOptions Fields**:

- `spec_path`: Path to the TVL specification file (required)
- `environment`: Named environment overlay from the spec
- `validate_constraints`: Whether to compile and validate structural constraints (default: `True`)
- `registry_resolver`: Resolver for registry domains (required if spec uses `registry://` domains)
- `apply_evaluation_set`: Wire TVL `evaluation_set.uri` to evaluation settings (default: `True`)
- `apply_configuration_space`: Apply TVL configuration space (default: `True`)
- `apply_objectives`: Apply TVL objectives (default: `True`)
- `apply_constraints`: Apply TVL constraints (default: `True`)
- `apply_budget`: Apply TVL budget settings (default: `True`)

### Grouped Option Bundles

#### `evaluation`

- **Type**: `EvaluationOptions | dict[str, Any] | None`
- **Default**: `None`

```python
from traigent.api.decorators import EvaluationOptions

# As an instance
@traigent.optimize(
    evaluation=EvaluationOptions(
        eval_dataset="qa_test.jsonl",
        custom_evaluator=my_evaluator,
        scoring_function=my_scorer,
    ),
    ...
)

# As a dict
@traigent.optimize(
    evaluation={
        "eval_dataset": "qa_test.jsonl",
        "custom_evaluator": my_evaluator,
    },
    ...
)
```

**EvaluationOptions Fields**:

- `eval_dataset`: Dataset path, list of paths, inline example list, or Dataset instance
- `custom_evaluator`: Custom evaluation function
- `scoring_function`: Custom scoring function
- `metric_functions`: Dict of metric name to evaluator functions

##### Evaluation Methods

Traigent supports multiple evaluation approaches through its parameters:

| Method | Parameter | Use Case |
|--------|-----------|----------|
| Exact Match | Default (no param needed) | Classification, yes/no, categories (binary pass/fail) |
| Graded Scoring | `scoring_function` | Custom graded evaluation (0.0-1.0) - partial matches, fuzzy matching |
| LLM-as-Judge | `custom_evaluator` | Complex quality assessment - nuanced evaluation, code review |
| Multi-Metric / RAGAS | `metric_functions` | Multiple dimensions, semantic similarity (via RAGAS) |

##### LLM-as-Judge Example

For tasks requiring nuanced evaluation (code quality, writing style, factual accuracy), implement an LLM-based evaluator:

```python
def llm_evaluator(output: str, expected: str, **kwargs) -> float:
    """Use an LLM to evaluate output quality."""
    # Your LLM evaluation logic here
    # See 06_custom_evaluator.py for complete implementation
    return 0.8  # Replace: return your calculated score (0.0 to 1.0)

@traigent.optimize(
    custom_evaluator=llm_evaluator,
    # ... other parameters
)
def my_function(prompt: str) -> str:
    ...
```

Complete Example: See [06_custom_evaluator.py](../../walkthrough/real/06_custom_evaluator.py) for a full LLM-as-Judge implementation.

#### `injection`

- **Type**: `InjectionOptions | dict[str, Any] | None`
- **Default**: `None`

```python
from traigent.api.decorators import InjectionOptions

@traigent.optimize(
    injection=InjectionOptions(
        injection_mode="context",  # or "parameter", "attribute", "seamless"
        config_param="config",  # required when injection_mode="parameter"
        auto_override_frameworks=True,
        framework_targets=["OpenAI", "Anthropic"],
    ),
    ...
)
```

**InjectionOptions Fields**:

- `injection_mode`: How to inject config ("context", "parameter", "attribute", "seamless")
- `config_param`: Parameter name when using "parameter" mode
- `auto_override_frameworks`: Auto-detect framework classes
- `framework_targets`: Explicit list of framework classes

#### `execution`

- **Type**: `ExecutionOptions | dict[str, Any] | None`
- **Default**: `None`

```python
from traigent.api.decorators import ExecutionOptions
from traigent.config.parallel import ParallelConfig

@traigent.optimize(
    execution=ExecutionOptions(
        execution_mode="edge_analytics",
        local_storage_path="./my_results",
        minimal_logging=True,
        parallel_config=ParallelConfig(thread_workers=4, trial_concurrency=2),
        privacy_enabled=False,
        max_total_examples=1000,
        samples_include_pruned=True,
    ),
    ...
)
```

**ExecutionOptions Fields**:

- `execution_mode`: "edge_analytics" (only supported mode in OSS)
- `local_storage_path`: Custom storage directory
- `minimal_logging`: Reduce log verbosity
- `parallel_config`: Concurrency configuration
- `privacy_enabled`: Redact sensitive data
- `max_total_examples`: Global sample budget
- `samples_include_pruned`: Count pruned trials in budget

**JavaScript Runtime Fields** (for `runtime="node"`):

- `runtime`: Runtime to execute trials ("python" or "node")
- `js_module`: Path to JS module containing trial function (required for node)
- `js_function`: Exported function name (default: "runTrial")
- `js_timeout`: Trial timeout in seconds (default: 300)
- `js_parallel_workers`: Number of parallel Node.js workers (default: 1)

**JavaScript Execution Example**:

```python
@traigent.optimize(
    execution={
        "runtime": "node",
        "js_module": "./dist/my-agent.js",
        "js_function": "runTrial",
        "js_timeout": 300.0,
        "js_parallel_workers": 4,
    },
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7],
    },
    evaluation={"eval_dataset": "dataset.jsonl"},
    objectives=["accuracy", "cost"],
    max_trials=20,
)
def optimize_js_agent(text: str) -> str:
    # Function body not executed when runtime="node"
    pass
```

See the [JS Bridge Guide](../guides/js-bridge.md) for complete documentation.

#### `mock`

- **Type**: `MockModeOptions | dict[str, Any] | None`
- **Default**: `None`

```python
from traigent.api.decorators import MockModeOptions

@traigent.optimize(
    mock=MockModeOptions(
        enabled=True,
        override_evaluator=True,
        base_accuracy=0.75,
        variance=0.25,
    ),
    ...
)
```

**MockModeOptions Fields**:

- `enabled`: Enable mock mode
- `override_evaluator`: Use mock evaluator
- `base_accuracy`: Base accuracy for mock results
- `variance`: Variance in mock results

### Runtime Overrides

The `**runtime_overrides` parameter accepts additional settings:

**Optimization Algorithm**:

```python
@traigent.optimize(
    algorithm="optuna",  # "grid", "random", "bayesian", "optuna"
    max_trials=50,
    timeout=3600,  # seconds
    ...
)
```

**Cost Controls**:

```python
@traigent.optimize(
    cost_limit=5.00,  # USD
    cost_approved=False,  # Prompt for approval
    ...
)
```

**Budget Controls**:

```python
@traigent.optimize(
    budget_limit=1000,  # Max samples
    budget_metric="samples",
    budget_include_pruned=True,
    ...
)
```

**Stop Conditions**:

```python
@traigent.optimize(
    plateau_window=10,  # trials
    plateau_epsilon=0.01,  # improvement threshold
    ...
)
```

## Complete Examples

### Basic Single-Objective Optimization

```python
import traigent

@traigent.optimize(
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4"],
        "temperature": [0.1, 0.5, 0.9],
    },
    evaluation={"eval_dataset": "qa_test.jsonl"},
)
def answer_question(question: str) -> str:
    llm = OpenAI()  # Parameters auto-optimized
    return llm.complete(question)
```

### Multi-Objective with Constraints

```python
@traigent.optimize(
    objectives=["accuracy", "cost", "latency"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4", "claude-3-haiku"],
        "temperature": (0.0, 1.0),
        "max_tokens": [100, 500, 1000],
    },
    constraints=[
        lambda cfg: cfg["max_tokens"] <= 500 if cfg["model"] == "gpt-4" else True,
        lambda cfg, metrics: metrics.get("cost", 0) <= 0.10 if metrics else True,
    ],
    evaluation={"eval_dataset": "support_tickets.jsonl"},
    execution={
        "execution_mode": "edge_analytics",
        "parallel_config": {"thread_workers": 4},
    },
)
def process_ticket(ticket: str) -> str:
    return support_chain.run(ticket)
```

### Edge Analytics with Privacy

```python
@traigent.optimize(
    objectives=["accuracy", "safety"],
    configuration_space={
        "model": ["gpt-4", "claude-3-sonnet"],
        "temperature": [0.1, 0.3, 0.5],
    },
    evaluation={"eval_dataset": "medical_qa.jsonl"},
    execution={
        "execution_mode": "edge_analytics",
        "local_storage_path": "./medical_optimizations",
        "privacy_enabled": True,
        "minimal_logging": True,
    },
)
def medical_assistant(query: str) -> str:
    return process_medical_query(query)
```

### Using ObjectiveSchema for Weighted Objectives

```python
from traigent.core.objectives import ObjectiveSchema, ObjectiveDefinition

@traigent.optimize(
    objectives=ObjectiveSchema(
        definitions=[
            ObjectiveDefinition(name="accuracy", weight=3.0, orientation="maximize"),
            ObjectiveDefinition(name="cost", weight=1.0, orientation="minimize"),
            ObjectiveDefinition(name="latency", weight=0.5, orientation="minimize"),
        ]
    ),
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4"],
        "temperature": (0.0, 1.0),
    },
    evaluation={"eval_dataset": "dataset.jsonl"},
)
def my_agent(query: str) -> str:
    return process(query)
```

## Related Documentation

- [Complete Function Specification](./complete-function-specification.md) - Full API reference
- [Evaluation Guide](../user-guide/evaluation_guide.md) - Understanding accuracy metrics, evaluation methods, and troubleshooting
- [Injection Modes Guide](../user-guide/injection_modes.md) - Configuration injection patterns
- [Execution Modes Guide](../guides/execution-modes.md) - Execution mode details
- [Thread Pool Examples](./thread-pool-examples.md) - Context propagation with threads
- [Telemetry Documentation](./telemetry.md) - Data collection and privacy
- [Custom Evaluator Example](../../walkthrough/real/06_custom_evaluator.py) - LLM-as-Judge implementation
