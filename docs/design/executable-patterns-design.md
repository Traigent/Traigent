# Design Document: Executable Patterns for LLM Optimization

**Status**: Design Exploration
**Author**: Claude (with user collaboration)
**Date**: 2026-01-25

**Related Document**: [Algorithmic Foundations](./algorithmic-foundations.md) - CS algorithms and data structures informing pattern design

---

## 1. Problem Statement

When engineers build LLM-powered systems, they often encounter repeatable patterns that trade off cost, quality, and latency:

- **Bundling**: Group related tasks (text extraction, entity resolution) in a single LLM call
- **Voting**: Run multiple times, aggregate via majority vote
- **Cascade**: Try cheap model first, escalate to expensive on low confidence
- **Split-Aggregate**: Break complex query into sub-queries, merge results

These patterns are currently implemented ad-hoc by users. They should be:
1. **Reusable** - Standard library of proven patterns
2. **Optimizable** - Optimizer explores pattern selection alongside parameters
3. **Trade-off aware** - Explicit cost/quality/latency characteristics

### Key Question Resolved

> Are these tuned variables or higher-level constructs?

**Answer**: Higher-level constructs (Patterns) that *contain* tuned variables. Patterns carry:
- Semantic meaning (what the pattern does)
- Executable behavior (how it runs)
- Trade-off profiles (expected cost/quality/latency characteristics)
- Pattern-specific configuration spaces

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pattern optimizability | **Optimizable** | Optimizer explores pattern selection as categorical variable |
| Primary goal | **Reusable templates** | Engineers pick from proven pattern catalog |
| Composition model | **Flat only** | Single pattern per task - keeps tractable |
| Pattern nature | **Executable** | Patterns include `execute()` method - reusable implementation |
| Trade-off handling | **Multi-objective** | Cost, latency, quality as explicit Pareto objectives |

---

## 3. Architecture

### 3.1 Abstraction Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                         PATTERNS                             │
│  Higher-level constructs with known trade-offs               │
│  Examples: Voting, Bundling, Cascade, Split-Aggregate        │
│                                                              │
│  Each pattern contains:                                      │
│  • config_space: Pattern-specific tunable variables          │
│  • trade_offs: Multi-objective cost/quality/latency profile  │
│  • execute(): Reusable implementation                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PATTERN PARAMETERS                        │
│  Tuned variables scoped to each pattern                      │
│  Examples:                                                   │
│  • Voting: vote_count, vote_strategy, parallel               │
│  • Bundling: bundle_size, error_handling                     │
│  • Cascade: confidence_threshold, cheap_model                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     TASK VARIABLES                           │
│  Standard tuned variables (always active)                    │
│  Examples: temperature, model, max_tokens, prompt_strategy   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Pattern Class Hierarchy

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from traigent.api.config_space import ConfigSpace
from traigent.api.parameter_ranges import Choices, IntRange, Range


@dataclass
class PatternResult:
    """Result of pattern execution."""
    output: Any
    metadata: dict[str, Any]  # Pattern-specific execution details


class ExecutablePattern(ABC):
    """Base class for executable optimization patterns."""

    name: str
    description: str
    config_space: ConfigSpace
    trade_offs: "TradeOffProfile"

    @abstractmethod
    def execute(
        self,
        task_fn: Callable,
        inputs: Any,
        config: dict,
        context: "ExecutionContext",
    ) -> PatternResult:
        """Execute the task using this pattern's strategy.

        Args:
            task_fn: The user's decorated function
            inputs: Input data for the task
            config: Merged config (pattern params + task params)
            context: Execution context (budget, metrics, etc.)

        Returns:
            PatternResult with output and execution metadata
        """
        pass

    def estimate_cost(self, config: dict) -> float:
        """Pre-execution cost estimate from trade-off profile."""
        return self.trade_offs.cost_model.estimate(config)

    def validate_config(self, config: dict) -> list[str]:
        """Validate config against pattern constraints."""
        return self.config_space.validate(config)


class SinglePattern(ExecutablePattern):
    """Baseline pattern - direct execution without wrapping."""

    name = "single"
    description = "Direct execution without pattern wrapping"
    config_space = ConfigSpace(tvars={})

    def execute(self, task_fn, inputs, config, context):
        result = task_fn(inputs, config)
        return PatternResult(output=result, metadata={})
```

### 3.3 Trade-Off Profile System

Trade-off profiles declare how patterns affect each objective:

```python
from abc import ABC, abstractmethod
from math import sqrt


class ObjectiveModel(ABC):
    """Base class for objective estimation models."""

    @abstractmethod
    def estimate(self, config: dict) -> float:
        """Estimate objective value given config."""
        pass


class LinearCostModel(ObjectiveModel):
    """Cost scales linearly with a config parameter."""

    def __init__(self, multiplier_param: str, base_cost: float = 1.0):
        self.multiplier_param = multiplier_param
        self.base_cost = base_cost

    def estimate(self, config: dict) -> float:
        multiplier = config.get(self.multiplier_param, 1)
        return self.base_cost * multiplier


class VarianceReductionModel(ObjectiveModel):
    """Quality variance reduced by sqrt(N) - ensemble effect."""

    def __init__(self, n_param: str, base_variance: float = 1.0):
        self.n_param = n_param
        self.base_variance = base_variance

    def estimate(self, config: dict) -> float:
        n = config.get(self.n_param, 1)
        return self.base_variance / sqrt(max(n, 1))


class SublinearCostModel(ObjectiveModel):
    """Cost scales sublinearly - batching efficiency."""

    def __init__(self, batch_param: str, efficiency: float = 0.8):
        self.batch_param = batch_param
        self.efficiency = efficiency

    def estimate(self, config: dict) -> float:
        batch_size = config.get(self.batch_param, 1)
        # Cost = 1 / (batch_size ^ efficiency)
        return 1.0 / (batch_size ** self.efficiency)


@dataclass
class TradeOffProfile:
    """Declares how a pattern affects each objective."""

    cost_model: ObjectiveModel
    latency_model: ObjectiveModel
    quality_model: ObjectiveModel

    def estimate_objectives(self, config: dict) -> dict[str, float]:
        """Estimate all objective values before execution."""
        return {
            "cost": self.cost_model.estimate(config),
            "latency": self.latency_model.estimate(config),
            "quality": self.quality_model.estimate(config),
        }
```

### 3.4 Pattern Catalog

```python
class PatternCatalog:
    """Collection of available patterns for optimization."""

    def __init__(self, patterns: list[ExecutablePattern]):
        self.patterns = {p.name: p for p in patterns}

    def as_choices(self) -> Choices:
        """Pattern selection as a tunable categorical variable."""
        return Choices(
            list(self.patterns.keys()),
            name="execution_pattern",
            default="single",
        )

    def as_config_space(self) -> ConfigSpace:
        """Flatten to unified config space with conditional activation."""
        tvars = {"execution_pattern": self.as_choices()}
        constraints = []

        for pattern in self.patterns.values():
            # Prefix pattern-specific vars
            for var_name, var_def in pattern.config_space.tvars.items():
                prefixed_name = f"{pattern.name}__{var_name}"
                tvars[prefixed_name] = var_def

                # Add conditional constraint
                constraints.append(
                    when(execution_pattern.not_equals(pattern.name))
                    .then(disabled(prefixed_name))
                )

        return ConfigSpace(tvars=tvars, constraints=constraints)

    def get_pattern(self, name: str) -> ExecutablePattern:
        """Get pattern by name."""
        return self.patterns[name]

    def execute(
        self,
        pattern_name: str,
        task_fn: Callable,
        inputs: Any,
        config: dict,
        context: ExecutionContext,
    ) -> PatternResult:
        """Execute using the named pattern."""
        pattern = self.get_pattern(pattern_name)
        return pattern.execute(task_fn, inputs, config, context)
```

---

## 4. Standard Pattern Library

### 4.1 Voting Pattern

```python
class VotingPattern(ExecutablePattern):
    """Execute N times, aggregate via voting strategy."""

    name = "voting"
    description = "Run N times, aggregate via majority/weighted vote"

    config_space = ConfigSpace(
        tvars={
            "vote_count": IntRange(3, 9, step=2),  # Odd for tie-breaking
            "vote_strategy": Choices(["majority", "weighted", "unanimous"]),
            "parallel": Choices([True, False], default=True),
        },
        constraints=[
            # Unanimous requires exact agreement - no early exit
            when(vote_strategy.equals("unanimous")).then(
                early_consensus.equals(1.0)
            )
        ],
    )

    trade_offs = TradeOffProfile(
        cost_model=LinearCostModel("vote_count"),
        latency_model=ParallelLatencyModel("vote_count", "parallel"),
        quality_model=VarianceReductionModel("vote_count"),
    )

    def execute(self, task_fn, inputs, config, context):
        vote_count = config["vote_count"]
        parallel = config.get("parallel", True)

        # Execute votes
        if parallel:
            with ThreadPoolExecutor() as pool:
                results = list(pool.map(
                    lambda _: task_fn(inputs, config),
                    range(vote_count)
                ))
        else:
            results = [task_fn(inputs, config) for _ in range(vote_count)]

        # Aggregate
        aggregated = self._aggregate(results, config["vote_strategy"])

        return PatternResult(
            output=aggregated,
            metadata={
                "vote_count": vote_count,
                "individual_results": results,
                "agreement_ratio": self._compute_agreement(results),
            }
        )
```

### 4.2 Bundling Pattern

```python
class BundlingPattern(ExecutablePattern):
    """Bundle N related items in single LLM call."""

    name = "bundling"
    description = "Group related tasks to reduce per-item cost"

    config_space = ConfigSpace(
        tvars={
            "bundle_size": IntRange(2, 10),
            "error_handling": Choices(["fail_all", "partial", "retry"]),
        }
    )

    trade_offs = TradeOffProfile(
        cost_model=SublinearCostModel("bundle_size", efficiency=0.8),
        latency_model=BatchLatencyModel("bundle_size"),
        quality_model=ErrorCorrelationModel("bundle_size"),
    )

    def execute(self, task_fn, inputs, config, context):
        bundle_size = config["bundle_size"]
        error_handling = config["error_handling"]

        # Bundle inputs
        bundles = self._create_bundles(inputs, bundle_size)
        results = []

        for bundle in bundles:
            try:
                bundle_result = task_fn(bundle, config)
                results.extend(self._unbundle(bundle_result))
            except Exception as e:
                if error_handling == "fail_all":
                    raise
                elif error_handling == "retry":
                    # Retry unbundled
                    for item in bundle:
                        results.append(task_fn([item], config)[0])
                else:  # partial
                    results.extend([None] * len(bundle))

        return PatternResult(
            output=results,
            metadata={"bundle_size": bundle_size, "bundle_count": len(bundles)}
        )
```

### 4.3 Cascade Pattern

```python
class CascadePattern(ExecutablePattern):
    """Try cheap model first, escalate on low confidence."""

    name = "cascade"
    description = "Gate expensive calls behind cheap confidence check"

    config_space = ConfigSpace(
        tvars={
            "confidence_threshold": Range(0.7, 0.95),
            "cheap_model": Choices.model(tier="small"),
            "expensive_model": Choices.model(tier="large"),
            "max_escalations": IntRange(1, 3, default=1),
        }
    )

    trade_offs = TradeOffProfile(
        cost_model=CascadeCostModel("confidence_threshold"),
        latency_model=ConditionalLatencyModel("confidence_threshold"),
        quality_model=ConfidenceGatedQualityModel("confidence_threshold"),
    )

    def execute(self, task_fn, inputs, config, context):
        threshold = config["confidence_threshold"]
        cheap = config["cheap_model"]
        expensive = config["expensive_model"]

        # Try cheap first
        cheap_config = {**config, "model": cheap}
        result, confidence = task_fn(inputs, cheap_config, return_confidence=True)

        if confidence >= threshold:
            return PatternResult(
                output=result,
                metadata={"escalated": False, "model_used": cheap}
            )

        # Escalate to expensive
        expensive_config = {**config, "model": expensive}
        result = task_fn(inputs, expensive_config)

        return PatternResult(
            output=result,
            metadata={"escalated": True, "model_used": expensive}
        )
```

---

## 5. Integration with Existing Traigent

### 5.1 Using Existing Multi-Objective Infrastructure

Traigent already has:
- `ObjectiveSchema` with weighted sum, harmonic, Chebyshev aggregation
- `ParetoFrontCalculator` with hypervolume computation
- Cost/latency tracking in `TrialResult.metrics`

Patterns plug directly into this:

```python
# User's decorator
@optimize(
    patterns=STANDARD_PATTERNS,
    objectives=ObjectiveSchema(
        objectives=[
            ObjectiveDefinition("accuracy", orientation="maximize"),
            ObjectiveDefinition("cost", orientation="minimize"),
            ObjectiveDefinition("latency", orientation="minimize"),
        ],
        aggregation_mode=AggregationMode.CHEBYSHEV,  # Balanced trade-offs
    ),
    temperature=Range(0.0, 1.0),
)
def extract_entities(text: str, config: dict) -> list[Entity]:
    ...
```

### 5.2 Optimizer Integration

Trade-off profiles inform the optimizer:

1. **Budget-aware exploration**: Filter candidates by estimated cost
2. **Pareto-guided exploration**: Focus on under-explored frontier regions
3. **Expected hypervolume improvement**: Use trade-offs as priors

```python
class PatternAwareOptimizer(BaseOptimizer):
    def __init__(self, pattern_catalog: PatternCatalog, **kwargs):
        super().__init__(**kwargs)
        self.patterns = pattern_catalog

    def suggest_next_trial(self, history, budget_remaining):
        candidates = self.generate_candidates()

        # Filter by budget
        affordable = [
            c for c in candidates
            if self.patterns.get_pattern(c["execution_pattern"])
                .estimate_cost(c) <= budget_remaining
        ]

        # Select using acquisition function with trade-off priors
        return self.select_by_ehvi(affordable, history)
```

### 5.3 Config Space Flattening

Patterns flatten to standard `ConfigSpace`:

```
Input:
  patterns: [voting, bundling, cascade]
  temperature: Range(0.0, 1.0)

Output ConfigSpace:
  execution_pattern: Choices(["voting", "bundling", "cascade"])
  temperature: Range(0.0, 1.0)
  voting__vote_count: IntRange(3, 9) [when execution_pattern == "voting"]
  voting__vote_strategy: Choices([...]) [when execution_pattern == "voting"]
  bundling__bundle_size: IntRange(2, 10) [when execution_pattern == "bundling"]
  cascade__threshold: Range(0.7, 0.95) [when execution_pattern == "cascade"]
```

---

## 6. SE Pattern Mapping

| Your Pattern | SE Analog | Key Trade-off |
|--------------|-----------|---------------|
| **Voting** | Scatter-Gather, Quorum | Cost × N vs variance / √N |
| **Bundling** | Batch Processing | Cost ÷ N vs error correlation |
| **Cascade** | Chain of Responsibility | Latency vs expected cost |
| **Split-Aggregate** | Splitter + Aggregator | Complexity vs parallelism |
| **Retry** | Retry with Backoff | Cost vs reliability |
| **Fallback** | Circuit Breaker | Quality vs availability |

---

## 7. Future Extensions (Out of Scope)

The following are explicitly deferred:
- **Pattern composition**: Nesting patterns (voting over bundles)
- **Pattern discovery**: Learning new patterns from trial history
- **Adaptive patterns**: Patterns that adjust at runtime

---

## 8. Open Questions

1. **Confidence return**: How do tasks return confidence for cascade pattern?
2. **Pattern-specific metrics**: Should patterns define their own metrics beyond the three objectives?
3. **Pattern compatibility**: Are all patterns compatible with all task types?

---

## 9. External Review Feedback

### 9.1 Codex GPT-5.2 Review (2026-01-25)

**Overall Assessment**: Architecture is sound, with important refinements needed.

#### Abstraction Hierarchy - ✅ Approved with Notes

> "A `Pattern` as a reusable execution strategy + its own `config_space` is a clean unit for search."

**Warning raised**: Pattern selection being optimizable makes `Pattern` both *policy* and *search variable*.

**Recommendation**: Keep a clear interface boundary:
```python
Pattern.execute(config, context) -> outputs + measures
```
Ensure optimizers don't depend on pattern internals.

**On conditional constraints**: "Flatten to ConfigSpace with conditionals works, but you're effectively modeling a hierarchical space. Ensure conditional constraints are expressive enough (mutual exclusivity, required params, derived params) and debuggable when invalid combos arise."

#### Trade-Offs as Estimators - ✅ Approved with Refinements

**Key insight**: Treat trade-offs as **priors with uncertainty** (mean + variance/quantiles), not point estimates.

**Requirements added**:
1. **Calibration path**: Update estimates from observed runs (per model/provider, prompt length, tool use, batching)
2. **Avoid staleness**: Estimates must be refreshable or they become misleading
3. **Keep docstrings**: Documentation is still valuable, but don't rely on docs for pruning/scheduling decisions

#### Missing Considerations Identified

| Gap | Description | Priority |
|-----|-------------|----------|
| **Uncertainty + Learning** | How estimates get trained/updated, cold-start defaults, avoiding exploration bias | High |
| **Runtime Measures Contract** | Standardize what `execute()` returns: quality metrics, cost, latency, failures. Define how partial/failed runs are scored. | High |
| **Control-Plane Concerns** | Caching/deduping, early stopping, fallbacks, resource/budget guardrails (pre-check + post-cost accounting) especially under parallel trials | Medium |
| **Observability** | Per-pattern traces and "why chosen" explanations so users can trust/debug the optimizer | Medium |

---

## 10. Design Refinements (Post-Review)

### 10.1 Uncertainty-Aware Trade-Off Profiles

```python
@dataclass
class UncertaintyAwareEstimate:
    """Estimate with uncertainty bounds."""
    mean: float
    variance: float
    confidence_interval: tuple[float, float]  # 95% CI
    sample_count: int  # How many observations informed this


class CalibratedObjectiveModel(ObjectiveModel):
    """ObjectiveModel with online calibration from observed runs."""

    def __init__(self, prior_model: ObjectiveModel):
        self.prior = prior_model
        self.observations: list[tuple[dict, float]] = []

    def estimate(self, config: dict) -> UncertaintyAwareEstimate:
        prior_estimate = self.prior.estimate(config)
        if not self.observations:
            # Cold start - use prior with high uncertainty
            return UncertaintyAwareEstimate(
                mean=prior_estimate,
                variance=1.0,  # High uncertainty
                confidence_interval=(prior_estimate * 0.5, prior_estimate * 2.0),
                sample_count=0,
            )
        # Bayesian update from observations
        return self._bayesian_update(config, prior_estimate)

    def record_observation(self, config: dict, actual_value: float):
        """Update model with observed value."""
        self.observations.append((config, actual_value))
```

### 10.2 Standardized Runtime Measures Contract

```python
@dataclass
class PatternExecutionResult:
    """Standardized result from pattern execution."""

    # Core outputs
    output: Any
    status: Literal["success", "partial", "failed"]

    # Required measures (always populated)
    measures: PatternMeasures

    # Pattern-specific metadata
    metadata: dict[str, Any]


@dataclass
class PatternMeasures:
    """Standardized measures every pattern must report."""

    # Quality (task-specific, but required)
    quality_score: float | None  # None if failed

    # Cost (always tracked)
    cost_usd: float
    token_count: int
    llm_calls: int

    # Latency
    latency_ms: float

    # Failure info
    error_count: int
    partial_results: int  # Items that succeeded in partial failure
```

### 10.3 Budget Guardrail Integration

```python
class ExecutablePattern(ABC):

    def execute_with_guardrails(
        self,
        task_fn: Callable,
        inputs: Any,
        config: dict,
        context: ExecutionContext,
    ) -> PatternExecutionResult:
        """Execute with pre-check and post-accounting."""

        # Pre-check: Can we afford this?
        estimated_cost = self.estimate_cost(config)
        if not context.budget_guardrail.check_before_trial(estimated_cost):
            raise BudgetExceededError(
                f"Pattern {self.name} estimated cost {estimated_cost} "
                f"exceeds remaining budget {context.budget_guardrail.remaining}"
            )

        # Execute
        result = self.execute(task_fn, inputs, config, context)

        # Post-accounting: Track actual cost
        context.budget_guardrail.add_cost(result.measures.cost_usd)

        # Calibrate trade-off model
        self.trade_offs.cost_model.record_observation(
            config, result.measures.cost_usd
        )

        return result
```

---

## 11. Next Steps

1. Validate this design with stakeholders
2. Prototype `VotingPattern` as proof-of-concept
3. Implement `CalibratedObjectiveModel` for uncertainty-aware estimation
4. Define `PatternMeasures` contract and integrate with `TrialResult`
5. Add budget guardrail integration to pattern execution
6. Define integration points in `OptimizedFunction` and `OptimizationOrchestrator`
7. Add pattern catalog to `traigent.api`

---

## 12. Iterative Workflow Model (MapReduce-Inspired)

### 12.1 Motivation

For batch processing of many items (e.g., 100+ extraction tasks), simple patterns are insufficient. We need a **multi-phase iterative workflow** inspired by MapReduce:

- **Progressive refinement**: Start cheap, escalate for failures
- **Bundled execution**: Amortize LLM call overhead
- **Judge integration**: Validate results, route failures
- **Budget-aware iteration**: Stop when quality/budget goals met

### 12.2 Workflow Architecture

```text
ROUND 1: Cheap + Bundled (high throughput)
  ├── Success (85%) → Done
  ├── Low Confidence (10%) → Queue for Round 2
  └── Failed (5%) → Queue for Round 2

ROUND 2: Better Model + Smaller Bundles
  ├── Success (80% of remaining) → Done
  └── Failed/Low Confidence → Queue for Round 3

ROUND 3: Expensive + Voting
  └── Final attempt with redundancy
```

### 12.3 Core Abstractions

```python
@dataclass
class WorkItem:
    """Item in the work queue."""
    id: str
    input: Any
    status: Literal["pending", "processing", "success", "failed", "low_confidence"]
    result: Any | None = None
    confidence: float | None = None
    attempts: int = 0
    round_history: list[RoundResult] = field(default_factory=list)


@dataclass
class RoundStrategy:
    """Strategy for a single round."""
    round: int | str  # Round number or "retry" / "low_confidence"

    # Partitioning (Map phase)
    partition: Literal["size", "semantic", "difficulty", "random"]
    bundle_size: int

    # Execution
    execution: Literal["bundled", "parallel", "voting"]
    model: str
    votes: int = 1

    # Budget constraints
    max_cost_per_item: float | None = None


@dataclass
class WorkflowConfig:
    """Configuration for iterative workflow."""

    # Stopping conditions
    max_rounds: int = 5
    max_budget_usd: float = 10.0
    target_success_rate: float = 0.95
    confidence_threshold: float = 0.8
    max_attempts_per_item: int = 3

    # Round strategies (optimizable)
    round_strategies: list[RoundStrategy] = field(default_factory=list)

    # Judge configuration
    judge_model: str = "gpt-4o-mini"
    judge_bundle_size: int = 20
```

### 12.4 Workflow Execution

```python
class IterativeWorkflow:
    """Multi-phase MapReduce-style workflow."""

    def execute(self, inputs: list[Any]) -> WorkflowResult:
        self.work_queue = [WorkItem(id=str(i), input=inp, status="pending")
                          for i, inp in enumerate(inputs)]

        round_number = 0
        while self._should_continue(round_number):
            # 1. SPLIT (Map) - partition pending items
            pending = [w for w in self.work_queue if w.status == "pending"]
            strategy = self._select_strategy(round_number, pending)
            partitions = self._split(pending, strategy)

            # 2. EXECUTE (bundled LLM calls)
            results = []
            for partition in partitions:
                bundle_result = self._execute_bundle(partition, strategy)
                results.extend(bundle_result)

            # 3. JUDGE (validate/score - also bundled)
            judged = self._judge_results(results)

            # 4. REDUCE (update work queue)
            self._reduce(judged, round_number)
            round_number += 1

        return self._collect_results()
```

### 12.5 Round Strategy Progression (Default)

| Round | Model | Bundle Size | Execution | Target |
|-------|-------|-------------|-----------|--------|
| 1 | gpt-4o-mini | 10 | bundled | Easy cases (80%+) |
| 2 | gpt-4o | 5 | bundled | Medium difficulty |
| 3 | gpt-4o | 1 | voting (3) | Hard cases |
| 4 | claude-opus | 1 | individual | Final attempt |

### 12.6 Cost Efficiency Example

```text
100 items, naive approach:
  100 × gpt-4o individual calls = $5.00

100 items, iterative workflow:
  Round 1: 10 bundled calls × gpt-4o-mini = $0.10 → 85 success
  Round 2: 3 bundled calls × gpt-4o = $0.15 → 12 success
  Round 3: 3 × 3 voting calls × gpt-4o = $0.25 → 2 success
  Total: $0.50 for 99% success rate (10× cheaper)
```

### 12.7 Optimizer Integration

The workflow configuration becomes an optimizable space:

```python
workflow_config_space = ConfigSpace(tvars={
    # Round 1
    "r1_bundle_size": IntRange(5, 20),
    "r1_model": Choices(["gpt-4o-mini", "gpt-4o"]),

    # Round 2 (retries)
    "r2_bundle_size": IntRange(1, 10),
    "r2_model": Choices(["gpt-4o", "claude-sonnet"]),
    "r2_votes": IntRange(1, 5),

    # Judge
    "judge_model": Choices(["gpt-4o-mini", "gpt-4o"]),
    "confidence_threshold": Range(0.6, 0.95),

    # Stopping
    "max_rounds": IntRange(2, 5),
    "target_success_rate": Range(0.9, 0.99),
})
```

### 12.8 Relationship to Patterns

The iterative workflow **orchestrates** patterns:

```text
┌─────────────────────────────────────────────────────────────┐
│                  ITERATIVE WORKFLOW                          │
│  (Orchestrates rounds, manages work queue)                   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌───────────┐   ┌───────────┐   ┌───────────┐
       │  Round 1  │   │  Round 2  │   │  Round 3  │
       │ Bundling  │   │ Bundling  │   │  Voting   │
       │  Pattern  │   │  Pattern  │   │  Pattern  │
       └───────────┘   └───────────┘   └───────────┘
              │               │               │
              ▼               ▼               ▼
       ┌─────────────────────────────────────────┐
       │            JUDGE PATTERN                 │
       │  (Validates results, routes failures)    │
       └─────────────────────────────────────────┘
```

---

## Appendix A: Review History

| Date | Reviewer | Model | Key Feedback |
|------|----------|-------|--------------|
| 2026-01-25 | Codex | GPT-5.2 | Uncertainty in estimates, runtime measures contract, observability |

---

## 13. Scheduling Algorithms for Work Ordering

### 13.1 The Scheduling Problem

Given N work items, determine:
- **Order**: Which items to process first
- **Resource**: Which model/config to use
- **Grouping**: How to bundle items
- **Adaptation**: How to re-prioritize based on results

### 13.2 Classic Algorithms Applied

| Algorithm | LLM Application | When to Use |
|-----------|-----------------|-------------|
| **SJF (Shortest Job First)** | Easy items first | Maximize early throughput |
| **Priority** | Important items first | Business-critical items |
| **MLFQ** | Adaptive tier demotion | Unknown difficulty distribution |
| **EDF (Earliest Deadline)** | Deadline-critical first | Latency SLA requirements |
| **Fair Share** | Balance across types | Multiple item categories |

### 13.3 Difficulty Estimation

```python
class DifficultyEstimator:
    """Estimate processing difficulty for scheduling."""

    def estimate(self, item: WorkItem) -> float:
        """Return difficulty score 0-1 (0=easy, 1=hard)."""
        features = {
            "length": len(item.input) / 10000,
            "complexity": self._text_complexity(item.input),
            "entity_count": self._count_entities(item.input),
            "previous_failures": item.attempts / 3,
            "historical_failure_rate": self._cluster_failure_rate(item),
        }

        weights = {
            "length": 0.15,
            "complexity": 0.20,
            "entity_count": 0.15,
            "previous_failures": 0.20,
            "historical_failure_rate": 0.30,
        }

        return sum(features[k] * weights[k] for k in features)

    def update(self, item: WorkItem, success: bool):
        """Update historical stats for online learning."""
        cluster = self._get_cluster(item)
        self.stats[cluster]["attempts"] += 1
        if not success:
            self.stats[cluster]["failures"] += 1
```

### 13.4 Multilevel Feedback Queue (MLFQ)

Items start in easy queue, demote on failure:

```text
┌─────────────────────────────────────────────────────────────┐
│ EASY QUEUE (bundle=10, model=mini)                          │
│ ┌───┐┌───┐┌───┐┌───┐┌───┐ ← New items start here            │
│ └───┘└───┘└───┘└───┘└───┘                                   │
│        │ failure                                             │
│        ▼                                                     │
│ MEDIUM QUEUE (bundle=5, model=4o)                           │
│ ┌───┐┌───┐┌───┐ ← Demoted from easy                         │
│ └───┘└───┘└───┘                                             │
│        │ failure                                             │
│        ▼                                                     │
│ HARD QUEUE (bundle=1, model=opus, votes=3)                  │
│ ┌───┐┌───┐ ← Demoted from medium                            │
│ └───┘└───┘                                                  │
└─────────────────────────────────────────────────────────────┘
```

```python
class MLFQScheduler:
    """Multilevel Feedback Queue scheduler."""

    def __init__(self):
        self.queues = {
            "easy": [],
            "medium": [],
            "hard": [],
        }
        self.strategies = {
            "easy": RoundStrategy(bundle_size=10, model="gpt-4o-mini"),
            "medium": RoundStrategy(bundle_size=5, model="gpt-4o"),
            "hard": RoundStrategy(bundle_size=1, model="claude-opus", votes=3),
        }

    def add_item(self, item: WorkItem):
        self.queues["easy"].append(item)

    def demote(self, item: WorkItem):
        for tier, next_tier in [("easy", "medium"), ("medium", "hard")]:
            if item in self.queues[tier]:
                self.queues[tier].remove(item)
                self.queues[next_tier].append(item)
                return

    def get_next_batch(self) -> tuple[list[WorkItem], RoundStrategy]:
        for tier in ["easy", "medium", "hard"]:
            if self.queues[tier]:
                strategy = self.strategies[tier]
                batch = self.queues[tier][:strategy.bundle_size]
                return batch, strategy
        return [], None
```

### 13.5 Scheduler Configuration Space

```python
scheduling_config_space = ConfigSpace(tvars={
    # Scheduler type
    "scheduler": Choices(["sjf", "priority", "mlfq", "edf", "fair"]),

    # Difficulty estimation
    "difficulty_method": Choices(["heuristic", "embedding", "historical"]),

    # MLFQ-specific
    "mlfq_queues": IntRange(2, 5),
    "mlfq_demotion_threshold": IntRange(1, 3),

    # Priority weights
    "priority_importance_weight": Range(0.0, 1.0),
    "priority_difficulty_weight": Range(0.0, 1.0),
})
```

---

## Appendix B: Distributed Computing Inspirations

| Distributed Concept | LLM Analog |
|---------------------|------------|
| MapReduce | Split → Bundle Execute → Judge → Reduce |
| Speculative Execution | Hedge pattern - race models |
| Checkpointing | Resume long extractions |
| Circuit Breaker | Disable failing models |
| Back-pressure | Budget guardrails |
| Data Locality | Bundle semantically related items |

---

## Appendix C: CS Algorithm Inspirations

| CS Problem | LLM Pattern | Key Insight |
|------------|-------------|-------------|
| Sorting/Selection | Tournament judge | Pairwise comparison cheaper than scoring |
| Graph Coloring | Compatible bundling | Bundle items without conflicts |
| TCP Congestion | Adaptive rate limiting | AIMD for API calls |
| Leader Election | Quorum voting | Early termination on consensus |
| Byzantine Fault | Hallucination tolerance | 3f+1 models for f faulty |
| Bandits | Model routing | Thompson sampling for selection |
| Caching | Semantic memoization | Cache by embedding similarity |
| Shortest Path | Cascade routing | Minimize expected cost path |
