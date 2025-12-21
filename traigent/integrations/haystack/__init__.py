"""Haystack Pipeline Integration for Traigent.

This module provides pipeline introspection and optimization capabilities
for Haystack AI pipelines, using TVL (Tuning Variable Language) terminology.

Terminology (aligned with TVL):
- TVAR (Tuned Variable): A parameter that can be optimized
- DiscoveredTVAR: A TVAR discovered through pipeline introspection
- TVARScope: A namespace/container for TVARs (maps to Haystack Component)
- PipelineSpec: The complete discovered pipeline structure with all scopes
- Connection: A data flow edge between TVARScopes
- ExplorationSpace: Optimization search space with TVARs and constraints
- CategoricalConstraint: Constraint for discrete choice parameters
- NumericalConstraint: Constraint for numerical parameters with bounds

Example usage:
    from traigent.integrations.haystack import from_pipeline, PipelineSpec

    pipeline = Pipeline()
    # ... add components ...

    # For discovery (pipeline structure):
    pipeline_spec = from_pipeline(pipeline)
    print(pipeline_spec.scopes)  # List of discovered TVARScopes
    print(pipeline_spec.connections)  # List of data flow connections

    # For optimization (search space):
    space = from_pipeline(pipeline, as_exploration_space=True)
    print(space.tunable_tvars)  # Dict of tunable TVARs with constraints

Backwards Compatibility:
    The old names (ConfigSpace, Component, Edge, Parameter) are still available
    as aliases for the new names to support existing code during migration.
"""

from __future__ import annotations

from .advanced_optimization import (
    HaystackOptimizer,
    OptimizationDirection,
    OptimizationResult,
    OptimizationTarget,
    TrialResult,
    compute_pareto_frontier,
    export_optimization_history,
    get_hyperparameter_importance,
    rank_by_metric,
)
from .attribution import (
    ComponentAttribution,
    NodeMetrics,
    compute_attribution,
    compute_cost_contribution,
    compute_latency_contribution,
    compute_quality_contribution,
    compute_sensitivity_scores,
    export_attribution,
    extract_node_metrics,
    get_attribution_ranked,
)
from .configspace import (
    TVAR,
    CategoricalConstraint,
    ConditionalConstraint,
    Configuration,
    ExplorationSpace,
    NumericalConstraint,
    TVARConstraint,
)
from .cost_tracking import (
    CostResult,
    HaystackCostTracker,
    TokenUsage,
    extract_token_usage,
    get_cost_metrics,
)
from .evaluation import (
    EvaluationDataset,
    EvaluationExample,
    validate_dataset,
)
from .evaluator import HaystackEvaluator
from .execution import (
    ExampleResult,
    RunResult,
    apply_config,
    execute_with_config,
)
from .introspection import (
    PARAMETER_SEMANTICS,
    TVAR_SEMANTICS,
    from_pipeline,
)
from .latency_tracking import (
    LatencyStats,
    compute_latency_stats,
    extract_latencies_from_results,
    get_latency_metrics,
)
from .metric_constraints import (
    ConstraintCheckResult,
    ConstraintViolation,
    MetricConstraint,
    check_constraints,
    cost_constraint,
    filter_by_constraints,
    get_best_satisfying,
    latency_constraint,
    quality_constraint,
)

# New TVL-aligned types (preferred); Backwards-compatible aliases (deprecated)
from .models import (
    Component,
    ConfigSpace,
    Connection,
    DiscoveredTVAR,
    Edge,
    Parameter,
    PipelineSpec,
    TVARScope,
)
from .production import (
    ApplyBackup,
    CLIResult,
    ConfigMismatchError,
    ConfigValidationError,
    OptimizationConfig,
    TunedConfig,
    apply_config_production,
    create_cli_result,
    export_experiment_history,
    export_tuned_config,
    load_artifacts,
    load_optimization_config,
    load_tuned_config,
    rollback_config,
    save_artifacts,
    save_optimization_config,
)

__all__ = [
    # Core function
    "from_pipeline",
    # New TVL-aligned types (preferred)
    "PipelineSpec",
    "TVARScope",
    "DiscoveredTVAR",
    "Connection",
    "TVAR_SEMANTICS",
    # Optimization types (Epic 2) - aligned with TVL Glossary v2.0
    "ExplorationSpace",  # 𝒳 - feasible configuration set
    "TVAR",  # tᵢ - tuned variable
    "TVARConstraint",  # Domain constraint (Dᵢ)
    "CategoricalConstraint",  # C^str for discrete choices
    "NumericalConstraint",  # C^str for numerical ranges
    "ConditionalConstraint",  # C^str for inter-TVAR dependencies
    "Configuration",  # θ - assignment to all TVARs
    # Evaluation types (Epic 3)
    "EvaluationDataset",  # Collection of evaluation examples
    "EvaluationExample",  # Single evaluation example
    "validate_dataset",  # Dataset validation function
    # Execution types (Epic 3)
    "execute_with_config",  # Run pipeline with config against dataset
    "apply_config",  # Apply config to pipeline
    "RunResult",  # Result of a pipeline run
    "ExampleResult",  # Result for a single example
    # Evaluator (Epic 3) - integrates with core optimization infrastructure
    "HaystackEvaluator",  # BaseEvaluator implementation for Haystack pipelines
    # Cost tracking (Epic 4)
    "HaystackCostTracker",  # Token/cost tracking for Haystack pipelines
    "TokenUsage",  # Token usage data structure
    "CostResult",  # Cost calculation result
    "extract_token_usage",  # Extract tokens from pipeline output
    "get_cost_metrics",  # Convert cost result to metrics dict
    # Latency tracking (Epic 4)
    "LatencyStats",  # Latency statistics (p50, p95, p99)
    "compute_latency_stats",  # Compute latency percentiles
    "extract_latencies_from_results",  # Extract latencies from results
    "get_latency_metrics",  # Convert latency stats to metrics dict
    # Metric constraints (Epic 4)
    "MetricConstraint",  # Post-evaluation metric constraint
    "ConstraintCheckResult",  # Result of constraint checking
    "ConstraintViolation",  # Individual constraint violation
    "check_constraints",  # Check multiple constraints
    "cost_constraint",  # Helper for cost constraints
    "latency_constraint",  # Helper for latency constraints
    "quality_constraint",  # Helper for quality metric constraints
    # Result filtering (Epic 4 - Story 4.5)
    "filter_by_constraints",  # Filter results by constraint satisfaction
    "get_best_satisfying",  # Get best result satisfying constraints
    # Advanced optimization (Epic 5)
    "HaystackOptimizer",  # Main optimizer for Haystack pipelines
    "OptimizationTarget",  # Target definition (metric + direction)
    "OptimizationDirection",  # Maximize or minimize
    "OptimizationResult",  # Result container with Pareto frontier
    "TrialResult",  # Single trial result
    "compute_pareto_frontier",  # Compute Pareto-optimal configs
    "rank_by_metric",  # Rank configs by metric
    "get_hyperparameter_importance",  # Hyperparameter importance analysis
    "export_optimization_history",  # Export history to JSON/CSV
    # Attribution & Insights (Epic 6)
    "NodeMetrics",  # Per-run metrics for a component
    "ComponentAttribution",  # Attribution insights for a component
    "compute_attribution",  # Compute full attribution analysis
    "compute_quality_contribution",  # Quality impact per component
    "compute_cost_contribution",  # Cost fraction per component
    "compute_latency_contribution",  # Latency fraction per component
    "compute_sensitivity_scores",  # Parameter sensitivity analysis
    "extract_node_metrics",  # Extract per-node metrics from run
    "get_attribution_ranked",  # Get ranked component list
    "export_attribution",  # Export attribution to JSON/CSV
    # Production Readiness (Epic 7)
    "OptimizationConfig",  # Configuration for optimization run
    "ConfigValidationError",  # Config validation error
    "ConfigMismatchError",  # Config mismatch error
    "load_optimization_config",  # Load config from YAML/JSON
    "save_optimization_config",  # Save config to YAML/JSON
    "ApplyBackup",  # Backup for rollback
    "apply_config_production",  # Apply config with validation
    "rollback_config",  # Rollback to previous config
    "TunedConfig",  # Tuned configuration with metadata
    "export_tuned_config",  # Export as TVL
    "load_tuned_config",  # Load from TVL
    "export_experiment_history",  # Export full history (was in Epic 5)
    "save_artifacts",  # Save all artifacts to directory
    "load_artifacts",  # Load artifacts from directory
    "CLIResult",  # CLI result for CI/CD
    "create_cli_result",  # Create CLI result from OptimizationResult
    # Backwards-compatible aliases (deprecated, for migration)
    "ConfigSpace",
    "Component",
    "Edge",
    "Parameter",
    "PARAMETER_SEMANTICS",
]
