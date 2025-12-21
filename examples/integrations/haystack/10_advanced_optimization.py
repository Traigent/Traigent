#!/usr/bin/env python3
"""Example 10: Advanced Optimization with Pareto Analysis.

This example demonstrates Epic 5 features:
- Bayesian optimization (TPE sampler)
- Evolutionary optimization (NSGA-II)
- Multi-objective optimization
- Pareto frontier computation
- Hyperparameter importance analysis
- Optimization history export

These features build on Epic 4's constraint system and provide
sophisticated optimization strategies for production ML pipelines.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

# Import all Epic 5 components
from traigent.integrations.haystack import (
    ExplorationSpace,
    OptimizationDirection,
    OptimizationResult,
    OptimizationTarget,
    TrialResult,
    compute_pareto_frontier,
    export_optimization_history,
    get_hyperparameter_importance,
    rank_by_metric,
)

# Constants for parameter names
PARAM_TEMP = "generator.temperature"
PARAM_TOPK = "retriever.top_k"


def create_mock_pipeline() -> Any:
    """Create a mock pipeline for demonstration."""

    class MockPipeline:
        def __init__(self):
            self.components = {
                "generator": MockComponent("generator", temperature=0.7),
                "retriever": MockComponent("retriever", top_k=5),
            }

        def get_component(self, name: str) -> Any:
            return self.components.get(name)

        def run(self, _data: dict) -> dict:
            return {"answer": "Mock response", "documents": []}

    class MockComponent:
        def __init__(self, name: str, **defaults):
            self.name = name
            for k, v in defaults.items():
                setattr(self, k, v)

    return MockPipeline()


def example_optimization_targets():
    """Example 1: Define multi-objective optimization targets.

    OptimizationTarget defines what metrics to optimize and in which
    direction. Supports weighted multi-objective optimization.
    """
    print("=" * 60)
    print("Example 1: Optimization Targets")
    print("=" * 60)

    # Single objective: maximize accuracy
    accuracy_target = OptimizationTarget(
        metric_name="accuracy",
        direction="maximize",
    )
    print(f"Accuracy target: {accuracy_target.metric_name}")
    print(f"  Direction: {accuracy_target.direction}")
    print(f"  Weight: {accuracy_target.weight}")

    # Multiple objectives with weights
    targets = [
        OptimizationTarget("accuracy", "maximize", weight=1.0),
        OptimizationTarget("cost", "minimize", weight=0.5),
        OptimizationTarget("latency_p95", "minimize", weight=0.3),
    ]

    print("\nMulti-objective targets:")
    for target in targets:
        is_max = target.direction == OptimizationDirection.MAXIMIZE
        dir_str = "MAX" if is_max else "MIN"
        print(f"  {target.metric_name}: {dir_str} (weight={target.weight})")

    # Enum-based direction specification
    target_enum = OptimizationTarget(
        metric_name="f1_score",
        direction=OptimizationDirection.MAXIMIZE,
    )
    direction_value = target_enum.direction.value
    print(f"\nUsing enum: {target_enum.metric_name} -> {direction_value}")


def example_trial_results():
    """Example 2: Working with trial results.

    TrialResult captures the outcome of a single optimization trial,
    including configuration, metrics, and constraint status.
    """
    print("\n" + "=" * 60)
    print("Example 2: Trial Results")
    print("=" * 60)

    # Create sample trial results
    trials = [
        TrialResult(
            trial_id="trial_001",
            config={PARAM_TEMP: 0.5, PARAM_TOPK: 3},
            metrics={"accuracy": 0.82, "cost": 0.015, "latency_p95": 1.2},
            constraints_satisfied=True,
            duration=45.2,
        ),
        TrialResult(
            trial_id="trial_002",
            config={PARAM_TEMP: 0.9, PARAM_TOPK: 10},
            metrics={"accuracy": 0.91, "cost": 0.045, "latency_p95": 2.8},
            constraints_satisfied=True,
            duration=78.5,
        ),
        TrialResult(
            trial_id="trial_003",
            config={PARAM_TEMP: 0.3, PARAM_TOPK: 1},
            metrics={"accuracy": 0.65, "cost": 0.008, "latency_p95": 0.5},
            constraints_satisfied=False,  # Violated min accuracy constraint
            duration=22.1,
        ),
        TrialResult(
            trial_id="trial_004",
            config={PARAM_TEMP: 0.7, PARAM_TOPK: 5},
            metrics={},  # Failed trial - no metrics
            constraints_satisfied=False,
            duration=120.0,
        ),
    ]

    print(f"Created {len(trials)} trial results:\n")
    for trial in trials:
        status = "SUCCESS" if trial.is_successful else "FAILED"
        constraint = "SATISFIED" if trial.constraints_satisfied else "VIOLATED"
        print(f"  {trial.trial_id}:")
        print(f"    Status: {status}")
        print(f"    Config: {trial.config}")
        if trial.is_successful:
            print(f"    Accuracy: {trial.metrics.get('accuracy', 'N/A')}")
            print(f"    Constraints: {constraint}")
        else:
            print("    Error: No metrics (failed)")
        print(f"    Duration: {trial.duration:.1f}s")
        print()

    return trials


def example_pareto_frontier(trials: list[TrialResult]):
    """Example 3: Compute Pareto frontier for multi-objective optimization.

    The Pareto frontier contains all non-dominated solutions - configs
    where no other config is better in ALL objectives simultaneously.
    """
    print("=" * 60)
    print("Example 3: Pareto Frontier Computation")
    print("=" * 60)

    # Define objectives for Pareto analysis
    targets = [
        OptimizationTarget("accuracy", "maximize"),
        OptimizationTarget("cost", "minimize"),
    ]

    print("Computing Pareto frontier for:")
    print("  - Maximize accuracy")
    print("  - Minimize cost")
    print()

    # Compute Pareto frontier
    pareto_configs, pareto_metrics = compute_pareto_frontier(trials, targets)

    print(f"Found {len(pareto_configs)} Pareto-optimal configurations:\n")
    for i, (config, metrics) in enumerate(zip(pareto_configs, pareto_metrics)):
        print(f"  Pareto #{i + 1}:")
        print(f"    Config: {config}")
        print(f"    Accuracy: {metrics.get('accuracy', 'N/A')}")
        print(f"    Cost: ${metrics.get('cost', 0):.3f}")
        print()

    # Three-objective Pareto
    targets_3obj = [
        OptimizationTarget("accuracy", "maximize"),
        OptimizationTarget("cost", "minimize"),
        OptimizationTarget("latency_p95", "minimize"),
    ]

    print("Three-objective Pareto frontier:")
    print("  - Maximize accuracy")
    print("  - Minimize cost")
    print("  - Minimize latency")

    pareto_3, _ = compute_pareto_frontier(trials, targets_3obj)
    print(f"\nFound {len(pareto_3)} Pareto-optimal configurations")


def example_ranking(trials: list[TrialResult]):
    """Example 4: Rank configurations by primary metric.

    Ranking provides a simple way to find the best configurations
    for a single objective while respecting constraints.
    """
    print("\n" + "=" * 60)
    print("Example 4: Configuration Ranking")
    print("=" * 60)

    # Rank by accuracy (maximize)
    print("Ranking by accuracy (highest first):")
    ranked = rank_by_metric(
        trials,
        metric_name="accuracy",
        direction=OptimizationDirection.MAXIMIZE,
    )

    for i, trial in enumerate(ranked):
        acc = trial.metrics.get("accuracy", "N/A")
        print(f"  #{i + 1}: accuracy={acc:.2f}")
        print(f"       config={trial.config}")

    # Rank by cost (minimize)
    print("\nRanking by cost (lowest first):")
    ranked_cost = rank_by_metric(
        trials,
        metric_name="cost",
        direction=OptimizationDirection.MINIMIZE,
    )

    for i, trial in enumerate(ranked_cost):
        print(f"  #{i + 1}: cost=${trial.metrics.get('cost', 0):.3f}")
        print(f"       config={trial.config}")


def example_hyperparameter_importance(trials: list[TrialResult]):
    """Example 5: Analyze hyperparameter importance.

    Identifies which parameters have the most impact on the
    target metric, helping prioritize tuning efforts.
    """
    print("\n" + "=" * 60)
    print("Example 5: Hyperparameter Importance Analysis")
    print("=" * 60)

    # Need more trials for importance analysis (minimum 5)
    # Create additional mock trials
    extended_trials = list(trials) + [
        TrialResult(
            trial_id="trial_005",
            config={PARAM_TEMP: 0.6, PARAM_TOPK: 7},
            metrics={"accuracy": 0.86, "cost": 0.025, "latency_p95": 1.5},
            constraints_satisfied=True,
            duration=50.0,
        ),
        TrialResult(
            trial_id="trial_006",
            config={PARAM_TEMP: 0.4, PARAM_TOPK: 4},
            metrics={"accuracy": 0.75, "cost": 0.018, "latency_p95": 1.0},
            constraints_satisfied=True,
            duration=35.0,
        ),
        TrialResult(
            trial_id="trial_007",
            config={PARAM_TEMP: 0.8, PARAM_TOPK: 8},
            metrics={"accuracy": 0.89, "cost": 0.038, "latency_p95": 2.2},
            constraints_satisfied=True,
            duration=65.0,
        ),
    ]

    # Compute importance for accuracy
    importance = get_hyperparameter_importance(
        extended_trials, target_metric="accuracy"
    )

    if importance:
        print("Hyperparameter importance for accuracy:\n")
        # Sort by importance
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for param, score in sorted_params:
            bar = "#" * int(score * 20)
            print(f"  {param}: {score:.3f} {bar}")
    else:
        print("Insufficient trials for importance analysis")
        print("(Need at least 5 successful trials with varied configs)")


def example_export_history(trials: list[TrialResult]):
    """Example 6: Export optimization history.

    Export trial history to JSON or CSV for analysis,
    visualization, or integration with other tools.
    """
    print("\n" + "=" * 60)
    print("Example 6: Export Optimization History")
    print("=" * 60)

    # Create result container
    result = OptimizationResult(
        best_config={PARAM_TEMP: 0.9, PARAM_TOPK: 10},
        best_metrics={"accuracy": 0.91, "cost": 0.045},
        history=trials,
        total_trials=len(trials),
        duration=265.8,
    )

    # Export to dict (in-memory)
    dict_output = export_optimization_history(result, format="dict")
    print("Exported to dict:")
    print(f"  Keys: {list(dict_output.keys())}")
    print(f"  Total trials: {dict_output['summary']['total_trials']}")
    success_count = sum(1 for t in trials if t.is_successful)
    success_rate = success_count / len(trials) if trials else 0
    print(f"  Success rate: {success_rate:.1%}")

    # Export to temporary files for demonstration
    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON file
        json_path = Path(tmpdir) / "optimization_history.json"
        export_optimization_history(result, format="json", output_path=str(json_path))
        print(f"\nJSON export size: {json_path.stat().st_size} bytes")

        # Preview JSON structure
        with open(json_path) as f:
            data = json.load(f)
            print(f"  Trials in export: {len(data['trials'])}")

        # CSV file
        csv_path = Path(tmpdir) / "optimization_history.csv"
        export_optimization_history(result, format="csv", output_path=str(csv_path))
        print(f"\nCSV export size: {csv_path.stat().st_size} bytes")

        # Preview CSV content
        with open(csv_path) as f:
            lines = f.readlines()
            print(f"  Header: {lines[0].strip()[:60]}...")
            print(f"  Rows: {len(lines) - 1}")


def example_optimizer_strategies():
    """Example 7: Compare different optimization strategies.

    Demonstrates configuration of Bayesian, evolutionary (NSGA-II),
    random, and grid search strategies.
    """
    print("\n" + "=" * 60)
    print("Example 7: Optimization Strategies")
    print("=" * 60)

    strategies = ["bayesian", "tpe", "evolutionary", "nsga2", "random", "grid"]

    print("Available optimization strategies:\n")
    for strategy in strategies:
        print(f"  {strategy.upper()}:")
        print(f"    {get_strategy_description(strategy)}")
        print()

    print("Strategy selection guide:")
    print("  - bayesian/tpe: Best for expensive evaluations, few trials")
    print("  - evolutionary/nsga2: Best for multi-objective Pareto")
    print("  - random: Good baseline, useful for high-dimensional spaces")
    print("  - grid: Exhaustive search for small discrete spaces")


def get_strategy_description(strategy: str) -> str:
    """Get description for optimization strategy."""
    descriptions = {
        "bayesian": "Uses TPE sampler for sample-efficient optimization",
        "tpe": "Tree-structured Parzen Estimator (same as bayesian)",
        "evolutionary": "NSGA-II for multi-objective Pareto optimization",
        "nsga2": "Non-dominated Sorting Genetic Algorithm II",
        "random": "Random sampling for baseline comparison",
        "grid": "Exhaustive grid search over discrete space",
    }
    return descriptions.get(strategy, "Unknown strategy")


def example_exploration_space():
    """Example 8: Define exploration spaces for optimization.

    ExplorationSpace defines the search space for hyperparameter
    optimization, supporting continuous, discrete, and categorical.
    """
    print("\n" + "=" * 60)
    print("Example 8: Exploration Space Definition")
    print("=" * 60)

    space = ExplorationSpace()

    # Add various parameter types
    space.add_continuous(PARAM_TEMP, 0.1, 1.0)
    space.add_discrete(PARAM_TOPK, [1, 3, 5, 10, 20])
    space.add_categorical("generator.model", ["gpt-3.5", "gpt-4", "claude"])
    space.add_boolean("use_reranking")

    print("Exploration space parameters:")
    for param in space.parameters:
        print(f"  {param.name}:")
        print(f"    Type: {param.type}")
        if hasattr(param, "low") and hasattr(param, "high"):
            print(f"    Range: [{param.low}, {param.high}]")
        if hasattr(param, "choices") and param.choices:
            print(f"    Choices: {param.choices}")

    # Convert to config space dict
    config_space = space.to_dict()
    print(f"\nConfig space dict: {len(config_space)} parameters")


def example_time_budgeted_optimization():
    """Example 9: Time-budgeted optimization.

    Configure optimization with a time limit for production scenarios.
    """
    print("\n" + "=" * 60)
    print("Example 9: Time-Budgeted Optimization")
    print("=" * 60)

    print("Time budget configuration:")
    print("  When creating HaystackOptimizer, set timeout_seconds:")
    print()
    print("  optimizer = HaystackOptimizer(")
    print("      evaluator=evaluator,")
    print("      config_space=space,")
    print("      n_trials=1000,        # High trial limit")
    print("      timeout_seconds=300,  # But 5 minute budget")
    print("  )")
    print()
    print("  Optimization will stop when EITHER:")
    print("    - n_trials reached")
    print("    - timeout_seconds exceeded")


def example_warm_start_config():
    """Example 10: Warm-start configuration.

    Resume optimization from previous results.
    """
    print("\n" + "=" * 60)
    print("Example 10: Warm-Start Configuration")
    print("=" * 60)

    # Simulate previous results
    previous_trials = [
        TrialResult(
            trial_id="prev_1",
            config={PARAM_TEMP: 0.5},
            metrics={"accuracy": 0.78},
        ),
        TrialResult(
            trial_id="prev_2",
            config={PARAM_TEMP: 0.7},
            metrics={"accuracy": 0.85},
        ),
    ]

    print(f"Previous trials: {len(previous_trials)}")
    for trial in previous_trials:
        print(f"  {trial.trial_id}: {trial.metrics}")

    print("\nTo warm-start optimization:")
    print("  1. Load previous TrialResults from checkpoint")
    print("  2. Call optimizer.warm_start(previous_trials)")
    print("  3. Then call optimizer.optimize()")
    print()
    print("  # Example:")
    print("  await optimizer.warm_start(previous_trials)")
    print("  result = await optimizer.optimize()")
    print()
    print("The optimizer will use previous trials to inform")
    print("future suggestions (especially useful for Bayesian).")


def example_parallel_execution():
    """Example 11: Parallel experiment execution.

    Configure parallel evaluation for faster optimization.
    """
    print("\n" + "=" * 60)
    print("Example 11: Parallel Execution")
    print("=" * 60)

    print("Parallel execution configuration:")
    print()
    print("  optimizer = HaystackOptimizer(")
    print("      evaluator=evaluator,")
    print("      config_space=space,")
    print("      n_trials=100,")
    print("      n_parallel=4,  # Run 4 trials concurrently")
    print("  )")
    print()
    print("Benefits:")
    print("  - Faster optimization (4x with n_parallel=4)")
    print("  - Better resource utilization")
    print("  - Useful when evaluations are I/O-bound")
    print()
    print("Considerations:")
    print("  - Bayesian optimization may be less sample-efficient")
    print("  - Ensure evaluator is thread-safe")
    print("  - Monitor resource usage (memory, API rate limits)")


def example_complete_workflow():
    """Example 12: Complete optimization workflow summary.

    Shows how all Epic 5 components fit together.
    """
    print("\n" + "=" * 60)
    print("Example 12: Complete Optimization Workflow")
    print("=" * 60)

    workflow_text = """
Complete workflow for Haystack pipeline optimization:

1. SETUP
   - Create pipeline and evaluation dataset
   - Define exploration space (search space)
   - Configure constraints (from Epic 4)

2. OPTIMIZATION TARGETS
   targets = [
       OptimizationTarget("accuracy", "maximize", weight=1.0),
       OptimizationTarget("cost", "minimize", weight=0.3),
   ]

3. CREATE OPTIMIZER
   optimizer = HaystackOptimizer(
       evaluator=evaluator,
       config_space=space,
       targets=targets,
       strategy="bayesian",  # or "evolutionary"
       n_trials=50,
       n_parallel=2,
       timeout_seconds=3600,
   )

4. RUN OPTIMIZATION
   result = await optimizer.optimize(
       progress_callback=lambda n, r: print(f"Trial {n}")
   )

5. ANALYZE RESULTS
   # Get best configuration
   print(f"Best: {result.best_config}")
   print(f"Metrics: {result.best_metrics}")

   # Get Pareto frontier for multi-objective
   for cfg, met in zip(result.pareto_configs, result.pareto_metrics):
       print(f"Pareto: {cfg} -> {met}")

   # Analyze parameter importance
   importance = get_hyperparameter_importance(result.history, "accuracy")

6. EXPORT & CHECKPOINT
   export_optimization_history(result, format="json", path="out.json")
   export_optimization_history(result, format="csv", path="out.csv")
"""
    print(workflow_text)


def main():
    """Run all Epic 5 examples."""
    print("\n" + "#" * 60)
    print("# Epic 5: Advanced Optimization & Pareto Analysis")
    print("#" * 60)

    # Run examples
    example_optimization_targets()
    trials = example_trial_results()
    example_pareto_frontier(trials)
    example_ranking(trials)
    example_hyperparameter_importance(trials)
    example_export_history(trials)
    example_optimizer_strategies()
    example_exploration_space()
    example_time_budgeted_optimization()
    example_warm_start_config()
    example_parallel_execution()
    example_complete_workflow()

    print("\n" + "#" * 60)
    print("# All Epic 5 examples completed!")
    print("#" * 60)


if __name__ == "__main__":
    main()
