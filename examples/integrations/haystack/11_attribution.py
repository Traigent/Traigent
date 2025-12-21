#!/usr/bin/env python3
"""Example 11: Attribution & Insights.

This example demonstrates Epic 6 features:
- Per-node metrics collection
- Quality contribution analysis
- Cost and latency contribution analysis
- Parameter sensitivity analysis
- Optimization recommendations

These features help identify which pipeline components are causing
quality/cost/latency issues and provide actionable recommendations.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from traigent.integrations.haystack import (
    ComponentAttribution,
    NodeMetrics,
    TrialResult,
    compute_attribution,
    compute_cost_contribution,
    compute_latency_contribution,
    compute_quality_contribution,
    compute_sensitivity_scores,
    export_attribution,
    get_attribution_ranked,
)


def create_sample_trials() -> list[Any]:
    """Create sample trials for attribution analysis."""

    # Create mock trial results with node metrics
    class MockTrial:
        def __init__(
            self,
            trial_id: str,
            config: dict,
            metrics: dict,
            node_metrics: dict,
        ):
            self.trial_id = trial_id
            self.config = config
            self.metrics = metrics
            self.node_metrics = node_metrics
            self.is_successful = True

    trials = [
        MockTrial(
            "trial_001",
            {"generator.temperature": 0.3, "retriever.top_k": 3},
            {"accuracy": 0.70, "total_cost": 0.025, "latency_p95": 1.1},
            {
                "generator": NodeMetrics(
                    "generator",
                    invocation_count=10,
                    total_latency_ms=800,
                    total_cost=0.020,
                    total_input_tokens=5000,
                    total_output_tokens=1500,
                ),
                "retriever": NodeMetrics(
                    "retriever",
                    invocation_count=10,
                    total_latency_ms=300,
                    total_cost=0.005,
                ),
            },
        ),
        MockTrial(
            "trial_002",
            {"generator.temperature": 0.5, "retriever.top_k": 5},
            {"accuracy": 0.78, "total_cost": 0.032, "latency_p95": 1.3},
            {
                "generator": NodeMetrics(
                    "generator",
                    invocation_count=10,
                    total_latency_ms=900,
                    total_cost=0.025,
                    total_input_tokens=6000,
                    total_output_tokens=2000,
                ),
                "retriever": NodeMetrics(
                    "retriever",
                    invocation_count=10,
                    total_latency_ms=400,
                    total_cost=0.007,
                ),
            },
        ),
        MockTrial(
            "trial_003",
            {"generator.temperature": 0.7, "retriever.top_k": 7},
            {"accuracy": 0.85, "total_cost": 0.040, "latency_p95": 1.5},
            {
                "generator": NodeMetrics(
                    "generator",
                    invocation_count=10,
                    total_latency_ms=1000,
                    total_cost=0.030,
                    total_input_tokens=7000,
                    total_output_tokens=2500,
                ),
                "retriever": NodeMetrics(
                    "retriever",
                    invocation_count=10,
                    total_latency_ms=500,
                    total_cost=0.010,
                ),
            },
        ),
        MockTrial(
            "trial_004",
            {"generator.temperature": 0.9, "retriever.top_k": 10},
            {"accuracy": 0.88, "total_cost": 0.050, "latency_p95": 1.8},
            {
                "generator": NodeMetrics(
                    "generator",
                    invocation_count=10,
                    total_latency_ms=1200,
                    total_cost=0.038,
                    total_input_tokens=8000,
                    total_output_tokens=3000,
                ),
                "retriever": NodeMetrics(
                    "retriever",
                    invocation_count=10,
                    total_latency_ms=600,
                    total_cost=0.012,
                ),
            },
        ),
        MockTrial(
            "trial_005",
            {"generator.temperature": 0.6, "retriever.top_k": 6},
            {"accuracy": 0.82, "total_cost": 0.036, "latency_p95": 1.4},
            {
                "generator": NodeMetrics(
                    "generator",
                    invocation_count=10,
                    total_latency_ms=950,
                    total_cost=0.028,
                    total_input_tokens=6500,
                    total_output_tokens=2200,
                ),
                "retriever": NodeMetrics(
                    "retriever",
                    invocation_count=10,
                    total_latency_ms=450,
                    total_cost=0.008,
                ),
            },
        ),
    ]

    return trials


def example_node_metrics():
    """Example 1: Per-node metrics collection.

    NodeMetrics captures per-component metrics for each run,
    including invocations, latency, cost, and token usage.
    """
    print("=" * 60)
    print("Example 1: Per-Node Metrics Collection")
    print("=" * 60)

    # Create sample node metrics
    generator_metrics = NodeMetrics(
        component_name="generator",
        invocation_count=10,
        total_latency_ms=1000.0,
        total_cost=0.03,
        total_input_tokens=5000,
        total_output_tokens=2000,
        error_count=0,
    )

    retriever_metrics = NodeMetrics(
        component_name="retriever",
        invocation_count=10,
        total_latency_ms=400.0,
        total_cost=0.01,
    )

    print("Generator metrics:")
    print(f"  Invocations: {generator_metrics.invocation_count}")
    print(f"  Total latency: {generator_metrics.total_latency_ms:.0f}ms")
    print(f"  Avg latency: {generator_metrics.avg_latency_ms:.0f}ms")
    print(f"  Total cost: ${generator_metrics.total_cost:.3f}")
    print(f"  Avg cost per call: ${generator_metrics.avg_cost:.4f}")
    print(f"  Input tokens: {generator_metrics.total_input_tokens}")
    print(f"  Output tokens: {generator_metrics.total_output_tokens}")

    print("\nRetriever metrics:")
    print(f"  Invocations: {retriever_metrics.invocation_count}")
    print(f"  Total latency: {retriever_metrics.total_latency_ms:.0f}ms")
    print(f"  Avg latency: {retriever_metrics.avg_latency_ms:.0f}ms")

    # Convert to dict for export
    print("\nAs dict:", generator_metrics.to_dict())


def example_quality_contribution(trials: list):
    """Example 2: Quality contribution analysis.

    Compute each component's contribution to overall quality
    using variance analysis across optimization trials.
    """
    print("\n" + "=" * 60)
    print("Example 2: Quality Contribution Analysis")
    print("=" * 60)

    quality = compute_quality_contribution(trials, target_metric="accuracy")

    print("Quality contribution by component:\n")
    for component, score in sorted(
        quality.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        bar = "#" * int(abs(score) * 20)
        sign = "+" if score > 0 else "-"
        print(f"  {component}: {sign}{abs(score):.3f} {bar}")

    print("\nInterpretation:")
    print("  - Higher scores = greater impact on quality when tuned")
    print("  - Focus optimization efforts on high-impact components")


def example_cost_latency_contribution(trials: list):
    """Example 3: Cost and latency contribution analysis.

    Identify which components consume the most resources.
    """
    print("\n" + "=" * 60)
    print("Example 3: Cost & Latency Contribution")
    print("=" * 60)

    cost = compute_cost_contribution(trials)
    latency = compute_latency_contribution(trials)

    print("Resource contribution by component:\n")
    print(f"  {'Component':<15} {'Cost %':>10} {'Latency %':>12}")
    print("  " + "-" * 40)

    components = set(cost.keys()) | set(latency.keys())
    for component in sorted(components):
        cost_pct = cost.get(component, 0) * 100
        latency_pct = latency.get(component, 0) * 100
        print(f"  {component:<15} {cost_pct:>9.1f}% {latency_pct:>11.1f}%")

    print("\nInsights:")
    # Find highest cost component
    if cost:
        max_cost = max(cost.items(), key=lambda x: x[1])
        print(f"  - {max_cost[0]} dominates cost ({max_cost[1]:.0%})")
    if latency:
        max_latency = max(latency.items(), key=lambda x: x[1])
        print(f"  - {max_latency[0]} dominates latency ({max_latency[1]:.0%})")


def example_sensitivity_analysis(trials: list):
    """Example 4: Parameter sensitivity analysis.

    Identify which parameters have the highest impact on outcomes.
    """
    print("\n" + "=" * 60)
    print("Example 4: Parameter Sensitivity Analysis")
    print("=" * 60)

    sensitivity = compute_sensitivity_scores(trials, target_metric="accuracy")

    print("Parameter sensitivity to accuracy:\n")
    for component, params in sensitivity.items():
        print(f"  {component}:")
        for param, score in sorted(params.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(score * 20)
            print(f"    {param}: {score:.3f} {bar}")

    print("\nInterpretation:")
    print("  - Higher scores = parameter value strongly affects accuracy")
    print("  - Focus tuning on high-sensitivity parameters")


def example_full_attribution(trials: list):
    """Example 5: Full attribution analysis.

    Compute complete attribution including recommendations.
    """
    print("\n" + "=" * 60)
    print("Example 5: Full Attribution Analysis")
    print("=" * 60)

    attribution = compute_attribution(trials, target_metric="accuracy")

    print("Component Attribution Summary:\n")
    for name, attr in attribution.items():
        print(f"  {name}:")
        print(f"    Quality contribution: {attr.quality_contribution:+.3f}")
        print(f"    Cost contribution: {attr.cost_contribution:.1%}")
        print(f"    Latency contribution: {attr.latency_contribution:.1%}")
        print(f"    Most sensitive param: {attr.most_sensitive_param}")
        print(f"    Optimization opportunity: {attr.optimization_opportunity}")
        if attr.recommendation:
            print(f"    Recommendation: {attr.recommendation}")
        print()


def example_ranked_attribution(trials: list):
    """Example 6: Ranked component list.

    Get components ranked by different criteria.
    """
    print("=" * 60)
    print("Example 6: Ranked Attribution")
    print("=" * 60)

    attribution = compute_attribution(trials, target_metric="accuracy")

    # Rank by quality impact
    print("\nRanked by quality impact:")
    ranked = get_attribution_ranked(attribution, by="quality")
    for i, attr in enumerate(ranked):
        print(f"  #{i+1}: {attr.component_name} ({attr.quality_contribution:+.3f})")

    # Rank by optimization opportunity
    print("\nRanked by optimization opportunity:")
    ranked = get_attribution_ranked(attribution, by="opportunity")
    for i, attr in enumerate(ranked):
        print(f"  #{i+1}: {attr.component_name} ({attr.optimization_opportunity})")

    # Rank by cost
    print("\nRanked by cost contribution:")
    ranked = get_attribution_ranked(attribution, by="cost")
    for i, attr in enumerate(ranked):
        print(f"  #{i+1}: {attr.component_name} ({attr.cost_contribution:.1%})")


def example_export_attribution(trials: list):
    """Example 7: Export attribution analysis.

    Export attribution to JSON or CSV for visualization.
    """
    print("\n" + "=" * 60)
    print("Example 7: Export Attribution")
    print("=" * 60)

    attribution = compute_attribution(trials, target_metric="accuracy")

    # Export to dict
    data = export_attribution(attribution, format="dict")
    print("Exported to dict:")
    print(f"  Components: {list(data['components'].keys())}")
    print(f"  Summary: {data['summary']}")

    # Export to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON export
        json_path = Path(tmpdir) / "attribution.json"
        export_attribution(attribution, format="json", output_path=str(json_path))
        print(f"\nJSON export: {json_path.stat().st_size} bytes")

        with open(json_path) as f:
            data = json.load(f)
            print(f"  Structure: {list(data.keys())}")

        # CSV export
        csv_path = Path(tmpdir) / "attribution.csv"
        export_attribution(attribution, format="csv", output_path=str(csv_path))
        print(f"\nCSV export: {csv_path.stat().st_size} bytes")

        with open(csv_path) as f:
            lines = f.readlines()
            print(f"  Rows: {len(lines)} (including header)")
            print(f"  Header: {lines[0].strip()[:50]}...")


def example_optimization_workflow():
    """Example 8: Complete attribution workflow.

    Shows how to use attribution in an optimization context.
    """
    print("\n" + "=" * 60)
    print("Example 8: Attribution in Optimization Workflow")
    print("=" * 60)

    workflow = """
Complete workflow for pipeline optimization with attribution:

1. RUN OPTIMIZATION
   result = await optimizer.optimize(...)

2. COMPUTE ATTRIBUTION
   attribution = compute_attribution(
       trials=result.history,
       target_metric="accuracy"
   )

3. ANALYZE RESULTS
   # Get ranked components by optimization potential
   ranked = get_attribution_ranked(attribution, by="opportunity")

   for attr in ranked:
       if attr.optimization_opportunity == "high":
           print(f"Focus on: {attr.component_name}")
           print(f"  Reason: {attr.recommendation}")

4. IDENTIFY BOTTLENECKS
   # Find cost/latency bottlenecks
   for name, attr in attribution.items():
       if attr.cost_contribution > 0.5:
           print(f"Cost bottleneck: {name}")
       if attr.latency_contribution > 0.5:
           print(f"Latency bottleneck: {name}")

5. GUIDE NEXT ITERATION
   # Focus search space on high-impact parameters
   for name, attr in attribution.items():
       if attr.most_sensitive_param:
           print(f"Tune {attr.most_sensitive_param} for {name}")

6. EXPORT FOR REPORTING
   export_attribution(attribution, format="json", output_path="report.json")
"""
    print(workflow)


def main():
    """Run all Epic 6 examples."""
    print("\n" + "#" * 60)
    print("# Epic 6: Attribution & Insights")
    print("#" * 60)

    # Create sample trials
    trials = create_sample_trials()

    # Run examples
    example_node_metrics()
    example_quality_contribution(trials)
    example_cost_latency_contribution(trials)
    example_sensitivity_analysis(trials)
    example_full_attribution(trials)
    example_ranked_attribution(trials)
    example_export_attribution(trials)
    example_optimization_workflow()

    print("\n" + "#" * 60)
    print("# All Epic 6 examples completed!")
    print("#" * 60)


if __name__ == "__main__":
    main()
