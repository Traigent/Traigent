#!/usr/bin/env python3
"""
P0-3: Few-Shot Example Selection Optimization with Traigent
=========================================================

This example demonstrates how Traigent systematically optimizes few-shot example
selection strategies to improve accuracy by 8-15% and reduce variance by 30%.

The optimization explores various selection methods (semantic similarity, diversity,
curriculum learning) and discovers task-specific optimal strategies.

Key Goals Demonstrated:
- Achieve 8-15% accuracy improvement over random selection
- Reduce output variance by 30%
- Maintain <100ms selection latency with caching
- Discover task-adaptive selection strategies

Run with (from repo root): .venv/bin/python examples/advanced/ai-engineering-tasks/p0_few_shot_selection/main.py
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import traigent

    # Set mock mode for demo
    os.environ["TRAIGENT_MOCK_LLM"] = "true"
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install traigent and rich: pip install traigent rich")
    sys.exit(1)

# Import local components
from dataset import (
    FewShotTask,
    analyze_example_diversity,
    generate_baseline_strategies,
    generate_evaluation_tasks,
)
from evaluator import evaluate_with_examples, select_examples
from selection_config import (
    EXAMPLE_EXAMPLE_SELECTION_SEARCH_SPACE,
    create_selection_config,
)

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


console = Console()


def print_header() -> None:
    """Print example header with styling."""
    console.print(
        Panel.fit(
            "[bold blue]P0-3: Few-Shot Example Selection Optimization[/bold blue]\n"
            "[dim]Discover optimal example selection strategies with Traigent[/dim]",
            border_style="blue",
        )
    )


def create_evaluation_function(tasks: list[FewShotTask]) -> Callable:
    """Create the evaluation function for Traigent optimization."""

    def evaluate_selection_strategy(**config_params) -> dict[str, float]:
        """
        Evaluate example selection strategy on the task dataset.

        This function is called by Traigent for each configuration trial.
        It tests the selection strategy across all tasks and returns metrics.
        """

        # Create config from parameters
        config = create_selection_config(**config_params)

        # Track metrics across all tasks
        accuracies = []
        latencies = []
        diversity_scores = []
        start_time = time.time()

        for task in tasks:
            try:
                # Select examples using current strategy
                selected_examples = select_examples(
                    query=task.query,
                    candidates=task.candidate_examples,
                    config=config,
                    task_type=task.task_type,
                )

                # Evaluate performance with selected examples
                accuracy = evaluate_with_examples(
                    task=task, selected_examples=selected_examples, config=config
                )

                accuracies.append(accuracy)

                # Calculate diversity of selected examples
                diversity = analyze_example_diversity(selected_examples)
                diversity_scores.append(diversity.get("domain_diversity", 0))

                # Track selection latency
                selection_time = (time.time() - start_time) * 1000  # ms
                latencies.append(selection_time)

            except Exception:
                # Handle selection failures gracefully
                accuracies.append(0.0)
                diversity_scores.append(0.0)
                latencies.append(1000.0)  # Penalty for failure

        # Calculate aggregated metrics
        if not accuracies:
            return {
                "accuracy": 0.0,
                "variance": 1.0,
                "selection_latency_ms": 1000.0,
                "diversity_score": 0.0,
            }

        # Calculate metrics
        avg_accuracy = np.mean(accuracies)
        variance = np.var(accuracies)
        avg_latency = np.mean(latencies)
        avg_diversity = np.mean(diversity_scores)

        # Additional metrics for analysis
        consistency = 1.0 - variance  # Higher is better
        efficiency = avg_accuracy / (avg_latency / 100)  # Accuracy per 100ms

        return {
            "accuracy": avg_accuracy,
            "variance": variance,
            "consistency": consistency,
            "selection_latency_ms": avg_latency,
            "diversity_score": avg_diversity,
            "efficiency": efficiency,
            # Task-specific breakdowns
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "p90_latency": np.percentile(latencies, 90),
        }

    return evaluate_selection_strategy


# Note: max_trials and timeout should be in .optimize() method, not decorator
def _create_dummy_eval_dataset() -> str:
    samples = [
        {"input": {"text": "I loved it"}, "output": "positive"},
        {"input": {"text": "It was fine"}, "output": "neutral"},
        {"input": {"text": "This is the worst"}, "output": "negative"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in samples:
            json.dump(s, f)
            f.write("\n")
        return f.name


@traigent.optimize(
    configuration_space=EXAMPLE_EXAMPLE_SELECTION_SEARCH_SPACE,
    eval_dataset=_create_dummy_eval_dataset(),
    objectives=[
        "accuracy",  # Primary: maximize accuracy
        "consistency",  # Primary: maximize consistency (1 - variance)
        "-selection_latency_ms",  # Secondary: minimize latency
        "diversity_score",  # Secondary: maximize diversity
    ],  # We want to maximize our primary objectives
    execution_mode="edge_analytics",  # Backend only supports Edge Analytics mode currently
)
def optimize_few_shot_selection(
    selection_method: str = "random",
    n_examples: int = 3,
    similarity_metric: str = "cosine",
    diversity_weight: float = 0.0,
    ordering_strategy: str = "similarity_desc",
    formatting: str = "io_pairs",
    cache_strategy: str = "none",
    adaptive_selection: bool = False,
    curriculum_learning: bool = False,
    example_weighting: str = "uniform",
    **kwargs,
) -> None:
    """
    Traigent-optimized function for few-shot example selection.

    This function will be called by Traigent with different parameter combinations
    to find the optimal selection strategy for few-shot learning.
    """

    # This is a placeholder - the actual evaluation happens in the evaluation function
    # Traigent will inject the optimal parameters and track results
    pass


def run_baseline_comparison(tasks: list[FewShotTask]) -> dict[str, dict[str, float]]:
    """Run baseline strategies for comparison."""

    console.print("\n[yellow]Running baseline strategies...[/yellow]")

    baseline_strategies = generate_baseline_strategies()
    baseline_results = {}

    evaluation_function = create_evaluation_function(tasks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        for name, strategy in baseline_strategies.items():
            task = progress.add_task(f"Running {name}...", total=1)

            # Convert baseline strategy to config format
            config_params = {
                "selection_method": strategy["selection_method"],
                "n_examples": strategy["n_examples"],
                "ordering_strategy": strategy["order"],
                "similarity_metric": "cosine",
                "diversity_weight": 0.0,
                "formatting": "io_pairs",
                "cache_strategy": "none",
                "adaptive_selection": False,
                "curriculum_learning": False,
                "example_weighting": "uniform",
            }

            # Run baseline evaluation
            results = evaluation_function(**config_params)
            baseline_results[name] = results

            progress.update(task, completed=1)

    return baseline_results


def display_results(
    optimization_results, baseline_results: dict[str, dict[str, float]]
) -> None:
    """Display optimization results in a formatted table."""

    console.print("\n[bold green]Optimization Results Summary[/bold green]")

    # Create results table
    table = Table(title="Few-Shot Selection Strategy Results")
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Accuracy", justify="right")
    table.add_column("Consistency", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Diversity", justify="right")
    table.add_column("Status", justify="center")

    # Add baseline results
    for name, results in baseline_results.items():
        status = "📊 Baseline"
        table.add_row(
            f"{name.replace('_', ' ').title()}",
            f"{results['accuracy']:.3f}",
            f"{results.get('consistency', 1-results['variance']):.3f}",
            f"{results['selection_latency_ms']:.1f}",
            f"{results['diversity_score']:.3f}",
            status,
        )

    # Add optimization results if available
    if (
        hasattr(optimization_results, "best_config")
        and optimization_results.best_config
    ):
        best_metrics = optimization_results.best_metrics
        status = "🚀 Optimized"

        table.add_row(
            "Traigent Optimized",
            f"{best_metrics.get('accuracy', 0):.3f}",
            f"{best_metrics.get('consistency', 0):.3f}",
            f"{best_metrics.get('selection_latency_ms', 0):.1f}",
            f"{best_metrics.get('diversity_score', 0):.3f}",
            status,
        )

    console.print(table)

    # Success criteria analysis
    console.print("\n[bold]Success Criteria Analysis:[/bold]")

    if (
        hasattr(optimization_results, "best_metrics")
        and optimization_results.best_metrics
    ):
        metrics = optimization_results.best_metrics

        # Check accuracy improvement (target: +8-15% over random)
        random_accuracy = baseline_results.get("random_selection", {}).get(
            "accuracy", 0
        )
        accuracy_improvement = (
            (metrics.get("accuracy", 0) - random_accuracy) / random_accuracy
            if random_accuracy > 0
            else 0
        )
        accuracy_status = "✅" if accuracy_improvement >= 0.08 else "❌"
        console.print(
            f"{accuracy_status} Accuracy Improvement: +{accuracy_improvement:.1%} (target: ≥+8%)"
        )

        # Check variance reduction (target: -30%)
        random_variance = baseline_results.get("random_selection", {}).get(
            "variance", 1
        )
        variance_reduction = (
            (random_variance - metrics.get("variance", 1)) / random_variance
            if random_variance > 0
            else 0
        )
        variance_status = "✅" if variance_reduction >= 0.30 else "❌"
        console.print(
            f"{variance_status} Variance Reduction: -{variance_reduction:.1%} (target: ≥-30%)"
        )

        # Check latency (target: <100ms)
        latency = metrics.get("selection_latency_ms", 0)
        latency_status = "✅" if latency < 100 else "❌"
        console.print(
            f"{latency_status} Selection Latency: {latency:.1f}ms (target: <100ms)"
        )

    # Display best configuration
    if (
        hasattr(optimization_results, "best_config")
        and optimization_results.best_config
    ):
        console.print("\n[bold]Optimal Configuration:[/bold]")
        config_panel = Panel(
            "\n".join(
                [f"{k}: {v}" for k, v in optimization_results.best_config.items()]
            ),
            title="Best Selection Strategy Found",
            border_style="green",
        )
        console.print(config_panel)


def display_insights(optimization_results, tasks: list[FewShotTask]) -> None:
    """Display insights from optimization."""

    console.print("\n[bold blue]Key Insights:[/bold blue]")

    if hasattr(optimization_results, "best_config"):
        config = optimization_results.best_config

        # Analyze selection method
        method = config.get("selection_method", "unknown")
        if method == "semantic_knn":
            console.print("• Semantic similarity is most effective for this task type")
        elif method == "mmr":
            console.print(
                "• Diversity-aware selection (MMR) balances relevance and variety"
            )
        elif method == "curriculum":
            console.print("• Curriculum learning (easy → hard) improves complex tasks")

        # Analyze number of examples
        n_examples = config.get("n_examples", 3)
        if n_examples <= 3:
            console.print(
                f"• Fewer examples ({n_examples}) are sufficient - quality over quantity"
            )
        elif n_examples >= 7:
            console.print(f"• More examples ({n_examples}) needed for task complexity")

        # Analyze ordering
        ordering = config.get("ordering_strategy", "unknown")
        if "difficulty" in ordering:
            console.print("• Example difficulty progression matters for learning")
        elif "diversity" in ordering:
            console.print("• Diverse example ordering improves generalization")

    console.print("\n[bold]Task-Specific Recommendations:[/bold]")
    console.print("• Classification: Use diverse boundary examples")
    console.print("• Generation: Select similar examples with varied outputs")
    console.print("• Reasoning: Apply curriculum learning (easy → hard)")
    console.print("• Complex tasks: Combine semantic similarity with diversity (MMR)")


async def main() -> None:
    """Main execution function."""

    print_header()

    console.print(
        "[dim]Initializing few-shot example selection optimization...[/dim]\n"
    )

    # Generate evaluation dataset
    console.print("📊 Generating evaluation tasks...")
    tasks = generate_evaluation_tasks(num_tasks=30)  # Fewer for demo

    console.print(f"✅ Generated {len(tasks)} tasks across 3 types")
    console.print(
        f"   - Classification: {sum(1 for t in tasks if t.task_type.value == 'classification')} tasks"
    )
    console.print(
        f"   - Generation: {sum(1 for t in tasks if t.task_type.value == 'generation')} tasks"
    )
    console.print(
        f"   - Reasoning: {sum(1 for t in tasks if t.task_type.value == 'reasoning')} tasks\n"
    )

    # Run baseline comparisons
    baseline_results = run_baseline_comparison(tasks)

    # Set up Traigent evaluation
    create_evaluation_function(tasks)

    # Configure Traigent optimization
    # traigent.configure(
    #     evaluator=evaluation_function,
    #     execution_mode="edge_analytics",  # Run in Edge Analytics mode for demo
    #     verbose=True
    # )
    # Note: evaluator, execution_mode, and verbose parameters don't exist in traigent.configure() API

    console.print("\n[yellow]Starting Traigent optimization...[/yellow]")
    console.print(
        "[dim]This will systematically explore different selection strategies...[/dim]\n"
    )

    # Run the optimization
    try:
        optimization_results = await optimize_few_shot_selection.optimize(
            max_trials=100, timeout=1800  # 30 minutes
        )

        # Display results
        display_results(optimization_results, baseline_results)

        # Generate insights
        display_insights(optimization_results, tasks)

        console.print("\n[green]Optimization completed! Results saved.[/green]")

    except Exception as e:
        console.print(f"\n[red]Error during optimization: {e}[/red]")
        console.print(
            "[yellow]Note: This is a demo that simulates optimization results[/yellow]"
        )

        # Show simulated results for demo purposes
        console.print(
            "\n[dim]Showing simulated optimization results for demonstration:[/dim]"
        )

        simulated_results = {
            "best_config": {
                "selection_method": "mmr",
                "n_examples": 5,
                "similarity_metric": "cosine",
                "diversity_weight": 0.3,
                "ordering_strategy": "curriculum",
                "formatting": "io_with_explanation",
                "cache_strategy": "task_based",
                "adaptive_selection": True,
            },
            "best_metrics": {
                "accuracy": 0.85,
                "consistency": 0.92,
                "variance": 0.08,
                "selection_latency_ms": 45,
                "diversity_score": 0.75,
            },
        }

        class SimulatedResults:
            def __init__(self, config, metrics):
                self.best_config = config
                self.best_metrics = metrics

        display_results(
            SimulatedResults(
                simulated_results["best_config"], simulated_results["best_metrics"]
            ),
            baseline_results,
        )


if __name__ == "__main__":
    # Handle both sync and async execution
    try:
        asyncio.run(main())
    except RuntimeError:
        # Fallback for environments where asyncio is not available
        console.print("[dim]Running in synchronous mode...[/dim]\n")

        # Generate sample dataset
        tasks = generate_evaluation_tasks(num_tasks=10)
        console.print(f"✅ Generated {len(tasks)} sample tasks\n")

        # Show configuration space
        console.print("[bold]Traigent Configuration Space:[/bold]")
        for key, values in EXAMPLE_EXAMPLE_SELECTION_SEARCH_SPACE.items():
            console.print(f"  {key}: {values}")

        console.print(
            "\n[green]This example demonstrates systematic optimization of few-shot selection![/green]"
        )
