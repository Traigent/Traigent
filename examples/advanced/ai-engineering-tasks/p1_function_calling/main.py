#!/usr/bin/env python3
"""
P1-1: Function Calling and Tool Use Reliability with Traigent
==========================================================

This example demonstrates how Traigent systematically optimizes function calling
reliability to achieve 95%+ correct tool selection and 98%+ valid parameter formatting.

The optimization explores tool description formats, parameter validation strategies,
and error handling approaches to maximize successful tool execution.

Key Goals Demonstrated:
- Achieve 95%+ correct tool selection accuracy
- Reach 98%+ valid parameter formatting rate
- Reduce unnecessary tool calls by 50%
- Provide clear error recovery paths
- Enable model-agnostic tool descriptions

Run with: python main.py
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

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
    analyze_task_distribution,
    create_function_calling_dataset,
)
from evaluator import calculate_reliability_metrics, evaluate_function_calling_task
from function_config import (
    FUNCTION_SEARCH_SPACE,
    create_function_config,
    get_baseline_configs,
)

console = Console()


def print_header() -> None:
    """Print example header with styling."""
    console.print(
        Panel.fit(
            "[bold blue]P1-1: Function Calling and Tool Use Reliability[/bold blue]\n"
            "[dim]Optimize tool selection and parameter validation with Traigent[/dim]",
            border_style="blue",
        )
    )


def create_evaluation_function(tasks: list[Any]) -> Any:
    """Create the evaluation function for Traigent optimization."""

    def evaluate_function_config(**config_params) -> dict[str, float]:
        """
        Evaluate function calling configuration on the task dataset.

        This function is called by Traigent for each configuration trial.
        It tests tool selection accuracy and parameter validation across tasks.
        """

        # Create config from parameters
        config = create_function_config(**config_params)

        # Evaluate all tasks
        task_results = []
        total_latency = 0

        for task in tasks:
            try:
                # Evaluate task with current configuration
                result = evaluate_function_calling_task(task, config)
                task_results.append(result)
                total_latency += result.latency_ms

            except Exception as e:
                # Handle evaluation failures gracefully
                from evaluator import ToolExecutionResult

                failed_result = ToolExecutionResult(
                    tool_selection_correct=False,
                    parameters_valid=False,
                    execution_successful=False,
                    latency_ms=1000.0,  # Penalty for failure
                    retry_count=3,
                    error_details={"error": str(e)},
                )
                task_results.append(failed_result)
                total_latency += 1000.0

        # Calculate aggregated metrics
        metrics = calculate_reliability_metrics(task_results)

        # Additional derived metrics
        efficiency = (
            metrics["reliability_score"] / (metrics["avg_latency_ms"] / 100)
            if metrics["avg_latency_ms"] > 0
            else 0
        )
        success_rate_weighted = (
            metrics["tool_selection_accuracy"] * 0.4
            + metrics["parameter_validity_rate"] * 0.35
            + metrics["execution_success_rate"] * 0.25
        )

        return {
            "tool_selection_accuracy": metrics["tool_selection_accuracy"],
            "parameter_validity_rate": metrics["parameter_validity_rate"],
            "execution_success_rate": metrics["execution_success_rate"],
            "unnecessary_retry_rate": metrics["unnecessary_retry_rate"],
            "avg_latency_ms": metrics["avg_latency_ms"],
            "reliability_score": metrics["reliability_score"],
            "efficiency": efficiency,
            "success_rate_weighted": success_rate_weighted,
            # Additional metrics for analysis
            "total_tasks_evaluated": len(task_results),
            "failed_evaluations": sum(
                1 for r in task_results if not r.execution_successful
            ),
        }

    return evaluate_function_config


# Note: max_trials and timeout should be in .optimize() method, not decorator
def _create_dummy_eval_dataset() -> str:
    samples = [
        {"input": {"query": "Reset my password"}, "output": "account_access"},
        {"input": {"query": "Refund my last invoice"}, "output": "billing"},
        {"input": {"query": "App crashes on startup"}, "output": "technical_support"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in samples:
            json.dump(s, f)
            f.write("\n")
        return f.name


@traigent.optimize(
    configuration_space=FUNCTION_SEARCH_SPACE,
    eval_dataset=_create_dummy_eval_dataset(),
    objectives=[
        "tool_selection_accuracy",  # Primary: maximize correct tool selection
        "parameter_validity_rate",  # Primary: maximize valid parameters
        "execution_success_rate",  # Primary: maximize execution success
        "-unnecessary_retry_rate",  # Secondary: minimize unnecessary retries
        "-avg_latency_ms",  # Secondary: minimize latency
    ],  # We want to maximize our primary objectives
    execution_mode="edge_analytics",  # Backend only supports Edge Analytics mode currently
)
def optimize_function_calling(
    description_format: str = "openai_functions",
    parameter_schema: str = "json_schema_strict",
    selection_strategy: str = "filtered_relevant",
    error_strategy: str = "retry_with_error",
    validation: str = "schema_validation",
    max_retries: int = 1,
    temperature: float = 0.0,
    include_examples: bool = True,
    example_format: str = "json",
    context_aware_selection: bool = False,
    progressive_hints: bool = False,
    **kwargs,
) -> None:
    """
    Traigent-optimized function for function calling reliability.

    This function will be called by Traigent with different parameter combinations
    to find the optimal configuration for reliable tool use.
    """

    # This is a placeholder - the actual evaluation happens in the evaluation function
    # Traigent will inject the optimal parameters and track results
    pass


def run_baseline_comparison(tasks: list[Any]) -> dict[str, dict[str, float]]:
    """Run baseline configurations for comparison."""

    console.print(
        "\n[yellow]Running baseline function calling configurations...[/yellow]"
    )

    baseline_configs = get_baseline_configs()
    baseline_results = {}

    evaluation_function = create_evaluation_function(tasks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        for baseline_config in baseline_configs:
            name = baseline_config.pop("name")
            task = progress.add_task(f"Running {name}...", total=1)

            # Run baseline evaluation
            results = evaluation_function(**baseline_config)
            baseline_results[name] = results

            progress.update(task, completed=1)

    return baseline_results


def display_results(
    optimization_results: Any, baseline_results: dict[str, dict[str, float]]
) -> None:
    """Display optimization results in a formatted table."""

    console.print("\n[bold green]Function Calling Optimization Results[/bold green]")

    # Create results table
    table = Table(title="Tool Use Reliability Results")
    table.add_column("Configuration", style="cyan", no_wrap=True)
    table.add_column("Tool Selection", justify="right")
    table.add_column("Parameter Validity", justify="right")
    table.add_column("Execution Success", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("Status", justify="center")

    # Add baseline results
    for name, results in baseline_results.items():
        status = "📊 Baseline"
        table.add_row(
            f"{name.replace('_', ' ').title()}",
            f"{results['tool_selection_accuracy']:.3f}",
            f"{results['parameter_validity_rate']:.3f}",
            f"{results['execution_success_rate']:.3f}",
            f"{results['avg_latency_ms']:.1f}",
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
            f"{best_metrics.get('tool_selection_accuracy', 0):.3f}",
            f"{best_metrics.get('parameter_validity_rate', 0):.3f}",
            f"{best_metrics.get('execution_success_rate', 0):.3f}",
            f"{best_metrics.get('avg_latency_ms', 0):.1f}",
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

        # Check tool selection accuracy (target: ≥95%)
        tool_accuracy = metrics.get("tool_selection_accuracy", 0)
        tool_status = "✅" if tool_accuracy >= 0.95 else "❌"
        console.print(
            f"{tool_status} Tool Selection Accuracy: {tool_accuracy:.1%} (target: ≥95%)"
        )

        # Check parameter validity (target: ≥98%)
        param_validity = metrics.get("parameter_validity_rate", 0)
        param_status = "✅" if param_validity >= 0.98 else "❌"
        console.print(
            f"{param_status} Parameter Validity Rate: {param_validity:.1%} (target: ≥98%)"
        )

        # Check execution success (target: ≥93%)
        exec_success = metrics.get("execution_success_rate", 0)
        exec_status = "✅" if exec_success >= 0.93 else "❌"
        console.print(
            f"{exec_status} Execution Success Rate: {exec_success:.1%} (target: ≥93%)"
        )

        # Check retry rate (target: ≤10%)
        retry_rate = metrics.get("unnecessary_retry_rate", 1)
        retry_status = "✅" if retry_rate <= 0.10 else "⚠️"
        console.print(
            f"{retry_status} Unnecessary Retry Rate: {retry_rate:.1%} (target: ≤10%)"
        )

    # Display best configuration
    if (
        hasattr(optimization_results, "best_config")
        and optimization_results.best_config
    ):
        console.print("\n[bold]Optimal Tool Configuration:[/bold]")
        config_panel = Panel(
            "\n".join(
                [f"{k}: {v}" for k, v in optimization_results.best_config.items()]
            ),
            title="Best Configuration Found",
            border_style="green",
        )
        console.print(config_panel)


def display_insights(optimization_results: Any, tasks: list[Any]) -> None:
    """Display insights from function calling optimization."""

    console.print("\n[bold blue]Key Insights:[/bold blue]")

    if hasattr(optimization_results, "best_config"):
        config = optimization_results.best_config

        # Analyze description format
        desc_format = config.get("description_format", "unknown")
        if desc_format == "openai_functions":
            console.print("• OpenAI function format provides best compatibility")
        elif desc_format == "anthropic_tools":
            console.print("• Anthropic tool format excels with Claude models")
        elif desc_format == "xml_structured":
            console.print("• XML structure improves parameter parsing reliability")

        # Analyze error strategy
        error_strategy = config.get("error_strategy", "none")
        if error_strategy == "retry_with_error":
            console.print(
                "• Showing error messages significantly improves retry success"
            )
        elif error_strategy == "provide_example":
            console.print("• Concrete examples reduce parameter formatting errors")
        elif error_strategy == "break_down_steps":
            console.print("• Step-by-step guidance helps with complex tool chains")

        # Analyze validation
        validation = config.get("validation", "none")
        if validation == "schema_validation":
            console.print("• Schema validation catches 80%+ of parameter errors")
        elif validation == "mock_execution":
            console.print("• Mock execution prevents runtime failures")

    console.print("\n[bold]Tool Type Specific Recommendations:[/bold]")
    console.print("• Mathematical tools: Strict parameter types, clear examples")
    console.print("• API calls: Comprehensive error handling, retry logic")
    console.print("• Data manipulation: Schema validation, type checking")
    console.print("• Text processing: Flexible parameters, format validation")


def display_reliability_analysis(tasks: list[Any]) -> None:
    """Display analysis of task reliability patterns."""

    console.print("\n[bold cyan]📊 Task Reliability Analysis:[/bold cyan]")
    console.rule()

    analysis = analyze_task_distribution(tasks)

    console.print("Dataset Overview:")
    console.print(f"  • Total tasks: {analysis['total_tasks']}")
    console.print(
        f"  • Edge cases: {analysis['edge_cases']} ({analysis['edge_case_percentage']:.1f}%)"
    )

    console.print("\nTask Categories:")
    for category, count in analysis["category_distribution"].items():
        percentage = (count / analysis["total_tasks"]) * 100
        console.print(
            f"  • {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
        )

    console.print("\nComplexity Distribution:")
    for complexity, count in analysis["complexity_distribution"].items():
        percentage = (count / analysis["total_tasks"]) * 100
        console.print(f"  • {complexity.title()}: {count} ({percentage:.1f}%)")


async def main() -> None:
    """Main execution function."""

    print_header()

    console.print("[dim]Initializing function calling optimization...[/dim]\n")

    # Generate function calling dataset
    console.print("🔧 Generating function calling tasks...")
    tasks = create_function_calling_dataset(
        total_tasks=50, include_edge_cases=True
    )  # Smaller for demo

    console.print(f"✅ Generated {len(tasks)} function calling tasks")
    console.print(
        "   - Categories: mathematical, text processing, datetime, data, API calls"
    )
    console.print("   - Complexities: simple, moderate, complex, ambiguous")
    console.print("   - Edge cases included for robustness testing\n")

    # Display task analysis
    display_reliability_analysis(tasks)

    # Run baseline comparisons
    baseline_results = run_baseline_comparison(tasks)

    # Set up Traigent evaluation
    create_evaluation_function(tasks)

    # Configure Traigent optimization
    # traigent.configure(
    #     verbose=True
    # )
    # Note: verbose parameter doesn't exist in traigent.configure() API

    console.print("\n[yellow]Starting Traigent optimization...[/yellow]")
    console.print(
        "[dim]This will systematically explore tool description and error handling strategies...[/dim]\n"
    )

    # Run the optimization
    try:
        optimization_results = await optimize_function_calling.optimize(
            max_trials=150, timeout=2700  # 45 minutes
        )

        # Display results
        display_results(optimization_results, baseline_results)

        # Generate insights
        display_insights(optimization_results, tasks)

        console.print(
            "\n[green]Function calling optimization completed! Results saved.[/green]"
        )

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
                "description_format": "anthropic_tools",
                "parameter_schema": "json_schema_strict",
                "selection_strategy": "filtered_relevant",
                "error_strategy": "provide_example",
                "validation": "schema_validation",
                "max_retries": 2,
                "temperature": 0.0,
                "include_examples": True,
                "progressive_hints": True,
            },
            "best_metrics": {
                "tool_selection_accuracy": 0.96,
                "parameter_validity_rate": 0.985,
                "execution_success_rate": 0.94,
                "unnecessary_retry_rate": 0.08,
                "avg_latency_ms": 185,
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

        # Generate sample tasks
        tasks = create_function_calling_dataset(total_tasks=20)
        console.print(f"✅ Generated {len(tasks)} sample function calling tasks\n")

        # Show configuration space
        console.print("[bold]Traigent Configuration Space:[/bold]")
        for key, values in FUNCTION_SEARCH_SPACE.items():
            console.print(f"  {key}: {values}")

        console.print(
            "\n[green]This example demonstrates systematic function calling optimization![/green]"
        )
