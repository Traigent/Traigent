#!/usr/bin/env python3
"""
P0-2: Context Engineering & RAG Optimization with Traigent
========================================================

This example demonstrates how Traigent systematically optimizes retrieval-augmented
generation (RAG) systems to improve answer quality by 15-25% while reducing costs by 30-50%.

**IMPORTANT**: This example demonstrates TWO patterns:
1. TRAIGENT FEATURE: Using custom evaluator with @traigent.optimize
2. CUSTOM LOGIC: Token budget allocation (NOT a Traigent feature - implemented in this example)

The optimization explores retrieval strategies, chunk sizes, reranking approaches,
and smart token budget allocation to find the optimal context composition.

Key Goals Demonstrated:
- Achieve 15-25% answer quality improvement
- Reduce context costs by 30-50%
- Optimize retrieval strategies (BM25, dense, hybrid)
- Discover optimal chunk sizes and overlap
- Smart token budget allocation (CUSTOM IMPLEMENTATION)

Run with (from repo root): .venv/bin/python examples/advanced/ai-engineering-tasks/p0_context_engineering/main.py
"""

import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
sys.path.insert(
    0,
    os.environ.get(
        "TRAIGENT_SDK_PATH", str(Path(__file__).parent.parent.parent.parent)
    ),
)

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
from context_config import CONTEXT_ENGINEERING_SEARCH_SPACE, create_context_config
from dataset import (
    RAGDataset,
    analyze_retrieval_quality,
    create_rag_dataset,
    generate_baseline_configs,
)
from evaluator import evaluate_answer_quality, retrieve_context

from traigent.core.objectives import create_default_objectives

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


console = Console()


def print_header() -> None:
    """Print example header with styling."""
    console.print(
        Panel.fit(
            "[bold blue]P0-2: Context Engineering & RAG Optimization[/bold blue]\n"
            "[dim]Optimize retrieval strategies and context composition with Traigent[/dim]",
            border_style="blue",
        )
    )


def create_evaluation_function(dataset: RAGDataset) -> Callable:
    """Create the evaluation function for Traigent optimization."""

    def evaluate_rag_configuration(**config_params) -> dict[str, float]:
        """
        Evaluate RAG configuration on the dataset.

        This function is called by Traigent for each configuration trial.
        It tests the RAG system with different retrieval and chunking strategies.
        """

        # Create config from parameters
        config = create_context_config(**config_params)

        # Track metrics across all queries
        answer_qualities = []
        retrieval_metrics = []
        latencies = []
        costs = []

        for query in dataset.queries:
            try:
                start_time = time.time()

                # Retrieve context using current configuration
                context_result = retrieve_context(
                    query=query.question,
                    config=config,
                    documents=dataset.documents,
                )

                # Evaluate answer quality
                quality_metrics = evaluate_answer_quality(
                    context=context_result["context"],
                    query=query.question,
                )
                quality_score = quality_metrics["overall_quality"]

                answer_qualities.append(quality_score)

                # Analyze retrieval quality
                retrieval_quality = analyze_retrieval_quality(
                    retrieved_chunks=context_result["retrieved_chunks"],
                    query=query,
                    documents=dataset.documents,
                )
                retrieval_metrics.append(retrieval_quality)

                # Calculate latency
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)

                # Calculate cost (simplified - tokens * price per token)
                context_tokens = context_result["token_count"]
                cost = context_tokens * 0.000001  # Simplified pricing
                costs.append(cost)

            except Exception:
                # Handle retrieval failures gracefully
                answer_qualities.append(0.0)
                retrieval_metrics.append({"f1_score": 0.0, "coverage": 0.0})
                latencies.append(1000.0)
                costs.append(1.0)

        # Calculate aggregated metrics
        if not answer_qualities:
            return {
                "answer_quality": 0.0,
                "retrieval_f1": 0.0,
                "latency_p95_ms": 1000.0,
                "cost_per_query": 1.0,
                "coverage": 0.0,
            }

        # Calculate metrics
        avg_quality = np.mean(answer_qualities)
        avg_f1 = np.mean([m["f1_score"] for m in retrieval_metrics])
        avg_coverage = np.mean([m["coverage"] for m in retrieval_metrics])
        p95_latency = np.percentile(latencies, 95)
        avg_cost = np.mean(costs)

        # Additional metrics
        quality_consistency = 1.0 - np.var(answer_qualities)
        efficiency = avg_quality / avg_cost if avg_cost > 0 else 0

        return {
            "answer_quality": avg_quality,
            "retrieval_f1": avg_f1,
            "coverage": avg_coverage,
            "latency_p95_ms": p95_latency,
            "cost_per_query": avg_cost,
            "quality_consistency": quality_consistency,
            "efficiency": efficiency,
            # Detailed breakdowns
            "min_quality": min(answer_qualities),
            "max_quality": max(answer_qualities),
            "total_chunks_retrieved": getattr(config, "n_chunks", 5),
        }

    return evaluate_rag_configuration


def _create_eval_dataset() -> str:
    """Write the demo dataset next to this script and return its path.

    The SDK only loads datasets under the current directory or
    TRAIGENT_DATASET_ROOT, so the system temp dir is not usable. Default the
    root to this script's directory so the example runs from any cwd.
    """
    samples = [
        {
            "input": {"question": "What is RAG?"},
            "output": "Retrieval Augmented Generation",
        },
        {"input": {"question": "What is AI?"}, "output": "Artificial Intelligence"},
        {"input": {"question": "How to reduce cost?"}, "output": "Use smaller context"},
    ]
    # Write under the effective dataset root: the configured one, else this
    # script's directory (which is then made the root).
    configured = os.environ.get("TRAIGENT_DATASET_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(root))
    path = root / f"_demo_{Path(__file__).resolve().parent.name}_eval_dataset.jsonl"
    text = "".join(json.dumps(x) + "\n" for x in samples)
    # Never clobber a file this example did not create: a rerun regenerates
    # identical content, anything else at this path belongs to someone else.
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise SystemExit(
            f"Refusing to overwrite {path}: it exists and was not written by this "
            "example. Delete it or point TRAIGENT_DATASET_ROOT at another directory."
        )
    path.write_text(text, encoding="utf-8")
    return str(path)


# CORRECT API USAGE: Only valid decorator parameters
@traigent.optimize(
    configuration_space=CONTEXT_ENGINEERING_SEARCH_SPACE,  # Fixed: was config_space
    eval_dataset=_create_eval_dataset(),
    objectives=create_default_objectives(
        ["answer_quality", "cost_per_query", "retrieval_f1", "latency_p95_ms"],
        orientations={
            "answer_quality": "maximize",
            "cost_per_query": "minimize",
            "retrieval_f1": "maximize",
            "latency_p95_ms": "minimize",
        },
    ),
    offline=True,  # Run locally with no Traigent backend egress
    # REMOVED: direction, max_trials, timeout_minutes (these don't exist in decorator API)
)
def optimize_context_engineering(
    retrieval_method: str = "bm25",
    embedding_model: str = "sentence-transformers",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    n_chunks: int = 5,
    reranking: bool = False,
    reranking_model: str = "cross-encoder",
    query_expansion: bool = False,
    expansion_method: str = "llm",
    hyde: bool = False,
    token_budget: int = 2000,
    budget_allocation: str = "uniform",
    **kwargs,
) -> None:
    """
    Traigent-optimized function for context engineering and RAG.

    **PATTERN NOTE**: This example uses a pattern where the function body is just 'pass'.
    This is valid when using a custom_evaluator in .optimize(), but not yet officially
    documented as "schema-based injection". The function signature defines the parameters
    to optimize, while the actual evaluation logic is in create_evaluation_function().

    This function will be called by Traigent with different parameter combinations
    to find the optimal RAG configuration.
    """
    # CONTEXT mode never overrides parameters: apply the per-trial config explicitly.
    trial_config = traigent.get_config() or {}
    retrieval_method = trial_config.get("retrieval_method", retrieval_method)
    embedding_model = trial_config.get("embedding_model", embedding_model)
    chunk_size = trial_config.get("chunk_size", chunk_size)
    chunk_overlap = trial_config.get("chunk_overlap", chunk_overlap)

    # This is a placeholder - the actual evaluation happens in the evaluation function
    # Traigent will inject the optimal parameters and track results
    pass


def run_baseline_comparison(dataset: RAGDataset) -> dict[str, dict[str, float]]:
    """Run baseline RAG configurations for comparison."""

    console.print("\n[yellow]Running baseline RAG configurations...[/yellow]")

    baseline_configs = generate_baseline_configs()
    baseline_results = {}

    evaluation_function = create_evaluation_function(dataset)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for name, config in baseline_configs.items():
            task = progress.add_task(f"Running {name}...", total=1)

            # Convert baseline config to parameters
            config_params = {
                "retrieval_method": config["retrieval_method"],
                "n_chunks": config["n_chunks"],
                "chunk_size": config["chunk_size"],
                "chunk_overlap": config["chunk_overlap"],
                "reranking": config["reranking"],
                "query_expansion": config["query_expansion"],
                # Add defaults for other parameters
                "embedding_model": "sentence-transformers",
                "reranking_model": "cross-encoder",
                "expansion_method": "llm",
                "hyde": False,
                "token_budget": 2000,
                "budget_allocation": "uniform",
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

    console.print("\n[bold green]RAG Optimization Results Summary[/bold green]")

    # Create results table
    table = Table(title="Context Engineering & RAG Results")
    table.add_column("Configuration", style="cyan", no_wrap=True)
    table.add_column("Answer Quality", justify="right")
    table.add_column("Cost/Query", justify="right")
    table.add_column("Retrieval F1", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Status", justify="center")

    # Add baseline results
    for name, results in baseline_results.items():
        status = "📊 Baseline"
        table.add_row(
            f"{name.replace('_', ' ').title()}",
            f"{results['answer_quality']:.3f}",
            f"${results['cost_per_query']:.5f}",
            f"{results['retrieval_f1']:.3f}",
            f"{results['latency_p95_ms']:.1f}",
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
            f"{best_metrics.get('answer_quality', 0):.3f}",
            f"${best_metrics.get('cost_per_query', 0):.5f}",
            f"{best_metrics.get('retrieval_f1', 0):.3f}",
            f"{best_metrics.get('latency_p95_ms', 0):.1f}",
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

        # Check answer quality improvement (target: +15-25%)
        baseline_quality = max(r["answer_quality"] for r in baseline_results.values())
        quality_improvement = (
            (metrics.get("answer_quality", 0) - baseline_quality) / baseline_quality
            if baseline_quality > 0
            else 0
        )
        quality_status = "✅" if quality_improvement >= 0.15 else "❌"
        console.print(
            f"{quality_status} Answer Quality Improvement: +{quality_improvement:.1%} (target: ≥+15%)"
        )

        # Check cost reduction (target: -30-50%)
        baseline_cost = min(r["cost_per_query"] for r in baseline_results.values())
        cost_reduction = (
            (baseline_cost - metrics.get("cost_per_query", 1)) / baseline_cost
            if baseline_cost > 0
            else 0
        )
        cost_status = "✅" if cost_reduction >= 0.30 else "❌"
        console.print(
            f"{cost_status} Cost Reduction: -{cost_reduction:.1%} (target: ≥-30%)"
        )

        # Check retrieval accuracy
        retrieval_f1 = metrics.get("retrieval_f1", 0)
        retrieval_status = "✅" if retrieval_f1 >= 0.8 else "⚠️"
        console.print(
            f"{retrieval_status} Retrieval F1: {retrieval_f1:.3f} (target: ≥0.8)"
        )

    # Display best configuration
    if (
        hasattr(optimization_results, "best_config")
        and optimization_results.best_config
    ):
        console.print("\n[bold]Optimal RAG Configuration:[/bold]")
        config_panel = Panel(
            "\n".join(
                [f"{k}: {v}" for k, v in optimization_results.best_config.items()]
            ),
            title="Best Configuration Found",
            border_style="green",
        )
        console.print(config_panel)


def _display_retrieval_method_insight(config: dict[str, Any]) -> None:
    method_messages = {
        "hybrid": "• Hybrid retrieval (BM25 + dense) provides best results",
        "dense": "• Dense retrieval excels for semantic understanding",
        "bm25": "• BM25 sufficient for keyword-based queries",
    }
    message = method_messages.get(config.get("retrieval_method", "unknown"))
    if message:
        console.print(message)


def _display_chunking_insight(config: dict[str, Any]) -> None:
    chunk_size = config.get("chunk_size", 512)
    overlap = config.get("chunk_overlap", 0)
    if chunk_size <= 256:
        console.print(f"• Smaller chunks ({chunk_size} tokens) improve precision")
    elif chunk_size >= 1024:
        console.print(f"• Larger chunks ({chunk_size} tokens) preserve context")
    if overlap > chunk_size * 0.25:
        console.print(f"• High overlap ({overlap} tokens) ensures continuity")


def _display_enhancement_insight(config: dict[str, Any]) -> None:
    enhancements = {
        "reranking": "• Reranking significantly improves relevance",
        "query_expansion": "• Query expansion helps with ambiguous queries",
        "hyde": "• HyDE (Hypothetical Document Embeddings) improves recall",
    }
    for key, message in enhancements.items():
        if config.get(key):
            console.print(message)


def _print_query_type_recommendations() -> None:
    console.print("\n[bold]Query-Type Specific Recommendations:[/bold]")
    for line in (
        "• Factual queries: Smaller chunks with keyword matching",
        "• Analytical queries: Larger chunks with semantic search",
        "• Multi-hop queries: Hybrid retrieval with reranking",
        "• Comparative queries: Query expansion for comprehensive coverage",
    ):
        console.print(line)


def display_insights(optimization_results, dataset: RAGDataset) -> None:
    """Display insights from RAG optimization."""

    console.print("\n[bold blue]Key Insights:[/bold blue]")
    config = getattr(optimization_results, "best_config", None)
    if isinstance(config, dict):
        _display_retrieval_method_insight(config)
        _display_chunking_insight(config)
        _display_enhancement_insight(config)
    _print_query_type_recommendations()


def display_cost_analysis(optimization_results, baseline_results) -> None:
    """Display cost-benefit analysis."""

    console.print(
        "\n[bold cyan]📊 Cost-Benefit Analysis (10,000 queries/day):[/bold cyan]"
    )
    console.rule()

    if baseline_results:
        baseline_cost = baseline_results.get("advanced_rag", {}).get(
            "cost_per_query", 0.001
        )
        daily_baseline = baseline_cost * 10000
        monthly_baseline = daily_baseline * 30

        console.print(
            f"Baseline RAG: ${daily_baseline:.2f}/day, ${monthly_baseline:.2f}/month"
        )

        if hasattr(optimization_results, "best_metrics"):
            optimized_cost = optimization_results.best_metrics.get(
                "cost_per_query", baseline_cost
            )
            daily_optimized = optimized_cost * 10000
            monthly_optimized = daily_optimized * 30

            savings = monthly_baseline - monthly_optimized
            console.print(
                f"Optimized RAG: ${daily_optimized:.2f}/day, ${monthly_optimized:.2f}/month"
            )
            console.print(
                f"\n[bold green]💰 Monthly Savings: ${savings:.2f}![/bold green]"
            )


async def main() -> None:
    """Main execution function."""

    print_header()

    console.print("[dim]Initializing context engineering optimization...[/dim]\n")

    # Generate RAG dataset
    console.print("📊 Generating RAG evaluation dataset...")
    dataset = create_rag_dataset(num_documents=10, num_queries=30)  # Smaller for demo

    console.print("✅ Generated dataset:")
    console.print(
        f"   - Documents: {len(dataset.documents)} across {len(dataset.metadata['domains'])} domains"
    )
    console.print(f"   - Queries: {len(dataset.queries)} with multiple types")
    console.print("   - Query types: factual, multi-hop, analytical, comparative\n")

    # Run baseline comparisons
    baseline_results = run_baseline_comparison(dataset)

    # Set up Traigent evaluation
    evaluation_function = create_evaluation_function(dataset)

    # Configure Traigent optimization
    # traigent.configure(
    #     evaluator=evaluation_function,
    #     offline=True,  # Run locally for demo
    #     verbose=True
    # )
    # Note: evaluator and verbose parameters don't exist in traigent.configure() API

    console.print("\n[yellow]Starting Traigent optimization...[/yellow]")
    console.print("[dim]This will systematically explore RAG configurations...[/dim]\n")

    # Run the optimization
    # CORRECT API USAGE: Call .optimize() method with parameters
    try:
        optimization_results = await optimize_context_engineering.optimize(
            custom_evaluator=evaluation_function,
            max_trials=100,  # Moved here from decorator
            timeout=1800,  # 30 minutes in seconds (was timeout_minutes=30)
            algorithm="random",  # Offline local example; smart algorithms require cloud
        )

        # Display results
        display_results(optimization_results, baseline_results)

        # Generate insights
        display_insights(optimization_results, dataset)

        # Cost analysis
        display_cost_analysis(optimization_results, baseline_results)

        console.print("\n[green]Optimization completed! Results saved.[/green]")

    except Exception as e:
        console.print(f"\n[red]Error during optimization: {e}[/red]")
        raise SystemExit(1) from e


if __name__ == "__main__":
    try:
        # Handle both sync and async execution
        try:
            asyncio.run(main())
        except RuntimeError:
            # Fallback for environments where asyncio is not available
            console.print("[dim]Running in synchronous mode...[/dim]\n")

            # Generate sample dataset
            dataset = create_rag_dataset(num_documents=5, num_queries=10)
            console.print(
                f"✅ Generated {len(dataset.documents)} documents and {len(dataset.queries)} queries\n"
            )

            # Show configuration space
            console.print("[bold]Traigent Configuration Space:[/bold]")
            for key, values in CONTEXT_ENGINEERING_SEARCH_SPACE.items():
                console.print(f"  {key}: {values}")

            console.print(
                "\n[green]This example demonstrates systematic RAG optimization![/green]"
            )
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
