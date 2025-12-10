"""Enhanced CLI interface for TraiGent SDK."""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from traigent import get_version_info
from traigent.api.functions import get_available_strategies
from traigent.cli.auth_commands import auth
from traigent.cli.hooks_commands import hooks
from traigent.cli.local_commands import register_edge_analytics_commands
from traigent.utils.logging import setup_logging
from traigent.utils.persistence import PersistenceManager
from traigent.utils.validation import OptimizationValidator
from traigent.visualization.plots import create_quick_plot

console = Console()
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_workspace_path(path: Path, description: str) -> Path:
    """Resolve a path and ensure it lives within the repository workspace."""
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise click.ClickException(
            f"{description} must reside within the Traigent workspace ({WORKSPACE_ROOT})"
        ) from exc
    return resolved


@click.group()
@click.version_option(
    version=get_version_info()["version"],
    prog_name="TraiGent",
    message="%(prog)s %(version)s",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging (errors only)")
def cli(verbose: bool, debug: bool, quiet: bool) -> None:
    """TraiGent SDK - Open-source LLM optimization toolkit.

    TraiGent makes it effortless to optimize your LLM applications with
    a simple decorator.

    Examples:
        traigent info              # Show version information
        traigent algorithms        # List available algorithms
        traigent --verbose info    # Verbose output
        traigent --quiet info      # Suppress logs
    """
    if quiet:
        setup_logging("ERROR")
    elif debug:
        setup_logging("DEBUG")
    elif verbose:
        setup_logging("INFO")
    else:
        setup_logging("WARNING")


@cli.command()
def info() -> None:
    """Show TraiGent SDK version and system information."""
    version_info = get_version_info()

    console.print("\n[bold blue]TraiGent SDK[/bold blue]")
    console.print(f"Version: [green]{version_info['version']}[/green]")
    console.print(f"Python: {version_info['python_version'].split()[0]}")
    console.print(f"Platform: {version_info['platform']}")

    # Features table
    console.print("\n[bold]Features:[/bold]")
    features_table = Table(show_header=True, header_style="bold magenta")
    features_table.add_column("Feature")
    features_table.add_column("Status")

    for feature, enabled in version_info["features"].items():
        status = "[green]✓ Enabled[/green]" if enabled else "[red]✗ Disabled[/red]"
        features_table.add_row(feature.replace("_", " ").title(), status)

    console.print(features_table)

    # Integrations table
    console.print("\n[bold]Integrations:[/bold]")
    integrations_table = Table(show_header=True, header_style="bold magenta")
    integrations_table.add_column("Framework")
    integrations_table.add_column("Status")

    for framework, enabled in version_info["integrations"].items():
        status = (
            "[green]✓ Available[/green]" if enabled else "[yellow]○ Planned[/yellow]"
        )
        integrations_table.add_row(framework.title(), status)

    console.print(integrations_table)
    console.print()


@cli.command()
def algorithms() -> None:
    """List available optimization algorithms."""
    strategies = get_available_strategies()

    console.print("\n[bold blue]Available Optimization Algorithms[/bold blue]\n")

    for name, info in strategies.items():
        console.print(f"[bold green]{info['name']}[/bold green] ([cyan]{name}[/cyan])")
        console.print(f"  {info['description']}")
        console.print(f"  Best for: {info['best_for']}")

        # Capabilities
        capabilities = []
        if info["supports_continuous"]:
            capabilities.append("continuous")
        if info["supports_categorical"]:
            capabilities.append("categorical")
        if info["deterministic"]:
            capabilities.append("deterministic")

        console.print(f"  Supports: {', '.join(capabilities)}")

        # Parameters
        if info["parameters"]:
            console.print("  Parameters:")
            for param, desc in info["parameters"].items():
                console.print(f"    • {param}: {desc}")

        console.print()


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--algorithm", "-a", default="grid", help="Optimization algorithm to use")
@click.option(
    "--max-trials", "-n", type=int, default=10, help="Maximum number of trials"
)
@click.option(
    "--max-total-examples",
    type=int,
    help="Global sample budget across all trials",
)
@click.option(
    "--samples-include-pruned/--samples-exclude-pruned",
    default=True,
    show_default=True,
    help="Include pruned trials when applying the sample budget",
)
@click.option("--timeout", "-t", type=float, help="Optimization timeout in seconds")
@click.option("--function", "-f", help="Specific function name to optimize")
@click.option(
    "--output-dir", "-o", default=".traigent", help="Directory to save results"
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option(
    "--tvl-spec",
    type=click.Path(exists=True),
    help="Load configuration from a TVL spec",
)
@click.option("--tvl-environment", help="TVL environment overlay to apply")
def optimize(
    file_path: str,
    algorithm: str,
    max_trials: int,
    max_total_examples: int | None,
    samples_include_pruned: bool,
    timeout: float,
    function: str,
    output_dir: str,
    verbose: bool,
    tvl_spec: str | None,
    tvl_environment: str | None,
) -> None:
    """Optimize a Python file containing @traigent.optimize decorators.

    This command will execute the optimization for functions decorated with
    @traigent.optimize in the specified Python file.

    Args:
        file_path: Path to Python file containing optimizable functions
    """
    import asyncio
    import importlib.util
    import inspect
    import sys
    from pathlib import Path

    console.print(f"\n[bold yellow]Running optimization on {file_path}[/bold yellow]")
    console.print(f"Algorithm: {algorithm}")
    console.print(f"Max trials: {max_trials}")
    if timeout:
        console.print(f"Timeout: {timeout}s")
    if max_total_examples is not None:
        status = "including" if samples_include_pruned else "excluding"
        console.print(
            f"Sample budget: {max_total_examples} examples ({status} pruned trials)"
        )
    elif not samples_include_pruned:
        console.print("Pruned trials will be excluded from the sample budget.")

    # Load the Python file as a module
    file_path_obj = _resolve_workspace_path(Path(file_path), "File path")

    # Validate output directory before loading the module
    output_dir_candidate = Path(output_dir)
    if not output_dir_candidate.is_absolute():
        output_dir_candidate = WORKSPACE_ROOT / output_dir_candidate
    output_dir_resolved = _resolve_workspace_path(
        output_dir_candidate, "Output directory"
    )

    # Add parent directory to Python path temporarily
    parent_dir = file_path_obj.parent
    sys.path.insert(0, str(parent_dir))

    try:
        # Import the module
        spec = importlib.util.spec_from_file_location("user_module", file_path_obj)
        if spec is None or spec.loader is None:
            console.print("[red]Error: Failed to load Python file[/red]")
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules["user_module"] = module
        spec.loader.exec_module(module)

        # Find functions with optimize method (decorated with @traigent.optimize)
        # Note: @traigent.optimize returns an OptimizedFunction instance, not a raw function
        optimizable_functions = []
        for name, obj in inspect.getmembers(module):
            # Skip private/internal attributes and modules
            if name.startswith("_") or inspect.ismodule(obj):
                continue

            # Check if it's decorated with @traigent.optimize
            # The decorator returns an OptimizedFunction instance with an optimize method
            is_optimized = False
            if inspect.isfunction(obj) and hasattr(obj, "optimize"):
                is_optimized = True
            elif (
                hasattr(obj, "__class__")
                and obj.__class__.__name__ == "OptimizedFunction"
            ):
                is_optimized = True
            elif hasattr(obj, "optimize") and callable(getattr(obj, "optimize", None)):
                # Additional check for objects with optimize method
                if hasattr(obj, "func") or callable(obj):
                    is_optimized = True

            if is_optimized:
                if function is None or name == function:
                    optimizable_functions.append((name, obj))

        if not optimizable_functions:
            if function:
                console.print(
                    f"[red]No optimizable function named '{function}' found[/red]"
                )
            else:
                console.print(
                    "[red]No functions with @traigent.optimize decorator found[/red]"
                )
            console.print(
                "\nMake sure your functions are decorated with @traigent.optimize"
            )
            return

        console.print(
            f"\n[green]Found {len(optimizable_functions)} optimizable function(s)[/green]"
        )

        # Initialize persistence manager
        persistence = PersistenceManager(str(output_dir_resolved))

        # Run optimization for each function
        async def run_optimizations() -> list[Any]:
            results = []
            for func_name, func in optimizable_functions:
                console.print(f"\n[bold blue]Optimizing: {func_name}[/bold blue]")

                try:
                    # Run optimization
                    optimize_kwargs = {
                        "algorithm": algorithm,
                        "max_trials": max_trials,
                        "timeout": timeout,
                        "verbose": verbose,
                        "samples_include_pruned": samples_include_pruned,
                    }
                    if max_total_examples is not None:
                        optimize_kwargs["max_total_examples"] = max_total_examples
                    if tvl_spec:
                        optimize_kwargs["tvl_spec"] = tvl_spec
                    if tvl_environment:
                        optimize_kwargs["tvl_environment"] = tvl_environment

                    result = await func.optimize(**optimize_kwargs)

                    # Display results
                    console.print(
                        f"✅ Best score: [green]{result.best_score:.3f}[/green]"
                    )
                    console.print(f"   Best config: {result.best_config}")
                    console.print(f"   Total trials: {len(result.trials)}")
                    console.print(
                        f"   Successful trials: {len(result.successful_trials)}"
                    )

                    # Save result
                    result_name = f"{func_name}_{algorithm}_{max_trials}"
                    persistence.save_result(result, result_name)
                    console.print(f"   Results saved as: {result_name}")

                    results.append((func_name, result))

                except Exception as e:
                    console.print(f"❌ [red]Error optimizing {func_name}: {e}[/red]")
                    if verbose:
                        import traceback

                        console.print(traceback.format_exc())

            return results

        # Run all optimizations
        results = asyncio.run(run_optimizations())

        # Summary
        console.print("\n[bold green]Optimization Complete![/bold green]")
        console.print(f"Total functions optimized: {len(results)}")
        console.print(f"Results saved to: {output_dir_resolved}")

        # Display summary table
        if results:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Function", style="cyan")
            table.add_column("Best Score", justify="right")
            table.add_column("Best Config", style="green")
            table.add_column("Trials", justify="right")

            for func_name, result in results:
                # Truncate config if too long
                config_str = str(result.best_config)
                if len(config_str) > 50:
                    config_str = config_str[:47] + "..."

                table.add_row(
                    func_name,
                    f"{result.best_score:.3f}",
                    config_str,
                    str(len(result.trials)),
                )

            console.print("\n[bold]Summary:[/bold]")
            console.print(table)

    except Exception as e:
        console.print(f"[red]Error loading file: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
    finally:
        # Remove from Python path
        if str(parent_dir) in sys.path:
            sys.path.remove(str(parent_dir))


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option(
    "--objectives",
    "-o",
    multiple=True,
    help="Objectives to validate (e.g., accuracy, latency)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed validation output")
def validate(dataset_path: str, objectives: tuple[str, ...], verbose: bool) -> None:
    """Validate dataset format and optimization configuration."""
    console.print(f"\n[bold blue]Validating Dataset: {dataset_path}[/bold blue]\n")

    from traigent.utils.validation import Validators

    # Validate dataset path
    path_result = Validators.validate_dataset(dataset_path)

    if path_result.is_valid:
        console.print("✅ [green]Dataset path validation passed[/green]")
    else:
        console.print("❌ [red]Dataset path validation failed[/red]")
        console.print(path_result.get_feedback())
        return

    # Validate dataset content (same as path validation)
    content_result = path_result

    if content_result.is_valid:
        console.print("✅ [green]Dataset content validation passed[/green]")
    else:
        console.print("❌ [red]Dataset content validation failed[/red]")

    if verbose or not content_result.is_valid:
        console.print(content_result.get_feedback())

    # Validate objectives if provided
    if objectives:
        from traigent.utils.validation import Validators

        obj_result = Validators.validate_objectives(list(objectives))

        if obj_result.is_valid:
            console.print("✅ [green]Objectives validation passed[/green]")
        else:
            console.print("❌ [red]Objectives validation failed[/red]")
            console.print(obj_result.get_feedback())


@cli.command()
@click.option(
    "--storage-dir", "-d", default=".traigent", help="TraiGent storage directory"
)
def results(storage_dir: str) -> None:
    """List and manage optimization results."""
    console.print("\n[bold blue]TraiGent Optimization Results[/bold blue]\n")

    persistence = PersistenceManager(storage_dir)
    all_results = persistence.list_results()

    if not all_results:
        console.print("[yellow]No optimization results found[/yellow]")
        console.print(f"Results are stored in: {persistence.base_dir}")
        return

    # Create results table with no_wrap to prevent truncation
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", no_wrap=True, min_width=20)
    table.add_column("Function", style="green", no_wrap=True, min_width=15)
    table.add_column("Algorithm", no_wrap=True)
    table.add_column("Best Score", justify="right")
    table.add_column("Trials", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Date", style="dim")

    for result_info in all_results:
        table.add_row(
            result_info["name"],
            result_info["function_name"],
            result_info["algorithm"],
            f"{result_info['best_score']:.3f}",
            str(result_info["total_trials"]),
            f"{result_info.get('success_rate', 0):.1%}",
            result_info.get("created_at", "").split("T")[0],  # Just date part
        )

    console.print(table)


@cli.command()
@click.argument("result_name")
@click.option(
    "--storage-dir", "-d", default=".traigent", help="TraiGent storage directory"
)
@click.option(
    "--plot-type",
    "-p",
    type=click.Choice(["progress", "pareto", "report"]),
    default="progress",
    help="Type of plot to generate",
)
def plot(result_name: str, storage_dir: str, plot_type: str) -> None:
    """Generate plots for optimization results."""
    console.print(
        f"\n[bold blue]Generating {plot_type} plot for: {result_name}[/bold blue]\n"
    )

    try:
        persistence = PersistenceManager(storage_dir)
        result = persistence.load_result(result_name)

        # Generate plot
        plot_output = create_quick_plot(result, plot_type)
        console.print(plot_output)

    except FileNotFoundError:
        console.print(f"[red]Result '{result_name}' not found[/red]")
        console.print("Use 'traigent results' to see available results")
    except Exception as e:
        console.print(f"[red]Error generating plot: {e}[/red]")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def validate_config(config_file: str) -> None:
    """Validate optimization configuration file."""
    console.print(f"\n[bold blue]Validating Configuration: {config_file}[/bold blue]\n")

    try:
        with open(config_file) as f:
            config = json.load(f)

        # Extract configuration components
        config_space = config.get("configuration_space", {})
        objectives = config.get("objectives", ["accuracy"])
        dataset_path = config.get("eval_dataset")
        algorithm = config.get("algorithm", "grid")

        # Validate configuration
        result = OptimizationValidator.validate_optimization_config(
            config_space, objectives, dataset_path, algorithm
        )

        if result.is_valid:
            console.print("✅ [green]Configuration validation passed[/green]")
        else:
            console.print("❌ [red]Configuration validation failed[/red]")

        console.print(result.get_feedback())

    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in config file: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Error reading config file: {e}[/red]")


@cli.command()
@click.option(
    "--template",
    "-t",
    type=click.Choice(["basic", "multi-objective", "langchain", "openai"]),
    default="basic",
    help="Example template to generate",
)
@click.option("--output", "-o", default="traigent_example.py", help="Output file name")
def generate(template: str, output: str) -> None:
    """Generate example code templates."""
    console.print(
        f"\n[bold blue]Generating {template} template: {output}[/bold blue]\n"
    )

    templates = {
        "basic": _get_basic_template(),
        "multi-objective": _get_multi_objective_template(),
        "langchain": _get_langchain_template(),
        "openai": _get_openai_template(),
    }

    template_code = templates[template]

    # Validate output path
    output_path = Path(output).resolve()

    # Security: Check for directory traversal
    if (
        ".." in output
        or str(output_path).startswith("/etc")
        or str(output_path).startswith("/sys")
    ):
        console.print("[red]Error: Invalid output path[/red]")
        return

    # Check if file already exists
    if output_path.exists():
        if not click.confirm(f"File {output} already exists. Overwrite?"):
            console.print("[yellow]Generation cancelled[/yellow]")
            return

    # Write template to file
    with open(output_path, "w") as f:
        f.write(template_code)

    console.print(f"✅ [green]Template generated: {output}[/green]")

    # Show syntax highlighted preview
    syntax = Syntax(template_code, "python", theme="monokai", line_numbers=True)
    console.print("\n[bold]Preview:[/bold]")
    console.print(syntax)


@cli.command()
def examples() -> None:
    """Show comprehensive usage examples."""
    console.print("\n[bold blue]TraiGent SDK Usage Examples[/bold blue]\n")

    examples = [
        {
            "title": "1. Basic Function Optimization",
            "code": """import traigent

        @traigent.optimize(
            eval_dataset="evals.jsonl",
            configuration_space={
                "model": ["gpt-4o-mini", "GPT-4o"],
                "temperature": [0.0, 0.5, 1.0]
            }
        )
        def my_agent(query: str, **config) -> str:
            # Your LLM logic here with config parameters
            return process_query(query, **config)

        # Run optimization
        import asyncio
        results = asyncio.run(my_agent.optimize())
        best_config = my_agent.get_best_config()""",
        },
        {
            "title": "2. Multi-Objective with Constraints",
            "code": """@traigent.optimize(
            eval_dataset="evals.jsonl",
            objectives=["accuracy", "cost", "latency"],
            configuration_space={
                "model": ["gpt-4o-mini", "GPT-4o"],
                "max_tokens": (100, 2000)
            }
        )
        def content_generator(prompt: str, **config) -> str:
            return generate_content(prompt, **config)

        # Add constraints
        from traigent.utils.constraints import model_cost_constraint
        content_generator.add_constraint(model_cost_constraint(max_cost_per_1k_tokens=0.05))

        # Find Pareto-optimal solutions
        results = asyncio.run(content_generator.optimize())""",
        },
        {
            "title": "3. OpenAI Integration",
            "code": """import traigent
from traigent.integrations import enable_openai_optimization

# Enable OpenAI SDK optimization
enable_openai_optimization()

@traigent.optimize(
    eval_dataset="data/qa_dataset.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0]
    }
)
def chat_agent(query: str) -> str:
    import openai
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

# Run optimization
import asyncio
results = asyncio.run(chat_agent.optimize())
print(f"Best config: {results.best_config}")""",
        },
        {
            "title": "4. Production Features",
            "code": """# With persistence and callbacks
        from traigent import PersistenceManager, ProgressBarCallback

        persistence = PersistenceManager(".my_optimizations")
        callbacks = [ProgressBarCallback()]

        @traigent.optimize(
            eval_dataset="evals.jsonl",
            objectives=["accuracy", "latency"],
            configuration_space={"temperature": (0.0, 1.0)},
            callbacks=callbacks
        )
        def my_function(input_text: str, **config) -> str:
            return process(input_text, **config)

        # Run with persistence
        result = asyncio.run(my_function.optimize())
        persistence.save_result(result, "my_optimization_v1")""",
        },
    ]

    for example in examples:
        console.print(f"[bold green]{example['title']}[/bold green]")
        syntax = Syntax(example["code"], "python", theme="monokai")
        console.print(syntax)
        console.print()


def _get_basic_template() -> str:
    """Get basic optimization template."""
    return '''"""Basic TraiGent optimization example."""

import asyncio
import traigent

# Dataset format (JSONL): {"input": {"question": "..."}, "output": "expected_result"}
# Use the example dataset or create your own:
DATASET = "examples/datasets/quickstart/qa_samples.jsonl"


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],  # Use list for grid/random search
        "max_tokens": [100, 250, 500]
    }
)
def my_function(question: str, **config) -> str:
    """Your function to optimize."""
    # Replace with your actual logic
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 150)

    # Example: call your LLM here
    result = f"Answer to '{question}' using {model} (temp={temperature})"
    return result


async def main():
    """Run optimization."""
    print("Starting optimization...")

    # Run optimization
    result = await my_function.optimize(
        algorithm="random",
        max_trials=10
    )

    print(f"Best score: {result.best_score:.3f}")
    print(f"Best config: {result.best_config}")

    # Use optimized function
    optimized_result = my_function("test input")
    print(f"Optimized result: {optimized_result}")


if __name__ == "__main__":
    asyncio.run(main())
'''


def _get_multi_objective_template() -> str:
    """Get multi-objective optimization template."""
    return '''"""Multi-objective optimization with TraiGent."""

import asyncio
import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

# Dataset format (JSONL): {"input": {"question": "..."}, "output": "expected_result"}
# Use the example dataset or create your own:
DATASET = "examples/datasets/quickstart/qa_samples.jsonl"

# Define custom objectives with weights
custom_objectives = ObjectiveSchema.from_objectives([
    ObjectiveDefinition("accuracy", orientation="maximize", weight=0.7),
    ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
])


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=custom_objectives,
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],  # Use list for grid/random search
        "max_tokens": [100, 250, 500]
    }
)
def multi_objective_function(question: str, **config) -> str:
    """Function with multiple objectives to optimize.

    TraiGent automatically tracks:
    - accuracy: compared against expected output
    - cost: token usage costs
    """
    # Your LLM call here - TraiGent tracks metrics automatically
    model = config.get("model", "gpt-4o-mini")
    return f"Answer to '{question}' using {model}"


async def main():
    """Run multi-objective optimization."""
    print("Starting multi-objective optimization...")
    print("Balancing: accuracy (70%), cost (30%)")

    result = await multi_objective_function.optimize(
        algorithm="random",
        max_trials=20
    )

    print(f"\\nTotal trials: {len(result.trials)}")
    print(f"Best score: {result.best_score:.3f}")
    print(f"Best config: {result.best_config}")

    # Show trade-off analysis
    if hasattr(result, 'trials') and result.trials:
        print("\\nTop 3 configurations:")
        sorted_trials = sorted(result.trials, key=lambda t: t.score, reverse=True)[:3]
        for i, trial in enumerate(sorted_trials, 1):
            print(f"  {i}. {trial.config} -> score={trial.score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
'''


def _get_langchain_template() -> str:
    """Get LangChain integration template."""
    return '''"""LangChain integration with TraiGent."""

import asyncio
import traigent

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("LangChain not installed. Install with: pip install langchain-openai")
    exit(1)

# Dataset format (JSONL): {"input": {"question": "..."}, "output": "expected_result"}
# Use the example dataset or create your own:
DATASET = "examples/datasets/quickstart/qa_samples.jsonl"


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0],
        "max_tokens": [100, 250, 500]
    }
)
def langchain_agent(question: str) -> str:
    """LangChain-based agent that TraiGent will optimize.

    TraiGent automatically intercepts ChatOpenAI parameters and
    injects optimized values during trials.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # TraiGent will override
        temperature=0.7,      # TraiGent will override
        max_tokens=150        # TraiGent will override
    )
    response = llm.invoke(f"Question: {question}\\nAnswer:")
    return response.content


async def main():
    """Run LangChain optimization."""
    print("Optimizing LangChain agent...")

    result = await langchain_agent.optimize(
        algorithm="random",
        max_trials=10
    )

    print(f"Best score: {result.best_score:.3f}")
    print(f"Best config: {result.best_config}")

    # Apply best config for future calls
    langchain_agent.apply_best_config(result)

    # Test with optimized parameters
    answer = langchain_agent("What is machine learning?")
    print(f"\\nOptimized answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
'''


def _get_openai_template() -> str:
    """Get OpenAI integration template."""
    return '''"""OpenAI SDK integration with TraiGent."""

import asyncio
import traigent
from traigent.integrations import enable_openai_optimization

# Enable OpenAI SDK optimization (intercepts openai.chat.completions.create)
enable_openai_optimization()

# Dataset format (JSONL): {"input": {"question": "..."}, "output": "expected_result"}
# Use the example dataset or create your own:
DATASET = "examples/datasets/quickstart/qa_samples.jsonl"


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0],
        "max_tokens": [100, 250, 500]
    },
    max_trials=15
)
def chat_agent(question: str) -> str:
    """Chat function that TraiGent will optimize."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # TraiGent will override this
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,  # TraiGent will override this
        max_tokens=150    # TraiGent will override this
    )
    return response.choices[0].message.content


async def main():
    """Run optimization and use the result."""
    print("Optimizing OpenAI chat completion...")

    # Run optimization
    result = await chat_agent.optimize()

    print(f"Best score: {result.best_score:.3f}")
    print(f"Best config: {result.best_config}")

    # Apply best config for future calls
    chat_agent.apply_best_config(result)

    # Test with optimized parameters
    response = chat_agent("Explain quantum computing in simple terms")
    print(f"\\nOptimized response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
'''


@cli.command()
@click.argument("module_path", type=str)
@click.option("--functions", help="Comma-separated function names to check")
@click.option(
    "--threshold", default=10, type=int, help="Improvement threshold percentage"
)
@click.option("--objectives", help="Comma-separated objectives to check")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be checked without running optimization",
)
def check(
    module_path: str, functions: str, threshold: int, objectives: str, dry_run: bool
) -> None:
    """Validate TraiGent optimization improves over default parameters.

    This command automatically discovers functions decorated with @traigent.optimize
    and validates that optimization actually improves over the function's default parameters
    using Pareto efficiency analysis.

    Args:
        module_path: Path to Python file containing optimizable functions

    Examples:
        traigent check my_module.py                    # Check all functions
        traigent check my_module.py --functions="func1,func2"  # Check specific functions
        traigent check my_module.py --threshold=15     # Require 15% improvement
        traigent check my_module.py --dry-run          # Preview what would be checked
    """
    import asyncio

    from traigent.cli.function_discovery import (
        discover_optimized_functions,
        print_discovery_summary,
        validate_discovered_functions,
    )
    from traigent.cli.optimization_validator import OptimizationValidator

    console.print("\n[bold blue]🔍 TraiGent Optimization Validation[/bold blue]")
    console.print(f"Module: [cyan]{module_path}[/cyan]")
    console.print(f"Threshold: [yellow]{threshold}%[/yellow]")

    if dry_run:
        console.print(
            "[dim]Running in dry-run mode (no optimization will be executed)[/dim]"
        )

    try:
        # Parse function filter
        function_filter = None
        if functions:
            function_filter = [name.strip() for name in functions.split(",")]
            console.print(f"Functions: [green]{', '.join(function_filter)}[/green]")

        # Parse objectives filter
        objectives_filter = None
        if objectives:
            objectives_filter = [obj.strip() for obj in objectives.split(",")]
            console.print(
                f"Objectives: [magenta]{', '.join(objectives_filter)}[/magenta]"
            )

        # Discover optimized functions
        console.print("\n[bold yellow]📋 Step 1: Function Discovery[/bold yellow]")
        discovered_functions = discover_optimized_functions(
            module_path, function_filter
        )

        # Validate discovered functions
        validation_issues = validate_discovered_functions(discovered_functions)
        print_discovery_summary(discovered_functions, validation_issues)

        if not discovered_functions:
            console.print(
                f"\n[red]❌ No optimizable functions found in {module_path}[/red]"
            )
            console.print(
                "Make sure your functions are decorated with @traigent.optimize"
            )
            return

        # Filter by objectives if specified
        if objectives_filter:
            filtered_functions = []
            for func_info in discovered_functions:
                if any(obj in func_info.objectives for obj in objectives_filter):
                    filtered_functions.append(func_info)
            discovered_functions = filtered_functions

            if not discovered_functions:
                console.print(
                    f"\n[red]❌ No functions found with objectives: {objectives_filter}[/red]"
                )
                return

        if dry_run:
            console.print(
                f"\n[green]✅ Dry run completed - found {len(discovered_functions)} function(s) ready for validation[/green]"
            )
            return

        # Run optimization validation
        console.print("\n[bold yellow]⚖️  Step 2: Optimization Validation[/bold yellow]")
        validator = OptimizationValidator(threshold_pct=threshold)

        async def run_validations() -> tuple[list[Any], int, int]:
            results = []
            passed_count = 0
            failed_count = 0

            for func_info in discovered_functions:
                console.print(f"\n[bold cyan]Validating: {func_info.name}[/bold cyan]")

                try:
                    result = await validator.validate_optimization(func_info)
                    results.append(result)

                    # Print result summary
                    if result.should_block:
                        console.print(f"[red]{result.get_summary()}[/red]")
                        failed_count += 1
                    else:
                        console.print(f"[green]{result.get_summary()}[/green]")
                        passed_count += 1

                except Exception as e:
                    console.print(
                        f"[red]❌ Error validating {func_info.name}: {e}[/red]"
                    )
                    failed_count += 1

            return results, passed_count, failed_count

        # Run all validations
        results, passed_count, failed_count = asyncio.run(run_validations())

        # Print final summary
        console.print("\n[bold blue]📊 Validation Summary[/bold blue]")
        console.print(f"Total functions: {len(discovered_functions)}")
        console.print(f"[green]✅ Passed: {passed_count}[/green]")
        console.print(f"[red]❌ Failed: {failed_count}[/red]")

        # Show detailed results for failed functions
        failed_results = [r for r in results if r.should_block]
        if failed_results:
            console.print("\n[bold red]🚨 Failed Validations:[/bold red]")
            for result in failed_results:
                console.print(f"\n{result.get_detailed_report()}")

        # Exit with appropriate code for git hooks
        if failed_count > 0:
            console.print(
                f"\n[red]❌ Optimization validation failed for {failed_count} function(s)[/red]"
            )
            console.print("Optimization does not improve over default parameters")
            console.print("To bypass: git push --no-verify")
            exit(1)
        else:
            console.print(
                "\n[green]✅ All optimizations validated successfully![/green]"
            )
            console.print("Optimized configurations are superior to default parameters")
            exit(0)

    except Exception as e:
        console.print(f"[red]Error during validation: {e}[/red]")
        exit(1)


# Register local commands
register_edge_analytics_commands(cli)

cli.add_command(auth)
cli.add_command(hooks)

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
