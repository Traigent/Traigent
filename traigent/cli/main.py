"""Enhanced CLI interface for Traigent SDK."""

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
from traigent.utils.secure_path import (
    PathTraversalError,
    safe_open,
    safe_write_text,
    validate_path,
)
from traigent.utils.validation import OptimizationValidator
from traigent.visualization.plots import create_quick_plot

console = Console()
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# Style constant for table headers
_TABLE_HEADER_STYLE = "bold magenta"

# Message constants to avoid duplication (SonarQube S1192)
_MSG_RESULTS_LIST_HINT = "Use 'traigent results list' to see available results"
_MSG_BEST_SCORE = "Best Score"
_MSG_TOTAL_TRIALS = "Total Trials"
_MSG_SUCCESS_RATE = "Success Rate"
_STYLE_HIGHLIGHT = "[yellow]"


def _resolve_workspace_path(
    path: Path,
    description: str,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a path and ensure it lives within the repository workspace."""
    try:
        return validate_path(
            path.expanduser(),
            WORKSPACE_ROOT,
            must_exist=must_exist,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(f"{description} does not exist: {exc}") from exc
    except PathTraversalError as exc:
        raise click.ClickException(
            f"{description} must reside within the Traigent workspace ({WORKSPACE_ROOT})"
        ) from exc


def _print_optimization_header(
    file_path: str,
    algorithm: str,
    max_trials: int,
    timeout: float,
    max_total_examples: int | None,
    samples_include_pruned: bool,
) -> None:
    """Print the optimization header with configuration details."""
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


def _is_optimizable_function(obj: Any) -> bool:
    """Check if an object is decorated with @traigent.optimize."""
    import inspect

    # Check if it's decorated with @traigent.optimize
    # The decorator returns an OptimizedFunction instance with an optimize method
    if inspect.isfunction(obj) and hasattr(obj, "optimize"):
        return True
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "OptimizedFunction":
        return True
    # Additional check for objects with optimize method
    if hasattr(obj, "optimize") and callable(getattr(obj, "optimize", None)):
        if hasattr(obj, "func") or callable(obj):
            return True
    return False


def _find_optimizable_functions(
    module: Any, function_filter: str | None
) -> list[tuple[str, Any]]:
    """Find functions with @traigent.optimize decorator in a module."""
    import inspect

    optimizable_functions = []
    for name, obj in inspect.getmembers(module):
        # Skip private/internal attributes and modules
        if name.startswith("_") or inspect.ismodule(obj):
            continue

        if _is_optimizable_function(obj):
            if function_filter is None or name == function_filter:
                optimizable_functions.append((name, obj))

    return optimizable_functions


def _build_optimize_kwargs(
    algorithm: str,
    max_trials: int,
    timeout: float,
    verbose: bool,
    samples_include_pruned: bool,
    max_total_examples: int | None,
    tvl_spec: str | None,
    tvl_environment: str | None,
) -> dict[str, Any]:
    """Build keyword arguments for the optimize call."""
    optimize_kwargs: dict[str, Any] = {
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
    return optimize_kwargs


def _display_optimization_result(
    func_name: str,
    result: Any,
    persistence: PersistenceManager,
    algorithm: str,
    max_trials: int,
) -> None:
    """Display and save the optimization result for a single function."""
    console.print(f"✅ Best score: [green]{result.best_score:.3f}[/green]")
    console.print(f"   Best config: {result.best_config}")
    console.print(f"   Total trials: {len(result.trials)}")
    console.print(f"   Successful trials: {len(result.successful_trials)}")

    # Save result
    result_name = f"{func_name}_{algorithm}_{max_trials}"
    persistence.save_result(result, result_name)
    console.print(f"   Results saved as: {result_name}")


def _display_optimization_summary(
    results: list[tuple[str, Any]], output_dir_resolved: Path
) -> None:
    """Display the final optimization summary table."""
    console.print("\n[bold green]Optimization Complete![/bold green]")
    console.print(f"Total functions optimized: {len(results)}")
    console.print(f"Results saved to: {output_dir_resolved}")

    if not results:
        return

    table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
    table.add_column("Function", style="cyan")
    table.add_column(_MSG_BEST_SCORE, justify="right")
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


def _handle_no_optimizable_functions(function_filter: str | None) -> None:
    """Print error message when no optimizable functions are found."""
    if function_filter:
        console.print(
            f"[red]No optimizable function named '{function_filter}' found[/red]"
        )
    else:
        console.print("[red]No functions with @traigent.optimize decorator found[/red]")
    console.print("\nMake sure your functions are decorated with @traigent.optimize")


def _load_user_module(file_path_obj: Path) -> Any | None:
    """Load a Python file as a module. Returns the module or None on failure."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("user_module", file_path_obj)
    if spec is None or spec.loader is None:
        console.print("[red]Error: Failed to load Python file[/red]")
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_module"] = module
    spec.loader.exec_module(module)
    return module


def _resolve_output_dir(output_dir: str) -> Path:
    """Resolve and validate the output directory path."""
    output_dir_candidate = Path(output_dir)
    if not output_dir_candidate.is_absolute():
        output_dir_candidate = WORKSPACE_ROOT / output_dir_candidate
    return _resolve_workspace_path(output_dir_candidate, "Output directory")


def _parse_comma_separated_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated string into a list of trimmed values."""
    if not value:
        return None
    return [item.strip() for item in value.split(",")]


def _filter_functions_by_objectives(
    discovered_functions: list[Any], objectives_filter: list[str]
) -> list[Any]:
    """Filter discovered functions to only include those with specified objectives."""
    return [
        func_info
        for func_info in discovered_functions
        if any(obj in func_info.objectives for obj in objectives_filter)
    ]


def _print_check_summary(
    discovered_functions: list[Any],
    passed_count: int,
    failed_count: int,
    results: list[Any],
) -> None:
    """Print the final validation summary for the check command."""
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


def _handle_check_exit(failed_count: int) -> None:
    """Handle exit logic for the check command."""
    if failed_count > 0:
        console.print(
            f"\n[red]❌ Optimization validation failed for {failed_count} function(s)[/red]"
        )
        console.print("Optimization does not improve over default parameters")
        console.print("To bypass: git push --no-verify")
        exit(1)
    else:
        console.print("\n[green]✅ All optimizations validated successfully![/green]")
        console.print("Optimized configurations are superior to default parameters")
        exit(0)


def _print_check_header(module_path: str, threshold: int, dry_run: bool) -> None:
    """Print the header for the check command."""
    console.print("\n[bold blue]🔍 Traigent Optimization Validation[/bold blue]")
    console.print(f"Module: [cyan]{module_path}[/cyan]")
    console.print(f"Threshold: [yellow]{threshold}%[/yellow]")
    if dry_run:
        console.print(
            "[dim]Running in dry-run mode (no optimization will be executed)[/dim]"
        )


def _print_filter_info(
    function_filter: list[str] | None, objectives_filter: list[str] | None
) -> None:
    """Print filter information for the check command."""
    if function_filter:
        console.print(f"Functions: [green]{', '.join(function_filter)}[/green]")
    if objectives_filter:
        console.print(f"Objectives: [magenta]{', '.join(objectives_filter)}[/magenta]")


def _run_single_optimization(
    func_name: str,
    func: Any,
    optimize_kwargs: dict[str, Any],
    persistence: PersistenceManager,
    algorithm: str,
    max_trials: int,
    verbose: bool,
) -> tuple[str, Any] | None:
    """Run optimization for a single function. Returns (name, result) or None on error."""
    import traceback

    console.print(f"\n[bold blue]Optimizing: {func_name}[/bold blue]")
    try:
        import asyncio

        result = asyncio.run(func.optimize(**optimize_kwargs))
        _display_optimization_result(
            func_name, result, persistence, algorithm, max_trials
        )
        return (func_name, result)
    except Exception as e:
        console.print(f"❌ [red]Error optimizing {func_name}: {e}[/red]")
        if verbose:
            console.print(traceback.format_exc())
        return None


@click.group()
@click.version_option(
    version=get_version_info()["version"],
    prog_name="Traigent",
    message="%(prog)s %(version)s",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging (errors only)")
def cli(verbose: bool, debug: bool, quiet: bool) -> None:
    """Traigent SDK - Open-source LLM optimization toolkit.

    Traigent makes it effortless to optimize your LLM applications with
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
    """Show Traigent SDK version and system information."""
    version_info = get_version_info()

    console.print("\n[bold blue]Traigent SDK[/bold blue]")
    console.print(f"Version: [green]{version_info['version']}[/green]")
    console.print(f"Python: {version_info['python_version'].split()[0]}")
    console.print(f"Platform: {version_info['platform']}")

    # Features table
    console.print("\n[bold]Features:[/bold]")
    features_table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
    features_table.add_column("Feature")
    features_table.add_column("Status")

    for feature, enabled in version_info["features"].items():
        status = "[green]✓ Enabled[/green]" if enabled else "[red]✗ Disabled[/red]"
        features_table.add_row(feature.replace("_", " ").title(), status)

    console.print(features_table)

    # Integrations table
    console.print("\n[bold]Integrations:[/bold]")
    integrations_table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
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
    import sys
    import traceback
    from pathlib import Path

    _print_optimization_header(
        file_path,
        algorithm,
        max_trials,
        timeout,
        max_total_examples,
        samples_include_pruned,
    )

    # Resolve paths
    file_path_obj = _resolve_workspace_path(
        Path(file_path), "File path", must_exist=True
    )
    output_dir_resolved = _resolve_output_dir(output_dir)

    # Add parent directory to Python path temporarily
    parent_dir = file_path_obj.parent
    sys.path.insert(0, str(parent_dir))

    try:
        module = _load_user_module(file_path_obj)
        if module is None:
            return

        optimizable_functions = _find_optimizable_functions(module, function)
        if not optimizable_functions:
            _handle_no_optimizable_functions(function)
            return

        console.print(
            f"\n[green]Found {len(optimizable_functions)} optimizable function(s)[/green]"
        )

        optimize_kwargs = _build_optimize_kwargs(
            algorithm,
            max_trials,
            timeout,
            verbose,
            samples_include_pruned,
            max_total_examples,
            tvl_spec,
            tvl_environment,
        )
        persistence = PersistenceManager(str(output_dir_resolved))

        async def run_optimizations() -> list[tuple[str, Any]]:
            results: list[tuple[str, Any]] = []
            for func_name, func in optimizable_functions:
                console.print(f"\n[bold blue]Optimizing: {func_name}[/bold blue]")
                try:
                    result = await func.optimize(**optimize_kwargs)
                    _display_optimization_result(
                        func_name, result, persistence, algorithm, max_trials
                    )
                    results.append((func_name, result))
                except Exception as e:
                    console.print(f"❌ [red]Error optimizing {func_name}: {e}[/red]")
                    if verbose:
                        console.print(traceback.format_exc())
            return results

        results = asyncio.run(run_optimizations())
        _display_optimization_summary(results, output_dir_resolved)

    except Exception as e:
        console.print(f"[red]Error loading file: {e}[/red]")
        if verbose:
            console.print(traceback.format_exc())
    finally:
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

    from traigent.evaluators.base import _resolve_dataset_source
    from traigent.utils.exceptions import ValidationError
    from traigent.utils.validation import Validators

    # Step 1: Validate using runtime path resolution (same as traigent.optimize)
    # This ensures CLI validation matches runtime behavior
    try:
        resolved_path, _registry_entry = _resolve_dataset_source(dataset_path)
        console.print("✅ [green]Dataset path validation passed[/green]")
        if verbose:
            console.print(f"   Resolved path: {resolved_path}")
    except ValidationError as e:
        console.print("❌ [red]Dataset path validation failed[/red]")
        console.print(f"   [red]{e}[/red]")
        console.print(
            "\n[yellow]Hint:[/yellow] Dataset paths must reside under the current "
            "working directory or the path specified by TRAIGENT_DATASET_ROOT."
        )
        return

    # Step 2: Validate dataset content format
    path_result = Validators.validate_dataset(dataset_path)

    if path_result.is_valid:
        console.print("✅ [green]Dataset content validation passed[/green]")
    else:
        console.print("❌ [red]Dataset content validation failed[/red]")

    if verbose or not path_result.is_valid:
        console.print(path_result.get_feedback())

    # Validate objectives if provided
    if objectives:
        obj_result = Validators.validate_objectives(list(objectives))

        if obj_result.is_valid:
            console.print("✅ [green]Objectives validation passed[/green]")
        else:
            console.print("❌ [red]Objectives validation failed[/red]")
            console.print(obj_result.get_feedback())


# -----------------------------------------------------------------------------
# Helper functions for results commands (reduces cognitive complexity - S3776)
# -----------------------------------------------------------------------------


def _format_metric_value(value: Any) -> str:
    """Format a metric value for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_diff_colored(diff: float, fmt: str = ".4f") -> str:
    """Format a numeric diff with color coding (green=positive, red=negative)."""
    if diff > 0:
        return f"[green]+{diff:{fmt}}[/]"
    if diff < 0:
        return f"[red]{diff:{fmt}}[/]"
    return f"{diff:+{fmt}}"


def _print_result_summary(result: Any) -> None:
    """Print the summary section of a result."""
    console.print("[bold]Summary[/bold]")
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Key", style="dim")
    summary_table.add_column("Value")

    summary_table.add_row("Function", getattr(result, "function_name", "unknown"))
    summary_table.add_row("Algorithm", getattr(result, "algorithm", "unknown"))
    summary_table.add_row(_MSG_TOTAL_TRIALS, str(len(result.trials)))
    summary_table.add_row("Successful Trials", str(len(result.successful_trials or [])))
    summary_table.add_row(_MSG_BEST_SCORE, f"{result.best_score:.4f}")
    summary_table.add_row("Stop Reason", result.stop_reason or "unknown")
    if hasattr(result, "duration") and result.duration:
        summary_table.add_row("Duration", f"{result.duration:.1f}s")

    console.print(summary_table)


def _print_config_json(
    config: dict[str, Any] | None, title: str = "Best Configuration"
) -> None:
    """Print a config dict as formatted JSON."""
    console.print(f"\n[bold]{title}[/bold]")
    if config:
        config_syntax = Syntax(json.dumps(config, indent=2), "json", theme="monokai")
        console.print(config_syntax)
    else:
        console.print("[yellow]No best configuration found[/yellow]")


def _print_metrics_table(metrics: dict[str, Any] | None) -> None:
    """Print a metrics table, filtering out internal metrics."""
    if not metrics:
        return
    console.print("\n[bold]Best Metrics[/bold]")
    metrics_table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")
    for metric, value in metrics.items():
        if not metric.startswith("_"):
            metrics_table.add_row(metric, _format_metric_value(value))
    console.print(metrics_table)


def _print_trials_table(trials: list[Any], max_display: int = 50) -> None:
    """Print a table of trials with config and score."""
    if not trials:
        return
    console.print(f"\n[bold]All Trials ({len(trials)})[/bold]")
    trials_table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
    trials_table.add_column("#", justify="right")
    trials_table.add_column("Config")
    trials_table.add_column("Score", justify="right")
    trials_table.add_column("Status")

    for i, trial in enumerate(trials[:max_display], 1):
        config_str = str(trial.config)
        if len(config_str) > 40:
            config_str = config_str[:37] + "..."
        primary_score = trial.metrics.get("overall") or trial.metrics.get("score")
        score_str = f"{primary_score:.4f}" if primary_score is not None else "N/A"
        status = getattr(trial, "status", "completed")
        trials_table.add_row(str(i), config_str, score_str, status)

    console.print(trials_table)
    if len(trials) > max_display:
        console.print(f"[dim]... and {len(trials) - max_display} more trials[/dim]")


def _build_comparison_summary_table(r1: Any, r2: Any, name1: str, name2: str) -> Table:
    """Build the main comparison summary table."""
    table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
    table.add_column("Metric", style="bold")
    table.add_column(name1, style="cyan")
    table.add_column(name2, style="green")
    table.add_column("Δ", justify="right")

    # Best score
    score_diff = r2.best_score - r1.best_score
    table.add_row(
        _MSG_BEST_SCORE,
        f"{r1.best_score:.4f}",
        f"{r2.best_score:.4f}",
        _format_diff_colored(score_diff),
    )

    # Trials count
    r1_total, r2_total = len(r1.trials), len(r2.trials)
    table.add_row(
        _MSG_TOTAL_TRIALS, str(r1_total), str(r2_total), str(r2_total - r1_total)
    )

    # Success rate
    sr1 = len(r1.successful_trials or []) / max(r1_total, 1)
    sr2 = len(r2.successful_trials or []) / max(r2_total, 1)
    table.add_row(
        _MSG_SUCCESS_RATE,
        f"{sr1:.1%}",
        f"{sr2:.1%}",
        _format_diff_colored(sr2 - sr1, ".1%"),
    )

    return table


def _build_config_comparison_table(
    config1: dict[str, Any] | None,
    config2: dict[str, Any] | None,
    name1: str,
    name2: str,
) -> Table:
    """Build a config comparison table highlighting differences."""
    config_table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
    config_table.add_column("Parameter")
    config_table.add_column(name1)
    config_table.add_column(name2)

    all_keys = set((config1 or {}).keys()) | set((config2 or {}).keys())
    for key in sorted(all_keys):
        v1 = (config1 or {}).get(key, "-")
        v2 = (config2 or {}).get(key, "-")
        style = _STYLE_HIGHLIGHT if v1 != v2 else ""
        config_table.add_row(key, f"{style}{v1}[/]", f"{style}{v2}[/]")

    return config_table


def _build_metrics_comparison_table(
    metrics1: dict[str, Any] | None,
    metrics2: dict[str, Any] | None,
    name1: str,
    name2: str,
) -> Table | None:
    """Build a metrics comparison table, or None if no metrics."""
    if not metrics1 and not metrics2:
        return None

    metrics_table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
    metrics_table.add_column("Metric")
    metrics_table.add_column(name1, justify="right")
    metrics_table.add_column(name2, justify="right")
    metrics_table.add_column("Δ", justify="right")

    all_metrics = set((metrics1 or {}).keys()) | set((metrics2 or {}).keys())
    for metric in sorted(all_metrics):
        if metric.startswith("_"):
            continue
        m1 = (metrics1 or {}).get(metric)
        m2 = (metrics2 or {}).get(metric)
        diff_str = "-"
        if isinstance(m1, (int, float)) and isinstance(m2, (int, float)):
            diff_str = f"{m2 - m1:+.4f}"
        metrics_table.add_row(
            metric,
            _format_metric_value(m1) if m1 else "-",
            _format_metric_value(m2) if m2 else "-",
            diff_str,
        )

    return metrics_table


def _parse_weights(weights_str: str) -> dict[str, float] | None:
    """Parse and normalize weight string like 'accuracy=0.7,cost=0.3'.

    Returns normalized weights dict or None if parsing fails.
    """
    try:
        weight_dict: dict[str, float] = {}
        for pair in weights_str.split(","):
            key, value = pair.strip().split("=")
            weight_dict[key.strip()] = float(value.strip())

        # Normalize weights to sum to 1
        total = sum(weight_dict.values())
        if total > 0:
            weight_dict = {k: v / total for k, v in weight_dict.items()}
        return weight_dict
    except ValueError:
        return None


def _calculate_trial_score(trial: Any, weight_dict: dict[str, float]) -> float | None:
    """Calculate weighted score for a trial. Returns None if no valid metrics."""
    if not trial.metrics:
        return None

    score = 0.0
    has_metrics = False
    for metric, weight in weight_dict.items():
        if metric in trial.metrics:
            metric_value = trial.metrics[metric]
            if isinstance(metric_value, (int, float)):
                score += weight * metric_value
                has_metrics = True

    return score if has_metrics else None


def _build_rerank_table(
    scored_trials: list[tuple[float, Any]], top_n: int = 5
) -> Table:
    """Build a table showing top re-ranked configurations."""
    table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
    table.add_column("Rank", justify="right")
    table.add_column("New Score", justify="right")
    table.add_column("Original Score", justify="right")
    table.add_column("Config")

    for i, (new_score, trial) in enumerate(scored_trials[:top_n], 1):
        config_str = json.dumps(trial.config)
        if len(config_str) > 50:
            config_str = config_str[:47] + "..."
        orig_score_val = trial.metrics.get("overall") or trial.metrics.get("score")
        orig_score = f"{orig_score_val:.4f}" if orig_score_val is not None else "N/A"
        table.add_row(str(i), f"{new_score:.4f}", orig_score, config_str)

    return table


@cli.group()
def results() -> None:
    """Browse, compare, and manage optimization results.

    Examples:
        traigent results list                    # List all runs
        traigent results show my_run             # Show detailed view
        traigent results compare run1 run2       # Side-by-side comparison
        traigent results rerank my_run --weights accuracy=0.8,cost=0.2
    """
    pass


@results.command("list")
@click.option(
    "--storage-dir", "-d", default=".traigent", help="Traigent storage directory"
)
def results_list(storage_dir: str) -> None:
    """List all optimization results with summary."""
    console.print("\n[bold blue]Traigent Optimization Results[/bold blue]\n")

    persistence = PersistenceManager(storage_dir)
    all_results = persistence.list_results()

    if not all_results:
        console.print("[yellow]No optimization results found[/yellow]")
        console.print(f"Results are stored in: {persistence.base_dir}")
        return

    # Create results table with no_wrap to prevent truncation
    table = Table(show_header=True, header_style=_TABLE_HEADER_STYLE)
    table.add_column("Name", style="cyan", no_wrap=True, min_width=20)
    table.add_column("Function", style="green", no_wrap=True, min_width=15)
    table.add_column("Algorithm", no_wrap=True)
    table.add_column(_MSG_BEST_SCORE, justify="right")
    table.add_column("Trials", justify="right")
    table.add_column(_MSG_SUCCESS_RATE, justify="right")
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


@results.command("show")
@click.argument("result_name")
@click.option(
    "--storage-dir", "-d", default=".traigent", help="Traigent storage directory"
)
@click.option("--trials", "-t", is_flag=True, help="Show all trial details")
def results_show(result_name: str, storage_dir: str, trials: bool) -> None:
    """Show detailed view of a specific optimization run.

    Examples:
        traigent results show my_run              # Summary view
        traigent results show my_run --trials     # Include all trials
    """
    console.print(f"\n[bold blue]Optimization Result: {result_name}[/bold blue]\n")

    try:
        persistence = PersistenceManager(storage_dir)
        result = persistence.load_result(result_name)

        _print_result_summary(result)
        _print_config_json(result.best_config)
        _print_metrics_table(result.best_metrics)

        if trials:
            _print_trials_table(result.trials)

    except FileNotFoundError:
        console.print(f"[red]Result '{result_name}' not found[/red]")
        console.print(_MSG_RESULTS_LIST_HINT)
    except Exception as e:
        console.print(f"[red]Error loading result: {e}[/red]")


@results.command("compare")
@click.argument("result1")
@click.argument("result2")
@click.option(
    "--storage-dir", "-d", default=".traigent", help="Traigent storage directory"
)
def results_compare(result1: str, result2: str, storage_dir: str) -> None:
    """Compare two optimization runs side-by-side.

    Examples:
        traigent results compare run_v1 run_v2
    """
    console.print(f"\n[bold blue]Comparing: {result1} vs {result2}[/bold blue]\n")

    try:
        persistence = PersistenceManager(storage_dir)
        r1 = persistence.load_result(result1)
        r2 = persistence.load_result(result2)

        # Summary comparison table
        summary_table = _build_comparison_summary_table(r1, r2, result1, result2)
        console.print(summary_table)

        # Config comparison
        console.print("\n[bold]Best Configurations[/bold]")
        config_table = _build_config_comparison_table(
            r1.best_config, r2.best_config, result1, result2
        )
        console.print(config_table)

        # Metrics comparison
        metrics_table = _build_metrics_comparison_table(
            r1.best_metrics, r2.best_metrics, result1, result2
        )
        if metrics_table:
            console.print("\n[bold]Metrics Comparison[/bold]")
            console.print(metrics_table)

    except FileNotFoundError as e:
        console.print(f"[red]Result not found: {e}[/red]")
        console.print(_MSG_RESULTS_LIST_HINT)
    except Exception as e:
        console.print(f"[red]Error comparing results: {e}[/red]")


@results.command("rerank")
@click.argument("result_name")
@click.option(
    "--weights",
    "-w",
    required=True,
    help="Objective weights as key=value pairs (e.g., accuracy=0.7,cost=0.3)",
)
@click.option(
    "--storage-dir", "-d", default=".traigent", help="Traigent storage directory"
)
def results_rerank(result_name: str, weights: str, storage_dir: str) -> None:
    """Recalculate best config with different objective weights.

    This re-scores all trials using new weights without re-running optimization.

    Examples:
        traigent results rerank my_run --weights accuracy=0.8,cost=0.2
        traigent results rerank my_run -w "accuracy=0.5,latency=0.3,cost=0.2"
    """
    console.print(
        f"\n[bold blue]Re-ranking: {result_name} with custom weights[/bold blue]\n"
    )

    # Parse weights
    weight_dict = _parse_weights(weights)
    if weight_dict is None:
        console.print("[red]Invalid weights format[/red]")
        console.print("Use format: --weights accuracy=0.7,cost=0.3")
        return

    console.print("[bold]Weights (normalized):[/bold]")
    for k, v in weight_dict.items():
        console.print(f"  {k}: {v:.2%}")

    try:
        persistence = PersistenceManager(storage_dir)
        result = persistence.load_result(result_name)

        if not result.trials:
            console.print("[yellow]No trials found in this result[/yellow]")
            return

        # Re-score trials using helper
        scored_trials = [
            (score, trial)
            for trial in result.trials
            if (score := _calculate_trial_score(trial, weight_dict)) is not None
        ]

        if not scored_trials:
            console.print(
                "[yellow]No trials have the specified metrics. "
                f"Available metrics: {list(result.trials[0].metrics.keys())}[/yellow]"
            )
            return

        # Sort by new score (highest first)
        scored_trials.sort(key=lambda x: x[0], reverse=True)

        # Show results table
        console.print("\n[bold]Top 5 Configurations (re-ranked)[/bold]")
        console.print(_build_rerank_table(scored_trials))

        # Show new best config
        _, best_trial = scored_trials[0]
        console.print("\n[bold green]New Best Configuration:[/bold green]")
        config_syntax = Syntax(
            json.dumps(best_trial.config, indent=2), "json", theme="monokai"
        )
        console.print(config_syntax)

        # Compare with original best
        if result.best_config != best_trial.config:
            console.print(
                "\n[yellow]Note: This differs from the original best config.[/yellow]"
            )
            console.print("[dim]Original best config:[/dim]")
            console.print(json.dumps(result.best_config, indent=2))

    except FileNotFoundError:
        console.print(f"[red]Result '{result_name}' not found[/red]")
        console.print(_MSG_RESULTS_LIST_HINT)
    except Exception as e:
        console.print(f"[red]Error re-ranking: {e}[/red]")


@cli.command()
@click.argument("result_name")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path (default: configs/<result_name>.json)",
)
@click.option(
    "--format",
    "-f",
    "export_format",
    type=click.Choice(["slim", "full"]),
    default="slim",
    help="Export format: slim (~1KB, git-friendly) or full (includes trials)",
)
@click.option(
    "--storage-dir", "-d", default=".traigent", help="Traigent storage directory"
)
def export(
    result_name: str, output: str | None, export_format: str, storage_dir: str
) -> None:
    """Export an optimization result to a portable config file.

    This exports the best configuration from a previous optimization run
    to a JSON file suitable for version control and deployment.

    The slim format (~1KB) contains only:
    - config: The best parameter values
    - metrics: Objective scores achieved
    - function_name, exported_at, traigent_version

    The full format additionally includes:
    - Complete trial history
    - Optimization metadata
    - Configuration space

    Examples:
        traigent export my_run -o configs/prod.json      # Export for git
        traigent export my_run --format=full              # Include all data
        traigent export my_run                            # Default: configs/my_run.json
    """
    console.print(
        f"\n[bold blue]Exporting: {result_name} ({export_format} format)[/bold blue]\n"
    )

    try:
        persistence = PersistenceManager(storage_dir)
        result = persistence.load_result(result_name)

        if not result.best_config:
            console.print("[red]No best configuration found in this result[/red]")
            return

        # Determine output path
        if output is None:
            output_path = Path("configs") / f"{result_name}.json"
        else:
            output_path = Path(output)

        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build export data based on format
        from traigent import __version__

        if export_format == "slim":
            export_data: dict[str, Any] = {
                "config": result.best_config,
                "function_name": getattr(result, "function_name", result_name),
                "exported_at": __import__("datetime").datetime.now().isoformat(),
                "traigent_version": __version__,
            }
            if result.best_metrics:
                export_data["metrics"] = {
                    k: v
                    for k, v in result.best_metrics.items()
                    if not k.startswith("_")
                }
        else:  # full
            export_data = {
                "config": result.best_config,
                "function_name": getattr(result, "function_name", result_name),
                "exported_at": __import__("datetime").datetime.now().isoformat(),
                "traigent_version": __version__,
                "trials": [
                    {
                        "trial_id": t.trial_id,
                        "config": t.config,
                        "metrics": t.metrics,
                        "status": getattr(t, "status", "completed"),
                    }
                    for t in (result.trials or [])
                ],
                "optimization": {
                    "total_trials": len(result.trials),
                    "stop_reason": result.stop_reason,
                },
            }
            if result.best_metrics:
                export_data["metrics"] = result.best_metrics

        # Write to file with path validation
        safe_write_text(
            output_path,
            json.dumps(export_data, indent=2, default=str),
            WORKSPACE_ROOT,
            encoding="utf-8",
        )

        console.print(f"✅ [green]Exported to: {output_path}[/green]")

        # Show preview
        config_preview = json.dumps(result.best_config, indent=2)
        if len(config_preview) > 200:
            config_preview = config_preview[:197] + "..."
        console.print(f"\n[bold]Best config:[/bold]\n{config_preview}")

        # Show usage hint
        console.print("\n[dim]To use in production:[/dim]")
        console.print(
            f'[cyan]@traigent.optimize(load_from="{output_path}", ...)[/cyan]'
        )

    except FileNotFoundError:
        console.print(f"[red]Result '{result_name}' not found[/red]")
        console.print("Use 'traigent results' to see available results")
    except Exception as e:
        console.print(f"[red]Error exporting result: {e}[/red]")


@cli.command()
@click.argument("result_name")
@click.option(
    "--storage-dir", "-d", default=".traigent", help="Traigent storage directory"
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
        config_path = _resolve_workspace_path(
            Path(config_file),
            "Config file",
            must_exist=True,
        )
        with safe_open(config_path, WORKSPACE_ROOT, mode="r", encoding="utf-8") as f:
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

    output_path = _resolve_workspace_path(
        Path(output),
        "Output file",
    )

    # Check if file already exists
    if output_path.exists():
        if not click.confirm(f"File {output} already exists. Overwrite?"):
            console.print("[yellow]Generation cancelled[/yellow]")
            return

    # Write template to file
    safe_write_text(output_path, template_code, WORKSPACE_ROOT, encoding="utf-8")

    console.print(f"✅ [green]Template generated: {output}[/green]")

    # Show syntax highlighted preview
    syntax = Syntax(template_code, "python", theme="monokai", line_numbers=True)
    console.print("\n[bold]Preview:[/bold]")
    console.print(syntax)


@cli.command()
def examples() -> None:
    """Show comprehensive usage examples."""
    console.print("\n[bold blue]Traigent SDK Usage Examples[/bold blue]\n")

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
    return '''"""Basic Traigent optimization example."""

import asyncio
import traigent

# Dataset format (JSONL): {"input": {"question": "..."}, "output": "expected_result"}
# Use the example dataset or create your own:
DATASET = "data/qa_samples.jsonl"


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
    return '''"""Multi-objective optimization with Traigent."""

import asyncio
import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

# Dataset format (JSONL): {"input": {"question": "..."}, "output": "expected_result"}
# Use the example dataset or create your own:
DATASET = "data/qa_samples.jsonl"

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

    Traigent automatically tracks:
    - accuracy: compared against expected output
    - cost: token usage costs
    """
    # Your LLM call here - Traigent tracks metrics automatically
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
    return '''"""LangChain integration with Traigent."""

import asyncio
import traigent

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("LangChain not installed. Install with: pip install langchain-openai")
    exit(1)

# Dataset format (JSONL): {"input": {"question": "..."}, "output": "expected_result"}
# Use the example dataset or create your own:
DATASET = "data/qa_samples.jsonl"


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
    """LangChain-based agent that Traigent will optimize.

    Traigent automatically intercepts ChatOpenAI parameters and
    injects optimized values during trials.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Traigent will override
        temperature=0.7,      # Traigent will override
        max_tokens=150        # Traigent will override
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
    return '''"""OpenAI SDK integration with Traigent."""

import asyncio
import traigent
from traigent.integrations import enable_openai_optimization

# Enable OpenAI SDK optimization (intercepts openai.chat.completions.create)
enable_openai_optimization()

# Dataset format (JSONL): {"input": {"question": "..."}, "output": "expected_result"}
# Use the example dataset or create your own:
DATASET = "data/qa_samples.jsonl"


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
    """Chat function that Traigent will optimize."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Traigent will override this
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,  # Traigent will override this
        max_tokens=150    # Traigent will override this
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
    """Validate Traigent optimization improves over default parameters.

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

    _print_check_header(module_path, threshold, dry_run)

    try:
        function_filter = _parse_comma_separated_list(functions)
        objectives_filter = _parse_comma_separated_list(objectives)
        _print_filter_info(function_filter, objectives_filter)

        console.print("\n[bold yellow]📋 Step 1: Function Discovery[/bold yellow]")
        discovered_functions = discover_optimized_functions(
            module_path, function_filter
        )

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

        if objectives_filter:
            discovered_functions = _filter_functions_by_objectives(
                discovered_functions, objectives_filter
            )
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

        console.print(
            "\n[bold yellow]⚖️  Step 2: Optimization Validation[/bold yellow]"
        )
        validator = OptimizationValidator(threshold_pct=threshold)

        async def run_validations() -> tuple[list[Any], int, int]:
            results: list[Any] = []
            passed, failed = 0, 0
            for func_info in discovered_functions:
                console.print(f"\n[bold cyan]Validating: {func_info.name}[/bold cyan]")
                try:
                    result = await validator.validate_optimization(func_info)
                    results.append(result)
                    if result.should_block:
                        console.print(f"[red]{result.get_summary()}[/red]")
                        failed += 1
                    else:
                        console.print(f"[green]{result.get_summary()}[/green]")
                        passed += 1
                except Exception as e:
                    console.print(
                        f"[red]❌ Error validating {func_info.name}: {e}[/red]"
                    )
                    failed += 1
            return results, passed, failed

        results, passed_count, failed_count = asyncio.run(run_validations())
        _print_check_summary(discovered_functions, passed_count, failed_count, results)
        _handle_check_exit(failed_count)

    except Exception as e:
        console.print(f"[red]Error during validation: {e}[/red]")
        exit(1)


# Register local commands
register_edge_analytics_commands(cli)

cli.add_command(auth)
cli.add_command(hooks)
from traigent.cli.detect_tvars_command import detect_tvars  # noqa: E402

cli.add_command(detect_tvars)

from traigent.cli.generate_config_command import generate_config  # noqa: E402

cli.add_command(generate_config)

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
