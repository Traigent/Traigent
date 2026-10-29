"""Function discovery engine for Traigent optimization validation system."""

# Traceability: CONC-Layer-API CONC-Quality-Maintainability CONC-Quality-Usability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from rich.console import Console

from traigent.cli.validation_types import OptimizedFunction
from traigent.utils.logging import get_logger
from traigent.utils.module_loading import (
    find_optimized_functions,
    load_user_module,
    looks_like_dotted_module,
    resolve_dotted_module,
)
from traigent.utils.secure_path import PathTraversalError, validate_path

logger = get_logger(__name__)
console = Console()


def _resolve_module_path(module_path: str, workspace_root: str | Path | None) -> Path:
    workspace_base = (
        Path(workspace_root).expanduser().resolve()
        if workspace_root is not None
        else Path.cwd().resolve()
    )
    candidate = Path(module_path).expanduser()
    if looks_like_dotted_module(str(module_path)) and not candidate.exists():
        # ``pkg.mod``: locate the source file, then apply the same workspace
        # containment rule as for a path.
        candidate = resolve_dotted_module(str(module_path), workspace_base)
    try:
        module_path_obj = validate_path(
            candidate,
            workspace_base,
            must_exist=True,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Module file not found: {candidate}") from None
    except (PathTraversalError, ValueError) as exc:
        raise ValueError(
            f"Module path must reside inside the current workspace ({workspace_base}): {module_path}"
        ) from exc

    if module_path_obj.suffix != ".py":
        raise ValueError(f"Module must be a Python file (.py): {module_path_obj}")
    return module_path_obj


def discover_optimized_functions(
    module_path: str,
    function_filter: list[str] | None = None,
    workspace_root: str | Path | None = None,
) -> list[OptimizedFunction]:
    """Discover all functions decorated with @traigent.optimize in a module.

    Args:
        module_path: Path to Python file containing optimizable functions
        function_filter: Optional list of function names to include (None = all functions)
        workspace_root: Optional user workspace root. Defaults to the caller's CWD.

    Returns:
        List of OptimizedFunction objects containing function metadata

    Raises:
        ImportError: If module cannot be loaded
        ValueError: If module path is invalid
    """
    logger.info(f"Discovering optimized functions in {module_path}")

    module_path_obj = _resolve_module_path(module_path, workspace_root)

    module = load_user_module(module_path_obj)
    discovered_functions = []

    for name, obj in find_optimized_functions(module, function_filter):
        try:
            func_info = _extract_function_info(name, obj)
            discovered_functions.append(func_info)
            logger.debug(f"Discovered optimized function: {func_info}")
        except Exception as e:
            logger.warning(f"Failed to extract info from function {name}: {e}")
            continue

    logger.info(f"Discovered {len(discovered_functions)} optimized function(s)")
    return discovered_functions


def _extract_function_info(name: str, func: Any) -> OptimizedFunction:
    """Extract information from a decorated function.

    Args:
        name: Function name
        func: Function object (should have optimize method or be OptimizedFunction)

    Returns:
        OptimizedFunction with extracted metadata
    """
    if not hasattr(func, "optimize"):
        raise ValueError(
            f"Function {name} is not decorated with @traigent.optimize"
        ) from None

    # The decorator returns an OptimizedFunction instance
    optimized_wrapper = func

    # Extract decorator configuration from OptimizedFunction attributes
    decorator_config = {
        "eval_dataset": getattr(optimized_wrapper, "eval_dataset", None),
        "objectives": getattr(optimized_wrapper, "objectives", ["accuracy"]),
        "configuration_space": getattr(optimized_wrapper, "configuration_space", {}),
        "default_config": getattr(optimized_wrapper, "default_config", {}),
        "constraints": getattr(optimized_wrapper, "constraints", []),
        "injection_mode": getattr(optimized_wrapper, "injection_mode", "context"),
        "config_param": getattr(optimized_wrapper, "config_param", None),
    }

    # Get the original function (before decoration)
    original_func = getattr(optimized_wrapper, "func", None)
    if original_func is None:
        # If we can't get the original function, use the wrapper for signature analysis
        original_func = func

    # Extract default parameters from function signature
    default_params = _extract_default_parameters(original_func)

    # Extract objectives with proper type safety
    objectives_raw = decorator_config.get("objectives")
    if isinstance(objectives_raw, list):
        objectives = [str(obj) for obj in objectives_raw]
    else:
        objectives = ["accuracy"]

    return OptimizedFunction(
        name=name,
        func=optimized_wrapper,  # Use the wrapped function with optimize method
        decorator_config=decorator_config,
        default_params=default_params,
        eval_dataset=(
            str(decorator_config.get("eval_dataset"))
            if decorator_config.get("eval_dataset") is not None
            else None
        ),
        objectives=objectives,
    )


def _extract_default_parameters(func: Any) -> dict[str, Any]:
    """Extract default parameter values from function signature.

    Args:
        func: Function to inspect

    Returns:
        Dictionary mapping parameter names to default values
    """
    try:
        sig = inspect.signature(func)
        default_params = {}

        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                default_params[param_name] = param.default

        return default_params

    except Exception as e:
        logger.warning(f"Failed to extract default parameters from {func}: {e}")
        return {}


def classify_discovered_functions(
    functions: list[OptimizedFunction],
) -> tuple[list[str], list[str]]:
    """Split discovery findings into ``(blocking, notes)``.

    ``blocking`` holds exactly what the real run turns into a BLOCKED verdict
    (``check_prerequisites``); ``notes`` are advisory and never block.
    """
    from traigent.cli.optimization_validator import check_prerequisites

    blocking: list[str] = []
    notes: list[str] = []
    for func_info in functions:
        blocking.extend(
            f"{func_info.name}: {issue}" for issue in check_prerequisites(func_info)
        )
        if func_info.baseline_source == "none":
            notes.append(
                f"{func_info.name}: Note - no default_config and no signature "
                "defaults, so the baseline runs with an empty config. That is "
                "fine if the function handles an empty traigent.get_config(); "
                "add default_config={...} to the decorator to pin the baseline."
            )
    return blocking, notes


def validate_discovered_functions(functions: list[OptimizedFunction]) -> list[str]:
    """Validate discovered functions and return any issues found.

    Args:
        functions: List of discovered functions to validate

    Returns:
        Blocking issues followed by advisory notes (empty if all valid)
    """
    blocking, notes = classify_discovered_functions(functions)
    return blocking + notes


def print_discovery_summary(
    functions: list[OptimizedFunction],
    issues: list[str],
    notes: list[str] | None = None,
) -> None:
    """Print a summary of function discovery results.

    Args:
        functions: List of discovered functions
        issues: Blocking validation issues
        notes: Advisory, non-blocking notes
    """
    if not functions:
        console.print("[red]No functions with @traigent.optimize decorator found[/red]")
        console.print("Make sure your functions are decorated with @traigent.optimize")
        return

    console.print(f"\n[green]Found {len(functions)} optimizable function(s):[/green]")

    for func_info in functions:
        status_indicators = []
        if func_info.has_dataset:
            status_indicators.append("[green]📊 dataset[/green]")
        else:
            status_indicators.append("[red]❌ no dataset[/red]")

        if func_info.baseline_source == "default_config":
            status_indicators.append("[green]⚙️  default_config[/green]")
        elif func_info.baseline_source == "signature":
            status_indicators.append(
                f"[green]⚙️  {len(func_info.default_params)} defaults[/green]"
            )
        else:
            status_indicators.append("[dim]⚙️  empty baseline[/dim]")

        objectives_str = ", ".join(func_info.objectives)
        status_indicators.append(f"[cyan]🎯 {objectives_str}[/cyan]")

        indicators = " ".join(status_indicators)
        console.print(f"  • [bold]{func_info.name}[/bold]: {indicators}")

    if issues:
        console.print(f"\n[yellow]Validation Issues ({len(issues)}):[/yellow]")
        for issue in issues:
            console.print(f"  ⚠️  {issue}")
    if notes:
        console.print("\n[dim]Notes:[/dim]")
        for note in notes:
            console.print(f"  [dim]ℹ️  {note}[/dim]")
