"""Function discovery engine for Traigent optimization validation system."""

# Traceability: CONC-Layer-API CONC-Quality-Maintainability CONC-Quality-Usability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from traigent.cli.validation_types import OptimizedFunction
from traigent.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def discover_optimized_functions(
    module_path: str, function_filter: list[str] | None = None
) -> list[OptimizedFunction]:
    """Discover all functions decorated with @traigent.optimize in a module.

    Args:
        module_path: Path to Python file containing optimizable functions
        function_filter: Optional list of function names to include (None = all functions)

    Returns:
        List of OptimizedFunction objects containing function metadata

    Raises:
        ImportError: If module cannot be loaded
        ValueError: If module path is invalid
    """
    logger.info(f"Discovering optimized functions in {module_path}")

    # Validate and resolve module path
    module_path_obj = Path(module_path).expanduser().resolve()

    # Security check for path traversal
    if ".." in str(module_path_obj) or str(module_path_obj).startswith("/etc"):
        raise ValueError(f"Invalid module path: {module_path_obj}")
    try:
        module_path_obj.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Module path must reside inside the Traigent workspace ({PROJECT_ROOT}): {module_path_obj}"
        ) from exc

    if not module_path_obj.exists():
        raise FileNotFoundError(f"Module file not found: {module_path_obj}")

    if not module_path_obj.suffix == ".py":
        raise ValueError(f"Module must be a Python file (.py): {module_path_obj}")

    # Load the Python file as a module
    parent_dir = module_path_obj.parent
    sys.path.insert(0, str(parent_dir))

    discovered_functions = []

    try:
        # Import the module
        spec = importlib.util.spec_from_file_location("user_module", module_path_obj)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec from {module_path_obj}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find functions with optimize method (decorated with @traigent.optimize)
        for name, obj in inspect.getmembers(module):
            # Skip private/internal attributes and modules
            if name.startswith("_") or inspect.ismodule(obj):
                continue

            # Check if it's a regular function with optimize method OR an OptimizedFunction instance
            is_optimized_function = False

            if inspect.isfunction(obj) and hasattr(obj, "optimize"):
                is_optimized_function = True
            elif (
                hasattr(obj, "__class__")
                and obj.__class__.__name__ == "OptimizedFunction"
            ):
                is_optimized_function = True
            elif hasattr(obj, "optimize") and callable(getattr(obj, "optimize", None)):
                # Additional check to ensure this isn't just a module or other object
                if hasattr(obj, "func") or callable(obj):
                    is_optimized_function = True

            if not is_optimized_function:
                continue

            # Apply function filter if provided
            if function_filter is not None and name not in function_filter:
                continue

            try:
                # Extract function information
                func_info = _extract_function_info(name, obj)
                discovered_functions.append(func_info)
                logger.debug(f"Discovered optimized function: {func_info}")

            except Exception as e:
                logger.warning(f"Failed to extract info from function {name}: {e}")
                continue

    finally:
        # Remove from Python path
        if str(parent_dir) in sys.path:
            sys.path.remove(str(parent_dir))

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


def validate_discovered_functions(functions: list[OptimizedFunction]) -> list[str]:
    """Validate discovered functions and return any issues found.

    Args:
        functions: List of discovered functions to validate

    Returns:
        List of validation issue messages (empty if all valid)
    """
    issues = []

    for func_info in functions:
        # Check for evaluation dataset
        if not func_info.has_dataset:
            issues.append(f"{func_info.name}: Missing evaluation dataset")

        # Check for objectives
        if not func_info.objectives:
            issues.append(f"{func_info.name}: No objectives specified")

        # Check for configuration space
        config_space = func_info.decorator_config.get("configuration_space", {})
        if not config_space:
            issues.append(f"{func_info.name}: No configuration space specified")

        # Warn about missing defaults (not blocking)
        if not func_info.has_defaults:
            issues.append(
                f"{func_info.name}: Warning - No default parameters found in function signature"
            )

    return issues


def print_discovery_summary(
    functions: list[OptimizedFunction], issues: list[str]
) -> None:
    """Print a summary of function discovery results.

    Args:
        functions: List of discovered functions
        issues: List of validation issues
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

        if func_info.has_defaults:
            status_indicators.append(
                f"[green]⚙️  {len(func_info.default_params)} defaults[/green]"
            )
        else:
            status_indicators.append("[yellow]⚠️  no defaults[/yellow]")

        objectives_str = ", ".join(func_info.objectives)
        status_indicators.append(f"[cyan]🎯 {objectives_str}[/cyan]")

        indicators = " ".join(status_indicators)
        console.print(f"  • [bold]{func_info.name}[/bold]: {indicators}")

    if issues:
        console.print(f"\n[yellow]Validation Issues ({len(issues)}):[/yellow]")
        for issue in issues:
            console.print(f"  ⚠️  {issue}")
