"""Shared helper functions for walkthrough examples.

Common utilities for API key validation, logging setup, and environment configuration.
"""

from __future__ import annotations

import logging
import os
import runpy
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

import traigent
from traigent.config.backend_config import BackendConfig

if TYPE_CHECKING:
    from typing import Any

    from traigent.core.result import OptimizationResult


# Estimated times for real examples (in seconds) - from test_all_examples.sh
EXAMPLE_ESTIMATED_TIMES: dict[str, int] = {
    "01_tuning_qa.py": 94,  # ~1m 34s
    "02_zero_code_change.py": 78,  # ~1m 18s
    "03_parameter_mode.py": 76,  # ~1m 16s
    "04_multi_objective.py": 300,  # ~5m 0s (48 combinations, 6 models)
    "05_rag_parallel.py": 55,  # ~0m 55s
    "06_custom_evaluator.py": 73,  # ~1m 13s
    "07_multi_provider.py": 120,  # ~2m 0s (tests multiple providers)
    "08_privacy_modes.py": 104,  # ~1m 44s
}


def _format_duration(seconds: int) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"~{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    return f"~{minutes}m {secs}s"


def print_estimated_time(example_name: str) -> None:
    """Print estimated runtime for a real example.

    Skips output when running via test_all_examples.sh (which shows it already).

    Args:
        example_name: Name of the example file (e.g., "01_tuning_qa.py")
    """
    # Skip if running via shell script (which already shows estimated time)
    if os.getenv("TRAIGENT_BATCH_MODE", "").lower() in ("1", "true", "yes"):
        return
    estimated = EXAMPLE_ESTIMATED_TIMES.get(example_name)
    if estimated:
        print(f"Estimated time: {_format_duration(estimated)}")


def configure_logging() -> None:
    """Configure Traigent logging and cloud-first backend defaults for walkthroughs."""
    os.environ.setdefault("TRAIGENT_ENV", "development")
    os.environ.setdefault("TRAIGENT_BACKEND_URL", BackendConfig.get_cloud_backend_url())
    os.environ.setdefault("TRAIGENT_API_URL", BackendConfig.get_cloud_api_url())

    log_level = os.getenv("TRAIGENT_LOG_LEVEL", "WARNING").upper()
    traigent.configure(logging_level=log_level)

    # Suppress noisy warnings from third-party libraries
    logging.getLogger("tokencost.costs").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def is_valid_traigent_key(value: str) -> bool:
    """Validate Traigent API key format.

    Args:
        value: API key string to validate

    Returns:
        True if key matches expected format, False otherwise
    """
    prefix_lengths = {"tg_": 64, "uk_": 46}
    for prefix, expected_length in prefix_lengths.items():
        if value.startswith(prefix):
            if len(value) != expected_length:
                return False
            allowed = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"  # pragma: allowlist secret
            )
            return all(ch in allowed for ch in value[len(prefix) :])
    return False


def sanitize_traigent_api_key() -> None:
    """Remove invalid Traigent API keys from environment.

    Checks TRAIGENT_API_KEY; removes if invalid format.
    """
    key = os.getenv("TRAIGENT_API_KEY")
    if key and not is_valid_traigent_key(key):
        print(
            "WARNING: Ignoring invalid TRAIGENT_API_KEY for this run. "
            "Set a valid Traigent key to enable cloud features."
        )
        os.environ.pop("TRAIGENT_API_KEY", None)


def _is_truthy_env(var_name: str) -> bool:
    return os.getenv(var_name, "").lower() in ("1", "true", "yes")


def maybe_run_mock_example(
    example_path: str,
    *,
    required_env_vars: tuple[str, ...] = ("OPENAI_API_KEY",),
) -> None:
    """Run the matching mock walkthrough when required provider keys are missing.

    This keeps first-run walkthrough usage seamless: the user can invoke a real
    example directly, see a short warning, and still get a successful mock run.

    Args:
        example_path: Current script ``__file__`` path.
        required_env_vars: Provider key env vars that enable real execution.

    Raises:
        SystemExit: After executing the matching mock example.
    """
    should_force_mock = _is_truthy_env("TRAIGENT_MOCK_LLM")
    has_required_key = any(os.getenv(var_name) for var_name in required_env_vars)

    if not should_force_mock and has_required_key:
        return

    example_file = Path(example_path).resolve()
    mock_path = example_file.parent.parent / "mock" / example_file.name
    if not mock_path.exists():
        missing_keys = ", ".join(required_env_vars)
        raise SystemExit(
            f"No mock fallback exists for {example_file.name}. "
            f"Set one of: {missing_keys}"
        )

    repo_root = example_file.parents[2]
    display_path = mock_path.relative_to(repo_root).as_posix()
    if should_force_mock:
        reason = "TRAIGENT_MOCK_LLM is enabled"
    else:
        reason = f"Missing {' or '.join(required_env_vars)}"

    print(
        f"WARNING: {reason}. Running {display_path} instead. "
        "Set the required API key to use real LLM calls."
    )

    os.environ["TRAIGENT_MOCK_LLM"] = "true"
    os.environ.setdefault("OPENAI_API_KEY", "mock-key-for-demos")
    os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")
    runpy.run_path(str(mock_path), run_name="__main__")
    raise SystemExit(0)


def require_openai_key(example_name: str) -> None:
    """Exit with error if OPENAI_API_KEY is not set.

    Args:
        example_name: Name of the example file (used in error message)

    Raises:
        SystemExit: If OPENAI_API_KEY environment variable is not set
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Export it to run real examples: "
            'export OPENAI_API_KEY="your-key". '  # pragma: allowlist secret
            f"To run without a key, use walkthrough/mock/{example_name}."
        )


def setup_example_logger(name: str) -> logging.Logger:
    """Create a logger for walkthrough examples with simple formatting.

    Args:
        name: Logger name (typically the example module name)

    Returns:
        Configured logger instance
    """
    example_logger = logging.getLogger(f"traigent.walkthrough.{name}")
    if not example_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        example_logger.addHandler(handler)
    example_logger.setLevel(logging.INFO)
    example_logger.propagate = False
    return example_logger


def _format_objective(obj: Any) -> str:
    """Format a single objective for display."""
    name = obj.name if hasattr(obj, "name") else str(obj)
    weight = getattr(obj, "weight", None)
    orientation = getattr(obj, "orientation", None)
    if weight is not None and orientation:
        return f"{name} ({orientation}, w={weight})"
    return name


def _format_objectives(objectives: Any) -> str:
    """Format objectives for display."""
    if hasattr(objectives, "objectives"):
        return ", ".join(_format_objective(obj) for obj in objectives.objectives)
    if isinstance(objectives, (list, tuple)):
        return ", ".join(str(o) for o in objectives)
    return str(objectives)


def print_optimization_config(
    objectives: Any,
    configuration_space: dict[str, list[Any]],
) -> None:
    """Print optimization configuration summary before running.

    Args:
        objectives: List of objective names or ObjectiveSchema instance
        configuration_space: Dictionary mapping parameter names to their possible values
    """
    objectives_str = _format_objectives(objectives)

    # Calculate combinations
    param_counts = [len(values) for values in configuration_space.values()]
    total_combinations = reduce(lambda x, y: x * y, param_counts, 1)

    # Build breakdown string (e.g., "4 models × 2 temperatures")
    breakdown_parts = [
        f"{len(values)} {param}" for param, values in configuration_space.items()
    ]
    breakdown_str = " × ".join(breakdown_parts)

    print("\nOptimization Configuration:")
    print(f"  Objectives: {objectives_str}")
    print(f"  Configuration space: {total_combinations} combinations ({breakdown_str})")
    for param, values in configuration_space.items():
        values_str = ", ".join(repr(v) for v in values)
        print(f"    - {param}: [{values_str}]")


def _find_best_trial(trials: list, metric_names: list[str]) -> Any:
    """Find the best trial by weighted score or accuracy."""
    best_trial = trials[0]
    for trial in trials:
        score = getattr(trial, "weighted_score", None)
        if score is not None:
            best_score = getattr(best_trial, "weighted_score", float("-inf"))
            if best_score < score:
                best_trial = trial
        elif metric_names and "accuracy" in metric_names:
            trial_acc = getattr(trial, "metrics", {}).get("accuracy", 0)
            best_acc = getattr(best_trial, "metrics", {}).get("accuracy", 0)
            if trial_acc > best_acc:
                best_trial = trial
    return best_trial


def _format_config_value(val: Any) -> str:
    """Format a configuration value for display."""
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def _format_metric_value(metric: str, val: float) -> str:
    """Format a metric value for display."""
    if metric == "cost":
        return f"${val:.5f}"
    if metric == "latency":
        return f"{val:.3f}s"
    return f"{val:.1%}"


def _get_objective_info(objectives: Any) -> list[tuple[str, str]]:
    """Extract objective names and orientations from objectives definition.

    Returns list of (name, orientation) tuples. Orientation is 'maximize' or 'minimize'.
    """
    if hasattr(objectives, "objectives"):
        return [
            (obj.name, getattr(obj, "orientation", "maximize"))
            for obj in objectives.objectives
        ]
    if isinstance(objectives, (list, tuple)):
        # Default orientations: accuracy=maximize, cost/latency=minimize
        result = []
        for o in objectives:
            name = str(o)
            if name in ("cost", "latency"):
                result.append((name, "minimize"))
            else:
                result.append((name, "maximize"))
        return result
    return []


def _find_best_per_objective(
    trials: list, metric_info: list[tuple[str, str]]
) -> dict[str, int]:
    """Find the best trial index for each objective metric.

    Returns dict mapping metric name to trial index (0-based).
    """
    best_indices = {}
    for metric_name, orientation in metric_info:
        best_idx = 0
        best_val = getattr(trials[0], "metrics", {}).get(metric_name, 0)
        # For minimize objectives, we want the smallest value
        # For maximize objectives, we want the largest value
        is_minimize = orientation == "minimize"
        for i, trial in enumerate(trials[1:], 1):
            val = getattr(trial, "metrics", {}).get(metric_name, 0)
            is_better = val < best_val if is_minimize else val > best_val
            if is_better:
                best_val = val
                best_idx = i
        best_indices[metric_name] = best_idx
    return best_indices


# ANSI color codes
class _Colors:
    """ANSI color codes for terminal output."""

    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        """Disable colors (for non-TTY output)."""
        cls.BOLD = cls.GREEN = cls.YELLOW = cls.CYAN = cls.DIM = cls.RESET = ""


def _check_color_support() -> bool:
    """Check if terminal supports colors."""
    import sys

    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return os.getenv("NO_COLOR") is None and os.getenv("TERM") != "dumb"


def _calc_col_widths(
    trials: list, param_names: list[str], metric_names: list[str]
) -> dict[str, int]:
    """Calculate column widths dynamically based on content."""
    col_widths: dict[str, int] = {"#": 4}
    for param in param_names:
        max_len = max(
            len(_format_config_value(getattr(t, "config", {}).get(param, "?")))
            for t in trials
        )
        col_widths[param] = max(len(param), max_len) + 1
    for metric in metric_names:
        max_len = max(
            len(_format_metric_value(metric, getattr(t, "metrics", {}).get(metric, 0)))
            for t in trials
        )
        col_widths[metric] = max(len(metric), max_len) + 1
    return col_widths


def _build_table_row(
    trial: Any,
    idx: int,
    param_names: list[str],
    metric_names: list[str],
    col_widths: dict[str, int],
    best_per_objective: dict[str, int],
    is_overall_best: bool,
    mock_metrics: dict[str, float] | None = None,
    trailing_params: list[str] | None = None,
) -> list[str]:
    """Build a single table row string."""
    C = _Colors
    config = getattr(trial, "config", {})
    metrics = getattr(trial, "metrics", {})

    # Row prefix for best overall
    row_prefix = f"{C.GREEN}★{C.RESET}" if is_overall_best else " "
    row_parts = [f"{row_prefix}{idx + 1:>{col_widths['#'] - 1}}"]

    # Config values
    for param in param_names:
        val = _format_config_value(config.get(param, "?"))
        row_parts.append(f"{val:^{col_widths[param]}}")

    # Metric values with highlighting for best
    for metric in metric_names:
        # Use mock_metrics override if provided (for simulated costs in mock mode)
        if mock_metrics and metric in mock_metrics:
            val = mock_metrics[metric]
        else:
            val = metrics.get(metric, 0)
        formatted = _format_metric_value(metric, val)
        if best_per_objective.get(metric) == idx:
            cell = f"{C.GREEN}{C.BOLD}{formatted:^{col_widths[metric]}}{C.RESET}"
        else:
            cell = f"{formatted:^{col_widths[metric]}}"
        row_parts.append(cell)

    # Trailing config values (shown after metrics)
    if trailing_params:
        for param in trailing_params:
            val = _format_config_value(config.get(param, "?"))
            row_parts.append(f"{val:^{col_widths[param]}}")

    return row_parts


def _get_mock_cost_for_trial(trial: Any, task_type: str, dataset_size: int) -> float:
    """Calculate mock cost for a trial based on model."""
    from utils.mock_answers import get_mock_cost

    config = getattr(trial, "config", {})
    model = config.get("model", "gpt-3.5-turbo")
    return get_mock_cost(model, task_type, dataset_size)


def _get_mock_latency_for_trial(trial: Any, task_type: str) -> float:
    """Calculate mock latency for a trial based on model."""
    from utils.mock_answers import get_mock_latency

    config = getattr(trial, "config", {})
    model = config.get("model", "gpt-3.5-turbo")
    return get_mock_latency(model, task_type)


def print_results_table(
    results: OptimizationResult,
    config_space: dict[str, list[Any]],
    objectives: Any,
    is_mock: bool = False,
    task_type: str = "simple_qa",
    dataset_size: int = 20,
) -> None:
    """Print a comparison table of all trial results.

    Args:
        results: The optimization results from Traigent
        config_space: Configuration space used for optimization
        objectives: The objectives used for optimization
        is_mock: Whether this is a mock run (affects displayed label and cost simulation)
        task_type: Task type for mock cost estimation (default: simple_qa)
        dataset_size: Dataset size for mock cost estimation (default: 20)
    """
    trials = getattr(results, "trials", [])
    if not trials:
        print("\nNo trials to display.")
        return

    # Check color support
    if not _check_color_support():
        _Colors.disable()
    C = _Colors

    # Get objective and parameter info
    objective_info = _get_objective_info(objectives)
    sample_metrics = getattr(trials[0], "metrics", {})
    metric_info = [(n, o) for n, o in objective_info if n in sample_metrics]
    metric_names = [n for n, _ in metric_info]

    # All config params shown in order from config_space
    param_names = list(config_space.keys())
    trailing_params = []  # No trailing params - all shown in config_space order

    # Calculate mock costs and latencies for each trial (for mock mode)
    mock_costs: list[float] = []
    mock_latencies: list[float] = []
    if is_mock and "cost" in metric_names:
        mock_costs = [
            _get_mock_cost_for_trial(t, task_type, dataset_size) for t in trials
        ]
    if is_mock and "latency" in metric_names:
        mock_latencies = [_get_mock_latency_for_trial(t, task_type) for t in trials]

    # Find best trials (using mock values if in mock mode)
    if mock_costs or mock_latencies:
        # Create modified metric_info for best calculation with mock values
        best_per_objective = {}
        for metric_name, orientation in metric_info:
            if metric_name == "cost" and mock_costs:
                # Use mock costs for finding best cost
                is_minimize = orientation == "minimize"
                best_idx = 0
                best_val = mock_costs[0]
                for i, cost in enumerate(mock_costs[1:], 1):
                    is_better = cost < best_val if is_minimize else cost > best_val
                    if is_better:
                        best_val = cost
                        best_idx = i
                best_per_objective[metric_name] = best_idx
            elif metric_name == "latency" and mock_latencies:
                # Use mock latencies for finding best latency
                is_minimize = orientation == "minimize"
                best_idx = 0
                best_val = mock_latencies[0]
                for i, latency in enumerate(mock_latencies[1:], 1):
                    is_better = (
                        latency < best_val if is_minimize else latency > best_val
                    )
                    if is_better:
                        best_val = latency
                        best_idx = i
                best_per_objective[metric_name] = best_idx
            else:
                # Use actual metrics for other objectives
                best_idx = 0
                best_val = getattr(trials[0], "metrics", {}).get(metric_name, 0)
                is_minimize = orientation == "minimize"
                for i, trial in enumerate(trials[1:], 1):
                    val = getattr(trial, "metrics", {}).get(metric_name, 0)
                    is_better = val < best_val if is_minimize else val > best_val
                    if is_better:
                        best_val = val
                        best_idx = i
                best_per_objective[metric_name] = best_idx
    else:
        best_per_objective = _find_best_per_objective(trials, metric_info)

    # Use proper weighted scoring from OptimizationResult
    # Note: In mock mode, trial.metrics may have cost=0 and similar latency for all
    # trials, so weighted scoring may favor accuracy. The displayed costs/latencies
    # are estimates for visualization only.
    try:
        weighted_result = results.calculate_weighted_scores(objective_schema=objectives)
        best_config = weighted_result.get("best_weighted_config", {})
    except Exception:
        # Fallback to simple accuracy-based selection
        best_trial = _find_best_trial(trials, metric_names)
        best_config = getattr(best_trial, "config", {})

    # Calculate column widths (use mock costs for width calculation if in mock mode)
    col_widths: dict[str, int] = {"#": 4}
    for param in param_names:
        max_len = max(
            len(_format_config_value(getattr(t, "config", {}).get(param, "?")))
            for t in trials
        )
        col_widths[param] = max(len(param), max_len) + 1
    for idx, metric in enumerate(metric_names):
        if metric == "cost" and mock_costs:
            max_len = max(len(_format_metric_value(metric, c)) for c in mock_costs)
        elif metric == "latency" and mock_latencies:
            max_len = max(
                len(_format_metric_value(metric, lat)) for lat in mock_latencies
            )
        else:
            max_len = max(
                len(
                    _format_metric_value(
                        metric, getattr(t, "metrics", {}).get(metric, 0)
                    )
                )
                for t in trials
            )
        col_widths[metric] = max(len(metric), max_len) + 1
    # Calculate column widths for trailing params (shown after metrics)
    for param in trailing_params:
        max_len = max(
            len(_format_config_value(getattr(t, "config", {}).get(param, "?")))
            for t in trials
        )
        col_widths[param] = max(len(param), max_len) + 1

    # Add trailing params after metrics
    all_cols = ["#"] + param_names + metric_names + trailing_params
    total_width = sum(col_widths[c] for c in all_cols) + len(all_cols) * 3 - 1

    # Box drawing characters
    H, V = "─", "│"
    TL, TR, BL, BR = "┌", "┐", "└", "┘"
    B, L, R, X = "┴", "├", "┤", "┼"

    # Print table header
    mode_label = "MOCK" if is_mock else "REAL"
    title = f" Trial Results ({mode_label} - {len(trials)} trials) "
    padding = (total_width - len(title)) // 2

    print()
    print(f"{C.BOLD}{TL}{H * total_width}{TR}{C.RESET}")
    print(
        f"{C.BOLD}{V}{' ' * padding}{title}{' ' * (total_width - padding - len(title))}{V}{C.RESET}"
    )
    print(f"{C.BOLD}{L}{H * total_width}{R}{C.RESET}")

    # Column headers
    header_parts = [f"{C.BOLD}{'#':^{col_widths['#']}}{C.RESET}"]
    header_parts.extend(f"{C.CYAN}{p:^{col_widths[p]}}{C.RESET}" for p in param_names)
    header_parts.extend(
        f"{C.YELLOW}{m:^{col_widths[m]}}{C.RESET}" for m in metric_names
    )
    # Trailing params after metrics (cyan like other config params)
    header_parts.extend(
        f"{C.CYAN}{p:^{col_widths[p]}}{C.RESET}" for p in trailing_params
    )
    print(f"{V} " + f" {V} ".join(header_parts) + f" {V}")

    # Separator
    print(f"{L}" + X.join(H * (col_widths[c] + 2) for c in all_cols) + f"{R}")

    # Data rows
    for i, trial in enumerate(trials):
        config = getattr(trial, "config", {})
        is_overall_best = config == best_config

        # Build mock_metrics override for this trial
        mock_metrics = {}
        if mock_costs:
            mock_metrics["cost"] = mock_costs[i]
        if mock_latencies:
            mock_metrics["latency"] = mock_latencies[i]

        row_parts = _build_table_row(
            trial,
            i,
            param_names,
            metric_names,
            col_widths,
            best_per_objective,
            is_overall_best,
            mock_metrics if mock_metrics else None,
            trailing_params,
        )
        print(f"{V} " + f" {V} ".join(row_parts) + f" {V}")

    # Bottom border
    print(f"{BL}" + B.join(H * (col_widths[c] + 2) for c in all_cols) + f"{BR}")

    # Legend
    legend = [f"{C.GREEN}★{C.RESET} Overall Best"]
    legend.extend(f"{C.GREEN}{C.BOLD}{m}{C.RESET} = Best {m}" for m in metric_names)
    print(f"{C.DIM}Legend: {', '.join(legend)}{C.RESET}")

    if is_mock:
        print(
            f"{C.DIM}Note: Mock mode - costs and latencies are estimated based on model characteristics.{C.RESET}"
        )


def print_cost_estimate(
    models: list[str],
    dataset_size: int,
    task_type: str = "simple_qa",
    num_trials: int | None = None,
) -> None:
    """Print cost estimate before running real examples.

    Args:
        models: List of model names to test
        dataset_size: Number of examples in evaluation dataset
        task_type: Type of task (affects token estimates)
        num_trials: Number of optimization trials (if None, assumes testing all models)
    """
    from utils.cost_estimator import estimate_cost

    estimate = estimate_cost(
        models=models,
        dataset_size=dataset_size,
        task_type=task_type,
        num_trials=num_trials,
    )

    total = estimate["total_cost"]
    range_min = estimate["cost_range"]["min"]
    range_max = estimate["cost_range"]["max"]

    print("\n⚠️  Cost Estimate:")
    print(f"   Estimated: ${total:.2f} (range: ${range_min:.2f} - ${range_max:.2f})")
    print(f"   Based on {dataset_size} examples × {estimate['num_trials']} trials")
    print("   Prices are estimates and may vary.")
