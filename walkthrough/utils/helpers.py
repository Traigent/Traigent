"""Shared helper functions for walkthrough examples.

Common utilities for API key validation, logging setup, and environment configuration.
"""

from __future__ import annotations

import logging
import os
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

import traigent
from traigent.config.backend_config import BackendConfig

if TYPE_CHECKING:
    from typing import Any

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


def _find_repo_root(start_path: Path) -> Path:
    """Walk upward until the repository root (identified by pyproject.toml)."""
    for candidate in (start_path.parent, *start_path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start_path.parents[2]


def _required_real_example_message(
    example_file: Path,
    required_env_vars: tuple[str, ...],
) -> str:
    """Build a clear hard-failure message for real walkthrough examples."""
    repo_root = _find_repo_root(example_file)
    real_display_path = example_file.relative_to(repo_root).as_posix()
    mock_display_path = (
        (example_file.parent.parent / "mock" / example_file.name)
        .relative_to(repo_root)
        .as_posix()
    )
    env_list = " or ".join(required_env_vars)
    primary_env = required_env_vars[0]
    return (
        f"ERROR: {env_list} environment variable is required for real examples.\n\n"
        "To run this example:\n"
        f"  export {primary_env}=your-key-here\n"
        f"  python {real_display_path}\n\n"
        "For mock examples without API keys, use:\n"
        f"  python {mock_display_path}"
    )


def maybe_run_mock_example(
    example_path: str,
    *,
    required_env_vars: tuple[str, ...] = ("OPENAI_API_KEY",),
) -> None:
    """Validate that real walkthroughs have the provider keys they require.

    Args:
        example_path: Current script ``__file__`` path.
        required_env_vars: Provider key env vars that enable real execution.

    Raises:
        SystemExit: When the real example should not proceed.
    """
    should_force_mock = os.getenv("TRAIGENT_MOCK_LLM", "").lower() in (
        "1",
        "true",
        "yes",
    )
    has_required_key = any(os.getenv(var_name) for var_name in required_env_vars)
    example_file = Path(example_path).resolve()
    if should_force_mock:
        raise SystemExit(
            "ERROR: walkthrough/real examples do not fall back to mock mode. "
            "Unset TRAIGENT_MOCK_LLM and provide the required API key, or run the "
            f"mock example directly.\n\n{_required_real_example_message(example_file, required_env_vars)}"
        )
    if not has_required_key:
        raise SystemExit(
            _required_real_example_message(example_file, required_env_vars)
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


def _get_mock_cost_for_trial(trial: Any, task_type: str, dataset_size: int) -> float:
    """Calculate mock cost for a trial based on model."""
    from utils.mock_answers import get_mock_cost

    config = getattr(trial, "config", {})
    model = config.get("model", "gpt-3.5-turbo")
    return float(get_mock_cost(model, task_type, dataset_size))


def _get_mock_latency_for_trial(trial: Any, task_type: str) -> float:
    """Calculate mock latency for a trial based on model, in MILLISECONDS.

    The results table renders ``latency`` as ``f"{val:.0f}ms"``, so the seconds
    value from the mock table must be scaled or every cell renders ``0ms``.
    """
    from utils.mock_answers import get_mock_latency

    config = getattr(trial, "config", {})
    model = config.get("model", "gpt-3.5-turbo")
    return float(get_mock_latency(model, task_type)) * 1000.0


def _build_metric_overrides(
    result: Any,
    *,
    is_mock: bool,
    task_type: str,
    dataset_size: int,
    reported_metrics: tuple[str, ...] = (),
) -> dict[str, list[float]] | None:
    """Build walkthrough-only display metric overrides for mock runs.

    Metrics the example reports itself (``reported_metrics``) are the numbers
    that actually drove selection, so they are displayed as recorded. Every other
    mock metric is an artifact of local execution - a ``cost`` of 0.0 the example
    never reported, or the wall-clock of a scaled-down ``time.sleep`` - and is
    replaced with the static-table estimate the footer note promises.
    """
    if not is_mock:
        return None

    trials = getattr(result, "trials", [])
    if not trials:
        return None

    sample_metrics = getattr(trials[0], "metrics", {}) or {}
    overrides: dict[str, list[float]] = {}
    if "cost" in sample_metrics and "cost" not in reported_metrics:
        overrides["cost"] = [
            _get_mock_cost_for_trial(trial, task_type, dataset_size) for trial in trials
        ]
    if "latency" in sample_metrics and "latency" not in reported_metrics:
        overrides["latency"] = [
            _get_mock_latency_for_trial(trial, task_type) for trial in trials
        ]
    return overrides or None


def build_results_table_callback(
    *,
    is_mock: bool = False,
    task_type: str = "simple_qa",
    dataset_size: int = 20,
    show_progress: bool = False,
    reported_metrics: tuple[str, ...] = (),
) -> Any:
    """Build the single SDK results-table callback used by walkthroughs.

    The SDK owns rendering. This helper only supplies walkthrough display context:
    MOCK/REAL title labels and estimated mock cost/latency overrides.

    Args:
        is_mock: Whether the example runs against simulated LLM responses.
        task_type: Mock-table task key used for the cost/latency estimates.
        dataset_size: Example count the estimated per-trial cost is scaled by.
        show_progress: Render a live progress bar instead of the final table only.
        reported_metrics: Metrics this example simulates itself from the shared
            mock tables (``traigent.with_usage`` for ``cost``, a ``latency``
            entry in ``metric_functions``). They drove config selection, so the
            table shows them exactly as recorded instead of re-estimating them.
    """
    from traigent.utils.callbacks import ProgressBarCallback, ResultsTableCallback

    mode_label = "MOCK" if is_mock else "REAL"
    footer_note = (
        "Note: Mock mode - cost and latency are simulated from a static pricing/latency "
        "table - no real API spend."
        if is_mock
        else None
    )

    def metric_override_factory(result: Any) -> dict[str, list[float]] | None:
        return _build_metric_overrides(
            result,
            is_mock=is_mock,
            task_type=task_type,
            dataset_size=dataset_size,
            reported_metrics=reported_metrics,
        )

    callback_cls = ProgressBarCallback if show_progress else ResultsTableCallback
    return callback_cls(
        mode_label=mode_label,
        metric_override_factory=metric_override_factory,
        footer_note=footer_note,
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
