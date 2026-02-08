"""Pipeline construction helpers for OptimizedFunction.optimize().

Pure functions that resolve, validate, and assemble the components
needed by the optimization orchestrator.  Extracted from
``OptimizedFunction`` to reduce god-object size.
"""

# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from traigent.api.functions import _GLOBAL_CONFIG, get_global_parallel_config
from traigent.config.parallel import (
    coerce_parallel_config,
    merge_parallel_configs,
    resolve_parallel_config,
)
from traigent.config.types import ExecutionMode, TraigentConfig, resolve_execution_mode
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.env_config import is_mock_llm
from traigent.utils.logging import get_logger
from traigent.utils.validation import validate_config_space

if TYPE_CHECKING:
    from traigent.config.parallel import ParallelConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Parameter resolution
# ---------------------------------------------------------------------------


def resolve_execution_parameters(
    algorithm: str | None,
    max_trials: int | None,
    configuration_space: dict[str, Any] | None,
    *,
    fallback_config_space: dict[str, Any],
    fallback_algorithm: str,
    fallback_max_trials: int | None,
) -> tuple[str, int | None, dict[str, Any]]:
    """Resolve and validate execution parameters.

    Args:
        algorithm: Algorithm from optimize() call
        max_trials: Max trials from optimize() call
        configuration_space: Config space from optimize() call
        fallback_config_space: Config space from decorator
        fallback_algorithm: Algorithm from decorator
        fallback_max_trials: Max trials from decorator

    Returns:
        Tuple of (resolved_algorithm, resolved_max_trials, effective_config_space)
    """
    effective_config_space = (
        configuration_space
        if configuration_space is not None
        else fallback_config_space
    )

    resolved_algorithm: str = algorithm if algorithm else fallback_algorithm

    resolved_max_trials = max_trials if max_trials is not None else fallback_max_trials
    if resolved_max_trials is not None and resolved_max_trials < 0:
        raise ValueError("max_trials must be non-negative")

    if not effective_config_space or effective_config_space == {}:
        raise ValueError(
            "Configuration space cannot be empty. Please specify at least one parameter to optimize "
            "either in the @traigent.optimize decorator or in the optimize() method call. "
            "Example: optimize(configuration_space={'temperature': [0.0, 0.5, 1.0]})"
        )

    if configuration_space is not None:
        validate_config_space(configuration_space)

    return resolved_algorithm, resolved_max_trials, effective_config_space


# ---------------------------------------------------------------------------
# Phase 2: TraigentConfig creation
# ---------------------------------------------------------------------------


def create_traigent_config(
    execution_mode: str,
    local_storage_path: str | None,
    minimal_logging: bool,
    privacy_enabled: bool,
) -> TraigentConfig:
    """Create TraigentConfig for the optimization run.

    Args:
        execution_mode: Execution mode string
        local_storage_path: Optional local storage path
        minimal_logging: Whether minimal logging is enabled
        privacy_enabled: Whether privacy mode is enabled

    Returns:
        Configured TraigentConfig instance
    """
    return TraigentConfig(
        execution_mode=cast(
            Literal[
                "edge_analytics",
                "privacy",
                "hybrid",
                "standard",
                "cloud",
                "hybrid_api",
            ],
            execution_mode,
        ),
        local_storage_path=local_storage_path,
        minimal_logging=minimal_logging,
        privacy_enabled=privacy_enabled,
    )


# ---------------------------------------------------------------------------
# Phase 3: Dataset preparation
# ---------------------------------------------------------------------------


def prepare_optimization_dataset(
    dataset: Dataset,
    algorithm_kwargs: dict[str, Any],
    *,
    max_examples: int | None,
    max_total_examples: int | None,
    samples_include_pruned: bool,
) -> tuple[Dataset, int | None, bool]:
    """Prepare the optimization dataset by applying caps and extracting kwargs.

    Args:
        dataset: Loaded dataset
        algorithm_kwargs: Algorithm kwargs (may be mutated to pop keys)
        max_examples: Maximum examples per trial
        max_total_examples: Maximum total examples across all trials
        samples_include_pruned: Whether pruned samples count toward budget

    Returns:
        Tuple of (dataset, max_total_examples, samples_include_pruned)
    """
    max_examples_value = algorithm_kwargs.get("max_examples") or max_examples
    if (
        max_examples_value is not None
        and isinstance(max_examples_value, int)
        and max_examples_value > 0
        and len(dataset.examples) > max_examples_value
    ):
        capped_dataset = Dataset(
            examples=dataset.examples[:max_examples_value],
            name=dataset.name if hasattr(dataset, "name") else "dataset",
            description=f"{dataset.description if hasattr(dataset, 'description') else 'Dataset'} (capped to {max_examples_value} examples)",
            metadata={
                **getattr(dataset, "metadata", {}),
                "original_count": len(dataset.examples),
                "capped_count": max_examples_value,
            },
        )
        logger.info(
            f"Dataset capped from {len(dataset.examples)} to {max_examples_value} examples"
        )
        dataset = capped_dataset

    max_total_examples_value = algorithm_kwargs.pop("max_total_examples", None)
    if max_total_examples_value is None:
        max_total_examples_value = max_total_examples

    samples_include_pruned_value = algorithm_kwargs.pop("samples_include_pruned", None)
    if samples_include_pruned_value is None:
        samples_include_pruned_value = samples_include_pruned

    return dataset, max_total_examples_value, bool(samples_include_pruned_value)


# ---------------------------------------------------------------------------
# Phase 5: Parallel configuration
# ---------------------------------------------------------------------------


def resolve_effective_parallel_config(
    algorithm_kwargs: dict[str, Any],
    *,
    decorator_parallel_config: ParallelConfig | None,
    config_space_size: int,
    is_async_func: bool,
) -> tuple[int | None, int | None, int | None]:
    """Resolve parallel configuration settings.

    Args:
        algorithm_kwargs: Algorithm kwargs (may be mutated to pop parallel_config)
        decorator_parallel_config: Parallel config from decorator
        config_space_size: Estimated search space size
        is_async_func: Whether the optimized function is async

    Returns:
        Tuple of (effective_parallel_trials, effective_batch_size, effective_thread_workers)
    """
    runtime_parallel_config = coerce_parallel_config(
        algorithm_kwargs.pop("parallel_config", None)
    )

    global_parallel_config = get_global_parallel_config()
    merged_parallel_config, merged_sources = merge_parallel_configs(
        [
            (global_parallel_config, "global"),
            (decorator_parallel_config, "decorator"),
            (runtime_parallel_config, "runtime"),
        ]
    )
    if merged_sources:
        logger.debug("Parallel configuration merge sources: %s", merged_sources)

    default_thread_workers = (
        merged_parallel_config.thread_workers
        if merged_parallel_config.thread_workers is not None
        else int(_GLOBAL_CONFIG.get("parallel_workers", 1))
    )

    resolved_parallel = resolve_parallel_config(
        merged_parallel_config,
        default_thread_workers=default_thread_workers,
        config_space_size=config_space_size,
        detected_function_kind="async" if is_async_func else "sync",
        sources=merged_sources,
    )

    for warning in resolved_parallel.warnings:
        logger.warning(warning)

    logger.info(
        "Resolved parallel configuration: mode=%s, trial_concurrency=%s, example_concurrency=%s, thread_workers=%s",
        resolved_parallel.mode,
        resolved_parallel.trial_concurrency,
        resolved_parallel.example_concurrency,
        resolved_parallel.thread_workers,
    )

    return (
        resolved_parallel.trial_concurrency,
        resolved_parallel.example_concurrency,
        resolved_parallel.thread_workers,
    )


# ---------------------------------------------------------------------------
# Phase 7: Evaluator construction
# ---------------------------------------------------------------------------


def resolve_custom_evaluator(
    custom_evaluator: Callable[..., Any] | None,
    *,
    mock_mode_config: dict[str, Any] | None,
    decorator_custom_evaluator: Callable[..., Any] | None,
) -> Callable[..., Any] | None:
    """Resolve the effective custom evaluator based on mock mode settings.

    Args:
        custom_evaluator: Custom evaluator from optimize() call
        mock_mode_config: Mock mode configuration
        decorator_custom_evaluator: Custom evaluator from decorator

    Returns:
        The custom evaluator to use, or None if LocalEvaluator should be used.
    """
    mock_mode_env = is_mock_llm()
    mock_config = mock_mode_config or {}
    mock_enabled = mock_config.get("enabled", True)
    override_evaluator = mock_config.get("override_evaluator", True)

    provided_custom_evaluator = custom_evaluator or decorator_custom_evaluator
    has_custom = provided_custom_evaluator is not None

    if mock_mode_env and mock_enabled and override_evaluator and has_custom:
        logger.info(
            "Mock mode enabled: overriding custom evaluator with LocalEvaluator"
        )
        return None

    return provided_custom_evaluator if has_custom else None


def build_metric_functions(
    metric_functions: dict[str, Callable[..., Any]] | None,
    scoring_function: Callable[..., Any] | None,
    objectives: Sequence[str],
) -> dict[str, Callable[..., Any]]:
    """Build the effective metric functions dictionary.

    Args:
        metric_functions: Explicit metric functions from decorator
        scoring_function: Scoring function from decorator
        objectives: List of objective names

    Returns:
        Dict mapping metric names to callable functions.
    """
    effective_metric_functions: dict[str, Callable[..., Any]] = dict(
        metric_functions or {}
    )

    if scoring_function is None:
        return effective_metric_functions

    target_metric: str | None = None
    if "accuracy" in objectives:
        target_metric = "accuracy"
    elif "score" in objectives:
        target_metric = "score"

    if target_metric and target_metric not in effective_metric_functions:
        effective_metric_functions[target_metric] = scoring_function

    return effective_metric_functions


def resolve_effective_workers(
    effective_batch_size: int | None,
    effective_thread_workers: int | None,
) -> int:
    """Resolve the effective number of workers, applying thread limit.

    Args:
        effective_batch_size: Batch size for example concurrency
        effective_thread_workers: Thread worker limit

    Returns:
        Effective worker count (>= 1).
    """
    effective_workers = max(1, int(effective_batch_size or 1))
    if effective_thread_workers and effective_workers > effective_thread_workers:
        logger.debug(
            "Clamping example concurrency from %s to thread worker limit %s",
            effective_workers,
            effective_thread_workers,
        )
        effective_workers = effective_thread_workers
    return effective_workers


def create_effective_evaluator(
    timeout: float | None,
    custom_evaluator: Callable[..., Any] | None,
    effective_batch_size: int | None,
    effective_thread_workers: int | None,
    effective_privacy_enabled: bool,
    *,
    objectives: Sequence[str],
    js_runtime_config: Any,
    execution_mode: str,
    mock_mode_config: dict[str, Any] | None,
    metric_functions: dict[str, Callable[..., Any]] | None,
    scoring_function: Callable[..., Any] | None,
    decorator_custom_evaluator: Callable[..., Any] | None,
    hybrid_api_endpoint: str | None = None,
    hybrid_api_capability_id: str | None = None,
    hybrid_api_transport: Any | None = None,
    hybrid_api_transport_type: str = "auto",
    hybrid_api_batch_size: int = 1,
    hybrid_api_batch_parallelism: int = 1,
    hybrid_api_keep_alive: bool = True,
    hybrid_api_heartbeat_interval: float = 30.0,
    hybrid_api_timeout: float | None = None,
    hybrid_api_auth_header: str | None = None,
    hybrid_api_auto_discover_tvars: bool = False,
) -> tuple[BaseEvaluator, Any]:
    """Create the appropriate evaluator for the optimization run.

    Args:
        timeout: Evaluation timeout
        custom_evaluator: Custom evaluator from optimize() call
        effective_batch_size: Example concurrency batch size
        effective_thread_workers: Thread worker limit
        effective_privacy_enabled: Whether privacy mode is enabled
        objectives: Objective names
        js_runtime_config: JS runtime configuration (or None)
        execution_mode: Execution mode string
        mock_mode_config: Mock mode configuration
        metric_functions: Explicit metric functions
        scoring_function: Scoring function
        decorator_custom_evaluator: Custom evaluator from decorator

    Returns:
        Tuple of (evaluator, js_process_pool_or_None)
    """
    effective_evaluator = resolve_custom_evaluator(
        custom_evaluator,
        mock_mode_config=mock_mode_config,
        decorator_custom_evaluator=decorator_custom_evaluator,
    )

    if effective_evaluator:
        if not callable(effective_evaluator):
            raise ValueError("custom_evaluator must be callable") from None
        return (
            CustomEvaluatorWrapper(
                custom_evaluator=effective_evaluator,
                metrics=list(objectives),
                timeout=timeout or 60.0,
                capture_llm_metrics=True,
            ),
            None,
        )

    execution_mode_enum = resolve_execution_mode(execution_mode)
    if execution_mode_enum is ExecutionMode.HYBRID_API:
        from traigent.evaluators.hybrid_api import HybridAPIEvaluator

        if hybrid_api_transport is None and not hybrid_api_endpoint:
            raise ValueError(
                "hybrid_api execution mode requires hybrid_api_endpoint "
                "or a preconfigured hybrid_api_transport."
            )

        request_timeout = (
            hybrid_api_timeout if hybrid_api_timeout is not None else timeout
        )
        if request_timeout is None:
            request_timeout = 300.0

        batch_size_value = max(1, int(hybrid_api_batch_size))
        batch_parallelism_value = max(1, int(hybrid_api_batch_parallelism))

        evaluator = HybridAPIEvaluator(
            api_endpoint=hybrid_api_endpoint,
            transport=hybrid_api_transport,
            transport_type=cast(
                Literal["http", "mcp", "auto"], hybrid_api_transport_type
            ),
            capability_id=hybrid_api_capability_id,
            auto_discover_tvars=hybrid_api_auto_discover_tvars,
            batch_size=batch_size_value,
            batch_parallelism=batch_parallelism_value,
            keep_alive=hybrid_api_keep_alive,
            heartbeat_interval=hybrid_api_heartbeat_interval,
            timeout=request_timeout,
            auth_header=hybrid_api_auth_header,
            metrics=list(objectives),
        )
        return evaluator, None

    # Check if JS runtime is configured
    js_config = js_runtime_config
    if js_config is not None and getattr(js_config, "is_js_runtime", False):
        from traigent.evaluators.js_evaluator import JSEvaluator

        js_parallel_workers = getattr(js_config, "js_parallel_workers", 1)
        process_pool = None

        if js_parallel_workers > 1:
            from traigent.bridges.process_pool import JSProcessPool, JSProcessPoolConfig

            pool_config = JSProcessPoolConfig(
                max_workers=js_parallel_workers,
                module_path=js_config.js_module,
                function_name=js_config.js_function,
                trial_timeout=js_config.js_timeout,
            )
            process_pool = JSProcessPool(pool_config)
            logger.info(
                "Created JS process pool with %d workers for parallel execution",
                js_parallel_workers,
            )

        return (
            JSEvaluator(
                js_module=js_config.js_module,
                js_function=js_config.js_function,
                js_timeout=js_config.js_timeout,
                process_pool=process_pool,
            ),
            process_pool,
        )

    effective_metric_fns = build_metric_functions(
        metric_functions, scoring_function, objectives
    )
    effective_workers = resolve_effective_workers(
        effective_batch_size, effective_thread_workers
    )

    return (
        LocalEvaluator(
            metrics=list(objectives),
            timeout=timeout or 60.0,
            max_workers=effective_workers,
            detailed=True,
            execution_mode=execution_mode,
            privacy_enabled=effective_privacy_enabled,
            mock_mode_config=mock_mode_config,
            metric_functions=effective_metric_fns or None,
        ),
        None,
    )


# ---------------------------------------------------------------------------
# Workflow traces
# ---------------------------------------------------------------------------


def create_workflow_traces_tracker(
    _traigent_config: TraigentConfig,  # noqa: ARG001
) -> Any:
    """Create workflow traces tracker if backend is configured.

    Args:
        _traigent_config: Traigent configuration (reserved for future use)

    Returns:
        WorkflowTracesTracker instance if backend is configured, None otherwise
    """
    backend_url = os.environ.get("TRAIGENT_BACKEND_URL")
    api_key = os.environ.get("TRAIGENT_API_KEY")

    if not backend_url or not api_key:
        return None

    if os.environ.get("TRAIGENT_TRACES_ENABLED", "").lower() == "false":
        return None

    try:
        from traigent.integrations.observability.workflow_traces import (
            WorkflowTracesTracker,
        )

        tracker = WorkflowTracesTracker(
            backend_url=backend_url,
            auth_token=api_key,
        )
        logger.debug(f"Auto-initialized workflow traces tracker for {backend_url}")
        return tracker

    except ImportError:
        logger.debug("Workflow traces module not available, skipping trace collection")
        return None
    except Exception as exc:
        logger.debug(f"Failed to initialize workflow traces tracker: {exc}")
        return None


# ---------------------------------------------------------------------------
# Orchestrator kwargs
# ---------------------------------------------------------------------------


def collect_orchestrator_kwargs(
    algorithm_kwargs: dict[str, Any],
    samples_include_pruned_value: bool,
    *,
    default_config: dict[str, Any] | None,
    constraints: list[Any] | None,
    agents: list[Any] | None,
    agent_prefixes: dict[str, Any] | None,
    agent_measures: dict[str, Any] | None,
    global_measures: dict[str, Any] | None,
    promotion_gate: Any | None,
    invocations_per_example: int = 1,
) -> dict[str, Any]:
    """Collect optional kwargs for orchestrator from algorithm_kwargs and attrs.

    Args:
        algorithm_kwargs: Algorithm kwargs dict
        samples_include_pruned_value: Whether pruned samples count toward budget
        default_config: Default configuration
        constraints: Constraint functions
        agents: Agent list
        agent_prefixes: Agent prefix mapping
        agent_measures: Agent measures mapping
        global_measures: Global measures mapping
        promotion_gate: Promotion gate configuration
        invocations_per_example: Number of invocations per example (default: 1)

    Returns:
        Dict of orchestrator keyword arguments.
    """
    kwargs: dict[str, Any] = {
        "cache_policy": algorithm_kwargs.get("cache_policy", "allow_repeats"),
        "samples_include_pruned": samples_include_pruned_value,
        "invocations_per_example": invocations_per_example,
    }

    optional_keys = [
        "budget_limit",
        "budget_metric",
        "budget_include_pruned",
        "plateau_window",
        "plateau_epsilon",
        "cost_limit",
        "cost_approved",
        "tie_breakers",
        "tvl_parameter_agents",
    ]
    for key in optional_keys:
        if key in algorithm_kwargs:
            kwargs[key] = algorithm_kwargs[key]

    optional_attrs = [
        ("default_config", default_config, lambda v: v.copy()),
        ("constraints", constraints, None),
        ("agents", agents, None),
        ("agent_prefixes", agent_prefixes, None),
        ("agent_measures", agent_measures, None),
        ("global_measures", global_measures, None),
        ("promotion_gate", promotion_gate, None),
    ]
    for attr_name, value, transform in optional_attrs:
        if value is not None:
            kwargs[attr_name] = transform(value) if transform else value

    return kwargs
