"""Standalone API functions for Traigent SDK."""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from traigent.api.types import OptimizationResult, StrategyConfig
from traigent.config.api_keys import _API_KEY_MANAGER
from traigent.config.context import get_applied_config
from traigent.config.context import get_config as _get_context_config
from traigent.config.context import get_trial_context
from traigent.config.feature_flags import flag_registry
from traigent.config.parallel import (
    ParallelConfig,
    coerce_parallel_config,
    merge_parallel_configs,
)
from traigent.config.types import TraigentConfig
from traigent.optimizers import list_optimizers
from traigent.utils.exceptions import ConfigAccessWarning, OptimizationStateError
from traigent.utils.insights import (
    get_optimization_insights as _get_optimization_insights,
)
from traigent.utils.logging import get_logger, setup_logging
from traigent.utils.validation import Validators, validate_or_raise

logger = get_logger(__name__)

# Global configuration
_GLOBAL_CONFIG: dict[str, Any] = {
    "default_storage_backend": "edge_analytics",
    "parallel_workers": 1,
    "cache_policy": "memory",
    "logging_level": "INFO",
    "api_keys": {},  # Kept for backward compatibility
    "parallel_config": ParallelConfig(),
}


if TYPE_CHECKING:
    from traigent.core.objectives import ObjectiveSchema


def configure(
    default_storage_backend: str | None = None,
    parallel_workers: int | None = None,
    cache_policy: str | None = None,
    logging_level: str | None = None,
    api_keys: dict[str, str] | None = None,
    feature_flags: dict[str, Any] | None = None,
    parallel_config: ParallelConfig | dict[str, Any] | None = None,
    objectives: ObjectiveSchema | Sequence[str] | None = None,
) -> bool:
    """Configure global Traigent SDK settings.

    Args:
        default_storage_backend: Default storage ("edge_analytics", "s3", "gcs")
        parallel_workers: Default number of parallel workers
        cache_policy: Cache policy ("memory", "disk", "distributed")
        logging_level: Logging verbosity ("DEBUG", "INFO", "WARNING", "ERROR")
        api_keys: API keys for external services
        objectives: Default objectives (list of names or ObjectiveSchema)

    Returns:
        True if configuration successful

    Example:
        >>> traigent.configure(
        ...     logging_level="DEBUG",
        ...     parallel_workers=4,
        ...     api_keys={"openai": "sk-..."}
        ... )
        True
    """
    if default_storage_backend is not None:
        _GLOBAL_CONFIG["default_storage_backend"] = default_storage_backend

    if parallel_workers is not None:
        result = Validators.validate_positive_int(parallel_workers, "parallel_workers")
        validate_or_raise(result)
        _GLOBAL_CONFIG["parallel_workers"] = parallel_workers

    if cache_policy is not None:
        valid_policies = ["memory", "disk", "distributed"]
        result = Validators.validate_choices(
            cache_policy, "cache_policy", valid_policies
        )
        validate_or_raise(result)
        _GLOBAL_CONFIG["cache_policy"] = cache_policy

    if logging_level is not None:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        result = Validators.validate_choices(
            logging_level, "logging_level", valid_levels
        )
        validate_or_raise(result)
        _GLOBAL_CONFIG["logging_level"] = logging_level
        setup_logging(level=logging_level)

    if api_keys is not None:
        result = Validators.validate_type(api_keys, dict, "api_keys")
        validate_or_raise(result)
        _GLOBAL_CONFIG["api_keys"].update(api_keys)  # Keep for backward compatibility
        # Also set in the secure manager
        for provider, key in api_keys.items():
            _API_KEY_MANAGER.set_api_key(provider, key, source="code")

    if feature_flags is not None:
        result = Validators.validate_type(feature_flags, dict, "feature_flags")
        validate_or_raise(result)
        flag_registry.apply_config(feature_flags)

    _apply_parallel_config(parallel_config, parallel_workers=parallel_workers)
    _apply_objectives(objectives)

    logger.info("Updated global configuration")
    return True


def _apply_parallel_config(
    parallel_config: ParallelConfig | dict[str, Any] | None,
    *,
    parallel_workers: int | None,
) -> None:
    if parallel_config is None:
        return

    coerced = coerce_parallel_config(parallel_config)
    if coerced is None:
        logger.debug(
            "parallel_config explicitly set to None; leaving existing value unchanged"
        )
        return

    existing_config = _GLOBAL_CONFIG.get("parallel_config", ParallelConfig())
    merged_config, _ = merge_parallel_configs(
        [
            (existing_config, "global-default"),
            (coerced, "configure"),
        ]
    )
    _GLOBAL_CONFIG["parallel_config"] = merged_config
    if coerced.thread_workers is not None and parallel_workers is None:
        # Keep legacy parallel_workers in sync for components still reading it
        result = Validators.validate_positive_int(
            coerced.thread_workers, "parallel_config.thread_workers"
        )
        validate_or_raise(result)
        _GLOBAL_CONFIG["parallel_workers"] = coerced.thread_workers


def _apply_objectives(objectives: ObjectiveSchema | Sequence[str] | None) -> None:
    if objectives is None:
        return

    from traigent.core.objectives import normalize_objectives, schema_to_objective_names

    schema = normalize_objectives(objectives)
    _GLOBAL_CONFIG["objective_schema"] = schema
    _GLOBAL_CONFIG["objectives"] = schema_to_objective_names(schema)


def initialize(  # noqa: C901
    api_key: str | None = None,
    api_url: str | None = None,
    config: TraigentConfig | None = None,
    **kwargs: Any,
) -> bool:
    """Initialize Traigent SDK for local or cloud operation.

    This function configures the SDK for integration with the Traigent backend,
    enabling seamless optimization with experiment tracking and storage.

    Args:
        api_key: API key for Traigent backend authentication (defaults to env var)
        api_url: Traigent backend URL (defaults to centralized config)
        config: TraigentConfig object with execution mode and settings
        **kwargs: Additional configuration parameters

    Returns:
        True if initialization successful

    Example::

        # Edge Analytics mode initialization (backend URL from env or config)
        # Set TRAIGENT_API_KEY environment variable for security
        config = traigent.TraigentConfig.edge_analytics_mode()
        traigent.initialize(config=config)

        # Cloud mode with explicit URL
        traigent.initialize(
            api_key=os.getenv("TRAIGENT_API_KEY"),  # Use env var
            api_url="https://portal.traigent.ai"
        )

        # Using environment variables (recommended)
        # export TRAIGENT_BACKEND_URL="https://portal.traigent.ai"
        # export TRAIGENT_API_KEY="your-key-here"  # pragma: allowlist secret
        traigent.initialize()
    """

    from traigent.config.backend_config import BackendConfig

    _configure_api_keys(api_key, BackendConfig)
    _configure_backend_url(api_url, BackendConfig)

    if config:
        _apply_config_settings(config)

    _apply_additional_overrides(kwargs)
    _configure_logging_settings(config)

    logger.info("Traigent SDK initialized successfully")
    return True


def _configure_api_keys(api_key: str | None, backend_config: Any) -> None:
    """Configure API keys from explicit value or environment."""

    if api_key:
        _API_KEY_MANAGER.set_api_key("traigent", api_key, source="initialization")
        logger.info("Traigent API key configured")
        return

    env_key = backend_config.get_api_key()
    if env_key:
        _API_KEY_MANAGER.set_api_key("traigent", env_key, source="environment")
        logger.info("Traigent API key configured from environment")


def _configure_backend_url(api_url: str | None, backend_config: Any) -> None:
    """Resolve and store the backend API URL."""

    if api_url:
        origin, path = backend_config.split_api_url(api_url)
        if origin:
            resolved_api_url = (
                f"{origin}{path or backend_config.get_default_api_path()}"
            )
        else:
            resolved_api_url = api_url.rstrip("/")
        _GLOBAL_CONFIG["traigent_api_url"] = resolved_api_url
        logger.info(f"Traigent backend API set to: {resolved_api_url}")
        return

    backend_api_url = backend_config.get_backend_api_url()
    _GLOBAL_CONFIG["traigent_api_url"] = backend_api_url
    logger.info(f"Traigent backend API configured: {backend_api_url}")


def _apply_config_settings(config: TraigentConfig) -> None:
    """Merge TraigentConfig values into the global configuration."""

    if config.execution_mode:
        _GLOBAL_CONFIG["execution_mode"] = config.execution_mode

    if config.local_storage_path:
        _GLOBAL_CONFIG["local_storage_path"] = config.local_storage_path

    if config.minimal_logging is not None:
        _GLOBAL_CONFIG["minimal_logging"] = config.minimal_logging

    if config.auto_sync is not None:
        _GLOBAL_CONFIG["auto_sync"] = config.auto_sync

    _GLOBAL_CONFIG["default_storage_backend"] = (
        "edge_analytics" if config.is_edge_analytics_mode() else "cloud"
    )

    logger.info(f"Traigent configured for {config.execution_mode} mode")


def _apply_additional_overrides(overrides: dict[str, Any]) -> None:
    """Merge arbitrary keyword overrides into the global configuration."""

    for key, value in overrides.items():
        _GLOBAL_CONFIG[key] = value


def _configure_logging_settings(config: TraigentConfig | None) -> None:
    """Configure logging level based on provided configuration."""

    if config and config.minimal_logging:
        setup_logging(level="WARNING")
        return

    setup_logging(level=_GLOBAL_CONFIG.get("logging_level", "INFO"))


def get_global_config() -> dict[str, Any]:
    """Get current global configuration.

    Returns:
        Dictionary with current global settings
    """
    return _GLOBAL_CONFIG.copy()


def get_api_key(provider: str) -> str | None:
    """Get API key for a provider, checking environment variables first.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic')

    Returns:
        The API key if found, None otherwise
    """
    return cast(str | None, _API_KEY_MANAGER.get_api_key(provider))


def _coerce_config_dict(
    config: object,
    *,
    source: str,
    current_state: str,
    expected_states: list[str],
) -> dict[str, Any]:
    """Normalize config objects to a plain dict and raise on invalid types."""
    if isinstance(config, TraigentConfig):
        return cast(dict[str, Any], config.to_dict())
    if isinstance(config, dict):
        return cast(dict[str, Any], dict(config))

    raise OptimizationStateError(
        f"{source} config has invalid type: {type(config).__name__}. "
        "Expected TraigentConfig or dict. This may indicate corrupted context state.",
        current_state=current_state,
        expected_states=expected_states,
    )


def get_config() -> dict[str, Any]:
    """Get the active configuration in any lifecycle context.

    This unified accessor works both during optimization trials and after
    applying the best configuration to your function.

    Important - When to use get_config() vs get_trial_config():
        - **get_config()** (recommended): Works in all contexts - during optimization
          trials and after calling ``apply_best_config()``. Use this for most cases.
        - **get_trial_config()**: Strict validation - raises ``OptimizationStateError``
          if called outside an active trial. Use when you need explicit trial context.

    .. warning::
        Traigent does NOT automatically inject config into function parameters.
        Function parameters like ``model: str = "gpt-4"`` will NOT be overridden
        by Traigent during optimization. Use ``get_config()`` to access trial values.

    Returns:
        Dictionary with the currently active configuration.

    Raises:
        OptimizationStateError: If no configuration is available (e.g., called
            outside an optimized function without apply_best_config()).

    Example::

        @traigent.optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
        def my_func(query: str):
            cfg = traigent.get_config()  # Works during trials and after apply_best_config
            return call_llm(model=cfg["model"])
    """
    trial_ctx = get_trial_context()
    if trial_ctx is not None:
        return _coerce_config_dict(
            _get_context_config(),
            source="Trial",
            current_state="OPTIMIZING",
            expected_states=["OPTIMIZING"],
        )

    # Check applied config first, then fall back to context config
    applied_config = get_applied_config()
    if applied_config is not None:
        return _coerce_config_dict(
            applied_config,
            source="Applied",
            current_state="CONFIG_APPLIED",
            expected_states=["CONFIG_APPLIED", "UNOPTIMIZED"],
        )

    # Fall back to context config (set by ConfigurationContext in wrappers)
    context_config = cast(object, _get_context_config())
    if isinstance(context_config, dict) and context_config:
        return _coerce_config_dict(
            context_config,
            source="Context",
            current_state="CONFIG_APPLIED",
            expected_states=["CONFIG_APPLIED", "UNOPTIMIZED"],
        )
    if isinstance(context_config, TraigentConfig):
        config_dict = cast(dict[str, Any], context_config.to_dict())
        # Check if config has any meaningful values (not just defaults)
        if any(v is not None for k, v in config_dict.items() if k != "execution_mode"):
            return config_dict

    raise OptimizationStateError(
        "No config available. Run optimize() and apply_best_config(), "
        "or call this inside an optimized function during an active trial.",
        current_state="NO_ACTIVE_CONFIG",
        expected_states=["UNOPTIMIZED", "CONFIG_APPLIED", "OPTIMIZING"],
    )


def get_trial_config() -> dict[str, Any]:
    """Get the configuration for the current optimization trial.

    This function should ONLY be called from within an optimized function
    during an active optimization run. It returns the trial-specific
    configuration values being tested.

    Raises:
        OptimizationStateError: If called outside an active optimization trial.
            This ensures you're accessing the right configuration at the right time.

    Returns:
        Dictionary with the current trial's configuration parameters.

    Example::

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.5, 0.8]}
        )
        def my_function(query: str) -> str:
            config = traigent.get_trial_config()  # Gets trial-specific config
            return call_llm(model=config["model"], temperature=config["temperature"])

        # Run optimization - get_trial_config() works inside the function
        result = traigent.optimize(my_function, dataset=my_data)
        # Access best config via result
        print(result.best_config)
    """
    # Check if we're in an active trial context
    trial_ctx = get_trial_context()
    if trial_ctx is None:
        raise OptimizationStateError(
            "get_trial_config() can only be called during an active optimization trial. "
            "For post-optimization access, call traigent.get_config() inside your "
            "function or use my_function.current_config / OptimizationResult.best_config.",
            current_state="NO_ACTIVE_TRIAL",
            expected_states=["OPTIMIZING"],
        )

    # Validate trial context has required fields
    if not isinstance(trial_ctx, dict):
        raise OptimizationStateError(
            f"Trial context is corrupted - expected dict but got {type(trial_ctx).__name__}.",
            current_state="INVALID_TRIAL_CONTEXT",
            expected_states=["OPTIMIZING"],
        )

    if "trial_id" not in trial_ctx:
        raise OptimizationStateError(
            "Trial context is missing 'trial_id'. This may indicate the context "
            "was not properly initialized via TrialContext.",
            current_state="INCOMPLETE_TRIAL_CONTEXT",
            expected_states=["OPTIMIZING"],
        )

    # Get the actual config from context
    config = cast(object, _get_context_config())
    return _coerce_config_dict(
        config,
        source="Trial",
        current_state="INVALID_CONFIG_TYPE",
        expected_states=["OPTIMIZING"],
    )


def get_current_config() -> dict[str, Any]:
    """Get the current optimization configuration.

    .. deprecated::
        Use :func:`get_config` for lifecycle-safe access. This function is
        deprecated because "current" is ambiguous - it could mean the trial
        config during optimization or the applied config after optimization.

        - During/after optimization: Use ``traigent.get_config()``
        - After optimization: Use ``result.best_config`` or ``func.current_config``

    Returns:
        Dictionary with current configuration parameters.

    Note:
        This function will emit a deprecation warning. Unlike get_trial_config(),
        it returns an empty dict instead of raising an error when called outside
        an optimization trial (for backward compatibility).
    """
    warnings.warn(
        "get_current_config() is deprecated. Use traigent.get_config() for active "
        "configs (during and after optimization), or access func.current_config / "
        "result.best_config directly.",
        ConfigAccessWarning,
        stacklevel=2,
    )

    config = _get_context_config()

    # Convert TraigentConfig to dict if needed
    if isinstance(config, TraigentConfig):
        return cast(dict[str, Any], config.to_dict())
    if isinstance(config, dict):
        return cast(dict[str, Any], dict(config))

    # Fallback for unexpected config types
    # This branch handles edge cases where config is neither TraigentConfig nor dict
    return {}


def override_config(
    objectives: list[str] | None = None,
    configuration_space: dict[str, Any] | None = None,
    constraints: list[Callable[..., Any]] | None = None,
    max_trials: int | None = None,
    timeout: int | None = None,
    max_total_examples: int | None = None,
    samples_include_pruned: bool | None = None,
) -> dict[str, Any]:
    """Create configuration override for optimization runs.

    Args:
        objectives: New objectives to optimize for
        configuration_space: Modified parameter space
        constraints: Additional or replacement constraints
        max_trials: Override number of optimization trials
        timeout: Override optimization timeout
        max_total_examples: Global sample budget across all trials
        samples_include_pruned: Whether pruned trials count toward the sample budget

    Returns:
        Configuration override dict for use with .optimize()

    Example::

        # Override to focus on cost efficiency
        cost_config = traigent.override_config(
            objectives=["cost", "accuracy"],
            configuration_space={"model": ["gpt-4o-mini"]},
            max_trials=20
        )
        results = my_agent.optimize(config_override=cost_config)
    """
    override: dict[str, Any] = {}

    if objectives is not None:
        override["objectives"] = objectives

    if configuration_space is not None:
        override["configuration_space"] = configuration_space

    if constraints is not None:
        override["constraints"] = constraints

    if max_trials is not None:
        if max_trials < 0:
            raise ValueError("max_trials must be non-negative")
        override["max_trials"] = max_trials

    if timeout is not None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        override["timeout"] = timeout

    if max_total_examples is not None:
        if max_total_examples <= 0:
            raise ValueError("max_total_examples must be > 0")
        override["max_total_examples"] = max_total_examples

    if samples_include_pruned is not None:
        override["samples_include_pruned"] = bool(samples_include_pruned)

    return override


def _validate_budget_configuration_inputs(
    budget_usd: float,
    min_instances: int,
    reserve_ratio: float,
    max_parallel_workers: int | None,
    model_pricing: Mapping[str, float],
) -> None:
    """Validate inputs for budget-aware optimization configuration."""
    if budget_usd <= 0:
        raise ValueError("budget_usd must be > 0")
    if min_instances < 1:
        raise ValueError("min_instances must be >= 1")
    if not 0 <= reserve_ratio < 1:
        raise ValueError("reserve_ratio must be in [0, 1)")
    if max_parallel_workers is not None and max_parallel_workers < 1:
        raise ValueError("max_parallel_workers must be >= 1 when provided")
    if not model_pricing:
        raise ValueError("model_pricing must not be empty")


def _normalize_model_pricing(model_pricing: Mapping[str, float]) -> dict[str, float]:
    """Normalize model pricing inputs to validated positive floats."""
    normalized_pricing: dict[str, float] = {}
    for model, raw_cost in model_pricing.items():
        if not model:
            raise ValueError("model_pricing keys must be non-empty model names")
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model_pricing['{model}'] must be numeric, got {raw_cost!r}"
            ) from exc
        if cost <= 0:
            raise ValueError(f"model_pricing['{model}'] must be > 0")
        normalized_pricing[model] = cost
    return normalized_pricing


def configure_for_budget(
    *,
    budget_usd: float,
    model_pricing: Mapping[str, float],
    min_instances: int = 1,
    reserve_ratio: float = 0.10,
    max_parallel_workers: int | None = None,
    return_diagnostics: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Build budget-aware optimization overrides from model pricing.

    This helper pre-filters candidate models to those that can run at least
    ``min_instances`` evaluations within budget, then derives trial/parallel
    limits to keep optimization spend bounded.

    Args:
        budget_usd: Total spend budget in USD.
        model_pricing: Mapping of model name to estimated cost per evaluation.
        min_instances: Minimum evaluations each selected model should support.
        reserve_ratio: Fraction of budget to reserve as safety headroom.
        max_parallel_workers: Optional upper bound for parallel workers.
        return_diagnostics: When True, also return a diagnostics mapping as
            ``(overrides, diagnostics)``.

    Returns:
        Recommended ``@optimize`` keyword overrides. The returned mapping is
        intentionally safe to pass directly as ``@optimize(**overrides)``.
        When ``return_diagnostics=True``, returns ``(overrides, diagnostics)``.
    """
    _validate_budget_configuration_inputs(
        budget_usd,
        min_instances,
        reserve_ratio,
        max_parallel_workers,
        model_pricing,
    )
    normalized_pricing = _normalize_model_pricing(model_pricing)

    effective_budget = budget_usd * (1.0 - reserve_ratio)
    affordable_items = sorted(
        (
            (model, cost)
            for model, cost in normalized_pricing.items()
            if cost * min_instances <= effective_budget
        ),
        key=lambda item: (item[1], item[0]),
    )

    if not affordable_items:
        cheapest_model, cheapest_cost = min(
            normalized_pricing.items(), key=lambda item: item[1]
        )
        raise ValueError(
            "No models can satisfy the requested minimum coverage within budget. "
            f"Cheapest model '{cheapest_model}' needs ${cheapest_cost * min_instances:.4f} "
            f"for {min_instances} evaluations, but effective budget is ${effective_budget:.4f}."
        )

    affordable_models = [model for model, _cost in affordable_items]
    cheapest_affordable_cost = affordable_items[0][1]
    max_instances = max(
        min_instances, int(effective_budget // cheapest_affordable_cost)
    )

    default_workers = int(_GLOBAL_CONFIG.get("parallel_workers", 1))
    worker_cap = (
        max_parallel_workers if max_parallel_workers is not None else default_workers
    )
    parallel_workers = max(1, min(max_instances, worker_cap))
    parallel_config = ParallelConfig.from_legacy(
        parallel_trials=parallel_workers,
        parallel_workers=parallel_workers,
    )

    overrides: dict[str, Any] = {
        "configuration_space": {"model": affordable_models},
        "max_trials": max_instances,
        "parallel_config": parallel_config,
        "cost_limit": float(budget_usd),
    }
    diagnostics = {
        "max_instances": max_instances,
        "parallel_workers": parallel_workers,
        "budget_usd": float(budget_usd),
        "effective_budget_usd": effective_budget,
        "selected_model_pricing": dict(affordable_items),
    }
    if return_diagnostics:
        return overrides, diagnostics
    return overrides


def set_strategy(
    algorithm: str = "tpe",
    algorithm_config: dict[str, Any] | None = None,
    parallel_workers: int | None = None,
    resource_limits: dict[str, Any] | None = None,
) -> StrategyConfig:
    """Configure optimization strategy and execution parameters.

    Args:
        algorithm: Optimization algorithm ("tpe", "random", "grid", "bayesian").
            Default is "tpe" (Tree-structured Parzen Estimator) which is always
            available with Optuna. "bayesian" (Gaussian Process) requires
            the traigent-advanced-algorithms plugin.
        algorithm_config: Algorithm-specific parameters
        parallel_workers: Number of parallel evaluation workers
        resource_limits: Memory, time, and compute constraints

    Returns:
        StrategyConfig object for use with optimization

    Example::

        strategy = traigent.set_strategy(
            algorithm="tpe",
            algorithm_config={
                "n_startup_trials": 10,
                "multivariate": True
            },
            parallel_workers=4
        )
        results = my_agent.optimize(strategy=strategy)
    """
    # Validate algorithm
    available_algorithms = get_available_strategies()
    if algorithm not in available_algorithms:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(available_algorithms.keys())}"
        )

    # Use global default if not specified
    workers = (
        parallel_workers
        if parallel_workers is not None
        else int(_GLOBAL_CONFIG["parallel_workers"])
    )

    strategy = StrategyConfig(
        algorithm=algorithm,
        algorithm_config=algorithm_config or {},
        parallel_workers=workers,
        resource_limits=resource_limits or {},
    )

    # Validate the strategy
    strategy.validate()

    logger.debug(f"Created strategy config: {algorithm}")

    return strategy


def get_available_strategies() -> dict[str, dict[str, Any]]:
    """Get information about available optimization strategies.

    Returns:
        Dict mapping strategy names to their capabilities and parameters

    Example:
        >>> strategies = traigent.get_available_strategies()
        >>> strategies["bayesian"]["description"]
        'Gaussian Process-based optimization with acquisition functions'
    """
    algorithms = list_optimizers()

    strategies = {}

    for algorithm in algorithms:
        if algorithm == "grid":
            strategies[algorithm] = {
                "name": "Grid Search",
                "description": "Exhaustive search over parameter combinations",
                "supports_continuous": False,
                "supports_categorical": True,
                "deterministic": True,
                "parameters": {
                    "parameter_order": (
                        "Map parameter names to numeric priorities. Lower values "
                        "vary slowest; higher values vary fastest."
                    ),
                    "order": "Alias for parameter_order.",
                },
                "best_for": "Small parameter spaces, exhaustive evaluation",
            }

        elif algorithm == "random":
            strategies[algorithm] = {
                "name": "Random Search",
                "description": "Random sampling from parameter space",
                "supports_continuous": True,
                "supports_categorical": True,
                "deterministic": False,
                "parameters": {
                    "max_trials": "Maximum number of trials (default: 100)",
                    "random_seed": "Random seed for reproducibility",
                },
                "best_for": "Quick exploration, continuous parameters",
            }

        elif algorithm == "bayesian":
            strategies[algorithm] = {
                "name": "Bayesian Optimization",
                "description": "Gaussian Process-based optimization with acquisition functions",
                "supports_continuous": True,
                "supports_categorical": True,
                "deterministic": False,
                "parameters": {
                    "acquisition_function": "expected_improvement or upper_confidence_bound",
                    "initial_random_samples": "Number of random trials before GP (default: 5)",
                    "xi": "Exploration parameter for EI (default: 0.01)",
                    "kappa": "Exploration parameter for UCB (default: 2.576)",
                    "random_seed": "Random seed for reproducibility",
                },
                "best_for": "Expensive evaluations, continuous optimization, sample efficiency",
            }

        # Add more algorithms as they are implemented

        else:
            strategies[algorithm] = {
                "name": algorithm.title(),
                "description": "Custom optimization algorithm",
                "supports_continuous": True,
                "supports_categorical": True,
                "deterministic": False,
                "parameters": {},
                "best_for": "Custom use cases",
            }

    return strategies


def _check_integration(module_path: str) -> bool:
    """Check if an integration module is available."""
    try:
        import importlib

        importlib.import_module(module_path)
        return True
    except ImportError:
        return False


def get_version_info() -> dict[str, Any]:
    """Get Traigent SDK version and capability information.

    Returns:
        Dict with version, supported features, and system info

    Example:
        >>> info = traigent.get_version_info()
        >>> 'version' in info
        True
    """
    import platform
    import sys

    from traigent import __version__

    algorithms = list_optimizers()
    if not algorithms:
        try:
            from traigent.optimizers.registry import _register_builtin_optimizers

            _register_builtin_optimizers()
            algorithms = list_optimizers()
        except Exception:
            algorithms = []

    # Query plugin registry for available features
    from traigent.plugins import (
        FEATURE_ADVANCED_ALGORITHMS,
        FEATURE_ANALYTICS,
        FEATURE_CLOUD,
        FEATURE_MULTI_OBJECTIVE,
        FEATURE_PARALLEL,
        FEATURE_SEAMLESS,
        FEATURE_TRACING,
        get_plugin_registry,
    )

    registry = get_plugin_registry()

    return {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "algorithms": algorithms,
        "features": {
            # Base features (always available)
            "grid_search": True,
            "random_search": True,
            "tpe_optimization": True,  # TPE is default, always available with Optuna
            "constraint_handling": True,
            "async_evaluation": True,
            "result_persistence": True,
            # Plugin-provided features (query registry)
            "bayesian_optimization": (
                registry.has_feature(FEATURE_ADVANCED_ALGORITHMS)
                or "bayesian" in algorithms  # Fallback: check if registered
            ),
            "multi_objective": registry.has_feature(FEATURE_MULTI_OBJECTIVE) or True,
            "parallel_evaluation": registry.has_feature(FEATURE_PARALLEL) or True,
            "seamless_injection": registry.has_feature(FEATURE_SEAMLESS) or True,
            "cloud_execution": registry.has_feature(FEATURE_CLOUD),
            "tracing": registry.has_feature(FEATURE_TRACING),
            "analytics": registry.has_feature(FEATURE_ANALYTICS),
            "visualization": True,  # Basic visualization in base
        },
        "integrations": {
            "langchain": _check_integration("traigent.integrations.llms.langchain"),
            "openai": _check_integration("traigent.integrations.llms.openai"),
            "mlflow": _check_integration("traigent.integrations.observability.mlflow"),
            "wandb": _check_integration("traigent.integrations.observability.wandb"),
        },
        "plugins": registry.list_plugins(),  # List installed plugins
        "global_config": _GLOBAL_CONFIG.copy(),
    }


def get_optimization_insights(results: OptimizationResult) -> dict[str, Any]:
    """Generate comprehensive insights from optimization results.

    Analyzes optimization results to provide business intelligence about parameter
    performance, configuration trade-offs, and optimization effectiveness.

    Args:
        results: OptimizationResult from completed optimization

    Returns:
        Dictionary containing structured insights with:
        - top_configurations: Ranked list of best configurations with trade-off analysis
        - performance_summary: Overall optimization statistics and improvements
        - parameter_insights: Analysis of parameter importance and impact
        - recommendations: Actionable recommendations based on analysis

    Example::

        results = my_function.optimize()
        insights = traigent.get_optimization_insights(results)
        print("Top 3 configurations discovered:")
        for i, config in enumerate(insights['top_configurations'][:3]):
            print(f"{i+1}. {config['config']} -> {config['score']:.2%} accuracy")
    """
    return cast(dict[str, Any], _get_optimization_insights(results))


def get_global_parallel_config() -> ParallelConfig:
    """Return the globally configured parallel settings."""

    raw_config = _GLOBAL_CONFIG.get("parallel_config")
    if isinstance(raw_config, ParallelConfig):
        return raw_config
    coerced = coerce_parallel_config(raw_config)
    if coerced is None:
        coerced = ParallelConfig()
    _GLOBAL_CONFIG["parallel_config"] = coerced
    return coerced


def with_usage(
    text: str,
    total_cost: float,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    response_time_ms: float | None = None,
) -> str | dict[str, Any]:
    """Wrap a response with usage metadata if in optimization mode.

    Returns text directly in production, or a dict with metadata
    during optimization. The metadata is extracted and injected into
    metrics after cost calculation.

    Args:
        text: The actual response content (must be a string)
        total_cost: Pre-computed cost in USD (REQUIRED - not recalculated from tokens)
        input_tokens: Number of input tokens consumed (informational, for UI display).
            If only one token count is provided, the other defaults to 0.
        output_tokens: Number of output tokens generated (informational, for UI display).
        response_time_ms: Response time in milliseconds (optional, for latency tracking).
            If only one token count is provided, the other defaults to 0.

    Returns:
        In production: text unchanged
        During optimization: dict with structure:
            {
                "text": str,
                "__traigent_meta__": {
                    "total_cost": float,
                    "usage": {  # Optional
                        "input_tokens": int,
                        "output_tokens": int
                    }
                }
            }

    Raises:
        TypeError: If text is not a string or total_cost is not numeric

    Validation:
        The returned __traigent_meta__ structure is validated at runtime using
        type guards from traigent.core.meta_types.TraigentMetadata:
        - text must be string (raises TypeError)
        - total_cost must be numeric (raises TypeError)
        - tokens must be integers if provided
        - negative values are clamped to 0 with warning

    Type Safety:
        See traigent.core.meta_types.TraigentMetadata for the canonical TypedDict
        definition. The structure is validated using is_traigent_metadata() type guard.

    Note:
        - `total_cost` is REQUIRED because cost is not recalculated from tokens
        - Token counts are informational only (for UI display)
        - Cost is injected AFTER SDK's cost calculation, overriding any calculated value
        - Custom metric functions receive the full dict output (including __traigent_meta__)
        - `__traigent_meta__` is a reserved key. If your function output already contains
          this key for other purposes, it will be treated as usage metadata.

    Example:
        @traigent.optimize(...)
        def my_workflow(question: str):
            # Accumulate usage from internal LLM calls
            total_in = total_out = total_cost = 0
            for step in workflow:
                response = llm.call(...)
                total_in += response.usage.prompt_tokens
                total_out += response.usage.completion_tokens
                total_cost += calculate_cost(response)  # User must calculate

            return traigent.with_usage(
                text=answer,
                total_cost=total_cost,
                input_tokens=total_in,
                output_tokens=total_out,
            )

    Multi-Agent Workflows:
        If the optional cloud workflow DTOs are installed, you can use
        AgentCostBreakdown and WorkflowCostSummary for per-agent cost tracking:

        >>> from traigent.cloud.agent_dtos import AgentCostBreakdown, WorkflowCostSummary
        >>> agent1 = AgentCostBreakdown(
        ...     agent_id="researcher",
        ...     agent_name="Research Agent",
        ...     input_tokens=100, output_tokens=50, total_tokens=150,
        ...     input_cost=0.001, output_cost=0.002, total_cost=0.003,
        ...     model_used="gpt-4o-mini"
        ... )
        >>> workflow = WorkflowCostSummary(
        ...     workflow_id="workflow-001",
        ...     workflow_name="Research",
        ...     agent_breakdowns=[agent1]
        ... )
        >>> return traigent.with_usage(
        ...     text=result,
        ...     total_cost=workflow.total_cost,
        ...     input_tokens=workflow.total_input_tokens,
        ...     output_tokens=workflow.total_output_tokens
        ... )

        In OSS-only builds, compute the aggregate totals directly and pass the
        numeric values to ``with_usage()`` without importing cloud DTOs.
    """
    # Enforce string type
    if not isinstance(text, str):
        raise TypeError(
            f"with_usage() requires text to be a string, got {type(text).__name__}. "
            "Convert your response to a string before calling with_usage()."
        )

    # Only wrap in dict during optimization
    if get_trial_context() is None:
        return text

    result: dict[str, Any] = {"text": text}

    # Build metadata - always include total_cost (required)
    meta: dict[str, Any] = {"total_cost": float(total_cost)}

    # Only include usage metadata if any field is explicitly provided (not None)
    # This avoids overwriting existing extracted values with zeros
    if (
        input_tokens is not None
        or output_tokens is not None
        or response_time_ms is not None
    ):
        usage: dict[str, int | float] = {}
        if input_tokens is not None:
            usage["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            usage["output_tokens"] = int(output_tokens)
        if response_time_ms is not None:
            usage["response_time_ms"] = float(response_time_ms)
        meta["usage"] = usage

    result["__traigent_meta__"] = meta

    return result
