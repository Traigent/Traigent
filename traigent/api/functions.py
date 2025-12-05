"""Standalone API functions for TraiGent SDK."""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from traigent.api.types import OptimizationResult, StrategyConfig
from traigent.config.api_keys import _API_KEY_MANAGER
from traigent.config.context import get_config
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
    """Configure global TraiGent SDK settings.

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

    if parallel_config is not None:
        coerced = coerce_parallel_config(parallel_config)
        if coerced is None:
            logger.debug(
                "parallel_config explicitly set to None; leaving existing value unchanged"
            )
        else:
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

    if objectives is not None:
        from traigent.core.objectives import (
            normalize_objectives,
            schema_to_objective_names,
        )

        schema = normalize_objectives(objectives)
        _GLOBAL_CONFIG["objective_schema"] = schema
        _GLOBAL_CONFIG["objectives"] = schema_to_objective_names(schema)

    logger.info("Updated global configuration")
    return True


def initialize(  # noqa: C901
    api_key: str | None = None,
    api_url: str | None = None,
    config: TraigentConfig | None = None,
    **kwargs: Any,
) -> bool:
    """Initialize TraiGent SDK for local or cloud operation.

    This function configures the SDK for integration with the Traigent backend,
    enabling seamless optimization with experiment tracking and storage.

    Args:
        api_key: API key for Traigent backend authentication (defaults to env var)
        api_url: Traigent backend URL (defaults to centralized config)
        config: TraigentConfig object with execution mode and settings
        **kwargs: Additional configuration parameters

    Returns:
        True if initialization successful

    Example:
        >>> # Edge Analytics mode initialization (backend URL from env or config)
        >>> # Set TRAIGENT_API_KEY environment variable for security
        >>> config = traigent.TraigentConfig.edge_analytics_mode()
        >>> traigent.initialize(config=config)

        >>> # Cloud mode with explicit URL
        >>> traigent.initialize(
        ...     api_key=os.getenv("TRAIGENT_API_KEY"),  # Use env var
        ...     api_url="https://api.traigent.ai"
        ... )

        >>> # Using environment variables (recommended)
        >>> # export TRAIGENT_BACKEND_URL="http://localhost:5000"
        >>> # export TRAIGENT_API_KEY="your-key-here"
        >>> traigent.initialize()
    """

    from traigent.config.backend_config import BackendConfig

    _configure_api_keys(api_key, BackendConfig)
    _configure_backend_url(api_url, BackendConfig)

    if config:
        _apply_config_settings(config)

    _apply_additional_overrides(kwargs)
    _configure_logging_settings(config)

    logger.info("TraiGent SDK initialized successfully")
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

    logger.info(f"TraiGent configured for {config.execution_mode} mode")


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
    return _API_KEY_MANAGER.get_api_key(provider)


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

    Example:
        >>> @traigent.optimize(
        ...     configuration_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.5, 0.8]}
        ... )
        ... def my_function(query: str) -> str:
        ...     config = traigent.get_trial_config()  # Gets trial-specific config
        ...     return call_llm(model=config["model"], temperature=config["temperature"])
        ...
        >>> # Run optimization - get_trial_config() works inside the function
        >>> result = traigent.optimize(my_function, dataset=my_data)
        >>> # Access best config via result
        >>> print(result.best_config)
    """
    from traigent.config.context import get_trial_context

    # Check if we're in an active trial context
    trial_ctx = get_trial_context()
    if trial_ctx is None:
        raise OptimizationStateError(
            "get_trial_config() can only be called during an active optimization trial. "
            "If you need the function's applied configuration, access it via "
            "my_function.current_config or the OptimizationResult.best_config.",
            current_state="NO_ACTIVE_TRIAL",
            expected_states=["OPTIMIZING"],
        )

    # Validate trial context has required fields
    if not isinstance(trial_ctx, dict):
        raise OptimizationStateError(
            "Trial context is corrupted - expected dict but got "
            f"{type(trial_ctx).__name__}.",
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
    config = get_config()

    # Convert TraigentConfig to dict if needed
    if isinstance(config, TraigentConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return dict(config)

    # Unexpected config type - this should never happen in normal operation
    # Raise error instead of silently returning empty dict to surface the bug
    raise OptimizationStateError(
        f"Trial config has invalid type: {type(config).__name__}. "
        "Expected TraigentConfig or dict. This may indicate corrupted context state.",
        current_state="INVALID_CONFIG_TYPE",
        expected_states=["OPTIMIZING"],
    )


def get_current_config() -> dict[str, Any]:
    """Get the current optimization configuration.

    .. deprecated::
        Use :func:`get_trial_config` instead. This function is deprecated
        because "current" is ambiguous - it could mean the trial config
        during optimization or the applied config after optimization.

        - During optimization: Use ``traigent.get_trial_config()``
        - After optimization: Use ``result.best_config`` or ``func.current_config``

    Returns:
        Dictionary with current configuration parameters.

    Note:
        This function will emit a deprecation warning. Unlike get_trial_config(),
        it returns an empty dict instead of raising an error when called outside
        an optimization trial (for backward compatibility).
    """
    warnings.warn(
        "get_current_config() is deprecated. Use get_trial_config() instead "
        "during optimization, or access func.current_config / result.best_config "
        "after optimization completes.",
        ConfigAccessWarning,
        stacklevel=2,
    )

    config = get_config()

    # Convert TraigentConfig to dict if needed
    if isinstance(config, TraigentConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return dict(config)

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

    Example:
        >>> # Override to focus on cost efficiency
        >>> cost_config = traigent.override_config(
        ...     objectives=["cost", "accuracy"],
        ...     configuration_space={"model": ["gpt-4o-mini"]},
        ...     max_trials=20
        ... )
        >>> results = my_agent.optimize(config_override=cost_config)
    """
    override: dict[str, Any] = {}

    if objectives is not None:
        override["objectives"] = objectives

    if configuration_space is not None:
        override["configuration_space"] = configuration_space

    if constraints is not None:
        override["constraints"] = constraints

    if max_trials is not None:
        if max_trials < 1:
            raise ValueError("max_trials must be >= 1")
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


def set_strategy(
    algorithm: str = "bayesian",
    algorithm_config: dict[str, Any] | None = None,
    parallel_workers: int | None = None,
    resource_limits: dict[str, Any] | None = None,
) -> StrategyConfig:
    """Configure optimization strategy and execution parameters.

    Args:
        algorithm: Optimization algorithm ("bayesian", "grid", "random", "genetic")
        algorithm_config: Algorithm-specific parameters
        parallel_workers: Number of parallel evaluation workers
        resource_limits: Memory, time, and compute constraints

    Returns:
        StrategyConfig object for use with optimization

    Example:
        >>> strategy = traigent.set_strategy(
        ...     algorithm="bayesian",
        ...     algorithm_config={
        ...         "acquisition_function": "expected_improvement",
        ...         "initial_random_samples": 5
        ...     },
        ...     parallel_workers=4
        ... )
        >>> results = my_agent.optimize(strategy=strategy)
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
        >>> print(strategies["bayesian"]["description"])
        >>> print(strategies["bayesian"]["parameters"])
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
                "parameters": {"description": "No additional parameters"},
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
    """Get TraiGent SDK version and capability information.

    Returns:
        Dict with version, supported features, and system info

    Example:
        >>> info = traigent.get_version_info()
        >>> print(f"TraiGent SDK v{info['version']}")
        >>> print(f"Available algorithms: {info['algorithms']}")
    """
    import platform
    import sys

    from traigent import __version__

    return {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "algorithms": list_optimizers(),
        "features": {
            "grid_search": True,
            "random_search": True,
            "bayesian_optimization": True,  # Available with scikit-learn
            "multi_objective": True,
            "constraint_handling": True,
            "async_evaluation": True,
            "parallel_evaluation": True,
            "result_persistence": True,  # Available in Sprint 2
            "visualization": True,  # Available in Sprint 3
        },
        "integrations": {
            "langchain": _check_integration("traigent.integrations.llms.langchain"),
            "openai": _check_integration("traigent.integrations.llms.openai"),
            "mlflow": _check_integration("traigent.integrations.observability.mlflow"),
            "wandb": _check_integration("traigent.integrations.observability.wandb"),
        },
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

    Example:
        >>> results = my_function.optimize()
        >>> insights = traigent.get_optimization_insights(results)
        >>> print("💡 Top 3 configurations discovered:")
        >>> for i, config in enumerate(insights['top_configurations'][:3]):
        ...     print(f"{i+1}. {config['config']} → {config['score']:.2%} accuracy, ${config.get('cost_analysis', {}).get('cost_per_query', 0):.3f}/1K queries")
    """
    return _get_optimization_insights(results)


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
