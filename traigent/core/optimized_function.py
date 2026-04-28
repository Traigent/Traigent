"""Core optimized function implementation.
# Traceability: CONC-Layer-Core CONC-Quality-Usability CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

Custom Evaluator Interface:
    Custom evaluation functions should have the signature:

    def custom_evaluate(func: Callable, config: dict[str, Any], example: EvaluationExample) -> ExampleResult

    Where:
    - func: The function being optimized
    - config: Configuration parameters to test
    - example: EvaluationExample from the dataset (contains input_data, expected_output, metadata)

    The function should return an ExampleResult with:
    - example_id: Unique identifier for this example
    - input_data: The input data used
    - expected_output: Expected output from the dataset
    - actual_output: Actual output from the function
    - metrics: Dict of metric names to float values
    - execution_time: Time taken to execute
    - success: Boolean indicating if evaluation succeeded
    - error_message: Optional error message if evaluation failed
"""

# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import copy
import inspect
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.config import get_provider
from traigent.config.parallel import coerce_parallel_config, merge_parallel_configs
from traigent.config.types import ExecutionMode, TraigentConfig, resolve_execution_mode
from traigent.core.ci_approval import check_ci_approval
from traigent.core.config_state_manager import ConfigStateManager, OptimizationState
from traigent.core.objectives import (
    ObjectiveSchema,
    create_default_objectives,
    normalize_objectives,
    schema_to_objective_names,
)
from traigent.core.optimization_pipeline import (
    HybridAPIEvaluatorOptions,
    collect_orchestrator_kwargs,
    create_effective_evaluator,
    create_traigent_config,
    create_workflow_traces_tracker,
    prepare_optimization_dataset,
    resolve_effective_parallel_config,
    resolve_execution_parameters,
)
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    load_inline_dataset,
)
from traigent.integrations.framework_override import override_context
from traigent.optimizers import get_optimizer
from traigent.tvl.options import TVLOptions
from traigent.tvl.spec_loader import load_tvl_spec
from traigent.utils.env_config import is_mock_llm
from traigent.utils.exceptions import (
    AuthenticationError,
    ConfigurationError,
    OptimizationError,
    TVLValidationError,
    ValidationError,
)
from traigent.utils.incentives import show_upgrade_hint
from traigent.utils.logging import get_logger
from traigent.utils.validation import (
    validate_config_space,
    validate_dataset_path,
    validate_objectives,
)

logger = get_logger(__name__)


def _resolve_callbacks(
    explicit_callbacks: list[Any] | None,
    decorator_callbacks: list[Any] | None,
    progress_bar: bool | None,
) -> list[Any]:
    """Resolve callbacks with optional auto-injection of ProgressBarCallback.

    Used by both ``optimize()`` and ``optimize_sync()`` to keep behavior
    consistent. Auto-enables a progress bar in interactive terminals unless
    the caller explicitly disables it.

    A lightweight :class:`ResultsTableCallback` is always appended (unless
    the full ``ProgressBarCallback`` is active, which already renders the
    table) so that users see a results summary even in non-interactive
    environments.

    Args:
        explicit_callbacks: Callbacks passed directly to optimize().
        decorator_callbacks: Callbacks stored on the decorator/OptimizedFunction.
        progress_bar: ``True`` to force, ``False`` to suppress, ``None`` for auto.

    Returns:
        Resolved list of callback instances.
    """
    from traigent.utils.callbacks import ProgressBarCallback, ResultsTableCallback

    callbacks = list(explicit_callbacks or decorator_callbacks or [])
    has_progress = any(isinstance(cb, ProgressBarCallback) for cb in callbacks)

    if progress_bar is not False and not has_progress:
        # True = always inject; None = inject only in interactive terminals
        if progress_bar is True or sys.stdin.isatty():
            callbacks.insert(0, ProgressBarCallback())
            has_progress = True

    has_table = any(isinstance(cb, ResultsTableCallback) for cb in callbacks)

    # ProgressBarCallback already renders the table, so avoid duplicate table
    # output when a standalone ResultsTableCallback is also present.
    if has_progress and has_table:
        callbacks = [cb for cb in callbacks if not isinstance(cb, ResultsTableCallback)]
        has_table = False

    # Always ensure a results table is rendered on completion.
    # ProgressBarCallback already does this, so only add the standalone
    # ResultsTableCallback when the progress bar is absent.
    if not has_progress and not has_table:
        callbacks.append(ResultsTableCallback())

    return callbacks


# Module-level flag to ensure cost warning is emitted only once per process
_COST_WARNING_EMITTED = False

# Error message for invalid configuration space type
_CONFIG_SPACE_TYPE_ERROR = "Configuration space must be a dictionary"

_CLOUD_FALLBACK_POLICIES = frozenset({"auto", "warn", "never"})


def _emit_cost_warning_once() -> None:
    """Emit cost warning once per process when optimization starts.

    This warning informs users that optimization will make multiple LLM API calls
    and that cost estimates are approximations. The warning is suppressed in mock mode.
    """
    global _COST_WARNING_EMITTED
    if _COST_WARNING_EMITTED:
        return
    if is_mock_llm():
        return

    _COST_WARNING_EMITTED = True

    # ANSI color codes for terminal styling
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Check if terminal supports colors (not redirected to file)
    use_colors = sys.stderr.isatty()

    if use_colors:
        msg = (
            f"\n{YELLOW}{BOLD}[!] COST WARNING{RESET}\n"
            f"{YELLOW}Traigent optimization will make multiple LLM API calls.{RESET}\n"
            f"Cost estimates are approximations based on {CYAN}litellm{RESET} library pricing.\n"
            f"Actual billing is determined by your LLM provider.\n\n"
            f"{BOLD}Configuration:{RESET}\n"
            f"  - Custom pricing file:   {CYAN}TRAIGENT_CUSTOM_MODEL_PRICING_FILE{RESET}\n"
            f"  - Custom pricing JSON:   {CYAN}TRAIGENT_CUSTOM_MODEL_PRICING_JSON{RESET}\n"
            f"  - Disable for testing:   {CYAN}TRAIGENT_MOCK_LLM=true{RESET}\n"
            f"  - Full details:          {CYAN}DISCLAIMER.md{RESET}\n"
        )
    else:
        msg = (
            "\n[!] COST WARNING\n"
            "Traigent optimization will make multiple LLM API calls.\n"
            "Cost estimates are approximations based on litellm library pricing.\n"
            "Actual billing is determined by your LLM provider.\n\n"
            "Configuration:\n"
            "  - Custom pricing file:   TRAIGENT_CUSTOM_MODEL_PRICING_FILE\n"
            "  - Custom pricing JSON:   TRAIGENT_CUSTOM_MODEL_PRICING_JSON\n"
            "  - Disable for testing:   TRAIGENT_MOCK_LLM=true\n"
            "  - Full details:          DISCLAIMER.md\n"
        )

    # Use warnings module for filterability; fallback to stderr on encoding errors
    import warnings

    try:
        warnings.warn(msg, UserWarning, stacklevel=2)
    except UnicodeEncodeError:
        # Fallback for ASCII-only locales
        try:
            print(msg, file=sys.stderr)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode(), file=sys.stderr)
    sys.stderr.flush()


class OptimizedFunction:
    """Wrapper for functions decorated with @traigent.optimize.

    This class provides the optimization interface for decorated functions,
    including methods to run optimization, get results, and analyze performance.
    """

    _csm: ConfigStateManager

    def __init__(
        self,
        func: Callable[..., Any],
        eval_dataset: (
            str | list[str | dict[str, Any] | EvaluationExample] | Dataset | None
        ) = None,
        objectives: list[str] | ObjectiveSchema | None = None,
        configuration_space: dict[str, Any] | None = None,
        config_space: dict[str, Any] | None = None,  # Backward compatibility
        default_config: dict[str, Any] | None = None,
        constraints: list[Callable[..., bool]] | None = None,
        injection_mode: str = "context",
        config_param: str | None = None,
        auto_override_frameworks: bool = False,
        framework_targets: list[str] | None = None,
        execution_mode: str = "edge_analytics",
        local_storage_path: str | None = None,
        minimal_logging: bool = True,
        custom_evaluator: Callable[..., Any] | None = None,
        scoring_function: Callable[..., Any] | None = None,
        metric_functions: dict[str, Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize optimized function wrapper.

        Args:
            func: Original function to optimize
            eval_dataset: Evaluation dataset path(s), inline examples, or Dataset object
            objectives: List of objectives to optimize or ObjectiveSchema
            configuration_space: Parameter search space
            default_config: Default configuration values
            constraints: List of constraint functions
            injection_mode: How to inject configuration ("context", "parameter", "decorator")
            config_param: Parameter name for injection_mode="parameter"
            auto_override_frameworks: Enable automatic framework parameter overrides
            framework_targets: List of framework class names to override (e.g., ["openai.OpenAI"])
            execution_mode: Execution mode ("edge_analytics", "privacy", "standard", "cloud")
            local_storage_path: Custom path for local storage (Edge Analytics mode only)
            minimal_logging: Use minimal logging in Edge Analytics mode
            custom_evaluator: Custom evaluation function for advanced use cases
            scoring_function: Simple scoring function that returns a score or dict of scores
            metric_functions: Dict of metric name to scoring function
            **kwargs: Additional configuration
        """
        # Extract decorator-provided metadata before core storage
        self._requested_execution_mode = kwargs.pop("requested_execution_mode", None)
        # Config persistence parameters
        self._auto_load_best = kwargs.pop("auto_load_best", False)
        self._load_from = kwargs.pop("load_from", None)
        # Store core parameters
        self._store_core_parameters(
            func,
            eval_dataset,
            objectives,
            default_config,
            constraints,
            injection_mode,
            config_param,
            auto_override_frameworks,
            framework_targets,
            execution_mode,
            local_storage_path,
            minimal_logging,
            custom_evaluator,
            scoring_function,
            metric_functions,
        )

        # Handle configuration space with backward compatibility
        self._setup_configuration_space(configuration_space, config_space)

        # Store additional parameters from kwargs
        self._store_additional_parameters(kwargs)

        # Initialize provider and validate configuration
        self._initialize_provider_and_validate()

        # Initialize state and function wrapper
        self._initialize_state_and_wrapper()

        # Auto-load config if requested
        self._maybe_auto_load_config()

        logger.debug(
            f"Created OptimizedFunction for {getattr(func, '__name__', str(func))}"
        )

    def _store_core_parameters(
        self,
        func,
        eval_dataset,
        objectives,
        default_config,
        constraints,
        injection_mode,
        config_param,
        auto_override_frameworks,
        framework_targets,
        execution_mode,
        local_storage_path,
        minimal_logging,
        custom_evaluator,
        scoring_function,
        metric_functions,
    ) -> None:
        """Store core initialization parameters."""
        self.func = func
        self.eval_dataset = eval_dataset

        # Handle ObjectiveSchema creation
        resolved_schema = normalize_objectives(objectives)
        if resolved_schema is None:
            resolved_schema = create_default_objectives(["accuracy"])

        self.objective_schema = resolved_schema

        self.default_config = default_config or {}
        self.constraints = constraints or []
        self.injection_mode = injection_mode
        self.config_param = config_param
        self.auto_override_frameworks = auto_override_frameworks
        self.framework_targets = framework_targets or []

        # Execution mode configuration
        requested_mode = getattr(self, "_requested_execution_mode", None)
        try:
            effective_mode_enum = resolve_execution_mode(execution_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from None

        privacy_alias_requested = False
        if effective_mode_enum is ExecutionMode.PRIVACY:
            effective_mode_enum = ExecutionMode.HYBRID
            privacy_alias_requested = True

        self._effective_execution_mode = effective_mode_enum
        if (
            requested_mode
            and requested_mode.lower() == "privacy"
            and effective_mode_enum is ExecutionMode.HYBRID
        ):
            display_mode = "privacy"
            privacy_alias_requested = True
        elif requested_mode:
            display_mode = requested_mode.lower()
        else:
            display_mode = effective_mode_enum.value
        self.execution_mode = display_mode
        self._privacy_alias_requested = privacy_alias_requested
        self.local_storage_path = local_storage_path
        self.minimal_logging = minimal_logging

        # Evaluation configuration
        self.custom_evaluator = custom_evaluator
        self.scoring_function = scoring_function
        self.metric_functions = metric_functions

    def _is_cloud_execution_mode(self) -> bool:
        """Return True when configured for managed cloud execution."""
        effective_mode = getattr(self, "_effective_execution_mode", None)
        if isinstance(effective_mode, ExecutionMode):
            mode_enum = effective_mode
        else:
            mode_enum = resolve_execution_mode(
                effective_mode, default=resolve_execution_mode(self.execution_mode)
            )
        return mode_enum is ExecutionMode.CLOUD

    def _setup_configuration_space(self, configuration_space, config_space) -> None:
        """Setup configuration space with backward compatibility."""
        # Backward compatibility: support both config_space and configuration_space
        if config_space is not None and configuration_space is None:
            try:
                self.configuration_space = config_space
            except ValidationError as e:
                if _CONFIG_SPACE_TYPE_ERROR in str(e):
                    raise TypeError(str(e)) from None
                raise
        else:
            try:
                self.configuration_space = configuration_space or {}
            except ValidationError as e:
                if _CONFIG_SPACE_TYPE_ERROR in str(e):
                    raise TypeError(str(e)) from None
                else:
                    raise

    def _store_callbacks(self, kwargs: dict[str, Any], sentinel: object) -> None:
        """Store callbacks parameter, normalizing to a list."""
        callbacks_value = kwargs.pop("callbacks", sentinel)
        if callbacks_value is sentinel or callbacks_value is None:
            self.callbacks = None
            return

        if isinstance(callbacks_value, list):
            self.callbacks = callbacks_value
            kwargs["callbacks"] = list(callbacks_value)
        elif isinstance(callbacks_value, tuple):
            normalized_callbacks = list(callbacks_value)
            self.callbacks = normalized_callbacks
            kwargs["callbacks"] = list(normalized_callbacks)
        else:
            normalized_callbacks = [callbacks_value]
            self.callbacks = normalized_callbacks
            kwargs["callbacks"] = list(normalized_callbacks)

    def _store_optional_param(
        self,
        kwargs: dict[str, Any],
        sentinel: object,
        key: str,
        default: Any,
        as_bool: bool = False,
    ) -> Any:
        """Store an optional parameter with sentinel-based defaults.

        Args:
            kwargs: The kwargs dict to read from and update.
            sentinel: Sentinel object for detecting missing values.
            key: The parameter key name.
            default: Default value when sentinel is detected.
            as_bool: If True, coerce the value to bool.

        Returns:
            The resolved value.
        """
        value = kwargs.pop(key, sentinel)
        if value is sentinel:
            resolved = default
        elif as_bool:
            resolved = bool(value)
            kwargs[key] = resolved
        else:
            resolved = value
            kwargs[key] = value
        return resolved

    def _store_additional_parameters(self, kwargs) -> None:
        """Store additional parameters from kwargs."""
        sentinel = object()

        # Decorator-provided runtime defaults
        self.algorithm = kwargs.pop("algorithm", "random")
        kwargs["algorithm"] = self.algorithm

        self.max_trials = kwargs.pop("max_trials", 50)
        kwargs["max_trials"] = self.max_trials

        self.timeout = kwargs.pop("timeout", None)
        kwargs["timeout"] = self.timeout

        save_to_value = kwargs.pop("save_to", sentinel)
        self.save_to = (
            None
            if save_to_value is sentinel or save_to_value is None
            else save_to_value
        )
        if self.save_to is not None:
            kwargs["save_to"] = self.save_to

        self._store_callbacks(kwargs, sentinel)

        self.use_cloud_service = self._store_optional_param(
            kwargs, sentinel, "use_cloud_service", False, as_bool=True
        )
        default_cloud_fallback_policy = (
            "never"
            if getattr(self, "_effective_execution_mode", None) is ExecutionMode.CLOUD
            else "auto"
        )
        raw_cloud_fallback_policy = kwargs.pop("cloud_fallback_policy", sentinel)
        if raw_cloud_fallback_policy is sentinel or raw_cloud_fallback_policy is None:
            self.cloud_fallback_policy = default_cloud_fallback_policy
        else:
            if not isinstance(raw_cloud_fallback_policy, str):
                raise ValueError(
                    "cloud_fallback_policy must be one of: auto, warn, never"
                )
            resolved_cloud_fallback_policy = raw_cloud_fallback_policy.strip().lower()
            if resolved_cloud_fallback_policy not in _CLOUD_FALLBACK_POLICIES:
                raise ValueError(
                    "cloud_fallback_policy must be one of: auto, warn, never"
                )
            self.cloud_fallback_policy = resolved_cloud_fallback_policy
        kwargs["cloud_fallback_policy"] = self.cloud_fallback_policy
        self.framework_target = self._store_optional_param(
            kwargs, sentinel, "framework_target", None
        )

        # Hybrid API evaluator configuration (execution_mode="hybrid_api")
        self.hybrid_api_endpoint = kwargs.pop("hybrid_api_endpoint", None)
        self.hybrid_api_tunable_id = kwargs.pop("tunable_id", None)
        self.hybrid_api_transport = kwargs.pop("hybrid_api_transport", None)
        self.hybrid_api_transport_type = kwargs.pop("hybrid_api_transport_type", "auto")
        self.hybrid_api_batch_size = kwargs.pop("hybrid_api_batch_size", 1)
        self.hybrid_api_batch_parallelism = kwargs.pop(
            "hybrid_api_batch_parallelism", 1
        )
        self.hybrid_api_keep_alive = bool(kwargs.pop("hybrid_api_keep_alive", True))
        self.hybrid_api_heartbeat_interval = kwargs.pop(
            "hybrid_api_heartbeat_interval", 30.0
        )
        self.hybrid_api_timeout = kwargs.pop("hybrid_api_timeout", None)
        self.hybrid_api_auth_header = kwargs.pop("hybrid_api_auth_header", None)
        self.hybrid_api_auto_discover_tvars = bool(
            kwargs.pop("hybrid_api_auto_discover_tvars", False)
        )

        # Execution knobs
        provided_parallel = coerce_parallel_config(kwargs.pop("parallel_config", None))
        combined_parallel, sources = merge_parallel_configs(
            [(provided_parallel, "decorator")]
        )
        self.parallel_config = combined_parallel
        self.parallel_config_sources = sources
        kwargs["parallel_config"] = combined_parallel
        if sources:
            kwargs["_parallel_config_sources"] = sources

        self.privacy_enabled = self._store_optional_param(
            kwargs, sentinel, "privacy_enabled", False, as_bool=True
        )
        if getattr(self, "_privacy_alias_requested", False):
            self.privacy_enabled = True
            kwargs["privacy_enabled"] = True
            self._privacy_alias_requested = False

        # Mock mode configuration
        self.mock_mode_config = self._store_optional_param(
            kwargs, sentinel, "mock_mode_config", None
        )
        self.max_examples = self._store_optional_param(
            kwargs, sentinel, "max_examples", None
        )
        self.max_total_examples = self._store_optional_param(
            kwargs, sentinel, "max_total_examples", None
        )
        self.samples_include_pruned = self._store_optional_param(
            kwargs, sentinel, "samples_include_pruned", True, as_bool=True
        )
        self.optimization_history_limit = kwargs.pop("optimization_history_limit", 100)
        if (
            not isinstance(self.optimization_history_limit, int)
            or self.optimization_history_limit < 1
        ):
            raise ValueError("optimization_history_limit must be >= 1")

        # Multi-agent configuration
        self.agents = self._store_optional_param(kwargs, sentinel, "agents", None)
        self.agent_prefixes = self._store_optional_param(
            kwargs, sentinel, "agent_prefixes", None
        )
        self.agent_measures = self._store_optional_param(
            kwargs, sentinel, "agent_measures", None
        )
        self.global_measures = self._store_optional_param(
            kwargs, sentinel, "global_measures", None
        )

        # JS runtime configuration
        self.js_runtime_config = self._store_optional_param(
            kwargs, sentinel, "js_runtime_config", None
        )

        # JS process pool (created lazily for parallel JS execution)
        self._js_process_pool: Any = None

        # Safety constraints
        self.safety_constraints = self._store_optional_param(
            kwargs, sentinel, "safety_constraints", None
        )

        # TVL promotion gate for statistical best-config selection
        self.promotion_gate = self._store_optional_param(
            kwargs, sentinel, "promotion_gate", None
        )

        self.kwargs = kwargs
        excluded_runtime_keys = {
            "algorithm",
            "max_trials",
            "timeout",
            "save_to",
            "callbacks",
            "parallel_config",
            "_parallel_config_sources",
            "use_cloud_service",
            "cloud_fallback_policy",
            "framework_target",
            "privacy_enabled",
            "mock_mode_config",
            "max_total_examples",
            "samples_include_pruned",
            # Multi-agent configuration
            "agents",
            "agent_prefixes",
            "agent_measures",
            "global_measures",
            # Safety constraints
            "safety_constraints",
        }
        self._decorator_runtime_overrides = {
            key: value
            for key, value in kwargs.items()
            if key not in excluded_runtime_keys
        }

        # Cloud service client (initialized lazily)
        self._cloud_client: Any | None = None

    def _initialize_provider_and_validate(self) -> None:
        """Initialize configuration provider and validate inputs."""
        # Validate basic inputs first
        self._validate_basic_inputs()

        # Get configuration provider
        # Convert enum to string value if needed
        injection_mode_str = (
            self.injection_mode.value
            if hasattr(self.injection_mode, "value")
            else self.injection_mode
        )
        try:
            self._provider = get_provider(
                injection_mode_str, config_param=self.config_param
            )
        except ConfigurationError as e:
            # Normalize provider/config errors to ValueError for decorator tests
            raise ValueError(str(e)) from None

        # Validate inputs
        self._validate_configuration()

    def _initialize_state_and_wrapper(self) -> None:
        """Initialize optimization state and function wrapper."""
        self._csm = ConfigStateManager(
            func=self.func,
            default_config=self.default_config,
            local_storage_path=getattr(self, "local_storage_path", None),
            configuration_space=getattr(self, "configuration_space", None),
            auto_load_best=getattr(self, "_auto_load_best", False),
            load_from=getattr(self, "_load_from", None),
            setup_wrapper_callback=self._setup_function_wrapper,
            optimization_history_limit=self.optimization_history_limit,
        )

        # Make function callable with current config
        self._setup_function_wrapper()

    # Backward-compatible proxy properties for state managed by ConfigStateManager.
    # Tests and internal optimization flow access these directly.
    @property  # noqa: F811
    def _state(self) -> OptimizationState:  # type: ignore[override]
        return self._csm._state

    @_state.setter
    def _state(self, value: OptimizationState) -> None:
        self._csm._state = value

    @property
    def _state_lock(self) -> threading.RLock:
        return self._csm._state_lock  # type: ignore[no-any-return]

    @property  # noqa: F811
    def _optimization_results(self) -> OptimizationResult | None:  # type: ignore[override]
        return self._csm._optimization_results  # type: ignore[no-any-return]

    @_optimization_results.setter
    def _optimization_results(self, value: OptimizationResult | None) -> None:
        self._csm._optimization_results = value

    @property  # noqa: F811
    def _optimization_history(self) -> list[OptimizationResult]:  # type: ignore[override]
        return self._csm._optimization_history  # type: ignore[no-any-return]

    @_optimization_history.setter
    def _optimization_history(self, value: list[OptimizationResult]) -> None:
        self._csm._optimization_history = value

    @property  # noqa: F811
    def _current_config(self) -> dict[str, Any]:  # type: ignore[override]
        return self._csm._current_config  # type: ignore[no-any-return]

    @_current_config.setter
    def _current_config(self, value: dict[str, Any]) -> None:
        self._csm._current_config = value

    @property  # noqa: F811
    def _best_config(self) -> dict[str, Any] | None:  # type: ignore[override]
        return self._csm._best_config  # type: ignore[no-any-return]

    @_best_config.setter
    def _best_config(self, value: dict[str, Any] | None) -> None:
        self._csm._best_config = value

    def _estimate_search_space_size(self) -> int:
        """Best-effort estimation of configuration combinations."""

        space = getattr(self, "configuration_space", {}) or {}
        if not isinstance(space, dict) or not space:
            return 0

        total = 1
        for values in space.values():
            if isinstance(values, (list, tuple, set)):
                total *= max(len(values), 1)
            else:
                return 0
        return total

    @property
    def config_space(self) -> dict[str, Any]:
        """Backward compatibility property for configuration_space."""
        return self.configuration_space

    @property
    def objectives(self) -> list[str]:
        """Objective names derived from the active objective schema."""
        result: list[str] = schema_to_objective_names(self.objective_schema)
        return result

    def _validate_basic_inputs(self) -> None:
        """Validate basic inputs and raise appropriate exceptions."""
        # Validate function
        if not callable(self.func):
            raise TypeError("func must be callable") from None

        # Validate max_trials and timeout for negative values
        if self.max_trials < 0:
            raise ValueError("max_trials must be non-negative")

        if self.timeout is not None and self.timeout < 0:
            raise ValueError("timeout must be non-negative")

    def _validate_configuration(self) -> None:
        """Validate optimization configuration."""
        self._validate_objectives()
        self._validate_config_space()
        self._validate_dataset()

    def _validate_objectives(self) -> None:
        """Validate objectives configuration."""
        try:
            validate_objectives(self.objectives)
        except ValidationError as e:
            if "At least one objective must be specified" in str(e):
                raise ValueError(str(e)) from None
            elif "Objectives must be a list" in str(e):
                raise TypeError(str(e)) from e
            else:
                raise

    def _validate_config_space(self) -> None:
        """Validate configuration space."""
        if self.configuration_space:
            try:
                validate_config_space(self.configuration_space)
            except ValidationError as e:
                if _CONFIG_SPACE_TYPE_ERROR in str(e):
                    raise TypeError(str(e)) from None
                else:
                    raise
        elif self.configuration_space == {}:
            if self._allows_empty_configuration_space():
                logger.debug(
                    "Allowing empty local configuration_space because "
                    "hybrid_api_auto_discover_tvars is enabled"
                )
                return
            # Empty config space should raise ValueError with helpful message
            raise ValueError(
                "Configuration space cannot be empty. Please specify at least one parameter to optimize "
                "in the @traigent.optimize decorator. Example: configuration_space={'temperature': [0.0, 0.5, 1.0], "
                "'model': ['gpt-3.5-turbo', 'gpt-4']}"
            )

    def _allows_empty_configuration_space(self) -> bool:
        """Whether empty config space is allowed for this function."""
        effective_mode = getattr(self, "_effective_execution_mode", None)
        return bool(
            effective_mode is ExecutionMode.HYBRID_API
            and self.hybrid_api_auto_discover_tvars
        )

    def _validate_dataset(self) -> None:
        """Validate dataset configuration."""
        if isinstance(self.eval_dataset, str):
            # Skip dataset validation in tests when the file doesn't exist
            if not (
                os.environ.get("PYTEST_CURRENT_TEST")
                or self.eval_dataset in ["test.jsonl", "data.jsonl"]
            ):
                validate_dataset_path(self.eval_dataset)
            return

        if isinstance(self.eval_dataset, list):
            if all(isinstance(item, str) for item in self.eval_dataset):
                if not os.environ.get("PYTEST_CURRENT_TEST"):
                    validate_dataset_path(self.eval_dataset)
                return

            if all(
                isinstance(item, (dict, EvaluationExample))
                for item in self.eval_dataset
            ):
                load_inline_dataset(self.eval_dataset)
                return

            raise ConfigurationError(
                "eval_dataset list must contain only dataset paths or inline examples"
            )

    def _setup_function_wrapper(self) -> None:
        """Setup function wrapper that uses current configuration."""
        # Use provider to inject configuration
        self._wrapped_func = self._provider.inject_config(
            self.func, self._current_config, self.config_param
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the optimized function callable."""
        # If framework overrides are enabled, use them during function call
        if self.auto_override_frameworks and self.framework_targets:
            with override_context(self.framework_targets):
                return self._wrapped_func(*args, **kwargs)
        else:
            return self._wrapped_func(*args, **kwargs)

    def _prepare_algorithm_kwargs(
        self, algorithm_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge decorator overrides into algorithm kwargs and validate."""
        decorator_overrides = getattr(self, "_decorator_runtime_overrides", {})
        if decorator_overrides:
            merged = {**decorator_overrides, **algorithm_kwargs}
        else:
            merged = dict(algorithm_kwargs)

        if "parallel_trials" in merged:
            raise ValueError(
                "parallel_trials is not a valid parameter for optimize(). "
                "Use parallel_config={'trial_concurrency': N} instead."
            )
        return merged

    def _validate_objectives_input(
        self,
        objectives: ObjectiveSchema | Sequence[str] | None,
        algorithm_kwargs: dict[str, Any],
    ) -> tuple[
        ObjectiveSchema | Sequence[str] | None, ObjectiveSchema | Sequence[str] | None
    ]:
        """Validate and extract objectives from inputs.

        Returns:
            Tuple of (objectives, legacy_objectives) after validation.
        """
        legacy_objectives = algorithm_kwargs.pop("objectives", None)
        legacy_orientations = algorithm_kwargs.pop("objective_orientations", None)
        legacy_weights = algorithm_kwargs.pop("objective_weights", None)

        if legacy_orientations is not None or legacy_weights is not None:
            raise ValueError(
                "objective_orientations/objective_weights are no longer supported. "
                "Provide an ObjectiveSchema instead."
            )
        if objectives is not None and legacy_objectives is not None:
            raise ValueError(
                "objectives provided both via parameter and inside algorithm_kwargs"
            )
        return objectives, legacy_objectives

    def _process_tvl_options(
        self,
        tvl_spec: str | Path | None,
        tvl_environment: str | None,
        tvl: TVLOptions | dict[str, Any] | None,
        algorithm_kwargs: dict[str, Any],
        algorithm: str | None,
        max_trials: int | None,
        timeout: float | None,
        configuration_space: dict[str, Any] | None,
        objectives: ObjectiveSchema | Sequence[str] | None,
    ) -> tuple[
        str | None,
        int | None,
        float | None,
        dict[str, Any] | None,
        ObjectiveSchema | Sequence[str] | None,
        dict[str, Any] | None,
    ]:
        """Process TVL options and return updated values.

        Returns:
            Tuple of (algorithm, max_trials, timeout, configuration_space, objectives, tvl_state)
        """
        runtime_tvl_spec_kw = algorithm_kwargs.pop("tvl_spec", None)
        runtime_tvl_env_kw = algorithm_kwargs.pop("tvl_environment", None)
        runtime_tvl_bundle_kw = algorithm_kwargs.pop("tvl", None)

        tvl_options_runtime = self._resolve_runtime_tvl_options(
            tvl_spec if tvl_spec is not None else runtime_tvl_spec_kw,
            tvl_environment if tvl_environment is not None else runtime_tvl_env_kw,
            tvl if tvl is not None else runtime_tvl_bundle_kw,
        )

        if not tvl_options_runtime:
            return algorithm, max_trials, timeout, configuration_space, objectives, None

        try:
            tvl_artifact = load_tvl_spec(**tvl_options_runtime.to_kwargs())  # type: ignore[arg-type]
        except TVLValidationError as exc:
            raise ValidationError(exc.message) from exc

        if configuration_space is None:
            configuration_space = tvl_artifact.configuration_space
        if objectives is None and tvl_artifact.objective_schema is not None:
            objectives = tvl_artifact.objective_schema

        tvl_state = self._apply_runtime_tvl_artifact(tvl_artifact)
        algorithm, max_trials, timeout = self._apply_tvl_runtime_overrides(
            algorithm,
            max_trials,
            timeout,
            algorithm_kwargs,
            tvl_artifact.runtime_overrides(),
        )
        return (
            algorithm,
            max_trials,
            timeout,
            configuration_space,
            objectives,
            tvl_state,
        )

    def _restore_tvl_state(self, tvl_state: dict[str, Any] | None) -> None:
        """Restore state modified by TVL processing."""
        if not tvl_state:
            return
        if "constraints" in tvl_state:
            self.constraints = tvl_state["constraints"]
        if "default_config" in tvl_state:
            self.default_config = tvl_state["default_config"]

    def _apply_discovered_objective_schema(
        self, state: dict[str, Any], discovered_spec: dict[str, Any]
    ) -> None:
        """Apply discovered objective schema and preserve prior state."""
        discovered_schema = discovered_spec.get("objective_schema")
        if discovered_schema is None:
            return
        state["objective_schema"] = self.objective_schema
        self.objective_schema = discovered_schema

    def _apply_discovered_constraints(
        self, state: dict[str, Any], discovered_spec: dict[str, Any]
    ) -> None:
        """Merge discovered constraints into the current optimization state."""
        discovered_constraints = discovered_spec.get("constraints")
        if not (isinstance(discovered_constraints, list) and discovered_constraints):
            return
        state["constraints"] = list(self.constraints or [])
        self.constraints = list(self.constraints or []) + discovered_constraints

    def _apply_discovered_default_config(
        self, state: dict[str, Any], discovered_spec: dict[str, Any]
    ) -> None:
        """Replace default config from discovery results when provided."""
        discovered_defaults = discovered_spec.get("default_config")
        if not (isinstance(discovered_defaults, dict) and discovered_defaults):
            return
        state["default_config"] = copy.deepcopy(self.default_config)
        self.default_config = discovered_defaults.copy()

    def _apply_discovered_runtime_overrides(
        self,
        algorithm: str | None,
        max_trials: int | None,
        evaluation_timeout: float | None,
        algorithm_kwargs: dict[str, Any],
        discovered_spec: dict[str, Any],
    ) -> tuple[str | None, int | None, float | None]:
        """Apply runtime overrides returned by hybrid discovery."""
        runtime_overrides = discovered_spec.get("runtime_overrides")
        if not (isinstance(runtime_overrides, dict) and runtime_overrides):
            return algorithm, max_trials, evaluation_timeout
        return self._apply_tvl_runtime_overrides(
            algorithm,
            max_trials,
            evaluation_timeout,
            algorithm_kwargs,
            runtime_overrides,
        )

    def _apply_discovered_promotion_policy(
        self, state: dict[str, Any], discovered_spec: dict[str, Any]
    ) -> None:
        """Apply discovered promotion policy if present."""
        promotion_policy = discovered_spec.get("promotion_policy")
        if promotion_policy is None:
            return

        from traigent.tvl.promotion_gate import PromotionGate

        state["promotion_gate"] = getattr(self, "promotion_gate", None)
        self.promotion_gate = PromotionGate.from_policy(
            promotion_policy=promotion_policy,
            objective_schema=self.objective_schema,
        )

    @staticmethod
    def _merge_discovered_metrics(
        objectives: Sequence[str], discovered_measures: Sequence[Any]
    ) -> list[str]:
        """Build a unique evaluator metric list preserving existing order."""
        merged_metrics: list[str] = []
        for metric_name in [*objectives, *discovered_measures]:
            if isinstance(metric_name, str) and metric_name not in merged_metrics:
                merged_metrics.append(metric_name)
        return merged_metrics

    def _apply_discovered_measures(
        self,
        state: dict[str, Any],
        evaluator: BaseEvaluator,
        discovered_spec: dict[str, Any],
    ) -> None:
        """Update evaluator metrics from discovered measures."""
        discovered_measures = discovered_spec.get("measures")
        if not isinstance(discovered_measures, list):
            return
        if not (
            hasattr(evaluator, "metrics")
            and isinstance(getattr(evaluator, "metrics", None), list)
        ):
            return

        state["evaluator_metrics"] = list(evaluator.metrics)
        merged_metrics = self._merge_discovered_metrics(
            self.objectives, discovered_measures
        )
        if merged_metrics:
            evaluator.metrics = merged_metrics

    async def _apply_hybrid_discovery_overrides(
        self,
        evaluator: BaseEvaluator,
        *,
        algorithm: str | None,
        max_trials: int | None,
        evaluation_timeout: float | None,
        effective_config_space: dict[str, Any],
        algorithm_kwargs: dict[str, Any],
    ) -> tuple[str | None, int | None, float | None, dict[str, Any], dict[str, Any]]:
        """Apply hybrid config-space discovery data to runtime settings."""
        from traigent.hybrid.discovery import merge_config_spaces

        state: dict[str, Any] = {}

        discover = getattr(evaluator, "discover_config_space", None)
        if not callable(discover):
            return (
                algorithm,
                max_trials,
                evaluation_timeout,
                effective_config_space,
                state,
            )

        discovered_config_space = await discover()
        if not discovered_config_space:
            raise ValueError(
                "Hybrid API config-space discovery returned no tunables. "
                "Verify GET /traigent/v1/config-space."
            )

        effective_config_space = merge_config_spaces(
            discovered_config_space,
            effective_config_space if effective_config_space else None,
        )

        discovered_spec = getattr(evaluator, "optimization_spec", None) or {}
        if not isinstance(discovered_spec, dict):
            return (
                algorithm,
                max_trials,
                evaluation_timeout,
                effective_config_space,
                state,
            )

        self._apply_discovered_objective_schema(state, discovered_spec)
        self._apply_discovered_constraints(state, discovered_spec)
        self._apply_discovered_default_config(state, discovered_spec)
        algorithm, max_trials, evaluation_timeout = (
            self._apply_discovered_runtime_overrides(
                algorithm,
                max_trials,
                evaluation_timeout,
                algorithm_kwargs,
                discovered_spec,
            )
        )
        self._apply_discovered_promotion_policy(state, discovered_spec)
        self._apply_discovered_measures(state, evaluator, discovered_spec)

        return (
            algorithm,
            max_trials,
            evaluation_timeout,
            effective_config_space,
            state,
        )

    def _restore_hybrid_discovery_state(
        self,
        state: dict[str, Any] | None,
        evaluator: BaseEvaluator | None = None,
    ) -> None:
        """Restore state mutated by hybrid discovery application."""
        if not state:
            return
        if "constraints" in state:
            self.constraints = state["constraints"]
        if "default_config" in state:
            self.default_config = state["default_config"]
        if "objective_schema" in state:
            self.objective_schema = state["objective_schema"]
        if "promotion_gate" in state:
            self.promotion_gate = state["promotion_gate"]
        if (
            evaluator is not None
            and "evaluator_metrics" in state
            and hasattr(evaluator, "metrics")
        ):
            evaluator.metrics = state["evaluator_metrics"]

    async def optimize(
        self,
        algorithm: str | None = None,
        max_trials: int | None = None,
        timeout: float | None = None,
        save_to: str | None = None,
        custom_evaluator: Callable[..., Any] | None = None,
        callbacks: list[Callable[..., Any]] | None = None,
        configuration_space: dict[str, Any] | None = None,
        objectives: ObjectiveSchema | Sequence[str] | None = None,
        tvl_spec: str | Path | None = None,
        tvl_environment: str | None = None,
        tvl: TVLOptions | dict[str, Any] | None = None,
        progress_bar: bool | None = None,
        **algorithm_kwargs: Any,
    ) -> OptimizationResult:
        """Run optimization on the function.

        Args:
            algorithm: Optimization algorithm to use. If None, uses the
                algorithm specified in the decorator (self.algorithm).
            max_trials: Maximum number of trials
            timeout: Maximum optimization time in seconds
            save_to: Path to save results
            custom_evaluator: Custom evaluation function that takes (func, config, input_data)
                            and returns metrics dict. If provided, overrides default evaluation.
            callbacks: List of callback objects for progress tracking
            configuration_space: Override configuration space for this optimization run.
                                Takes precedence over decorator configuration_space.
            objectives: Optional override objectives (list of names or ObjectiveSchema)
            tvl_spec: Optional TVL spec path to load at runtime.
            tvl_environment: Environment overlay to apply when loading the spec.
            tvl: Structured TVL options (dict or TVLOptions) for runtime overrides.
            progress_bar: Controls the live progress bar during optimization.
                ``True`` forces a progress bar even in non-interactive mode,
                ``False`` suppresses it, ``None`` (default) auto-enables in
                interactive terminals (``sys.stdin.isatty()``).
            **algorithm_kwargs: Additional algorithm-specific parameters.
                For grid search (algorithm="grid"):
                    - parameter_order: dict[str, int | float] controlling iteration order.
                      Lower values = varies slowest, higher values = varies fastest.
                      Alias: ``order``.
                      Example: ``parameter_order={"model": 0, "temperature": 1}``

        Returns:
            OptimizationResult with trial results and best configuration

        Raises:
            OptimizationError: If optimization fails
        """
        logger.info(f"Starting optimization of {self.func.__name__}")
        _emit_cost_warning_once()

        algorithm_kwargs = self._prepare_algorithm_kwargs(algorithm_kwargs)
        objectives, legacy_objectives = self._validate_objectives_input(
            objectives, algorithm_kwargs
        )

        # Process TVL options
        algorithm, max_trials, timeout, configuration_space, objectives, tvl_state = (
            self._process_tvl_options(
                tvl_spec,
                tvl_environment,
                tvl,
                algorithm_kwargs,
                algorithm,
                max_trials,
                timeout,
                configuration_space,
                objectives,
            )
        )

        # Normalize configuration_space to handle Range/IntRange/LogRange/Choices objects
        if configuration_space is not None:
            from traigent.api.parameter_ranges import normalize_configuration_space

            configuration_space, _ = normalize_configuration_space(configuration_space)

        runtime_objective_input = (
            objectives if objectives is not None else legacy_objectives
        )
        try:
            runtime_schema = normalize_objectives(runtime_objective_input)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

        original_schema = self.objective_schema
        if runtime_schema is not None:
            self.objective_schema = runtime_schema

        timeout = timeout if timeout is not None else getattr(self, "timeout", None)
        save_to = save_to if save_to is not None else getattr(self, "save_to", None)
        callbacks = _resolve_callbacks(
            callbacks, getattr(self, "callbacks", None), progress_bar
        )

        try:
            validate_objectives(self.objectives)
            result = await self._execute_optimization(
                algorithm=algorithm,
                max_trials=max_trials,
                timeout=timeout,
                save_to=save_to,
                custom_evaluator=custom_evaluator,
                callbacks=callbacks,
                configuration_space=configuration_space,
                algorithm_kwargs=algorithm_kwargs,
            )
        finally:
            if runtime_schema is not None:
                self.objective_schema = original_schema
            self._restore_tvl_state(tvl_state)

        return result

    def optimize_sync(
        self,
        algorithm: str | None = None,
        max_trials: int | None = None,
        timeout: float | None = None,
        save_to: str | None = None,
        custom_evaluator: Callable[..., Any] | None = None,
        callbacks: list[Callable[..., Any]] | None = None,
        configuration_space: dict[str, Any] | None = None,
        objectives: ObjectiveSchema | Sequence[str] | None = None,
        tvl_spec: str | Path | None = None,
        tvl_environment: str | None = None,
        tvl: TVLOptions | dict[str, Any] | None = None,
        progress_bar: bool | None = None,
        **algorithm_kwargs: Any,
    ) -> OptimizationResult:
        """Run optimization synchronously (convenience wrapper).

        This is a synchronous wrapper around optimize() for users who don't need
        async functionality. It handles the event loop creation internally.

        For async code, use the async optimize() method directly:
            result = await func.optimize(...)

        Args:
            algorithm: Optimization algorithm to use
            max_trials: Maximum number of trials
            timeout: Maximum optimization time in seconds
            save_to: Path to save results
            custom_evaluator: Custom evaluation function
            callbacks: List of callback objects for progress tracking
            configuration_space: Override configuration space
            objectives: Optional override objectives
            tvl_spec: Optional TVL spec path
            tvl_environment: Environment overlay for TVL spec
            tvl: Structured TVL options
            progress_bar: ``True`` to force, ``False`` to suppress, ``None``
                (default) auto-enables in interactive terminals.
            **algorithm_kwargs: Additional algorithm parameters

        Returns:
            OptimizationResult with trial results and best configuration

        Example:
            # Simple synchronous usage (no asyncio.run needed)
            result = my_function.optimize_sync(max_trials=10)
            print(result.best_config)

            # Equivalent async usage
            result = asyncio.run(my_function.optimize(max_trials=10))
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.optimize(
            algorithm=algorithm,
            max_trials=max_trials,
            timeout=timeout,
            save_to=save_to,
            custom_evaluator=custom_evaluator,
            callbacks=callbacks,
            configuration_space=configuration_space,
            objectives=objectives,
            tvl_spec=tvl_spec,
            tvl_environment=tvl_environment,
            tvl=tvl,
            progress_bar=progress_bar,
            **algorithm_kwargs,
        )

        if loop is not None and loop.is_running():
            # Already in an async context - can't use asyncio.run
            # Create a new thread to run the coroutine
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # No event loop running - safe to use asyncio.run
            return asyncio.run(coro)

    def _resolve_runtime_tvl_options(
        self,
        spec: str | Path | None,
        environment: str | None,
        bundle: TVLOptions | dict[str, Any] | None,
    ) -> TVLOptions | None:
        if bundle is None and spec is None:
            if environment:
                raise ValueError("tvl_environment requires a tvl_spec path")
            return None

        options: TVLOptions | None
        if isinstance(bundle, TVLOptions):
            options = bundle
        elif isinstance(bundle, dict):
            options = TVLOptions.model_validate(bundle)
        else:
            options = None

        if options is None and spec is None:
            raise ValueError("tvl options require a spec path")

        if options is None and spec is not None:
            options = TVLOptions(spec_path=str(spec), environment=environment)
            return options

        if options is None:
            raise ValueError("TVL options could not be resolved")
        if spec is not None and Path(options.spec_path) != Path(spec):
            raise ValueError("Conflicting TVL specs provided at runtime")
        if environment:
            options = options.merged_with(environment=environment)
        return options

    def _apply_runtime_tvl_artifact(self, artifact) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if artifact.constraints:
            state["constraints"] = list(self.constraints or [])
            self.constraints = list(self.constraints or []) + artifact.constraints
        if artifact.default_config:
            state["default_config"] = copy.deepcopy(self.default_config)
            self.default_config = artifact.default_config.copy()
        return state

    @staticmethod
    def _apply_tvl_runtime_overrides(
        algorithm: str | None,
        max_trials: int | None,
        timeout: float | None,
        algorithm_kwargs: dict[str, Any],
        overrides: dict[str, Any],
    ) -> tuple[str | None, int | None, float | None]:
        updated_algorithm = algorithm or overrides.get("algorithm")
        updated_max_trials = (
            max_trials if max_trials is not None else overrides.get("max_trials")
        )
        updated_timeout = timeout if timeout is not None else overrides.get("timeout")

        for key in (
            "parallel_config",
            "max_total_examples",
            "samples_include_pruned",
            "plateau_window",
            "plateau_epsilon",
            "tie_breakers",
            "tvl_parameter_agents",
        ):
            if key in overrides and key not in algorithm_kwargs:
                algorithm_kwargs[key] = overrides[key]

        # Fail fast on invalid parallel_trials usage - users must use parallel_config
        if "parallel_trials" in overrides or "parallel_trials" in algorithm_kwargs:
            raise ValueError(
                "parallel_trials is not a valid parameter for optimize(). "
                "Use parallel_config={'trial_concurrency': N} instead."
            )

        return updated_algorithm, updated_max_trials, updated_timeout

    def _hybrid_api_evaluator_kwargs(
        self,
        *,
        force_auto_discover_tvars: bool | None = None,
    ) -> dict[str, Any]:
        """Build hybrid API kwargs for evaluator construction."""
        auto_discover = (
            force_auto_discover_tvars
            if force_auto_discover_tvars is not None
            else self.hybrid_api_auto_discover_tvars
        )
        return {
            "hybrid_api_options": HybridAPIEvaluatorOptions(
                endpoint=self.hybrid_api_endpoint,
                tunable_id=self.hybrid_api_tunable_id,
                transport=self.hybrid_api_transport,
                transport_type=self.hybrid_api_transport_type,
                batch_size=self.hybrid_api_batch_size,
                batch_parallelism=self.hybrid_api_batch_parallelism,
                keep_alive=self.hybrid_api_keep_alive,
                heartbeat_interval=self.hybrid_api_heartbeat_interval,
                timeout=self.hybrid_api_timeout,
                auth_header=self.hybrid_api_auth_header,
                auto_discover_tvars=auto_discover,
            ),
        }

    def _create_effective_evaluator(
        self,
        timeout: float | None,
        custom_evaluator: Callable[..., Any] | None,
        effective_batch_size: int | None,
        effective_thread_workers: int | None,
        effective_privacy_enabled: bool,
        *,
        force_auto_discover_tvars: bool | None = None,
    ) -> BaseEvaluator:
        """Create the appropriate evaluator. Delegates to optimization_pipeline."""
        evaluator, js_pool = create_effective_evaluator(
            timeout=timeout,
            custom_evaluator=custom_evaluator,
            effective_batch_size=effective_batch_size,
            effective_thread_workers=effective_thread_workers,
            effective_privacy_enabled=effective_privacy_enabled,
            objectives=self.objectives,
            js_runtime_config=getattr(self, "js_runtime_config", None),
            execution_mode=self.execution_mode,
            mock_mode_config=self.mock_mode_config,
            metric_functions=self.metric_functions,
            scoring_function=self.scoring_function,
            decorator_custom_evaluator=self.custom_evaluator,
            **self._hybrid_api_evaluator_kwargs(
                force_auto_discover_tvars=force_auto_discover_tvars
            ),
        )
        if js_pool is not None:
            self._js_process_pool = js_pool
        return evaluator

    def _build_optimization_orchestrator(
        self,
        optimizer: Any,
        evaluator: BaseEvaluator,
        max_trials: int | None,
        max_total_examples_value: int | None,
        timeout: float | None,
        callbacks: list[Callable[..., Any]] | None,
        traigent_config: TraigentConfig,
        effective_parallel_trials: int | None,
        samples_include_pruned_value: bool,
        algorithm_kwargs: dict[str, Any],
    ) -> OptimizationOrchestrator:
        """Build the optimization orchestrator with all configuration."""
        orchestrator_kwargs = collect_orchestrator_kwargs(
            algorithm_kwargs,
            samples_include_pruned_value,
            default_config=getattr(self, "default_config", None),
            constraints=getattr(self, "constraints", None),
            agents=getattr(self, "agents", None),
            agent_prefixes=getattr(self, "agent_prefixes", None),
            agent_measures=getattr(self, "agent_measures", None),
            global_measures=getattr(self, "global_measures", None),
            promotion_gate=getattr(self, "promotion_gate", None),
        )

        # Auto-initialize workflow traces tracker if backend is configured
        workflow_traces_tracker = create_workflow_traces_tracker(traigent_config)

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=max_trials,
            max_total_examples=max_total_examples_value,
            timeout=timeout,
            callbacks=callbacks or [],
            config=traigent_config,
            parallel_trials=effective_parallel_trials,
            objectives=self.objectives,
            objective_schema=self.objective_schema,
            workflow_traces_tracker=workflow_traces_tracker,
            **orchestrator_kwargs,
        )

        orchestrator.samples_include_pruned = samples_include_pruned_value
        return orchestrator

    async def _run_and_finalize_optimization(
        self,
        orchestrator: OptimizationOrchestrator,
        dataset: Dataset,
        effective_config_space: dict[str, Any],
        save_to: str | None,
    ) -> OptimizationResult:
        """Run optimization with context managers and finalize results."""
        from traigent.config.context import ConfigurationSpaceContext

        # Set state to OPTIMIZING before starting
        self._state = OptimizationState.OPTIMIZING

        try:
            with ConfigurationSpaceContext(effective_config_space):
                if self.auto_override_frameworks and self.framework_targets:
                    with override_context(self.framework_targets):
                        result = await orchestrator.optimize(
                            func=self._wrapped_func,
                            dataset=dataset,
                        )
                else:
                    result = await orchestrator.optimize(
                        func=self._wrapped_func,
                        dataset=dataset,
                    )

            # Store results
            self._optimization_results = result
            self._csm.append_optimization_result(result)

            # Update current config to best found
            if result.best_config:
                self._current_config = result.best_config.copy()
                self._best_config = result.best_config.copy()
                self._setup_function_wrapper()

            # Set state to OPTIMIZED on success
            self._state = OptimizationState.OPTIMIZED

        except Exception:
            # Set state to ERROR on failure
            self._state = OptimizationState.ERROR
            raise
        finally:
            # Clean up JS process pool if it was created
            if self._js_process_pool is not None:
                try:
                    await self._js_process_pool.shutdown()
                    logger.debug("JS process pool shut down successfully")
                except Exception as e:
                    logger.warning("Error shutting down JS process pool: %s", e)
                finally:
                    self._js_process_pool = None

        # Save results if requested
        if save_to:
            self.save_optimization_results(save_to)

        logger.info(
            f"Optimization completed: {len(result.trials)} trials, "
            f"best score: {'N/A' if result.best_score is None else f'{result.best_score:.4f}'}"
        )

        # Show upgrade hints after optimization completion (Edge Analytics mode only)
        if self.traigent_config.is_edge_analytics_mode():  # type: ignore[has-type]
            try:
                show_upgrade_hint(
                    "session_complete",
                    trial_count=len(result.trials),
                    best_score=result.best_score,
                )
            except Exception as e:
                logger.debug(f"Failed to show upgrade hint: {e}")

        return result  # type: ignore[no-any-return]

    async def _try_cloud_execution(
        self,
        dataset: Dataset,
        max_trials: int | None,
        timeout: float | None,
        effective_config_space: dict[str, Any],
        algorithm_kwargs: dict[str, Any],
    ) -> OptimizationResult | None:
        """Try cloud execution, returning None if fallback to local is needed."""
        use_cloud = self._is_cloud_execution_mode() and (
            max_trials is None or max_trials > 0
        )
        if not use_cloud:
            return None

        from traigent.cloud.client import (
            CloudRemoteExecutionUnavailableError,
            CloudServiceError,
        )

        try:
            return await self._optimize_with_cloud_service(
                dataset,
                max_trials,
                timeout,
                configuration_space=effective_config_space,
                **algorithm_kwargs,
            )
        except (AuthenticationError, ConfigurationError, ValidationError):
            raise
        except CloudRemoteExecutionUnavailableError:
            raise
        except CloudServiceError as e:
            if self.cloud_fallback_policy == "never":
                raise
            logger.warning("Cloud optimization failed, falling back to local: %s", e)
        except OSError as e:  # Includes TimeoutError and ConnectionError (subclasses)
            if self.cloud_fallback_policy == "never":
                raise
            logger.warning(
                "Cloud optimization failed (transient), falling back to local: %s", e
            )
        except Exception as e:
            if self.cloud_fallback_policy == "never":
                raise
            logger.warning(
                "Cloud optimization failed unexpectedly, falling back to local: %s",
                e,
                exc_info=True,
            )
        return None

    def _apply_mock_config_overrides(
        self, algorithm: str, optimizer_kwargs: dict[str, Any]
    ) -> str:
        """No-op retained for backward compatibility.

        Historically this method consulted ``self.mock_mode_config`` to
        override the optimizer algorithm and to inject ``random_seed`` into
        ``optimizer_kwargs``. As part of the F5 retirement of the mock-mode
        flag, ``mock_mode_config`` is now fully inert: callers may still pass
        the parameter through public APIs, but it must not change optimizer
        selection or seeding. A stray production config in the past silently
        rerouted real optimizations to a different algorithm with a fixed
        seed, so we now ignore it entirely. Real seeding should go through
        the normal ``algorithm_kwargs`` / ``random_seed`` parameter path.
        """
        return algorithm

    async def _execute_optimization(
        self,
        *,
        algorithm: str | None,
        max_trials: int | None,
        timeout: float | None,
        save_to: str | None,
        custom_evaluator: Callable[..., Any] | None,
        callbacks: list[Callable[..., Any]] | None,
        configuration_space: dict[str, Any] | None,
        algorithm_kwargs: dict[str, Any],
    ) -> OptimizationResult:
        """Execute optimization assuming objective schema is already resolved.

        This method orchestrates the optimization process by delegating to
        specialized helper methods for each phase of execution.
        """
        hybrid_auto_discovery_enabled = self._allows_empty_configuration_space()
        effective_privacy_enabled = bool(getattr(self, "privacy_enabled", False))
        hybrid_discovery_state: dict[str, Any] | None = None
        precreated_evaluator: BaseEvaluator | None = None

        if hybrid_auto_discovery_enabled:
            # Discovery must happen before we resolve parallel settings because
            # the remote config-space can override optimizer/runtime parameters.
            # Use neutral worker settings for this bootstrap evaluator instance.
            precreated_evaluator = self._create_effective_evaluator(
                timeout=timeout,
                custom_evaluator=custom_evaluator,
                effective_batch_size=None,
                effective_thread_workers=None,
                effective_privacy_enabled=effective_privacy_enabled,
                force_auto_discover_tvars=True,
            )

            pre_discovery_space = (
                configuration_space
                if configuration_space is not None
                else self.configuration_space
            )
            try:
                (
                    algorithm,
                    max_trials,
                    timeout,
                    discovered_config_space,
                    hybrid_discovery_state,
                ) = await self._apply_hybrid_discovery_overrides(
                    evaluator=precreated_evaluator,
                    algorithm=algorithm,
                    max_trials=max_trials,
                    evaluation_timeout=timeout,
                    effective_config_space=pre_discovery_space or {},
                    algorithm_kwargs=algorithm_kwargs,
                )
            except Exception:
                close = getattr(precreated_evaluator, "close", None)
                if callable(close):
                    await close()
                raise
            configuration_space = discovered_config_space

        # Phase 1: Resolve and validate parameters
        algorithm, max_trials, effective_config_space = resolve_execution_parameters(
            algorithm,
            max_trials,
            configuration_space,
            fallback_config_space=self.configuration_space,
            fallback_algorithm=cast(str, getattr(self, "algorithm", "grid")),
            fallback_max_trials=getattr(self, "max_trials", None),
        )

        # Phase 2: Create TraigentConfig and check CI approval
        traigent_config = create_traigent_config(
            execution_mode=self.execution_mode,
            local_storage_path=self.local_storage_path,
            minimal_logging=self.minimal_logging,
            privacy_enabled=getattr(self, "privacy_enabled", False),
        )
        self.traigent_config = traigent_config
        self._check_ci_approval()

        # Phase 3: Prepare dataset
        dataset = self._load_dataset()
        dataset, max_total_examples_value, samples_include_pruned_value = (
            prepare_optimization_dataset(
                dataset,
                algorithm_kwargs,
                max_examples=getattr(self, "max_examples", None),
                max_total_examples=getattr(self, "max_total_examples", None),
                samples_include_pruned=getattr(self, "samples_include_pruned", True),
            )
        )
        if max_total_examples_value is not None:
            self.max_total_examples = max_total_examples_value

        # Phase 4: Try cloud execution if applicable
        cloud_result = await self._try_cloud_execution(
            dataset, max_trials, timeout, effective_config_space, algorithm_kwargs
        )
        if cloud_result is not None:
            return cloud_result

        # Phase 5: Resolve parallel configuration
        effective_parallel_trials, effective_batch_size, effective_thread_workers = (
            resolve_effective_parallel_config(
                algorithm_kwargs,
                decorator_parallel_config=self.parallel_config,
                config_space_size=self._estimate_search_space_size(),
                is_async_func=asyncio.iscoroutinefunction(self.func),
            )
        )

        # Phase 5.5: Pop legacy cost-estimation param before optimizer creation.
        # It is not consumed by orchestrator and should not leak into optimizer kwargs.
        algorithm_kwargs.pop("invocations_per_example", None)

        # Phase 6: Determine privacy and create evaluator
        if precreated_evaluator is not None:
            evaluator = precreated_evaluator
        else:
            evaluator = self._create_effective_evaluator(
                timeout=timeout,
                custom_evaluator=custom_evaluator,
                effective_batch_size=effective_batch_size,
                effective_thread_workers=effective_thread_workers,
                effective_privacy_enabled=effective_privacy_enabled,
            )

        # Phase 7: Create optimizer
        optimizer_kwargs = algorithm_kwargs.copy()
        if max_trials:
            optimizer_kwargs["max_trials"] = max_trials

        # Apply mock config overrides if present
        algorithm = self._apply_mock_config_overrides(algorithm, optimizer_kwargs)

        optimizer = get_optimizer(
            algorithm, effective_config_space, self.objectives, **optimizer_kwargs
        )

        # Update TraigentConfig with final privacy setting
        traigent_config.privacy_enabled = effective_privacy_enabled

        # Phase 8: Build orchestrator
        orchestrator = self._build_optimization_orchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=max_trials,
            max_total_examples_value=max_total_examples_value,
            timeout=timeout,
            callbacks=callbacks,
            traigent_config=traigent_config,
            effective_parallel_trials=effective_parallel_trials,
            samples_include_pruned_value=samples_include_pruned_value,
            algorithm_kwargs=algorithm_kwargs,
        )

        # Phase 9: Run optimization and finalize
        try:
            return await self._run_and_finalize_optimization(
                orchestrator=orchestrator,
                dataset=dataset,
                effective_config_space=effective_config_space,
                save_to=save_to,
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise OptimizationError(f"Optimization failed: {e}") from e
        finally:
            self._restore_hybrid_discovery_state(hybrid_discovery_state, evaluator)

    async def _optimize_with_cloud_service(
        self,
        dataset: Dataset,
        max_trials: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run optimization through the reserved Traigent Cloud path.

        Remote cloud execution is not available yet. The cloud client is
        expected to fail closed with guidance to use hybrid for portal-tracked
        optimization.

        Args:
            dataset: Evaluation dataset
            max_trials: Maximum number of trials
            timeout: Optimization timeout
            **kwargs: Additional arguments

        Returns:
            OptimizationResult from cloud service when the future path is implemented
        """
        from traigent.cloud.client import TraigentCloudClient

        # Initialize cloud client if not already done
        if self._cloud_client is None:
            self._cloud_client = TraigentCloudClient(enable_fallback=False)

        if max_trials is not None and max_trials <= 0:
            logger.info("Cloud optimization skipped due to max_trials=0.")
            return self._build_empty_result("cloud_service")

        async with self._cloud_client as client:
            # Extract configuration_space from kwargs if provided
            config_space_override = kwargs.pop("configuration_space", None)
            effective_config_space = self._resolve_cloud_config_space(
                config_space_override
            )
            cloud_result = await self._execute_cloud_service_optimization(
                client,
                dataset,
                effective_config_space,
                max_trials,
            )
            cloud_payload = await self._extract_cloud_result_payload(cloud_result)
            result = self._build_cloud_optimization_result(cloud_payload)

            # Store results
            self._optimization_results = result
            self._csm.append_optimization_result(result)

            # Update current config and best config (consistent with local optimization)
            if result.best_config:
                self._current_config = result.best_config.copy()
                self._best_config = result.best_config.copy()
                self._setup_function_wrapper()

            logger.info(
                "Cloud optimization completed: %s trials, %.1f%% cost reduction",
                cloud_payload["trials_count"],
                cloud_payload["cost_reduction"] * 100,
            )

            return result

    @staticmethod
    async def _resolve_awaitable_value(value: Any) -> Any:
        """Await values only when necessary."""
        return await value if inspect.isawaitable(value) else value

    def _resolve_cloud_config_space(
        self, config_space_override: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Resolve the configuration space used for cloud optimization."""
        if config_space_override is not None:
            return config_space_override
        return self.configuration_space

    async def _execute_cloud_service_optimization(
        self,
        client: Any,
        dataset: Dataset,
        effective_config_space: dict[str, Any],
        max_trials: int | None,
    ) -> Any:
        """Run the cloud optimization request inside the configuration context."""
        from traigent.config.context import ConfigurationSpaceContext

        with ConfigurationSpaceContext(effective_config_space):
            cloud_candidate = await client.optimize_function(
                function_name=self.func.__name__,
                dataset=dataset,
                configuration_space=effective_config_space,
                objectives=self.objectives,
                max_trials=max_trials if max_trials is not None else 50,
                local_function=self.func,
            )
            return await self._resolve_awaitable_value(cloud_candidate)

    async def _resolve_cloud_result_attribute(
        self, cloud_result: Any, attribute: str, default: Any
    ) -> Any:
        """Resolve a cloud-result attribute that may be awaitable."""
        value = getattr(cloud_result, attribute, default)
        return await self._resolve_awaitable_value(value)

    @staticmethod
    def _coerce_cloud_metric_map(raw_metrics: Any) -> dict[str, float]:
        """Normalize cloud metrics into a string-to-float mapping."""
        if not isinstance(raw_metrics, dict):
            return {}

        best_metrics: dict[str, float] = {}
        for key, raw_value in raw_metrics.items():
            try:
                best_metrics[str(key)] = float(raw_value)
            except (TypeError, ValueError):
                continue
        return best_metrics

    @staticmethod
    def _coerce_cloud_int(raw_value: Any, default: int = 0) -> int:
        """Convert cloud numeric fields to ints with a stable fallback."""
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_cloud_float(raw_value: Any, default: float = 0.0) -> float:
        """Convert cloud numeric fields to floats with a stable fallback."""
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_optional_cloud_int(raw_value: Any) -> int | None:
        """Convert optional cloud integer fields when valid."""
        return raw_value if isinstance(raw_value, int) else None

    def _select_cloud_best_score(self, best_metrics: dict[str, float]) -> float:
        """Resolve the best score using the primary objective when available."""
        primary_objective = self.objectives[0] if self.objectives else None
        if primary_objective and primary_objective in best_metrics:
            return best_metrics[primary_objective]
        if best_metrics:
            return next(iter(best_metrics.values()))
        return 0.0

    async def _extract_cloud_result_payload(self, cloud_result: Any) -> dict[str, Any]:
        """Normalize cloud result attributes into a local payload dict."""
        best_config_raw = await self._resolve_cloud_result_attribute(
            cloud_result, "best_config", {}
        )
        best_metrics_raw = await self._resolve_cloud_result_attribute(
            cloud_result, "best_metrics", {}
        )
        trials_count_raw = await self._resolve_cloud_result_attribute(
            cloud_result, "trials_count", 0
        )
        cost_reduction_raw = await self._resolve_cloud_result_attribute(
            cloud_result, "cost_reduction", 0.0
        )
        optimization_time_raw = await self._resolve_cloud_result_attribute(
            cloud_result, "optimization_time", 0.0
        )
        subset_used_raw = await self._resolve_cloud_result_attribute(
            cloud_result, "subset_used", False
        )
        subset_size_raw = await self._resolve_cloud_result_attribute(
            cloud_result, "subset_size", None
        )

        best_config = (
            best_config_raw.copy() if isinstance(best_config_raw, dict) else {}
        )
        best_metrics = self._coerce_cloud_metric_map(best_metrics_raw)
        return {
            "best_config": best_config,
            "best_metrics": best_metrics,
            "trials_count": self._coerce_cloud_int(trials_count_raw),
            "cost_reduction": self._coerce_cloud_float(cost_reduction_raw),
            "optimization_time": self._coerce_cloud_float(optimization_time_raw),
            "subset_used": bool(subset_used_raw),
            "subset_size": self._coerce_optional_cloud_int(subset_size_raw),
            "best_score": self._select_cloud_best_score(best_metrics),
        }

    def _build_cloud_optimization_result(self, cloud_result: Any) -> OptimizationResult:
        """Convert a cloud result payload into a standard optimization result."""
        from traigent.api.types import TrialResult, TrialStatus

        best_config = cloud_result["best_config"]
        best_metrics = cloud_result["best_metrics"]
        optimization_time = cloud_result["optimization_time"]
        mock_trial = TrialResult(
            trial_id="cloud_best",
            config=best_config,
            metrics=best_metrics,
            status=TrialStatus.COMPLETED,
            duration=optimization_time,
            timestamp=datetime.now(UTC),
            metadata={},
        )

        return OptimizationResult(
            trials=[mock_trial],  # Cloud service doesn't expose all trials
            best_config=best_config,
            best_score=cloud_result["best_score"],
            optimization_id=f"cloud_{int(time.time())}",
            duration=optimization_time,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=self.objectives,
            algorithm="cloud_service",
            timestamp=datetime.now(UTC),
            metadata={
                "cloud_service": True,
                "cost_reduction": cloud_result["cost_reduction"],
                "subset_used": cloud_result["subset_used"],
                "subset_size": cloud_result["subset_size"],
                "trials_count": cloud_result["trials_count"],
            },
        )

    def _load_dataset(self) -> Dataset:
        """Load evaluation dataset.

        Returns:
            Dataset object for evaluation

        Raises:
            ConfigurationError: If dataset cannot be loaded
        """
        if isinstance(self.eval_dataset, Dataset):
            return self.eval_dataset

        elif isinstance(self.eval_dataset, str):
            try:
                return Dataset.from_jsonl(self.eval_dataset)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load dataset from {self.eval_dataset}: {e}"
                ) from e

        elif isinstance(self.eval_dataset, list):
            if all(isinstance(item, str) for item in self.eval_dataset):
                # Multiple datasets - combine them
                all_examples = []
                for path in self.eval_dataset:
                    try:
                        dataset = Dataset.from_jsonl(path)
                        all_examples.extend(dataset.examples)
                    except Exception as e:
                        raise ConfigurationError(
                            f"Failed to load dataset from {path}: {e}"
                        ) from e

                return Dataset(
                    examples=all_examples,
                    name="combined_dataset",
                    description=f"Combined dataset from {len(self.eval_dataset)} files",
                )

            if all(
                isinstance(item, (dict, EvaluationExample))
                for item in self.eval_dataset
            ):
                try:
                    return load_inline_dataset(self.eval_dataset)
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to load inline dataset: {e}"
                    ) from e

            raise ConfigurationError(
                "eval_dataset list must contain only dataset paths or inline examples"
            )

        else:
            raise ConfigurationError(f"Invalid dataset type: {type(self.eval_dataset)}")

    def _build_empty_result(self, algorithm: str) -> OptimizationResult:
        """Create a result representing a skipped optimization run."""

        best_config = (
            self._current_config.copy() if hasattr(self, "_current_config") else {}
        )
        now = datetime.now(UTC)

        return OptimizationResult(
            trials=[],
            best_config=best_config,
            best_score=None,
            optimization_id=f"no_trials_{int(time.time())}",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=self.objectives,
            algorithm=algorithm,
            timestamp=now,
            metadata={"reason": "max_trials=0"},
        )

    def _check_ci_approval(self) -> None:
        """Check if approval is required and granted for CI runs."""
        check_ci_approval(self.traigent_config)

    def get_best_config(self) -> dict[str, Any] | None:
        """Get the best configuration found during optimization."""
        return self._csm.get_best_config()  # type: ignore[no-any-return]

    def get_optimization_results(self) -> OptimizationResult | None:
        """Get the latest optimization results."""
        return self._csm.get_optimization_results()  # type: ignore[no-any-return]

    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get history of all optimization runs."""
        return self._csm.get_optimization_history()  # type: ignore[no-any-return]

    def is_optimization_complete(self) -> bool:
        """Check if optimization has been completed."""
        return self._csm.is_optimization_complete()  # type: ignore[no-any-return]

    def reset_optimization(self) -> None:
        """Reset optimization state and restore default configuration."""
        self._csm.reset_optimization()

    def save_optimization_results(self, path: str) -> None:
        """Save optimization results to file."""
        self._csm.save_optimization_results(path)

    def load_optimization_results(self, path: str) -> None:
        """Load optimization results from file."""
        self._csm.load_optimization_results(path)

    @property
    def state(self) -> OptimizationState:
        """Get the current lifecycle state of this optimized function."""
        return self._csm.state

    @property
    def best_config(self) -> dict[str, Any] | None:
        """Get the best configuration found during optimization."""
        return self._csm.best_config  # type: ignore[no-any-return]

    @property
    def current_config(self) -> dict[str, Any]:
        """Get the configuration this function uses when called."""
        return self._csm.current_config  # type: ignore[no-any-return]

    def _maybe_auto_load_config(self) -> None:
        """Auto-load configuration if requested. Delegates to ConfigStateManager."""
        self._csm.maybe_auto_load_config()

    def set_config(self, config: dict[str, Any]) -> None:
        """Set current configuration manually."""
        self._csm.set_config(config)

    def apply_best_config(self, results: OptimizationResult | None = None) -> bool:
        """Apply best configuration from optimization results."""
        return self._csm.apply_best_config(  # type: ignore[no-any-return]
            results,
            get_wrapped_func=lambda: self._wrapped_func,
            set_wrapped_func=lambda f: setattr(self, "_wrapped_func", f),
        )

    def export_config(
        self,
        path: str | Path,
        *,
        format: str = "slim",  # noqa: A002
        include_metadata: bool = True,
    ) -> Path:
        """Export the best configuration to a file."""
        return self._csm.export_config(  # type: ignore[no-any-return]
            path, format=format, include_metadata=include_metadata
        )

    def _load_config_from_path(self, path: str) -> dict[str, Any] | None:
        """Load config from a file path. Delegates to ConfigStateManager."""
        return self._csm._load_config_from_path(path)  # type: ignore[no-any-return]

    def _find_latest_config_path(self) -> str | None:
        """Find the latest saved config path. Delegates to ConfigStateManager."""
        return self._csm._find_latest_config_path()  # type: ignore[no-any-return]

    def cleanup(self, *, preserve_config: bool = True) -> None:
        """Clean up optimization artifacts to free memory.

        Call this after you're done analyzing optimization results
        and no longer need the trial history. This is useful for
        long-running services to prevent memory leaks.

        Args:
            preserve_config: If True (default), keeps the best_config applied.
                If False, reverts to default_config.

        Example:
            # After optimization and analysis
            result = my_func.optimize(...)
            analyze_results(result)

            # Clean up to free memory, but keep best config
            my_func.cleanup()

            # Or reset completely
            my_func.cleanup(preserve_config=False)
        """
        # Clear optimization history
        self._optimization_history.clear()
        self._optimization_results = None

        # Clear any accumulated stats
        if hasattr(self, "_stats"):
            self._stats.clear()

        # Clear metrics cache if present
        if hasattr(self, "_metrics_cache"):
            self._metrics_cache.clear()

        # Optionally revert config
        if not preserve_config:
            self._current_config = self.default_config.copy()
            self._best_config = None
            self._state = OptimizationState.UNOPTIMIZED
            self._setup_function_wrapper()

        # Clean provider state if provider supports it
        if hasattr(self, "_config_provider") and self._config_provider:
            if hasattr(self._config_provider, "cleanup"):
                self._config_provider.cleanup()

        logger.debug(f"Cleaned up optimization artifacts for {self.func.__name__}")

    def reset(self) -> None:
        """Fully reset the function to its initial state.

        This clears all optimization history, reverts to default_config,
        and sets state to UNOPTIMIZED. Use this to run a fresh optimization.

        Equivalent to calling cleanup(preserve_config=False).

        Example:
            # Run optimization
            result1 = my_func.optimize(...)

            # Reset for a new optimization run
            my_func.reset()

            # Run fresh optimization
            result2 = my_func.optimize(...)
        """
        self.cleanup(preserve_config=False)
        logger.info(f"Reset {self.func.__name__} to initial state")

    @property
    def configuration_space(self) -> dict[str, Any]:
        """Get configuration space."""
        return self._configuration_space

    @configuration_space.setter
    def configuration_space(self, value: dict[str, Any]) -> None:
        """Set configuration space with validation."""
        if value:
            validate_config_space(value)
        self._configuration_space = value or {}

    def __repr__(self) -> str:
        """String representation of optimized function."""
        return (
            f"OptimizedFunction({self.func.__name__}, "
            f"objectives={self.objectives}, "
            f"config_space_size={len(self.configuration_space)}, "
            f"optimized={self._optimization_results is not None})"
        )

    @property
    def __name__(self) -> str:
        """Get function name."""
        return getattr(self.func, "__name__", "OptimizedFunction")

    @property
    def __doc__(self) -> str | None:  # type: ignore[override]
        """Get function docstring."""
        return getattr(self.func, "__doc__", None)

    def get_doc(self) -> str | None:
        """Get function docstring."""
        return getattr(self.func, "__doc__", None)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the function with current configuration."""
        return self(*args, **kwargs)
