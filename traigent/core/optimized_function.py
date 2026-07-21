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
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeVar, cast

from traigent.api.strategy_presets import (
    NormalizedStrategyPreset,
    is_strategy_preset_name,
    normalize_strategy_preset,
)
from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.config import get_provider
from traigent.config.parallel import coerce_parallel_config, merge_parallel_configs
from traigent.config.types import (
    ExecutionIntent,
    ExecutionMode,
    ResolvedExecutionPolicy,
    TraigentConfig,
    normalize_algorithm_name,
    resolve_execution_policy,
    validate_execution_mode,
)
from traigent.core.ci_approval import check_ci_approval
from traigent.core.config_state_manager import ConfigStateManager, OptimizationState
from traigent.core.cost_enforcement import is_cost_preapproved, normalize_cost_approved
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.execution_policy_runtime import (
    SOURCE_CLOUD_BRAIN,
    SOURCE_EXPLICIT_LOCAL,
    SOURCE_LOCAL_FALLBACK,
    SOURCE_OFFLINE,
    CloudBrainUnavailableError,
    backend_optimization_strategy_for_algorithm,
    backend_egress_disabled,
    exception_is_connectivity,
    initial_result_source,
    is_offline_requested,
    mark_local_fallback,
    policy_allows_cloud_fallback,
    policy_from_config,
    policy_is_cloud_brain,
    policy_is_cloud_required,
    policy_requires_cloud,
    unsupported_backend_smart_algorithm_message,
)
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
from traigent.defaults import DEFAULT_MAX_TRIALS
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
from traigent.utils.artifact_fingerprints import build_artifact_fingerprints
from traigent.utils.cost_calculator import (
    UnknownModelError,
    find_models_missing_price_coverage,
)
from traigent.utils.env_config import is_mock_llm, is_strict_cost_accounting
from traigent.utils.exceptions import (
    AuthenticationError,
    ConfigurationError,
    OptimizationError,
    TVLValidationError,
    ValidationError,
)
from traigent.utils.function_identity import is_coroutine_callable
from traigent.utils.incentives import show_upgrade_hint
from traigent.utils.logging import get_logger
from traigent.utils.validation import (
    validate_config_space,
    validate_dataset_path,
    validate_objectives,
)

logger = get_logger(__name__)

# Type parameters for the @optimize decorator's generic return type.
# _P captures the wrapped function's parameter spec; _R captures its return type.
_P = ParamSpec("_P")
_R = TypeVar("_R")


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
_MODEL_CONFIG_KEYS = frozenset({"model", "model_name", "model_id", "engine"})


def _is_model_config_key(key: object) -> bool:
    return isinstance(key, str) and key.strip().lower() in _MODEL_CONFIG_KEYS


def _iter_config_model_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]

    if isinstance(value, Mapping):
        choices = value.get("values") or value.get("choices")
        if choices is not None:
            return _iter_config_model_values(choices)
        fixed_value = value.get("value")
        if isinstance(fixed_value, str):
            return [fixed_value]
        return []

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, str) and item.strip()]

    range_values = getattr(value, "values", None)
    if isinstance(range_values, Sequence) and not isinstance(
        range_values, (str, bytes, bytearray)
    ):
        return [item for item in range_values if isinstance(item, str) and item.strip()]

    return []


def _dedupe_model_ids(model_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(model_ids))


def _extract_config_space_model_ids(config_space: Mapping[str, Any]) -> list[str]:
    model_ids: list[str] = []
    for key, value in config_space.items():
        if _is_model_config_key(key):
            model_ids.extend(_iter_config_model_values(value))
    return _dedupe_model_ids(model_ids)


def _extract_config_model_ids(config: Mapping[str, Any]) -> list[str]:
    model_ids: list[str] = []
    for key, value in config.items():
        if _is_model_config_key(key) and isinstance(value, str) and value.strip():
            model_ids.append(value)
    return _dedupe_model_ids(model_ids)


def _format_model_id_list(model_ids: Sequence[str]) -> str:
    return ", ".join(f"`{model_id}`" for model_id in model_ids)


def _format_unpriced_model_warning(
    model_ids: Sequence[str], occurrences: Mapping[str, int] | None = None
) -> str:
    formatted = _format_model_id_list(model_ids)
    # When occurrence counts are available (the runtime path, #1597), make the
    # warning quantitative and explicit that $0 means UNKNOWN spend — not
    # verified-free — so a reported total_cost is understood as a lower bound.
    total_calls = sum(occurrences.get(m, 0) for m in model_ids) if occurrences else 0
    call_note = (
        f" across {total_calls} call{'s' if total_calls != 1 else ''}"
        if total_calls
        else ""
    )
    unknown_spend_note = (
        " These are recorded as $0 because pricing is UNKNOWN, not because "
        "usage was free — any reported total_cost is a lower bound on actual "
        "spend."
        if total_calls
        else ""
    )
    if len(model_ids) == 1:
        return (
            f"Cost for {formatted} is unavailable{call_note} — results will "
            f"report $0 for it.{unknown_spend_note} Set "
            "TRAIGENT_CUSTOM_MODEL_PRICING_JSON, or contact "
            "Traigent to add coverage."
        )
    return (
        f"Costs for {formatted} are unavailable{call_note} — results will "
        f"report $0 for them.{unknown_spend_note} Set "
        "TRAIGENT_CUSTOM_MODEL_PRICING_JSON, or contact "
        "Traigent to add coverage."
    )


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
            f"Traigent limits are best-effort local guardrails, not provider billing caps.\n"
            f"Actual billing is determined by your LLM provider.\n\n"
            f"{BOLD}Configuration:{RESET}\n"
            f"  - Custom pricing file:   {CYAN}TRAIGENT_CUSTOM_MODEL_PRICING_FILE{RESET}\n"
            f"  - Custom pricing JSON:   {CYAN}TRAIGENT_CUSTOM_MODEL_PRICING_JSON{RESET}\n"
            f"  - Local mock helper:     {CYAN}from traigent.testing import enable_mock_mode_for_quickstart{RESET}\n"
            f"  - Then call:            {CYAN}enable_mock_mode_for_quickstart(){RESET}\n"
            f"  - Legacy env mock:       {CYAN}TRAIGENT_MOCK_LLM=true{RESET} (non-production only)\n"
            f"  - Set provider caps:     billing limits in your LLM/cloud provider account\n"
            f"  - Full details:          {CYAN}DISCLAIMER.md{RESET}\n"
        )
    else:
        msg = (
            "\n[!] COST WARNING\n"
            "Traigent optimization will make multiple LLM API calls.\n"
            "Cost estimates are approximations based on litellm library pricing.\n"
            "Traigent limits are best-effort local guardrails, not provider billing caps.\n"
            "Actual billing is determined by your LLM provider.\n\n"
            "Configuration:\n"
            "  - Custom pricing file:   TRAIGENT_CUSTOM_MODEL_PRICING_FILE\n"
            "  - Custom pricing JSON:   TRAIGENT_CUSTOM_MODEL_PRICING_JSON\n"
            "  - Local mock helper:     from traigent.testing import enable_mock_mode_for_quickstart\n"
            "  - Then call:             enable_mock_mode_for_quickstart()\n"
            "  - Legacy env mock:       TRAIGENT_MOCK_LLM=true (non-production only)\n"
            "  - Set provider caps:     billing limits in your LLM/cloud provider account\n"
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


class OptimizedFunction(Generic[_P, _R]):
    """Wrapper for functions decorated with @traigent.optimize.

    This class provides the optimization interface for decorated functions,
    including methods to run optimization, get results, and analyze performance.

    The class is generic over the wrapped function's parameter spec (*_P*) and
    return type (*_R*), so static type-checkers can preserve the original
    function signature through the decorator and still expose the
    :meth:`optimize` / :meth:`optimize_sync` method surface.
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
        default_config: dict[str, Any] | None = None,
        constraints: list[Callable[..., bool]] | None = None,
        injection_mode: str = "context",
        config_param: str | None = None,
        auto_override_frameworks: bool = False,
        framework_targets: list[str] | None = None,
        execution_mode: str = "local",
        local_storage_path: str | None = None,
        minimal_logging: bool = True,
        custom_evaluator: Callable[..., Any] | None = None,
        scoring_function: Callable[..., Any] | None = None,
        metric_functions: dict[str, Callable[..., Any]] | None = None,
        evaluator_definition_id: str | None = None,
        effectuation: bool = False,
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
            execution_mode: Execution mode ("local", "hybrid", "hybrid_api"; "edge_analytics" and "privacy" are legacy aliases)
            local_storage_path: Custom path for local storage (local mode only)
            minimal_logging: Use minimal logging in local mode
            custom_evaluator: Custom evaluation function for advanced use cases
            scoring_function: Simple scoring function that returns a score or dict of scores
            metric_functions: Dict of metric name to scoring function
            effectuation: Opt-in executable TVAR effectuation.
            **kwargs: Additional configuration
        """
        # Extract decorator-provided metadata before core storage
        max_trials_explicit = kwargs.pop("_max_trials_explicit", None)
        self._max_trials_uses_sdk_default = (
            "max_trials" not in kwargs
            if max_trials_explicit is None
            else not bool(max_trials_explicit)
        )
        self._max_trials_default_notice_logged = False
        self._requested_execution_mode = kwargs.pop("requested_execution_mode", None)
        provided_execution_policy = kwargs.pop("execution_policy", None)
        self.execution_policy = (
            provided_execution_policy
            if isinstance(provided_execution_policy, ResolvedExecutionPolicy)
            else None
        )
        self.external_service_evaluator: Any = None
        self.hybrid_api_options: Any = None
        self.offline: bool = False
        # Surrogate (pre-screen) scorer; set post-construction by the optimize
        # decorator and read by _create_effective_evaluator. None when unconfigured.
        self._surrogate_evaluator: Callable[..., Any] | None = None
        # Explicit surrogate descriptor id; set post-construction by the optimize
        # decorator (``surrogate_evaluator_name``). None -> id derived from the
        # callable. A runtime optimize() name overrides this decorator value.
        self._surrogate_evaluator_name: str | None = None
        # Config persistence parameters
        self._auto_load_best = kwargs.pop("auto_load_best", False)
        self._load_from = kwargs.pop("load_from", None)
        self._config_id = kwargs.pop("config_id", None)
        self._best_config_source = kwargs.pop("best_config_source", "off")
        self._best_config_strict = kwargs.pop("best_config_strict", False)
        self._best_config_cache_dir = kwargs.pop("best_config_cache_dir", None)
        self._best_config_cache_ttl_seconds = kwargs.pop(
            "best_config_cache_ttl_seconds", 24 * 60 * 60
        )
        self._best_config_stale_ok_ttl_seconds = kwargs.pop(
            "best_config_stale_ok_ttl_seconds", None
        )
        self._best_config_environment = kwargs.pop("best_config_environment", None)
        self._enable_auto_load_dev_logs = kwargs.pop("enable_auto_load_dev_logs", None)
        # Guided-generation defaults configured at decoration time; consumed by
        # optimize_with_guidance when not overridden at the call site.
        self.prompt_rewrite_options = kwargs.pop("prompt_rewrite", None)
        self.grow_dataset_options = kwargs.pop("grow_dataset", None)
        self.skill_train_options = kwargs.pop("skill_train", None)
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
            evaluator_definition_id,
            effectuation,
        )

        # Handle configuration space
        self._setup_configuration_space(configuration_space)

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
        evaluator_definition_id,
        effectuation,
    ) -> None:
        """Store core initialization parameters."""
        self.func = func
        self.eval_dataset = eval_dataset
        # Guided generation may swap in a grown dataset that must persist across
        # rounds (see set_eval_dataset_override / optimize_with_guidance).
        self._dataset_override: Dataset | None = None

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
        if self.execution_policy is None:
            self.execution_policy = resolve_execution_policy(
                execution_mode=execution_mode,
                source_hint="optimized_function",
            )

        try:
            effective_mode_enum = validate_execution_mode(execution_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from None

        privacy_alias_requested = any(
            isinstance(value, str) and value.strip().lower() == "privacy"
            for value in (execution_mode, self._requested_execution_mode)
        )

        self._effective_execution_mode = effective_mode_enum
        self.execution_mode = effective_mode_enum.value
        self.offline = self.execution_policy.offline
        self._privacy_alias_requested = privacy_alias_requested
        self._privacy_alias_enabled = False
        self.local_storage_path = local_storage_path
        self.minimal_logging = minimal_logging

        # Evaluation configuration
        self.custom_evaluator = custom_evaluator
        self.scoring_function = scoring_function
        self.metric_functions = metric_functions
        self.evaluator_definition_id = (
            evaluator_definition_id.strip()
            if isinstance(evaluator_definition_id, str)
            and evaluator_definition_id.strip()
            else None
        )
        self.effectuation = bool(effectuation)

    def _is_cloud_execution_mode(self) -> bool:
        return False

    def _setup_configuration_space(self, configuration_space) -> None:
        """Setup configuration space."""
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

        self.max_trials = kwargs.pop("max_trials", DEFAULT_MAX_TRIALS)
        kwargs["max_trials"] = self.max_trials

        # Experiment display name: decorator param > TRAIGENT_EXPERIMENT_NAME > self-describing default > func.__name__
        # _experiment_name holds ONLY the explicit decorator value (None when absent).
        # _default_experiment_name holds the precomputed self-describing default string
        # (func name + objectives + knobs) that decorators.py computed at decoration time.
        # The experiment_name getter resolves lazily so TRAIGENT_EXPERIMENT_NAME can be
        # changed AFTER decoration and still take effect.
        self._experiment_name: str | None = kwargs.pop("experiment_name", None)
        self._default_experiment_name: str | None = kwargs.pop(
            "_default_experiment_name", None
        )

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
        raw_cloud_fallback_policy = kwargs.pop("cloud_fallback_policy", sentinel)
        if (
            raw_cloud_fallback_policy is not sentinel
            and raw_cloud_fallback_policy is not None
        ):
            import warnings

            warnings.warn(
                "cloud_fallback_policy is deprecated and has no effect. "
                "Remote cloud execution has been removed.",
                DeprecationWarning,
                stacklevel=6,
            )
        self.cloud_fallback_policy = "auto"
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
            self._privacy_alias_enabled = True
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
        self.smart_pruning = self._store_optional_param(
            kwargs, sentinel, "smart_pruning", None
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

        # Safety constraints
        self.safety_constraints = self._store_optional_param(
            kwargs, sentinel, "safety_constraints", None
        )

        # TVL promotion gate for statistical best-config selection
        self.promotion_gate = self._store_optional_param(
            kwargs, sentinel, "promotion_gate", None
        )

        # Advisory strategy preset for task-local selection metadata.
        self.strategy_preset = self._store_optional_param(
            kwargs, sentinel, "strategy_preset", None
        )

        # Warm-start: seed a new run from a prior experiment's learned configs.
        self.warm_start_from = self._store_optional_param(
            kwargs, sentinel, "warm_start_from", None
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
            "smart_pruning",
            # Multi-agent configuration
            "agents",
            "agent_prefixes",
            "agent_measures",
            "global_measures",
            # Safety constraints
            "safety_constraints",
            "strategy_preset",
            "warm_start_from",
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
            config_id=getattr(self, "_config_id", None),
            best_config_source=getattr(self, "_best_config_source", "off"),
            best_config_strict=getattr(self, "_best_config_strict", False),
            best_config_cache_dir=getattr(self, "_best_config_cache_dir", None),
            best_config_cache_ttl_seconds=getattr(
                self, "_best_config_cache_ttl_seconds", 24 * 60 * 60
            ),
            best_config_stale_ok_ttl_seconds=getattr(
                self, "_best_config_stale_ok_ttl_seconds", None
            ),
            best_config_environment=getattr(self, "_best_config_environment", None),
            enable_auto_load_dev_logs=getattr(self, "_enable_auto_load_dev_logs", None),
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
    def objectives(self) -> list[str]:
        """Objective names derived from the active objective schema."""
        result: list[str] = schema_to_objective_names(self.objective_schema)
        return result

    def _validate_basic_inputs(self) -> None:
        """Validate basic inputs and raise appropriate exceptions."""
        # Validate function
        if not callable(self.func):
            raise TypeError("func must be callable") from None

        if self.max_trials is not None and self.max_trials <= 0:
            raise ValueError("max_trials must be a positive integer")

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
                validate_config_space(
                    self.configuration_space, default_config=self.default_config
                )
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

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Make the optimized function callable.

        The signature is typed generically so that a decorated function
        preserves its original parameter / return types under mypy/pyright.
        """
        wrapped_func = self._wrapped_func
        # If framework overrides are enabled, use them during function call
        if self.auto_override_frameworks and self.framework_targets:
            with override_context(self.framework_targets):
                return cast(_R, wrapped_func(*args, **kwargs))
        else:
            return cast(_R, wrapped_func(*args, **kwargs))

    def _trial_callable_for_config_space(
        self,
        effective_config_space: dict[str, Any],
    ) -> Callable[..., Any]:
        if not getattr(self, "effectuation", False):
            return cast(Callable[..., Any], self._wrapped_func)

        from traigent.effectuation import compile_effectuation

        application = compile_effectuation(
            self._wrapped_func,
            effective_config_space,
            enabled=True,
        )
        return cast(Callable[..., Any], application.wrapped_callable)

    # Decorator-only parameters that MUST NOT be accepted as call-time
    # ``.optimize(**algorithm_kwargs)`` keys. Before this guard they were
    # silently absorbed into ``BaseOptimizer.algorithm_config`` with zero
    # effect (issue #1683, Bug A: ``warm_start_from`` passed at call time was
    # structurally dead but raised no error — no-silent-legacy policy).
    #
    # Keys listed here are the ``_OPTIMIZE_DEFAULTS`` decorator options that
    # are (a) not explicit ``optimize()`` signature parameters and (b) never
    # consumed from ``algorithm_kwargs`` anywhere downstream. Keys that ARE
    # legitimately consumed downstream (``parallel_config``,
    # ``max_total_examples``, ``samples_include_pruned``, ``max_examples``,
    # ``plateau_window``, ``plateau_epsilon``, ``semantic_saturation``,
    # ``cache_policy``, ``cost_limit``, ``cost_approved``, ``metric_*``,
    # ``tie_breakers``, ``tvl_parameter_agents``, ``invocations_per_example``,
    # algorithm-specific options like ``seed``/``parameter_order``) must stay
    # OFF this list. General allowlist validation of every unknown kwarg is a
    # tracked follow-up (see issue #1683).
    _DECORATOR_ONLY_OPTIMIZE_PARAMS: frozenset[str] = frozenset(
        {
            "warm_start_from",
            "eval_dataset",
            "experiment_name",
            "default_config",
            "constraints",
            "safety_constraints",
            "injection_mode",
            "config_param",
            "agents",
            "agent_prefixes",
            "agent_measures",
            "global_measures",
            "auto_load_best",
            "load_from",
            "config_id",
            "best_config_source",
            "best_config_strict",
            "best_config_cache_dir",
            "best_config_cache_ttl_seconds",
            "best_config_stale_ok_ttl_seconds",
            "enable_auto_load_dev_logs",
            "smart_pruning",
            "mock_mode_config",
            "evaluator",
            "local_storage_path",
            "minimal_logging",
            "scoring_function",
            "metric_functions",
            "evaluation",
            "injection",
            "execution",
            "mock",
            "offline",
            "framework_targets",
            "auto_override_frameworks",
            "effectuation",
            "auto_detect_tvars",
            "auto_detect_tvars_mode",
            "auto_detect_tvars_min_confidence",
            "auto_detect_tvars_include",
            "auto_detect_tvars_exclude",
        }
    )

    def _prepare_algorithm_kwargs(
        self, algorithm_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge decorator overrides into algorithm kwargs and validate."""
        # Hard-fail on decorator-only params passed at call time (issue #1683
        # Bug A). Previously these were silently swallowed into the
        # optimizer's algorithm_config and had no effect.
        rejected = self._DECORATOR_ONLY_OPTIMIZE_PARAMS.intersection(algorithm_kwargs)
        if rejected:
            rejected_names = ", ".join(sorted(rejected))
            raise TypeError(
                f"{rejected_names} is a @traigent.optimize decorator argument "
                "and is not accepted by .optimize() at call time; move it to "
                f"the decorator: @traigent.optimize({sorted(rejected)[0]}=...). "
                "Previously this was silently ignored (issue #1683)."
            )

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

        # Normalize discovered dict payloads into the typed PromotionPolicy —
        # a raw dict would read as NON-strict in _is_strict_evidence_mode and
        # then fail OPEN on gate exceptions (FR-SDK-FAIL-CLOSED-PROMOTION-V1).
        if isinstance(promotion_policy, dict):
            from traigent.tvl.models import PromotionPolicy

            promotion_policy = PromotionPolicy.from_dict(promotion_policy)

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

    @staticmethod
    def _resolve_runtime_strategy_argument(
        *,
        strategy: str | None,
        strategy_params: Mapping[str, Any] | None,
        algorithm: str | None,
    ) -> tuple[str | None, str | None]:
        """Resolve runtime strategy into a preset name or deprecated algorithm alias."""
        if strategy is None:
            if strategy_params is not None:
                normalize_strategy_preset(None, strategy_params)
            return None, algorithm

        if is_strategy_preset_name(strategy) or strategy_params is not None:
            return strategy, algorithm

        if algorithm is not None and algorithm != strategy:
            raise TypeError(
                "Conflicting optimization selector: received both "
                f"'algorithm={algorithm}' and 'strategy={strategy}'. "
                "Use only 'algorithm'."
            )
        warnings.warn(
            "'strategy' as an optimizer selector is deprecated; "
            "use 'algorithm' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return None, strategy

    @staticmethod
    def _resolve_effective_strategy_preset(
        *,
        decorator_preset: NormalizedStrategyPreset | None,
        runtime_strategy: str | None,
        strategy_params: Mapping[str, Any] | None,
    ) -> NormalizedStrategyPreset | None:
        """Resolve runtime preset override against a decorator-level preset."""
        if runtime_strategy is None:
            return decorator_preset

        if decorator_preset is not None:
            raise ValueError("runtime strategy cannot override a decorator strategy.")

        runtime_preset = normalize_strategy_preset(runtime_strategy, strategy_params)
        return runtime_preset

    def _apply_runtime_strategy_preset(
        self,
        preset: NormalizedStrategyPreset | None,
        objectives: ObjectiveSchema | Sequence[str] | None,
    ) -> tuple[
        ObjectiveSchema | Sequence[str] | None,
        list[Callable[..., bool]],
        NormalizedStrategyPreset | None,
    ]:
        """Apply runtime preset objectives without adding search constraints."""
        original_constraints = list(self.constraints or [])
        original_preset = getattr(self, "strategy_preset", None)
        if preset is None:
            return objectives, original_constraints, original_preset
        if objectives is not None:
            raise ValueError(
                "strategy presets are mutually exclusive with explicit objectives. "
                "Use either strategy=... or objectives=..., not both."
            )
        self.strategy_preset = preset
        return list(preset.objectives), original_constraints, original_preset

    async def optimize(
        self,
        algorithm: str | None = None,
        max_trials: int | None = None,
        timeout: float | None = None,
        save_to: str | None = None,
        custom_evaluator: Callable[..., Any] | None = None,
        surrogate_evaluator: Callable[..., Any] | None = None,
        surrogate_evaluator_name: str | None = None,
        callbacks: list[Callable[..., Any]] | None = None,
        configuration_space: dict[str, Any] | None = None,
        objectives: ObjectiveSchema | Sequence[str] | None = None,
        tvl_spec: str | Path | None = None,
        tvl_environment: str | None = None,
        tvl: TVLOptions | dict[str, Any] | None = None,
        strategy: str | None = None,
        strategy_params: Mapping[str, Any] | None = None,
        progress_bar: bool | None = None,
        budget: ExecutionBudget | None = None,
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
            surrogate_evaluator: Optional cheap pre-screen scorer over captured
                outputs. When provided, overrides any decorator-level surrogate
                (same runtime-over-decorator precedence as ``custom_evaluator``).
            surrogate_evaluator_name: Optional explicit id for the surrogate
                descriptor's ``evaluator_id``. Overrides a decorator-level name;
                when omitted the id is derived from the callable.
            callbacks: List of callback objects for progress tracking
            configuration_space: Override configuration space for this optimization run.
                                Takes precedence over decorator configuration_space.
            objectives: Optional override objectives (list of names or ObjectiveSchema)
            tvl_spec: Optional TVL spec path to load at runtime.
            tvl_environment: Environment overlay to apply when loading the spec.
            tvl: Structured TVL options (dict or TVLOptions) for runtime overrides.
            strategy: Optional advisory strategy preset name. Non-preset values retain
                the deprecated optimizer-alias behavior.
            strategy_params: Typed parameters for the selected strategy preset.
            progress_bar: Controls the live progress bar during optimization.
                ``True`` forces a progress bar even in non-interactive mode,
                ``False`` suppresses it, ``None`` (default) auto-enables in
                interactive terminals (``sys.stdin.isatty()``).
            budget: Optional **experimental** shared cumulative ``ExecutionBudget``
                (issue #1980). Pass the *same* instance to several ``optimize()``
                calls (e.g. baseline -> search -> holdout) to cap the *total*
                cost / examples / wall-clock across all of them; per-operation
                limits still apply and can never exceed the shared cap. Examples
                and the deadline are hard limits; the monetary cap is a lower bound
                when cost is unobservable (see ``ExecutionBudget`` docs). ``None``
                (default) leaves behavior unchanged.
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

        runtime_strategy_name, algorithm = self._resolve_runtime_strategy_argument(
            strategy=strategy,
            strategy_params=strategy_params,
            algorithm=algorithm,
        )
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

        original_schema = self.objective_schema
        strategy_original_constraints: list[Callable[..., bool]] | None = None
        strategy_original_preset: NormalizedStrategyPreset | None = None
        decorator_preset = getattr(self, "strategy_preset", None)
        effective_preset = self._resolve_effective_strategy_preset(
            decorator_preset=decorator_preset,
            runtime_strategy=runtime_strategy_name,
            strategy_params=strategy_params,
        )
        if decorator_preset is not None and objectives is not None:
            raise ValueError(
                "strategy presets are mutually exclusive with explicit objectives. "
                "Use either strategy=... or objectives=..., not both."
            )
        if decorator_preset is None:
            (
                objectives,
                strategy_original_constraints,
                strategy_original_preset,
            ) = self._apply_runtime_strategy_preset(effective_preset, objectives)

        runtime_objective_input = (
            objectives if objectives is not None else legacy_objectives
        )
        try:
            runtime_schema = normalize_objectives(runtime_objective_input)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

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
                surrogate_evaluator=surrogate_evaluator,
                surrogate_evaluator_name=surrogate_evaluator_name,
                callbacks=callbacks,
                configuration_space=configuration_space,
                algorithm_kwargs=algorithm_kwargs,
                execution_budget=budget,
            )
        finally:
            if runtime_schema is not None:
                self.objective_schema = original_schema
            if strategy_original_constraints is not None:
                self.constraints = strategy_original_constraints
                self.strategy_preset = strategy_original_preset
            self._restore_tvl_state(tvl_state)

        return result

    def optimize_sync(
        self,
        algorithm: str | None = None,
        max_trials: int | None = None,
        timeout: float | None = None,
        save_to: str | None = None,
        custom_evaluator: Callable[..., Any] | None = None,
        surrogate_evaluator: Callable[..., Any] | None = None,
        surrogate_evaluator_name: str | None = None,
        callbacks: list[Callable[..., Any]] | None = None,
        configuration_space: dict[str, Any] | None = None,
        objectives: ObjectiveSchema | Sequence[str] | None = None,
        tvl_spec: str | Path | None = None,
        tvl_environment: str | None = None,
        tvl: TVLOptions | dict[str, Any] | None = None,
        strategy: str | None = None,
        strategy_params: Mapping[str, Any] | None = None,
        progress_bar: bool | None = None,
        budget: ExecutionBudget | None = None,
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
            strategy: Optional advisory strategy preset name.
            strategy_params: Typed parameters for the selected strategy preset.
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
            surrogate_evaluator=surrogate_evaluator,
            surrogate_evaluator_name=surrogate_evaluator_name,
            callbacks=callbacks,
            configuration_space=configuration_space,
            objectives=objectives,
            tvl_spec=tvl_spec,
            tvl_environment=tvl_environment,
            tvl=tvl,
            strategy=strategy,
            strategy_params=strategy_params,
            progress_bar=progress_bar,
            budget=budget,
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
            "semantic_saturation",
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
        surrogate_evaluator: Callable[..., Any] | None = None,
        surrogate_evaluator_name: str | None = None,
        force_auto_discover_tvars: bool | None = None,
    ) -> BaseEvaluator:
        """Create the appropriate evaluator. Delegates to optimization_pipeline."""
        evaluator, _auxiliary_resource = create_effective_evaluator(
            timeout=timeout,
            custom_evaluator=custom_evaluator,
            effective_batch_size=effective_batch_size,
            effective_thread_workers=effective_thread_workers,
            effective_privacy_enabled=effective_privacy_enabled,
            objectives=self.objectives,
            execution_mode=self.execution_mode,
            mock_mode_config=self.mock_mode_config,
            metric_functions=self.metric_functions,
            scoring_function=self.scoring_function,
            decorator_custom_evaluator=self.custom_evaluator,
            # Surrogate (pre-screen) scorer: runtime optimize()-arg overrides the
            # decorator value (same precedence as custom_evaluator); stashed on the
            # evaluator so the trial lifecycle can score captured outputs.
            surrogate_evaluator=surrogate_evaluator,
            decorator_surrogate_evaluator=self._surrogate_evaluator,
            surrogate_evaluator_name=surrogate_evaluator_name,
            decorator_surrogate_evaluator_name=self._surrogate_evaluator_name,
            **self._hybrid_api_evaluator_kwargs(
                force_auto_discover_tvars=force_auto_discover_tvars
            ),
        )
        return cast(BaseEvaluator, evaluator)

    def _build_artifact_fingerprint_payload(
        self,
        *,
        dataset: Dataset,
        configuration_space: dict[str, Any],
        custom_evaluator: Callable[..., Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """Build privacy-safe artifact fingerprints for session creation."""
        external: Any = None
        if self.execution_mode == ExecutionMode.HYBRID_API.value:
            external = {
                "kind": "hybrid_api",
                "endpoint": getattr(self, "hybrid_api_endpoint", None),
            }
        elif getattr(self, "external_service_evaluator", None) is not None:
            external = getattr(self, "external_service_evaluator", None)

        return cast(
            dict[str, dict[str, Any]],
            build_artifact_fingerprints(
                dataset=dataset,
                func=self.func,
                custom_evaluator=custom_evaluator or self.custom_evaluator,
                scoring_function=self.scoring_function,
                metric_functions=self.metric_functions,
                external=external,
                configuration_space=configuration_space,
            ),
        )

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
        artifact_fingerprint_payload: dict[str, dict[str, Any]],
        requested_algorithm: str | None = None,
        execution_budget: ExecutionBudget | None = None,
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
            safety_constraints=getattr(self, "safety_constraints", None),
            warm_start_from=getattr(self, "warm_start_from", None),
        )
        orchestrator_kwargs["requested_algorithm"] = requested_algorithm

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
            strategy_preset=getattr(self, "strategy_preset", None),
            smart_pruning=getattr(self, "smart_pruning", None),
            **orchestrator_kwargs,
        )

        orchestrator.samples_include_pruned = samples_include_pruned_value
        orchestrator.artifact_fingerprints = artifact_fingerprint_payload.get(
            "artifact_fingerprints"
        )
        orchestrator.fingerprint_meta = artifact_fingerprint_payload.get(
            "fingerprint_meta"
        )
        orchestrator.evaluator_definition_id = self.evaluator_definition_id
        # RFC 0001 §3.4: forward the user-attached knob resolver so the
        # public optimize() path resolves Fixed/CVAR bindings in-trial.
        # Attribute seam (like promotion_gate): set
        # ``wrapped.knob_resolver = KnobResolver(...)`` before optimizing;
        # absent ⇒ byte-identical legacy behavior.
        knob_resolver = getattr(self, "knob_resolver", None)
        if knob_resolver is not None:
            orchestrator.knob_resolver = knob_resolver
        # Cumulative ExecutionBudget attribute seam (issue #1980), same pattern as
        # knob_resolver: set AFTER construction (never an __init__ param, which
        # would run _configure_stop_conditions/_setup_cost_enforcer before the
        # budget could act). Absent ⇒ byte-identical legacy behavior — every
        # orchestrator seam reads it via getattr(self, "execution_budget", None).
        if execution_budget is not None:
            orchestrator.execution_budget = execution_budget
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

        trial_func = self._trial_callable_for_config_space(effective_config_space)

        # Reset the unpriced-at-runtime model registry so this run starts clean.
        # The runtime cost path records models that priced to $0 despite non-zero
        # tokens (e.g. a model hard-coded in the optimized function body that no
        # pricing table covers, invisible to the config-space preflight); we drain
        # it onto the result below to surface a user-visible warning (#1407).
        from traigent.utils.cost_calculator import reset_unpriced_runtime_models

        reset_unpriced_runtime_models()

        # Set state to OPTIMIZING before starting
        self._state = OptimizationState.OPTIMIZING

        try:
            with ConfigurationSpaceContext(effective_config_space):
                if self.auto_override_frameworks and self.framework_targets:
                    with override_context(self.framework_targets):
                        result = await orchestrator.optimize(
                            func=trial_func,
                            dataset=dataset,
                            function_name=self.experiment_name,
                        )
                else:
                    result = await orchestrator.optimize(
                        func=trial_func,
                        dataset=dataset,
                        function_name=self.experiment_name,
                    )

            # Store results
            self._optimization_results = result
            self._csm.append_optimization_result(result)

            # Update current config to best found
            if result.best_config:
                self.apply_best_config(result)

            # Set state to OPTIMIZED on success
            self._state = OptimizationState.OPTIMIZED

        except Exception:
            # Set state to ERROR on failure
            self._state = OptimizationState.ERROR
            raise
        # Surface any models that priced to $0 at runtime despite non-zero tokens
        # (#1407). Strict accounting already raised mid-run via
        # ``cost_from_tokens(strict=True)`` (fail-closed), so anything collected
        # here is the non-strict warn-and-continue case — make it visible on the
        # result instead of leaving only a buried log.
        self._attach_unpriced_model_warning(result)

        # Surface the shared cumulative ExecutionBudget's consumed/remaining state
        # onto the result via metadata (issue #1980). No-op when no budget was
        # attached -> result shape unchanged.
        self._attach_execution_budget_snapshot(result, orchestrator)

        # Save results if requested
        if save_to:
            self.save_optimization_results(save_to)

        logger.info(
            f"Optimization completed: {len(result.trials)} trials, "
            f"best score: {'N/A' if result.best_score is None else f'{result.best_score:.4f}'}"
        )

        # Show upgrade hints after optimization completion (Edge Analytics mode only)
        if self.traigent_config.is_local_mode():  # type: ignore[has-type]
            try:
                show_upgrade_hint(
                    "session_complete",
                    trial_count=len(result.trials),
                    best_score=result.best_score,
                )
            except Exception as e:
                logger.debug(f"Failed to show upgrade hint: {e}")

        return result  # type: ignore[no-any-return]

    def _attach_unpriced_model_warning(self, result: OptimizationResult) -> None:
        """Attach a user-visible warning for models that priced to $0 at runtime.

        Closes the runtime half of the cost-coverage gap (#1407): the pre-run
        preflight only inspects config-space model ids, so a model HARD-CODED in
        the optimized function body is invisible to it. When such a model is
        unpriced, the non-strict runtime cost path records $0 with only a buried
        log. Here we lift the collected ids onto the result (``warnings`` /
        ``warning_codes`` and ``metadata``) using the SAME remediation surface as
        the preflight, so the user can opt into custom pricing or acknowledge the
        gap. Strict accounting never reaches this surface — it fails closed
        mid-run via ``cost_from_tokens(strict=True)``.
        """
        try:
            from traigent.utils.cost_calculator import (
                get_unpriced_runtime_models,
                get_unpriced_runtime_occurrences,
            )

            unpriced = get_unpriced_runtime_models()
            occurrences = get_unpriced_runtime_occurrences()
        except Exception:  # pragma: no cover - defensive; never break a run
            return

        if not unpriced:
            return

        message = _format_unpriced_model_warning(unpriced, occurrences)
        if message not in result.warnings:
            result.warnings.append(message)
        if "UNPRICED_MODEL_RUNTIME" not in result.warning_codes:
            result.warning_codes.append("UNPRICED_MODEL_RUNTIME")
        if isinstance(result.metadata, dict):
            result.metadata.setdefault("unpriced_models_runtime", list(unpriced))
            # Quantified detail (#1597): how many calls per model recorded $0
            # because pricing was UNKNOWN, not because usage was free — so any
            # consumer of result.total_cost knows it is a lower bound, not
            # verified spend, whenever this is non-empty.
            result.metadata.setdefault(
                "unpriced_models_runtime_call_counts", dict(occurrences)
            )

        warnings.warn(message, UserWarning, stacklevel=2)
        logger.warning(
            "Models priced at $0 at runtime (real spend under-reported): %s. %s",
            ", ".join(unpriced),
            message,
        )

    def _attach_execution_budget_snapshot(
        self,
        result: OptimizationResult,
        orchestrator: OptimizationOrchestrator,
    ) -> None:
        """Surface a shared cumulative ExecutionBudget's state onto the result.

        Additive (issue #1980), beside :meth:`_attach_unpriced_model_warning`:
        writes ``result.metadata["execution_budget"]`` (consumed / remaining per
        dimension, ``untracked_trials``, ``cost_tracking``) and appends the
        ``"EXECUTION_BUDGET_UNTRACKED_COST"`` warning code when cost tracking is
        incomplete — so a caller reading the monetary cap knows it was a lower
        bound, not a hard guarantee. No-op when no budget was attached, keeping the
        result shape byte-identical for the absent path.
        """
        budget: ExecutionBudget | None = getattr(orchestrator, "execution_budget", None)
        if budget is None:
            return

        # Fold the enforcer's unknown-cost mode and any unpriced-runtime models
        # into the budget's honesty flag, so cost_tracking reflects every silent
        # $0 path, not only per-trial ``cost is None`` debits.
        try:
            cost_enforcer = getattr(orchestrator, "cost_enforcer", None)
            if (
                cost_enforcer is not None
                and cost_enforcer.get_status().unknown_cost_mode
            ):
                budget.mark_cost_untracked()
        except Exception:  # pragma: no cover - defensive; never break finalize
            logger.debug("ExecutionBudget unknown-cost fold-in failed", exc_info=True)
        if "UNPRICED_MODEL_RUNTIME" in result.warning_codes:
            budget.mark_cost_untracked()

        snapshot = budget.snapshot()
        if isinstance(result.metadata, dict):
            result.metadata["execution_budget"] = snapshot.as_dict()

        if snapshot.cost_tracking != "complete":
            if "EXECUTION_BUDGET_UNTRACKED_COST" not in result.warning_codes:
                result.warning_codes.append("EXECUTION_BUDGET_UNTRACKED_COST")
            message = (
                "ExecutionBudget: cost tracking was "
                f"{snapshot.cost_tracking!r} ({snapshot.untracked_trials} of "
                f"{snapshot.trials} trial(s) had unobservable cost). The consumed "
                "cost is a LOWER BOUND and the monetary cap is not a hard "
                "guarantee on this run; the examples and deadline caps stayed hard."
            )
            if message not in result.warnings:
                result.warnings.append(message)

    async def _try_cloud_execution(
        self,
        dataset: Dataset,
        max_trials: int | None,
        timeout: float | None,
        effective_config_space: dict[str, Any],
        algorithm_kwargs: dict[str, Any],
        traigent_config: TraigentConfig,
        artifact_fingerprint_payload: dict[str, dict[str, Any]],
        effective_privacy_enabled: bool,
        effective_parallel_trials: int | None,
        samples_include_pruned_value: bool,
        callbacks: list[Callable[..., Any]] | None,
        save_to: str | None,
        requested_algorithm: str | None = None,
        execution_budget: ExecutionBudget | None = None,
    ) -> OptimizationResult | None:
        """Try cloud-brain execution, returning None for allowed local fallback."""

        policy = policy_from_config(traigent_config)
        if (
            policy is None
            or getattr(self, "external_service_evaluator", None) is not None
            or self.execution_mode == ExecutionMode.HYBRID_API.value
            or is_offline_requested(policy)
        ):
            if policy_is_cloud_required(policy) and is_offline_requested(policy):
                requested = getattr(policy, "algorithm", None) or "smart"
                raise ConfigurationError(
                    f"Smart optimization ('{requested}') requires the Traigent "
                    "managed cloud service, but offline mode is set "
                    "(offline=True or TRAIGENT_OFFLINE/TRAIGENT_OFFLINE_MODE). "
                    "The local SDK runs only 'grid' and 'random'. Either drop "
                    "offline mode and connect to a Traigent backend that "
                    "provides smart optimization, or use algorithm='grid' / "
                    "algorithm='random' to run locally."
                )
            return None

        if not (policy_is_cloud_brain(policy) or policy_is_cloud_required(policy)):
            return None

        if max_trials is not None and max_trials <= 0:
            return None

        backend_optimization_strategy = None
        if policy_is_cloud_required(policy):
            backend_optimization_strategy = backend_optimization_strategy_for_algorithm(
                policy.algorithm
            )
            if backend_optimization_strategy is None:
                raise ConfigurationError(
                    unsupported_backend_smart_algorithm_message(policy.algorithm)
                )

        try:
            from traigent.cloud.client import TraigentCloudClient
            from traigent.cloud.remote_guidance import (
                TraigentCloudRemoteGuidanceAdapter,
            )
            from traigent.config.backend_config import BackendConfig
            from traigent.optimizers.interactive_optimizer import InteractiveOptimizer

            cloud_client = TraigentCloudClient(
                api_key=BackendConfig.get_api_key(),
                base_url=BackendConfig.get_backend_url(),
                enable_fallback=False,
                no_egress=backend_egress_disabled(traigent_config),
            )
            remote_service = TraigentCloudRemoteGuidanceAdapter(cloud_client)
            optimizer_kwargs = dict(algorithm_kwargs)
            optimizer_kwargs.pop("cost_approved", None)
            optimizer_kwargs.pop("invocations_per_example", None)
            if max_trials:
                optimizer_kwargs["max_trials"] = max_trials

            optimizer = InteractiveOptimizer(
                effective_config_space,
                self.objectives,
                remote_service=remote_service,
                dataset_metadata={
                    "size": len(dataset),
                    "name": getattr(dataset, "name", "dataset"),
                },
                optimization_strategy=backend_optimization_strategy
                or {
                    "algorithm": policy.algorithm,
                    "source": SOURCE_CLOUD_BRAIN,
                },
                artifact_fingerprints=artifact_fingerprint_payload.get(
                    "artifact_fingerprints"
                ),
                fingerprint_meta=artifact_fingerprint_payload.get("fingerprint_meta"),
                evaluator_definition_id=self.evaluator_definition_id,
                context=traigent_config,
                **optimizer_kwargs,
            )
            evaluator = self._create_effective_evaluator(
                timeout=timeout,
                custom_evaluator=None,
                effective_batch_size=None,
                effective_thread_workers=None,
                effective_privacy_enabled=effective_privacy_enabled,
            )
            orchestrator = self._build_optimization_orchestrator(
                optimizer=optimizer,
                evaluator=evaluator,
                max_trials=max_trials,
                max_total_examples_value=getattr(self, "max_total_examples", None),
                timeout=timeout,
                callbacks=callbacks,
                traigent_config=traigent_config,
                effective_parallel_trials=effective_parallel_trials,
                samples_include_pruned_value=samples_include_pruned_value,
                algorithm_kwargs=algorithm_kwargs,
                artifact_fingerprint_payload=artifact_fingerprint_payload,
                requested_algorithm=requested_algorithm,
                execution_budget=execution_budget,
            )
            orchestrator._cloud_guidance_client = cloud_client
            return await self._run_and_finalize_optimization(
                orchestrator=orchestrator,
                dataset=dataset,
                effective_config_space=effective_config_space,
                save_to=save_to,
            )
        except (AuthenticationError, ConfigurationError, ValidationError):
            raise
        except Exception as e:
            if policy_requires_cloud(policy) or not policy_allows_cloud_fallback(
                policy
            ):
                raise
            if not exception_is_connectivity(e):
                raise
            reason = str(getattr(e, "reason", None) or e)
            mark_local_fallback(traigent_config, reason)
            if not (
                isinstance(e, CloudBrainUnavailableError)
                and e.stage == "session-create"
            ):
                logger.warning(
                    "traigent.cloud_brain_fallback source=%s fallback_reason=%s "
                    "stage=next-trial",
                    SOURCE_LOCAL_FALLBACK,
                    reason,
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
        mock_config = self.mock_mode_config
        if not isinstance(mock_config, Mapping):
            return algorithm

        inert_keys = sorted(
            key for key in ("optimizer", "sampler", "random_seed") if key in mock_config
        )
        if inert_keys:
            logger.warning(
                "mock_mode_config keys %s are inert post-F5 and no longer select "
                "optimizers or seed runs; pass algorithm/random_seed via "
                "decorated.optimize(algorithm=..., random_seed=...) instead.",
                ", ".join(inert_keys),
            )
        return algorithm

    def _preflight_model_cost_coverage(
        self,
        effective_config_space: Mapping[str, Any],
        *,
        cost_approved: bool = False,
    ) -> None:
        """Warn or fail before trials when a real run includes unpriced models."""
        if is_mock_llm():
            return

        model_ids = _extract_config_space_model_ids(effective_config_space)
        if not model_ids:
            model_ids = _dedupe_model_ids(
                _extract_config_model_ids(getattr(self, "_current_config", {}))
                + _extract_config_model_ids(getattr(self, "default_config", {}))
            )
        if not model_ids:
            return

        missing = find_models_missing_price_coverage(model_ids)
        if not missing:
            return

        formatted = _format_model_id_list(missing)
        if is_strict_cost_accounting():
            raise UnknownModelError(
                "Cost coverage preflight failed before any optimization trial "
                f"started: pricing is unavailable for {formatted}. "
                "Set TRAIGENT_CUSTOM_MODEL_PRICING_JSON or "
                "TRAIGENT_CUSTOM_MODEL_PRICING_FILE with explicit per-token "
                "pricing, or contact Traigent to add coverage."
            )

        message = _format_unpriced_model_warning(missing)

        if is_cost_preapproved(cost_approved):
            warnings.warn(message, UserWarning, stacklevel=3)
            logger.info(
                "Proceeding with unpriced models because cost execution was "
                "pre-approved: %s",
                ", ".join(missing),
            )
            return

        if not sys.stdin.isatty():
            raise UnknownModelError(
                "Cost coverage preflight failed before any optimization trial "
                "started: pricing is unavailable for "
                f"{formatted}. Cannot prompt for confirmation in a "
                "non-interactive shell. Set TRAIGENT_CUSTOM_MODEL_PRICING_JSON "
                "or TRAIGENT_CUSTOM_MODEL_PRICING_FILE with explicit per-token "
                "pricing, set TRAIGENT_COST_APPROVED=true to acknowledge this "
                "cost concern, or run interactively."
            )

        plural = "model has" if len(missing) == 1 else "models have"
        print(
            f"""
================================================================================
Traigent Unpriced Model Warning
================================================================================

The following {len(missing)} {plural} NO known pricing:
  {formatted}

Their optimization results will report $0 cost for those calls, but your LLM
provider may still bill you.

Remediation:
  - Set TRAIGENT_CUSTOM_MODEL_PRICING_JSON with explicit per-token pricing
  - Contact Traigent to add model pricing coverage
  - Set TRAIGENT_COST_APPROVED=true to skip this prompt after acknowledging
    this cost concern

================================================================================
""",
            file=sys.stderr,
        )

        try:
            print(
                "Proceed despite unpriced models? [y/N]: ",
                end="",
                file=sys.stderr,
                flush=True,
            )
            choice = input().strip().lower()
        except (EOFError, KeyboardInterrupt) as exc:
            raise UnknownModelError(
                "Cost coverage preflight failed before any optimization trial "
                "started: user confirmation was interrupted for unpriced "
                f"models {formatted}."
            ) from exc

        if choice not in {"y", "yes"}:
            raise UnknownModelError(
                "Cost coverage preflight failed before any optimization trial "
                "started: user declined to proceed with unpriced models "
                f"{formatted}."
            )

        warnings.warn(message, UserWarning, stacklevel=3)
        logger.info(
            "Proceeding with unpriced models after interactive user confirmation: %s",
            ", ".join(missing),
        )

    def _policy_for_runtime_algorithm(
        self,
        stored_policy: ResolvedExecutionPolicy | None,
        runtime_algorithm: str | None,
    ) -> ResolvedExecutionPolicy | None:
        """Re-derive cloud-vs-local routing from the resolved runtime algorithm.

        The policy stored on the instance is resolved at construction time
        *without* knowing the algorithm that ``optimize(...)`` is finally called
        with. A runtime override such as ``optimize(algorithm="grid")`` (or
        ``optimize(algorithm="bayesian")``) must route from that resolved runtime
        algorithm, exactly like the decorator path
        (``decorators._resolve_execution_policy_from_options``). Otherwise a stale
        ``auto`` policy (intent ``CLOUD_BRAIN``) keeps the runtime algorithm on
        the backend-guided/typed cloud session:

        * a local override (``grid``/``random``) samples one config per trial on
          the cloud session instead of running the exhaustive local
          ``GridSearchOptimizer`` (#1421);
        * a smart override (``bayesian``/``tpe``/``cmaes``/``nsga2``/``optuna*``)
          silently reuses the laxer ``CLOUD_BRAIN`` policy, which either runs the
          *auto* strategy under a different name or falls back to local — instead
          of the fallback-forbidden ``CLOUD_REQUIRED`` semantics a smart algorithm
          demands (#1681).

        The override — not the construction-time default — decides cloud-vs-local
        so that:

        * a local override flips a cloud-capable policy to ``LOCAL_ONLY``;
        * a smart override yields ``CLOUD_REQUIRED`` (no silent local fallback);
        * ``auto`` (or an unchanged algorithm) keeps its stored cloud routing;
        * ``offline`` / ``require_cloud`` / ``HYBRID_API`` intent is preserved;
        * unknown algorithm names still fail through the existing optimizer
          lookup (``get_optimizer``) rather than being rejected early here,
          keeping error semantics unchanged.
        """

        if stored_policy is None:
            return None

        # Already local (offline or explicit grid/random decorator): nothing to
        # flip — and never silently relax an offline/no-egress guarantee. A smart
        # runtime override on an offline wrapper still fails closed downstream via
        # get_optimizer(), which rejects smart algorithms for local execution.
        if stored_policy.intent is ExecutionIntent.LOCAL_ONLY:
            return stored_policy

        normalized_runtime = normalize_algorithm_name(runtime_algorithm)

        # No effective override: the runtime algorithm already matches the policy
        # the wrapper resolved at construction. Keep the stored object (identity)
        # so genuinely cloud algorithms keep their exact cloud routing untouched.
        if normalized_runtime == stored_policy.algorithm:
            return stored_policy

        legacy_mode = (
            ExecutionMode.HYBRID_API
            if stored_policy.legacy_execution_mode is ExecutionMode.HYBRID_API
            else None
        )
        try:
            return resolve_execution_policy(
                algorithm=normalized_runtime,
                offline=stored_policy.offline,
                require_cloud=stored_policy.require_cloud,
                execution_mode=legacy_mode,
                source_hint="optimize_runtime",
            )
        except ValueError:
            # Unknown/unvalidated runtime algorithm name: preserve the stored
            # policy so the downstream optimizer lookup (get_optimizer) raises the
            # canonical "Unknown optimizer" error instead of a policy-resolution
            # ValueError leaking from here (error semantics unchanged, #1421).
            return stored_policy

    async def _execute_optimization(
        self,
        *,
        algorithm: str | None,
        max_trials: int | None,
        timeout: float | None,
        save_to: str | None,
        custom_evaluator: Callable[..., Any] | None,
        surrogate_evaluator: Callable[..., Any] | None = None,
        surrogate_evaluator_name: str | None = None,
        callbacks: list[Callable[..., Any]] | None,
        configuration_space: dict[str, Any] | None,
        algorithm_kwargs: dict[str, Any],
        execution_budget: ExecutionBudget | None = None,
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
                surrogate_evaluator=surrogate_evaluator,
                surrogate_evaluator_name=surrogate_evaluator_name,
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
        requested_max_trials = max_trials
        algorithm, max_trials, effective_config_space = resolve_execution_parameters(
            algorithm,
            max_trials,
            configuration_space,
            fallback_config_space=self.configuration_space,
            fallback_algorithm=cast(str, getattr(self, "algorithm", "grid")),
            fallback_max_trials=getattr(self, "max_trials", None),
            default_config=self.default_config,
        )
        used_implicit_default_max_trials = (
            requested_max_trials is None
            and getattr(self, "_max_trials_uses_sdk_default", False)
            and max_trials == DEFAULT_MAX_TRIALS
        )
        if (
            used_implicit_default_max_trials
            and not self._max_trials_default_notice_logged
        ):
            logger.info(
                "Using default max_trials=%d; pass max_trials=... to change.",
                max_trials,
            )
            self._max_trials_default_notice_logged = True

        stored_policy = getattr(self, "execution_policy", None)
        if not isinstance(stored_policy, ResolvedExecutionPolicy):
            stored_policy = None
        # Re-derive cloud-vs-local routing from the *resolved runtime* algorithm
        # so a runtime override such as optimize(algorithm="grid") stays local
        # and exhaustive even when the construction-time policy was a
        # cloud-capable ``auto`` (issue #1421).
        execution_policy = self._policy_for_runtime_algorithm(stored_policy, algorithm)
        external_evaluator = (
            getattr(self, "external_service_evaluator", None) is not None
            or self.execution_mode == ExecutionMode.HYBRID_API.value
        )
        result_source = initial_result_source(
            execution_policy,
            external_evaluator=external_evaluator,
        )
        no_egress = result_source == SOURCE_OFFLINE

        # Phase 2: Create TraigentConfig and check CI approval
        traigent_config = create_traigent_config(
            execution_mode=self.execution_mode,
            local_storage_path=self.local_storage_path,
            minimal_logging=self.minimal_logging,
            privacy_enabled=(
                bool(getattr(self, "privacy_enabled", False))
                and not bool(getattr(self, "_privacy_alias_enabled", False))
            ),
            execution_policy=execution_policy,
            no_egress=no_egress,
            result_source=result_source,
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

        # Phase 3.5: cost coverage preflight before any cloud or local trial dispatch.
        self._preflight_model_cost_coverage(
            effective_config_space,
            cost_approved=normalize_cost_approved(
                algorithm_kwargs.get("cost_approved", False)
            ),
        )

        artifact_fingerprint_payload = self._build_artifact_fingerprint_payload(
            dataset=dataset,
            configuration_space=effective_config_space,
            custom_evaluator=custom_evaluator,
        )

        # Phase 4: Resolve parallel configuration
        effective_parallel_trials, effective_batch_size, effective_thread_workers = (
            resolve_effective_parallel_config(
                algorithm_kwargs,
                decorator_parallel_config=self.parallel_config,
                config_space_size=self._estimate_search_space_size(),
                is_async_func=is_coroutine_callable(self.func),
            )
        )

        # Phase 5: Try cloud execution if applicable
        cloud_result = await self._try_cloud_execution(
            dataset,
            max_trials,
            timeout,
            effective_config_space,
            algorithm_kwargs,
            traigent_config,
            artifact_fingerprint_payload,
            effective_privacy_enabled,
            effective_parallel_trials,
            samples_include_pruned_value,
            callbacks,
            save_to,
            requested_algorithm=algorithm,
            execution_budget=execution_budget,
        )
        if cloud_result is not None:
            return cloud_result

        if (
            traigent_config.result_source
            in {SOURCE_LOCAL_FALLBACK, SOURCE_OFFLINE, SOURCE_EXPLICIT_LOCAL}
            and algorithm == "auto"
        ):
            algorithm = "random"

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
                surrogate_evaluator=surrogate_evaluator,
                surrogate_evaluator_name=surrogate_evaluator_name,
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
        traigent_config.privacy_enabled = effective_privacy_enabled and not bool(
            getattr(self, "_privacy_alias_enabled", False)
        )

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
            artifact_fingerprint_payload=artifact_fingerprint_payload,
            requested_algorithm=algorithm,
            execution_budget=execution_budget,
        )

        # Phase 9: Run optimization and finalize
        try:
            return await self._run_and_finalize_optimization(
                orchestrator=orchestrator,
                dataset=dataset,
                effective_config_space=effective_config_space,
                save_to=save_to,
            )
        except OptimizationError:
            raise
        except Exception as e:
            from traigent.knobs import ResolutionError

            logger.error(f"Optimization failed: {e}")
            if isinstance(e, ResolutionError):
                # RFC 0001 §3.4: the typed fail-closed governance rejection
                # IS the public contract — never dilute it into a generic
                # OptimizationError.
                raise
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
                self.apply_best_config(result)

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
                function_name=self.experiment_name,
                dataset=dataset,
                configuration_space=effective_config_space,
                objectives=self.objectives,
                max_trials=max_trials if max_trials is not None else DEFAULT_MAX_TRIALS,
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

    def optimize_with_guidance(
        self,
        provider: Any,
        *,
        plan_kind: Any = "benchmark_guide",
        rewrite_llm: Any = None,
        prompt_rewrite: Any = None,
        grow_dataset: Any = None,
        prompt_param: str | None = None,
        weak_examples: Sequence[tuple[Any, Any, Any]] = (),
        **optimize_kwargs: Any,
    ) -> Any:
        """Run guided generation: optimize, fetch an opaque backend GuidancePlan,
        generate locally with the user's own LLM, and re-optimize across rounds.

        ``provider`` supplies the GuidancePlan (a ``GuidancePlanProvider``);
        generation runs on ``rewrite_llm`` (a callable ``fn(prompt) -> str`` or an
        already-constructed client). Content never leaves the client. Returns the
        best ``OptimizationResult`` across rounds.
        """
        from traigent.generation import (
            DatasetGrowthOptions,
            ExampleSynthesizer,
            GuidanceLoop,
            PromptRewriteOptions,
            PromptRewriter,
            resolve_rewrite_llm,
        )
        from traigent.generation.models import PlanKind
        from traigent.utils.example_id import (
            compute_dataset_hash,
            generate_stable_example_id,
        )

        kind = plan_kind if isinstance(plan_kind, PlanKind) else PlanKind(plan_kind)
        llm = resolve_rewrite_llm(rewrite_llm)

        def _coerce(spec: Any, cls: type) -> Any:
            if spec is None:
                return cls()
            if isinstance(spec, cls):
                return spec
            return cls(**spec)

        prompt_opts = _coerce(
            (
                prompt_rewrite
                if prompt_rewrite is not None
                else self.prompt_rewrite_options
            ),
            PromptRewriteOptions,
        )
        growth_opts = _coerce(
            grow_dataset if grow_dataset is not None else self.grow_dataset_options,
            DatasetGrowthOptions,
        )

        config_space = dict(self.configuration_space or {})
        dataset = self._load_dataset()

        is_rewrite = kind is PlanKind.PROMPT_REWRITE
        rewriter = PromptRewriter(llm, prompt_opts) if is_rewrite else None
        synthesizer = ExampleSynthesizer(llm, growth_opts) if not is_rewrite else None

        # Best-effort stable-id -> example map for resolving plan seeds locally.
        ds_hash = compute_dataset_hash(getattr(dataset, "name", "dataset"))
        id_to_example = {
            generate_stable_example_id(ds_hash, i): ex
            for i, ex in enumerate(getattr(dataset, "examples", []))
        }

        def _seed_resolver(seed_ref: str) -> Any:
            return id_to_example.get(seed_ref)

        def _optimize_round(cs: dict[str, Any], ds: Any) -> Any:
            self.set_eval_dataset_override(ds)
            return self.optimize_sync(configuration_space=cs, **optimize_kwargs)

        loop = GuidanceLoop(
            provider=provider,
            rewriter=rewriter,
            synthesizer=synthesizer,
            prompt_options=prompt_opts,
            growth_options=growth_opts,
        )
        try:
            outcome = loop.run(
                optimize_round=_optimize_round,
                config_space=config_space,
                dataset=dataset,
                plan_kind=kind,
                prompt_param=prompt_param,
                seed_resolver=None if is_rewrite else _seed_resolver,
                weak_examples=weak_examples,
            )
        finally:
            self.set_eval_dataset_override(None)
        return outcome.best_result

    def train_skill(
        self,
        *,
        document: str,
        optimizer_llm: Any = None,
        skill_train: Any = None,
        doc_param: str | None = None,
        selection_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        **fixed_config: Any,
    ) -> Any:
        """Train a text skill document behind a strict selection gate.

        Privacy semantics, precisely: no Traigent-managed optimizer is invoked —
        optimizer calls go only to the caller-supplied ``optimizer_llm``, which
        RECEIVES ROLLOUT CONTENTS (inputs, expected/actual outputs, metrics) in
        its reflection prompts. Candidate evaluation runs through this
        function's configured execution path; end-to-end local training is
        guaranteed only when ``execution_mode`` is local/edge. Under hybrid
        modes, trial payloads (including the candidate document as a
        configuration value) may be submitted to the backend; a warning is
        emitted in that case.
        """

        from traigent.api.parameter_ranges import Choices
        from traigent.generation import (
            SkillTrainOptions,
            merge_prompt_candidates,
            resolve_rewrite_llm,
        )
        from traigent.generation.skill_train.reflection import Reflector
        from traigent.generation.skill_train.trainer import RolloutRecord, SkillTrainer

        def _coerce(spec: Any) -> SkillTrainOptions:
            if spec is None:
                return SkillTrainOptions()
            if isinstance(spec, SkillTrainOptions):
                return spec
            return SkillTrainOptions(**spec)

        options = _coerce(
            skill_train if skill_train is not None else self.skill_train_options
        )
        llm = resolve_rewrite_llm(optimizer_llm)
        reflector = Reflector(llm, model_hint=options.optimizer_model)
        effective_mode = getattr(self, "execution_mode", None)
        if effective_mode and effective_mode != "local":
            logger.warning(
                "train_skill candidate evaluation follows this function's "
                "execution mode (%s): trial payloads, including the candidate "
                "document as a configuration value, may reach the backend. "
                "End-to-end local training requires local mode.",
                effective_mode,
            )
        config_space = dict(self.configuration_space or {})
        resolved_doc_param = self._resolve_skill_doc_param(
            config_space, explicit=doc_param or options.doc_param
        )
        pinned = self._preflight_skill_fixed_config(
            config_space,
            doc_param=resolved_doc_param,
            fixed_config=fixed_config,
        )
        dataset = self._load_dataset()

        def _evaluate_document(
            text: str, split_dataset: Dataset
        ) -> tuple[float, list[RolloutRecord]]:
            evaluation_space: dict[str, Any] = {
                name: Choices([value]) for name, value in pinned.items()
            }
            evaluation_space[resolved_doc_param] = Choices([text])
            self.set_eval_dataset_override(split_dataset)
            try:
                result = self.optimize_sync(
                    configuration_space=evaluation_space,
                    max_trials=1,
                )
            finally:
                self.set_eval_dataset_override(None)
            return self._skill_result_to_rollouts(result, options)

        trainer = SkillTrainer(
            dataset=dataset,
            evaluate_fn=_evaluate_document,
            reflector=reflector,
            options=options,
            selection_dataset=selection_dataset,
            test_dataset=test_dataset,
            artifacts_root=self.local_storage_path,
        )
        result = trainer.run(document)
        result.summary = {
            **result.summary,
            "doc_param": resolved_doc_param,
            "evaluation_basis": result.evaluation_basis,
        }
        if resolved_doc_param in config_space:
            merged = merge_prompt_candidates(
                config_space, resolved_doc_param, [result.best_document]
            )
            result.summary["merged_config_space"] = {
                **config_space,
                resolved_doc_param: merged,
            }
        return result

    def set_eval_dataset_override(self, dataset: Dataset | None) -> None:
        """Pin the dataset returned by ``_load_dataset``.

        Used by guided generation so a grown dataset persists across
        re-optimization rounds. Pass ``None`` to clear.
        """
        self._dataset_override = dataset

    def _resolve_skill_doc_param(
        self,
        config_space: dict[str, Any],
        *,
        explicit: str | None,
    ) -> str:
        from traigent.api.parameter_ranges import TextDocument

        if explicit:
            if explicit not in config_space:
                available = ", ".join(sorted(config_space)) or "<empty>"
                raise ValueError(
                    f"train_skill doc_param {explicit!r} is not a configuration-space "
                    f"parameter (available: {available}). Add it to the config space "
                    "(e.g. as a TextDocument) so the trained document is actually "
                    "wired into the function."
                )
            return explicit

        candidates: list[str] = []
        for name, value in config_space.items():
            if isinstance(value, TextDocument):
                candidates.append(name)

        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                "train_skill requires doc_param because the config space has no "
                "TextDocument parameter to train."
            )
        raise ValueError(
            "train_skill requires doc_param because multiple TextDocument "
            f"parameters could hold the document: {sorted(candidates)}"
        )

    def _preflight_skill_fixed_config(
        self,
        config_space: dict[str, Any],
        *,
        doc_param: str,
        fixed_config: dict[str, Any],
    ) -> dict[str, Any]:
        pinned: dict[str, Any] = {}
        unpinned: list[str] = []
        default_config = dict(getattr(self, "default_config", {}) or {})

        for name, spec in config_space.items():
            if name == doc_param:
                continue
            if name in fixed_config:
                pinned[name] = fixed_config[name]
                continue
            if name in default_config:
                pinned[name] = default_config[name]
                continue

            default_getter = getattr(spec, "get_default", None)
            if callable(default_getter):
                default_value = default_getter()
                if default_value is not None:
                    pinned[name] = default_value
                    continue

            values = getattr(spec, "values", None)
            if values is not None and len(values) == 1:
                pinned[name] = list(values)[0]
                continue
            if isinstance(spec, (list, tuple)) and len(spec) == 1:
                pinned[name] = spec[0]
                continue
            if not isinstance(spec, (list, tuple, dict)) and values is None:
                pinned[name] = spec
                continue
            unpinned.append(name)

        if unpinned:
            raise ValueError(
                "train_skill requires every non-document config parameter to be "
                "pinned. Provide fixed_config values or single defaults for: "
                + ", ".join(sorted(unpinned))
            )
        return pinned

    def _skill_result_to_rollouts(
        self, result: Any, options: Any
    ) -> tuple[float, list[Any]]:
        if not getattr(result, "trials", None):
            raise RuntimeError("train_skill optimization produced no trials")
        trial = result.trials[0]
        score = self._skill_extract_score(result, trial, options.score_metric)
        entries: Any = []
        metadata = getattr(trial, "metadata", None)
        if isinstance(metadata, dict):
            entries = metadata.get("example_results") or []
        rollouts = [
            self._skill_entry_to_rollout(
                entry, options.score_metric, options.failure_threshold
            )
            for entry in entries
        ]
        return score, rollouts

    def _skill_extract_score(
        self, result: Any, trial: Any, score_metric: str | None
    ) -> float:
        metrics = dict(getattr(trial, "metrics", {}) or {})
        result_metrics = dict(getattr(result, "metrics", {}) or {})
        if score_metric is not None:
            for source in (metrics, result_metrics):
                value = source.get(score_metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)

        best_score = getattr(result, "best_score", None)
        if isinstance(best_score, (int, float)) and not isinstance(best_score, bool):
            return float(best_score)

        objective_names = list(getattr(result, "objectives", []) or [])
        for name in [*(objective_names[:1]), "accuracy", "score"]:
            value = metrics.get(name, result_metrics.get(name))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)

        for source in (metrics, result_metrics):
            for value in source.values():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
        raise RuntimeError("train_skill could not resolve a numeric trial score")

    def _skill_entry_to_rollout(
        self,
        entry: Any,
        score_metric: str | None,
        failure_threshold: float,
    ) -> Any:
        from traigent.generation.skill_train.trainer import RolloutRecord

        def _get(name: str, default: Any = None) -> Any:
            if isinstance(entry, Mapping):
                return entry.get(name, default)
            return getattr(entry, name, default)

        metrics_raw = _get("metrics", {}) or {}
        metrics = {
            key: float(value)
            for key, value in dict(metrics_raw).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        success_value = _get("success", None)
        if success_value is None:
            success_value = bool(getattr(entry, "is_successful", False))
        success = bool(success_value)
        metric = score_metric or self._skill_resolve_metric_name(metrics)
        metric_value = metrics.get(metric) if metric else None
        is_failure = not success or (
            isinstance(metric_value, (int, float))
            and float(metric_value) < failure_threshold
        )
        return RolloutRecord(
            example_id=str(_get("example_id", "")),
            input_data=_get("input_data", {}),
            expected=_get("expected_output", _get("expected")),
            actual=_get("actual_output", _get("actual")),
            metrics=metrics,
            success=success,
            is_failure=is_failure,
        )

    @staticmethod
    def _skill_resolve_metric_name(metrics: dict[str, float]) -> str | None:
        for preferred in ("accuracy", "score", "primary"):
            if preferred in metrics:
                return preferred
        if metrics:
            return sorted(metrics)[0]
        return None

    def _load_dataset(self) -> Dataset:
        """Load evaluation dataset.

        Returns:
            Dataset object for evaluation

        Raises:
            ConfigurationError: If dataset cannot be loaded
        """
        # A guided-generation round may have grown the dataset; the override takes
        # precedence so synthesized examples persist across re-optimization rounds.
        if self._dataset_override is not None:
            return self._dataset_override

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

    def clear_override(self) -> bool:
        """Clear a sticky set_config/apply_best_config override."""
        return self._csm.clear_override()  # type: ignore[no-any-return]

    @property
    def best_config_snapshot(self):
        """Return the active immutable best-config snapshot."""
        return self._csm.best_config_snapshot

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

    def export_best_config(
        self,
        directory: str | Path = ".traigent/best-configs",
        *,
        config_id: str | None = None,
        include_metadata: bool = True,
    ) -> Path:
        """Export the best configuration as a canonical repo runtime spec."""
        return self._csm.export_best_config(  # type: ignore[no-any-return]
            directory, config_id=config_id, include_metadata=include_metadata
        )

    def publish_best_config(self, *, target: str = "cloud") -> dict[str, Any]:
        """Publish the best config to a durable remote target when supported."""
        return self._csm.publish_best_config(target=target)  # type: ignore[no-any-return]

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
    def experiment_name(self) -> str:
        """Resolved experiment display name for portal/storage.

        Resolution order (highest to lowest priority):
        1. ``experiment_name`` passed to ``@traigent.optimize()`` — stored in ``_experiment_name``.
        2. ``TRAIGENT_EXPERIMENT_NAME`` environment variable — checked at access time, so
           setting it after decoration still takes effect.
        3. Self-describing default precomputed at decoration time (func name + objectives + knobs)
           — stored in ``_default_experiment_name``.
        4. Decorated function's ``__name__`` (bare fallback when no objectives/knobs exist).
        """
        if self._experiment_name is not None:
            return self._experiment_name
        env_name = os.environ.get("TRAIGENT_EXPERIMENT_NAME")
        if env_name:
            return env_name
        if self._default_experiment_name is not None:
            return self._default_experiment_name
        return self.__name__

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
