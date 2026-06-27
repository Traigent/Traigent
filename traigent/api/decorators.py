"""Main decorator for Traigent SDK.

This module provides the primary @optimize decorator that enables zero-code-change
optimization for any function containing LLM invocations. The decorator automatically
detects and optimizes LLM parameters without requiring any changes to the existing code.
Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Maintainability CONC-Quality-Compatibility FUNC-API-ENTRY FUNC-INVOKERS REQ-API-001 REQ-INJ-002 SYNC-OptimizationFlow

Examples:
    Basic usage with seamless optimization:

    >>> @traigent.optimize(
    ...     eval_dataset="qa_test.jsonl",
    ...     objectives=["accuracy"],
    ...     configuration_space={
    ...         "model": ["gpt-3.5-turbo", "gpt-4"],
    ...         "temperature": [0.1, 0.5, 0.9]
    ...     }
    ... )
    ... def answer_question(question: str) -> str:
    ...     llm = OpenAI()  # Parameters will be auto-optimized
    ...     return llm.complete(question)

    Multi-objective optimization with constraints:

    >>> @traigent.optimize(
    ...     eval_dataset="customer_support.jsonl",
    ...     objectives=["accuracy", "cost", "latency"],
    ...     configuration_space={
    ...         "model": ["gpt-3.5-turbo", "gpt-4", "claude-haiku-4-5-20251001"],
    ...         "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
    ...         "max_tokens": [100, 500, 1000]
    ...     },
    ...     constraints=[
    ...         lambda config: config["temperature"] < 0.8 if config["model"] == "gpt-4" else True,
    ...         lambda config, metrics: metrics.get("cost", 0) <= 0.10
    ...     ]
    ... )
    ... def process_ticket(ticket: str) -> str:
    ...     # Your existing code remains unchanged
    ...     return llm_chain.process(ticket)
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, cast

if TYPE_CHECKING:
    from traigent.api.config_space import ConfigSpace
    from traigent.api.constraints import BoolExpr, Constraint
    from traigent.api.safety import CompoundSafetyConstraint, SafetyConstraint

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from traigent.api.functions import _GLOBAL_CONFIG
from traigent.api.parameter_ranges import (
    ParameterRange,
    TextDocument,
    is_inline_param_definition,
    normalize_configuration_space,
)
from traigent.api.strategy_presets import (
    NormalizedStrategyPreset,
    UnknownStrategyPresetError,
    is_strategy_preset_name,
    normalize_strategy_preset,
)
from traigent.api.types import AgentDefinition
from traigent.config.parallel import (
    ParallelConfig,
    coerce_parallel_config,
    merge_parallel_configs,
)
from traigent.config.types import (
    ExecutionIntent,
    ExecutionMode,
    InjectionMode,
    ResolvedExecutionPolicy,
    is_traigent_disabled,
    resolve_execution_policy,
    validate_algorithm_name,
)
from traigent.core.objectives import (
    ObjectiveSchema,
    create_default_objectives,
    normalize_objectives,
)
from traigent.core.optimized_function import OptimizedFunction
from traigent.defaults import DEFAULT_MAX_TRIALS
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.tvl.options import TVLOptions
from traigent.tvl.promotion_gate import PromotionGate
from traigent.tvl.spec_loader import TVLSpecArtifact, load_tvl_spec
from traigent.utils.exceptions import (
    ConfigurationError,
    TVLValidationError,
    ValidationError,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


__all__ = [
    "EvaluationOptions",
    "InjectionOptions",
    "HybridAPIOptions",
    "ExternalServiceEvaluator",
    "ExecutionOptions",
    "MockModeOptions",
    "optimize",
]


class EvaluationOptions(BaseModel):
    """Grouped evaluation settings used by the optimize decorator."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    eval_dataset: (
        str | list[str | dict[str, Any] | EvaluationExample] | Dataset | None
    ) = None
    custom_evaluator: Callable[..., Any] | None = None
    scoring_function: Callable[..., Any] | None = None
    metric_functions: dict[str, Callable[..., Any]] | None = None


class InjectionOptions(BaseModel):
    """Configuration bundle controlling how optimized configs are injected.

    Attributes:
        injection_mode: How to inject config ("context", "parameter", "seamless").
            - "context" (default): Thread-safe contextvars, access via get_config()
            - "parameter": Explicit config param in function signature
            - "seamless": Zero code change, AST transformation
        config_param: Parameter name for injection_mode="parameter".
        auto_override_frameworks: Whether to auto-override framework calls.
        framework_targets: List of framework names to target.
        effectuation: Opt-in executable TVAR effectuation. Defaults to False,
            preserving existing trial-call behavior.

    Note:
        ATTRIBUTE mode was removed in v2.x due to thread-safety issues.
        Use "context" (recommended) or "seamless" instead.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    injection_mode: str | InjectionMode = InjectionMode.CONTEXT
    config_param: str | None = None
    # Default to False - requires traigent-integrations plugin for framework overrides
    # Set to True explicitly when using framework integrations
    auto_override_frameworks: bool = False
    framework_targets: list[str] | None = None
    effectuation: bool = False

    @field_validator("injection_mode", mode="before")
    @classmethod
    def validate_injection_mode(cls, v: str | InjectionMode) -> str | InjectionMode:
        """Validate injection_mode and reject removed modes."""
        if isinstance(v, str) and v in ("attribute", "decorator"):
            raise ValueError(
                f"injection_mode='{v}' has been removed in v2.x. "
                'Use "context" (recommended), "parameter", or "seamless" instead.'
            )
        return v


class HybridAPIOptions(BaseModel):
    """Deprecated compatibility bundle for external-service evaluators."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    endpoint: str | None = None
    tunable_id: str | None = None
    transport: Any | None = None
    transport_type: str = "auto"
    batch_size: int = Field(default=1, ge=1)
    batch_parallelism: int = Field(default=1, ge=1)
    keep_alive: bool = True
    heartbeat_interval: float = Field(default=30.0, gt=0)
    timeout: float | None = None
    auth_header: str | None = None
    auto_discover_tvars: bool = False


class ExternalServiceEvaluator(BaseModel):
    """Evaluator bundle for external services used by optimization."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    kind: Literal["hybrid_api"] = "hybrid_api"
    hybrid_api: HybridAPIOptions = Field(default_factory=HybridAPIOptions)


class ExecutionOptions(BaseModel):
    """Execution and orchestration preferences for optimization runs.

    Note:
        ``runtime`` and ``js_*`` execution fields were removed with the
        temporary Python-orchestrated JS bridge. Use the ``traigent-js`` npm
        package for JavaScript optimization.

    Attributes:
        algorithm: Optimizer selector. ``auto`` (default) resolves to the
            managed optimizer policy, ``grid``/``random`` stay local, and known
            smart algorithms require managed orchestration.
        offline: Force local-only, zero-egress resolution.
        local_storage_path: Path for local result storage.
        minimal_logging: Whether to minimize logging output.
        parallel_config: Configuration for parallel execution.
        max_total_examples: Maximum total examples across all trials.
        samples_include_pruned: Whether to include pruned trials in sample count.
        reps_per_trial: Number of repetitions per configuration for statistical
            stability. Only the default ``1`` (no repetition) is available in the
            OSS SDK; passing any other value raises ``pydantic.ValidationError`` at
            construction time. Per-configuration repetition requires Traigent
            Enterprise (contact ``sales@traigent.ai``).
        reps_aggregation: How to aggregate metrics across repetitions. Only the
            default ``"mean"`` is available in the OSS SDK; passing any other
            value raises ``pydantic.ValidationError`` at
            construction time. Per-configuration repetition aggregation
            requires Traigent Enterprise (contact ``sales@traigent.ai``).
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        validate_assignment=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_js_bridge_fields(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            _reject_removed_js_bridge_options(data)
        return data

    algorithm: str = "auto"
    offline: bool = False
    evaluator: ExternalServiceEvaluator | dict[str, Any] | None = None
    local_storage_path: str | None = None
    minimal_logging: bool = True
    parallel_config: ParallelConfig | dict[str, Any] | None = None
    max_total_examples: int | None = None
    samples_include_pruned: bool = True
    reps_per_trial: int = 1
    reps_aggregation: str = "mean"

    @model_validator(mode="before")
    @classmethod
    def _warn_legacy_execution_fields(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            _warn_for_legacy_execution_options(set(data))
            if isinstance(data.get("evaluator"), HybridAPIOptions):
                import warnings

                warnings.warn(
                    "HybridAPIOptions as an evaluator is deprecated. Pass an "
                    "ExternalServiceEvaluator or evaluator options dict instead.",
                    DeprecationWarning,
                    stacklevel=4,
                )
                data = {
                    **data,
                    "evaluator": ExternalServiceEvaluator(hybrid_api=data["evaluator"]),
                }
            legacy_hybrid_values = {
                option_key: data[legacy_key]
                for legacy_key, option_key in _LEGACY_HYBRID_API_OPTION_MAP.items()
                if legacy_key in data
            }
            if legacy_hybrid_values:
                HybridAPIOptions.model_validate(legacy_hybrid_values)
        return data

    @field_validator("reps_per_trial")
    @classmethod
    def _reject_non_default_reps_per_trial(cls, v: int) -> int:
        # Per-configuration repetition is enterprise-gated; fail at the contract
        # boundary (construction) instead of late at runtime so callers see the
        # gate before the @optimize decorator is applied.
        if v != 1:
            raise ValueError(
                "reps_per_trial != 1 is not available in this version. "
                "Per-configuration repetitions for statistical stability "
                "require Traigent Enterprise. Contact sales@traigent.ai."
            )
        return v

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, v: str) -> str:
        return validate_algorithm_name(v)

    @field_validator("offline")
    @classmethod
    def _validate_offline(cls, v: bool) -> bool:
        if not isinstance(v, bool):
            raise ValueError("offline must be a bool")
        return v

    @field_validator("reps_aggregation")
    @classmethod
    def _reject_non_default_reps_aggregation(cls, v: str) -> str:
        if v != "mean":
            raise ValueError(
                "reps_aggregation != 'mean' is not available in this version. "
                "Per-configuration repetition aggregation requires Traigent "
                "Enterprise. Contact sales@traigent.ai."
            )
        return v


class MockModeOptions(BaseModel):
    """Fine-grained configuration for mock mode behaviour.

    .. deprecated::
        **All MockModeOptions fields are inert in the current SDK.**
        Mock mode is enabled by calling ``traigent.testing.enable_mock_mode_for_quickstart()``
        from local tutorial or test code, not via this object. The
        legacy ``TRAIGENT_MOCK_LLM=true`` env var remains available outside
        production for shell fixtures and backwards compatibility, but
        direct user-set env-var activation emits ``DeprecationWarning``. ``enabled``,
        ``override_evaluator``, ``base_accuracy``, and ``variance`` are
        retained on the schema for backwards compatibility so existing
        serialized configs round-trip without breaking, but the
        optimization pipeline ignores all of them. In mock mode the LLM
        call layer is intercepted with canned/deterministic responses;
        the scoring path (built-in metrics, custom evaluators, and the
        ``LocalEvaluator`` accuracy calculator) is unchanged — there is
        no random-score fabrication. Walkthrough scripts under
        ``walkthrough/mock/`` use their own helper ``get_mock_accuracy``
        for example scoring; that helper is example-only and is not part
        of the SDK runtime behavior.

        This deprecation is doc-only. The fields will be removed in a
        future major version.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    enabled: bool = True  # inert; see class docstring
    override_evaluator: bool = True  # inert; see class docstring
    base_accuracy: float = 0.75  # inert; see class docstring
    variance: float = 0.25  # inert; see class docstring


BundleModel = TypeVar("BundleModel", bound=BaseModel)
# ParamSpec / TypeVar for the @optimize decorator's generic return type.
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _coerce_bundle(
    value: Any, model_cls: type[BundleModel], parameter_name: str
) -> BundleModel | None:
    if value is None:
        return None
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        if model_cls is ExecutionOptions:
            _reject_removed_js_bridge_options(value)
        return cast(BundleModel, model_cls.model_validate(value))
    raise TypeError(
        f"{parameter_name} must be a dict or {model_cls.__name__}, got {type(value).__name__}"
    )


def _validate_custom_evaluator_signature(evaluator: Callable[..., Any]) -> None:
    """Validate that custom_evaluator has the expected signature.

    The custom_evaluator must accept (func, config, example) and return ExampleResult.
    This catches interface mismatches early at decoration time.

    Args:
        evaluator: The custom evaluator callable to validate.

    Raises:
        ValidationError: If the evaluator signature doesn't match expectations.
    """
    # Get the callable to inspect (handle both functions and class instances)
    if callable(evaluator) and not inspect.isfunction(evaluator):
        # It's a class instance with __call__, inspect the __call__ method
        # Note: We're accessing __call__ for signature inspection, not testing callability
        callable_to_check = getattr(evaluator, "__call__", evaluator)  # noqa: B004
    else:
        callable_to_check = evaluator

    try:
        sig = inspect.signature(callable_to_check)
    except (ValueError, TypeError):
        # Can't inspect signature (e.g., built-in), skip validation
        logger.debug(
            "Could not inspect custom_evaluator signature, skipping validation"
        )
        return

    params = list(sig.parameters.values())

    # Filter out 'self' for bound methods
    if params and params[0].name == "self":
        params = params[1:]

    # Check parameter count (must accept at least 3: func, config, example)
    required_params = [
        p
        for p in params
        if p.default is inspect.Parameter.empty
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    param_names = [p.name for p in params]
    metric_evaluator_params = {"prediction", "expected", "input_data"}
    has_metric_evaluator_signature = metric_evaluator_params <= set(param_names)

    # Error if signature looks like metric evaluator (has prediction, expected, input_data)
    if has_metric_evaluator_signature:
        raise ValidationError(
            f"custom_evaluator signature mismatch.\n\n"
            f"Expected: custom_evaluator(func, config, example) -> ExampleResult\n"
            f"Got:      {evaluator.__class__.__name__}({', '.join(param_names)})\n\n"
            f"This looks like a metric evaluator. Did you mean to use:\n"
            f"    evaluation=EvaluationOptions(\n"
            f"        metric_functions={{'accuracy': my_evaluator}}\n"
            f"    )\n"
            f"instead of:\n"
            f"    evaluation=EvaluationOptions(\n"
            f"        custom_evaluator=my_evaluator\n"
            f"    )"
        )

    if len(required_params) < 3:
        raise ValidationError(
            f"custom_evaluator must accept (func, config, example), "
            f"got {len(required_params)} required parameters: {param_names}"
        )

    # Warn if parameter names suggest wrong interface (but don't error)
    suspicious_names = {"prediction", "expected", "input_data", "output", "response"}
    found_suspicious = suspicious_names & set(param_names)
    if found_suspicious:
        logger.warning(
            "custom_evaluator parameter names %s suggest metric evaluator interface. "
            "custom_evaluator expects (func, config, example), not (prediction, expected, input_data). "
            "Did you mean to use metric_functions instead?",
            found_suspicious,
        )


def _warn_context_mode_param_shadowing(
    func: Callable[..., Any],
    configuration_space: Any,
    injection_mode: Any,
    config_param: str | None,
) -> None:
    """Warn when CONTEXT-mode tuned knobs shadow the wrapped function's params.

    In the default ``injection_mode=InjectionMode.CONTEXT`` the optimizer does
    **not** override function parameters — the per-trial config lives in
    contextvars and must be read via ``traigent.get_config()``. If a function
    already declares the tuned knobs as parameters (the common "wrap a pre-existing
    agent function" shape) and never reads ``get_config()``, every trial silently
    receives the signature defaults, so the optimizer reports a "best config" for a
    sweep that never actually varied those parameters (see issue #1372). This emits
    a loud warning so the no-op is not silent. It is advisory (never raises): a
    function that intentionally reads ``get_config()`` is detected and skipped.
    """
    # Only CONTEXT mode shadows parameters; PARAMETER/SEAMLESS inject explicitly.
    if injection_mode not in (InjectionMode.CONTEXT, "context"):
        return
    if not configuration_space:
        return
    try:
        config_keys = set(configuration_space.keys())
    except AttributeError:
        return
    if not config_keys:
        return

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - unintrospectable callable
        return

    param_names = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    if config_param:
        param_names.discard(config_param)

    shadowed = sorted(config_keys & param_names)
    if not shadowed:
        return

    # Best-effort: if the body already reads the per-trial config, the knobs are
    # honored and the overlap is intentional — do not warn.
    try:
        source = inspect.getsource(func)
        if "get_config" in source or "current_config" in source:
            return
    except (OSError, TypeError):  # pragma: no cover - source unavailable
        pass

    import warnings

    func_name = getattr(func, "__name__", repr(func))
    message = (
        f"@traigent.optimize: the tuned variable(s) {shadowed} are declared as "
        f"parameters of '{func_name}', but injection_mode is CONTEXT (the default), "
        f"which does NOT override function parameters. Unless the body reads "
        f"traigent.get_config(), every trial will run with the signature defaults "
        f"and the optimization will silently sweep nothing (a false-positive 'best "
        f"config'). To actually vary {shadowed}: read them via traigent.get_config() "
        f'inside the function, or use injection_mode="seamless" (zero code change) / '
        f'injection_mode="parameter".'
    )
    warnings.warn(message, UserWarning, stacklevel=3)
    logger.warning("%s", message)


_DEFAULT_SENTINEL = object()
_AUTO_DETECT_TVARS_MODES = frozenset({"off", "suggest", "apply"})

_OPTIMIZE_DEFAULTS: dict[str, Any] = {
    "eval_dataset": None,
    "objectives": None,
    "configuration_space": None,
    "experiment_name": None,
    "default_config": None,
    "warm_start_from": None,
    "constraints": None,
    "safety_constraints": None,
    "tvl_spec": None,
    "tvl_environment": None,
    "tvl": None,
    "injection_mode": InjectionMode.CONTEXT,
    "config_param": None,
    "auto_override_frameworks": False,  # Requires traigent-integrations plugin
    "framework_targets": None,
    "effectuation": False,
    "algorithm": "auto",
    "offline": False,
    "evaluator": None,
    "local_storage_path": None,
    "minimal_logging": True,
    "parallel_config": None,
    "max_total_examples": None,
    "samples_include_pruned": True,
    "mock_mode_config": None,
    "custom_evaluator": None,
    "scoring_function": None,
    "metric_functions": None,
    "evaluation": None,
    "injection": None,
    "execution": None,
    "mock": None,
    "max_trials": DEFAULT_MAX_TRIALS,
    # Early stopping parameters
    "plateau_window": None,  # Stop if no improvement for N trials
    "plateau_epsilon": None,  # Improvement threshold for plateau detection
    # Multi-agent configuration
    "agents": None,  # Explicit agent definitions
    "agent_prefixes": None,  # Prefix-based agent inference
    "agent_measures": None,  # Agent-to-measures mapping
    "global_measures": None,  # Global (non-agent) measures
    # Config persistence (Phase 1 of optimization-persistency feature)
    "auto_load_best": False,  # Auto-load best config on decoration
    "load_from": None,  # Explicit path to load config from
    "config_id": None,  # Stable best-config identifier
    "best_config_source": "off",  # off|repo|cloud|repo_then_cloud|cloud_then_repo
    "best_config_strict": False,  # Fail startup/refresh on invalid active source
    "best_config_cache_dir": None,  # Local cloud best-config cache
    "best_config_cache_ttl_seconds": 24 * 60 * 60,  # Fresh cache window
    "best_config_stale_ok_ttl_seconds": None,  # Offline stale-cache reuse window
    "enable_auto_load_dev_logs": None,  # Back-compat dev-log auto-load toggle
    # Tuned variable auto-detection
    "auto_detect_tvars": False,  # Log suggestions when no configuration_space is set
    "auto_detect_tvars_mode": None,  # "off" | "suggest" | "apply"
    "auto_detect_tvars_min_confidence": "medium",  # "high" | "medium" | "low"
    "auto_detect_tvars_include": None,  # Optional names to include
    "auto_detect_tvars_exclude": None,  # Optional names to exclude
}

_LEGACY_HYBRID_API_OPTION_MAP = {
    "hybrid_api_endpoint": "endpoint",
    "tunable_id": "tunable_id",
    "hybrid_api_transport": "transport",
    "hybrid_api_transport_type": "transport_type",
    "hybrid_api_batch_size": "batch_size",
    "hybrid_api_batch_parallelism": "batch_parallelism",
    "hybrid_api_keep_alive": "keep_alive",
    "hybrid_api_heartbeat_interval": "heartbeat_interval",
    "hybrid_api_timeout": "timeout",
    "hybrid_api_auth_header": "auth_header",
    "hybrid_api_auto_discover_tvars": "auto_discover_tvars",
}
_LEGACY_EXECUTION_OPTION_KEYS = frozenset(
    {
        "execution_mode",
        "privacy_enabled",
        "cloud_fallback_policy",
        *_LEGACY_HYBRID_API_OPTION_MAP.keys(),
    }
)
_DIRECT_OPTION_KEYS = (
    frozenset(_OPTIMIZE_DEFAULTS.keys()) | _LEGACY_EXECUTION_OPTION_KEYS
)
_JS_BRIDGE_REMOVED_MESSAGE = (
    "JS bridge removed. Use the traigent-js npm package for JavaScript optimization."
)
_JS_BRIDGE_REMOVED_PARAMETERS = frozenset(
    (
        "runtime",
        "js_module",
        "js_function",
        "js_timeout",
        "js_parallel_workers",
        "js_use_npx",
        "js_runner_path",
        "js_node_executable",
    )
)
_REMOVED_PARAMETERS = frozenset(
    (
        "auto_optimize",
        "trigger",
        "batch_size",
        "parallel_trials",
        *_JS_BRIDGE_REMOVED_PARAMETERS,
    )
)
_ALLOWED_RUNTIME_OVERRIDE_KEYS = frozenset(
    (
        "metric_limit",
        "metric_name",
        "metric_include_pruned",
        "plateau_window",
        "plateau_epsilon",
        "cost_limit",
        "cost_approved",
        "tie_breakers",
        "tvl_parameter_agents",
    )
)


def _warn_for_legacy_execution_options(keys: set[str]) -> None:
    """Emit coarse deprecation warnings for legacy execution option names."""
    import warnings

    if keys & {"execution_mode"}:
        warnings.warn(
            "execution_mode is deprecated as a public optimizer selector. Use "
            "algorithm='auto'|'grid'|'random' and offline=True for no egress.",
            DeprecationWarning,
            stacklevel=4,
        )
    if keys & {"privacy_enabled"}:
        warnings.warn(
            "privacy_enabled is deprecated and has no effect. Use offline=True "
            "for no egress.",
            DeprecationWarning,
            stacklevel=4,
        )
    if keys & {"cloud_fallback_policy"}:
        warnings.warn(
            "cloud_fallback_policy is deprecated and has no effect. Set "
            "TRAIGENT_REQUIRE_CLOUD=1 to disable fallback.",
            DeprecationWarning,
            stacklevel=4,
        )
    if keys & set(_LEGACY_HYBRID_API_OPTION_MAP):
        warnings.warn(
            "Flat hybrid_api_* optimize options are deprecated. Pass an "
            "ExternalServiceEvaluator or evaluator options dict instead.",
            DeprecationWarning,
            stacklevel=4,
        )


def _removed_parameter_message(parameter_name: str) -> str:
    if parameter_name in _JS_BRIDGE_REMOVED_PARAMETERS:
        return _JS_BRIDGE_REMOVED_MESSAGE
    return (
        f"{parameter_name} parameter has been removed. Use supported arguments such as "
        "parallel_config or ExecutionOptions instead."
    )


def _reject_removed_js_bridge_options(options: Mapping[str, Any]) -> None:
    if set(options) & _JS_BRIDGE_REMOVED_PARAMETERS:
        raise TypeError(_JS_BRIDGE_REMOVED_MESSAGE)


def get_optimize_default(parameter_name: str) -> Any:
    """Return the optimize() default for a parameter.

    Args:
        parameter_name: Name of the optimize() parameter.

    Returns:
        The default value configured for the requested parameter.
    """
    if parameter_name not in _OPTIMIZE_DEFAULTS:
        raise KeyError(f"{parameter_name!r} is not a recognized optimize() parameter.")
    return _OPTIMIZE_DEFAULTS[parameter_name]


@dataclass
class LegacyOptimizeArgs:
    """Container for legacy optimize() arguments."""

    eval_dataset: (
        str | list[str | dict[str, Any] | EvaluationExample] | Dataset | None
    ) = None
    objectives: list[str] | ObjectiveSchema | None = None
    experiment_name: str | None = None
    configuration_space: dict[str, Any] | None = None
    default_config: dict[str, Any] | None = None
    warm_start_from: str | None = None
    constraints: list[Callable[..., Any]] | None = None
    safety_constraints: list[Any] | None = (
        None  # SafetyConstraint | CompoundSafetyConstraint
    )
    tvl_spec: str | None = None
    tvl_environment: str | None = None
    tvl: TVLOptions | dict[str, Any] | None = None
    injection_mode: str | InjectionMode | None = None
    config_param: str | None = None
    auto_override_frameworks: bool | None = None
    framework_targets: list[str] | None = None
    effectuation: bool | None = None
    execution_mode: str | None = None
    hybrid_api_endpoint: str | None = None
    tunable_id: str | None = None
    hybrid_api_transport: Any | None = None
    hybrid_api_transport_type: str | None = None
    hybrid_api_batch_size: int | None = None
    hybrid_api_batch_parallelism: int | None = None
    hybrid_api_keep_alive: bool | None = None
    hybrid_api_heartbeat_interval: float | None = None
    hybrid_api_timeout: float | None = None
    hybrid_api_auth_header: str | None = None
    hybrid_api_auto_discover_tvars: bool | None = None
    cloud_fallback_policy: str | None = None
    local_storage_path: str | None = None
    minimal_logging: bool | None = None
    parallel_config: ParallelConfig | dict[str, Any] | None = None
    privacy_enabled: bool | None = None
    max_total_examples: int | None = None
    samples_include_pruned: bool | None = None
    mock_mode_config: dict[str, Any] | None = None
    custom_evaluator: Callable[..., Any] | None = None
    scoring_function: Callable[..., Any] | None = None
    metric_functions: dict[str, Callable[..., Any]] | None = None
    evaluation: EvaluationOptions | dict[str, Any] | None = None
    injection: InjectionOptions | dict[str, Any] | None = None
    execution: ExecutionOptions | dict[str, Any] | None = None
    evaluator: ExternalServiceEvaluator | dict[str, Any] | None = None
    mock: MockModeOptions | dict[str, Any] | None = None
    algorithm: str | None = None
    offline: bool | None = None
    max_trials: int | None = None
    # Multi-agent configuration
    agents: dict[str, AgentDefinition] | None = None
    agent_prefixes: list[str] | None = None
    agent_measures: dict[str, list[str]] | None = None
    global_measures: list[str] | None = None
    # Config persistence
    auto_load_best: bool | None = None
    load_from: str | None = None
    config_id: str | None = None
    best_config_source: str | None = None
    best_config_strict: bool | None = None
    best_config_cache_dir: str | None = None
    best_config_cache_ttl_seconds: int | None = None
    best_config_stale_ok_ttl_seconds: int | None = None
    enable_auto_load_dev_logs: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> LegacyOptimizeArgs:
        """Build a legacy arguments bundle from raw values.

        Args:
            values: Mapping of optimize keyword names to values captured from
                legacy call sites.

        Returns:
            A populated ``LegacyOptimizeArgs`` instance with extras separated.
        """
        recognized: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in values.items():
            if key in _DIRECT_OPTION_KEYS:
                recognized[key] = value
            else:
                extra[key] = value
        init_kwargs: dict[str, Any] = {**recognized, "extra": extra}
        return cls(**init_kwargs)

    def iter_known_values(self) -> list[tuple[str, Any]]:
        """Return key/value pairs for supported optimize parameters.

        Returns:
            List of ``(parameter_name, value)`` tuples including None entries
            for unset parameters.
        """
        return [
            ("eval_dataset", self.eval_dataset),
            ("objectives", self.objectives),
            ("experiment_name", self.experiment_name),
            ("configuration_space", self.configuration_space),
            ("default_config", self.default_config),
            ("warm_start_from", self.warm_start_from),
            ("constraints", self.constraints),
            ("safety_constraints", self.safety_constraints),
            ("tvl_spec", self.tvl_spec),
            ("tvl_environment", self.tvl_environment),
            ("tvl", self.tvl),
            ("injection_mode", self.injection_mode),
            ("config_param", self.config_param),
            ("auto_override_frameworks", self.auto_override_frameworks),
            ("framework_targets", self.framework_targets),
            ("effectuation", self.effectuation),
            ("execution_mode", self.execution_mode),
            ("hybrid_api_endpoint", self.hybrid_api_endpoint),
            ("tunable_id", self.tunable_id),
            ("hybrid_api_transport", self.hybrid_api_transport),
            ("hybrid_api_transport_type", self.hybrid_api_transport_type),
            ("hybrid_api_batch_size", self.hybrid_api_batch_size),
            ("hybrid_api_batch_parallelism", self.hybrid_api_batch_parallelism),
            ("hybrid_api_keep_alive", self.hybrid_api_keep_alive),
            ("hybrid_api_heartbeat_interval", self.hybrid_api_heartbeat_interval),
            ("hybrid_api_timeout", self.hybrid_api_timeout),
            ("hybrid_api_auth_header", self.hybrid_api_auth_header),
            ("hybrid_api_auto_discover_tvars", self.hybrid_api_auto_discover_tvars),
            ("cloud_fallback_policy", self.cloud_fallback_policy),
            ("local_storage_path", self.local_storage_path),
            ("minimal_logging", self.minimal_logging),
            ("parallel_config", self.parallel_config),
            ("privacy_enabled", self.privacy_enabled),
            ("max_total_examples", self.max_total_examples),
            ("samples_include_pruned", self.samples_include_pruned),
            ("mock_mode_config", self.mock_mode_config),
            ("custom_evaluator", self.custom_evaluator),
            ("scoring_function", self.scoring_function),
            ("metric_functions", self.metric_functions),
            ("evaluation", self.evaluation),
            ("injection", self.injection),
            ("execution", self.execution),
            ("evaluator", self.evaluator),
            ("mock", self.mock),
            ("algorithm", self.algorithm),
            ("offline", self.offline),
            ("max_trials", self.max_trials),
            ("agents", self.agents),
            ("agent_prefixes", self.agent_prefixes),
            ("agent_measures", self.agent_measures),
            ("global_measures", self.global_measures),
            ("auto_load_best", self.auto_load_best),
            ("load_from", self.load_from),
            ("config_id", self.config_id),
            ("best_config_source", self.best_config_source),
            ("best_config_strict", self.best_config_strict),
            ("best_config_cache_dir", self.best_config_cache_dir),
            ("best_config_cache_ttl_seconds", self.best_config_cache_ttl_seconds),
            ("best_config_stale_ok_ttl_seconds", self.best_config_stale_ok_ttl_seconds),
            ("enable_auto_load_dev_logs", self.enable_auto_load_dev_logs),
        ]


def _resolve_option(
    parameter_name: str,
    current_value: Any,
    bundle_value: Any,
    defaults: dict[str, Any],
) -> Any:
    if bundle_value is None:
        return current_value

    default_value = defaults.get(parameter_name, _DEFAULT_SENTINEL)
    if default_value is _DEFAULT_SENTINEL:
        if current_value is not None and current_value != bundle_value:
            raise TypeError(
                f"Conflicting values for {parameter_name!r} supplied via both direct "
                "arguments and grouped options. Remove one of the definitions."
            )
        return current_value if current_value is not None else bundle_value

    if current_value != default_value:
        if current_value != bundle_value:
            raise TypeError(
                f"Conflicting values for {parameter_name!r} supplied via both direct "
                "arguments and grouped options. Remove one of the definitions."
            )
        return current_value

    return bundle_value


def _resolve_tvl_options(
    tvl_spec_value: str | Path | None,
    tvl_environment_value: str | None,
    tvl_bundle: TVLOptions | None,
) -> TVLOptions | None:
    if tvl_bundle is None and tvl_spec_value is None:
        if tvl_environment_value:
            raise TypeError(
                "tvl_environment requires a tvl_spec or structured tvl options"
            )
        return None

    if tvl_bundle is not None:
        bundle = tvl_bundle
        if tvl_spec_value and Path(bundle.spec_path) != Path(tvl_spec_value):
            raise TypeError("Conflicting tvl_spec definitions provided.")
        if tvl_environment_value:
            bundle = bundle.merged_with(environment=tvl_environment_value)
        return bundle

    return TVLOptions(spec_path=str(tvl_spec_value), environment=tvl_environment_value)


def _apply_tvl_artifact(
    *,
    artifact: TVLSpecArtifact,
    options: TVLOptions,
    configuration_space: dict[str, Any] | None,
    objectives: list[str] | ObjectiveSchema | None,
    constraints: list[Constraint | BoolExpr | Callable[..., Any]] | None,
    default_config: dict[str, Any] | None,
    runtime_overrides: dict[str, Any],
) -> tuple[
    dict[str, Any] | None,
    list[str] | ObjectiveSchema | None,
    list[Constraint | BoolExpr | Callable[..., Any]] | None,
    dict[str, Any] | None,
]:
    if options.apply_configuration_space:
        if configuration_space is not None:
            raise TypeError(
                "Provide configuration_space or tvl_spec (with apply_configuration_space), not both."
            )
        configuration_space = artifact.configuration_space

    if options.apply_objectives and artifact.objective_schema is not None:
        objectives = artifact.objective_schema

    if options.apply_constraints and artifact.constraints:
        # artifact.constraints are already callables from TVL; cast for type compatibility
        # Use string annotation since Constraint/BoolExpr are TYPE_CHECKING imports
        tvl_constraints: list[Constraint | BoolExpr | Callable[..., Any]] = cast(
            "list[Constraint | BoolExpr | Callable[..., Any]]", artifact.constraints
        )
        constraints = list(constraints or []) + tvl_constraints

    if options.apply_budget:
        overrides = artifact.runtime_overrides()
        for key, value in overrides.items():
            runtime_overrides.setdefault(key, value)

    if not default_config and artifact.default_config:
        default_config = artifact.default_config

    return configuration_space, objectives, constraints, default_config


def _parse_legacy_args(
    legacy: LegacyOptimizeArgs | dict[str, Any] | None,
) -> LegacyOptimizeArgs | None:
    """Parse legacy arguments into LegacyOptimizeArgs if provided."""
    if legacy is None:
        return None
    if isinstance(legacy, LegacyOptimizeArgs):
        return legacy
    if isinstance(legacy, dict):
        return LegacyOptimizeArgs.from_mapping(legacy)
    raise TypeError("legacy must be a LegacyOptimizeArgs instance or dict")


def _build_settings_recorder(
    combined_settings: dict[str, Any],
    provided_sources: dict[str, str],
) -> Callable[[str, Any, str], None]:
    """Build a recorder function for merging optimize options."""

    def record_option(key: str, value: Any, source: str) -> None:
        """Record a resolved option and track its configuration source."""
        if value is None:
            return
        if key in _REMOVED_PARAMETERS:
            raise TypeError(_removed_parameter_message(key))
        existing_source = provided_sources.get(key)
        if existing_source is not None:
            existing_value = combined_settings[key]
            default_value = _OPTIMIZE_DEFAULTS.get(key, _DEFAULT_SENTINEL)
            if (
                source == "optimize parameter"
                and default_value is not _DEFAULT_SENTINEL
                and value == default_value
            ):
                return
            if existing_value != value:
                raise TypeError(
                    f"Conflicting values for {key!r} supplied via both {existing_source} "
                    f"and {source}. Remove one of the definitions."
                )
            return
        combined_settings[key] = value
        provided_sources[key] = source

    return record_option


def _extract_inline_params(
    combined_runtime_overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract inline parameter definitions from runtime overrides.

    Returns:
        Tuple of (inline_params, remaining_overrides).
    """
    inline_params: dict[str, Any] = {}
    remaining_overrides: dict[str, Any] = {}
    for key, value in combined_runtime_overrides.items():
        if is_inline_param_definition(value):
            inline_params[key] = value
        else:
            remaining_overrides[key] = value
    return inline_params, remaining_overrides


def _normalize_runtime_override_aliases(
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Normalize runtime override keys."""
    return dict(overrides)


def _resolve_strategy_argument(
    *,
    strategy: str | None,
    strategy_params: Mapping[str, Any] | None,
    runtime_overrides: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Split public ``strategy`` into preset selection."""
    if strategy is None:
        if strategy_params is not None:
            normalize_strategy_preset(None, strategy_params)
        return None, runtime_overrides

    if is_strategy_preset_name(strategy) or strategy_params is not None:
        return strategy, runtime_overrides

    raise UnknownStrategyPresetError(strategy)


def _apply_strategy_preset_to_options(
    *,
    strategy_preset: NormalizedStrategyPreset | None,
    objectives: list[str] | ObjectiveSchema | None,
    constraints: list[Constraint | BoolExpr | Callable[..., Any]] | None,
) -> tuple[
    list[str] | ObjectiveSchema | None,
    list[Constraint | BoolExpr | Callable[..., Any]] | None,
]:
    """Apply advisory preset objectives without adding search constraints."""
    if strategy_preset is None:
        return objectives, constraints
    if objectives is not None:
        raise ValueError(
            "strategy presets are mutually exclusive with explicit objectives. "
            "Use either strategy=... or objectives=..., not both."
        )
    return list(strategy_preset.objectives), constraints


def _validate_runtime_overrides(remaining_overrides: dict[str, Any]) -> None:
    """Validate that runtime overrides don't contain unknown keys."""
    unknown_keys = (
        set(remaining_overrides.keys())
        - _DIRECT_OPTION_KEYS
        - _ALLOWED_RUNTIME_OVERRIDE_KEYS
    )
    if unknown_keys:
        raise TypeError(
            f"Unknown keyword arguments: {sorted(unknown_keys)}. "
            f"If you meant to define parameter ranges, use Range(), IntRange(), "
            f"Choices(), or tuple syntax. Example: temperature=Range(0.0, 1.0)"
        )


def _ensure_scope_name(scope_var_names: dict[int, str], name: str) -> None:
    """Ensure a parameter name appears in scope even without a live object id."""
    if name in scope_var_names.values():
        return

    synthetic_id = -1
    while synthetic_id in scope_var_names:
        synthetic_id -= 1
    scope_var_names[synthetic_id] = name


def _iter_constraint_scope_sources(
    raw_configuration_space: dict[str, Any] | ConfigSpace | None,
    inline_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect configuration mappings that contribute names to constraint scope."""
    sources: list[dict[str, Any]] = []
    if isinstance(raw_configuration_space, dict):
        sources.append(raw_configuration_space)
    else:
        tvars = getattr(raw_configuration_space, "tvars", None)
        if isinstance(tvars, Mapping):
            sources.append(dict(tvars))
    if inline_params:
        sources.append(inline_params)
    return sources


def _build_constraint_scope_var_names(
    raw_configuration_space: dict[str, Any] | ConfigSpace | None,
    inline_params: dict[str, Any],
    config_space_var_names: dict[int, str] | None,
) -> dict[int, str] | None:
    """Build var-name scope used for compile-time constraint validation."""
    scope_var_names: dict[int, str] = dict(config_space_var_names or {})

    for source in _iter_constraint_scope_sources(
        raw_configuration_space, inline_params
    ):
        for name, value in source.items():
            if not isinstance(name, str):
                continue
            _ensure_scope_name(scope_var_names, name)
            if isinstance(value, ParameterRange):
                scope_var_names[id(value)] = name

    return scope_var_names or None


def _augment_constraint_scope_var_names(
    scope_var_names: dict[int, str] | None,
    configuration_space: dict[str, Any] | None,
) -> dict[int, str] | None:
    """Augment scope map with normalized configuration-space keys."""
    if scope_var_names is None and not configuration_space:
        return None

    merged = dict(scope_var_names or {})
    for name in configuration_space or {}:
        if isinstance(name, str):
            _ensure_scope_name(merged, name)
    return merged


def _resolve_evaluation_bundle_options(
    evaluation_bundle: EvaluationOptions | None,
    eval_dataset: Any,
    custom_evaluator: Any,
    scoring_function: Any,
    metric_functions: Any,
    defaults: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    """Resolve evaluation options from bundle."""
    if evaluation_bundle is None:
        return eval_dataset, custom_evaluator, scoring_function, metric_functions

    return (
        _resolve_option(
            "eval_dataset", eval_dataset, evaluation_bundle.eval_dataset, defaults
        ),
        _resolve_option(
            "custom_evaluator",
            custom_evaluator,
            evaluation_bundle.custom_evaluator,
            defaults,
        ),
        _resolve_option(
            "scoring_function",
            scoring_function,
            evaluation_bundle.scoring_function,
            defaults,
        ),
        _resolve_option(
            "metric_functions",
            metric_functions,
            evaluation_bundle.metric_functions,
            defaults,
        ),
    )


def _resolve_injection_bundle_options(
    injection_bundle: InjectionOptions | None,
    injection_mode: Any,
    config_param: Any,
    auto_override_frameworks: Any,
    framework_targets: Any,
    effectuation: Any,
    defaults: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any]:
    """Resolve injection options from bundle."""
    if injection_bundle is None:
        return (
            injection_mode,
            config_param,
            auto_override_frameworks,
            framework_targets,
            effectuation,
        )

    return (
        _resolve_option(
            "injection_mode", injection_mode, injection_bundle.injection_mode, defaults
        ),
        _resolve_option(
            "config_param", config_param, injection_bundle.config_param, defaults
        ),
        _resolve_option(
            "auto_override_frameworks",
            auto_override_frameworks,
            injection_bundle.auto_override_frameworks,
            defaults,
        ),
        _resolve_option(
            "framework_targets",
            framework_targets,
            injection_bundle.framework_targets,
            defaults,
        ),
        _resolve_option(
            "effectuation", effectuation, injection_bundle.effectuation, defaults
        ),
    )


@dataclass(slots=True)
class ResolvedExecutionOptions:
    """Resolved execution options after merging direct and bundled settings."""

    algorithm: Any
    offline: Any
    evaluator: Any
    local_storage_path: Any
    minimal_logging: Any
    parallel_config: Any
    max_total_examples: Any
    samples_include_pruned: Any
    legacy_options: dict[str, Any] = field(default_factory=dict)


def _resolve_execution_bundle_options(
    execution_bundle: ExecutionOptions | None,
    base_options: ResolvedExecutionOptions,
    defaults: dict[str, Any],
) -> ResolvedExecutionOptions:
    """Resolve execution options from bundle.

    Enterprise-gated fields (``reps_per_trial``, ``reps_aggregation``) are
    rejected at ``ExecutionOptions`` construction time via field validators
    (see the tracked fix), so this resolver only handles option merging.
    """
    if execution_bundle is None:
        return base_options

    # Defense in depth: pydantic field_validator on ExecutionOptions already
    # rejects non-default values at construction time. This runtime check
    # catches any path that constructs the bundle through a different route.
    if execution_bundle.reps_per_trial != 1:
        raise NotImplementedError(
            "reps_per_trial is not available in this version. "
            "This feature requires Traigent Enterprise. "
            "Contact sales@traigent.ai for more information."
        )
    if execution_bundle.reps_aggregation != "mean":
        raise NotImplementedError(
            "reps_aggregation is not available in this version. "
            "This feature requires Traigent Enterprise. "
            "Contact sales@traigent.ai for more information."
        )

    extra = getattr(execution_bundle, "__pydantic_extra__", None) or {}
    legacy_options = {**base_options.legacy_options}
    for key in _LEGACY_EXECUTION_OPTION_KEYS:
        if key in extra:
            existing = legacy_options.get(key, _DEFAULT_SENTINEL)
            if existing is not _DEFAULT_SENTINEL and existing != extra[key]:
                raise TypeError(
                    f"Conflicting values for {key!r} supplied via both direct "
                    "arguments and ExecutionOptions. Remove one of the definitions."
                )
            legacy_options[key] = extra[key]

    return ResolvedExecutionOptions(
        algorithm=_resolve_option(
            "algorithm",
            base_options.algorithm,
            execution_bundle.algorithm,
            defaults,
        ),
        offline=_resolve_option(
            "offline",
            base_options.offline,
            execution_bundle.offline,
            defaults,
        ),
        evaluator=_resolve_option(
            "evaluator",
            base_options.evaluator,
            execution_bundle.evaluator,
            defaults,
        ),
        local_storage_path=_resolve_option(
            "local_storage_path",
            base_options.local_storage_path,
            execution_bundle.local_storage_path,
            defaults,
        ),
        minimal_logging=_resolve_option(
            "minimal_logging",
            base_options.minimal_logging,
            execution_bundle.minimal_logging,
            defaults,
        ),
        parallel_config=_resolve_option(
            "parallel_config",
            base_options.parallel_config,
            execution_bundle.parallel_config,
            defaults,
        ),
        max_total_examples=_resolve_option(
            "max_total_examples",
            base_options.max_total_examples,
            execution_bundle.max_total_examples,
            defaults,
        ),
        samples_include_pruned=_resolve_option(
            "samples_include_pruned",
            base_options.samples_include_pruned,
            execution_bundle.samples_include_pruned,
            defaults,
        ),
        legacy_options=legacy_options,
    )


def _resolve_injection_mode_enum(
    injection_mode: str | InjectionMode,
) -> str | InjectionMode:
    """Convert string injection mode to enum, handling removed modes."""
    if not isinstance(injection_mode, str):
        return injection_mode

    # Handle removed injection modes with helpful migration message
    if injection_mode in ("attribute", "decorator"):
        from traigent.utils.exceptions import ConfigurationError

        raise ConfigurationError(
            f"injection_mode='{injection_mode}' has been removed in v2.x.\n\n"
            "Migration guide:\n"
            '  Before: @traigent.optimize(injection_mode="attribute")\n'
            "          config = my_func.current_config\n\n"
            '  After:  @traigent.optimize(injection_mode="context")\n'
            "          config = traigent.get_config()\n\n"
            'For zero code changes, use injection_mode="seamless" instead.'
        )

    try:
        return InjectionMode(injection_mode)
    except ValueError:
        return injection_mode


def _resolve_execution_policy_from_options(
    *,
    algorithm: str,
    offline: bool,
    legacy_options: Mapping[str, Any],
    source_hint: str,
) -> ResolvedExecutionPolicy:
    """Resolve public and legacy execution controls into a policy object."""
    try:
        return resolve_execution_policy(
            algorithm=algorithm,
            offline=offline,
            execution_mode=legacy_options.get("execution_mode"),
            privacy_enabled=legacy_options.get("privacy_enabled"),
            cloud_fallback_policy=legacy_options.get("cloud_fallback_policy"),
            source_hint=source_hint,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from None


def _coerce_external_service_evaluator(
    value: Any,
) -> ExternalServiceEvaluator | None:
    """Normalize evaluator/external-service config to a typed bundle."""

    if value is None:
        return None
    if isinstance(value, ExternalServiceEvaluator):
        return value
    if isinstance(value, HybridAPIOptions):
        import warnings

        warnings.warn(
            "HybridAPIOptions as an evaluator is deprecated. Pass an "
            "ExternalServiceEvaluator or evaluator options dict instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return ExternalServiceEvaluator(hybrid_api=value)
    if isinstance(value, dict):
        if "hybrid_api" in value or "kind" in value:
            return cast(
                ExternalServiceEvaluator, ExternalServiceEvaluator.model_validate(value)
            )
        return ExternalServiceEvaluator(
            hybrid_api=HybridAPIOptions.model_validate(value)
        )

    # Some full-suite reload paths can hand us a pydantic model instance from a
    # previous import of this module. Accept the current public shape rather
    # than relying only on class identity.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
        hybrid_payload = (
            payload.get("hybrid_api") if isinstance(payload, dict) else None
        )
        hybrid_dump = getattr(hybrid_payload, "model_dump", None)
        if callable(hybrid_dump):
            payload["hybrid_api"] = hybrid_dump()
        if isinstance(payload, dict) and ("hybrid_api" in payload or "kind" in payload):
            return cast(
                ExternalServiceEvaluator,
                ExternalServiceEvaluator.model_validate(payload),
            )

    if hasattr(value, "hybrid_api"):
        hybrid_value = value.hybrid_api
        hybrid_dump = getattr(hybrid_value, "model_dump", None)
        hybrid_payload = hybrid_dump() if callable(hybrid_dump) else hybrid_value
        return cast(
            ExternalServiceEvaluator,
            ExternalServiceEvaluator.model_validate(
                {
                    "kind": getattr(value, "kind", "hybrid_api"),
                    "hybrid_api": hybrid_payload,
                }
            ),
        )

    if any(hasattr(value, field) for field in ("endpoint", "transport", "tunable_id")):
        hybrid_payload = {
            field: getattr(value, field)
            for field in HybridAPIOptions.model_fields
            if hasattr(value, field)
        }
        return ExternalServiceEvaluator(
            hybrid_api=HybridAPIOptions.model_validate(hybrid_payload)
        )

    raise TypeError(
        "evaluator must be an ExternalServiceEvaluator, external-service options dict, or None"
    )


def _merge_hybrid_api_options(
    current: HybridAPIOptions | None,
    legacy_values: dict[str, Any],
) -> HybridAPIOptions | None:
    """Merge deprecated flat evaluator kwargs into a typed option bundle."""

    if not legacy_values:
        return current

    import warnings

    warnings.warn(
        "Flat hybrid_api_* optimize options are deprecated. Pass an "
        "ExternalServiceEvaluator or evaluator options dict instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    current_data = current.model_dump() if current is not None else {}
    merged = dict(current_data)
    for legacy_key, option_key in _LEGACY_HYBRID_API_OPTION_MAP.items():
        if legacy_key not in legacy_values:
            continue
        value = legacy_values[legacy_key]
        if value is None:
            continue
        existing = merged.get(option_key)
        default = HybridAPIOptions.model_fields[option_key].default
        if existing not in (None, default) and existing != value:
            raise TypeError(
                f"Conflicting values for external-service option {option_key!r} "
                "supplied via both evaluator and deprecated flat arguments."
            )
        merged[option_key] = value
    return cast(HybridAPIOptions, HybridAPIOptions.model_validate(merged))


def _resolve_external_service_evaluator(
    evaluator: Any,
    legacy_options: Mapping[str, Any],
) -> ExternalServiceEvaluator | None:
    """Resolve evaluator config plus deprecated flat external-service kwargs."""

    resolved = _coerce_external_service_evaluator(evaluator)
    legacy_hybrid_values = {
        key: value
        for key, value in legacy_options.items()
        if key in _LEGACY_HYBRID_API_OPTION_MAP
    }
    hybrid_api_options = _merge_hybrid_api_options(
        resolved.hybrid_api if resolved is not None else None,
        legacy_hybrid_values,
    )
    if hybrid_api_options is None:
        return resolved
    return ExternalServiceEvaluator(hybrid_api=hybrid_api_options)


def _runtime_algorithm_for_policy(
    policy: ResolvedExecutionPolicy,
    external_evaluator: ExternalServiceEvaluator | None,
) -> str:
    """Map public algorithm policy to the runtime optimizer selector."""

    if policy.algorithm == "auto" and (
        policy.intent is ExecutionIntent.LOCAL_ONLY or external_evaluator is not None
    ):
        return "random"
    return policy.algorithm


def _runtime_execution_mode_for_policy(
    policy: ResolvedExecutionPolicy,
    external_evaluator: ExternalServiceEvaluator | None,
) -> ExecutionMode:
    """Compatibility mode for the current runtime constructor."""

    if external_evaluator is not None:
        return ExecutionMode.HYBRID_API
    if policy.intent is ExecutionIntent.LOCAL_ONLY:
        return ExecutionMode.EDGE_ANALYTICS
    return ExecutionMode.HYBRID


_OBJECTIVES_TYPE_ERROR = (
    "objectives must be a sequence of strings or an ObjectiveSchema"
)


def _validate_objectives(
    objectives: list[str] | ObjectiveSchema | None,
) -> None:
    """Validate objectives format."""
    if objectives is None or isinstance(objectives, ObjectiveSchema):
        return

    if isinstance(objectives, (str, bytes)):
        raise ValidationError(_OBJECTIVES_TYPE_ERROR)
    try:
        iter(objectives)
    except TypeError as exc:
        raise ValidationError(_OBJECTIVES_TYPE_ERROR) from exc
    if not all(isinstance(obj, str) for obj in objectives):
        raise ValidationError(
            "All objectives must be strings when provided as a sequence"
        )


def _resolve_objective_schema(
    objectives: list[str] | ObjectiveSchema | None,
) -> ObjectiveSchema:
    """Resolve objective schema from objectives or global config."""
    resolved_schema = normalize_objectives(objectives)
    if resolved_schema is not None:
        return resolved_schema

    global_schema = _GLOBAL_CONFIG.get("objective_schema")
    if isinstance(global_schema, ObjectiveSchema):
        return global_schema

    global_names = _GLOBAL_CONFIG.get("objectives") or []
    if global_names:
        return create_default_objectives(list(global_names))

    return create_default_objectives(["accuracy"])


def _resolve_actual_execution_mode(
    execution_mode: str,
) -> str:
    """Resolve legacy execution mode from provided or global config."""
    if (
        execution_mode == ExecutionMode.EDGE_ANALYTICS.value
        and "execution_mode" in _GLOBAL_CONFIG
    ):
        result: str = str(_GLOBAL_CONFIG["execution_mode"])
        logger.debug(f"Using execution mode from global config: {result}")
        return result
    logger.debug(f"Using explicitly provided execution mode: {execution_mode}")
    return execution_mode


def _log_execution_mode_warnings(
    execution_mode_enum: ExecutionMode,
    actual_execution_mode: str,
    local_storage_path: str | None,
    minimal_logging: bool,
) -> None:
    """Log warnings for incompatible runtime settings."""
    if minimal_logging and execution_mode_enum is not ExecutionMode.EDGE_ANALYTICS:
        logger.warning(
            "minimal_logging is only effective for offline local runs. "
            "It will be ignored for the selected managed execution path."
        )


def _check_deprecated_objective_kwargs(
    combined_runtime_overrides: dict[str, Any],
) -> None:
    """Check for and reject deprecated objective kwargs."""
    deprecated_objective_kwargs = {
        key
        for key in ("objective_orientations", "objective_weights")
        if key in combined_runtime_overrides
    }
    if deprecated_objective_kwargs:
        raise TypeError(
            "objective_orientations/objective_weights are no longer supported. "
            "Provide an ObjectiveSchema when you need explicit orientations or weights."
        )


def _apply_tvl_options_if_present(
    tvl_options: TVLOptions | None,
    configuration_space: dict[str, Any] | None,
    objectives: list[str] | ObjectiveSchema | None,
    constraints: list[Any] | None,
    default_config: dict[str, Any] | None,
    eval_dataset: Any,
    combined_runtime_overrides: dict[str, Any],
) -> tuple[
    dict[str, Any] | None,
    list[str] | ObjectiveSchema | None,
    list[Any] | None,
    dict[str, Any] | None,
    Any,
    PromotionGate | None,
]:
    """Apply TVL options if present, returning updated values.

    Returns:
        Tuple of (configuration_space, objectives, constraints, default_config,
        eval_dataset, promotion_gate).
    """
    if tvl_options is None:
        return (
            configuration_space,
            objectives,
            constraints,
            default_config,
            eval_dataset,
            None,  # promotion_gate
        )

    try:
        tvl_artifact = load_tvl_spec(**tvl_options.to_kwargs())  # type: ignore[arg-type]
    except TVLValidationError as exc:
        raise ValidationError(exc.message) from exc

    configuration_space, objectives, constraints, default_config = _apply_tvl_artifact(
        artifact=tvl_artifact,
        options=tvl_options,
        configuration_space=configuration_space,
        objectives=objectives,
        constraints=constraints,
        default_config=default_config,
        runtime_overrides=combined_runtime_overrides,
    )

    if (
        tvl_options.apply_evaluation_set
        and eval_dataset is None
        and tvl_artifact.evaluation_set is not None
    ):
        eval_dataset = tvl_artifact.evaluation_set.dataset

    # Create PromotionGate from TVL spec if promotion_policy is defined
    promotion_gate = PromotionGate.from_spec_artifact(tvl_artifact)
    if promotion_gate is not None:
        logger.debug("PromotionGate created from TVL spec promotion_policy")

    env_suffix = f" (env={tvl_options.environment})" if tvl_options.environment else ""
    logger.info("TVL spec %s applied%s", tvl_artifact.path, env_suffix)

    return (
        configuration_space,
        objectives,
        constraints,
        default_config,
        eval_dataset,
        promotion_gate,
    )


def _process_runtime_overrides(
    runtime_overrides: dict[str, Any],
    legacy_args: LegacyOptimizeArgs | None,
    record_option: Callable[[str, Any, str], None],
) -> dict[str, Any]:
    """Process runtime overrides, handling removed parameters."""
    combined_runtime_overrides: dict[str, Any] = {}
    if legacy_args:
        combined_runtime_overrides.update(
            _normalize_runtime_override_aliases(legacy_args.extra)
        )

    normalized_runtime_overrides = _normalize_runtime_override_aliases(
        runtime_overrides
    )
    _warn_for_legacy_execution_options(set(normalized_runtime_overrides))

    for key, value in normalized_runtime_overrides.items():
        if key in _REMOVED_PARAMETERS:
            raise TypeError(_removed_parameter_message(key))
        if key in _DIRECT_OPTION_KEYS:
            record_option(key, value, "keyword argument")
        else:
            combined_runtime_overrides[key] = value

    removed_in_runtime = set(combined_runtime_overrides) & _REMOVED_PARAMETERS
    if removed_in_runtime:
        if removed_in_runtime & _JS_BRIDGE_REMOVED_PARAMETERS:
            raise TypeError(_JS_BRIDGE_REMOVED_MESSAGE)
        raise TypeError(
            "The following optimize() parameters have been removed: "
            f"{sorted(removed_in_runtime)}. Use supported arguments such as parallel_config "
            "or ExecutionOptions instead."
        )

    return combined_runtime_overrides


def _process_config_space_constraints(
    configuration_space: dict[str, Any] | ConfigSpace | None,
    constraints: list[Any] | None,
) -> tuple[Any, Any, Any]:
    """Process ConfigSpace constraints, returning constraints and var_names."""
    config_space_constraints_attr = getattr(configuration_space, "constraints", None)
    config_space_has_constraints = bool(config_space_constraints_attr)
    if config_space_has_constraints and constraints:
        raise TypeError(
            "Cannot provide both ConfigSpace with constraints and explicit constraints. "
            "Either include constraints in your ConfigSpace or pass them separately."
        )

    config_space_constraints = None
    config_space_var_names = None
    if config_space_has_constraints:
        config_space_constraints = getattr(configuration_space, "constraints", None)
        config_space_var_names = getattr(configuration_space, "var_names", None)

    return (
        config_space_constraints,
        config_space_var_names,
        config_space_has_constraints,
    )


def _normalize_config_space_and_defaults(
    configuration_space: dict[str, Any] | ConfigSpace | None,
    inline_params: dict[str, Any],
    default_config: dict[str, Any] | None,
    config_space_constraints: Any,
    constraints: list[Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[Any] | None]:
    """Normalize configuration space and merge defaults."""
    normalized_space: dict[str, Any] | None = None
    if inline_params or configuration_space:
        normalized_space, param_defaults = normalize_configuration_space(
            configuration_space, inline_params
        )
        _restore_text_document_markers(
            normalized_space, configuration_space, inline_params
        )
        if param_defaults:
            default_config = {**param_defaults, **(default_config or {})}

    if config_space_constraints and not constraints:
        constraints = config_space_constraints

    return normalized_space, default_config, constraints


def _constraints_for_satisfiability(constraints: list[Any] | None) -> list[Any]:
    """Return structural constraints that the satisfiability checker can evaluate."""
    if not constraints:
        return []

    from traigent.api.constraints import BoolExpr, Constraint

    satisfiability_constraints: list[Constraint] = []
    for constraint in constraints:
        if isinstance(constraint, Constraint):
            satisfiability_constraints.append(constraint)
        elif isinstance(constraint, BoolExpr):
            satisfiability_constraints.append(Constraint(expr=constraint))
    return satisfiability_constraints


def _validate_constraint_satisfiability(
    raw_configuration_space: dict[str, Any] | ConfigSpace | None,
    inline_params: dict[str, Any],
    constraints: list[Any] | None,
) -> None:
    """Fail fast when structural constraints make a finite space impossible."""
    satisfiability_constraints = _constraints_for_satisfiability(constraints)
    if not satisfiability_constraints:
        return

    from traigent.api.config_space import ConfigSpace
    from traigent.api.validation_protocol import SatStatus

    if isinstance(raw_configuration_space, ConfigSpace):
        tvars = dict(raw_configuration_space.tvars)
        if inline_params:
            tvars.update(
                ConfigSpace.from_decorator_args(inline_params=inline_params).tvars
            )
        space = ConfigSpace(tvars=tvars, constraints=tuple(satisfiability_constraints))
    else:
        space = ConfigSpace.from_decorator_args(
            configuration_space=raw_configuration_space,
            inline_params=inline_params,
            constraints=satisfiability_constraints,
        )

    if not space.tvars:
        return

    result = space.check_satisfiability()
    if result.status == SatStatus.UNSAT:
        detail = result.message or "no valid configuration satisfies all constraints"
        raise ValueError(f"constraints are unsatisfiable: {detail}")


def _restore_text_document_markers(
    normalized_space: dict[str, Any],
    raw_configuration_space: dict[str, Any] | ConfigSpace | None,
    inline_params: dict[str, Any],
) -> None:
    """Keep train_skill auto-discovery markers after decorator normalization."""

    for source in _iter_constraint_scope_sources(
        raw_configuration_space, inline_params
    ):
        for name, value in source.items():
            if isinstance(value, TextDocument):
                normalized_space[name] = value


def _resolve_auto_detect_tvars_mode(
    auto_detect_tvars: Any, auto_detect_tvars_mode: Any
) -> str:
    """Resolve effective auto-detection mode with backward compatibility."""
    if not isinstance(auto_detect_tvars, bool):
        raise TypeError(
            "auto_detect_tvars must be a bool. "
            f"Received {type(auto_detect_tvars).__name__}."
        )
    if auto_detect_tvars_mode is None:
        return "suggest" if auto_detect_tvars else "off"
    if not isinstance(auto_detect_tvars_mode, str):
        raise TypeError(
            "auto_detect_tvars_mode must be one of "
            f"{sorted(_AUTO_DETECT_TVARS_MODES)}, got "
            f"{type(auto_detect_tvars_mode).__name__}."
        )
    mode = auto_detect_tvars_mode.strip().lower()
    if mode not in _AUTO_DETECT_TVARS_MODES:
        raise TypeError(
            "auto_detect_tvars_mode must be one of "
            f"{sorted(_AUTO_DETECT_TVARS_MODES)}, got {auto_detect_tvars_mode!r}."
        )
    return mode


def _coerce_auto_detect_tvars_min_confidence(value: Any) -> str:
    """Validate and normalize auto-detection minimum confidence."""
    from traigent.tuned_variables.detection_types import DetectionConfidence

    if isinstance(value, DetectionConfidence):
        return str(value.value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return str(DetectionConfidence(normalized).value)
        except ValueError as exc:
            raise TypeError(
                "auto_detect_tvars_min_confidence must be one of "
                "['high', 'medium', 'low']."
            ) from exc
    raise TypeError(
        "auto_detect_tvars_min_confidence must be a string ('high'|'medium'|'low')."
    )


def _coerce_auto_detect_tvars_name_filter(
    value: Any, option_name: str
) -> set[str] | None:
    """Validate include/exclude name filters for auto-detection."""
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    if isinstance(value, Collection):
        values = list(value)
        if any(not isinstance(v, str) for v in values):
            raise TypeError(f"{option_name} must contain only strings.")
        return set(values)
    raise TypeError(
        f"{option_name} must be None, a string, or a collection of strings."
    )


def _suggest_detected_tvars(
    func: Callable[..., Any],
    *,
    mode: str,
    min_confidence: str,
    include: Collection[str] | None = None,
    exclude: Collection[str] | None = None,
) -> dict[str, Any] | None:
    """Run detection on ``func`` and return filtered configuration candidates.

    Logs suggestion or auto-apply messages at INFO level.
    """
    try:
        from traigent.tuned_variables.detector import TunedVariableDetector

        detector = TunedVariableDetector()
        result = detector.detect_from_callable(func)
        if result.count == 0:
            return None

        config_space = result.to_configuration_space(
            format="ranges",
            min_confidence=min_confidence,
            include=include,
            exclude=exclude,
        )
        if not config_space:
            logger.info(
                "auto_detect_tvars: no candidates met filters in '%s' "
                "(min_confidence=%s, include=%s, exclude=%s).",
                func.__name__,
                min_confidence,
                sorted(include) if include is not None else None,
                sorted(exclude) if exclude is not None else None,
            )
            return None

        suggestions = [
            f"{c.name} ({c.candidate_type.value}, {c.confidence.value} confidence)"
            + (
                f" → {c.suggested_range.to_parameter_range_code()}"
                if c.suggested_range
                else ""
            )
            for c in result.candidates
            if c.name in config_space
        ]

        if mode == "apply":
            logger.info(
                "auto_detect_tvars: auto-applied %d tunable variable(s) to "
                "configuration_space in '%s': %s",
                len(config_space),
                func.__name__,
                suggestions,
            )
        else:
            logger.info(
                "auto_detect_tvars: detected %d tunable variable(s) in '%s'. "
                "Consider adding to configuration_space: %s",
                len(config_space),
                func.__name__,
                suggestions,
            )
        return cast("dict[str, Any]", config_space)
    except Exception:
        logger.debug("auto_detect_tvars: detection failed", exc_info=True)
        return None


def optimize(  # NOSONAR(S107)
    *,
    objectives: list[str] | ObjectiveSchema | None = None,
    configuration_space: dict[str, Any] | ConfigSpace | None = None,
    experiment_name: str | None = None,
    default_config: dict[str, Any] | None = None,
    warm_start_from: str | None = None,
    constraints: list[Constraint | BoolExpr | Callable[..., Any]] | None = None,
    safety_constraints: list[SafetyConstraint | CompoundSafetyConstraint] | None = None,
    tvl_spec: str | Path | None = None,
    tvl_environment: str | None = None,
    tvl: TVLOptions | dict[str, Any] | None = None,
    evaluation: EvaluationOptions | dict[str, Any] | None = None,
    injection: InjectionOptions | dict[str, Any] | None = None,
    effectuation: bool = False,
    execution: ExecutionOptions | dict[str, Any] | None = None,
    evaluator: ExternalServiceEvaluator | dict[str, Any] | None = None,
    mock: MockModeOptions | dict[str, Any] | None = None,
    algorithm: str = "auto",
    offline: bool = False,
    strategy: str | None = None,
    strategy_params: Mapping[str, Any] | None = None,
    # Multi-agent configuration
    agents: dict[str, AgentDefinition] | None = None,
    agent_prefixes: list[str] | None = None,
    agent_measures: dict[str, list[str]] | None = None,
    global_measures: list[str] | None = None,
    # Config persistence
    auto_load_best: bool = False,
    load_from: str | None = None,
    config_id: str | None = None,
    best_config_source: str = "off",
    best_config_strict: bool = False,
    best_config_cache_dir: str | None = None,
    best_config_cache_ttl_seconds: int = 24 * 60 * 60,
    best_config_stale_ok_ttl_seconds: int | None = None,
    enable_auto_load_dev_logs: bool | None = None,
    # Guided generation: configure here, then run via fn.optimize_with_guidance(provider)
    prompt_rewrite: dict[str, Any] | None = None,
    grow_dataset: dict[str, Any] | None = None,
    skill_train: dict[str, Any] | None = None,
    legacy: LegacyOptimizeArgs | dict[str, Any] | None = None,
    **runtime_overrides: Any,
) -> Callable[
    [Callable[_P, _R]], OptimizedFunction[_P, _R]
]:  # NOSONAR - stable public API intentionally exposes broad options
    """Decorator to make functions optimizable with Traigent.

    This is the main entry point for Traigent optimization. Decorate any function
    with @traigent.optimize to add zero-code-change optimization capabilities.
    The decorator automatically detects and optimizes LLM invocations without
    requiring any modifications to your existing code. Use the grouped bundles
    for structured parameters and the ``legacy`` argument to bridge the previous
    expansive signature when needed.

    Args:
        objectives: Target metrics to optimize. Accepts a list of names (Traigent
            infers sensible orientations and equal weights) or an ObjectiveSchema
            for explicit weights, orientations, and metadata. Omitted values fall
            back to ``traigent.configure(objectives=...)`` or ``["accuracy"]``.
            Mutually exclusive with strategy presets; use one path so the
            business-goal preset cannot silently override hand-set objectives.
        strategy: Optional advisory strategy preset name. Supported preset names
            are ``max_accuracy_then_cheapest_within_epsilon``,
            ``quality_floor_min_cost``, and ``pareto_frontier``.
            Use ``algorithm`` to select an optimizer by name.
        strategy_params: Typed parameters for the selected strategy preset.
            ``epsilon`` is required for
            ``max_accuracy_then_cheapest_within_epsilon`` and must be > 0 and
            <= 1. ``floor`` is required for ``quality_floor_min_cost`` and must
            be between 0 and 1 inclusive. ``pareto_frontier`` accepts no params.
        configuration_space: Dictionary describing the search space. Keys are
            parameter names; values can be discrete lists, numeric tuples, or nested
            dicts for composite parameters.
        experiment_name: Human-readable display name for this experiment shown in
            the Traigent portal and local storage. When ``None`` (default), the
            decorated function's ``__name__`` is used. Falls back to the
            ``TRAIGENT_EXPERIMENT_NAME`` environment variable if set and no
            explicit value is passed. Allows names with spaces, punctuation, and
            other characters not valid in Python identifiers, for example
            ``"Amir txt2sql v1 (ACL 0.8)"``.
        default_config: Baseline configuration materialized before the first trial.
            In seamless/attribute modes these override literal values during the
            initial run. In parameter mode the dict is converted to a TraigentConfig
            and provided as the ``config`` argument.

            Default Value Precedence (highest to lowest):
                1. Explicit ``default_config`` dict values
                2. ``ParameterRange.default`` values (e.g., ``Range(0.0, 1.0, default=0.7)``)
                3. Optimizer-suggested defaults (e.g., Optuna's suggest_* midpoint)

            Example showing precedence::

                >>> @traigent.optimize(
                ...     temperature=Range(0.0, 1.0, default=0.5),  # Precedence 2
                ...     default_config={"temperature": 0.7},      # Precedence 1 (wins)
                ... )
                ... def my_func(): ...

        constraints: Optional validators receiving ``config`` and ``metrics``. Return
            True to accept a configuration or False to skip it.
        safety_constraints: Not yet implemented - raises ``NotImplementedError``.
            See traigent-smartopt#26.

        TVL integration:
            tvl_spec: Path to a TVL spec. When provided (and ``tvl`` opts allow it)
                the spec becomes the authoritative configuration space/objective
                source.
            tvl_environment: Named environment overlay from the spec.
            tvl: Structured ``TVLOptions`` (or dict) controlling how the spec is
                applied (configuration/objectives/budgets/constraints).

        Evaluation options:
            evaluation: Grouped evaluation settings (EvaluationOptions or dict). Use
                this bundle to supply datasets and evaluators together. When present
                it replaces the individual arguments historically passed directly.
            eval_dataset: Dataset input accepted as a JSONL path, list of paths,
                inline example dicts, or a Dataset instance with ``input`` and
                ``expected_output`` columns.
            custom_evaluator: Callback ``(func, config, example) -> ExampleResult``.
            scoring_function: Evaluator ``(output, expected, llm_metrics)``.
            metric_functions: Mapping of metric name to evaluator functions.

        Injection options:
            injection: Grouped injection settings (InjectionOptions or dict) covering
                how optimized parameters are applied.
            injection_mode: Selector for the injection mechanism (context, parameter,
                seamless). "context" (default) uses thread-safe contextvars,
                "parameter" adds explicit config param, "seamless" uses AST
                transformation for zero code changes.
            config_param: Parameter name used when ``injection_mode="parameter"``.
            auto_override_frameworks: Toggle to auto-detect supported frameworks
                (LangChain, OpenAI, Anthropic, etc.) and override their parameters.
            framework_targets: Explicit list of framework classes to override.
            effectuation: Opt-in executable TVAR effectuation. Defaults to False.

        Execution options:
            execution: Grouped execution settings (ExecutionOptions or dict) spanning
                orchestration, storage, and parallelism.
            algorithm: Optimizer selector. ``auto`` (default) uses managed
                orchestration with local fallback, ``grid``/``random`` stay
                local, and known smart optimizers require managed orchestration.
            offline: Force local-only, zero-egress resolution.
            local_storage_path: Location for local result storage. Falls back
                to ``TRAIGENT_RESULTS_FOLDER`` or ``~/.traigent/`` when omitted.
            minimal_logging: Toggle for reduced logging noise in offline local runs.
            parallel_config: Consolidated parallel configuration (ParallelConfig
                or dict). Preferred path for controlling concurrency.
            max_total_examples: Global sample budget across all trials.
            samples_include_pruned: Whether pruned trials count toward the sample budget.

        Mock mode options:
            mock: Grouped mock-mode preferences (MockModeOptions or dict).
            mock_mode_config: Legacy mock-mode dict. **Inert** — the SDK
                no longer reads ``enabled``, ``override_evaluator``,
                ``base_accuracy``, or ``variance`` from this dict. Use
                ``traigent.testing.enable_mock_mode_for_quickstart()`` in
                local tutorial or test code to enable mock mode. The legacy
                ``TRAIGENT_MOCK_LLM`` env var remains supported outside
                production for shell fixtures but emits ``DeprecationWarning``
                when users set it directly. The parameter is retained for
                config round-trip;
                see the tracked fix.

        Cost safeguards:
            cost_limit: Maximum USD spending per optimization run. Defaults to
                TRAIGENT_RUN_COST_LIMIT env var or $2.00.
            cost_approved: Skip cost approval prompt. Use with caution in production.
            metric_limit: Soft cumulative stop for a named completed-trial metric.
                Requires metric_name. Use for counters such as total tokens or
                cumulative latency, not hard money-spend control.
            metric_name: Metric summed by metric_limit.

        Additional controls:
            legacy: Adapter for the legacy decorator signature. Accepts either a
                LegacyOptimizeArgs instance or a dict with the historic keyword
                arguments. Values provided here merge with the explicit parameters.
            **runtime_overrides: Stop-condition and budget knobs accepted by
                the decorator. The currently supported keys are:
                ``metric_limit``, ``metric_name``,
                ``metric_include_pruned``, ``plateau_window``,
                ``plateau_epsilon``, ``cost_limit``, ``cost_approved``,
                ``tie_breakers``, and ``tvl_parameter_agents``.
                Note: ``algorithm`` and ``max_trials`` are first-class
                parameters of this decorator (not in ``**runtime_overrides``);
                ``timeout`` is supported on
                ``OptimizedFunction.optimize(timeout=...)`` at call time,
                not on the decorator. Tuned-variable detection controls
                (``auto_detect_tvars`` and friends) are available as
                first-class parameters.

    Warning:
        Optimization runs multiple LLM calls. Use
        ``traigent.testing.enable_mock_mode_for_quickstart()`` for local testing.
        Cost estimates are approximations; actual billing is determined by your LLM provider.
        Traigent cost limits, alerts, thresholds, and stop conditions are best-effort
        local guardrails, not provider-side billing caps.
        See DISCLAIMER.md for full liability terms.

    Important - Configuration Access:
        Traigent does NOT automatically override function parameters. The default
        injection_mode="context" stores trial configuration in context variables.
        To access trial config, use ``traigent.get_trial_config()`` or the simpler
        ``traigent.get_config()`` which works both during and after optimization.

        WRONG - relying on parameter defaults to be overridden::

            @traigent.optimize(configuration_space={"model": ["gpt-4", "gpt-3.5"]})
            def my_func(model: str = "gpt-4"):  # model will stay "gpt-4"!
                return call_llm(model=model)

        CORRECT - explicitly fetch trial config::

            @traigent.optimize(configuration_space={"model": ["gpt-4", "gpt-3.5"]})
            def my_func():
                cfg = traigent.get_config()  # Works during trials and after apply_best_config
                return call_llm(model=cfg["model"])

        Note: ``get_trial_config()`` raises ``OptimizationStateError`` if called
        outside an active trial - use it when you want strict validation.
        ``get_config()`` is the recommended approach for most use cases.

    Returns:
        OptimizedFunction: Wrapper that adds optimization methods to your function:
            - optimize(): Run optimization and return results
            - get_best_config(): Get current best configuration
            - set_config(): Manually set configuration
            - reset(): Reset to default configuration

    Raises:
        ConfigurationError: If configuration is invalid or incompatible
        DatasetError: If evaluation dataset cannot be loaded
        FrameworkError: If framework override fails

    Examples:
        Basic usage with automatic framework override:

        >>> @traigent.optimize(
        ...     objectives=["accuracy"],
        ...     evaluation={"eval_dataset": "qa_test.jsonl"},
        ...     configuration_space={
        ...         "model": ["gpt-3.5-turbo", "gpt-4"],
        ...         "temperature": [0.1, 0.5, 0.9]
        ...     }
        ... )
        ... def answer_question(question: str) -> str:
        ...     # Your existing code - no changes needed!
        ...     llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
        ...     return llm.complete(question)

        Multi-objective optimization with constraints:

        >>> @traigent.optimize(
        ...     objectives=["accuracy", "cost"],
        ...     evaluation={"eval_dataset": "support_tickets.jsonl"},
        ...     configuration_space={
        ...         "model": ["gpt-3.5-turbo", "gpt-4"],
        ...         "temperature": (0.0, 1.0),
        ...         "max_tokens": [100, 500, 1000]
        ...     },
        ...     constraints=[
        ...         lambda cfg: cfg["max_tokens"] <= 500 if cfg["model"] == "gpt-4" else True,
        ...         lambda cfg, metrics: metrics.get("cost", 0) <= 0.10
        ...     ]
        ... )
        ... def handle_ticket(ticket: str) -> str:
        ...     return support_chain.run(ticket)

        Offline local optimization:

        >>> @traigent.optimize(
        ...     objectives=["accuracy", "safety"],
        ...     evaluation={"eval_dataset": "medical_qa.jsonl"},
        ...     execution={
        ...         "offline": True,
        ...         "local_storage_path": "./my_optimizations",
        ...         "minimal_logging": True,
        ...     },
        ...     configuration_space={
        ...         "model": ["gpt-4", "claude-haiku-4-5-20251001"],
        ...         "temperature": [0.1, 0.3, 0.5],
        ...         "safety_filter": ["strict", "moderate"]
        ...     }
        ... )
        ... def medical_assistant(patient_query: str) -> str:  # doctest: +SKIP
        ...     # Data never leaves your infrastructure
        ...     return process_medical_query(patient_query)

    Note:
        Set TRAIGENT_DISABLED=1 environment variable to disable all optimization.
        The decorator becomes a pass-through that returns the original function.
    """
    # Check if Traigent is disabled via environment variable
    # When disabled, @optimize becomes a no-op that returns the original function
    if is_traigent_disabled():
        logger.debug("Traigent disabled via TRAIGENT_DISABLED env var, returning no-op")

        def passthrough_decorator(
            func: Callable[_P, _R],
        ) -> OptimizedFunction[_P, _R]:
            """No-op decorator that returns the function unchanged when Traigent is disabled."""
            # Cast so the disabled path satisfies the declared return type without
            # wrapping: the caller only cares that the result is callable with the
            # same signature and exposes .optimize().
            return func  # type: ignore[return-value]

        return passthrough_decorator

    legacy_args = _parse_legacy_args(legacy)
    max_trials_explicit = runtime_overrides.get("max_trials") is not None
    if legacy_args is not None and legacy_args.max_trials is not None:
        max_trials_explicit = True

    combined_settings = dict(_OPTIMIZE_DEFAULTS)
    if "execution_mode" in _GLOBAL_CONFIG:
        combined_settings["execution_mode"] = _GLOBAL_CONFIG["execution_mode"]
    provided_sources: dict[str, str] = {}
    record_option = _build_settings_recorder(combined_settings, provided_sources)

    if legacy_args:
        for key, value in legacy_args.iter_known_values():
            record_option(key, value, "legacy arguments")

    preset_strategy_name, runtime_overrides = _resolve_strategy_argument(
        strategy=strategy,
        strategy_params=strategy_params,
        runtime_overrides=runtime_overrides,
    )
    strategy_preset = (
        normalize_strategy_preset(preset_strategy_name, strategy_params)
        if preset_strategy_name is not None
        else None
    )

    direct_inputs = {
        "objectives": objectives,
        "configuration_space": configuration_space,
        "experiment_name": experiment_name,
        "default_config": default_config,
        "warm_start_from": warm_start_from,
        "constraints": constraints,
        "safety_constraints": safety_constraints,
        "tvl_spec": tvl_spec,
        "tvl_environment": tvl_environment,
        "tvl": tvl,
        "evaluation": evaluation,
        "injection": injection,
        "effectuation": effectuation,
        "execution": execution,
        "evaluator": evaluator,
        "mock": mock,
        "algorithm": algorithm,
        "offline": offline,
        "agents": agents,
        "agent_prefixes": agent_prefixes,
        "agent_measures": agent_measures,
        "global_measures": global_measures,
        "auto_load_best": auto_load_best,
        "load_from": load_from,
        "config_id": config_id,
        "best_config_source": best_config_source,
        "best_config_strict": best_config_strict,
        "best_config_cache_dir": best_config_cache_dir,
        "best_config_cache_ttl_seconds": best_config_cache_ttl_seconds,
        "best_config_stale_ok_ttl_seconds": best_config_stale_ok_ttl_seconds,
        "enable_auto_load_dev_logs": enable_auto_load_dev_logs,
    }
    for key, value in direct_inputs.items():
        record_option(key, value, "optimize parameter")

    combined_runtime_overrides = _process_runtime_overrides(
        runtime_overrides, legacy_args, record_option
    )

    # Extract inline parameter definitions and validate remaining overrides
    inline_params, remaining_overrides = _extract_inline_params(
        combined_runtime_overrides
    )
    _validate_runtime_overrides(remaining_overrides)
    combined_runtime_overrides = remaining_overrides

    eval_dataset = combined_settings["eval_dataset"]
    objectives = combined_settings["objectives"]
    configuration_space = combined_settings["configuration_space"]
    default_config = combined_settings["default_config"]
    constraints = combined_settings["constraints"]
    safety_constraints = combined_settings["safety_constraints"]
    if safety_constraints:
        raise NotImplementedError(
            "safety_constraints are not yet implemented. "
            "Statistical chance-constraints are on the roadmap — track progress at "
            "https://github.com/Traigent/traigent-smartopt/issues/26"
        )

    # Process ConfigSpace constraints
    config_space_constraints, config_space_var_names, _ = (
        _process_config_space_constraints(configuration_space, constraints)
    )
    constraint_scope_var_names = _build_constraint_scope_var_names(
        configuration_space,
        inline_params,
        config_space_var_names,
    )

    raw_configuration_space = configuration_space

    # Normalize configuration_space and merge defaults
    configuration_space, default_config, constraints = (
        _normalize_config_space_and_defaults(
            configuration_space,
            inline_params,
            default_config,
            config_space_constraints,
            constraints,
        )
    )
    _validate_constraint_satisfiability(
        raw_configuration_space,
        inline_params,
        constraints,
    )

    # Note: Constraint normalization is deferred until after TVL artifact application
    # to allow merging of user constraints with TVL constraints
    injection_mode = combined_settings["injection_mode"]
    config_param = combined_settings["config_param"]
    auto_override_frameworks = combined_settings["auto_override_frameworks"]
    framework_targets = combined_settings["framework_targets"]
    effectuation = combined_settings["effectuation"]
    algorithm_value = validate_algorithm_name(combined_settings["algorithm"])
    offline_value = combined_settings["offline"]
    if not isinstance(offline_value, bool):
        raise TypeError("offline must be a bool")
    evaluator_value = combined_settings["evaluator"]
    legacy_execution_options = {
        key: combined_settings[key]
        for key in _LEGACY_EXECUTION_OPTION_KEYS
        if key in combined_settings
    }
    local_storage_path = combined_settings["local_storage_path"]
    minimal_logging = combined_settings["minimal_logging"]
    parallel_config = combined_settings["parallel_config"]
    max_total_examples = combined_settings["max_total_examples"]
    samples_include_pruned = combined_settings["samples_include_pruned"]
    mock_mode_config = combined_settings["mock_mode_config"]
    custom_evaluator = combined_settings["custom_evaluator"]
    scoring_function = combined_settings["scoring_function"]
    metric_functions = combined_settings["metric_functions"]
    tvl_spec_value = combined_settings["tvl_spec"]
    tvl_environment_value = combined_settings["tvl_environment"]
    # Multi-agent configuration
    agents_config = combined_settings["agents"]
    agent_prefixes_config = combined_settings["agent_prefixes"]
    agent_measures_config = combined_settings["agent_measures"]
    global_measures_config = combined_settings["global_measures"]
    # Config persistence
    auto_load_best_config = combined_settings["auto_load_best"]
    load_from_config = combined_settings["load_from"]
    config_id_value = combined_settings["config_id"]
    best_config_source_value = combined_settings["best_config_source"]
    best_config_strict_value = combined_settings["best_config_strict"]
    best_config_cache_dir_value = combined_settings["best_config_cache_dir"]
    best_config_cache_ttl_seconds_value = combined_settings[
        "best_config_cache_ttl_seconds"
    ]
    best_config_stale_ok_ttl_seconds_value = combined_settings[
        "best_config_stale_ok_ttl_seconds"
    ]
    enable_auto_load_dev_logs_value = combined_settings["enable_auto_load_dev_logs"]
    # Optimizer limits
    max_trials_value = combined_settings["max_trials"]
    if max_trials_value is not None and max_trials_value <= 0:
        raise ValueError("max_trials must be a positive integer")
    # Experiment display name (decorator > env var > func.__name__ at decoration time)
    experiment_name_value = combined_settings["experiment_name"]
    # Warm-start: prior experiment id to seed this run (empty string -> None).
    warm_start_from_value: str | None = combined_settings.get("warm_start_from") or None
    # Tuned variable auto-detection
    auto_detect_tvars_value = combined_settings["auto_detect_tvars"]
    auto_detect_tvars_mode_value = combined_settings["auto_detect_tvars_mode"]
    auto_detect_tvars_min_confidence_value = combined_settings[
        "auto_detect_tvars_min_confidence"
    ]
    auto_detect_tvars_include_value = combined_settings["auto_detect_tvars_include"]
    auto_detect_tvars_exclude_value = combined_settings["auto_detect_tvars_exclude"]

    effective_auto_detect_tvars_mode = _resolve_auto_detect_tvars_mode(
        auto_detect_tvars_value, auto_detect_tvars_mode_value
    )
    auto_detect_tvars_min_confidence = _coerce_auto_detect_tvars_min_confidence(
        auto_detect_tvars_min_confidence_value
    )
    auto_detect_tvars_include = _coerce_auto_detect_tvars_name_filter(
        auto_detect_tvars_include_value, "auto_detect_tvars_include"
    )
    auto_detect_tvars_exclude = _coerce_auto_detect_tvars_name_filter(
        auto_detect_tvars_exclude_value, "auto_detect_tvars_exclude"
    )

    defaults = dict(_OPTIMIZE_DEFAULTS)

    evaluation_bundle = _coerce_bundle(
        combined_settings["evaluation"], EvaluationOptions, "evaluation"
    )
    injection_bundle = _coerce_bundle(
        combined_settings["injection"], InjectionOptions, "injection"
    )
    execution_bundle = _coerce_bundle(
        combined_settings["execution"], ExecutionOptions, "execution"
    )
    mock_bundle = _coerce_bundle(combined_settings["mock"], MockModeOptions, "mock")
    tvl_bundle = _coerce_bundle(combined_settings["tvl"], TVLOptions, "tvl")

    # Resolve options from bundles
    eval_dataset, custom_evaluator, scoring_function, metric_functions = (
        _resolve_evaluation_bundle_options(
            evaluation_bundle,
            eval_dataset,
            custom_evaluator,
            scoring_function,
            metric_functions,
            defaults,
        )
    )

    # Validate custom_evaluator signature early to catch interface mismatches
    if custom_evaluator is not None:
        _validate_custom_evaluator_signature(custom_evaluator)

    (
        injection_mode,
        config_param,
        auto_override_frameworks,
        framework_targets,
        effectuation,
    ) = _resolve_injection_bundle_options(
        injection_bundle,
        injection_mode,
        config_param,
        auto_override_frameworks,
        framework_targets,
        effectuation,
        defaults,
    )

    base_execution_options = ResolvedExecutionOptions(
        algorithm=algorithm_value,
        offline=offline_value,
        evaluator=evaluator_value,
        local_storage_path=local_storage_path,
        minimal_logging=minimal_logging,
        parallel_config=parallel_config,
        max_total_examples=max_total_examples,
        samples_include_pruned=samples_include_pruned,
        legacy_options=legacy_execution_options,
    )
    resolved_execution = _resolve_execution_bundle_options(
        execution_bundle,
        base_execution_options,
        defaults,
    )
    algorithm_value = validate_algorithm_name(resolved_execution.algorithm)
    offline_value = resolved_execution.offline
    if not isinstance(offline_value, bool):
        raise TypeError("offline must be a bool")
    evaluator_value = resolved_execution.evaluator
    local_storage_path = resolved_execution.local_storage_path
    minimal_logging = resolved_execution.minimal_logging
    parallel_config = resolved_execution.parallel_config
    max_total_examples = resolved_execution.max_total_examples
    samples_include_pruned = resolved_execution.samples_include_pruned
    legacy_execution_options = resolved_execution.legacy_options
    external_service_evaluator = _resolve_external_service_evaluator(
        evaluator_value,
        legacy_execution_options,
    )
    hybrid_api_options = (
        external_service_evaluator.hybrid_api
        if external_service_evaluator is not None
        else None
    )
    hybrid_api_endpoint = hybrid_api_options.endpoint if hybrid_api_options else None
    tunable_id = hybrid_api_options.tunable_id if hybrid_api_options else None
    hybrid_api_transport = hybrid_api_options.transport if hybrid_api_options else None
    hybrid_api_transport_type = (
        hybrid_api_options.transport_type if hybrid_api_options else "auto"
    )
    hybrid_api_batch_size = hybrid_api_options.batch_size if hybrid_api_options else 1
    hybrid_api_batch_parallelism = (
        hybrid_api_options.batch_parallelism if hybrid_api_options else 1
    )
    hybrid_api_keep_alive = (
        hybrid_api_options.keep_alive if hybrid_api_options else True
    )
    hybrid_api_heartbeat_interval = (
        hybrid_api_options.heartbeat_interval if hybrid_api_options else 30.0
    )
    hybrid_api_timeout = hybrid_api_options.timeout if hybrid_api_options else None
    hybrid_api_auth_header = (
        hybrid_api_options.auth_header if hybrid_api_options else None
    )
    hybrid_api_auto_discover_tvars = (
        hybrid_api_options.auto_discover_tvars if hybrid_api_options else False
    )

    tvl_options = _resolve_tvl_options(
        tvl_spec_value, tvl_environment_value, tvl_bundle
    )
    (
        configuration_space,
        objectives,
        constraints,
        default_config,
        eval_dataset,
        promotion_gate,
    ) = _apply_tvl_options_if_present(
        tvl_options,
        configuration_space,
        objectives,
        constraints,
        default_config,
        eval_dataset,
        combined_runtime_overrides,
    )
    objectives, constraints = _apply_strategy_preset_to_options(
        strategy_preset=strategy_preset,
        objectives=objectives,
        constraints=constraints,
    )
    constraint_scope_var_names = _augment_constraint_scope_var_names(
        constraint_scope_var_names, configuration_space
    )

    if samples_include_pruned is None:
        samples_include_pruned = True

    if mock_bundle:
        mock_mode_config = _resolve_option(
            "mock_mode_config",
            mock_mode_config,
            mock_bundle.model_dump(exclude_none=True),
            defaults,
        )

    _validate_objectives(objectives)
    execution_policy = _resolve_execution_policy_from_options(
        algorithm=algorithm_value,
        offline=offline_value,
        legacy_options=legacy_execution_options,
        source_hint="optimize",
    )
    runtime_execution_mode = _runtime_execution_mode_for_policy(
        execution_policy,
        external_service_evaluator,
    )
    runtime_algorithm_value = _runtime_algorithm_for_policy(
        execution_policy,
        external_service_evaluator,
    )

    def decorator(func: Callable[_P, _R]) -> OptimizedFunction[_P, _R]:
        """Actual decorator function.

        Args:
            func: Function to optimize

        Returns:
            OptimizedFunction wrapper
        """
        _check_deprecated_objective_kwargs(combined_runtime_overrides)

        logger.debug(f"Decorating function {func.__name__} with @traigent.optimize")

        resolved_schema = _resolve_objective_schema(objectives)

        requested_execution_mode = legacy_execution_options.get("execution_mode")
        actual_execution_mode = runtime_execution_mode.value

        actual_injection_mode = _resolve_injection_mode_enum(injection_mode)

        _log_execution_mode_warnings(
            runtime_execution_mode,
            actual_execution_mode,
            local_storage_path,
            minimal_logging,
        )

        user_parallel_config = coerce_parallel_config(parallel_config)
        combined_parallel_config, parallel_sources = merge_parallel_configs(
            [(user_parallel_config, "decorator")]
        )
        if parallel_sources:
            logger.debug(
                "Decorator parallel configuration sources: %s", parallel_sources
            )

        # Normalize Constraint/BoolExpr objects to callables for the optimizer
        # This is done after TVL artifact application to include all constraints
        normalized_constraints: list[Callable[..., bool]] | None = None
        if constraints:
            from traigent.api.constraints import normalize_constraints

            normalized_constraints = normalize_constraints(
                constraints, constraint_scope_var_names
            )

        # Remove keys from runtime overrides that are passed explicitly below
        # to avoid "got multiple values for keyword argument" errors.
        # TVL budget application can inject max_trials into runtime_overrides
        # (via _apply_tvl_artifact), but we pass it explicitly as well.
        combined_runtime_overrides.pop("algorithm", None)
        combined_runtime_overrides.pop("max_trials", None)

        resolved_configuration_space = configuration_space
        resolved_default_config = default_config

        # Tuned variable auto-detection:
        # - suggest mode logs candidates
        # - apply mode auto-materializes configuration_space
        if (
            effective_auto_detect_tvars_mode != "off"
            and not resolved_configuration_space
        ):
            detected_config = _suggest_detected_tvars(
                func,
                mode=effective_auto_detect_tvars_mode,
                min_confidence=auto_detect_tvars_min_confidence,
                include=auto_detect_tvars_include,
                exclude=auto_detect_tvars_exclude,
            )
            if effective_auto_detect_tvars_mode == "apply" and detected_config:
                resolved_configuration_space, detected_defaults = (
                    normalize_configuration_space(detected_config)
                )
                if detected_defaults:
                    resolved_default_config = {
                        **detected_defaults,
                        **(resolved_default_config or {}),
                    }

        # #1372: loudly warn if CONTEXT-mode tuned knobs shadow the wrapped
        # function's own parameters (silent no-op sweep otherwise).
        _warn_context_mode_param_shadowing(
            func,
            resolved_configuration_space,
            actual_injection_mode,
            config_param,
        )

        optimized_func: OptimizedFunction[_P, _R] = OptimizedFunction(  # type: ignore[assignment]
            func=func,
            eval_dataset=eval_dataset,
            objectives=resolved_schema,
            configuration_space=resolved_configuration_space,
            algorithm=runtime_algorithm_value,
            default_config=resolved_default_config,
            constraints=normalized_constraints,
            safety_constraints=safety_constraints,
            injection_mode=actual_injection_mode,
            config_param=config_param,
            auto_override_frameworks=auto_override_frameworks,
            framework_targets=framework_targets,
            effectuation=bool(effectuation),
            execution_mode=runtime_execution_mode,
            hybrid_api_endpoint=hybrid_api_endpoint,
            tunable_id=tunable_id,
            hybrid_api_transport=hybrid_api_transport,
            hybrid_api_transport_type=hybrid_api_transport_type,
            hybrid_api_batch_size=hybrid_api_batch_size,
            hybrid_api_batch_parallelism=hybrid_api_batch_parallelism,
            hybrid_api_keep_alive=hybrid_api_keep_alive,
            hybrid_api_heartbeat_interval=hybrid_api_heartbeat_interval,
            hybrid_api_timeout=hybrid_api_timeout,
            hybrid_api_auth_header=hybrid_api_auth_header,
            hybrid_api_auto_discover_tvars=hybrid_api_auto_discover_tvars,
            local_storage_path=local_storage_path,
            minimal_logging=minimal_logging,
            max_total_examples=max_total_examples,
            samples_include_pruned=samples_include_pruned,
            parallel_config=combined_parallel_config,
            mock_mode_config=mock_mode_config,
            custom_evaluator=custom_evaluator,
            scoring_function=scoring_function,
            metric_functions=metric_functions,
            requested_execution_mode=requested_execution_mode,
            execution_policy=execution_policy,
            # Multi-agent configuration
            agents=agents_config,
            agent_prefixes=agent_prefixes_config,
            agent_measures=agent_measures_config,
            global_measures=global_measures_config,
            # Config persistence
            auto_load_best=auto_load_best_config,
            load_from=load_from_config,
            config_id=config_id_value,
            best_config_source=best_config_source_value,
            best_config_strict=best_config_strict_value,
            best_config_cache_dir=best_config_cache_dir_value,
            best_config_cache_ttl_seconds=best_config_cache_ttl_seconds_value,
            best_config_stale_ok_ttl_seconds=best_config_stale_ok_ttl_seconds_value,
            enable_auto_load_dev_logs=enable_auto_load_dev_logs_value,
            # TVL promotion gate for statistical best-config selection
            promotion_gate=promotion_gate,
            # Advisory strategy preset metadata/selection.
            strategy_preset=strategy_preset,
            # Optimizer limits (extracted from combined_settings)
            max_trials=max_trials_value,
            _max_trials_explicit=max_trials_explicit,
            # Experiment display name (overrides func.__name__ in portal/storage)
            experiment_name=experiment_name_value,
            # Warm-start: seed this run from a prior experiment's learned configs.
            warm_start_from=warm_start_from_value,
            # Guided-generation defaults (consumed by optimize_with_guidance)
            prompt_rewrite=prompt_rewrite,
            grow_dataset=grow_dataset,
            skill_train=skill_train,
            **combined_runtime_overrides,
        )
        optimized_func.execution_policy = execution_policy
        optimized_func.external_service_evaluator = external_service_evaluator
        optimized_func.hybrid_api_options = hybrid_api_options
        optimized_func.offline = execution_policy.offline

        effective_name = experiment_name_value or func.__name__
        logger.info(
            f"Created optimizable function: {func.__name__} (experiment_name={effective_name!r})"
        )

        # cast so mypy sees OptimizedFunction[_P, _R] rather than
        # OptimizedFunction[Any, Any] (the constructor takes Callable[..., Any]
        # to stay compatible with diverse callers; the generic parameters are
        # threaded through the factory, not the __init__ signature).
        return cast(OptimizedFunction[_P, _R], optimized_func)

    return decorator
