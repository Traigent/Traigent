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
    ...         "model": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
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
import warnings
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from traigent.api.config_space import ConfigSpace
    from traigent.api.constraints import BoolExpr, Constraint
    from traigent.api.safety import CompoundSafetyConstraint, SafetyConstraint

from pydantic import BaseModel, ConfigDict, field_validator

from traigent.api.functions import _GLOBAL_CONFIG
from traigent.api.parameter_ranges import (
    ParameterRange,
    is_inline_param_definition,
    normalize_configuration_space,
)
from traigent.api.types import AgentDefinition
from traigent.config.parallel import (
    ParallelConfig,
    coerce_parallel_config,
    merge_parallel_configs,
)
from traigent.config.types import (
    ExecutionMode,
    InjectionMode,
    is_traigent_disabled,
    resolve_execution_mode,
)
from traigent.core.objectives import (
    ObjectiveSchema,
    create_default_objectives,
    normalize_objectives,
)
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset
from traigent.tvl.options import TVLOptions
from traigent.tvl.promotion_gate import PromotionGate
from traigent.tvl.spec_loader import TVLSpecArtifact, load_tvl_spec
from traigent.utils.exceptions import TVLValidationError, ValidationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationOptions(BaseModel):
    """Grouped evaluation settings used by the optimize decorator."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    eval_dataset: str | list[str] | Dataset | None = None
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


class ExecutionOptions(BaseModel):
    """Execution and orchestration preferences for optimization runs.

    Attributes:
        execution_mode: Execution mode (edge_analytics, cloud, local, hybrid).
        local_storage_path: Path for local result storage.
        minimal_logging: Whether to minimize logging output.
        parallel_config: Configuration for parallel execution.
        privacy_enabled: Whether to enable privacy-preserving mode.
        max_total_examples: Maximum total examples across all trials.
        samples_include_pruned: Whether to include pruned trials in sample count.
        reps_per_trial: Number of repetitions per configuration for statistical stability.
            Running multiple repetitions helps account for LLM non-determinism.
            Default is 1 (no repetition). Set to 3-5 for noisy evaluations.
        reps_aggregation: How to aggregate metrics across repetitions.
            Options: "mean" (default), "median", "min", "max".
        runtime: Runtime to execute trials in ("python" or "node").
            When set to "node", trials are executed in a Node.js subprocess.
        js_module: Path to the JS module containing the trial function.
            Required when runtime="node".
        js_function: Name of the exported function to call in the JS module.
            Default is "runTrial".
        js_timeout: Timeout for JS trial execution in seconds.
            Default is 300 (5 minutes).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    execution_mode: str = "edge_analytics"
    local_storage_path: str | None = None
    minimal_logging: bool = True
    parallel_config: ParallelConfig | dict[str, Any] | None = None
    privacy_enabled: bool | None = None
    max_total_examples: int | None = None
    samples_include_pruned: bool = True
    reps_per_trial: int = 1
    reps_aggregation: str = "mean"
    # JS Bridge options
    runtime: str = "python"
    js_module: str | None = None
    js_function: str = "runTrial"
    js_timeout: float = 300.0
    js_parallel_workers: int = 1
    # Hybrid API options
    hybrid_api_endpoint: str | None = None
    tunable_id: str | None = None
    hybrid_api_transport: Any | None = None
    hybrid_api_transport_type: str = "auto"
    hybrid_api_batch_size: int = 1
    hybrid_api_batch_parallelism: int = 1
    hybrid_api_keep_alive: bool = True
    hybrid_api_heartbeat_interval: float = 30.0
    hybrid_api_timeout: float | None = None
    hybrid_api_auth_header: str | None = None
    hybrid_api_auto_discover_tvars: bool = False


class MockModeOptions(BaseModel):
    """Fine-grained configuration for mock mode behaviour."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    enabled: bool = True
    override_evaluator: bool = True
    base_accuracy: float = 0.75
    variance: float = 0.25


BundleModel = TypeVar("BundleModel", bound=BaseModel)


def _coerce_bundle(
    value: Any, model_cls: type[BundleModel], parameter_name: str
) -> BundleModel | None:
    if value is None:
        return None
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
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


_DEFAULT_SENTINEL = object()
_AUTO_DETECT_TVARS_MODES = frozenset({"off", "suggest", "apply"})

_OPTIMIZE_DEFAULTS: dict[str, Any] = {
    "eval_dataset": None,
    "objectives": None,
    "configuration_space": None,
    "default_config": None,
    "constraints": None,
    "safety_constraints": None,
    "tvl_spec": None,
    "tvl_environment": None,
    "tvl": None,
    "injection_mode": InjectionMode.CONTEXT,
    "config_param": None,
    "auto_override_frameworks": False,  # Requires traigent-integrations plugin
    "framework_targets": None,
    "execution_mode": "edge_analytics",
    "hybrid_api_endpoint": None,
    "tunable_id": None,
    "hybrid_api_transport": None,
    "hybrid_api_transport_type": "auto",
    "hybrid_api_batch_size": 1,
    "hybrid_api_batch_parallelism": 1,
    "hybrid_api_keep_alive": True,
    "hybrid_api_heartbeat_interval": 30.0,
    "hybrid_api_timeout": None,
    "hybrid_api_auth_header": None,
    "hybrid_api_auto_discover_tvars": False,
    "local_storage_path": None,
    "minimal_logging": True,
    "parallel_config": None,
    "privacy_enabled": None,
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
    "algorithm": "random",
    "max_trials": 50,
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
    # Tuned variable auto-detection
    "auto_detect_tvars": False,  # Log suggestions when no configuration_space is set
    "auto_detect_tvars_mode": None,  # "off" | "suggest" | "apply"
    "auto_detect_tvars_min_confidence": "medium",  # "high" | "medium" | "low"
    "auto_detect_tvars_include": None,  # Optional names to include
    "auto_detect_tvars_exclude": None,  # Optional names to exclude
}

_DIRECT_OPTION_KEYS = frozenset(_OPTIMIZE_DEFAULTS.keys())
_REMOVED_PARAMETERS = frozenset(
    ("auto_optimize", "trigger", "batch_size", "parallel_trials")
)
_ALLOWED_RUNTIME_OVERRIDE_KEYS = frozenset(
    (
        "budget_limit",
        "budget_metric",
        "budget_include_pruned",
        "plateau_window",
        "plateau_epsilon",
        "cost_limit",
        "cost_approved",
        "tie_breakers",
        "tvl_parameter_agents",
    )
)


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

    eval_dataset: str | list[str] | Dataset | None = None
    objectives: list[str] | ObjectiveSchema | None = None
    configuration_space: dict[str, Any] | None = None
    default_config: dict[str, Any] | None = None
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
    mock: MockModeOptions | dict[str, Any] | None = None
    algorithm: str | None = None
    max_trials: int | None = None
    # Multi-agent configuration
    agents: dict[str, AgentDefinition] | None = None
    agent_prefixes: list[str] | None = None
    agent_measures: dict[str, list[str]] | None = None
    global_measures: list[str] | None = None
    # Config persistence
    auto_load_best: bool | None = None
    load_from: str | None = None
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
            ("configuration_space", self.configuration_space),
            ("default_config", self.default_config),
            ("constraints", self.constraints),
            ("safety_constraints", self.safety_constraints),
            ("tvl_spec", self.tvl_spec),
            ("tvl_environment", self.tvl_environment),
            ("tvl", self.tvl),
            ("injection_mode", self.injection_mode),
            ("config_param", self.config_param),
            ("auto_override_frameworks", self.auto_override_frameworks),
            ("framework_targets", self.framework_targets),
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
            ("mock", self.mock),
            ("algorithm", self.algorithm),
            ("max_trials", self.max_trials),
            ("agents", self.agents),
            ("agent_prefixes", self.agent_prefixes),
            ("agent_measures", self.agent_measures),
            ("global_measures", self.global_measures),
            ("auto_load_best", self.auto_load_best),
            ("load_from", self.load_from),
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
            raise TypeError(
                f"{key} parameter has been removed. Use supported arguments such as "
                "parallel_config or ExecutionOptions instead."
            )
        existing_source = provided_sources.get(key)
        if existing_source is not None:
            existing_value = combined_settings[key]
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
    """Normalize deprecated runtime override aliases."""
    if "strategy" not in overrides:
        return dict(overrides)

    normalized = dict(overrides)
    strategy_value = normalized.pop("strategy")
    existing_algorithm = normalized.get("algorithm")
    if (
        existing_algorithm is not None
        and strategy_value is not None
        and existing_algorithm != strategy_value
    ):
        raise TypeError(
            "Conflicting optimization selector: received both "
            f"'algorithm={existing_algorithm}' and 'strategy={strategy_value}'. "
            "Use only 'algorithm'."
        )

    warnings.warn(
        "'strategy' is deprecated; use 'algorithm' instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    normalized["algorithm"] = (
        existing_algorithm if existing_algorithm is not None else strategy_value
    )
    return normalized


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


def _build_constraint_scope_var_names(
    raw_configuration_space: dict[str, Any] | ConfigSpace | None,
    inline_params: dict[str, Any],
    config_space_var_names: dict[int, str] | None,
) -> dict[int, str] | None:
    """Build var-name scope used for compile-time constraint validation."""
    scope_var_names: dict[int, str] = dict(config_space_var_names or {})

    sources: list[dict[str, Any]] = []
    if isinstance(raw_configuration_space, dict):
        sources.append(raw_configuration_space)
    elif hasattr(raw_configuration_space, "tvars"):
        tvars = getattr(raw_configuration_space, "tvars", None)
        if isinstance(tvars, dict):
            sources.append(tvars)
    if inline_params:
        sources.append(inline_params)

    for source in sources:
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
    defaults: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    """Resolve injection options from bundle."""
    if injection_bundle is None:
        return (
            injection_mode,
            config_param,
            auto_override_frameworks,
            framework_targets,
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
    )


@dataclass
class JSRuntimeConfig:
    """Configuration for JS runtime execution."""

    runtime: str = "python"
    js_module: str | None = None
    js_function: str = "runTrial"
    js_timeout: float = 300.0
    js_parallel_workers: int = 1

    @property
    def is_js_runtime(self) -> bool:
        """Return True if this is a JS runtime configuration.

        Returns:
            True if runtime is 'node', False otherwise.
        """
        return self.runtime == "node"


@dataclass(slots=True)
class ResolvedExecutionOptions:
    """Resolved execution options after merging direct and bundled settings."""

    execution_mode: Any
    hybrid_api_endpoint: Any
    tunable_id: Any
    hybrid_api_transport: Any
    hybrid_api_transport_type: Any
    hybrid_api_batch_size: Any
    hybrid_api_batch_parallelism: Any
    hybrid_api_keep_alive: Any
    hybrid_api_heartbeat_interval: Any
    hybrid_api_timeout: Any
    hybrid_api_auth_header: Any
    hybrid_api_auto_discover_tvars: Any
    local_storage_path: Any
    minimal_logging: Any
    parallel_config: Any
    privacy_enabled: Any
    max_total_examples: Any
    samples_include_pruned: Any
    js_runtime_config: JSRuntimeConfig | None


def _resolve_execution_bundle_options(
    execution_bundle: ExecutionOptions | None,
    base_options: ResolvedExecutionOptions,
    defaults: dict[str, Any],
) -> ResolvedExecutionOptions:
    """Resolve execution options from bundle and validate enterprise features."""
    if execution_bundle is None:
        return base_options

    # Validate enterprise-gated features
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

    # Build JS runtime config if runtime is "node"
    js_runtime_config = None
    if execution_bundle.runtime == "node":
        if not execution_bundle.js_module:
            raise ValueError(
                "js_module is required when runtime='node'. "
                "Specify the path to your JS module containing the trial function."
            )
        js_runtime_config = JSRuntimeConfig(
            runtime=execution_bundle.runtime,
            js_module=execution_bundle.js_module,
            js_function=execution_bundle.js_function,
            js_timeout=execution_bundle.js_timeout,
            js_parallel_workers=execution_bundle.js_parallel_workers,
        )
    elif execution_bundle.runtime not in ("python", "node"):
        raise ValueError(
            f"Invalid runtime '{execution_bundle.runtime}'. "
            "Supported values are 'python' (default) or 'node' (JavaScript)."
        )

    return ResolvedExecutionOptions(
        execution_mode=_resolve_option(
            "execution_mode",
            base_options.execution_mode,
            execution_bundle.execution_mode,
            defaults,
        ),
        hybrid_api_endpoint=_resolve_option(
            "hybrid_api_endpoint",
            base_options.hybrid_api_endpoint,
            execution_bundle.hybrid_api_endpoint,
            defaults,
        ),
        tunable_id=_resolve_option(
            "tunable_id",
            base_options.tunable_id,
            execution_bundle.tunable_id,
            defaults,
        ),
        hybrid_api_transport=_resolve_option(
            "hybrid_api_transport",
            base_options.hybrid_api_transport,
            execution_bundle.hybrid_api_transport,
            defaults,
        ),
        hybrid_api_transport_type=_resolve_option(
            "hybrid_api_transport_type",
            base_options.hybrid_api_transport_type,
            execution_bundle.hybrid_api_transport_type,
            defaults,
        ),
        hybrid_api_batch_size=_resolve_option(
            "hybrid_api_batch_size",
            base_options.hybrid_api_batch_size,
            execution_bundle.hybrid_api_batch_size,
            defaults,
        ),
        hybrid_api_batch_parallelism=_resolve_option(
            "hybrid_api_batch_parallelism",
            base_options.hybrid_api_batch_parallelism,
            execution_bundle.hybrid_api_batch_parallelism,
            defaults,
        ),
        hybrid_api_keep_alive=_resolve_option(
            "hybrid_api_keep_alive",
            base_options.hybrid_api_keep_alive,
            execution_bundle.hybrid_api_keep_alive,
            defaults,
        ),
        hybrid_api_heartbeat_interval=_resolve_option(
            "hybrid_api_heartbeat_interval",
            base_options.hybrid_api_heartbeat_interval,
            execution_bundle.hybrid_api_heartbeat_interval,
            defaults,
        ),
        hybrid_api_timeout=_resolve_option(
            "hybrid_api_timeout",
            base_options.hybrid_api_timeout,
            execution_bundle.hybrid_api_timeout,
            defaults,
        ),
        hybrid_api_auth_header=_resolve_option(
            "hybrid_api_auth_header",
            base_options.hybrid_api_auth_header,
            execution_bundle.hybrid_api_auth_header,
            defaults,
        ),
        hybrid_api_auto_discover_tvars=_resolve_option(
            "hybrid_api_auto_discover_tvars",
            base_options.hybrid_api_auto_discover_tvars,
            execution_bundle.hybrid_api_auto_discover_tvars,
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
        privacy_enabled=_resolve_option(
            "privacy_enabled",
            base_options.privacy_enabled,
            execution_bundle.privacy_enabled,
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
        js_runtime_config=js_runtime_config,
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


# Injection modes that are incompatible with JS runtime
_JS_INCOMPATIBLE_INJECTION_MODES = frozenset(
    {
        InjectionMode.CONTEXT,  # Uses Python's contextvars
        InjectionMode.SEAMLESS,  # Modifies Python source code
    }
)


def _validate_js_runtime_injection_mode(
    js_runtime_config: JSRuntimeConfig | None,
    injection_mode: str | InjectionMode,
) -> None:
    """Validate that injection mode is compatible with JS runtime.

    When runtime='node', Python-specific injection modes are not supported
    because the trial config is passed directly to the JS function via the
    NDJSON protocol.

    Args:
        js_runtime_config: JS runtime configuration (None if runtime='python')
        injection_mode: The injection mode being used

    Raises:
        ValueError: If injection mode is incompatible with JS runtime
    """
    if js_runtime_config is None or not js_runtime_config.is_js_runtime:
        return

    # Normalize to enum for comparison
    if isinstance(injection_mode, InjectionMode):
        mode_enum = injection_mode
    elif injection_mode in [m.value for m in InjectionMode]:
        mode_enum = InjectionMode(injection_mode)
    else:
        mode_enum = None

    if mode_enum in _JS_INCOMPATIBLE_INJECTION_MODES:
        raise ValueError(
            f"injection_mode='{mode_enum.value if mode_enum else injection_mode}' "
            f"is not compatible with runtime='node'. "
            f"When using JavaScript runtime, config is passed directly to the JS function "
            f"via the trial protocol. Use injection_mode='parameter' or omit it."
        )


def _resolve_execution_mode_enum(
    execution_mode: str | ExecutionMode,
    privacy_enabled: bool | None,
) -> tuple[ExecutionMode, bool | None]:
    """Resolve execution mode enum and handle privacy deprecation."""
    try:
        execution_mode_enum = resolve_execution_mode(execution_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from None

    if execution_mode_enum is ExecutionMode.PRIVACY:
        logger.warning(
            "execution_mode='privacy' is deprecated. Use execution_mode='hybrid' "
            "with privacy_enabled=True. Mapping automatically."
        )
        execution_mode_enum = ExecutionMode.HYBRID
        if privacy_enabled is None:
            privacy_enabled = True

    return execution_mode_enum, privacy_enabled


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
    """Resolve actual execution mode from provided or global config."""
    if (
        execution_mode == _OPTIMIZE_DEFAULTS["execution_mode"]
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
    """Log warnings for incompatible execution mode settings."""
    if local_storage_path and execution_mode_enum is ExecutionMode.CLOUD:
        logger.warning(
            "local_storage_path is ignored when execution_mode='cloud'. "
            "Cloud mode uses Traigent cloud storage."
        )

    if minimal_logging and execution_mode_enum is not ExecutionMode.EDGE_ANALYTICS:
        logger.warning(
            "minimal_logging is only effective in Edge Analytics mode. "
            f"It will be ignored in {actual_execution_mode} mode."
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

    for key, value in normalized_runtime_overrides.items():
        if key in _REMOVED_PARAMETERS:
            raise TypeError(
                "The following optimize() parameters have been removed: "
                f"[{key}]. Use supported arguments such as parallel_config "
                "or ExecutionOptions instead."
            )
        if key in _DIRECT_OPTION_KEYS:
            record_option(key, value, "keyword argument")
        else:
            combined_runtime_overrides[key] = value

    removed_in_runtime = set(combined_runtime_overrides) & _REMOVED_PARAMETERS
    if removed_in_runtime:
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
        if param_defaults:
            default_config = {**param_defaults, **(default_config or {})}

    if config_space_constraints and not constraints:
        constraints = config_space_constraints

    return normalized_space, default_config, constraints


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
        return value.value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return DetectionConfidence(normalized).value
        except ValueError as exc:
            raise TypeError(
                "auto_detect_tvars_min_confidence must be one of "
                "['high', 'medium', 'low']."
            ) from exc
    raise TypeError(
        "auto_detect_tvars_min_confidence must be a string " "('high'|'medium'|'low')."
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
        return config_space
    except Exception:
        logger.debug("auto_detect_tvars: detection failed", exc_info=True)
        return None


def optimize(
    *,
    objectives: list[str] | ObjectiveSchema | None = None,
    configuration_space: dict[str, Any] | ConfigSpace | None = None,
    default_config: dict[str, Any] | None = None,
    constraints: list[Constraint | BoolExpr | Callable[..., Any]] | None = None,
    safety_constraints: list[SafetyConstraint | CompoundSafetyConstraint] | None = None,
    tvl_spec: str | Path | None = None,
    tvl_environment: str | None = None,
    tvl: TVLOptions | dict[str, Any] | None = None,
    evaluation: EvaluationOptions | dict[str, Any] | None = None,
    injection: InjectionOptions | dict[str, Any] | None = None,
    execution: ExecutionOptions | dict[str, Any] | None = None,
    mock: MockModeOptions | dict[str, Any] | None = None,
    # Multi-agent configuration
    agents: dict[str, AgentDefinition] | None = None,
    agent_prefixes: list[str] | None = None,
    agent_measures: dict[str, list[str]] | None = None,
    global_measures: list[str] | None = None,
    # Config persistence
    auto_load_best: bool = False,
    load_from: str | None = None,
    legacy: LegacyOptimizeArgs | dict[str, Any] | None = None,
    **runtime_overrides: Any,
) -> Callable[
    [Callable[..., Any]], Any
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
        configuration_space: Dictionary describing the search space. Keys are
            parameter names; values can be discrete lists, numeric tuples, or nested
            dicts for composite parameters.
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
            eval_dataset: Dataset input accepted as a JSONL path, list of paths, or
                Dataset instance with ``input`` and ``expected_output`` columns.
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

        Execution options:
            execution: Grouped execution settings (ExecutionOptions or dict) spanning
                orchestration, storage, and parallelism.
            execution_mode: Execution mode ("cloud", "edge_analytics", "privacy",
                "standard"). Defaults to "edge_analytics".
            local_storage_path: Location for Edge Analytics storage. Falls back
                to ``TRAIGENT_RESULTS_FOLDER`` or ``~/.traigent/`` when omitted.
            minimal_logging: Toggle for reduced logging noise in Edge mode.
            parallel_config: Consolidated parallel configuration (ParallelConfig
                or dict). Preferred path for controlling concurrency.
            privacy_enabled: Flag enabling hybrid privacy safeguards.
            max_total_examples: Global sample budget across all trials.
            samples_include_pruned: Whether pruned trials count toward the sample budget.

        Mock mode options:
            mock: Grouped mock-mode preferences (MockModeOptions or dict).
            mock_mode_config: Dict controlling mock behaviour keys such as
                ``enabled``, ``override_evaluator``, ``base_accuracy``, ``variance``.

        Cost safeguards:
            cost_limit: Maximum USD spending per optimization run. Defaults to
                TRAIGENT_RUN_COST_LIMIT env var or $2.00.
            cost_approved: Skip cost approval prompt. Use with caution in production.

        Additional controls:
            legacy: Adapter for the legacy decorator signature. Accepts either a
                LegacyOptimizeArgs instance or a dict with the historic keyword
                arguments. Values provided here merge with the explicit parameters.
            **runtime_overrides: Runtime overrides such as ``algorithm``, ``max_trials``,
                ``timeout``, ``cache_policy``, or stop-condition knobs like
                ``budget_limit``, ``plateau_window``, ``cost_limit``, and ``cost_approved``.
                Tuned-variable detection controls are also available via runtime
                overrides: ``auto_detect_tvars`` (bool), ``auto_detect_tvars_mode``
                (``off|suggest|apply``), ``auto_detect_tvars_min_confidence``
                (``high|medium|low``), and optional include/exclude filters via
                ``auto_detect_tvars_include`` / ``auto_detect_tvars_exclude``.

    Warning:
        Optimization runs multiple LLM calls. Use TRAIGENT_MOCK_LLM=true for testing.
        Cost estimates are approximations; actual billing is determined by your LLM provider.
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

        Edge Analytics mode optimization for privacy and offline use:

        >>> @traigent.optimize(
        ...     objectives=["accuracy", "safety"],
        ...     evaluation={"eval_dataset": "medical_qa.jsonl"},
        ...     execution={
        ...         "execution_mode": "edge_analytics",
        ...         "local_storage_path": "./my_optimizations",
        ...         "minimal_logging": True,
        ...     },
        ...     configuration_space={
        ...         "model": ["gpt-4", "claude-2"],
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

        def passthrough_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """No-op decorator that returns the function unchanged when Traigent is disabled."""
            return func

        return passthrough_decorator

    legacy_args = _parse_legacy_args(legacy)

    combined_settings = dict(_OPTIMIZE_DEFAULTS)
    provided_sources: dict[str, str] = {}
    record_option = _build_settings_recorder(combined_settings, provided_sources)

    if legacy_args:
        for key, value in legacy_args.iter_known_values():
            record_option(key, value, "legacy arguments")

    direct_inputs = {
        "objectives": objectives,
        "configuration_space": configuration_space,
        "default_config": default_config,
        "constraints": constraints,
        "safety_constraints": safety_constraints,
        "tvl_spec": tvl_spec,
        "tvl_environment": tvl_environment,
        "tvl": tvl,
        "evaluation": evaluation,
        "injection": injection,
        "execution": execution,
        "mock": mock,
        "agents": agents,
        "agent_prefixes": agent_prefixes,
        "agent_measures": agent_measures,
        "global_measures": global_measures,
        "auto_load_best": auto_load_best,
        "load_from": load_from,
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

    # Process ConfigSpace constraints
    config_space_constraints, config_space_var_names, _ = (
        _process_config_space_constraints(configuration_space, constraints)
    )
    constraint_scope_var_names = _build_constraint_scope_var_names(
        configuration_space,
        inline_params,
        config_space_var_names,
    )

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

    # Note: Constraint normalization is deferred until after TVL artifact application
    # to allow merging of user constraints with TVL constraints
    injection_mode = combined_settings["injection_mode"]
    config_param = combined_settings["config_param"]
    auto_override_frameworks = combined_settings["auto_override_frameworks"]
    framework_targets = combined_settings["framework_targets"]
    execution_mode = combined_settings["execution_mode"]
    hybrid_api_endpoint = combined_settings["hybrid_api_endpoint"]
    tunable_id = combined_settings["tunable_id"]
    hybrid_api_transport = combined_settings["hybrid_api_transport"]
    hybrid_api_transport_type = combined_settings["hybrid_api_transport_type"]
    hybrid_api_batch_size = combined_settings["hybrid_api_batch_size"]
    hybrid_api_batch_parallelism = combined_settings["hybrid_api_batch_parallelism"]
    hybrid_api_keep_alive = combined_settings["hybrid_api_keep_alive"]
    hybrid_api_heartbeat_interval = combined_settings["hybrid_api_heartbeat_interval"]
    hybrid_api_timeout = combined_settings["hybrid_api_timeout"]
    hybrid_api_auth_header = combined_settings["hybrid_api_auth_header"]
    hybrid_api_auto_discover_tvars = combined_settings["hybrid_api_auto_discover_tvars"]
    local_storage_path = combined_settings["local_storage_path"]
    minimal_logging = combined_settings["minimal_logging"]
    parallel_config = combined_settings["parallel_config"]
    privacy_enabled = combined_settings["privacy_enabled"]
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
    algorithm_value = combined_settings["algorithm"]
    # Optimizer limits
    max_trials_value = combined_settings["max_trials"]
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
    ) = _resolve_injection_bundle_options(
        injection_bundle,
        injection_mode,
        config_param,
        auto_override_frameworks,
        framework_targets,
        defaults,
    )

    base_execution_options = ResolvedExecutionOptions(
        execution_mode=execution_mode,
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
        parallel_config=parallel_config,
        privacy_enabled=privacy_enabled,
        max_total_examples=max_total_examples,
        samples_include_pruned=samples_include_pruned,
        js_runtime_config=None,
    )
    resolved_execution = _resolve_execution_bundle_options(
        execution_bundle,
        base_execution_options,
        defaults,
    )
    execution_mode = resolved_execution.execution_mode
    hybrid_api_endpoint = resolved_execution.hybrid_api_endpoint
    tunable_id = resolved_execution.tunable_id
    hybrid_api_transport = resolved_execution.hybrid_api_transport
    hybrid_api_transport_type = resolved_execution.hybrid_api_transport_type
    hybrid_api_batch_size = resolved_execution.hybrid_api_batch_size
    hybrid_api_batch_parallelism = resolved_execution.hybrid_api_batch_parallelism
    hybrid_api_keep_alive = resolved_execution.hybrid_api_keep_alive
    hybrid_api_heartbeat_interval = resolved_execution.hybrid_api_heartbeat_interval
    hybrid_api_timeout = resolved_execution.hybrid_api_timeout
    hybrid_api_auth_header = resolved_execution.hybrid_api_auth_header
    hybrid_api_auto_discover_tvars = resolved_execution.hybrid_api_auto_discover_tvars
    local_storage_path = resolved_execution.local_storage_path
    minimal_logging = resolved_execution.minimal_logging
    parallel_config = resolved_execution.parallel_config
    privacy_enabled = resolved_execution.privacy_enabled
    max_total_examples = resolved_execution.max_total_examples
    samples_include_pruned = resolved_execution.samples_include_pruned
    js_runtime_config = resolved_execution.js_runtime_config

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

    def decorator(func: Callable[..., Any]) -> OptimizedFunction:
        """Actual decorator function.

        Args:
            func: Function to optimize

        Returns:
            OptimizedFunction wrapper
        """
        _check_deprecated_objective_kwargs(combined_runtime_overrides)

        logger.debug(f"Decorating function {func.__name__} with @traigent.optimize")

        resolved_schema = _resolve_objective_schema(objectives)

        requested_execution_mode = execution_mode
        actual_execution_mode = _resolve_actual_execution_mode(execution_mode)

        actual_injection_mode = _resolve_injection_mode_enum(injection_mode)

        # Validate injection mode is compatible with JS runtime
        _validate_js_runtime_injection_mode(js_runtime_config, actual_injection_mode)

        execution_mode_enum, effective_privacy_enabled = _resolve_execution_mode_enum(
            actual_execution_mode, privacy_enabled
        )
        actual_execution_mode = execution_mode_enum.value

        _log_execution_mode_warnings(
            execution_mode_enum,
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

        optimized_func = OptimizedFunction(
            func=func,
            eval_dataset=eval_dataset,
            objectives=resolved_schema,
            configuration_space=resolved_configuration_space,
            algorithm=algorithm_value,
            default_config=resolved_default_config,
            constraints=normalized_constraints,
            safety_constraints=safety_constraints,
            injection_mode=actual_injection_mode,
            config_param=config_param,
            auto_override_frameworks=auto_override_frameworks,
            framework_targets=framework_targets,
            execution_mode=execution_mode_enum,
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
            privacy_enabled=effective_privacy_enabled,
            mock_mode_config=mock_mode_config,
            custom_evaluator=custom_evaluator,
            scoring_function=scoring_function,
            metric_functions=metric_functions,
            requested_execution_mode=requested_execution_mode,
            # Multi-agent configuration
            agents=agents_config,
            agent_prefixes=agent_prefixes_config,
            agent_measures=agent_measures_config,
            global_measures=global_measures_config,
            # Config persistence
            auto_load_best=auto_load_best_config,
            load_from=load_from_config,
            # JS runtime configuration
            js_runtime_config=js_runtime_config,
            # TVL promotion gate for statistical best-config selection
            promotion_gate=promotion_gate,
            # Optimizer limits (extracted from combined_settings)
            max_trials=max_trials_value,
            **combined_runtime_overrides,
        )

        logger.info(f"Created optimizable function: {func.__name__}")

        return optimized_func

    return decorator
