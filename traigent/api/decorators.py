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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from traigent.api.config_space import ConfigSpace
    from traigent.api.constraints import BoolExpr, Constraint

from pydantic import BaseModel, ConfigDict

from traigent.api.functions import _GLOBAL_CONFIG
from traigent.api.parameter_ranges import (
    is_inline_param_definition,
    normalize_configuration_space,
)
from traigent.api.types import AgentDefinition
from traigent.config.parallel import (
    ParallelConfig,
    coerce_parallel_config,
    merge_parallel_configs,
)
from traigent.config.types import ExecutionMode, InjectionMode, resolve_execution_mode
from traigent.core.objectives import (
    ObjectiveSchema,
    create_default_objectives,
    normalize_objectives,
)
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset
from traigent.tvl.options import TVLOptions
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
        injection_mode: How to inject config ("context", "parameter", "attribute", "seamless").
        config_param: Parameter name for injection_mode="parameter".
        auto_override_frameworks: Whether to auto-override framework calls.
        framework_targets: List of framework names to target.
        allow_parallel_attribute: Opt-in to allow attribute mode with parallel trials.
            Attribute mode is unsafe for parallel trials (race condition on shared
            function attribute). Set to True only if you understand the risk and
            are using context-based access inside the function body.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    injection_mode: str | InjectionMode = InjectionMode.CONTEXT
    config_param: str | None = None
    # Default to False - requires traigent-integrations plugin for framework overrides
    # Set to True explicitly when using framework integrations
    auto_override_frameworks: bool = False
    framework_targets: list[str] | None = None
    allow_parallel_attribute: bool = False


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

_OPTIMIZE_DEFAULTS: dict[str, Any] = {
    "eval_dataset": None,
    "objectives": None,
    "configuration_space": None,
    "default_config": None,
    "constraints": None,
    "tvl_spec": None,
    "tvl_environment": None,
    "tvl": None,
    "injection_mode": InjectionMode.CONTEXT,
    "config_param": None,
    "auto_override_frameworks": False,  # Requires traigent-integrations plugin
    "framework_targets": None,
    "allow_parallel_attribute": False,
    "execution_mode": "edge_analytics",
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
}

_DIRECT_OPTION_KEYS = frozenset(_OPTIMIZE_DEFAULTS.keys())
_REMOVED_PARAMETERS = frozenset(
    ("auto_optimize", "trigger", "batch_size", "parallel_trials")
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
    tvl_spec: str | None = None
    tvl_environment: str | None = None
    tvl: TVLOptions | dict[str, Any] | None = None
    injection_mode: str | InjectionMode | None = None
    config_param: str | None = None
    auto_override_frameworks: bool | None = None
    framework_targets: list[str] | None = None
    execution_mode: str | None = None
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
            ("tvl_spec", self.tvl_spec),
            ("tvl_environment", self.tvl_environment),
            ("tvl", self.tvl),
            ("injection_mode", self.injection_mode),
            ("config_param", self.config_param),
            ("auto_override_frameworks", self.auto_override_frameworks),
            ("framework_targets", self.framework_targets),
            ("execution_mode", self.execution_mode),
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


def _validate_runtime_overrides(remaining_overrides: dict[str, Any]) -> None:
    """Validate that runtime overrides don't contain unknown keys."""
    unknown_keys = set(remaining_overrides.keys()) - _DIRECT_OPTION_KEYS
    if unknown_keys:
        raise TypeError(
            f"Unknown keyword arguments: {sorted(unknown_keys)}. "
            f"If you meant to define parameter ranges, use Range(), IntRange(), "
            f"Choices(), or tuple syntax. Example: temperature=Range(0.0, 1.0)"
        )


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
    allow_parallel_attribute: Any,
    defaults: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any]:
    """Resolve injection options from bundle."""
    if injection_bundle is None:
        return (
            injection_mode,
            config_param,
            auto_override_frameworks,
            framework_targets,
            allow_parallel_attribute,
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
            "allow_parallel_attribute",
            allow_parallel_attribute,
            injection_bundle.allow_parallel_attribute,
            defaults,
        ),
    )


def _resolve_execution_bundle_options(
    execution_bundle: ExecutionOptions | None,
    execution_mode: Any,
    local_storage_path: Any,
    minimal_logging: Any,
    parallel_config: Any,
    privacy_enabled: Any,
    max_total_examples: Any,
    samples_include_pruned: Any,
    defaults: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Resolve execution options from bundle and validate enterprise features."""
    if execution_bundle is None:
        return (
            execution_mode,
            local_storage_path,
            minimal_logging,
            parallel_config,
            privacy_enabled,
            max_total_examples,
            samples_include_pruned,
        )

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

    return (
        _resolve_option(
            "execution_mode", execution_mode, execution_bundle.execution_mode, defaults
        ),
        _resolve_option(
            "local_storage_path",
            local_storage_path,
            execution_bundle.local_storage_path,
            defaults,
        ),
        _resolve_option(
            "minimal_logging",
            minimal_logging,
            execution_bundle.minimal_logging,
            defaults,
        ),
        _resolve_option(
            "parallel_config",
            parallel_config,
            execution_bundle.parallel_config,
            defaults,
        ),
        _resolve_option(
            "privacy_enabled",
            privacy_enabled,
            execution_bundle.privacy_enabled,
            defaults,
        ),
        _resolve_option(
            "max_total_examples",
            max_total_examples,
            execution_bundle.max_total_examples,
            defaults,
        ),
        _resolve_option(
            "samples_include_pruned",
            samples_include_pruned,
            execution_bundle.samples_include_pruned,
            defaults,
        ),
    )


def _resolve_injection_mode_enum(
    injection_mode: str | InjectionMode,
) -> str | InjectionMode:
    """Convert string injection mode to enum, handling deprecations."""
    if not isinstance(injection_mode, str):
        return injection_mode

    if injection_mode == "decorator":
        warnings.warn(
            "injection_mode='decorator' is deprecated. Use 'attribute' instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        return InjectionMode.ATTRIBUTE

    try:
        return InjectionMode(injection_mode)
    except ValueError:
        return injection_mode


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
]:
    """Apply TVL options if present, returning updated values."""
    if tvl_options is None:
        return (
            configuration_space,
            objectives,
            constraints,
            default_config,
            eval_dataset,
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

    env_suffix = f" (env={tvl_options.environment})" if tvl_options.environment else ""
    logger.info("TVL spec %s applied%s", tvl_artifact.path, env_suffix)

    return configuration_space, objectives, constraints, default_config, eval_dataset


def _process_runtime_overrides(
    runtime_overrides: dict[str, Any],
    legacy_args: LegacyOptimizeArgs | None,
    record_option: Callable[[str, Any, str], None],
) -> dict[str, Any]:
    """Process runtime overrides, handling removed parameters."""
    combined_runtime_overrides: dict[str, Any] = {}
    if legacy_args:
        combined_runtime_overrides.update(legacy_args.extra)

    for key, value in runtime_overrides.items():
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


def optimize(
    *,
    objectives: list[str] | ObjectiveSchema | None = None,
    configuration_space: dict[str, Any] | ConfigSpace | None = None,
    default_config: dict[str, Any] | None = None,
    constraints: list[Constraint | BoolExpr | Callable[..., Any]] | None = None,
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
) -> Callable[[Callable[..., Any]], Any]:
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
                attribute, seamless). Available via the bundle or legacy support.
            config_param: Parameter name used when ``injection_mode="parameter"``.
            auto_override_frameworks: Toggle to auto-detect supported frameworks
                (LangChain, OpenAI, Anthropic, etc.) and override their parameters.
            framework_targets: Explicit list of framework classes to override.
            allow_parallel_attribute: Opt-in to allow attribute mode with parallel
                trials (unsafe by default due to shared function attributes).

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

    Warning:
        Optimization runs multiple LLM calls. Use TRAIGENT_MOCK_LLM=true for testing.
        Cost estimates are approximations; actual billing is determined by your LLM provider.
        See DISCLAIMER.md for full liability terms.

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
    """

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

    # Process ConfigSpace constraints
    config_space_constraints, config_space_var_names, _ = (
        _process_config_space_constraints(configuration_space, constraints)
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
    allow_parallel_attribute = combined_settings["allow_parallel_attribute"]
    execution_mode = combined_settings["execution_mode"]
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
        allow_parallel_attribute,
    ) = _resolve_injection_bundle_options(
        injection_bundle,
        injection_mode,
        config_param,
        auto_override_frameworks,
        framework_targets,
        allow_parallel_attribute,
        defaults,
    )

    (
        execution_mode,
        local_storage_path,
        minimal_logging,
        parallel_config,
        privacy_enabled,
        max_total_examples,
        samples_include_pruned,
    ) = _resolve_execution_bundle_options(
        execution_bundle,
        execution_mode,
        local_storage_path,
        minimal_logging,
        parallel_config,
        privacy_enabled,
        max_total_examples,
        samples_include_pruned,
        defaults,
    )

    tvl_options = _resolve_tvl_options(
        tvl_spec_value, tvl_environment_value, tvl_bundle
    )
    configuration_space, objectives, constraints, default_config, eval_dataset = (
        _apply_tvl_options_if_present(
            tvl_options,
            configuration_space,
            objectives,
            constraints,
            default_config,
            eval_dataset,
            combined_runtime_overrides,
        )
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
                constraints, config_space_var_names
            )

        optimized_func = OptimizedFunction(
            func=func,
            eval_dataset=eval_dataset,
            objectives=resolved_schema,
            configuration_space=configuration_space,
            default_config=default_config,
            constraints=normalized_constraints,
            injection_mode=actual_injection_mode,
            config_param=config_param,
            auto_override_frameworks=auto_override_frameworks,
            framework_targets=framework_targets,
            allow_parallel_attribute=allow_parallel_attribute,
            execution_mode=execution_mode_enum,
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
            **combined_runtime_overrides,
        )

        logger.info(f"Created optimizable function: {func.__name__}")

        return optimized_func

    return decorator
