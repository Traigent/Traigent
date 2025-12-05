"""Main decorator for TraiGent SDK.

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

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from pydantic import BaseModel, ConfigDict

from traigent.api.functions import _GLOBAL_CONFIG
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
    """Configuration bundle controlling how optimized configs are injected."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    injection_mode: str | InjectionMode = InjectionMode.CONTEXT
    config_param: str | None = None
    auto_override_frameworks: bool = True
    framework_targets: list[str] | None = None


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
        f"{parameter_name} must be a dict or {model_cls.__name__}, "
        f"got {type(value).__name__}"
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
    "auto_override_frameworks": True,
    "framework_targets": None,
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
    constraints: list[Callable[..., Any]] | None,
    default_config: dict[str, Any] | None,
    runtime_overrides: dict[str, Any],
) -> tuple[
    dict[str, Any] | None,
    list[str] | ObjectiveSchema | None,
    list[Callable[..., Any]] | None,
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
        constraints = list(constraints or []) + artifact.constraints

    if options.apply_budget:
        overrides = artifact.runtime_overrides()
        for key, value in overrides.items():
            runtime_overrides.setdefault(key, value)
    elif artifact.metadata:
        runtime_overrides.setdefault("tvl_metadata", artifact.metadata)

    if not default_config and artifact.default_config:
        default_config = artifact.default_config

    return configuration_space, objectives, constraints, default_config


def optimize(
    *,
    objectives: list[str] | ObjectiveSchema | None = None,
    configuration_space: dict[str, Any] | None = None,
    default_config: dict[str, Any] | None = None,
    constraints: list[Callable[..., Any]] | None = None,
    tvl_spec: str | Path | None = None,
    tvl_environment: str | None = None,
    tvl: TVLOptions | dict[str, Any] | None = None,
    evaluation: EvaluationOptions | dict[str, Any] | None = None,
    injection: InjectionOptions | dict[str, Any] | None = None,
    execution: ExecutionOptions | dict[str, Any] | None = None,
    mock: MockModeOptions | dict[str, Any] | None = None,
    legacy: LegacyOptimizeArgs | dict[str, Any] | None = None,
    **runtime_overrides: Any,
) -> Callable[[Callable[..., Any]], Any]:
    """Decorator to make functions optimizable with TraiGent.

    This is the main entry point for TraiGent optimization. Decorate any function
    with @traigent.optimize to add zero-code-change optimization capabilities.
    The decorator automatically detects and optimizes LLM invocations without
    requiring any modifications to your existing code. Use the grouped bundles
    for structured parameters and the ``legacy`` argument to bridge the previous
    expansive signature when needed.

    Args:
        objectives: Target metrics to optimize. Accepts a list of names (TraiGent
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

        Additional controls:
            legacy: Adapter for the legacy decorator signature. Accepts either a
                LegacyOptimizeArgs instance or a dict with the historic keyword
                arguments. Values provided here merge with the explicit parameters.
            **runtime_overrides: Runtime overrides such as ``algorithm``, ``max_trials``,
                ``timeout``, ``cache_policy``, or stop-condition knobs like
                ``budget_limit`` and ``plateau_window``.

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
        ... def medical_assistant(patient_query: str) -> str:
        ...     # Data never leaves your infrastructure
        ...     return process_medical_query(patient_query)
        >>> result = my_agent("Hello")
        >>>
        >>> # Run optimization
        >>> import asyncio
        >>> optimization_result = asyncio.run(my_agent.optimize())
        >>> best_config = my_agent.get_best_config()
    """

    if legacy is None:
        legacy_args: LegacyOptimizeArgs | None = None
    elif isinstance(legacy, LegacyOptimizeArgs):
        legacy_args = legacy
    elif isinstance(legacy, dict):
        legacy_args = LegacyOptimizeArgs.from_mapping(legacy)
    else:
        raise TypeError("legacy must be a LegacyOptimizeArgs instance or dict")

    combined_settings = dict(_OPTIMIZE_DEFAULTS)
    provided_sources: dict[str, str] = {}

    def record_option(key: str, value: Any, source: str) -> None:
        """Merge a single optimize option while guarding against duplicates.

        Args:
            key: Optimize parameter name being merged.
            value: Candidate value to apply for the parameter.
            source: Human-readable description of where the value originated.
        """
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
    }
    for key, value in direct_inputs.items():
        record_option(key, value, "optimize parameter")

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

    eval_dataset = combined_settings["eval_dataset"]
    objectives = combined_settings["objectives"]
    configuration_space = combined_settings["configuration_space"]
    default_config = combined_settings["default_config"]
    constraints = combined_settings["constraints"]
    injection_mode = combined_settings["injection_mode"]
    config_param = combined_settings["config_param"]
    auto_override_frameworks = combined_settings["auto_override_frameworks"]
    framework_targets = combined_settings["framework_targets"]
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

    if evaluation_bundle:
        eval_dataset = _resolve_option(
            "eval_dataset", eval_dataset, evaluation_bundle.eval_dataset, defaults
        )
        custom_evaluator = _resolve_option(
            "custom_evaluator",
            custom_evaluator,
            evaluation_bundle.custom_evaluator,
            defaults,
        )
        scoring_function = _resolve_option(
            "scoring_function",
            scoring_function,
            evaluation_bundle.scoring_function,
            defaults,
        )
        metric_functions = _resolve_option(
            "metric_functions",
            metric_functions,
            evaluation_bundle.metric_functions,
            defaults,
        )

    if injection_bundle:
        injection_mode = _resolve_option(
            "injection_mode",
            injection_mode,
            injection_bundle.injection_mode,
            defaults,
        )
        config_param = _resolve_option(
            "config_param", config_param, injection_bundle.config_param, defaults
        )
        auto_override_frameworks = _resolve_option(
            "auto_override_frameworks",
            auto_override_frameworks,
            injection_bundle.auto_override_frameworks,
            defaults,
        )
        framework_targets = _resolve_option(
            "framework_targets",
            framework_targets,
            injection_bundle.framework_targets,
            defaults,
        )

    if execution_bundle:
        execution_mode = _resolve_option(
            "execution_mode",
            execution_mode,
            execution_bundle.execution_mode,
            defaults,
        )
        local_storage_path = _resolve_option(
            "local_storage_path",
            local_storage_path,
            execution_bundle.local_storage_path,
            defaults,
        )
        minimal_logging = _resolve_option(
            "minimal_logging",
            minimal_logging,
            execution_bundle.minimal_logging,
            defaults,
        )
        parallel_config = _resolve_option(
            "parallel_config",
            parallel_config,
            execution_bundle.parallel_config,
            defaults,
        )
        privacy_enabled = _resolve_option(
            "privacy_enabled",
            privacy_enabled,
            execution_bundle.privacy_enabled,
            defaults,
        )
        max_total_examples = _resolve_option(
            "max_total_examples",
            max_total_examples,
            execution_bundle.max_total_examples,
            defaults,
        )
        samples_include_pruned = _resolve_option(
            "samples_include_pruned",
            samples_include_pruned,
            execution_bundle.samples_include_pruned,
            defaults,
        )

    tvl_options = _resolve_tvl_options(
        tvl_spec_value, tvl_environment_value, tvl_bundle
    )
    if tvl_options:
        try:
            tvl_artifact = load_tvl_spec(**tvl_options.to_kwargs())  # type: ignore[arg-type]
        except TVLValidationError as exc:
            raise ValidationError(exc.message) from exc

        (
            configuration_space,
            objectives,
            constraints,
            default_config,
        ) = _apply_tvl_artifact(
            artifact=tvl_artifact,
            options=tvl_options,
            configuration_space=configuration_space,
            objectives=objectives,
            constraints=constraints,
            default_config=default_config,
            runtime_overrides=combined_runtime_overrides,
        )
        logger.info(
            "TVL spec %s applied%s",
            tvl_artifact.path,
            f" (env={tvl_options.environment})" if tvl_options.environment else "",
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

    if objectives is not None and not isinstance(objectives, ObjectiveSchema):
        if isinstance(objectives, (str, bytes)):
            raise ValidationError(
                "objectives must be a sequence of strings or an ObjectiveSchema"
            )
        try:
            iter(objectives)
        except TypeError as exc:
            raise ValidationError(
                "objectives must be a sequence of strings or an ObjectiveSchema"
            ) from exc
        if not all(isinstance(obj, str) for obj in objectives):
            raise ValidationError(
                "All objectives must be strings when provided as a sequence"
            )

    def decorator(func: Callable[..., Any]) -> OptimizedFunction:
        """Actual decorator function.

        Args:
            func: Function to optimize

        Returns:
            OptimizedFunction wrapper
        """
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

        logger.debug(f"Decorating function {func.__name__} with @traigent.optimize")

        resolved_schema = normalize_objectives(objectives)
        if resolved_schema is None:
            global_schema = _GLOBAL_CONFIG.get("objective_schema")
            if isinstance(global_schema, ObjectiveSchema):
                resolved_schema = global_schema
            else:
                global_names = _GLOBAL_CONFIG.get("objectives") or []
                if global_names:
                    resolved_schema = create_default_objectives(list(global_names))

        if resolved_schema is None:
            resolved_schema = create_default_objectives(["accuracy"])

        requested_execution_mode = execution_mode
        actual_execution_mode = execution_mode

        if (
            actual_execution_mode == _OPTIMIZE_DEFAULTS["execution_mode"]
            and "execution_mode" in _GLOBAL_CONFIG
        ):
            actual_execution_mode = _GLOBAL_CONFIG["execution_mode"]
            logger.debug(
                f"Using execution mode from global config: {actual_execution_mode}"
            )
        else:
            logger.debug(
                f"Using explicitly provided execution mode: {actual_execution_mode}"
            )

        actual_injection_mode = injection_mode

        if isinstance(actual_injection_mode, str):
            if actual_injection_mode == "decorator":
                warnings.warn(
                    "injection_mode='decorator' is deprecated. Use 'attribute' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                actual_injection_mode = InjectionMode.ATTRIBUTE
            else:
                try:
                    actual_injection_mode = InjectionMode(actual_injection_mode)
                except ValueError:
                    pass

        effective_privacy_enabled = privacy_enabled
        try:
            execution_mode_enum = resolve_execution_mode(actual_execution_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from None

        if execution_mode_enum is ExecutionMode.PRIVACY:
            logger.warning(
                "execution_mode='privacy' is deprecated. Use execution_mode='hybrid' "
                "with privacy_enabled=True. Mapping automatically."
            )
            execution_mode_enum = ExecutionMode.HYBRID
            if effective_privacy_enabled is None:
                effective_privacy_enabled = True

        actual_execution_mode = execution_mode_enum.value

        if local_storage_path and execution_mode_enum is ExecutionMode.CLOUD:
            logger.warning(
                "local_storage_path is ignored when execution_mode='cloud'. "
                "Cloud mode uses TraiGent cloud storage."
            )

        if minimal_logging and execution_mode_enum is not ExecutionMode.EDGE_ANALYTICS:
            logger.warning(
                "minimal_logging is only effective in Edge Analytics mode. "
                f"It will be ignored in {actual_execution_mode} mode."
            )

        user_parallel_config = coerce_parallel_config(parallel_config)
        combined_parallel_config, parallel_sources = merge_parallel_configs(
            [(user_parallel_config, "decorator")]
        )
        if parallel_sources:
            logger.debug(
                "Decorator parallel configuration sources: %s", parallel_sources
            )

        optimized_func = OptimizedFunction(
            func=func,
            eval_dataset=eval_dataset,
            objectives=resolved_schema,
            configuration_space=configuration_space,
            default_config=default_config,
            constraints=constraints,
            injection_mode=actual_injection_mode,
            config_param=config_param,
            auto_override_frameworks=auto_override_frameworks,
            framework_targets=framework_targets,
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
            **combined_runtime_overrides,
        )

        logger.info(f"Created optimizable function: {func.__name__}")

        return optimized_func

    return decorator
