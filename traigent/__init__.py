"""Traigent SDK - Open-source LLM optimization toolkit.

Traigent makes it effortless to optimize your LLM applications with a simple decorator.

Example:
    >>> import traigent
    >>> from traigent import Range, IntRange, Choices, LogRange
    >>>
    >>> # New SE-friendly syntax with Range/Choices classes
    >>> @traigent.optimize(
    ...     eval_dataset="evals.jsonl",
    ...     objectives=["accuracy", "cost"],
    ...     temperature=Range(0.0, 2.0),
    ...     max_tokens=IntRange(100, 4096),
    ...     model=Choices(["gpt-4o-mini", "gpt-4o"]),
    ... )
    ... def my_function(query: str) -> str:
    ...     return process_query(query)
    >>>
    >>> # Legacy syntax still works
    >>> @traigent.optimize(
    ...     configuration_space={
    ...         "model": ["gpt-4o-mini", "gpt-4o"],
    ...         "temperature": (0.0, 1.0)
    ...     }
    ... )
    ... def my_function(query: str) -> str:
    ...     return process_query(query)
    >>>
    >>> results = my_function.optimize()
    >>> best_config = my_function.get_best_config()
"""

# ruff: noqa: E402

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import builtins
import os
import sys
import warnings
from importlib import import_module


# ---------------------------------------------------------------------------
# Quickstart bootstrap (must run BEFORE the heavy package imports below).
# ---------------------------------------------------------------------------
#
# ``traigent quickstart`` and ``python -m traigent.examples.quickstart`` both
# trigger ``traigent/__init__.py`` first (Python imports the parent package
# before resolving any submodule), which pulls in optional dependencies
# (LiteLLM model discovery, langfuse, etc.) that may attempt network at
# import time. The bundled quickstart is the load-bearing demo on the
# website funnel and MUST work with no API keys, no network, and no
# surprises — so we detect quickstart invocations by inspecting ``sys.argv``
# at the top of this module and seed the legacy env-var paths the SDK
# already knows about. For non-quickstart invocations this is a no-op,
# preserving the SDK's normal behavior for production code.
def _is_quickstart_invocation() -> bool:
    if not sys.argv:
        return False
    argv0 = sys.argv[0] or ""
    # ``python -m traigent.examples.quickstart`` — argv[0] is the path
    # to the quickstart's __main__.py
    if argv0.endswith(("quickstart/__main__.py", "quickstart\\__main__.py")):
        return True
    # ``traigent quickstart`` — argv[0] is the venv's bin/traigent script
    # and argv[1] is the subcommand name. Gate on argv[0] basename being
    # *exactly* the traigent CLI so an unrelated tool whose name happens
    # to contain "traigent" (e.g. ``my-traigent-util quickstart``) does
    # NOT silently override the user's OPENAI_API_KEY.
    if len(sys.argv) >= 2 and sys.argv[1] == "quickstart":
        basename = os.path.basename(argv0).lower()
        if basename in {"traigent", "traigent.exe"}:
            return True
    return False


if _is_quickstart_invocation():
    # Sentinel telling env_config's prod guard that this env-var write is
    # internal bootstrap, not user code. The prod hard-block still fires
    # if ENVIRONMENT=production (correct: mock mode is blocked even from
    # quickstart in prod), but the dev-mode deprecation warning meant for
    # users who set TRAIGENT_MOCK_LLM themselves is suppressed — they
    # ARE using the in-code path; the env var is just how we hand state
    # across the import boundary.
    os.environ["_TRAIGENT_QUICKSTART_BOOTSTRAP"] = "1"
    os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # Only force offline if the user hasn't supplied a portal key — they
    # may want results synced even while LLM calls stay mocked.
    if not os.environ.get("TRAIGENT_API_KEY"):
        os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")
    # OVERRIDE OPENAI_API_KEY (not setdefault): if a real key is sitting
    # in the parent shell, a mock-regression in the demo could otherwise
    # spend it. The placeholder cannot succeed against a real OpenAI
    # endpoint, which is the whole point.
    os.environ["OPENAI_API_KEY"] = "mock-key-for-demos"  # pragma: allowlist secret


# Suppress noisy FutureWarning from transitive deps (instructor → google.generativeai)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"instructor\.providers\.gemini"
)

from traigent._version import get_version

__version__ = get_version()
__author__ = "Traigent Team"
__email__ = "opensource@traigent.ai"


def _is_missing_optional_module(
    exc: ModuleNotFoundError, module_prefixes: tuple[str, ...]
) -> bool:
    missing_module = getattr(exc, "name", "")
    return any(
        missing_module == prefix or missing_module.startswith(f"{prefix}.")
        for prefix in module_prefixes
    )


_OPTIONAL_EXPORT_ERRORS: dict[str, str] = {}

try:
    # Multi-agent workflow cost tracking (DTO hardening)
    from traigent.cloud.agent_dtos import AgentCostBreakdown, WorkflowCostSummary
    from traigent.cloud.dtos import MeasuresDict
except ModuleNotFoundError as exc:
    if not _is_missing_optional_module(exc, ("traigent.cloud",)):
        raise

    _OPTIONAL_EXPORT_ERRORS.update(
        {
            "AgentCostBreakdown": (
                "module 'traigent' has no attribute 'AgentCostBreakdown'. "
                "Cloud workflow DTO exports are unavailable in this build."
            ),
            "WorkflowCostSummary": (
                "module 'traigent' has no attribute 'WorkflowCostSummary'. "
                "Cloud workflow DTO exports are unavailable in this build."
            ),
            "MeasuresDict": (
                "module 'traigent' has no attribute 'MeasuresDict'. "
                "Cloud workflow DTO exports are unavailable in this build."
            ),
        }
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Main decorator and configuration API
    "optimize": ("traigent.api.decorators", "optimize"),
    "ConfigSpace": ("traigent.api.config_space", "ConfigSpace"),
    "configure": ("traigent.api.functions", "configure"),
    "configure_for_budget": ("traigent.api.functions", "configure_for_budget"),
    "get_available_strategies": (
        "traigent.api.functions",
        "get_available_strategies",
    ),
    "get_config": ("traigent.api.functions", "get_config"),
    "get_current_config": ("traigent.api.functions", "get_current_config"),
    "get_optimization_insights": (
        "traigent.api.functions",
        "get_optimization_insights",
    ),
    "get_trial_config": ("traigent.api.functions", "get_trial_config"),
    "get_version_info": ("traigent.api.functions", "get_version_info"),
    "initialize": ("traigent.api.functions", "initialize"),
    "override_config": ("traigent.api.functions", "override_config"),
    "set_strategy": ("traigent.api.functions", "set_strategy"),
    "with_usage": ("traigent.api.functions", "with_usage"),
    # SE-friendly parameter range classes
    "Choices": ("traigent.api.parameter_ranges", "Choices"),
    "IntRange": ("traigent.api.parameter_ranges", "IntRange"),
    "LogRange": ("traigent.api.parameter_ranges", "LogRange"),
    "ParameterRange": ("traigent.api.parameter_ranges", "ParameterRange"),
    "Range": ("traigent.api.parameter_ranges", "Range"),
    # TVL constraint system
    "AndCondition": ("traigent.api.constraints", "AndCondition"),
    "BoolExpr": ("traigent.api.constraints", "BoolExpr"),
    "Condition": ("traigent.api.constraints", "Condition"),
    "Constraint": ("traigent.api.constraints", "Constraint"),
    "ConstraintScopeError": ("traigent.api.constraints", "ConstraintScopeError"),
    "NotCondition": ("traigent.api.constraints", "NotCondition"),
    "OrCondition": ("traigent.api.constraints", "OrCondition"),
    "WhenBuilder": ("traigent.api.constraints", "WhenBuilder"),
    "constraints_to_callables": (
        "traigent.api.constraints",
        "constraints_to_callables",
    ),
    "implies": ("traigent.api.constraints", "implies"),
    "normalize_constraints": ("traigent.api.constraints", "normalize_constraints"),
    "require": ("traigent.api.constraints", "require"),
    "when": ("traigent.api.constraints", "when"),
    # Core result and validation types
    "ConfigurationComparison": ("traigent.api.types", "ConfigurationComparison"),
    "OptimizationResult": ("traigent.api.types", "OptimizationResult"),
    "OptimizationStatus": ("traigent.api.types", "OptimizationStatus"),
    "ParetoFront": ("traigent.api.types", "ParetoFront"),
    "SensitivityAnalysis": ("traigent.api.types", "SensitivityAnalysis"),
    "StrategyConfig": ("traigent.api.types", "StrategyConfig"),
    "TrialError": ("traigent.api.types", "TrialError"),
    "TrialResult": ("traigent.api.types", "TrialResult"),
    "serialize_trials": ("traigent.api.types", "serialize_trials"),
    "ConstraintValidator": (
        "traigent.api.validation_protocol",
        "ConstraintValidator",
    ),
    "ConstraintViolation": (
        "traigent.api.validation_protocol",
        "ConstraintViolation",
    ),
    "ConstraintValidationResult": (
        "traigent.api.validation_protocol",
        "ValidationResult",
    ),
    "PythonConstraintValidator": (
        "traigent.api.validation_protocol",
        "PythonConstraintValidator",
    ),
    "SatResult": ("traigent.api.validation_protocol", "SatResult"),
    "SatStatus": ("traigent.api.validation_protocol", "SatStatus"),
    # Context and config helpers
    "copy_context_to_thread": ("traigent.config.context", "copy_context_to_thread"),
    "get_trial_context": ("traigent.config.context", "get_trial_context"),
    "TraigentConfig": ("traigent.config.types", "TraigentConfig"),
    "TraigentMetadata": ("traigent.core.meta_types", "TraigentMetadata"),
    "is_traigent_metadata": ("traigent.core.meta_types", "is_traigent_metadata"),
    "OptimizationState": ("traigent.core.optimized_function", "OptimizationState"),
    # Client surfaces
    "BenchmarkClient": ("traigent.cloud.benchmark_client", "BenchmarkClient"),
    "BenchmarkClientConfig": (
        "traigent.cloud.benchmark_client",
        "BenchmarkClientConfig",
    ),
    "CoreMetricsClient": ("traigent.core_metrics", "CoreMetricsClient"),
    "CoreMetricsConfig": ("traigent.core_metrics", "CoreMetricsConfig"),
    "EnterpriseAdminClient": ("traigent.admin", "EnterpriseAdminClient"),
    "EnterpriseAdminConfig": ("traigent.admin", "EnterpriseAdminConfig"),
    "EvaluationClient": ("traigent.evaluation", "EvaluationClient"),
    "EvaluationConfig": ("traigent.evaluation", "EvaluationConfig"),
    "ObservabilityClient": ("traigent.observability", "ObservabilityClient"),
    "ObservabilityConfig": ("traigent.observability", "ObservabilityConfig"),
    "ProjectManagementClient": ("traigent.projects", "ProjectManagementClient"),
    "ProjectManagementConfig": ("traigent.projects", "ProjectManagementConfig"),
    "PromptManagementClient": ("traigent.prompts", "PromptManagementClient"),
    "PromptManagementConfig": ("traigent.prompts", "PromptManagementConfig"),
    # Admin DTOs
    "SSOProviderType": ("traigent.admin", "SSOProviderType"),
    "TenantDTO": ("traigent.admin", "TenantDTO"),
    "TenantListResponse": ("traigent.admin", "TenantListResponse"),
    "TenantMembershipDTO": ("traigent.admin", "TenantMembershipDTO"),
    "TenantMembershipListResponse": (
        "traigent.admin",
        "TenantMembershipListResponse",
    ),
    "TenantMembershipRole": ("traigent.admin", "TenantMembershipRole"),
    "TenantMembershipStatus": ("traigent.admin", "TenantMembershipStatus"),
    "TenantSSOConfigDTO": ("traigent.admin", "TenantSSOConfigDTO"),
    # Evaluation DTOs
    "AnnotationQueueDTO": ("traigent.evaluation", "AnnotationQueueDTO"),
    "AnnotationQueueItemDTO": ("traigent.evaluation", "AnnotationQueueItemDTO"),
    "AnnotationQueueItemListResponse": (
        "traigent.evaluation",
        "AnnotationQueueItemListResponse",
    ),
    "AnnotationQueueItemStatus": (
        "traigent.evaluation",
        "AnnotationQueueItemStatus",
    ),
    "AnnotationQueueListResponse": (
        "traigent.evaluation",
        "AnnotationQueueListResponse",
    ),
    "AnnotationQueueStatus": ("traigent.evaluation", "AnnotationQueueStatus"),
    "BackfillResultDTO": ("traigent.evaluation", "BackfillResultDTO"),
    "Dataset": ("traigent.evaluators.base", "Dataset"),
    "EvaluationExample": ("traigent.evaluators.base", "EvaluationExample"),
    "EvaluationTargetRefDTO": ("traigent.evaluation", "EvaluationTargetRefDTO"),
    "EvaluationTargetType": ("traigent.evaluation", "EvaluationTargetType"),
    "EvaluatorDefinitionDTO": ("traigent.evaluation", "EvaluatorDefinitionDTO"),
    "EvaluatorListResponse": ("traigent.evaluation", "EvaluatorListResponse"),
    "EvaluatorRunDTO": ("traigent.evaluation", "EvaluatorRunDTO"),
    "EvaluatorRunListResponse": ("traigent.evaluation", "EvaluatorRunListResponse"),
    "EvaluatorRunStatus": ("traigent.evaluation", "EvaluatorRunStatus"),
    "JudgeConfigDTO": ("traigent.evaluation", "JudgeConfigDTO"),
    "MeasureValueType": ("traigent.evaluation", "MeasureValueType"),
    "ScoreRecordDTO": ("traigent.evaluation", "ScoreRecordDTO"),
    "ScoreRecordListResponse": ("traigent.evaluation", "ScoreRecordListResponse"),
    "ScoreSource": ("traigent.evaluation", "ScoreSource"),
    # Observability DTOs and helpers
    "CorrelationIds": ("traigent.observability", "CorrelationIds"),
    "ObserveContext": ("traigent.observability", "ObserveContext"),
    "ObservationDTO": ("traigent.observability", "ObservationDTO"),
    "ObservationRecord": ("traigent.observability", "ObservationRecord"),
    "ObservationType": ("traigent.observability", "ObservationType"),
    "PaginationInfo": ("traigent.observability", "PaginationInfo"),
    "SessionDTO": ("traigent.observability", "SessionDTO"),
    "SessionListResponse": ("traigent.observability", "SessionListResponse"),
    "SessionRecord": ("traigent.observability", "SessionRecord"),
    "ThumbRating": ("traigent.observability", "ThumbRating"),
    "TraceCollaborationState": (
        "traigent.observability",
        "TraceCollaborationState",
    ),
    "TraceCommentRecord": ("traigent.observability", "TraceCommentRecord"),
    "TraceCommentsResponse": ("traigent.observability", "TraceCommentsResponse"),
    "TraceDTO": ("traigent.observability", "TraceDTO"),
    "TraceFeedbackRecord": ("traigent.observability", "TraceFeedbackRecord"),
    "TraceFeedbackResponse": ("traigent.observability", "TraceFeedbackResponse"),
    "TraceFeedbackSummary": ("traigent.observability", "TraceFeedbackSummary"),
    "TraceListResponse": ("traigent.observability", "TraceListResponse"),
    "TraceObservationsResponse": (
        "traigent.observability",
        "TraceObservationsResponse",
    ),
    "TraceRecord": ("traigent.observability", "TraceRecord"),
    "get_default_observability_client": (
        "traigent.observability",
        "get_default_observability_client",
    ),
    "observe": ("traigent.observability", "observe"),
    "set_default_observability_client": (
        "traigent.observability",
        "set_default_observability_client",
    ),
    # Prompt DTOs and helpers
    "ChatPromptMessage": ("traigent.prompts", "ChatPromptMessage"),
    "PromptDetail": ("traigent.prompts", "PromptDetail"),
    "PromptListResponse": ("traigent.prompts", "PromptListResponse"),
    "PromptPlaygroundConfig": ("traigent.prompts", "PromptPlaygroundConfig"),
    "PromptPlaygroundResult": ("traigent.prompts", "PromptPlaygroundResult"),
    "PromptPlaygroundTokenUsage": (
        "traigent.prompts",
        "PromptPlaygroundTokenUsage",
    ),
    "PromptSummary": ("traigent.prompts", "PromptSummary"),
    "PromptType": ("traigent.prompts", "PromptType"),
    "PromptVersionRecord": ("traigent.prompts", "PromptVersionRecord"),
    "ResolvedPrompt": ("traigent.prompts", "ResolvedPrompt"),
    # Utilities
    "ConfigAccessWarning": ("traigent.utils.exceptions", "ConfigAccessWarning"),
    "ConstraintManager": ("traigent.utils.constraints", "ConstraintManager"),
    "DataIntegrityError": ("traigent.utils.exceptions", "DataIntegrityError"),
    "DTOSerializationError": (
        "traigent.utils.exceptions",
        "DTOSerializationError",
    ),
    "LoggingCallback": ("traigent.utils.callbacks", "LoggingCallback"),
    "MetricExtractionError": ("traigent.utils.exceptions", "MetricExtractionError"),
    "MultiObjectiveMetrics": (
        "traigent.utils.multi_objective",
        "MultiObjectiveMetrics",
    ),
    "OptimizationStateError": (
        "traigent.utils.exceptions",
        "OptimizationStateError",
    ),
    "OptimizationValidator": ("traigent.utils.validation", "OptimizationValidator"),
    "ParameterImportanceAnalyzer": (
        "traigent.utils.importance",
        "ParameterImportanceAnalyzer",
    ),
    "ParetoFrontCalculator": (
        "traigent.utils.multi_objective",
        "ParetoFrontCalculator",
    ),
    "PersistenceManager": ("traigent.utils.persistence", "PersistenceManager"),
    "PlotGenerator": ("traigent.visualization.plots", "PlotGenerator"),
    "ProgressBarCallback": ("traigent.utils.callbacks", "ProgressBarCallback"),
    "RetryConfig": ("traigent.utils.retry", "RetryConfig"),
    "StatisticsCallback": ("traigent.utils.callbacks", "StatisticsCallback"),
    "TraigentDeprecationWarning": (
        "traigent.utils.exceptions",
        "TraigentDeprecationWarning",
    ),
    "TraigentWarning": ("traigent.utils.exceptions", "TraigentWarning"),
    "ValidationResult": ("traigent.utils.validation", "ValidationResult"),
    "VendorPauseError": ("traigent.utils.exceptions", "VendorPauseError"),
    "create_quick_plot": ("traigent.visualization.plots", "create_quick_plot"),
    "get_default_callbacks": ("traigent.utils.callbacks", "get_default_callbacks"),
    "get_verbose_callbacks": ("traigent.utils.callbacks", "get_verbose_callbacks"),
    "max_tokens_constraint": ("traigent.utils.constraints", "max_tokens_constraint"),
    "model_cost_constraint": ("traigent.utils.constraints", "model_cost_constraint"),
    "retry": ("traigent.utils.retry", "retry"),
    "temperature_constraint": (
        "traigent.utils.constraints",
        "temperature_constraint",
    ),
    "validate_and_suggest": ("traigent.utils.validation", "validate_and_suggest"),
}


def _load_lazy_export(name: str):
    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module_name), attr_name)
    except ModuleNotFoundError as exc:
        if _is_missing_optional_module(exc, (module_name,)):
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}. "
                f"Export {name!r} requires optional module {module_name!r}."
            ) from exc
        raise

    globals()[name] = value
    return value


__all__ = [
    # Main decorator
    "optimize",
    # SE-friendly parameter range classes
    "Range",
    "IntRange",
    "LogRange",
    "Choices",
    "ParameterRange",
    # TVL constraint system
    "AndCondition",
    "BoolExpr",
    "Condition",
    "Constraint",
    "ConstraintScopeError",
    "ConfigSpace",
    "ConstraintValidator",
    "ConstraintViolation",
    "ConstraintValidationResult",
    "NotCondition",
    "OrCondition",
    "PythonConstraintValidator",
    "SatResult",
    "SatStatus",
    "WhenBuilder",
    "constraints_to_callables",
    "implies",
    "normalize_constraints",
    "require",
    "when",
    # Configuration functions
    "configure",
    "configure_for_budget",
    "initialize",
    "override_config",
    "get_available_strategies",
    "get_config",
    "get_current_config",  # Deprecated: use get_trial_config
    "get_trial_config",  # New: use during optimization trials
    "get_trial_context",  # New: check if in optimization context
    "get_optimization_insights",
    "get_version_info",
    "set_strategy",
    "with_usage",  # New: wrap multi-agent workflow responses with usage metadata
    # Configuration types
    "TraigentConfig",
    # Lifecycle and state
    "OptimizationState",
    # Thread context helpers
    "copy_context_to_thread",
    "CoreMetricsClient",
    "CoreMetricsConfig",
    "EnterpriseAdminClient",
    "EnterpriseAdminConfig",
    "ObservabilityClient",
    "ObservabilityConfig",
    "BenchmarkClient",
    "BenchmarkClientConfig",
    "Dataset",
    "EvaluationExample",
    "ProjectManagementClient",
    "ProjectManagementConfig",
    "EvaluationClient",
    "EvaluationConfig",
    "PromptManagementClient",
    "PromptManagementConfig",
    "PromptPlaygroundConfig",
    "PromptPlaygroundResult",
    "PromptPlaygroundTokenUsage",
    "SSOProviderType",
    "TenantDTO",
    "TenantListResponse",
    "TenantMembershipDTO",
    "TenantMembershipListResponse",
    "TenantMembershipRole",
    "TenantMembershipStatus",
    "TenantSSOConfigDTO",
    "AnnotationQueueDTO",
    "AnnotationQueueItemDTO",
    "AnnotationQueueItemListResponse",
    "AnnotationQueueItemStatus",
    "AnnotationQueueListResponse",
    "AnnotationQueueStatus",
    "BackfillResultDTO",
    "EvaluationTargetRefDTO",
    "EvaluationTargetType",
    "EvaluatorDefinitionDTO",
    "EvaluatorListResponse",
    "EvaluatorRunDTO",
    "EvaluatorRunListResponse",
    "EvaluatorRunStatus",
    "JudgeConfigDTO",
    "MeasureValueType",
    "ScoreRecordDTO",
    "ScoreRecordListResponse",
    "ScoreSource",
    "CorrelationIds",
    "ObserveContext",
    "ObservationDTO",
    "ObservationRecord",
    "ObservationType",
    "PaginationInfo",
    "SessionDTO",
    "SessionListResponse",
    "SessionRecord",
    "ThumbRating",
    "TraceCollaborationState",
    "TraceCommentRecord",
    "TraceCommentsResponse",
    "TraceDTO",
    "TraceFeedbackRecord",
    "TraceFeedbackResponse",
    "TraceFeedbackSummary",
    "TraceListResponse",
    "TraceObservationsResponse",
    "TraceRecord",
    "get_default_observability_client",
    "observe",
    "set_default_observability_client",
    "ChatPromptMessage",
    "PromptDetail",
    "PromptListResponse",
    "PromptSummary",
    "PromptType",
    "PromptVersionRecord",
    "ResolvedPrompt",
    "TraigentMetadata",
    "is_traigent_metadata",
    # Exceptions and warnings
    "TraigentWarning",
    "TraigentDeprecationWarning",
    "OptimizationStateError",
    "ConfigAccessWarning",
    "DataIntegrityError",
    "MetricExtractionError",
    "DTOSerializationError",
    # Sprint 2 features
    "PersistenceManager",
    "ProgressBarCallback",
    "LoggingCallback",
    "StatisticsCallback",
    "get_default_callbacks",
    "get_verbose_callbacks",
    "ParetoFrontCalculator",
    "MultiObjectiveMetrics",
    "ParameterImportanceAnalyzer",
    "RetryConfig",
    "retry",
    # Sprint 3 features
    "OptimizationValidator",
    "ValidationResult",
    "validate_and_suggest",
    "ConstraintManager",
    "temperature_constraint",
    "model_cost_constraint",
    "max_tokens_constraint",
    "PlotGenerator",
    "create_quick_plot",
    # Result types
    "OptimizationResult",
    "TrialError",
    "TrialResult",
    "serialize_trials",
    "SensitivityAnalysis",
    "ConfigurationComparison",
    "ParetoFront",
    "OptimizationStatus",
    "StrategyConfig",
]

if "AgentCostBreakdown" in globals():
    __all__.extend(
        [
            "AgentCostBreakdown",
            "WorkflowCostSummary",
            "MeasuresDict",
        ]
    )


def __getattr__(name: str):
    if name in _OPTIONAL_EXPORT_ERRORS:
        raise AttributeError(_OPTIONAL_EXPORT_ERRORS[name])
    if name in _LAZY_EXPORTS:
        return _load_lazy_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# NOTE: Legacy builtins injection removed in v0.9.0
# Use explicit imports: `import traigent`
