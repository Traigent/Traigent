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
import sys
import warnings

# Suppress noisy FutureWarning from transitive deps (instructor → google.generativeai)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"instructor\.providers\.gemini"
)

from traigent._version import get_version

__version__ = get_version()
__author__ = "Traigent Team"
__email__ = "opensource@traigent.ai"

# Multi-agent workflow cost tracking (DTO hardening)
from traigent.admin import (
    EnterpriseAdminClient,
    EnterpriseAdminConfig,
    SSOProviderType,
    TenantDTO,
    TenantListResponse,
    TenantMembershipDTO,
    TenantMembershipListResponse,
    TenantMembershipRole,
    TenantMembershipStatus,
    TenantSSOConfigDTO,
)
from traigent.api.config_space import ConfigSpace

# TVL constraint system
from traigent.api.constraints import (
    AndCondition,
    BoolExpr,
    Condition,
    Constraint,
    ConstraintScopeError,
    NotCondition,
    OrCondition,
    WhenBuilder,
    constraints_to_callables,
    implies,
    normalize_constraints,
    require,
    when,
)

# Public API exports
from traigent.api.decorators import optimize
from traigent.api.functions import (
    configure,
    configure_for_budget,
    get_available_strategies,
    get_config,
    get_current_config,
    get_optimization_insights,
    get_trial_config,
    get_version_info,
    initialize,
    override_config,
    set_strategy,
    with_usage,
)

# SE-friendly parameter range classes
from traigent.api.parameter_ranges import (
    Choices,
    IntRange,
    LogRange,
    ParameterRange,
    Range,
)

# Core types
from traigent.api.types import (
    ConfigurationComparison,
    OptimizationResult,
    OptimizationStatus,
    ParetoFront,
    SensitivityAnalysis,
    StrategyConfig,
    TrialError,
    TrialResult,
    serialize_trials,
)
from traigent.api.validation_protocol import (
    ConstraintValidator,
    ConstraintViolation,
    PythonConstraintValidator,
    SatResult,
    SatStatus,
)
from traigent.api.validation_protocol import (
    ValidationResult as ConstraintValidationResult,
)
from traigent.cloud.agent_dtos import AgentCostBreakdown, WorkflowCostSummary
from traigent.cloud.dtos import MeasuresDict

# Thread context helpers
from traigent.config.context import copy_context_to_thread, get_trial_context

# Configuration types
from traigent.config.types import TraigentConfig
from traigent.core.meta_types import TraigentMetadata, is_traigent_metadata

# Lifecycle and state management
from traigent.core.optimized_function import OptimizationState
from traigent.core_metrics import (
    CoreEntityCountsDTO,
    CoreExperimentTrendDTO,
    CoreMetricsClient,
    CoreMetricsConfig,
    CoreMetricsOverviewDTO,
    DailyCountPointDTO,
    FineTuningExportDTO,
    MeasureAggregateSummaryDTO,
)
from traigent.evaluation import (
    AnnotationQueueDTO,
    AnnotationQueueItemDTO,
    AnnotationQueueItemListResponse,
    AnnotationQueueItemStatus,
    AnnotationQueueListResponse,
    AnnotationQueueStatus,
    BackfillResultDTO,
    EvaluationClient,
    EvaluationConfig,
    EvaluationTargetRefDTO,
    EvaluationTargetType,
    EvaluatorDefinitionDTO,
    EvaluatorListResponse,
    EvaluatorRunDTO,
    EvaluatorRunListResponse,
    EvaluatorRunStatus,
    JudgeConfigDTO,
    MeasureValueType,
    ScoreRecordDTO,
    ScoreRecordListResponse,
    ScoreSource,
)
from traigent.observability import (
    CorrelationIds,
    ObservabilityClient,
    ObservabilityConfig,
    ObservationDTO,
    ObservationRecord,
    ObservationType,
    ObserveContext,
    PaginationInfo,
    SessionDTO,
    SessionListResponse,
    SessionRecord,
    ThumbRating,
    TraceCollaborationState,
    TraceCommentRecord,
    TraceCommentsResponse,
    TraceDTO,
    TraceFeedbackRecord,
    TraceFeedbackResponse,
    TraceFeedbackSummary,
    TraceListResponse,
    TraceObservationsResponse,
    TraceRecord,
    get_default_observability_client,
    observe,
    set_default_observability_client,
)
from traigent.projects import (
    ProjectDTO,
    ProjectListResponse,
    ProjectManagementClient,
    ProjectManagementConfig,
)
from traigent.prompts import (
    ChatPromptMessage,
    PromptDetail,
    PromptListResponse,
    PromptManagementClient,
    PromptManagementConfig,
    PromptSummary,
    PromptType,
    PromptVersionRecord,
    ResolvedPrompt,
)
from traigent.utils.callbacks import (
    LoggingCallback,
    ProgressBarCallback,
    StatisticsCallback,
    get_default_callbacks,
    get_verbose_callbacks,
)
from traigent.utils.constraints import (
    ConstraintManager,
    max_tokens_constraint,
    model_cost_constraint,
    temperature_constraint,
)

# Exceptions and warnings
from traigent.utils.exceptions import (
    ConfigAccessWarning,
    DataIntegrityError,
    DTOSerializationError,
    MetricExtractionError,
    OptimizationStateError,
    TraigentDeprecationWarning,
    TraigentWarning,
    VendorPauseError,
)
from traigent.utils.importance import ParameterImportanceAnalyzer
from traigent.utils.multi_objective import MultiObjectiveMetrics, ParetoFrontCalculator

# Sprint 2 features
from traigent.utils.persistence import PersistenceManager
from traigent.utils.retry import RetryConfig, retry

# Sprint 3 features
from traigent.utils.validation import (
    OptimizationValidator,
    ValidationResult,
    validate_and_suggest,
)
from traigent.visualization.plots import PlotGenerator, create_quick_plot


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
    # Multi-agent workflow cost tracking (DTO hardening)
    "EnterpriseAdminClient",
    "EnterpriseAdminConfig",
    "AgentCostBreakdown",
    "WorkflowCostSummary",
    "MeasuresDict",
    "CoreMetricsClient",
    "CoreMetricsConfig",
    "CoreEntityCountsDTO",
    "CoreExperimentTrendDTO",
    "CoreMetricsOverviewDTO",
    "DailyCountPointDTO",
    "FineTuningExportDTO",
    "MeasureAggregateSummaryDTO",
    "SSOProviderType",
    "TenantDTO",
    "TenantListResponse",
    "TenantMembershipDTO",
    "TenantMembershipListResponse",
    "TenantMembershipRole",
    "TenantMembershipStatus",
    "TenantSSOConfigDTO",
    "EvaluationClient",
    "EvaluationConfig",
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
    "CorrelationIds",
    "ObserveContext",
    "ObservabilityClient",
    "ObservabilityConfig",
    "ObservationDTO",
    "ObservationRecord",
    "ObservationType",
    "PaginationInfo",
    "SessionListResponse",
    "SessionRecord",
    "SessionDTO",
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
    "PromptManagementClient",
    "PromptManagementConfig",
    "PromptType",
    "ChatPromptMessage",
    "PromptSummary",
    "PromptDetail",
    "PromptVersionRecord",
    "PromptListResponse",
    "ResolvedPrompt",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# NOTE: Legacy builtins injection removed in v0.9.0
# Use explicit imports: `import traigent`
