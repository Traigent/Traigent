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


def _is_optimizer_cli_invocation() -> bool:
    if len(sys.argv) < 2 or sys.argv[1] != "optimizer":
        return False
    basename = os.path.basename(sys.argv[0] or "").lower()
    return basename in {"traigent", "traigent.exe"}


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

if _is_optimizer_cli_invocation():
    # Optimizer scan/decorate are static adoption helpers. They should not try
    # to refresh LiteLLM's remote model-cost map while merely parsing code.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


# Suppress noisy FutureWarning from transitive deps (instructor → google.generativeai)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"instructor\.providers\.gemini"
)

from traigent._version import get_version

__version__ = get_version()
__author__ = "Traigent Team"
__email__ = "opensource@traigent.ai"

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
from traigent.cloud.benchmark_client import BenchmarkClient, BenchmarkClientConfig

# Thread context helpers
from traigent.config.context import copy_context_to_thread, get_trial_context

# Configuration types
from traigent.config.types import TraigentConfig
from traigent.core.meta_types import TraigentMetadata, is_traigent_metadata

# Lifecycle and state management
from traigent.core.optimized_function import OptimizationState
from traigent.core_metrics import CoreMetricsClient, CoreMetricsConfig
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
from traigent.evaluators.base import Dataset, EvaluationExample
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
from traigent.projects import ProjectManagementClient, ProjectManagementConfig
from traigent.prompts import (
    ChatPromptMessage,
    PromptDetail,
    PromptListResponse,
    PromptManagementClient,
    PromptManagementConfig,
    PromptPlaygroundConfig,
    PromptPlaygroundResult,
    PromptPlaygroundTokenUsage,
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# NOTE: Legacy builtins injection removed in v0.9.0
# Use explicit imports: `import traigent`
