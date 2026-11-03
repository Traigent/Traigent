# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
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
import logging
import os
import sys
import warnings
from importlib import import_module

_logger = logging.getLogger(__name__)


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# LiteLLM network-download pin (must run before anything below can import
# litellm, and before the quickstart block).
# ---------------------------------------------------------------------------
#
# LiteLLM fetches its model-cost table (and, for Anthropic, its beta-header
# config) from raw.githubusercontent.com the first time it is imported,
# unless LITELLM_LOCAL_MODEL_COST_MAP / LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS
# are already set. On a slow or blocked network that is a ~5s stall plus a
# warning on every `import traigent` that transitively reaches litellm
# (traigent.utils.cost_calculator, most provider integrations). Pin to the
# bundled tables by default; TRAIGENT_LITELLM_LIVE_PRICES=1 opts back into
# the live fetch (e.g. to pick up new-model prices before Traigent's next
# release). ``setdefault`` so an explicit user value always wins.
# litellm calls a bare load_dotenv() on import while LITELLM_MODE is "DEV" (its
# default), ignoring TRAIGENT_SKIP_DOTENV (#2470). Honor the documented opt-out
# here: LITELLM_MODE=PRODUCTION gates only that dotenv load in litellm, and an
# explicit user value still wins. Only applied when the user opted out of .env.
if _is_truthy_env(os.environ.get("TRAIGENT_SKIP_DOTENV")):
    os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

if not _is_truthy_env(os.environ.get("TRAIGENT_LITELLM_LIVE_PRICES")):
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    if "litellm" in sys.modules:
        # Too late for this process: litellm already ran its own
        # module-level import-time fetch before `import traigent` executed
        # (the caller imported litellm first), and reloading an
        # already-imported module here is not safe. The setdefault above
        # still lands so anything that reads the env var afterward (e.g.
        # `traigent doctor`) sees the pin; only litellm's own network
        # fetch already happened. Debug, not warning: this is expected
        # whenever litellm is a deliberate first import.
        _logger.debug(
            "traigent: litellm was already imported before `import "
            "traigent`, so the LITELLM_LOCAL_MODEL_COST_MAP / "
            "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS pin could not prevent "
            "its import-time network fetch. Import traigent before "
            "litellm, or set those two env vars yourself, to avoid it. "
            "The environment variables are documented in the traigent-skills "
            "repo (install it with `npx skills add Traigent/traigent-skills`)."
        )


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
    # ``python -m traigent.examples.quickstart`` and quickstart companion
    # modules — argv[0] is the path to the module's .py file.
    if argv0.endswith(
        (
            "quickstart/__main__.py",
            "quickstart\\__main__.py",
            "quickstart/publish_and_verify.py",
            "quickstart\\publish_and_verify.py",
        )
    ):
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
    # Usually a no-op: the LiteLLM pin above already set this unless the
    # user opted into TRAIGENT_LITELLM_LIVE_PRICES. Kept here so quickstart
    # stays fully offline-deterministic even when that opt-out is set —
    # the demo's own "no network" promise outranks the user's live-price
    # preference for this one invocation.
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


# Env-only mock activation (TRAIGENT_MOCK_LLM=true set before ``import
# traigent``, or the quickstart bootstrap above) must block raw provider HTTP
# from the first line of user code, not only once LiteLLM metadata patching
# runs lazily (#2418). Production is skipped here exactly as ``is_mock_llm()``
# refuses mock mode there (the hard-block in ``env_config`` is unchanged). The
# guard is installed only when the flag is on, so a normal import stays light
# and untouched; a failed install fails closed by aborting the import.
if _is_truthy_env(os.environ.get("TRAIGENT_MOCK_LLM")) and os.environ.get(
    "ENVIRONMENT", "development"
).strip().lower() not in {"prod", "production"}:
    try:
        from traigent.utils.mock_transport_guard import install_mock_transport_guard

        install_mock_transport_guard()
    except Exception as _guard_exc:
        raise RuntimeError(
            "TRAIGENT_MOCK_LLM is on but the mock-mode transport guard could "
            "not be installed; refusing to continue without it (a raw LLM "
            "provider call could otherwise go out)."
        ) from _guard_exc

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

_TRAIGENT_API_DECORATORS = "traigent.api.decorators"
_TRAIGENT_API_FUNCTIONS = "traigent.api.functions"
_TRAIGENT_API_PARAMETER_RANGES = "traigent.api.parameter_ranges"
_TRAIGENT_API_CONSTRAINTS = "traigent.api.constraints"
_TRAIGENT_API_TYPES = "traigent.api.types"
_TRAIGENT_API_VALIDATION_PROTOCOL = "traigent.api.validation_protocol"
_TRAIGENT_CONFIG_TYPES = "traigent.config.types"
_TRAIGENT_ADMIN = "traigent.admin"
_TRAIGENT_EVALUATION = "traigent.evaluation"
_TRAIGENT_OBSERVABILITY = "traigent.observability"
_TRAIGENT_PROMPTS = "traigent.prompts"
_TRAIGENT_UTILS_EXCEPTIONS = "traigent.utils.exceptions"
_TRAIGENT_UTILS_CONSTRAINTS = "traigent.utils.constraints"
_TRAIGENT_UTILS_CALLBACKS = "traigent.utils.callbacks"
_TRAIGENT_UTILS_VALIDATION = "traigent.utils.validation"
_TRAIGENT_CONTRACT = "traigent.contract"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Main decorator and configuration API
    "optimize": (_TRAIGENT_API_DECORATORS, "optimize"),
    # Runtime requirements + reactive policies (lazy: no network, no TVL compiler)
    "requirement": ("traigent.runtime.decorator", "requirement"),
    "ConfigSpace": ("traigent.api.config_space", "ConfigSpace"),
    "configure": (_TRAIGENT_API_FUNCTIONS, "configure"),
    "configure_for_budget": (_TRAIGENT_API_FUNCTIONS, "configure_for_budget"),
    "get_available_strategies": (
        _TRAIGENT_API_FUNCTIONS,
        "get_available_strategies",
    ),
    "get_config": (_TRAIGENT_API_FUNCTIONS, "get_config"),
    "get_current_config": (_TRAIGENT_API_FUNCTIONS, "get_current_config"),
    "get_optimization_insights": (
        _TRAIGENT_API_FUNCTIONS,
        "get_optimization_insights",
    ),
    "get_trial_config": (_TRAIGENT_API_FUNCTIONS, "get_trial_config"),
    "get_version_info": (_TRAIGENT_API_FUNCTIONS, "get_version_info"),
    "initialize": (_TRAIGENT_API_FUNCTIONS, "initialize"),
    "override_config": (_TRAIGENT_API_FUNCTIONS, "override_config"),
    "set_strategy": (_TRAIGENT_API_FUNCTIONS, "set_strategy"),
    "with_usage": (_TRAIGENT_API_FUNCTIONS, "with_usage"),
    # Experimental: shared cumulative execution budget (issue #1980)
    "ExecutionBudget": ("traigent.core.execution_budget", "ExecutionBudget"),
    # SE-friendly parameter range classes
    "Choices": (_TRAIGENT_API_PARAMETER_RANGES, "Choices"),
    "IntRange": (_TRAIGENT_API_PARAMETER_RANGES, "IntRange"),
    "LogRange": (_TRAIGENT_API_PARAMETER_RANGES, "LogRange"),
    "ParameterRange": (_TRAIGENT_API_PARAMETER_RANGES, "ParameterRange"),
    "Range": (_TRAIGENT_API_PARAMETER_RANGES, "Range"),
    "TextDocument": (_TRAIGENT_API_PARAMETER_RANGES, "TextDocument"),
    # TVL constraint system
    "AndCondition": (_TRAIGENT_API_CONSTRAINTS, "AndCondition"),
    "BoolExpr": (_TRAIGENT_API_CONSTRAINTS, "BoolExpr"),
    "Condition": (_TRAIGENT_API_CONSTRAINTS, "Condition"),
    "Constraint": (_TRAIGENT_API_CONSTRAINTS, "Constraint"),
    "ConstraintScopeError": (_TRAIGENT_API_CONSTRAINTS, "ConstraintScopeError"),
    "NotCondition": (_TRAIGENT_API_CONSTRAINTS, "NotCondition"),
    "OrCondition": (_TRAIGENT_API_CONSTRAINTS, "OrCondition"),
    "WhenBuilder": (_TRAIGENT_API_CONSTRAINTS, "WhenBuilder"),
    "constraints_to_callables": (
        _TRAIGENT_API_CONSTRAINTS,
        "constraints_to_callables",
    ),
    "implies": (_TRAIGENT_API_CONSTRAINTS, "implies"),
    "normalize_constraints": (_TRAIGENT_API_CONSTRAINTS, "normalize_constraints"),
    "require": (_TRAIGENT_API_CONSTRAINTS, "require"),
    "when": (_TRAIGENT_API_CONSTRAINTS, "when"),
    # Core result and validation types
    "ConfigurationComparison": (_TRAIGENT_API_TYPES, "ConfigurationComparison"),
    "OptimizationResult": (_TRAIGENT_API_TYPES, "OptimizationResult"),
    "OptimizationStatus": (_TRAIGENT_API_TYPES, "OptimizationStatus"),
    "ParetoFront": (_TRAIGENT_API_TYPES, "ParetoFront"),
    "PresetSelection": (_TRAIGENT_API_TYPES, "PresetSelection"),
    "SensitivityAnalysis": (_TRAIGENT_API_TYPES, "SensitivityAnalysis"),
    "StrategyConfig": (_TRAIGENT_API_TYPES, "StrategyConfig"),
    "TrialError": (_TRAIGENT_API_TYPES, "TrialError"),
    "TrialResult": (_TRAIGENT_API_TYPES, "TrialResult"),
    "serialize_trials": (_TRAIGENT_API_TYPES, "serialize_trials"),
    "ExternalServiceEvaluator": (
        _TRAIGENT_API_DECORATORS,
        "ExternalServiceEvaluator",
    ),
    "HybridAPIOptions": (_TRAIGENT_API_DECORATORS, "HybridAPIOptions"),
    "ConstraintValidator": (
        _TRAIGENT_API_VALIDATION_PROTOCOL,
        "ConstraintValidator",
    ),
    "ConstraintViolation": (
        _TRAIGENT_API_VALIDATION_PROTOCOL,
        "ConstraintViolation",
    ),
    "ConstraintValidationResult": (
        _TRAIGENT_API_VALIDATION_PROTOCOL,
        "ValidationResult",
    ),
    "PythonConstraintValidator": (
        _TRAIGENT_API_VALIDATION_PROTOCOL,
        "PythonConstraintValidator",
    ),
    "SatResult": (_TRAIGENT_API_VALIDATION_PROTOCOL, "SatResult"),
    "SatStatus": (_TRAIGENT_API_VALIDATION_PROTOCOL, "SatStatus"),
    # Context and config helpers
    "copy_context_to_thread": ("traigent.config.context", "copy_context_to_thread"),
    "get_trial_context": ("traigent.config.context", "get_trial_context"),
    "ExecutionIntent": (_TRAIGENT_CONFIG_TYPES, "ExecutionIntent"),
    "ExecutionMode": (_TRAIGENT_CONFIG_TYPES, "ExecutionMode"),
    "ResolvedExecutionPolicy": (
        _TRAIGENT_CONFIG_TYPES,
        "ResolvedExecutionPolicy",
    ),
    "TraigentConfig": (_TRAIGENT_CONFIG_TYPES, "TraigentConfig"),
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
    # Economics telemetry emitter (WI-B). Root-exported to match the schema-owned
    # Python/JS parity manifest, which classifies EconomicsTelemetryClient as a
    # `matched` root export, and the JS SDK, which root-exports it from
    # `@traigent/sdk`. Lazy (like the other client surfaces): plain `import
    # traigent` does not import `traigent.economics` until this symbol is accessed.
    "EconomicsTelemetryClient": (
        "traigent.economics",
        "EconomicsTelemetryClient",
    ),
    "EnterpriseAdminClient": (_TRAIGENT_ADMIN, "EnterpriseAdminClient"),
    "EnterpriseAdminConfig": (_TRAIGENT_ADMIN, "EnterpriseAdminConfig"),
    "EvaluationClient": (_TRAIGENT_EVALUATION, "EvaluationClient"),
    "EvaluationConfig": (_TRAIGENT_EVALUATION, "EvaluationConfig"),
    "ObservabilityClient": (_TRAIGENT_OBSERVABILITY, "ObservabilityClient"),
    "ObservabilityConfig": (_TRAIGENT_OBSERVABILITY, "ObservabilityConfig"),
    "ProjectManagementClient": ("traigent.projects", "ProjectManagementClient"),
    "ProjectManagementConfig": ("traigent.projects", "ProjectManagementConfig"),
    "PromptManagementClient": (_TRAIGENT_PROMPTS, "PromptManagementClient"),
    "PromptManagementConfig": (_TRAIGENT_PROMPTS, "PromptManagementConfig"),
    # Admin DTOs
    "SSOProviderType": (_TRAIGENT_ADMIN, "SSOProviderType"),
    "TenantDTO": (_TRAIGENT_ADMIN, "TenantDTO"),
    "TenantListResponse": (_TRAIGENT_ADMIN, "TenantListResponse"),
    "TenantMembershipDTO": (_TRAIGENT_ADMIN, "TenantMembershipDTO"),
    "TenantMembershipListResponse": (
        _TRAIGENT_ADMIN,
        "TenantMembershipListResponse",
    ),
    "TenantMembershipRole": (_TRAIGENT_ADMIN, "TenantMembershipRole"),
    "TenantMembershipStatus": (_TRAIGENT_ADMIN, "TenantMembershipStatus"),
    "TenantSSOConfigDTO": (_TRAIGENT_ADMIN, "TenantSSOConfigDTO"),
    # Evaluation DTOs
    "AnnotationQueueDTO": (_TRAIGENT_EVALUATION, "AnnotationQueueDTO"),
    "AnnotationQueueItemCompleteResultDTO": (
        _TRAIGENT_EVALUATION,
        "AnnotationQueueItemCompleteResultDTO",
    ),
    "AnnotationQueueItemCreateResultDTO": (
        _TRAIGENT_EVALUATION,
        "AnnotationQueueItemCreateResultDTO",
    ),
    "AnnotationQueueItemDTO": (_TRAIGENT_EVALUATION, "AnnotationQueueItemDTO"),
    "AnnotationQueueItemListResponse": (
        _TRAIGENT_EVALUATION,
        "AnnotationQueueItemListResponse",
    ),
    "AnnotationQueueItemStatus": (
        _TRAIGENT_EVALUATION,
        "AnnotationQueueItemStatus",
    ),
    "AnnotationQueueListResponse": (
        _TRAIGENT_EVALUATION,
        "AnnotationQueueListResponse",
    ),
    "AnnotationQueueStatus": (_TRAIGENT_EVALUATION, "AnnotationQueueStatus"),
    "BackfillResultDTO": (_TRAIGENT_EVALUATION, "BackfillResultDTO"),
    "Dataset": ("traigent.evaluators.base", "Dataset"),
    "EvaluationExample": ("traigent.evaluators.base", "EvaluationExample"),
    "EvaluationTargetRefDTO": (_TRAIGENT_EVALUATION, "EvaluationTargetRefDTO"),
    "EvaluationTargetType": (_TRAIGENT_EVALUATION, "EvaluationTargetType"),
    "EvaluatorDefinitionDTO": (_TRAIGENT_EVALUATION, "EvaluatorDefinitionDTO"),
    "EvaluatorListResponse": (_TRAIGENT_EVALUATION, "EvaluatorListResponse"),
    "EvaluatorRunDTO": (_TRAIGENT_EVALUATION, "EvaluatorRunDTO"),
    "EvaluatorRunListResponse": (_TRAIGENT_EVALUATION, "EvaluatorRunListResponse"),
    "EvaluatorRunStatus": (_TRAIGENT_EVALUATION, "EvaluatorRunStatus"),
    "JudgeConfigDTO": (_TRAIGENT_EVALUATION, "JudgeConfigDTO"),
    "MeasureValueType": (_TRAIGENT_EVALUATION, "MeasureValueType"),
    "ScoreRecordDTO": (_TRAIGENT_EVALUATION, "ScoreRecordDTO"),
    "ScoreRecordListResponse": (_TRAIGENT_EVALUATION, "ScoreRecordListResponse"),
    "ScoreSource": (_TRAIGENT_EVALUATION, "ScoreSource"),
    "TypedMeasureDTO": (_TRAIGENT_EVALUATION, "TypedMeasureDTO"),
    # Observability DTOs and helpers
    "CorrelationIds": (_TRAIGENT_OBSERVABILITY, "CorrelationIds"),
    "ObserveContext": (_TRAIGENT_OBSERVABILITY, "ObserveContext"),
    "ObservationDTO": (_TRAIGENT_OBSERVABILITY, "ObservationDTO"),
    "ObservationRecord": (_TRAIGENT_OBSERVABILITY, "ObservationRecord"),
    "ObservationType": (_TRAIGENT_OBSERVABILITY, "ObservationType"),
    "PaginationInfo": (_TRAIGENT_OBSERVABILITY, "PaginationInfo"),
    "SessionDTO": (_TRAIGENT_OBSERVABILITY, "SessionDTO"),
    "SessionListResponse": (_TRAIGENT_OBSERVABILITY, "SessionListResponse"),
    "SessionRecord": (_TRAIGENT_OBSERVABILITY, "SessionRecord"),
    "ThumbRating": (_TRAIGENT_OBSERVABILITY, "ThumbRating"),
    "TraceCollaborationState": (
        _TRAIGENT_OBSERVABILITY,
        "TraceCollaborationState",
    ),
    "TraceCommentRecord": (_TRAIGENT_OBSERVABILITY, "TraceCommentRecord"),
    "TraceCommentsResponse": (_TRAIGENT_OBSERVABILITY, "TraceCommentsResponse"),
    "TraceDTO": (_TRAIGENT_OBSERVABILITY, "TraceDTO"),
    "TraceFeedbackRecord": (_TRAIGENT_OBSERVABILITY, "TraceFeedbackRecord"),
    "TraceFeedbackResponse": (_TRAIGENT_OBSERVABILITY, "TraceFeedbackResponse"),
    "TraceFeedbackSummary": (_TRAIGENT_OBSERVABILITY, "TraceFeedbackSummary"),
    "TraceListResponse": (_TRAIGENT_OBSERVABILITY, "TraceListResponse"),
    "TraceObservationsResponse": (
        _TRAIGENT_OBSERVABILITY,
        "TraceObservationsResponse",
    ),
    "TraceRecord": (_TRAIGENT_OBSERVABILITY, "TraceRecord"),
    "get_default_observability_client": (
        _TRAIGENT_OBSERVABILITY,
        "get_default_observability_client",
    ),
    "SpanResult": (_TRAIGENT_OBSERVABILITY, "SpanResult"),
    "add_agent_span": (_TRAIGENT_OBSERVABILITY, "add_agent_span"),
    "observe": (_TRAIGENT_OBSERVABILITY, "observe"),
    "set_default_observability_client": (
        _TRAIGENT_OBSERVABILITY,
        "set_default_observability_client",
    ),
    # Prompt DTOs and helpers
    "ChatPromptMessage": (_TRAIGENT_PROMPTS, "ChatPromptMessage"),
    "PromptDetail": (_TRAIGENT_PROMPTS, "PromptDetail"),
    "PromptListResponse": (_TRAIGENT_PROMPTS, "PromptListResponse"),
    "PromptPlaygroundConfig": (_TRAIGENT_PROMPTS, "PromptPlaygroundConfig"),
    "PromptPlaygroundResult": (_TRAIGENT_PROMPTS, "PromptPlaygroundResult"),
    "PromptPlaygroundTokenUsage": (
        _TRAIGENT_PROMPTS,
        "PromptPlaygroundTokenUsage",
    ),
    "PromptSummary": (_TRAIGENT_PROMPTS, "PromptSummary"),
    "PromptType": (_TRAIGENT_PROMPTS, "PromptType"),
    "PromptVersionRecord": (_TRAIGENT_PROMPTS, "PromptVersionRecord"),
    "ResolvedPrompt": (_TRAIGENT_PROMPTS, "ResolvedPrompt"),
    # Utilities
    "ConfigAccessWarning": (_TRAIGENT_UTILS_EXCEPTIONS, "ConfigAccessWarning"),
    "ConstraintManager": (_TRAIGENT_UTILS_CONSTRAINTS, "ConstraintManager"),
    "DataIntegrityError": (_TRAIGENT_UTILS_EXCEPTIONS, "DataIntegrityError"),
    "DTOSerializationError": (
        _TRAIGENT_UTILS_EXCEPTIONS,
        "DTOSerializationError",
    ),
    "LoggingCallback": (_TRAIGENT_UTILS_CALLBACKS, "LoggingCallback"),
    "MetricExtractionError": (_TRAIGENT_UTILS_EXCEPTIONS, "MetricExtractionError"),
    "MultiObjectiveMetrics": (
        "traigent.utils.multi_objective",
        "MultiObjectiveMetrics",
    ),
    "OptimizationStateError": (
        _TRAIGENT_UTILS_EXCEPTIONS,
        "OptimizationStateError",
    ),
    "PlatformCapabilityError": (
        _TRAIGENT_UTILS_EXCEPTIONS,
        "PlatformCapabilityError",
    ),
    "OptimizationValidator": (_TRAIGENT_UTILS_VALIDATION, "OptimizationValidator"),
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
    "ProgressBarCallback": (_TRAIGENT_UTILS_CALLBACKS, "ProgressBarCallback"),
    "RetryConfig": ("traigent.utils.retry", "RetryConfig"),
    "StatisticsCallback": (_TRAIGENT_UTILS_CALLBACKS, "StatisticsCallback"),
    "TraigentDeprecationWarning": (
        _TRAIGENT_UTILS_EXCEPTIONS,
        "TraigentDeprecationWarning",
    ),
    "TraigentWarning": (_TRAIGENT_UTILS_EXCEPTIONS, "TraigentWarning"),
    "ValidationResult": (_TRAIGENT_UTILS_VALIDATION, "ValidationResult"),
    "SnapshotBindingError": (_TRAIGENT_UTILS_EXCEPTIONS, "SnapshotBindingError"),
    "VendorPauseError": (_TRAIGENT_UTILS_EXCEPTIONS, "VendorPauseError"),
    "create_quick_plot": ("traigent.visualization.plots", "create_quick_plot"),
    "get_default_callbacks": (_TRAIGENT_UTILS_CALLBACKS, "get_default_callbacks"),
    "get_verbose_callbacks": (_TRAIGENT_UTILS_CALLBACKS, "get_verbose_callbacks"),
    "max_tokens_constraint": (_TRAIGENT_UTILS_CONSTRAINTS, "max_tokens_constraint"),
    "model_cost_constraint": (_TRAIGENT_UTILS_CONSTRAINTS, "model_cost_constraint"),
    "retry": ("traigent.utils.retry", "retry"),
    "temperature_constraint": (
        _TRAIGENT_UTILS_CONSTRAINTS,
        "temperature_constraint",
    ),
    "validate_and_suggest": (_TRAIGENT_UTILS_VALIDATION, "validate_and_suggest"),
    # No-execution evaluation-compatibility contract
    "validate_evaluation_contract": (
        _TRAIGENT_CONTRACT,
        "validate_evaluation_contract",
    ),
    "EvaluationContractReport": (
        _TRAIGENT_CONTRACT,
        "EvaluationContractReport",
    ),
    "ContractFinding": (_TRAIGENT_CONTRACT, "ContractFinding"),
    "ContractCode": (_TRAIGENT_CONTRACT, "ContractCode"),
    "EVALUATION_CONTRACT_VERSION": (
        _TRAIGENT_CONTRACT,
        "EVALUATION_CONTRACT_VERSION",
    ),
}

_DEPRECATED_LAZY_EXPORTS = {
    "ExecutionIntent",
    "ExecutionMode",
    "ResolvedExecutionPolicy",
    "HybridAPIOptions",
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
    if name in _DEPRECATED_LAZY_EXPORTS:
        warnings.warn(
            f"traigent.{name} is a deprecated compatibility alias and is no "
            "longer part of the public root export surface. Import the internal "
            "compatibility symbol from its defining submodule only when needed.",
            DeprecationWarning,
            stacklevel=3,
        )
    return value


__all__ = [
    # Main decorator
    "optimize",
    "requirement",
    # SE-friendly parameter range classes
    "Range",
    "IntRange",
    "LogRange",
    "Choices",
    "TextDocument",
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
    "ExecutionBudget",  # Experimental: shared cumulative execution budget (#1980)
    # Configuration types
    "TraigentConfig",
    "ExternalServiceEvaluator",
    # Lifecycle and state
    "OptimizationState",
    # Thread context helpers
    "copy_context_to_thread",
    "CoreMetricsClient",
    "CoreMetricsConfig",
    "EconomicsTelemetryClient",
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
    "AnnotationQueueItemCompleteResultDTO",
    "AnnotationQueueItemCreateResultDTO",
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
    "TypedMeasureDTO",
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
    "SpanResult",
    "add_agent_span",
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
    "PlatformCapabilityError",
    "SnapshotBindingError",
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
    "PresetSelection",
    "TrialError",
    "TrialResult",
    "serialize_trials",
    "SensitivityAnalysis",
    "ConfigurationComparison",
    "ParetoFront",
    "OptimizationStatus",
    "StrategyConfig",
    # No-execution evaluation-compatibility contract
    "validate_evaluation_contract",
    "EvaluationContractReport",
    "ContractFinding",
    "ContractCode",
    "EVALUATION_CONTRACT_VERSION",
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
