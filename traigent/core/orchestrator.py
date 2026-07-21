"""Optimization orchestration engine."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import copy
import inspect
import math
import os
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, cast

from traigent.api.agent_inference import (
    build_agent_configuration,
    extract_parameter_agents,
)
from traigent.api.strategy_presets import (
    NormalizedStrategyPreset,
    select_strategy_preset,
)
from traigent.api.types import (
    AgentConfiguration,
    AgentDefinition,
    OptimizationResult,
    OptimizationStatus,
    StopReason,
    TrialResult,
    TrialStatus,
)

# Type-only imports for optional dependencies
if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient
    from traigent.integrations.observability.workflow_traces import (
        SpanPayload,
        WorkflowTracesTracker,
    )

from traigent.config.types import ExecutionIntent, ExecutionMode, TraigentConfig
from traigent.core.backend_session_manager import (
    BackendSessionManager,
    session_aggregation_echoed,
)
from traigent.core.cache_policy import CachePolicyHandler
from traigent.core.cost_enforcement import (
    DEFAULT_COST_LIMIT_USD,
    CostEnforcer,
    CostEnforcerConfig,
    Permit,
    normalize_cost_approved,
    validate_cost_limit,
)
from traigent.core.cost_estimator import CostEstimator
from traigent.core.exception_handler import (
    VendorErrorCategory,
    classify_systematic_provider_failure,
    provider_failure_action_hint,
)
from traigent.core.execution_policy_runtime import (
    SOURCE_CLOUD_BRAIN,
    SOURCE_LOCAL_FALLBACK,
    CloudBrainUnavailableError,
    backend_optimization_strategy_for_algorithm,
    backend_egress_disabled,
    is_offline_requested,
    policy_from_config,
    policy_is_cloud_brain,
    policy_is_cloud_required,
    unsupported_backend_smart_algorithm_message,
)
from traigent.core.logger_facade import LoggerFacade
from traigent.core.metadata_helpers import merge_run_metrics_into_session_summary
from traigent.core.metric_registry import MetricRegistry, MetricSpec
from traigent.core.metrics_aggregator import (
    aggregate_metrics,
    build_safeguards_telemetry,
)
from traigent.core.objectives import ObjectiveSchema
from traigent.core.orchestrator_helpers import (
    allocate_parallel_ceilings,
    constraint_requires_metrics,
    normalize_parallel_trials,
    pre_trial_validate_config,
    prepare_evaluation_config,
    prepare_objectives,
    validate_constructor_arguments,
    validate_dataset,
)
from traigent.core.parallel_execution_manager import (
    ParallelExecutionManager,
    PermittedTrialResult,
)
from traigent.core.progress_manager import ProgressManager
from traigent.core.result_selection import (
    NO_RANKING_ELIGIBLE_TRIALS,
    TieBreaker,
    _primary_scores_tied,
    _secondary_metric_key,
    observed_metric_ranges,
    resolve_weighted_selection_schema,
    select_best_configuration,
)
from traigent.core.sample_budget import SampleBudgetManager
from traigent.core.stat_significance import compute_significance
from traigent.core.stop_condition_manager import StopConditionManager
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.core.utils import extract_examples_attempted
from traigent.core.workflow_trace_manager import WorkflowTraceManager
from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.metrics.registry import clone_registry
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.interactive_optimizer import CloudBrainOptimizationComplete
from traigent.tvl.promotion_gate import PromotionGate
from traigent.utils.callbacks import CallbackManager, OptimizationCallback, ProgressInfo
from traigent.utils.env_config import (  # noqa: F401
    is_backend_offline as is_backend_offline,
)
from traigent.utils.exceptions import (
    ConfigurationError,
    OptimizationError,
    VendorPauseError,
)
from traigent.utils.function_identity import (
    FunctionDescriptor,
    resolve_function_descriptor,
)
from traigent.utils.hashing import generate_run_label
from traigent.utils.logging import get_logger
from traigent.utils.objectives import (
    coerce_finite_objective_score,
    is_minimization_objective,
)
from traigent.utils.optimization_logger import OptimizationLogger

from .tracing import optimization_session_span, record_optimization_complete

logger = get_logger(__name__)

# Orchestrator constants
PROGRESS_LOG_INTERVAL = 10  # Log progress every N trials

# Stop reasons that already own an empty (0-trial) cloud-required run and must
# not be relabeled by the smart-managed-path fail-closed guard (issue #1681):
# an interrupted/timed-out/cancelled run, a cost-limit stop (#1684 owns that),
# or an explicit provider/connectivity error already surfaced elsewhere.
_EMPTY_SMART_RUN_OWNED_STOP_REASONS = frozenset(
    {
        "timeout",
        "user_cancelled",
        "cost_limit",
        "vendor_error",
        "network_error",
        "error",
    }
)
_OBJECTIVE_UNMATCHED_WARNING_CODE = "OBJECTIVE_UNMATCHED"
# Issue #1832: a declared, weighted, matched objective whose value is uniformly
# constant across the ranking-eligible trials cannot influence ``best_config`` —
# its weight is a silent no-op (e.g. cost/latency = 0 on a no-LLM-scored or
# free/unpriceable model run). Sibling of _OBJECTIVE_UNMATCHED_WARNING_CODE
# (#1691): same register/append/consume mechanism, warning-only (never fatal).
_OBJECTIVE_INERT_CONSTANT_WARNING_CODE = "OBJECTIVE_INERT_CONSTANT"
# Zero-span tolerance mirroring normalize_value()'s normative epsilon in
# objectives.py (``abs(max - min) < 1e-9``): an objective whose observed span is
# below this normalizes to a constant 0.5 on every trial, so its weight becomes a
# fixed additive term that cannot change the argmax. Using selection's OWN span
# test makes the #1832 inertness warning provably equivalent to actual inertness.
_OBJECTIVE_ZERO_SPAN_EPSILON = 1e-9


def _successful_trial_metric_names(trials: Sequence[TrialResult]) -> list[str]:
    metric_names: set[str] = set()
    for trial in trials:
        if not trial.is_successful:
            continue
        metrics = trial.metrics or {}
        metric_names.update(str(name) for name in metrics)
    return sorted(metric_names)


def _objective_metric_never_measured(
    trials: Sequence[TrialResult],
    objective: str,
) -> bool:
    saw_any_metric = False
    for trial in trials:
        if not trial.is_successful:
            continue
        metrics = trial.metrics or {}
        if metrics:
            saw_any_metric = True
        if objective in metrics:
            return False
    return saw_any_metric


def _format_unmatched_objective_warning(
    objective: str,
    available_metric_names: Sequence[str],
) -> tuple[str, str | None]:
    suggestion = None
    matches = get_close_matches(
        objective, list(available_metric_names), n=1, cutoff=0.6
    )
    if matches:
        suggestion = matches[0]

    available = ", ".join(available_metric_names) if available_metric_names else "none"
    suggestion_text = f" Did you mean '{suggestion}'?" if suggestion else ""
    return (
        f"No winner: objective metric '{objective}' was never measured in any "
        "successful trial; best_config is None. "
        f"Available metrics: {available}.{suggestion_text}",
        suggestion,
    )


def _detect_inert_constant_objectives(
    eligible_trials: Sequence[TrialResult],
    schema: ObjectiveSchema,
) -> list[str]:
    """Return declared weighted objectives that are inert because constant (#1832).

    An objective is inert when it is (a) declared with a nonzero normalized
    weight, (b) matched/measured — present and finite on *every* ranking-eligible
    trial — and (c) uniformly constant across that set (every trial's value equal
    within float tolerance). Such an objective's weight cannot change which config
    ranks best, so weighted selection silently collapses onto the remaining
    objective(s).

    ``eligible_trials`` MUST be the exact ranking-eligible set that selection
    ranked over (``SelectionResult.ranking_eligible_trial_ids``), not a re-derived
    set. Runs with fewer than two eligible trials are degenerate — every objective
    is trivially "constant" when there was no choice to make — and never warn.

    Constancy uses selection's OWN zero-span criterion — ``abs(max - min) < 1e-9``
    (``_OBJECTIVE_ZERO_SPAN_EPSILON``), the same absolute span test
    ``normalize_value`` (objectives.py) uses to collapse an objective to a constant
    0.5. This makes the warning provably equivalent to the objective being inert in
    selection: exact ``0.0``-valued objectives (the motivating case) are flagged,
    and a relatively-close-but-large-span objective is never falsely flagged.
    """
    if len(eligible_trials) < 2:
        return []
    inert: list[str] = []
    for obj in schema.objectives:
        if schema.get_normalized_weight(obj.name) <= 0:
            continue
        values: list[float] = []
        fully_matched = True
        for trial in eligible_trials:
            value = coerce_finite_objective_score(trial.get_metric(obj.name))
            if value is None:
                fully_matched = False
                break
            values.append(value)
        if not fully_matched or len(values) < 2:
            continue
        if abs(max(values) - min(values)) < _OBJECTIVE_ZERO_SPAN_EPSILON:
            inert.append(obj.name)
    return inert


def _format_inert_objective_warning(
    inert_objectives: Sequence[str],
    all_objective_names: Sequence[str],
    constant_value: float,
) -> str:
    inert_list = ", ".join(f"'{name}'" for name in inert_objectives)
    remaining = [name for name in all_objective_names if name not in inert_objectives]
    remaining_text = (
        ", ".join(f"'{name}'" for name in remaining)
        if remaining
        else "no other objectives"
    )
    plural = "objectives" if len(inert_objectives) > 1 else "objective"
    were = "were" if len(inert_objectives) > 1 else "was"
    return (
        f"Weighted {plural} {inert_list} {were} uniformly constant "
        f"(value {constant_value:g}) across every ranking-eligible trial, so the "
        "declared weight had no effect on best_config: the run effectively "
        f"optimized only {remaining_text}. This happens when an objective binds "
        "to a genuinely uniform measurement — e.g. cost/latency = 0 on a "
        "no-LLM-scored, free, or unpriceable-model run. Real priced runs whose "
        "cost/latency varies across configs are unaffected."
    )


class OptimizationOrchestrator:
    """Orchestrates the optimization process.

    Coordinates between optimizers and evaluators to run optimization trials,
    track progress, and manage the overall optimization lifecycle.
    """

    def __init__(
        self,
        optimizer: BaseOptimizer,
        evaluator: BaseEvaluator,
        max_trials: int | None = None,
        max_total_examples: int | None = None,
        timeout: float | None = None,
        callbacks: list[Callable[..., Any]] | None = None,
        config: TraigentConfig | None = None,
        parallel_trials: int | None = None,
        objectives: Sequence[str | None] | None = None,
        objective_schema: ObjectiveSchema | None = None,
        metric_registry: MetricRegistry | None = None,
        workflow_traces_tracker: WorkflowTracesTracker | None = None,
        use_versioned_logger: bool = False,
        file_version: str = "2",
        **kwargs: Any,
    ) -> None:
        """Initialize optimization orchestrator.

        Keyword Args:
            default_config: Optional baseline configuration evaluated once before
                optimizer-driven suggestions. Counts toward max_trials.
            workflow_traces_tracker: Optional tracker for collecting and submitting
                workflow spans to the backend for visualization.
        """

        self._initialized = False
        validate_constructor_arguments(
            optimizer,
            evaluator,
            max_trials,
            max_total_examples,
            timeout,
        )

        self.optimizer = optimizer
        self.evaluator = evaluator
        self._max_trials = max_trials
        self._max_total_examples = max_total_examples
        self.timeout = timeout
        self.traigent_config = config or TraigentConfig()
        self.config = kwargs
        self.parallel_trials = normalize_parallel_trials(parallel_trials)
        self.parallel_execution_manager = ParallelExecutionManager(
            parallel_trials=self.parallel_trials,
        )
        self.progress_manager = ProgressManager(
            total_trials=max_trials,
            algorithm_name=optimizer.__class__.__name__,
            objectives=(
                [obj for obj in objectives if obj is not None] if objectives else None
            ),
            log_interval=PROGRESS_LOG_INTERVAL,
        )

        raw_constraints = kwargs.pop("constraints", None)
        raw_safety_constraints = kwargs.pop("safety_constraints", None)
        default_config = kwargs.pop("default_config", None)
        combined_constraints = list(raw_constraints or [])
        combined_constraints.extend(raw_safety_constraints or [])
        self._init_constraints(combined_constraints)

        self.objectives, self.objective_schema = prepare_objectives(
            objectives, objective_schema
        )

        # TVL 0.9 tie-breaker and promotion gate configuration
        self._tie_breakers: dict[str, TieBreaker] = (
            kwargs.pop("tie_breakers", None) or {}
        )
        self._promotion_gate: PromotionGate | None = kwargs.pop("promotion_gate", None)
        self.strategy_preset: NormalizedStrategyPreset | None = kwargs.pop(
            "strategy_preset", None
        )
        self._warm_start_from: str | None = kwargs.pop("warm_start_from", None)
        self._smart_pruning: dict[str, Any] | None = kwargs.pop("smart_pruning", None)
        self._config_metrics_history: dict[str, dict[str, list[float]]] = {}
        self._incumbent_config_hash: str | None = None

        # Multi-agent configuration
        self._agent_configuration = self._init_agent_configuration(kwargs)

        # Derive band_target from objective_schema if available
        self._band_target = self._compute_band_target()

        self._default_config: dict[str, Any] | None = None
        self._default_config_used = False
        self._init_default_config(default_config)

        self.use_versioned_logger = use_versioned_logger
        self.file_version = file_version

        self._configure_evaluator_execution_mode()

        self.callback_manager = CallbackManager(
            cast(list[OptimizationCallback] | None, callbacks) if callbacks else None
        )
        self.metric_registry = (
            metric_registry.clone() if metric_registry is not None else clone_registry()
        )

        self._workflow_traces_tracker: WorkflowTracesTracker | None = (
            workflow_traces_tracker
        )

        self._backend_client: BackendIntegratedClient | None = None
        self.backend_client = self._initialize_backend_client()
        self._initialize_runtime_state()
        self.artifact_fingerprints: dict[str, str | None] | None = None
        self.fingerprint_meta: dict[str, Any] | None = None
        self.evaluator_definition_id: str | None = None

        # Interactive pause prompt adapter (None in non-interactive environments)
        from traigent.core.exception_handler import (
            PausePromptAdapter,
            TerminalPausePrompt,
        )

        self._prompt_adapter: PausePromptAdapter | None = (
            TerminalPausePrompt() if sys.stdin.isatty() else None
        )

        # Workflow trace manager for span collection and backend submission
        self._workflow_trace_manager = WorkflowTraceManager(
            workflow_traces_tracker=workflow_traces_tracker,
            backend_client=self.backend_client,
            function_descriptor=None,  # Set later in _initialize_optimization_run
            optimizer_config_space=(
                self.optimizer.config_space if self.optimizer.config_space else {}
            ),
            max_trials=self.max_trials,
            optimizer_class_name=self.optimizer.__class__.__name__,
            optimization_id=self._optimization_id,
        )
        self._register_declared_workflow_trace_nodes()

        self.backend_session_manager = BackendSessionManager(
            backend_client=self.backend_client,
            traigent_config=self.traigent_config,
            objectives=self.objectives,
            objective_schema=self.objective_schema,
            optimizer=self.optimizer,
            optimization_id=self._optimization_id,
            optimization_status=self._status,
            strategy_preset_metadata=(
                self.strategy_preset.to_metadata()
                if self.strategy_preset is not None
                else None
            ),
            smart_pruning=self._smart_pruning,
        )

        self.cache_policy_handler = CachePolicyHandler(
            traigent_config=self.traigent_config,
            optimizer=self.optimizer,
        )
        self._function_descriptor: FunctionDescriptor | None = None

        logger.debug(
            f"Created orchestrator with {optimizer.__class__.__name__} "
            f"and {evaluator.__class__.__name__}, {len(callbacks or [])} callbacks"
        )

        self._configure_stop_conditions()
        estimated_input_tokens, estimated_output_tokens = (
            self._extract_estimated_tokens_per_example(
                getattr(self.evaluator, "optimization_spec", None)
            )
        )

        self._cost_estimator = CostEstimator(
            cost_enforcer=self.cost_enforcer,
            max_trials=self._max_trials,
            max_total_examples=self._max_total_examples,
            model_name=self.traigent_config.model,
            candidate_models=self._extract_model_candidates_from_config_space(
                getattr(self.optimizer, "config_space", None)
            ),
            estimated_input_tokens_per_example=estimated_input_tokens,
            estimated_output_tokens_per_example=estimated_output_tokens,
        )

        self._trial_lifecycle = TrialLifecycle(self)
        self._initialized = True

    @staticmethod
    def _extract_raw_model_candidates(definition: Any) -> Sequence[Any] | None:
        """Extract raw candidate values from a config-space definition."""
        if isinstance(definition, str):
            return (definition,)
        if isinstance(definition, Sequence) and not isinstance(
            definition, (str, bytes, bytearray)
        ):
            return definition
        if not isinstance(definition, dict):
            return None

        values = definition.get("values")
        if isinstance(values, str):
            return (values,)
        if isinstance(values, Sequence) and not isinstance(
            values, (str, bytes, bytearray)
        ):
            return values
        return None

    @staticmethod
    def _normalize_model_candidates(
        raw_candidates: Sequence[Any] | None,
    ) -> tuple[str, ...]:
        """Normalize candidate model values into a deduplicated tuple."""
        if raw_candidates is None:
            return ()
        return tuple(
            dict.fromkeys(
                candidate.strip()
                for candidate in raw_candidates
                if isinstance(candidate, str) and candidate.strip()
            )
        )

    @staticmethod
    def _extract_model_candidates_from_config_space(
        config_space: dict[str, Any] | None,
    ) -> tuple[str, ...]:
        """Extract model candidates from optimizer config space for estimation."""
        if not isinstance(config_space, dict):
            return ()

        for key in ("model", "model_name"):
            normalized_candidates = (
                OptimizationOrchestrator._normalize_model_candidates(
                    OptimizationOrchestrator._extract_raw_model_candidates(
                        config_space.get(key)
                    )
                )
            )
            if normalized_candidates:
                return normalized_candidates

        return ()

    @staticmethod
    def _extract_estimated_tokens_per_example(
        optimization_spec: dict[str, Any] | None,
    ) -> tuple[int | None, int | None]:
        """Extract per-example token estimate from hybrid optimization metadata."""
        if not isinstance(optimization_spec, dict):
            return (None, None)

        estimate = optimization_spec.get("estimated_tokens_per_example")
        if not isinstance(estimate, dict):
            return (None, None)

        def _normalize(key: str) -> int | None:
            value = estimate.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
            return None

        return (_normalize("input_tokens"), _normalize("output_tokens"))

    def _init_constraints(
        self, raw_constraints: list[Callable[..., bool]] | None
    ) -> None:
        """Initialize pre and post evaluation constraints."""
        self._constraints_pre_eval: list[Callable[..., bool]] = []
        self._constraints_post_eval: list[Callable[..., bool]] = []
        if not raw_constraints:
            return
        for constraint in raw_constraints:
            if constraint_requires_metrics(constraint):
                self._constraints_post_eval.append(constraint)
            else:
                self._constraints_pre_eval.append(constraint)

    def _init_agent_configuration(
        self, kwargs: dict[str, Any]
    ) -> AgentConfiguration | None:
        """Initialize multi-agent configuration from kwargs."""
        explicit_agents: dict[str, AgentDefinition] | None = kwargs.pop("agents", None)
        agent_prefixes: list[str] | None = kwargs.pop("agent_prefixes", None)
        self._declared_trace_agents = dict(explicit_agents or {})
        self._declared_trace_agent_prefixes = list(agent_prefixes or [])
        agent_measures: dict[str, list[str]] | None = kwargs.pop("agent_measures", None)
        global_measures: list[str] | None = kwargs.pop("global_measures", None)
        tvl_parameter_agents: dict[str, str] | None = kwargs.pop(
            "tvl_parameter_agents", None
        )

        has_multi_agent_config = any(
            [
                explicit_agents,
                agent_prefixes,
                agent_measures,
                global_measures,
                tvl_parameter_agents,
            ]
        )
        if not has_multi_agent_config:
            return None

        parameter_agents = extract_parameter_agents(self.optimizer.config_space)
        if tvl_parameter_agents:
            merged_agents = dict(tvl_parameter_agents)
            merged_agents.update(parameter_agents)
            parameter_agents = merged_agents

        return cast(
            AgentConfiguration | None,
            build_agent_configuration(
                configuration_space=self.optimizer.config_space,
                explicit_agents=explicit_agents,
                agent_prefixes=agent_prefixes,
                agent_measures=agent_measures,
                global_measures=global_measures,
                parameter_agents=parameter_agents,
            ),
        )

    @staticmethod
    def _workflow_node_type_for_agent(agent: AgentDefinition) -> str:
        """Map public agent types to workflow graph node types."""

        if agent.agent_type in {"llm", "retriever", "router", "tool"}:
            return cast(str, agent.agent_type)
        return "agent"

    def _register_declared_workflow_trace_nodes(self) -> None:
        """Register declared multi-agent nodes in the workflow trace graph."""

        manager = getattr(self, "_workflow_trace_manager", None)
        if manager is None:
            return

        registered: set[str] = set()
        if self._agent_configuration is not None:
            for agent_id, agent in self._agent_configuration.agents.items():
                manager.register_node(
                    agent_id,
                    node_type=self._workflow_node_type_for_agent(agent),
                    display_name=agent.display_name,
                    tunable_params=list(agent.parameter_keys),
                    metadata={
                        "source": "declared_agent",
                        "measure_ids": list(agent.measure_ids),
                        "primary_model": agent.primary_model,
                        "order": agent.order,
                        "auto_inferred": self._agent_configuration.auto_inferred,
                    },
                )
                registered.add(agent_id)

        for prefix in getattr(self, "_declared_trace_agent_prefixes", []):
            if prefix in registered:
                continue
            manager.register_node(
                str(prefix),
                node_type="agent",
                metadata={"source": "agent_prefixes"},
            )

    def _compute_band_target(self) -> float | None:
        """Derive band_target from objective_schema if available."""
        if self.objective_schema is None or not self.objective_schema.objectives:
            return None
        primary_obj = self.objective_schema.objectives[0]
        if not hasattr(primary_obj, "band") or primary_obj.band is None:
            return None
        band = primary_obj.band
        if band.center is not None:
            return float(band.center)
        if band.low is not None and band.high is not None:
            return float((band.low + band.high) / 2.0)
        return None

    def _init_default_config(self, default_config: Any) -> None:
        """Initialize default config from provided value."""
        if isinstance(default_config, dict) and default_config:
            self._default_config = copy.deepcopy(default_config)
        elif default_config is not None:
            logger.debug(
                "Ignoring default_config with unexpected type: %s",
                type(default_config).__name__,
            )

    @staticmethod
    def _coerce_positive_limit(value: Any, *, name: str) -> float:
        limit = float(value)
        if limit <= 0:
            raise ValueError(f"{name} must be a positive number")
        return limit

    def _resolve_metric_limit_config(self) -> tuple[float | None, str | None, bool]:
        """Resolve metric_limit config."""
        raw_metric_limit = self.config.get("metric_limit")
        metric_include_pruned = bool(self.config.get("metric_include_pruned", True))

        if raw_metric_limit is not None:
            metric_name = self.config.get("metric_name")
            if metric_name is None:
                raise ValueError("metric_name is required when metric_limit is set")
            return (
                self._coerce_positive_limit(raw_metric_limit, name="metric_limit"),
                str(metric_name),
                metric_include_pruned,
            )

        return (None, None, metric_include_pruned)

    def _setup_convergence_condition(self) -> None:
        """Configure hypervolume-based convergence if specified in config."""
        convergence_metric = self.config.get("convergence_metric")
        convergence_window = self.config.get("convergence_window")
        convergence_threshold = self.config.get("convergence_threshold")

        if convergence_metric != "hypervolume_improvement":
            return
        if convergence_window is None or convergence_threshold is None:
            return
        if self.objective_schema is None:
            return

        directions = [obj.orientation for obj in self.objective_schema.objectives]
        if "band" in directions:
            logger.info(
                "Skipping hypervolume convergence: band objectives are not "
                "compatible with hypervolume computation"
            )
            return

        objective_names = [obj.name for obj in self.objective_schema.objectives]
        self._stop_condition_manager.add_convergence_condition(
            window=int(convergence_window),
            threshold=float(convergence_threshold),
            objective_names=objective_names,
            directions=cast(list[str], directions),
        )

    def _setup_cost_enforcer(self) -> None:
        """Initialize cost enforcer for cost limit enforcement."""
        cost_limit = self.config.get("cost_limit")
        cost_approved = normalize_cost_approved(self.config.get("cost_approved", False))
        cost_config = None
        if cost_limit is not None or cost_approved:
            # None means "use the default"; any provided value (including a
            # falsy 0) must pass config-time validation instead of being
            # silently coerced to the default (issue #1684 item 3).
            cost_config = CostEnforcerConfig(
                limit=(
                    DEFAULT_COST_LIMIT_USD
                    if cost_limit is None
                    else float(validate_cost_limit(cost_limit))
                ),
                approved=cost_approved,
            )
        self.cost_enforcer = CostEnforcer(config=cost_config)
        self.parallel_execution_manager.set_cost_enforcer(self.cost_enforcer)
        self._stop_condition_manager.register_cost_limit_condition(self.cost_enforcer)

    def _configure_stop_conditions(self) -> None:
        """Configure stop conditions and sample budget management."""
        plateau_window = int(self.config.get("plateau_window", 0) or 0)
        plateau_epsilon = (
            float(self.config.get("plateau_epsilon", 1e-6) or 1e-6)
            if plateau_window > 0
            else None
        )
        samples_include_pruned = bool(self.config.get("samples_include_pruned", True))
        metric_limit, metric_name, metric_include_pruned = (
            self._resolve_metric_limit_config()
        )

        self._stop_condition_manager = StopConditionManager(
            max_trials=self._max_trials,
            max_samples=self._max_total_examples,
            samples_include_pruned=samples_include_pruned,
            plateau_window=plateau_window or None,
            plateau_epsilon=plateau_epsilon,
            objective_schema=self.objective_schema,
            metric_limit=metric_limit,
            metric_name=metric_name,
            metric_include_pruned=metric_include_pruned,
            semantic_saturation=self.config.get("semantic_saturation"),
        )

        self._setup_convergence_condition()
        self._setup_cost_enforcer()

        self._samples_include_pruned = samples_include_pruned
        self._sample_budget_manager: SampleBudgetManager | None = (
            SampleBudgetManager(
                self._max_total_examples,
                include_pruned=self._samples_include_pruned,
            )
            if self._max_total_examples is not None
            else None
        )

    @property
    def backend_client(self) -> BackendIntegratedClient | None:
        """Return the active backend client (may be None for local-only runs)."""

        return self._backend_client

    @backend_client.setter
    def backend_client(self, client: BackendIntegratedClient | None) -> None:
        self._backend_client = client
        if hasattr(self, "backend_session_manager") and self.backend_session_manager:
            self.backend_session_manager.update_backend_client(client)

    @backend_client.deleter
    def backend_client(self) -> None:
        self._backend_client = None
        if hasattr(self, "backend_session_manager") and self.backend_session_manager:
            self.backend_session_manager.update_backend_client(None)

    @property
    def trial_count(self) -> int:
        """Get the number of completed trials."""
        return len(self._trials)

    @property
    def agent_configuration(self) -> AgentConfiguration | None:
        """Get the agent configuration for multi-agent experiments.

        Returns None for single-agent experiments (no grouping needed).
        """
        return self._agent_configuration

    def _configure_evaluator_execution_mode(self) -> None:
        if hasattr(self.evaluator, "execution_mode"):
            self.evaluator.execution_mode = self.traigent_config.execution_mode

        # Register hybrid lifecycle manager for cleanup if present
        # (HybridAPIEvaluator exposes lifecycle_manager property)
        self._hybrid_lifecycle_manager = getattr(
            self.evaluator, "lifecycle_manager", None
        )

    def _initialize_backend_client(self) -> BackendIntegratedClient | None:
        """Initialize backend client. Delegates to BackendSessionManager."""
        if backend_egress_disabled(self.traigent_config):
            return None
        return BackendSessionManager.create_backend_client(self.traigent_config)

    def _is_cloud_brain_run(self) -> bool:
        """Whether this orchestrator is running cloud-brain guidance."""

        policy = policy_from_config(self.traigent_config)
        return bool(
            (policy_is_cloud_brain(policy) or policy_is_cloud_required(policy))
            and getattr(self.traigent_config, "result_source", None)
            == SOURCE_CLOUD_BRAIN
            and not backend_egress_disabled(self.traigent_config)
        )

    def _backend_optimization_strategy_for_run(self) -> dict[str, str] | None:
        """Return the backend strategy for named managed algorithms."""

        policy = policy_from_config(self.traigent_config)
        if not policy_is_cloud_required(policy):
            return None
        strategy = backend_optimization_strategy_for_algorithm(policy.algorithm)
        if strategy is None:
            raise ConfigurationError(
                unsupported_backend_smart_algorithm_message(policy.algorithm)
            )
        return strategy

    def _optimizer_uses_remote_guidance(self) -> bool:
        """Whether the active optimizer would call remote next-trial guidance."""

        return bool(
            getattr(self.optimizer, "remote_service", None) is not None
            and callable(getattr(self.optimizer, "get_next_suggestion", None))
        )

    def _local_optimizer_planned_trials(self) -> int | None:
        """Trial count the LOCAL optimizer plans to run when unconstrained.

        Grid exposes ``total_combinations`` (the exhaustive cartesian size);
        random exposes its configured ``max_trials``. Returns None when the
        optimizer declares neither.
        """
        total = getattr(self.optimizer, "total_combinations", None)
        if isinstance(total, int) and not isinstance(total, bool) and total > 0:
            return total
        configured = getattr(self.optimizer, "max_trials", None)
        if (
            isinstance(configured, int)
            and not isinstance(configured, bool)
            and configured > 0
        ):
            return configured
        return None

    def _planned_backend_max_trials(self) -> int | None:
        """Wire ``max_trials`` for backend session-create.

        #1938: for connected local-decision runs (grid/random with egress
        enabled) the backend is a tracking sink, not the sequencer — the wire
        budget must equal the PLANNED local trial count so server-side budget
        logic never under- or over-states the plan. The plan is the local
        optimizer's own count (exhaustive grid size / configured random count)
        bounded by the run's ``max_trials`` cap when one is set. Cloud-brain /
        cloud-required runs pass the user's value through untouched.
        """
        policy = policy_from_config(self.traigent_config)
        if (
            policy is None
            or policy.intent is not ExecutionIntent.LOCAL_ONLY
            or policy.offline
        ):
            return self.max_trials
        planned = self._local_optimizer_planned_trials()
        if planned is None:
            return self.max_trials
        if self.max_trials is None:
            return planned
        return min(self.max_trials, planned)

    def _initialize_runtime_state(self) -> None:
        self._trials: list[TrialResult] = []
        # The backend session id for the active run. Read by the
        # certified-selection report guard to verify the incumbent's trial id
        # was backend-acknowledged before attesting a winner.
        self._active_session_id: str | None = None
        # RFC 0001 (knob bindings): optional resolver injecting Fixed/CVAR
        # values post-suggest, pre-validation. None => byte-identical legacy
        # behavior. Set via `orchestrator.knob_resolver = KnobResolver(...)`.
        self.knob_resolver: Any | None = None
        self._strict_withheld_promotions: list[str] = []
        self._certified_promotions = 0
        self._start_time: float | None = None
        self._status = OptimizationStatus.PENDING
        self._optimization_id = str(uuid.uuid4())
        self._stop_reason: StopReason | None = None
        # #1404: bounded auto-retry budget for transient vendor errors (429/503)
        # before the run loop gives up. Cumulative per category so the run always
        # terminates. Opt-in via TRAIGENT_VENDOR_MAX_RETRIES (default 0 = off).
        self._vendor_retry_counts: dict[VendorErrorCategory, int] = {}
        self._provider_call_attempts = 0
        self._provider_call_failures = 0
        self._provider_consecutive_call_failures = 0
        self._provider_failure_first_error: str | None = None
        self._provider_failure_category: str | None = None
        self._session_finalized = False
        self._logger: OptimizationLogger | None = None
        self._logger_v2: Any | None = None
        self._logger_facade = LoggerFacade()
        self._trials_prevented = 0
        self._examples_capped = 0
        self._cached_results_reused = 0
        self._cloud_guidance_client: Any | None = None
        self._ci_blocks = 0
        self._successful_trials = 0
        self._failed_trials = 0
        self._best_trial_cached: TrialResult | None = None
        self._consumed_examples = 0
        # Lock for protecting shared state mutations during parallel trial execution
        self._state_lock = asyncio.Lock()

    def _consume_default_config(self) -> dict[str, Any] | None:
        """Return the default config once to seed a baseline trial."""
        if self._default_config_used:
            return None

        self._default_config_used = True
        if not self._default_config:
            return None

        # Keep optimizer trial counts aligned with max_trials when applicable.
        # Some optimizers (random/optuna) track an internal _trial_count and can
        # stop early if it isn't incremented for the baseline trial.
        if hasattr(self.optimizer, "_trial_count"):
            try:
                self.optimizer._trial_count += 1  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Unable to increment optimizer trial count for baseline.")

        return copy.deepcopy(self._default_config)

    @property
    def status(self) -> OptimizationStatus:
        """Get the current optimization status."""
        return self._status

    @property
    def optimization_id(self) -> str:
        """Get the unique optimization ID."""
        return self._optimization_id

    @property
    def progress(self) -> float:
        """Get optimization progress as a percentage (0.0 to 1.0)."""
        if self.max_trials is None:
            # Unbounded runs have indeterminate completion percentage.
            return 0.0
        if self.max_trials == 0:
            # Explicit no-op optimization is immediately complete.
            return 1.0

        return min(self.trial_count / self.max_trials, 1.0)

    @property
    def best_result(self) -> TrialResult | None:
        """Get the best trial result so far."""
        if self._best_trial_cached is not None:
            return self._best_trial_cached

        rankable_trials = [trial for trial in self._trials if trial.is_successful]
        if not rankable_trials:
            return None

        # Find trial with best primary objective (assuming first objective is primary)
        if self.optimizer.objectives:
            primary_objective = self.optimizer.objectives[0]
            # Honor the declared orientation (issue #1959) so a minimize
            # objective whose name misses the heuristic patterns (e.g. 'brier',
            # 'spend', 'perplexity') is not misclassified as maximize.
            minimization = is_minimization_objective(
                primary_objective,
                orientation=(
                    self.objective_schema.get_orientation(primary_objective)
                    if self.objective_schema
                    else None
                ),
            )
            scored_trials = [
                (trial, score)
                for trial in rankable_trials
                if (
                    score := coerce_finite_objective_score(
                        trial.metrics.get(primary_objective)
                    )
                )
                is not None
            ]
            if not scored_trials:
                return None
            if minimization:
                best_trial = min(scored_trials, key=lambda item: item[1])[0]
            else:
                best_trial = max(scored_trials, key=lambda item: item[1])[0]
            self._best_trial_cached = best_trial
            return best_trial

        # If no objectives are defined, rankable_trials is known non-empty.
        return rankable_trials[-1]

    def _get_config_hash(self, config: dict[str, Any] | None) -> str:
        """Generate a hash for a configuration to track metrics across trials."""
        from traigent.utils.hashing import generate_config_hash

        result: str = generate_config_hash(config or {})
        return result

    def _track_trial_metrics(self, trial_result: TrialResult) -> str:
        """Track metrics for a trial result, grouped by config hash.

        Returns:
            The config hash for this trial.
        """
        config_hash = self._get_config_hash(trial_result.config)

        if config_hash not in self._config_metrics_history:
            self._config_metrics_history[config_hash] = {}

        # Add this trial's metrics to the history
        for metric_name, metric_value in (trial_result.metrics or {}).items():
            if isinstance(metric_value, (int, float)):
                if metric_name not in self._config_metrics_history[config_hash]:
                    self._config_metrics_history[config_hash][metric_name] = []
                self._config_metrics_history[config_hash][metric_name].append(
                    float(metric_value)
                )

        return config_hash

    def _has_sufficient_samples(
        self,
        candidate_metrics: dict[str, list[float]],
        incumbent_metrics: dict[str, list[float]],
        min_samples: int = 2,
    ) -> bool:
        """Check if both configs have enough samples for statistical comparison.

        Requires ALL objectives to have sufficient samples. This is intentional:
        epsilon-Pareto dominance testing requires complete metric data for all
        objectives to produce valid multi-objective comparisons. Partial data
        would lead to incorrect dominance conclusions.
        """
        objectives = self.optimizer.objectives or []
        return all(
            len(candidate_metrics.get(obj, [])) >= min_samples
            and len(incumbent_metrics.get(obj, [])) >= min_samples
            for obj in objectives
        )

    def _is_strict_evidence_mode(self) -> bool:
        """RFC 0001 §3.6 strict(M, c): the declared strict evidence modes.

        Disjuncts: promotion_policy.require_calibration ∨ chance_constraints
        ∨ a declared-governed CVAR in the resolved space (per-CVAR
        require_calibration, declaration-pinned certificate, or a
        certificate-backed / guaranteed-selection target). The per-CVAR
        disjuncts are GATE-INDEPENDENT: declaring a governed CVAR demands
        certificate-backed promotion evidence even when no promotion policy
        was configured. Governance is read from the resolver's DECLARED
        bindings (``has_governed_cvars``) — never inferred from runtime
        evidence; a duck-typed custom resolver without that method
        contributes no per-CVAR disjunct (declare strictness via the
        promotion policy in that case).
        """
        resolver = getattr(self, "knob_resolver", None)
        if resolver is not None:
            has_governed = getattr(resolver, "has_governed_cvars", None)
            if callable(has_governed) and has_governed():
                return True
        gate = self._promotion_gate
        if gate is None:
            return False
        policy = getattr(gate, "policy", None)
        if policy is None:
            return False
        if isinstance(policy, dict):
            # Defensive normalization: a raw dict policy must not silently
            # read as non-strict (it would then fail OPEN on gate errors).
            raw_policy = policy
            try:
                from traigent.tvl.models import PromotionPolicy

                policy = PromotionPolicy.from_dict(raw_policy)
            except Exception:
                # Unparseable policy declaring strict keys ⇒ fail CLOSED.
                return bool(
                    raw_policy.get("require_calibration")
                    or raw_policy.get("chance_constraints")
                )
        require_calibration = getattr(policy, "require_calibration", None)
        if require_calibration is not None and getattr(
            require_calibration, "enabled", False
        ):
            return True
        return bool(getattr(policy, "chance_constraints", None))

    def _build_session_objectives_payload(self) -> list[Any]:
        """Objectives for the session-create wire payload.

        With a declared ``ObjectiveSchema``, emit weighted objective dicts
        (``{"name", "orientation", "weight"}``) so the backend receives the
        user's weights (Traigent#1715); otherwise fall back to the bare
        metric-name list exactly as before.
        """
        if self.objective_schema is not None and self.objective_schema.objectives:
            return [
                {
                    "name": objective.name,
                    "orientation": objective.orientation,
                    "weight": objective.weight,
                }
                for objective in self.objective_schema.objectives
            ]
        return list(self.optimizer.objectives or [])

    def _build_session_default_config_payload(self) -> dict[str, Any] | None:
        """``default_config`` for the session-create wire payload.

        ``@optimize(default_config=...)`` is materialized locally (see
        ``_init_default_config`` / ``_consume_default_config``) but was never
        placed on the session-create payload, so backend warm-start seed
        projection never saw the user's declared baseline and the
        persisted/returned ``default_config`` was always empty. Peek at the
        stored value (do not consume it — consumption is reserved for
        seeding the actual baseline trial) and project it read-only, exactly
        as declared, omitting it entirely when unset so the payload stays
        byte-identical for runs without a default_config.
        """
        if not self._default_config:
            return None
        return copy.deepcopy(self._default_config)

    def _build_wire_governance(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Project the declared governance onto the wire (RFC 0001 P8).

        promotion_policy: from the promotion gate's declared policy.
        tvl_governance: cvar names/types/governed flags from the resolver's
        DECLARED bindings (the same source _is_strict_evidence_mode trusts) —
        never runtime or calibrated state. Both None for ungoverned sessions
        so their wire payload stays byte-identical.
        """
        from traigent.cloud.governance import (
            build_tvl_governance,
            promotion_policy_to_wire,
        )

        gate = self._promotion_gate
        policy = getattr(gate, "policy", None) if gate is not None else None
        wire_policy = promotion_policy_to_wire(policy)

        wire_governance = None
        resolver = getattr(self, "knob_resolver", None)
        space = getattr(resolver, "_space", None) if resolver is not None else None
        if space is not None:
            wire_governance = build_tvl_governance(space)
        return wire_policy, wire_governance

    def _build_certified_selection_report(self) -> dict[str, Any] | None:
        """Phase 8: the client-attested certified-selection finalize report.

        Built ONLY when every condition holds (mirror of the server's
        fail-closed rules so a 400 means a bug, not a flow): strict evidence
        mode, a certified incumbent exists, the resolver's DECLARED governed
        cvars are each backed by a CERTIFIED certificate with an issued
        hash. Any gap ⇒ None ⇒ the honest no-winner finalize. The incumbent's
        trial_id is the BACKEND id (trials are submitted under it), so the
        server can bind the winner to its own record.
        """
        if not self._is_strict_evidence_mode():
            return None
        # The FIRST trial seeds the incumbent as comparison initialization,
        # NOT as certification — terminal strict selection only names a
        # winner after >=1 explicit gate promotion (the _certified_promotions
        # guard in result assembly). Mirror it here so the wire report can
        # never overclaim a winner the SDK result itself refuses to certify.
        # An absent attribute reads 0 ⇒ fail closed.
        if not getattr(self, "_certified_promotions", 0):
            return None
        incumbent = self._best_trial_cached
        if incumbent is None or not getattr(incumbent, "trial_id", None):
            return None
        # The report binds the winner to its BACKEND trial id. If the
        # incumbent's id was never minted+acknowledged by the backend (the
        # slot request or result submission failed), the server cannot bind
        # it — sending a report would be a 400 at best and an attestation of
        # an unbound winner at worst. Fail closed: send NO report (the
        # risk-register rule for the unbindable case).
        manager = getattr(self, "backend_session_manager", None)
        ack = (
            getattr(manager, "is_trial_backend_acknowledged", None) if manager else None
        )
        session_id = getattr(self, "_active_session_id", None)
        if callable(ack):
            if session_id is None or not ack(session_id, incumbent.trial_id):
                logger.debug(
                    "certified_selection withheld: incumbent trial %s is not "
                    "backend-acknowledged (unbindable)",
                    incumbent.trial_id,
                )
                return None
        resolver = getattr(self, "knob_resolver", None)
        space = getattr(resolver, "_space", None) if resolver is not None else None
        if space is None:
            return None

        try:
            from traigent.knobs.bindings import Calibrated, is_governed
        except Exception:  # pragma: no cover - knobs is a hard dependency
            return None

        governed = [
            name
            for name, knob in dict(getattr(space, "knobs", {}) or {}).items()
            if isinstance(getattr(knob, "binding", None), Calibrated)
            and is_governed(knob.binding)
        ]
        if not governed:
            return None

        calibrated_inputs = getattr(resolver, "_calibrated_inputs", {}) or {}
        certificates: dict[str, Any] = {}
        for name in governed:
            certificate = getattr(calibrated_inputs.get(name), "certificate", None)
            if certificate is None:
                logger.debug(
                    "certified_selection withheld: governed cvar %s has no "
                    "certificate at finalize",
                    name,
                )
                return None
            certificates[name] = certificate

        from traigent.cloud.governance import build_certified_selection

        return cast(
            dict[str, Any] | None,
            build_certified_selection(incumbent.trial_id, certificates),
        )

    def _withhold_promotion(self, reason: str) -> bool:
        """Fail closed: record + log a strict-mode withheld promotion."""
        self._strict_withheld_promotions.append(reason)
        logger.warning(
            "Strict evidence mode: promotion withheld (%s); "
            "no winner-by-objective fallback",
            reason,
        )
        return False

    def _handle_promotion_decision(
        self,
        decision: Any,
        candidate_trial: TrialResult,
    ) -> bool:
        """Handle the promotion decision from PromotionGate."""
        if decision.decision == "promote":
            logger.info(
                "PromotionGate: promoting candidate config (reason: %s)",
                decision.reason,
            )
            return True
        if decision.decision == "reject":
            logger.debug(
                "PromotionGate: rejecting candidate config (reason: %s)",
                decision.reason,
            )
            return False
        # "no_decision": strict evidence modes fail CLOSED — insufficient
        # evidence never promotes and never falls back to simple comparison
        # (RFC 0001 P7). Non-strict keeps the legacy simple comparison.
        if self._is_strict_evidence_mode():
            return self._withhold_promotion(f"gate no_decision: {decision.reason}")
        logger.debug(
            "PromotionGate: no decision, using simple comparison (reason: %s)",
            decision.reason,
        )
        return self._simple_is_better(candidate_trial)

    def _evaluate_promotion(
        self,
        candidate_hash: str,
        candidate_trial: TrialResult,
    ) -> bool:
        """Evaluate whether candidate should be promoted over incumbent.

        Uses PromotionGate for statistical evaluation when available and
        sufficient samples exist. Falls back to simple score comparison.

        Returns:
            True if candidate should become the new incumbent.
        """
        # If no promotion gate or no incumbent, use simple logic
        if self._promotion_gate is None:
            if self._is_strict_evidence_mode():
                # RFC 0001 §3.6: a declared-governed CVAR demands
                # certificate-backed promotion evidence; with no promotion
                # gate there is no certification machinery, so the raw
                # comparison MUST NOT promote (it would otherwise be counted
                # as a certified promotion by _update_best_trial_cache and
                # launder an uncertified winner — fail open).
                return self._withhold_promotion(
                    "no promotion gate in strict evidence mode"
                )
            return self._simple_is_better(candidate_trial)
        if self._incumbent_config_hash is None:
            return True

        # Get metrics for both configs
        candidate_metrics = self._config_metrics_history.get(candidate_hash, {})
        incumbent_metrics = self._config_metrics_history.get(
            self._incumbent_config_hash, {}
        )

        # Check if we have sufficient samples for statistical comparison
        has_data = bool(candidate_metrics and incumbent_metrics)
        if not has_data or not self._has_sufficient_samples(
            candidate_metrics, incumbent_metrics
        ):
            if self._is_strict_evidence_mode():
                return self._withhold_promotion(
                    "insufficient samples for statistical comparison"
                )
            return self._simple_is_better(candidate_trial)

        # Use PromotionGate for statistical evaluation
        inc_metrics_seq = cast(dict[str, Sequence[float]], incumbent_metrics)
        cand_metrics_seq = cast(dict[str, Sequence[float]], candidate_metrics)
        try:
            decision = self._promotion_gate.evaluate(
                incumbent_metrics=inc_metrics_seq,
                candidate_metrics=cand_metrics_seq,
            )
            return self._handle_promotion_decision(decision, candidate_trial)
        except Exception as e:
            if self._is_strict_evidence_mode():
                # Gate exception fails CLOSED in strict modes (Rule 1: a
                # failing policy surface denies; it never silently degrades
                # to the permissive simple comparison).
                logger.error(
                    "PromotionGate evaluation failed in strict evidence "
                    "mode; promotion withheld: %s",
                    e,
                )
                return self._withhold_promotion(f"gate exception: {e}")
            logger.warning(
                "PromotionGate evaluation failed, using simple comparison: %s", e
            )
            return self._simple_is_better(candidate_trial)

    def _weighted_selection_schema(self) -> ObjectiveSchema | None:
        """Schema governing weighted incumbent ranking, or None for legacy.

        Mirrors the terminal-selection gate in ``result_selection`` (issue
        #1682): weighted ranking activates only for schemas with meaningful
        (non-uniform) weights over >1 non-banded objectives.
        """
        return resolve_weighted_selection_schema(self.objective_schema)

    def _weighted_is_better(
        self, schema: ObjectiveSchema, trial_result: TrialResult
    ) -> bool:
        """Compare candidate vs incumbent by the schema's weighted aggregate.

        Normalization uses the min-max ranges observed SO FAR (successful
        trials seen to date plus the incumbent and the candidate), matching
        terminal selection's observed-range normalization (issue #1682).
        Ranges evolve as trials arrive, so the incumbent comparison is
        recomputed under current ranges each time; live tracking is therefore
        an online approximation — the authoritative ``best_config`` comes
        from terminal selection over the full trial set.
        """
        observed: list[TrialResult] = [
            trial for trial in self._trials if trial.is_successful
        ]
        observed.append(trial_result)
        if self._best_trial_cached is not None:
            observed.append(self._best_trial_cached)
        ranges = observed_metric_ranges(
            observed, (obj.name for obj in schema.objectives)
        )

        new_weighted = schema.compute_weighted_score(
            trial_result.metrics or {}, ranges=ranges
        )
        if new_weighted is None or not math.isfinite(new_weighted):
            return False

        if self._best_trial_cached is None:
            return True

        current_weighted = schema.compute_weighted_score(
            self._best_trial_cached.metrics or {}, ranges=ranges
        )
        if current_weighted is None or not math.isfinite(current_weighted):
            return True

        if _primary_scores_tied(new_weighted, current_weighted):
            return self._secondary_tie_breaks_incumbent(
                trial_result, self.optimizer.objectives[0]
            )
        return bool(new_weighted > current_weighted)

    def _simple_is_better(self, trial_result: TrialResult) -> bool:
        """Check if trial_result is better than current best using simple comparison."""
        if not self.optimizer.objectives:
            return True

        weighted_schema = self._weighted_selection_schema()
        if weighted_schema is not None:
            # Honor declared ObjectiveSchema weights in live incumbent
            # tracking (issue #1682) via the shared objectives.py scorer.
            return self._weighted_is_better(weighted_schema, trial_result)

        primary_objective = self.optimizer.objectives[0]
        new_score_value = coerce_finite_objective_score(
            trial_result.get_metric(primary_objective)
            if hasattr(trial_result, "get_metric")
            else (trial_result.metrics or {}).get(primary_objective)
        )
        if new_score_value is None:
            return False

        if self._best_trial_cached is None:
            return True

        current_score_value = coerce_finite_objective_score(
            self._best_trial_cached.get_metric(primary_objective)
            if hasattr(self._best_trial_cached, "get_metric")
            else (self._best_trial_cached.metrics or {}).get(primary_objective)
        )
        if current_score_value is None:
            return True

        # Honor the declared orientation (issue #1960) so live incumbent
        # tracking uses the schema direction rather than a name heuristic.
        minimization = is_minimization_objective(
            primary_objective,
            orientation=(
                self.objective_schema.get_orientation(primary_objective)
                if self.objective_schema
                else None
            ),
        )
        if _primary_scores_tied(new_score_value, current_score_value):
            return self._secondary_tie_breaks_incumbent(trial_result, primary_objective)
        if minimization:
            return bool(new_score_value < current_score_value)
        return bool(new_score_value > current_score_value)

    def _secondary_tie_breaks_incumbent(
        self,
        trial_result: TrialResult,
        primary_objective: str,
    ) -> bool:
        """Return whether secondary declared objectives promote the candidate."""
        if len(self.optimizer.objectives) <= 1 or self._best_trial_cached is None:
            return False

        # Thread the declared orientation into the live secondary comparison
        # (issue #1955) so a minimize-oriented secondary (e.g. 'brier') promotes
        # the LOWER value, matching every terminal / post-hoc selection path.
        orientations = (
            {obj.name: str(obj.orientation) for obj in self.objective_schema.objectives}
            if self.objective_schema and self.objective_schema.objectives
            else None
        )

        candidate_key = _secondary_metric_key(
            trial_result.metrics or {},
            primary_objective,
            self.optimizer.objectives,
            orientations,
        )
        incumbent_key = _secondary_metric_key(
            self._best_trial_cached.metrics or {},
            primary_objective,
            self.optimizer.objectives,
            orientations,
        )
        return bool(candidate_key > incumbent_key)

    def _update_best_trial_cache(self, trial_result: TrialResult) -> None:
        if not self.optimizer.objectives:
            self._best_trial_cached = trial_result
            return

        primary_objective = self.optimizer.objectives[0]
        if (
            coerce_finite_objective_score(trial_result.get_metric(primary_objective))
            is None
        ):
            return

        # Track metrics for this trial
        config_hash = self._track_trial_metrics(trial_result)

        # First trial becomes incumbent
        if self._best_trial_cached is None:
            self._best_trial_cached = trial_result
            self._incumbent_config_hash = config_hash
            return

        # Evaluate if this trial should become the new best
        if self._evaluate_promotion(config_hash, trial_result):
            if self._is_strict_evidence_mode():
                # In strict mode a True here can only come from the gate's
                # explicit "promote": the no-gate lane, the no-decision /
                # insufficient-samples / exception lanes all withhold — so
                # counting it is sound: only gate-certified promotions
                # justify a certified winner.
                self._certified_promotions += 1
            self._best_trial_cached = trial_result
            self._incumbent_config_hash = config_hash

    def _remaining_sample_budget(self) -> float:
        """Return remaining sample budget.

        Note: reads ``_consumed_examples`` which is mutated under
        ``_state_lock``.  Callers in the optimisation loop already
        serialise via the loop structure, but any new call-site that
        runs concurrently **must** hold ``self._state_lock`` first.
        """
        if self._sample_budget_manager is not None:
            return float(self._sample_budget_manager.remaining())
        if self._max_total_examples is None:
            return float("inf")
        return max(self._max_total_examples - self._consumed_examples, 0)

    def _register_examples_attempted(self, trial_result: TrialResult) -> None:
        if self._sample_budget_manager is not None:
            self._consumed_examples = int(self._sample_budget_manager.consumed())
            return
        if (
            not self._samples_include_pruned
            and trial_result.status == TrialStatus.PRUNED
        ):
            return
        attempted = extract_examples_attempted(
            trial_result, default=None, check_example_results=True
        )
        if attempted is None:
            return
        self._consumed_examples += attempted

    def _extract_provider_failure_summary(
        self,
        trial_result: TrialResult,
    ) -> dict[str, Any] | None:
        """Extract fatal provider-call failure counts from trial metadata/error text."""

        metadata = trial_result.metadata or {}
        summary = metadata.get("provider_failure_summary")
        if isinstance(summary, dict):
            return summary

        if trial_result.status != TrialStatus.FAILED or not trial_result.error_message:
            return None

        category = classify_systematic_provider_failure(trial_result.error_message)
        if category is None:
            return None

        attempted = extract_examples_attempted(trial_result, default=1)
        if attempted <= 0:
            attempted = 1
        return {
            "attempted_calls": attempted,
            "fatal_failures": attempted,
            "failure_rate": 1.0,
            "category": category.value,
            "category_counts": {category.value: attempted},
            "sample_error": trial_result.error_message,
        }

    def _record_provider_failure_signal(self, trial_result: TrialResult) -> None:
        """Abort when provider/auth/quota failures dominate attempted calls."""

        summary = self._extract_provider_failure_summary(trial_result)
        if summary is None:
            self._provider_consecutive_call_failures = 0
            return

        attempted = max(int(summary.get("attempted_calls") or 0), 0)
        fatal_failures = max(int(summary.get("fatal_failures") or 0), 0)
        if attempted <= 0 or fatal_failures <= 0:
            return

        self._provider_call_attempts += attempted
        self._provider_call_failures += fatal_failures
        if fatal_failures >= attempted:
            self._provider_consecutive_call_failures += fatal_failures
        else:
            self._provider_consecutive_call_failures = 0

        sample_error = summary.get("sample_error")
        if self._provider_failure_first_error is None and sample_error:
            self._provider_failure_first_error = str(sample_error)
        category = summary.get("category")
        if self._provider_failure_category is None and category:
            self._provider_failure_category = str(category)

        failure_rate = (
            self._provider_call_failures / self._provider_call_attempts
            if self._provider_call_attempts
            else 0.0
        )
        consecutive_abort = self._provider_consecutive_call_failures >= 3
        high_fraction_abort = (
            self._provider_call_attempts >= 5
            and self._provider_call_failures >= 3
            and failure_rate >= 0.80
        )
        if not (consecutive_abort or high_fraction_abort):
            return

        hint = provider_failure_action_hint(self._provider_failure_category)
        first_error = self._provider_failure_first_error or "provider call failed"
        self._stop_reason = "vendor_error"
        self._status = OptimizationStatus.FAILED
        raise OptimizationError(
            "Aborting optimization because provider calls are failing "
            "systematically before they can be scored. "
            f"{self._provider_call_failures}/{self._provider_call_attempts} "
            f"attempted call(s) failed with {self._provider_failure_category or 'provider'} "
            f"errors. Likely cause: {hint} First provider error: {first_error}"
        )

    async def _cleanup_backend_client(self) -> None:
        """Close backend HTTP session if one was opened."""

        guidance_client = getattr(self, "_cloud_guidance_client", None)
        if guidance_client is not None:
            closer = getattr(guidance_client, "close", None)
            if callable(closer):
                try:
                    result = closer()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    logger.debug("Cloud guidance client cleanup failed: %s", exc)
            self._cloud_guidance_client = None

        client = self.backend_client
        if client is None:
            return

        closer = getattr(client, "close", None) or getattr(
            client, "_reset_http_session", None
        )
        if not closer:
            return

        try:
            try:
                result = closer()
            except TypeError:
                # Some closers require an argument
                result = closer("shutdown")

            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.debug("Backend client cleanup failed: %s", exc)

    async def _cleanup_hybrid_lifecycle(self) -> None:
        """Release hybrid API lifecycle manager if one was registered."""
        lifecycle_manager = getattr(self, "_hybrid_lifecycle_manager", None)
        if lifecycle_manager is None:
            return

        try:
            releaser = getattr(lifecycle_manager, "release", None)
            if releaser:
                result = releaser()
                if inspect.isawaitable(result):
                    await result
                logger.debug("Hybrid lifecycle manager released")
        except Exception as exc:
            logger.debug("Hybrid lifecycle cleanup failed: %s", exc)

    def _get_progress_info(self, current_trial: int) -> ProgressInfo:
        """Create progress information for callbacks."""
        elapsed_time = time.time() - self._start_time if self._start_time else 0.0

        # Count successful and failed trials
        successful_trials = self._successful_trials
        failed_trials = self._failed_trials

        # Get best score and config
        best_result = self.best_result
        best_score = None
        best_config = None
        if best_result and best_result.metrics and self.optimizer.objectives:
            primary_objective = self.optimizer.objectives[0]
            best_score = best_result.metrics.get(primary_objective)
            best_config = best_result.config

        # Estimate remaining time
        avg_trial_time = elapsed_time / len(self._trials) if self._trials else 0
        remaining_trials = (self.max_trials or 0) - len(self._trials)
        estimated_remaining = (
            avg_trial_time * remaining_trials if remaining_trials > 0 else None
        )

        return ProgressInfo(
            current_trial=current_trial,
            total_trials=self.max_trials or 0,
            completed_trials=len(self._trials),
            successful_trials=successful_trials,
            failed_trials=failed_trials,
            best_score=best_score,
            best_config=best_config,
            elapsed_time=elapsed_time,
            estimated_remaining=estimated_remaining,
            current_algorithm=self.optimizer.__class__.__name__,
        )

    # --- Consolidated logging helpers (V2 + legacy) ---
    def _initialize_logger(
        self,
        session_id: str | None,
        func: Callable[..., Any],
        dataset: Dataset,
        *,
        experiment_display_name: str | None = None,
    ) -> None:
        if not session_id:
            return

        experiment_name = experiment_display_name or (
            func.__name__ if hasattr(func, "__name__") else "unknown_function"
        )

        self._logger = OptimizationLogger(
            experiment_name=experiment_name,
            session_id=session_id,
            execution_mode=cast(
                ExecutionMode | str,
                self.traigent_config.execution_mode or "local",
            ),
        )
        self._logger_facade.attach(self._logger)

        self._logger_facade.log_session_start(
            config=self._build_logger_run_config(),
            objectives=(
                self.objective_schema
                if self.objective_schema
                else self.optimizer.objectives
            ),
            algorithm=self.optimizer.__class__.__name__,
            dataset_info={
                "size": len(dataset),
                "name": getattr(dataset, "name", "unknown"),
            },
        )

    def _build_logger_run_config(self) -> dict[str, Any]:
        resolved_algorithm = self.optimizer.__class__.__name__
        requested_algorithm = self.config.get("requested_algorithm")
        algorithm_config = getattr(self.optimizer, "algorithm_config", {}) or {}
        traigent_config = (
            self.traigent_config.to_dict()
            if hasattr(self.traigent_config, "to_dict")
            else {}
        )

        return {
            "schema_version": "2",
            "algorithm": requested_algorithm or resolved_algorithm,
            "requested_algorithm": requested_algorithm,
            "resolved_algorithm": resolved_algorithm,
            "max_trials": self.max_trials,
            "max_total_examples": self._max_total_examples,
            "configuration_space": getattr(self.optimizer, "config_space", {}) or {},
            "objectives": list(getattr(self.optimizer, "objectives", []) or []),
            "parallel_trials": self.parallel_trials,
            "timeout": self.timeout,
            "budget": {
                "cost_limit": self.config.get("cost_limit"),
                "metric_limit": self.config.get("metric_limit"),
                "metric_name": self.config.get("metric_name"),
                "max_total_examples": self._max_total_examples,
                "samples_include_pruned": self.samples_include_pruned,
            },
            "execution": {
                "mode": self.traigent_config.execution_mode,
                "result_source": getattr(self.traigent_config, "result_source", None),
                "no_egress": getattr(self.traigent_config, "no_egress", None),
                "privacy_enabled": self.traigent_config.privacy_enabled,
            },
            "optimizer_config": dict(algorithm_config),
            "traigent_config": traigent_config,
        }

    async def _handle_trial_result(
        self,
        trial_result: TrialResult,
        optimizer_config: dict[str, Any] | Any,
        current_trial_index: int,
        session_id: str | None,
        optuna_trial_id: int | None,
        *,
        log_on_success: bool,
        permit: Permit | None = None,
        submit_to_backend: bool = True,
    ) -> int:
        """Update orchestrator state after a single trial completes.

        Args:
            permit: The Permit object from permit acquisition.
                When provided, track_cost_async uses this to release the
                reservation exactly with single-release semantics, preventing
                over/under-release when EMA changes between acquisition and tracking.
        """

        # Protect shared state mutations with lock to prevent race conditions
        # during parallel trial execution (P1/P2/P3 fixes)
        async with self._state_lock:
            self._trials.append(trial_result)
            logger.info(
                "Trial #%d result: status=%s, config=%s, metrics=%s",
                current_trial_index + 1,
                trial_result.status,
                trial_result.config,
                trial_result.metrics,
            )
            self._log_trial(trial_result)

            # Track cost for cost limit enforcement
            # Create a default permit if none provided (for sequential trials without permit system)
            effective_permit = permit or Permit(id=0, amount=0.0, active=True)

            if trial_result.status == TrialStatus.CANCELLED:
                # For cancelled trials, check if they have cost data
                # (e.g., partial API calls before cancellation)
                trial_cost = self._extract_trial_cost(trial_result)
                if trial_cost is not None:
                    # Track actual cost incurred before cancellation
                    await self.cost_enforcer.track_cost_async(
                        cost=trial_cost,
                        permit=effective_permit,
                        trial_failed=True,
                        trial_id=trial_result.trial_id,
                    )
                # Note: For cost-limit cancellations, no permit was acquired so
                # nothing to release. For other cancellations (exceptions),
                # the parallel execution manager handles permit release.
            else:
                # For success/failed trials, track cost (which releases reservation)
                trial_cost = self._extract_trial_cost(trial_result)
                await self.cost_enforcer.track_cost_async(
                    cost=trial_cost,
                    permit=effective_permit,
                    trial_failed=trial_result.status == TrialStatus.FAILED,
                    trial_id=trial_result.trial_id,
                )

            if trial_result.is_successful:
                self._successful_trials += 1
                self.optimizer.update_best(trial_result)
                self._update_best_trial_cache(trial_result)
            elif trial_result.status == TrialStatus.FAILED:
                self._failed_trials += 1

            self._notify_optimizer_of_result(trial_result, optuna_trial_id)

            # Track consumed examples inside lock to prevent race conditions
            # on _consumed_examples during parallel trial execution
            self._register_examples_attempted(trial_result)
            self._record_provider_failure_signal(trial_result)

        submission_outcome: Any = None
        # #1939: no backend-client requirement — submit_trial persists the
        # trial LOCALLY even when there is no backend client (offline runs),
        # and only performs the remote submission when egress is enabled.
        if submit_to_backend and session_id:
            pre_submit_trial_id = (
                str(trial_result.trial_id)
                if getattr(trial_result, "trial_id", None) is not None
                else None
            )
            submission_outcome = await self.backend_session_manager.submit_trial(
                trial_result=trial_result,
                session_id=session_id,
                dataset_name=getattr(self, "_dataset_name", "dataset"),
            )
            post_submit_trial_id = (
                str(trial_result.trial_id)
                if getattr(trial_result, "trial_id", None) is not None
                else None
            )
            if post_submit_trial_id != pre_submit_trial_id:
                self._workflow_trace_manager.rebind_configuration_run_id(
                    pre_submit_trial_id,
                    post_submit_trial_id,
                )

        if hasattr(self.optimizer, "tell"):
            try:
                self.optimizer.tell(optimizer_config, trial_result)
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                logger.exception(
                    "Optimizer tell() failed for trial %s: %s", trial_result.trial_id, e
                )

        new_count = current_trial_index + 1

        progress_info = self._get_progress_info(new_count)
        self.callback_manager.on_trial_complete(trial_result, progress_info)

        should_log = False
        if new_count % PROGRESS_LOG_INTERVAL == 0:
            should_log = True
        elif log_on_success and trial_result.is_successful:
            should_log = True

        if should_log:
            self._log_progress(new_count)

        if getattr(submission_outcome, "optimization_complete", False) is True:
            reason = getattr(submission_outcome, "reason", None)
            raise CloudBrainOptimizationComplete(
                str(reason) if reason else "cloud brain completed optimization"
            )

        return new_count

    def _calculate_parallel_batch_caps(
        self,
        remaining: float,
        remaining_samples: float | None,
    ) -> tuple[int, int, bool]:
        """Calculate caps for parallel batch execution.

        Delegates to ParallelExecutionManager for computation.

        Returns:
            Tuple of (remaining_cap, target_batch_size, infinite_budget)
        """
        caps = self.parallel_execution_manager.calculate_batch_caps(
            remaining, remaining_samples
        )
        return caps.remaining_cap, caps.target_batch_size, caps.infinite_budget

    async def _try_hybrid_batch_generation(
        self, dataset: Dataset, remaining: int
    ) -> list[dict[str, Any]]:
        """Try to generate configs via hybrid/async batch generation."""
        try:
            remote_context = {
                "privacy_enabled": getattr(
                    self.traigent_config, "privacy_enabled", False
                ),
                "parallel_trials": remaining,
                "mode": "hybrid",
                "dataset_size": len(dataset),
            }
            return (
                await self.optimizer.generate_candidates_async(
                    remaining, remote_context=remote_context
                )
                or []
            )
        except (ValueError, TypeError, AttributeError, NotImplementedError) as e:
            logger.warning(
                "Async batch suggestion unavailable, falling back to sequential: %s", e
            )
            return []

    def _generate_sequential_configs(self, count: int) -> list[dict[str, Any]]:
        """Generate configs sequentially from the optimizer.

        Validates each config against pre-trial constraints
        (``config_space.validate``) before including it, to avoid
        wasting budget on configurations that violate structural
        constraints.
        """
        configs: list[dict[str, Any]] = []
        for _ in range(count):
            try:
                config = self.optimizer.suggest_next_trial(self._trials)
                config = self._apply_knob_resolution(config)
            except OptimizationError:
                break
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.exception(
                    "Optimizer failed to suggest trial during batch generation: %s", e
                )
                break
            # Pre-trial constraint validation — reject before execution
            if not pre_trial_validate_config(config, self._constraints_pre_eval):
                logger.debug("Config rejected by pre-trial constraints: %s", config)
                continue
            configs.append(config)
        return configs

    async def _generate_parallel_configs(
        self,
        dataset: Dataset,
        target_batch_size: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Generate batch of configs from optimizer.

        Returns:
            Tuple of (configs, used_async_batch)
        """
        configs: list[dict[str, Any]] = []
        remaining = target_batch_size

        baseline_config = self._consume_default_config()
        if baseline_config is not None:
            # baseline trials get the same Fixed/CVAR injection as suggested
            # ones — no bypass of R3/R6/R8 (RFC 0001)
            configs.append(self._apply_knob_resolution(baseline_config))
            remaining -= 1
            if remaining <= 0:
                return configs, False

        # Try hybrid/async generation if applicable
        used_async_batch = False
        if self.traigent_config.execution_mode_enum is ExecutionMode.HYBRID:
            async_configs = await self._try_hybrid_batch_generation(dataset, remaining)
            if async_configs:
                configs.extend(
                    self._apply_knob_resolution(async_config)
                    for async_config in async_configs
                )
                used_async_batch = True

        # Fall back to sequential generation
        if not used_async_batch:
            configs.extend(self._generate_sequential_configs(remaining))

        return configs, used_async_batch

    def _apply_knob_resolution(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve knob bindings for one suggestion (RFC 0001 §3.4).

        With no resolver configured this is an exact passthrough (legacy
        behavior byte-identical). With a resolver, Fixed and Calibrated
        values are injected fail-closed: any rejection (stale certificate,
        evidence leakage, missing producer, ...) raises ResolutionError —
        there is no silent fallback. Recorded fallback use is logged.
        """
        if self.knob_resolver is None:
            return config
        from traigent.knobs import ResolutionError, ResolutionRejection

        try:
            resolved = self.knob_resolver.resolve(config)
        except ResolutionError:
            raise
        except Exception as exc:
            # Internal resolver failures (ValueError, canonicalization, ...)
            # MUST surface as the typed fail-closed error — the suggest-site
            # except tuples catch ValueError and would silently break.
            raise ResolutionError(
                (ResolutionRejection.INFEASIBLE_VALUE,),
                f"resolver internal failure: {exc}",
            ) from exc
        if resolved.used_fallbacks:
            logger.warning(
                "Knob resolution used declared fallbacks for: %s",
                ", ".join(resolved.used_fallbacks),
            )
        return dict(resolved.config)

    async def _suggest_next_trial_config(self, dataset: Dataset) -> dict[str, Any]:
        """Get the next trial config, using async cloud guidance when available."""

        if self._optimizer_uses_remote_guidance() and (
            backend_egress_disabled(self.traigent_config)
            or getattr(self.traigent_config, "result_source", None)
            == SOURCE_LOCAL_FALLBACK
        ):
            reason = (
                getattr(self.traigent_config, "fallback_reason", None)
                or self.backend_session_manager.fallback_reason
                or "backend egress disabled"
            )
            raise CloudBrainUnavailableError("next-trial", str(reason))

        remote_context = {
            "privacy_enabled": getattr(self.traigent_config, "privacy_enabled", False),
            "mode": (
                SOURCE_CLOUD_BRAIN
                if self._is_cloud_brain_run()
                else self.traigent_config.execution_mode
            ),
            "dataset_size": len(dataset),
            "session_id": self._active_session_id,
            "completed_trials": len(self._trials),
        }
        config = await self.optimizer.suggest_next_trial_async(
            self._trials,
            remote_context=remote_context,
        )
        return self._apply_knob_resolution(config)

    def _build_trial_descriptors(
        self,
        configs: list[dict[str, Any]],
        dataset: Dataset,
    ) -> list[tuple[dict[str, Any], dict[str, Any] | Any, Dataset, int | None]]:
        """Build trial descriptors with subset handling.

        Returns:
            List of (original_config, eval_config, dataset_for_trial, optuna_id)
        """
        trial_descriptors: list[
            tuple[dict[str, Any], dict[str, Any] | Any, Dataset, int | None]
        ] = []

        for idx, cfg in enumerate(configs):
            cfg_copy = dict(cfg) if isinstance(cfg, dict) else cfg
            optuna_id = (
                cfg_copy.get("_optuna_trial_id") if isinstance(cfg_copy, dict) else None
            )

            cfg_eval = prepare_evaluation_config(cfg_copy)

            ds_for_trial = dataset
            if isinstance(cfg_copy, dict) and "__subset_indices__" in cfg_copy:
                try:
                    idxs = list(cfg_copy.get("__subset_indices__") or [])
                    examples = [
                        dataset.examples[j]
                        for j in idxs
                        if 0 <= j < len(dataset.examples)
                    ]
                    ds_for_trial = Dataset(
                        examples=examples,
                        name=f"{getattr(dataset, 'name', 'dataset')}_subset_{idx}",
                        description=getattr(
                            dataset,
                            "description",
                            "Traigent evaluation dataset",
                        ),
                    )
                except Exception as exc:
                    logger.debug(
                        "Subset indices provided but failed to build subset: %s",
                        exc,
                    )

            trial_descriptors.append((cfg, cfg_eval, ds_for_trial, optuna_id))

        return trial_descriptors

    def _allocate_trial_ceilings(
        self,
        trial_descriptors: list[
            tuple[dict[str, Any], dict[str, Any] | Any, Dataset, int | None]
        ],
    ) -> list[int | None]:
        """Allocate sample ceilings for parallel trials.

        This is a safety mechanism to ensure that when spinning up N parallel
        trials with X remaining budget, each trial gets at most X/N samples.
        This prevents any single trial from consuming more than its fair share
        when running in parallel.

        Args:
            trial_descriptors: List of trial descriptor tuples

        Returns:
            List of sample ceilings for each trial
        """
        if self._sample_budget_manager is None:
            return [len(desc[2].examples) for desc in trial_descriptors]

        remaining_float = self._sample_budget_manager.remaining()
        if math.isfinite(remaining_float):
            dataset_sizes = [len(desc[2].examples) for desc in trial_descriptors]
            return list(allocate_parallel_ceilings(dataset_sizes, int(remaining_float)))
        return cast(
            list[int | None], [len(desc[2].examples) for desc in trial_descriptors]
        )

    async def _schedule_and_run_parallel_trials(
        self,
        func: Callable[..., Any],
        trial_descriptors: list[
            tuple[dict[str, Any], dict[str, Any] | Any, Dataset, int | None]
        ],
        ceilings: list[int | None],
        session_id: str | None,
        trial_count: int,
    ) -> tuple[list[dict[str, Any]], list[int | None], list[PermittedTrialResult]]:
        """Schedule and execute parallel trials.

        Returns:
            Tuple of (scheduled_configs, scheduled_optuna_ids, results)
        """
        scheduled_configs: list[dict[str, Any]] = []
        scheduled_optuna_ids: list[int | None] = []
        tasks = []

        for descriptor, sample_ceiling in zip(
            trial_descriptors, ceilings, strict=False
        ):
            cfg, cfg_eval, ds_for_trial, optuna_id = descriptor
            if sample_ceiling is not None and sample_ceiling <= 0:
                logger.debug(
                    "Skipping parallel trial due to zero sample ceiling (config=%s)",
                    cfg,
                )
                self._trials_prevented += 1
                self._abandon_optuna_trial(
                    optuna_id,
                    reason="trial_skipped_zero_sample_ceiling",
                    config=cfg,
                    status=TrialStatus.CANCELLED,
                    pruned_step=0,
                )
                continue

            trial_index = trial_count + len(scheduled_configs)
            self.callback_manager.on_trial_start(trial_index, cfg)
            scheduled_configs.append(cfg)
            scheduled_optuna_ids.append(optuna_id)
            tasks.append(
                self._trial_lifecycle.run_trial(
                    func,
                    cfg,
                    ds_for_trial,
                    trial_index,
                    session_id,
                    optuna_trial_id=optuna_id,
                    sample_ceiling=sample_ceiling,
                )
            )

        if not tasks:
            return [], [], []

        # Use permit-based execution with cost enforcement
        # Each trial checks for a cost permit before execution
        # Trials that exceed the cost limit are cancelled (returned as None)
        (
            results,
            cancelled_count,
        ) = await self.parallel_execution_manager.run_with_cost_permits(
            tasks,
            cancel_sentinel=None,  # None indicates cost limit cancellation
        )

        # Track cancelled trials for reporting
        if cancelled_count > 0:
            self._trials_prevented += cancelled_count
            logger.info(
                "Prevented %d trial(s) due to cost limit during parallel execution",
                cancelled_count,
            )

        return scheduled_configs, scheduled_optuna_ids, results

    async def _process_parallel_results(
        self,
        scheduled_configs: list[dict[str, Any]],
        results: list[PermittedTrialResult],
        scheduled_optuna_ids: list[int | None],
        session_id: str | None,
        trial_count: int,
    ) -> int:
        """Process results from parallel trial execution.

        Results are wrapped in PermittedTrialResult to carry the Permit object
        through to track_cost_async for exact budget release with single-release
        semantics.

        Results may include:
        - TrialResult objects for successful trials
        - Exceptions (due to return_exceptions=True in gather)
        - None for trials cancelled due to cost limit
        """
        terminal_completion: CloudBrainOptimizationComplete | None = None
        submit_to_backend = True

        for batch_offset, (config, permitted_result, optuna_id) in enumerate(
            zip(scheduled_configs, results, scheduled_optuna_ids, strict=False)
        ):
            # Unwrap PermittedTrialResult to get result and permit
            result = permitted_result.result
            permit = permitted_result.permit

            # Use try/finally to ensure permit is released on any exception
            # during result processing (Gemini Phase 1.5 fix)
            try:
                # Handle cost limit cancellation (None sentinel from permit check)
                if result is None:
                    logger.info(
                        "Trial cancelled due to cost limit (config=%s)",
                        config,
                    )
                    trial_result = TrialResult(
                        trial_id=f"trial_{trial_count + batch_offset}",
                        config=config,
                        metrics={},
                        status=TrialStatus.CANCELLED,
                        duration=0.0,
                        timestamp=datetime.now(UTC),
                        error_message="Trial cancelled: cost limit reached",
                    )
                    # For cancelled trials, permit is denied (no permit acquired)
                # Handle exceptions from gather (P4 fix)
                elif isinstance(result, BaseException):
                    logger.warning(
                        "Trial failed with exception during parallel execution: %s",
                        result,
                        exc_info=result,
                    )
                    # Convert exception to failed TrialResult
                    # Use batch_offset (from enumerate) to ensure unique trial_id even before
                    # _handle_trial_result increments trial_count
                    trial_result = TrialResult(
                        trial_id=f"trial_{trial_count + batch_offset}",
                        config=config,
                        metrics={},
                        status=TrialStatus.FAILED,
                        duration=0.0,
                        timestamp=datetime.now(UTC),
                        error_message=str(result),
                    )
                    # For exceptions, permit was already released in parallel_execution_manager
                else:
                    trial_result = result

                trial_count = await self._handle_trial_result(
                    trial_result=trial_result,
                    optimizer_config=config,
                    current_trial_index=trial_count,
                    session_id=session_id,
                    optuna_trial_id=optuna_id,
                    log_on_success=False,
                    permit=permit,
                    submit_to_backend=submit_to_backend,
                )
            except CloudBrainOptimizationComplete as complete:
                if terminal_completion is None:
                    terminal_completion = complete
                submit_to_backend = False
                trial_count = len(self._trials)
            except Exception:
                # Release permit on any exception to prevent stranding
                # (only if permit is still active - wasn't already released)
                if permit.active and self.cost_enforcer is not None:
                    await self.cost_enforcer.release_permit_async(permit)
                    logger.debug(
                        "Released permit %d after exception in _process_parallel_results",
                        permit.id,
                    )
                raise
        if terminal_completion is not None:
            raise terminal_completion
        return trial_count

    def _apply_cap_and_prevent_excess(
        self,
        configs: list[dict[str, Any]],
        slice_cap: int,
    ) -> list[dict[str, Any]]:
        """Apply cap to configs and mark prevented trials as pruned.

        Args:
            configs: List of trial configurations
            slice_cap: Maximum number of configs to keep

        Returns:
            Capped list of configurations
        """
        original_count = len(configs)
        prevented_configs = configs[slice_cap:]
        capped_configs = configs[:slice_cap]

        if original_count > len(capped_configs):
            prevented_count = original_count - len(capped_configs)
            self._trials_prevented += prevented_count
            logger.info("Prevented %s trials due to cap", prevented_count)
            for cfg in prevented_configs:
                optuna_id = (
                    cfg.get("_optuna_trial_id") if isinstance(cfg, dict) else None
                )
                self._abandon_optuna_trial(
                    optuna_id,
                    reason="trial_prevented_by_cap",
                    config=cfg,
                    status=TrialStatus.PRUNED,
                    pruned_step=0,
                )
        return capped_configs

    def _apply_cache_policy_and_prune(
        self,
        configs: list[dict[str, Any]],
        function_name: str | None,
        dataset: Dataset,
    ) -> list[dict[str, Any]]:
        """Apply cache policy and mark filtered trials as pruned.

        Args:
            configs: List of trial configurations
            function_name: Name of the function being optimized
            dataset: The dataset being used

        Returns:
            Filtered list of configurations
        """
        cache_policy = self.config.get("cache_policy", "allow_repeats")
        if cache_policy == "allow_repeats":
            return configs

        dataset_name = getattr(dataset, "name", "unnamed_dataset")
        # Build a map from optuna_trial_id to config before filtering
        id_to_config: dict[int, dict[str, Any]] = {
            cast(int, cfg.get("_optuna_trial_id")): cfg
            for cfg in configs
            if isinstance(cfg, dict) and isinstance(cfg.get("_optuna_trial_id"), int)
        }
        before_ids = set(id_to_config.keys())

        filtered_configs: list[dict[str, Any]] = self.cache_policy_handler.apply_policy(
            configs, cache_policy, function_name or "unknown_function", dataset_name
        )

        after_ids = {
            cfg.get("_optuna_trial_id")
            for cfg in filtered_configs
            if isinstance(cfg, dict) and cfg.get("_optuna_trial_id") is not None
        }
        dropped_ids = before_ids - after_ids
        if dropped_ids:
            self._trials_prevented += len(dropped_ids)
            for optuna_id in dropped_ids:
                if isinstance(optuna_id, int):
                    self._abandon_optuna_trial(
                        optuna_id,
                        reason=f"trial_filtered_by_cache_policy:{cache_policy}",
                        config=id_to_config.get(optuna_id),
                        status=TrialStatus.PRUNED,
                        pruned_step=0,
                    )
        return filtered_configs

    def _maybe_log_checkpoint(self, trial_count: int) -> None:
        """Log a checkpoint if the trial count is at a logging interval."""
        if trial_count % PROGRESS_LOG_INTERVAL == 0:
            optimizer_state = {}
            if hasattr(self.optimizer, "get_state"):
                optimizer_state = self.optimizer.get_state()
            self._log_checkpoint(
                optimizer_state=optimizer_state,
                trials_history=self._trials,
                trial_count=trial_count,
            )

    async def _run_parallel_batch(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        session_id: str | None,
        function_name: str | None,
        trial_count: int,
        remaining: float,
        remaining_samples: float | None = None,
    ) -> tuple[int, str]:
        """Execute a batch of trials when parallel execution is enabled.

        This method orchestrates parallel trial execution by delegating to
        specialized helper methods for each phase.
        """
        # Phase 1: Calculate caps and check if we should continue
        remaining_cap, target_batch_size, infinite_budget = (
            self._calculate_parallel_batch_caps(remaining, remaining_samples)
        )

        if remaining_cap == 0:
            return trial_count, "break"

        # Phase 2: Generate configs from optimizer
        configs, _ = await self._generate_parallel_configs(dataset, target_batch_size)

        # Phase 3: Apply caps and prevent excess trials
        slice_cap = self.parallel_trials if infinite_budget else remaining_cap
        configs = self._apply_cap_and_prevent_excess(configs, slice_cap)

        # Phase 4: Apply cache policy
        configs = self._apply_cache_policy_and_prune(configs, function_name, dataset)

        if not configs:
            return trial_count, "break"

        # Phase 5: Build trial descriptors
        trial_descriptors = self._build_trial_descriptors(configs, dataset)

        if not trial_descriptors:
            return trial_count, "break"

        # Phase 6: Allocate sample ceilings
        ceilings = self._allocate_trial_ceilings(trial_descriptors)

        # Phase 7: Schedule and run trials
        (
            scheduled_configs,
            scheduled_optuna_ids,
            results,
        ) = await self._schedule_and_run_parallel_trials(
            func, trial_descriptors, ceilings, session_id, trial_count
        )

        if not results:
            return trial_count, "break"

        # Phase 8: Process results
        trial_count = await self._process_parallel_results(
            scheduled_configs, results, scheduled_optuna_ids, session_id, trial_count
        )

        # Phase 9: Check for batch-wide vendor errors (parallel mode)
        vendor_stop = await self._check_batch_vendor_errors(results)
        if vendor_stop:
            self._stop_reason = "vendor_error"
            return trial_count, "break"

        # Phase 10: Checkpoint logging
        self._maybe_log_checkpoint(trial_count)

        return trial_count, "continue"

    async def _submit_usage_analytics(self) -> None:
        """Submit usage analytics if enabled."""

        if not self.traigent_config.enable_usage_analytics:
            return

        if is_backend_offline() or backend_egress_disabled(self.traigent_config):
            logger.debug("Skipping analytics submission: backend egress is disabled")
            return

        try:
            from traigent.utils.local_analytics import LocalAnalytics

            analytics = LocalAnalytics(self.traigent_config)
            await asyncio.wait_for(analytics.submit_usage_stats(), timeout=10.0)
            logger.debug("Analytics submitted after optimization completion")
        except TimeoutError:
            logger.debug("Analytics submission timed out")
        except Exception as exc:
            logger.debug("Analytics submission failed: %s", exc)

    def collect_workflow_span(self, span_data: SpanPayload) -> None:
        """Collect a workflow span for later submission. Delegates to WorkflowTraceManager."""
        self._workflow_trace_manager.collect_span(span_data)

    async def _submit_workflow_traces(self, session_id: str | None = None) -> None:
        """Submit collected workflow traces. Delegates to WorkflowTraceManager."""
        if backend_egress_disabled(self.traigent_config):
            return
        await self._workflow_trace_manager.submit_traces(session_id)

    @staticmethod
    def _populate_experiment_cloud_url(result: OptimizationResult) -> None:
        """Populate backend experiment/run ids and portal URL from session metadata."""
        metadata = result.metadata or {}
        exp_id = metadata.get("experiment_id")
        if not exp_id:
            return
        run_id = metadata.get("experiment_run_id")
        result.experiment_id = str(exp_id)
        if run_id:
            result.experiment_run_id = str(run_id)
        try:
            from traigent.cloud.sync_manager import build_experiment_url
            from traigent.config.backend_config import BackendConfig

            result.cloud_url = build_experiment_url(
                BackendConfig.get_cloud_web_url(),
                str(exp_id),
                run_id=str(run_id) if run_id else None,
                project_id=metadata.get("project_id"),
                tenant_id=metadata.get("tenant_id"),
            )
        except Exception as exc:
            logger.debug("Cloud URL construction failed: %s", exc)

    def _initialize_optimization_run(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        function_name: str | None,
    ) -> str | None:
        """Perform common setup before entering the optimization loop."""

        logger.info(
            "Starting optimization %s: %s examples, max_trials=%s, max_total_examples=%s",
            self._optimization_id,
            len(dataset),
            self.max_trials,
            self.max_total_examples,
        )

        self._status = OptimizationStatus.RUNNING
        self._session_finalized = False
        self._start_time = time.time()
        self._successful_trials = 0
        self._failed_trials = 0
        self._best_trial_cached = None
        self._strict_withheld_promotions = []
        self._certified_promotions = 0

        # Reset stop conditions for fresh optimization run
        self._stop_condition_manager.reset()

        # Reset cost enforcer to prevent cost accumulation across multiple optimize() calls
        # on the same orchestrator instance (defensive programming)
        if self.cost_enforcer is not None:
            self.cost_enforcer.reset()

        descriptor = resolve_function_descriptor(func)
        self._function_descriptor = descriptor
        self._workflow_trace_manager._function_descriptor = descriptor

        self._dataset_name = getattr(dataset, "name", "dataset")

        # function_name may carry the user-supplied experiment_name override (from
        # @traigent.optimize(experiment_name=...)). The fully-qualified descriptor
        # identifier is still used for session keying; the override becomes the
        # human-readable display name forwarded to the portal.
        experiment_display_name: str | None = None
        if function_name and function_name != descriptor.identifier:
            logger.debug(
                "function_name='%s' supplied — using as experiment display name "
                "(session keyed by descriptor identifier '%s')",
                function_name,
                descriptor.identifier,
            )
            experiment_display_name = function_name

        # Create backend session using manager. Governance crosses the wire
        # content-free (Phase 8): the declared promotion policy and the
        # cvar summary from DECLARED bindings — never values or evidence.
        wire_policy, wire_governance = self._build_wire_governance()
        objectives_payload = self._build_session_objectives_payload()
        default_config_payload = self._build_session_default_config_payload()
        optimization_strategy_payload = self._backend_optimization_strategy_for_run()
        session_context = self.backend_session_manager.create_session(
            func=func,
            dataset=dataset,
            function_descriptor=descriptor,
            # #1938: connected local runs advertise the PLANNED local trial
            # count (never an over-/under-stated budget); other runs pass the
            # user's max_trials through unchanged.
            max_trials=self._planned_backend_max_trials(),
            max_total_examples=self.max_total_examples,
            start_time=self._start_time or time.time(),
            agent_configuration=self._agent_configuration,
            objectives=objectives_payload,
            default_config=default_config_payload,
            promotion_policy=wire_policy,
            tvl_governance=wire_governance,
            experiment_display_name=experiment_display_name,
            warm_start_from=self._warm_start_from,
            smart_pruning=self._smart_pruning,
            artifact_fingerprints=self.artifact_fingerprints,
            fingerprint_meta=self.fingerprint_meta,
            evaluator_definition_id=self.evaluator_definition_id,
            cost_limit=self.config.get("cost_limit"),
            optimization_strategy=optimization_strategy_payload,
        )
        session_id: str | None = session_context.session_id
        self._active_session_id = session_id
        self._bind_interactive_optimizer_session(
            session_id=session_id,
            function_name=experiment_display_name or descriptor.identifier,
            dataset=dataset,
        )

        if session_id:
            self._initialize_logger(
                session_id,
                func,
                dataset,
                experiment_display_name=experiment_display_name,
            )

        self.callback_manager.on_optimization_start(
            config_space=self.optimizer.config_space,
            objectives=self.optimizer.objectives,
            algorithm=self.optimizer.__class__.__name__,
        )

        return session_id

    def _bind_interactive_optimizer_session(
        self,
        *,
        session_id: str | None,
        function_name: str,
        dataset: Dataset,
    ) -> None:
        """Bind a backend-created session to InteractiveOptimizer."""

        if (
            self._optimizer_uses_remote_guidance()
            and getattr(self.traigent_config, "result_source", None)
            == SOURCE_LOCAL_FALLBACK
        ):
            reason = (
                getattr(self.traigent_config, "fallback_reason", None)
                or self.backend_session_manager.fallback_reason
                or "backend session creation failed"
            )
            raise CloudBrainUnavailableError("session-create", str(reason))

        if not self._is_cloud_brain_run():
            return

        if not session_id or not self.backend_session_manager.backend_tracking_enabled:
            raise CloudBrainUnavailableError(
                "session-create",
                "backend session was not created",
            )

        try:
            from traigent.cloud.models import (
                OptimizationSession,
                OptimizationSessionStatus,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - optional install guard
            raise CloudBrainUnavailableError(
                "session-create",
                "cloud models are unavailable",
                original=exc,
            ) from exc

        optimizer = cast(Any, self.optimizer)
        optimizer.session = OptimizationSession(
            session_id=session_id,
            function_name=function_name,
            configuration_space=getattr(self.optimizer, "config_space", {}) or {},
            objectives=list(getattr(self.optimizer, "objectives", []) or []),
            max_trials=self.max_trials if self.max_trials is not None else 10,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            metadata={
                "dataset_size": len(dataset),
                "source": SOURCE_CLOUD_BRAIN,
            },
        )
        optimizer.session_id = session_id
        optimizer._start_time = time.time()

    @property
    def max_trials(self) -> int | None:
        return self._max_trials

    @max_trials.setter
    def max_trials(self, value: int | None) -> None:
        self._max_trials = value
        if getattr(self, "_initialized", False):
            self._stop_condition_manager.update_max_trials(value)

    @property
    def max_total_examples(self) -> int | None:
        return self._max_total_examples

    @max_total_examples.setter
    def max_total_examples(self, value: int | None) -> None:
        self._max_total_examples = value
        if getattr(self, "_initialized", False):
            self._stop_condition_manager.update_max_samples(value)

    @property
    def samples_include_pruned(self) -> bool:
        return self._samples_include_pruned

    @samples_include_pruned.setter
    def samples_include_pruned(self, value: bool) -> None:
        self._samples_include_pruned = bool(value)
        if getattr(self, "_initialized", False):
            self._stop_condition_manager.update_samples_include_pruned(
                self._samples_include_pruned
            )

    @staticmethod
    def _validate_dataset(dataset: Dataset) -> None:
        """Ensure dataset is present and non-empty. Delegates to orchestrator_helpers."""
        validate_dataset(dataset)

    async def create_session(  # NOSONAR - async for API consistency with other orchestrator methods
        self, function_name: str | None = None, dataset_name: str | None = None
    ) -> str:
        """Create a session for optimization tracking.

        Args:
            function_name: Name of the function being optimized
            dataset_name: Name of the dataset being used

        Returns:
            Session ID string
        """
        descriptor = self._function_descriptor
        identifier = function_name or (
            descriptor.identifier if descriptor is not None else "unknown_function"
        )

        if backend_egress_disabled(self.traigent_config):
            mock_session_id = f"mock-session-{self._optimization_id[:8]}"
            logger.debug("Created local-only mock session: %s", mock_session_id)
            return mock_session_id

        if self.backend_client:
            # Ensure max_trials is not None
            max_trials_value = self.max_trials if self.max_trials is not None else 10
            logger.info(
                f"Creating session with max_trials={max_trials_value} (self.max_trials={self.max_trials})"
            )

            metadata = {
                "optimization_id": self._optimization_id,
                "max_trials": max_trials_value,
                "function_name": identifier,
                "function_display_name": (
                    descriptor.display_name if descriptor else identifier
                ),
                "evaluation_set": dataset_name or "default_evaluation",
            }
            if self.strategy_preset is not None:
                metadata["strategy_preset"] = self.strategy_preset.to_metadata()

            raw_result = self.backend_client.create_session(
                function_name=identifier,
                search_space=getattr(self.optimizer, "config_space", {}),
                optimization_goal="maximize",  # Default assumption
                metadata=metadata,
                smart_pruning=self._smart_pruning,
                artifact_fingerprints=self.artifact_fingerprints,
                fingerprint_meta=self.fingerprint_meta,
                evaluator_definition_id=self.evaluator_definition_id,
            )
            session_id = self.backend_session_manager.handle_session_creation_result(
                self.backend_session_manager.normalize_session_creation_result(
                    raw_result
                )
            )
            return cast(str, session_id)
        else:
            # Return a mock session ID if no backend client
            mock_session_id = f"mock-session-{self._optimization_id[:8]}"
            logger.debug(f"Created mock session: {mock_session_id}")
            return mock_session_id

    async def optimize(  # noqa: C901
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        function_name: str | None = None,
    ) -> OptimizationResult:
        """Run the optimization process.

        Args:
            func: Function to optimize
            dataset: Evaluation dataset
            function_name: Name of the function being optimized (for session tracking)

        Returns:
            OptimizationResult with all trial results

        Raises:
            OptimizationError: If optimization fails
        """
        # Validate dataset
        self._validate_dataset(dataset)

        # Perform standard initialization (logging, session creation, callbacks)
        session_id = self._initialize_optimization_run(func, dataset, function_name)

        function_identifier = (
            self._function_descriptor.identifier
            if self._function_descriptor is not None
            else function_name
        )

        # Start tracing span for the optimization session
        with optimization_session_span(
            function_name=function_identifier or func.__name__,
            max_trials=self.max_trials,
            timeout=self.timeout,
            algorithm=getattr(self.optimizer, "name", None),
            objectives=self.objectives,
            config_space=getattr(self.optimizer, "config_space", None),
        ) as session_span:
            return await self._run_optimization_with_tracing(
                func=func,
                dataset=dataset,
                session_id=session_id,
                function_identifier=function_identifier,
                session_span=session_span,
            )

    def _check_cost_approval(self, dataset: Dataset) -> None:
        """Check pre-run cost approval before optimization.

        Delegates to CostEstimator, which raises CostLimitExceeded (an
        OptimizationError subclass) when cost approval is declined. Mid-run
        cost limits are handled separately as graceful ``stop_reason="cost_limit"``
        results.
        """
        self._cost_estimator.check_cost_approval(dataset)

    def _is_cost_limit_reached(self) -> bool:
        """Return whether the shared cost enforcer has hit its run limit."""
        cost_enforcer = getattr(self, "cost_enforcer", None)
        return bool(cost_enforcer is not None and cost_enforcer.is_limit_reached)

    def _describe_budget_gate_block(self, remaining_trials: float) -> str:
        """Build the explicit budget-gate block message (issue #1684 item 3).

        Format: ``budget gate blocked N planned trials (estimated $X > limit $Y)``.
        """
        planned = (
            None if math.isinf(remaining_trials) else max(int(remaining_trials), 0)
        )
        cost_enforcer = getattr(self, "cost_enforcer", None)
        describe = getattr(cost_enforcer, "budget_block_message", None)
        if callable(describe):
            return str(describe(planned))
        blocked = "all remaining" if planned is None else str(planned)
        return f"budget gate blocked {blocked} planned trials (cost limit reached)"

    def _check_budget_limits(
        self, trial_count: int
    ) -> tuple[float, float, StopReason | None]:
        """Check trial and sample budget limits.

        Returns:
            Tuple of (remaining_trials, remaining_samples, stop_reason_if_any)
        """
        remaining = (
            float("inf") if self.max_trials is None else self.max_trials - trial_count
        )

        if self._is_cost_limit_reached():
            # Loud, explicit block: log the exact scope of what the budget
            # gate is cutting off (issue #1684 item 3). The WARNING log is the
            # user-visible surface today; ``_budget_gate_detail`` stages the
            # same string for the run-status/metadata surfacing follow-up
            # (owned by the status-logic unit) and is not read anywhere yet.
            detail = self._describe_budget_gate_block(remaining)
            logger.warning("%s (completed trials so far: %d)", detail, trial_count)
            self._budget_gate_detail = detail
            return remaining, 0, "cost_limit"

        if remaining <= 0:
            logger.info(f"Trial limit reached: {self.max_trials}")
            return remaining, 0, "max_trials_reached"

        remaining_samples = self._remaining_sample_budget()
        if remaining_samples <= 0:
            logger.info(
                "Sample limit reached: max_total_examples=%s", self._max_total_examples
            )
            return remaining, remaining_samples, "max_samples_reached"

        return remaining, remaining_samples, None

    def _apply_budget_stop(self, budget_stop: StopReason | None) -> bool:
        """Record a budget stop reason and return True if the loop should break."""
        if not budget_stop:
            return False
        if budget_stop == "cost_limit" or not self._stop_reason:
            self._stop_reason = budget_stop
        return True

    async def _run_optimization_loop(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        session_id: str | None,
        function_identifier: str | None,
    ) -> int:
        """Run the main optimization loop. Returns final trial count."""
        trial_count = 0

        while True:
            remaining, remaining_samples, budget_stop = self._check_budget_limits(
                trial_count
            )
            if self._apply_budget_stop(budget_stop):
                break

            stop_action = await self._check_stop_with_budget_pause(trial_count)
            if stop_action == "break":
                break
            if stop_action == "continue":
                continue

            trial_count, action = await self._dispatch_trial(
                func,
                dataset,
                session_id,
                function_identifier,
                trial_count,
                remaining,
                remaining_samples,
            )

            if action == "break":
                break

        return trial_count

    async def _check_stop_with_budget_pause(self, trial_count: int) -> str | None:
        """Check stop conditions; offer budget pause on cost limit.

        Returns:
            ``"break"`` to stop, ``"continue"`` to retry after raising limit,
            ``None`` to proceed normally.
        """
        if not self._should_stop(trial_count):
            return None
        if self._stop_reason == "cost_limit" and self._prompt_adapter is not None:
            decision = await self._handle_budget_limit_pause()
            if decision == "continue":
                self._stop_reason = None
                return "continue"
        return "break"

    async def _dispatch_trial(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        session_id: str | None,
        function_identifier: str | None,
        trial_count: int,
        remaining: float,
        remaining_samples: float | None,
    ) -> tuple[int, str]:
        """Run a single trial iteration (sequential or parallel) with vendor pause.

        Returns:
            ``(trial_count, action)`` where action is ``"continue"`` or ``"break"``.
        """
        try:
            if self.parallel_trials > 1:
                return await self._run_parallel_batch(
                    func=func,
                    dataset=dataset,
                    session_id=session_id,
                    function_name=function_identifier,
                    trial_count=trial_count,
                    remaining=remaining,
                    remaining_samples=remaining_samples,
                )
            return cast(
                tuple[int, str],
                await self._trial_lifecycle.run_sequential_trial(
                    func=func,
                    dataset=dataset,
                    session_id=session_id,
                    function_name=function_identifier,
                    trial_count=trial_count,
                ),
            )
        except VendorPauseError as e:
            decision = await self._handle_vendor_pause(e)
            if decision == "break":
                self._stop_reason = "vendor_error"
                return trial_count, "break"
            return trial_count, "continue"  # User chose to resume
        except CloudBrainOptimizationComplete as complete:
            self._stop_reason = cast(StopReason, complete.stop_reason)
            logger.info("Cloud brain completed optimization: %s", complete.reason)
            return len(self._trials), "break"

    @staticmethod
    def _vendor_retry_settings() -> tuple[int, float]:
        """Resolve bounded vendor-retry budget and base backoff from env (#1404).

        Returns ``(max_retries, base_backoff_seconds)``. Defaults to ``0``
        (opt-in) so the documented graceful fail-stop contract is unchanged
        unless the operator sets ``TRAIGENT_VENDOR_MAX_RETRIES``. A malformed
        value also yields the default.
        """

        def _read(name: str, default: float, *, minimum: float) -> float:
            raw = os.environ.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return default
            return value if value >= minimum else default

        max_retries = int(_read("TRAIGENT_VENDOR_MAX_RETRIES", 0.0, minimum=0.0))
        base_backoff = _read("TRAIGENT_VENDOR_RETRY_BACKOFF", 1.0, minimum=0.0)
        return max_retries, base_backoff

    async def _maybe_auto_retry_vendor_error(self, exc: VendorPauseError) -> str | None:
        """Bounded auto-retry/backoff for transient vendor errors (#1404).

        For *recoverable* categories (rate limit / service unavailable) a single
        transient ``429``/``503`` should not auto-stop an entire unattended run.
        Sleeps with bounded backoff (honoring ``Retry-After`` when the vendor
        exposes it) and returns ``"continue"`` up to ``TRAIGENT_VENDOR_MAX_RETRIES``
        times per category. Returns ``None`` when no auto-retry applies, so the
        caller falls back to the existing resume/stop prompt path. The budget is
        cumulative per category, guaranteeing the loop always terminates.
        """
        category = exc.category
        recoverable = {
            VendorErrorCategory.RATE_LIMIT,
            VendorErrorCategory.SERVICE_UNAVAILABLE,
        }
        if category not in recoverable:
            return None

        max_retries, base_backoff = self._vendor_retry_settings()
        if max_retries <= 0:
            return None

        used = self._vendor_retry_counts.get(category, 0)
        if used >= max_retries:
            logger.warning(
                "traigent.vendor_auto_retry exhausted category=%s attempts=%s/%s",
                category.value,
                used,
                max_retries,
            )
            return None

        self._vendor_retry_counts[category] = used + 1

        # Honor a vendor-provided Retry-After when available, else exponential.
        retry_after = getattr(exc.original_error, "retry_after", None)
        try:
            delay = float(retry_after) if retry_after is not None else None
        except (TypeError, ValueError):
            delay = None
        if delay is None or delay < 0:
            delay = base_backoff * (2**used)
        if delay is None:
            raise RuntimeError("vendor retry delay fallback did not produce a value")
        delay_seconds = float(delay)

        logger.warning(
            "traigent.vendor_auto_retry category=%s attempt=%s/%s backoff=%.2fs "
            "(set TRAIGENT_VENDOR_MAX_RETRIES=0 to disable)",
            category.value,
            used + 1,
            max_retries,
            delay_seconds,
        )
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        return "continue"

    async def _handle_vendor_pause(self, exc: VendorPauseError) -> str:
        """Prompt user to resume or stop after a vendor error.

        Returns:
            ``"continue"`` to retry the trial, ``"break"`` to stop.
        """
        if exc.category is not None:
            auto = await self._maybe_auto_retry_vendor_error(exc)
            if auto is not None:
                return auto
        if self._prompt_adapter is None or exc.category is None:
            return "break"
        decision = self._prompt_adapter.prompt_vendor_pause(
            exc.original_error or exc,
            exc.category,
        )
        if decision == "resume":
            logger.info("User chose to resume after vendor error: %s", exc.category)
            return "continue"
        logger.info("User chose to stop after vendor error: %s", exc.category)
        return "break"

    async def _handle_budget_limit_pause(self) -> str:
        """Prompt user to raise budget limit or stop.

        Returns:
            ``"continue"`` to retry with new limit, ``"break"`` to stop.
        """
        if self._prompt_adapter is None:
            return "break"
        status = self.cost_enforcer.get_status()
        decision = self._prompt_adapter.prompt_budget_pause(
            status.accumulated_cost_usd,
            self.cost_enforcer.config.limit,
        )
        if decision.startswith("raise:"):
            try:
                new_limit = float(decision.split(":")[1])
            except (ValueError, IndexError):
                logger.warning("Malformed budget response: %s", decision)
                return "break"
            self.cost_enforcer.update_limit(new_limit)
            logger.info("User raised cost limit to $%.2f", new_limit)
            return "continue"
        logger.info("User chose to stop at budget limit")
        return "break"

    async def _check_batch_vendor_errors(
        self, results: list[PermittedTrialResult]
    ) -> bool:
        """Check if an entire parallel batch failed with vendor errors.

        When all trials in a batch fail with vendor-like errors (rate limit,
        quota, service unavailable), prompt the user to resume or stop.

        Handles both paths the parallel executor can produce:
        - Raised exceptions (VendorPauseError or vendor-classifiable exceptions)
          carried in ``PermittedTrialResult.result``.
        - Failed ``TrialResult`` objects with a vendor-classifiable
          ``error_message``.

        Returns:
            True if the user chose to stop, False otherwise.
        """
        if not results:
            return False

        from traigent.core.exception_handler import classify_vendor_error

        # Non-recoverable categories get priority so we don't offer "resume" on
        # insufficient funds just because rate_limit happened to come first.
        category_priority = {
            VendorErrorCategory.AUTHENTICATION: 0,
            VendorErrorCategory.INSUFFICIENT_FUNDS: 1,
            VendorErrorCategory.QUOTA_EXHAUSTED: 2,
            VendorErrorCategory.SERVICE_UNAVAILABLE: 3,
            VendorErrorCategory.RATE_LIMIT: 4,
        }

        def _classify(r: PermittedTrialResult) -> VendorErrorCategory | None:
            payload = getattr(r, "result", r)
            if isinstance(payload, VendorPauseError):
                if payload.category is not None:
                    return payload.category
                # category is declared optional; cascade through original_error
                # then the message so an opaquely-wrapped original doesn't drop
                # the trial from the vendor count.
                if payload.original_error is not None:
                    category = classify_vendor_error(payload.original_error)
                    if category is not None:
                        return category
                return classify_vendor_error(RuntimeError(str(payload)))
            if isinstance(payload, Exception):
                return classify_vendor_error(payload)
            if (
                isinstance(payload, TrialResult)
                and payload.status == TrialStatus.FAILED
                and payload.error_message
            ):
                return classify_vendor_error(RuntimeError(payload.error_message))
            return None

        vendor_failures = 0
        best_category: VendorErrorCategory | None = None
        for r in results:
            category = _classify(r)
            if category is None:
                continue
            vendor_failures += 1
            if best_category is None or category_priority.get(
                category, len(category_priority)
            ) < category_priority.get(best_category, len(category_priority)):
                best_category = category

        if vendor_failures == 0 or vendor_failures < len(results):
            return False

        # Pass a category through so _handle_vendor_pause consults the prompt
        # adapter (when present) or defaults to "break" — matching sequential
        # mode semantics. Without a category the adapter would always be
        # bypassed and the user's "resume" preference ignored.
        exc = VendorPauseError(
            f"All {vendor_failures} parallel trials failed with vendor errors",
            category=best_category,
        )
        decision = await self._handle_vendor_pause(exc)
        return decision == "break"

    async def _finalize_optimization(
        self,
        result: OptimizationResult,
        session_id: str | None,
        session_span: Any,
    ) -> None:
        """Finalize optimization: record metrics, update backend, notify callbacks."""
        record_optimization_complete(
            session_span,
            trial_count=len(self._trials),
            best_score=result.best_score,
            best_config=result.best_config,
            stop_reason=self._stop_reason,
        )

        merge_run_metrics_into_session_summary(result)

        # Compute statistical significance badges per objective
        # Guarded: significance is post-processing and must not fail the run
        try:
            orientations: dict[str, str] | None = None
            if self.objective_schema and self.objective_schema.objectives:
                orientations = {
                    obj.name: str(obj.orientation)
                    for obj in self.objective_schema.objectives
                }
            significance = compute_significance(
                trials=result.trials,
                objectives=self.optimizer.objectives,
                objective_orientations=orientations,
                alpha=0.05,
            )
            if significance:
                result.metadata["statistical_significance"] = significance
        except Exception:
            logger.warning(
                "Statistical significance computation failed; "
                "optimization results are unaffected",
                exc_info=True,
            )

        # #1939: finalize the LOCAL session record first — offline / degraded
        # runs have no backend finalize path, and sync eligibility requires a
        # terminal local status ("completed"). No-op for connected
        # tracking-enabled runs (their backend finalize owns the local mirror).
        self.backend_session_manager.finalize_local_session(session_id, self._status)
        if session_id is not None and backend_egress_disabled(self.traigent_config):
            # Surface the syncable local session id on the result so offline
            # users can run `traigent sync <session_id>` directly.
            result.metadata["local_session_id"] = session_id

        persistence_status = "skipped"
        if (
            self.backend_session_manager.backend_tracking_enabled
            and not backend_egress_disabled(self.traigent_config)
            and session_id is not None
        ):
            try:
                await self.backend_session_manager.update_weighted_scores(
                    result, session_id
                )
                # Traigent#1720/#1724 (g2:agg-summary): fold the session-level
                # rollup into the existing finalize request body (mirroring
                # certified_selection) instead of the dead local-only
                # submit_result shim. Build the payload BEFORE finalize so it
                # rides the same request.
                agg_payload = (
                    self.backend_session_manager.build_session_aggregation_payload(
                        result, session_id
                    )
                )

                session_summary = None
                if not getattr(self, "_session_finalized", False):
                    certified_selection = self._build_certified_selection_report()
                    try:
                        session_summary = self.backend_session_manager.finalize_session(
                            session_id,
                            self._status,
                            certified_selection=certified_selection,
                            session_aggregation=agg_payload,
                        )
                    finally:
                        self._session_finalized = True
                self.backend_session_manager.attach_session_metadata(
                    result, session_id, session_summary
                )
                if agg_payload is not None and not session_aggregation_echoed(
                    session_summary
                ):
                    # A rollup was sent, but the finalize response did not
                    # echo it back — either an older backend without this
                    # field (safe to merge before the backend deploys) or the
                    # backend rejected/dropped it. Trial-level results and
                    # session finalize above did succeed, so this is not a
                    # full failure; but 'succeeded' must not be reported for
                    # a rollup the backend never confirmed.
                    persistence_status = "degraded"
                    result.metadata["persistence_degraded_reason"] = (
                        "session aggregation rollup was sent to the backend "
                        "finalize endpoint but was not acknowledged in the "
                        "response (the backend may not yet support "
                        "session_aggregation); trial-level results and "
                        "session finalize were persisted normally."
                    )
                    logger.warning(
                        "Session aggregation rollup for session_id=%s was not "
                        "acknowledged by the backend finalize response; "
                        "marking persistence_status=degraded (trial results "
                        "and finalize succeeded).",
                        session_id,
                    )
                else:
                    # agg_payload is None (nothing to persist — not a
                    # degradation) or the backend echoed it back (succeeded).
                    persistence_status = "succeeded"
            except Exception as exc:
                persistence_status = "failed"
                result.metadata["persistence_error"] = str(exc)
                logger.error(
                    "Backend persistence failed after optimization; backend session "
                    "left RUNNING. Run `traigent local sync` or check portal to "
                    "finalize or repair the session. Keeping result source=%s, "
                    "session_id=%s, and marking persistence_status=failed: %s",
                    result.source,
                    session_id,
                    exc,
                )

            # Populate experiment_id and cloud_url from session metadata
            self._populate_experiment_cloud_url(result)
        rejection_reason = self.backend_session_manager.backend_rejection_reason
        if rejection_reason:
            if persistence_status in {"skipped", "succeeded"}:
                persistence_status = "degraded"
            result.metadata["persistence_reason"] = "rejected"
            result.metadata["persistence_rejected"] = True
            result.metadata["persistence_rejection_reason"] = rejection_reason
            self.traigent_config.persistence_reason = "rejected"
            self.traigent_config.persistence_rejection_reason = rejection_reason

        # #1938: the backend closed the tracking session while the LOCAL
        # optimizer kept sequencing — the run completed its full local
        # enumeration, but backend tracking is only PARTIAL. Mark it so.
        if self.backend_session_manager.backend_remote_early_complete:
            if persistence_status in {"skipped", "succeeded"}:
                persistence_status = "degraded"
            result.metadata["backend_tracking"] = "partial"
            result.metadata["backend_tracking_partial_reason"] = (
                self.backend_session_manager.backend_remote_early_complete_reason
                or "backend closed the tracking session before the local "
                "enumeration finished"
            )

        result.metadata["persistence_status"] = persistence_status
        self.traigent_config.persistence_status = persistence_status

        await self._submit_usage_analytics()

        # Submit collected workflow traces and graph to backend
        # (#1938: never to a session the backend already closed server-side)
        if (
            self.backend_session_manager.backend_tracking_enabled
            and not self.backend_session_manager.backend_remote_early_complete
        ):
            await self._submit_workflow_traces(session_id)

        self.callback_manager.on_optimization_complete(result)

        # Issue #1265: if a backend-tracking run degraded to local-only, say so
        # prominently at the end (the per-trial warning fired once mid-run; this
        # is the final, unmissable summary tied to the result's source marker).
        if self.backend_session_manager.backend_degraded:
            rejection_reason = self.backend_session_manager.backend_rejection_reason
            if rejection_reason:
                logger.warning(
                    "⚠️  Optimization %s finished in LOCAL-ONLY mode "
                    "(source='local_fallback'): "
                    "your config was rejected by the backend: %s. Results "
                    "were computed and stored locally and are NOT on the cloud "
                    "backend; metadata includes persistence_reason='rejected'.",
                    self._optimization_id,
                    rejection_reason,
                )
            else:
                logger.warning(
                    "⚠️  Optimization %s finished in LOCAL-ONLY mode "
                    "(source='local_fallback'): "
                    "the Traigent backend was unreachable during the run, so results "
                    "were computed and stored locally and are NOT on the cloud "
                    "backend. They will sync on the next successful run.",
                    self._optimization_id,
                )

        cost_status = self.cost_enforcer.get_status()
        logger.info(
            f"Optimization {self._optimization_id} completed: "
            f"{len(self._trials)} trials, best score: "
            f"{'N/A' if result.best_score is None else f'{result.best_score:.4f}'}, "
            f"total cost: ${cost_status.accumulated_cost_usd:.4f}"
        )

    def _finalize_failed_backend_session(
        self,
        session_id: str | None,
        failure: BaseException,
    ) -> None:
        """Best-effort terminal finalize for exception exits."""
        # #1939: the LOCAL record must go terminal even when the backend
        # finalize below is skipped (offline / tracking disabled).
        self.backend_session_manager.finalize_local_session(
            session_id, OptimizationStatus.FAILED
        )
        if (
            getattr(self, "_session_finalized", False)
            or not self.backend_session_manager.backend_tracking_enabled
            or backend_egress_disabled(self.traigent_config)
            or session_id is None
        ):
            return

        terminal_reason = self._stop_reason or type(failure).__name__
        try:
            self.backend_session_manager.finalize_session(
                session_id,
                OptimizationStatus.FAILED,
            )
        except Exception as finalize_error:
            logger.warning(
                "Failed to finalize backend session %s as FAILED after %s: %s",
                session_id,
                terminal_reason,
                finalize_error,
                exc_info=True,
            )
        finally:
            self._session_finalized = True

    async def _finalize_user_cancelled_optimization(
        self,
        session_id: str | None,
        session_span: Any,
        interrupt: BaseException,
    ) -> OptimizationResult:
        """Finalize completed trials after a user interrupt."""
        self._stop_reason = "user_cancelled"
        self._status = OptimizationStatus.CANCELLED
        logger.warning(
            "Optimization %s interrupted by user (%s); stopping and finalizing "
            "with %d completed trial(s).",
            self._optimization_id,
            type(interrupt).__name__,
            len(self._trials),
        )
        result = self._create_optimization_result()
        try:
            await self._finalize_optimization(result, session_id, session_span)
        except Exception as finalize_error:
            logger.warning(
                "Finalization failed after user interrupt; returning partial "
                "optimization result with %d completed trial(s): %s",
                len(self._trials),
                finalize_error,
                exc_info=True,
            )
        # asyncio.run() delivers Ctrl-C to the running coroutine as
        # CancelledError before it can re-raise KeyboardInterrupt. This public
        # optimize() boundary intentionally returns the same cancelled partial
        # result as the direct KeyboardInterrupt path; a second cancellation
        # during finalization is not caught above and still propagates.
        return result

    def _fail_closed_on_empty_smart_managed_run(self) -> None:
        """Reject a cloud-required smart run that executed zero trials.

        A smart algorithm (``bayesian``/``tpe``/``cmaes``/``nsga2``/
        ``optuna*``) resolves to a ``CLOUD_REQUIRED`` policy whose managed
        cloud path must either run trials or raise. When that managed path
        returns without executing a single trial, the run would otherwise be
        finalized as a silent ``COMPLETED`` result with ``best_config=None`` —
        the exact silent-empty failure of issue #1681. Surface it as an
        actionable error instead of a hollow success.

        Deliberately narrow so it never hijacks a legitimate empty stop:

        * only fires for a genuinely empty run (``len(self._trials) == 0``);
        * only when the resolved policy is ``CLOUD_REQUIRED`` (a smart
          algorithm), never for local/hybrid/cloud-brain runs;
        * leaves an explicit ``max_trials<=0`` no-op run alone (mirrors the
          ``_try_cloud_execution`` guard for non-positive trial budgets);
        * defers to already-owned stop causes (timeout / user cancel / cost
          limit #1684 / vendor or network error) rather than relabeling them.
        """

        if self._trials:
            return
        policy = policy_from_config(self.traigent_config)
        if not policy_is_cloud_required(policy):
            return
        if self._max_trials is not None and self._max_trials <= 0:
            return
        if self._stop_reason in _EMPTY_SMART_RUN_OWNED_STOP_REASONS:
            return

        algorithm = getattr(policy, "algorithm", None) or "the requested algorithm"
        raise OptimizationError(
            f"Smart optimization ('{algorithm}') requires the Traigent managed "
            "cloud service, but the run finished without executing a single "
            "trial (0 trials, no best configuration). The managed backend "
            "path returned no trial guidance, and a cloud-required run must "
            "not silently report success. The local SDK runs only 'grid' and "
            "'random'; connect to a Traigent backend that provides smart "
            "optimization, or call "
            "optimize(algorithm='grid') / optimize(algorithm='random') to run "
            "locally."
        )

    async def _run_optimization_with_tracing(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        session_id: str | None,
        function_identifier: str | None,
        session_span: Any,
    ) -> OptimizationResult:
        """Run optimization with tracing enabled."""
        self._check_cost_approval(dataset)

        try:
            # Check for immediate timeout
            if self.timeout is not None and self.timeout == 0:
                self._stop_reason = "timeout"
                self._status = OptimizationStatus.CANCELLED
                return self._create_optimization_result()

            if self.timeout is not None and self.timeout > 0:
                # Hard wall-clock watchdog (issue #1266). The per-trial
                # _should_stop() timeout is only evaluated *between* trials, so a
                # single hung trial — a deadlocked sampler, a stuck LLM call, or a
                # worker thread aborted by a native (pyo3) panic whose future
                # never resolves — would otherwise hang the run forever despite
                # an explicit timeout. wait_for() enforces the deadline even when
                # control never returns to the loop's own check, turning an
                # indefinite deadlock into a clean, catchable stop that still
                # finalizes with the best trial found so far. A bounded grace
                # margin (25% of the budget, floored at 1s and capped at 5min)
                # lets the normal between-trial check stop cleanly first, while
                # still capping wasted LLM spend on a true hang.
                grace = min(max(self.timeout * 0.25, 1.0), 300.0)
                watchdog_deadline = self.timeout + grace
                try:
                    await asyncio.wait_for(
                        self._run_optimization_loop(
                            func, dataset, session_id, function_identifier
                        ),
                        timeout=watchdog_deadline,
                    )
                except TimeoutError:
                    self._stop_reason = "timeout"
                    logger.warning(
                        "Optimization %s exceeded its wall-clock deadline "
                        "(%.1fs incl. grace); a trial appears to have hung. "
                        "Stopping and finalizing with %d completed trial(s).",
                        self._optimization_id,
                        watchdog_deadline,
                        len(self._trials),
                    )
            else:
                await self._run_optimization_loop(
                    func, dataset, session_id, function_identifier
                )

            # A cloud-required (smart) run that executed zero trials must not be
            # reported as a silent COMPLETED with best_config=None: fail closed
            # with an actionable error instead (issue #1681).
            self._fail_closed_on_empty_smart_managed_run()

            # Set final status
            self._status = (
                OptimizationStatus.CANCELLED
                if self._stop_reason == "timeout"
                else OptimizationStatus.COMPLETED
            )

            result = self._create_optimization_result()
            await self._finalize_optimization(result, session_id, session_span)
            return result

        except (KeyboardInterrupt, asyncio.CancelledError) as interrupt:
            return await self._finalize_user_cancelled_optimization(
                session_id, session_span, interrupt
            )

        except OptimizationError as e:
            self._status = OptimizationStatus.FAILED
            if not self._stop_reason:
                self._stop_reason = "error"
            self._finalize_failed_backend_session(session_id, e)
            raise
        except Exception as e:
            from traigent.knobs import ResolutionError

            self._status = OptimizationStatus.FAILED
            if not self._stop_reason:
                self._stop_reason = "error"
            logger.error(f"Optimization {self._optimization_id} failed: {e}")
            self._finalize_failed_backend_session(session_id, e)
            if isinstance(e, ResolutionError):
                # RFC 0001 §3.4: the typed fail-closed governance rejection
                # IS the public contract — callers distinguish a stale
                # certificate / evidence leak from a generic optimization
                # failure. Never dilute it into OptimizationError.
                raise
            raise OptimizationError(f"Optimization failed: {e}") from e
        finally:
            await self._cleanup_backend_client()
            await self._cleanup_hybrid_lifecycle()

    def _abandon_optuna_trial(
        self,
        optuna_trial_id: int | None,
        *,
        reason: str,
        config: dict[str, Any] | None = None,
        status: TrialStatus = TrialStatus.PRUNED,
        pruned_step: int = 0,
    ) -> None:
        """Finalize an asked Optuna trial that will never be executed.

        Deferred cleanup: this ask/tell residual is guarded for local execution
        and should be genericized or removed with the parallel path follow-up.

        Creates a TrialResult for the abandoned trial and:
        1. Appends it to self._trials so it appears in OptimizationResult
        2. Logs the trial via the logger facade
        3. Notifies the optimizer so it can update its internal state

        Args:
            optuna_trial_id: The Optuna trial ID to abandon.
            reason: Human-readable reason for abandonment.
            config: The configuration that was to be evaluated (if available).
            status: Trial status (default PRUNED).
            pruned_step: Step at which pruning occurred.
        """

        if optuna_trial_id is None:
            return
        if not isinstance(optuna_trial_id, int):
            logger.debug(
                "Skipping abandon for non-int optuna_trial_id=%r", optuna_trial_id
            )
            return

        trial_result = TrialResult(
            trial_id=f"optuna_{optuna_trial_id}",
            config=config or {},
            metrics={},
            status=status,
            duration=0.0,
            timestamp=datetime.now(UTC),
            error_message=reason,
            metadata={
                "pruned_step": pruned_step,
                "stop_reason": reason,
                "abandoned": True,
            },
        )

        # Record the abandoned trial in the trials list and log it
        self._trials.append(trial_result)
        self._log_trial(trial_result)

        self._notify_optimizer_of_result(trial_result, optuna_trial_id)

    def _extract_prune_step(self, metadata: dict[str, Any] | None) -> int:
        """Extract prune step from trial metadata."""
        if not metadata:
            return 0
        for key in ("pruned_step", "pruned_at_step", "step"):
            raw_value = metadata.get(key)
            if raw_value is not None:
                try:
                    return int(raw_value)
                except (TypeError, ValueError):
                    continue
        return 0

    def _extract_objective_values(self, metrics: dict[str, Any]) -> list[float]:
        """Extract objective values from trial metrics."""
        values: list[float] = []
        for objective in self.optimizer.objectives:
            metric_value = metrics.get(objective)
            # Fallback for common objective aliases
            if metric_value is None and objective in {"accuracy", "success_rate"}:
                metric_value = metrics.get("accuracy") or metrics.get("success_rate")
            # The per-config ``cost`` metric can be decoupled from the real spend
            # (provider/meta-reported total) and arrive missing or 0.0 while
            # ``total_cost`` is correct. A ``minimize cost`` objective pinned at 0
            # is inert — every config looks free — so prefer ``total_cost`` when
            # the ``cost`` metric is absent or zero (#1423). ``total_cost`` is the
            # SDK's authoritative aggregate cost and is never wrongly zeroed.
            if objective == "cost" and not metric_value:
                total_cost = metrics.get("total_cost")
                if total_cost:
                    metric_value = total_cost
            if metric_value is None and objective in {"cost", "latency", "error"}:
                metric_value = metrics.get(objective, 0.0)
            values.append(float(metric_value if metric_value is not None else 0.0))
        return values

    def _notify_pruned_trial(
        self, trial_result: TrialResult, optuna_trial_id: int
    ) -> bool:
        """Notify optimizer of pruned/cancelled trial. Returns True if handled."""
        metadata = trial_result.metadata or {}
        prune_step = self._extract_prune_step(metadata)
        reason = (
            trial_result.error_message or metadata.get("stop_reason") or "trial_pruned"
        )

        if hasattr(self.optimizer, "report_trial_pruned"):
            self.optimizer.report_trial_pruned(optuna_trial_id, prune_step)  # type: ignore[attr-defined]
            return True

        if hasattr(self.optimizer, "report_trial_result"):
            self.optimizer.report_trial_result(  # type: ignore[arg-type, attr-defined]
                optuna_trial_id,
                None,
                metadata={
                    "state": "pruned",
                    "pruned_at_step": prune_step,
                    "reason": reason,
                },
            )
            return True

        if hasattr(self.optimizer, "report_trial_failure"):
            self.optimizer.report_trial_failure(optuna_trial_id, reason)  # type: ignore[attr-defined]
            return True

        return False

    def _notify_completed_trial(
        self, trial_result: TrialResult, optuna_trial_id: int
    ) -> bool:
        """Notify optimizer of completed trial. Returns True if handled."""
        if not hasattr(self.optimizer, "report_trial_result"):
            return False

        metrics = trial_result.metrics or {}
        values = self._extract_objective_values(metrics)
        payload: float | list[float] = values if len(values) > 1 else values[0]
        self.optimizer.report_trial_result(optuna_trial_id, payload)
        return True

    def _notify_failed_trial(
        self, trial_result: TrialResult, optuna_trial_id: int
    ) -> bool:
        """Notify optimizer of failed trial. Returns True if handled."""
        if not hasattr(self.optimizer, "report_trial_failure"):
            return False

        self.optimizer.report_trial_failure(
            optuna_trial_id, trial_result.error_message or "Unknown error"
        )
        return True

    def _notify_optimizer_of_result(
        self, trial_result: TrialResult, optuna_trial_id: int | None
    ) -> None:
        """Notify optimizers that support ask/tell about trial outcomes."""
        if optuna_trial_id is None or not isinstance(optuna_trial_id, int):
            if optuna_trial_id is not None:
                logger.debug(
                    "Skipping optimizer notification for non-int optuna_trial_id=%r",
                    optuna_trial_id,
                )
            return

        status = trial_result.status
        if status in {TrialStatus.PRUNED, TrialStatus.CANCELLED}:
            self._notify_pruned_trial(trial_result, optuna_trial_id)
        elif status == TrialStatus.COMPLETED:
            self._notify_completed_trial(trial_result, optuna_trial_id)
        elif status == TrialStatus.FAILED:
            self._notify_failed_trial(trial_result, optuna_trial_id)

    def _should_stop(self, trial_count: int) -> bool:
        """Check if optimization should stop.

        Args:
            trial_count: Current number of trials completed

        Returns:
            True if optimization should stop
        """
        # Evaluate reusable stop conditions first
        stop_triggered, reason = self._stop_condition_manager.should_stop(self._trials)
        if stop_triggered:
            # Map stop condition reasons to StopReason literals
            reason_mapping: dict[str | None, StopReason] = {
                "max_samples": "max_samples_reached",
                "max_trials": "max_trials_reached",
                "plateau": "plateau",
                "cost_limit": "cost_limit",
                "timeout": "timeout",
                "metric_limit": "metric_limit",
                "convergence": "convergence",
                "semantic_saturation": "semantic_saturation",
            }
            mapped_reason = reason_mapping.get(reason, "condition")
            if mapped_reason == "cost_limit" or not self._stop_reason:
                self._stop_reason = mapped_reason
            logger.info("Stopping: %s condition triggered", self._stop_reason)
            return True

        elapsed = 0.0
        if self.timeout and self._start_time:
            elapsed = time.time() - self._start_time

        logger.debug(
            "Stop check: trials=%s elapsed=%.2fs best_score=%s",
            trial_count,
            elapsed,
            getattr(self.optimizer, "_best_score", None),
        )

        if self.timeout and self._start_time and elapsed >= self.timeout:
            logger.info(f"Stopping: reached timeout ({self.timeout}s)")
            self._stop_reason = "timeout"
            return True

        optimizer_stop = self.optimizer.should_stop(self._trials)
        logger.debug("Optimizer should_stop returned %s", optimizer_stop)
        if optimizer_stop:
            logger.info("Stopping: optimizer requested stop")
            self._stop_reason = "optimizer"
            return True

        return False

    def _log_progress(self, trial_count: int) -> None:
        """Log optimization progress.

        Args:
            trial_count: Current number of trials
        """
        total_trials = len(self._trials)
        success_count = self._successful_trials

        best_score: float | None = None
        best_trial = self.best_result
        if best_trial and self.optimizer.objectives:
            primary_objective = self.optimizer.objectives[0]
            best_score = best_trial.get_metric(primary_objective)

        success_rate = (success_count / total_trials) if total_trials else 0.0

        elapsed = time.time() - self._start_time if self._start_time else 0.0

        logger.info(
            f"Progress: {trial_count} trials, "
            f"best score: {'N/A' if best_score is None else f'{best_score:.4f}'}, "
            f"success rate: {success_rate:.2%}, "
            f"elapsed: {elapsed:.1f}s"
        )

    def register_metric(self, spec: MetricSpec) -> None:
        """Register or override a metric specification for aggregation."""
        self.metric_registry.register(spec)

    def _log_trial(self, trial_result: TrialResult) -> None:
        """Legacy hook for logging trial results."""
        if self._logger_facade.logger is not self._logger:
            self._logger_facade.attach(self._logger)
        self._logger_facade.log_trial_result(trial_result)

    def _estimate_optimization_cost(self, dataset: Dataset) -> float:
        """Estimate optimization cost. Delegates to CostEstimator."""
        return cast(float, self._cost_estimator.estimate_optimization_cost(dataset))

    def _extract_trial_cost(self, trial_result: TrialResult) -> float | None:
        """Extract cost from trial result. Delegates to CostEstimator."""
        return cast(float | None, CostEstimator.extract_trial_cost(trial_result))

    def _log_checkpoint(
        self,
        optimizer_state: dict[str, Any],
        trials_history: list[TrialResult],
        trial_count: int,
    ) -> None:
        """Legacy hook for saving checkpoints."""
        if self._logger_facade.logger is not self._logger:
            self._logger_facade.attach(self._logger)
        self._logger_facade.save_checkpoint(
            optimizer_state=optimizer_state,
            trials_history=trials_history,
            trial_count=trial_count,
        )

    def _build_result_metadata(
        self,
        session_summary: dict[str, Any] | None,
        safeguards_telemetry: dict[str, Any],
        strategy_preset_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            **({} if session_summary is None else {"session_summary": session_summary}),
            "safeguards": safeguards_telemetry,
        }
        if strategy_preset_metadata is not None:
            metadata["strategy_preset"] = dict(strategy_preset_metadata)

        if self._function_descriptor is not None:
            descriptor = self._function_descriptor
            metadata.update(
                {
                    "function_identifier": descriptor.identifier,
                    "function_name": descriptor.display_name,
                    "function_module": descriptor.module,
                    "function_relative_path": descriptor.relative_path,
                    "function_slug": descriptor.slug,
                }
            )

        # Flag offline mode so callbacks can show appropriate hints
        if is_offline_requested(policy_from_config(self.traigent_config)):
            metadata["offline_mode"] = True

        # Warm-start provenance: surface the prior experiment id that seeded
        # this run so users can query result.metadata["warm_start_from"].
        # Only present for warm-started runs; absent otherwise.
        if self._warm_start_from:
            metadata["warm_start_from"] = self._warm_start_from

        return metadata

    def _create_optimization_result(self) -> OptimizationResult:
        """Create final optimization result.

        Returns:
            OptimizationResult with all trial data
        """
        strict_mode = self._is_strict_evidence_mode()
        certified_config: dict[str, Any] | None = None
        certified_score: float | None = None
        if (
            strict_mode
            and self._best_trial_cached is not None
            and self._certified_promotions > 0
        ):
            # Only an incumbent that won at least one explicit gate
            # promotion is a certified winner. The FIRST trial seeds the
            # incumbent as comparison initialization, NOT as certification —
            # a strict run with zero gate promotions returns
            # NO_CERTIFIED_SELECTION (review finding: first-incumbent
            # overclaim).
            config_space_keys = set(getattr(self.optimizer, "config_space", {}).keys())
            incumbent = self._best_trial_cached
            certified_config = {
                key: value
                for key, value in (incumbent.config or {}).items()
                if not config_space_keys or key in config_space_keys
            }
            primary = (
                self.optimizer.objectives[0] if self.optimizer.objectives else None
            )
            certified_score = (
                incumbent.get_metric(primary)
                if primary and hasattr(incumbent, "get_metric")
                else None
            )
        # Build objective orientation map from ObjectiveSchema when available
        # (F-B fix: honour declared orientation, not name-pattern heuristics).
        _obj_orientations: dict[str, str] | None = None
        if self.objective_schema and self.objective_schema.objectives:
            _obj_orientations = {
                obj.name: str(obj.orientation)
                for obj in self.objective_schema.objectives
            }
        primary_objective = (
            self.optimizer.objectives[0] if self.optimizer.objectives else None
        )
        selection = select_best_configuration(
            trials=self._trials,
            # Objectives-free runs are supported (weighted scoring disabled,
            # see BaseOptimizer); every sibling objectives[0] site in this
            # module guards this and the terminal selection must too (#1108
            # review NB). None ⇒ the selector's honest no-eligible shape;
            # strict mode is certificate-driven and never reads it.
            primary_objective=primary_objective,
            config_space_keys=set(getattr(self.optimizer, "config_space", {}).keys()),
            aggregate_configs=not self.traigent_config.is_local_mode(),
            tie_breakers=self._tie_breakers or None,
            band_target=self._band_target,
            objective_order=self.optimizer.objectives,
            comparability_mode=self.traigent_config.get_comparability_mode(),
            require_certified=strict_mode,
            certified_config=certified_config,
            certified_score=certified_score,
            objective_orientations=_obj_orientations,
            # Weighted-selection gating happens inside the selector: any
            # >1-objective non-banded schema — uniform/default weights
            # included — selects by the weighted aggregate (#1682, #1846);
            # single-objective and banded schemas keep legacy ranking.
            objective_schema=self.objective_schema,
        )
        best_config = selection.best_config
        best_score = selection.best_score
        best_config_margin = selection.best_config_margin
        # Issue #1866: surface a statistical-tie winner once per run. When the
        # winner-vs-runner-up margin is not significant, the "adopt best_config"
        # action is being taken on noise — name both configs, the objective, and
        # the p-value so the user can prefer the cheaper/faster config or add
        # examples. Additive: best_config itself is unchanged.
        if (
            isinstance(best_config_margin, dict)
            and best_config_margin.get("verdict") == "statistical_tie"
        ):
            logger.warning(
                "best_config is a STATISTICAL TIE on '%s': winner (trial %s) vs "
                "runner-up (trial %s) delta=%.4g, p=%.4g (alpha=%s). The winner "
                "is statistically interchangeable with the runner-up on the "
                "primary objective; prefer the cheaper/faster config or add "
                "examples rather than treating the margin as a decision.",
                best_config_margin.get("primary_objective"),
                selection.best_trial_id,
                best_config_margin.get("runner_up_trial_id"),
                best_config_margin.get("delta"),
                best_config_margin.get("p_value"),
                best_config_margin.get("alpha"),
            )
        session_summary = selection.session_summary
        result_status = self._status
        result_warnings: list[str] = []
        result_warning_codes: list[str] = []
        if (
            primary_objective
            and selection.reason_code == NO_RANKING_ELIGIBLE_TRIALS
            and _objective_metric_never_measured(self._trials, primary_objective)
        ):
            available_metric_names = _successful_trial_metric_names(self._trials)
            warning, suggestion = _format_unmatched_objective_warning(
                primary_objective,
                available_metric_names,
            )
            result_warnings.append(warning)
            result_warning_codes.append(_OBJECTIVE_UNMATCHED_WARNING_CODE)
            if session_summary is None:
                session_summary = {}
            else:
                session_summary = dict(session_summary)
            session_summary["reason"] = warning
            session_summary["reason_code"] = selection.reason_code
            session_summary["unmatched_objective"] = primary_objective
            session_summary["available_metrics"] = list(available_metric_names)
            if suggestion:
                session_summary["did_you_mean"] = suggestion
            if result_status == OptimizationStatus.COMPLETED:
                result_status = OptimizationStatus.FAILED
                self._status = result_status

        # Issue #1832 (sibling of #1691): warn when a declared, weighted,
        # matched objective is uniformly constant across the ranking-eligible
        # trials so its weight is a silent no-op. Warning-only — selection math,
        # status, and cost/latency binding are untouched; real priced runs with
        # varying cost/latency never trigger this.
        weighted_selection_schema = resolve_weighted_selection_schema(
            self.objective_schema
        )
        if (
            weighted_selection_schema is not None
            and selection.ranking_eligible_trial_ids
        ):
            eligible_ids = set(selection.ranking_eligible_trial_ids)
            eligible_trials = [
                trial for trial in self._trials if trial.trial_id in eligible_ids
            ]
            inert_objectives = _detect_inert_constant_objectives(
                eligible_trials, weighted_selection_schema
            )
            if inert_objectives:
                all_objective_names = [
                    obj.name for obj in weighted_selection_schema.objectives
                ]
                constant_value = (
                    coerce_finite_objective_score(
                        eligible_trials[0].get_metric(inert_objectives[0])
                    )
                    or 0.0
                )
                inert_warning = _format_inert_objective_warning(
                    inert_objectives, all_objective_names, constant_value
                )
                result_warnings.append(inert_warning)
                result_warning_codes.append(_OBJECTIVE_INERT_CONSTANT_WARNING_CODE)
                logger.warning(inert_warning)
                if session_summary is None:
                    session_summary = {}
                else:
                    session_summary = dict(session_summary)
                session_summary["inert_constant_objectives"] = list(inert_objectives)
        preset_selection = (
            select_strategy_preset(self.strategy_preset, self._trials)
            if self.strategy_preset is not None
            else None
        )
        strategy_preset_metadata = (
            preset_selection.to_metadata()
            if preset_selection is not None
            else (
                self.strategy_preset.to_metadata()
                if self.strategy_preset is not None
                else None
            )
        )

        # Calculate duration
        duration = time.time() - self._start_time if self._start_time else 0.0

        # Create convergence info
        successful_trials = [t for t in self._trials if t.is_successful]
        success_rate = (
            0.0
            if _OBJECTIVE_UNMATCHED_WARNING_CODE in result_warning_codes
            else (len(successful_trials) / len(self._trials) if self._trials else 0.0)
        )
        convergence_info = {
            "total_trials": len(self._trials),
            "successful_trials": len(successful_trials),
            "success_rate": success_rate,
            "algorithm": self.optimizer.__class__.__name__,
        }

        aggregated_metrics = aggregate_metrics(
            self._trials,
            self.metric_registry,
        )
        processed_metrics = aggregated_metrics.processed_metrics
        total_cost = aggregated_metrics.total_cost
        total_tokens = aggregated_metrics.total_tokens

        safeguards_telemetry = build_safeguards_telemetry(
            trials_prevented=self._trials_prevented,
            configs_deduplicated=self.cache_policy_handler.configs_deduplicated,
            examples_capped=self._examples_capped,
            cached_results_reused=self._cached_results_reused,
            ci_blocks=self._ci_blocks,
            cache_policy_used=self.cache_policy_handler.cache_policy_used,
        )

        # Generate human-readable run label
        func_name = "run"
        if self._function_descriptor is not None:
            func_name = (
                self._function_descriptor.display_name
                or self._function_descriptor.identifier
                or "run"
            )
        now = datetime.now(UTC)
        run_label = generate_run_label(func_name, self._optimization_id, now)

        result_metadata = self._build_result_metadata(
            session_summary,
            safeguards_telemetry,
            strategy_preset_metadata,
        )
        semantic_saturation = (
            self._stop_condition_manager.semantic_saturation_diagnostics(self._trials)
        )
        if semantic_saturation is not None:
            result_metadata["semantic_saturation"] = semantic_saturation

        if selection.best_trial_id:
            result_metadata["best_trial_id"] = selection.best_trial_id

        # Provenance marker (issue #1265): "backend" only when remote tracking
        # stayed healthy; "local" when an intentionally-local mode ran or a
        # backend-tracking run degraded to local-only mid-flight.
        source = self.backend_session_manager.result_source(len(self._trials))
        result_metadata["source"] = source
        fallback_reason = (
            getattr(self.traigent_config, "fallback_reason", None)
            or self.backend_session_manager.fallback_reason
        )
        if source == SOURCE_LOCAL_FALLBACK and fallback_reason:
            result_metadata["fallback_reason"] = fallback_reason
        rejection_reason = (
            self.backend_session_manager.backend_rejection_reason
            or getattr(self.traigent_config, "persistence_rejection_reason", None)
        )
        if rejection_reason:
            result_metadata["persistence_reason"] = "rejected"
            result_metadata["persistence_rejected"] = True
            result_metadata["persistence_rejection_reason"] = rejection_reason
        if getattr(self.traigent_config, "persistence_status", None):
            result_metadata["persistence_status"] = (
                self.traigent_config.persistence_status
            )

        # Create optimization result
        optimization_result = OptimizationResult(
            trials=self._trials.copy(),
            best_config=best_config,
            best_score=best_score,
            optimization_id=self._optimization_id,
            duration=duration,
            convergence_info=convergence_info,
            status=result_status,
            objectives=self.optimizer.objectives,
            algorithm=self.optimizer.__class__.__name__,
            timestamp=now,
            total_cost=total_cost if total_cost > 0 else None,
            total_tokens=total_tokens if total_tokens > 0 else None,
            metrics=processed_metrics,
            metadata=result_metadata,
            preset_selection=preset_selection,
            stop_reason=self._stop_reason,
            reason_code=selection.reason_code,
            run_label=run_label,
            warnings=result_warnings,
            warning_codes=result_warning_codes,
            source=source,
            best_config_margin=best_config_margin,
        )

        # Log optimization completion
        if self._logger:
            weighted_results = None
            if len(self.optimizer.objectives) > 1:
                # Thread the declared schema (same reason as
                # BackendSessionManager.update_weighted_scores): without it,
                # minimize orientations fall back to name-pattern autodetect
                # and the logged weighted results can contradict terminal
                # best_config (#1846 follow-up).
                weighted_results = optimization_result.calculate_weighted_scores(
                    objective_schema=self.objective_schema
                )

            self._logger_facade.log_session_end(
                optimization_result=optimization_result,
                weighted_results=weighted_results,
            )

        return optimization_result
