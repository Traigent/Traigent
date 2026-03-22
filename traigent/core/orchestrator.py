"""Optimization orchestration engine."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import copy
import inspect
import math
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from traigent.api.agent_inference import (
    build_agent_configuration,
    extract_parameter_agents,
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

from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.cache_policy import CachePolicyHandler
from traigent.core.cost_enforcement import (
    DEFAULT_COST_LIMIT_USD,
    CostEnforcer,
    CostEnforcerConfig,
    Permit,
)
from traigent.core.cost_estimator import CostEstimator
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
from traigent.core.result_selection import TieBreaker, select_best_configuration
from traigent.core.sample_budget import SampleBudgetManager
from traigent.core.stat_significance import compute_significance
from traigent.core.stop_condition_manager import StopConditionManager
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.core.utils import extract_examples_attempted
from traigent.core.workflow_trace_manager import WorkflowTraceManager
from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.metrics.registry import clone_registry
from traigent.optimizers.base import BaseOptimizer
from traigent.tvl.promotion_gate import PromotionGate
from traigent.utils.callbacks import CallbackManager, OptimizationCallback, ProgressInfo
from traigent.utils.env_config import is_backend_offline
from traigent.utils.exceptions import OptimizationError, VendorPauseError
from traigent.utils.function_identity import (
    FunctionDescriptor,
    resolve_function_descriptor,
)
from traigent.utils.hashing import generate_run_label
from traigent.utils.logging import get_logger
from traigent.utils.objectives import is_minimization_objective
from traigent.utils.optimization_logger import OptimizationLogger

from .tracing import optimization_session_span, record_optimization_complete

logger = get_logger(__name__)

# Orchestrator constants
PROGRESS_LOG_INTERVAL = 10  # Log progress every N trials


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
        default_config = kwargs.pop("default_config", None)
        self._init_constraints(raw_constraints)

        self.objectives, self.objective_schema = prepare_objectives(
            objectives, objective_schema
        )

        # TVL 0.9 tie-breaker and promotion gate configuration
        self._tie_breakers: dict[str, TieBreaker] = (
            kwargs.pop("tie_breakers", None) or {}
        )
        self._promotion_gate: PromotionGate | None = kwargs.pop("promotion_gate", None)
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

        self.backend_session_manager = BackendSessionManager(
            backend_client=self.backend_client,
            traigent_config=self.traigent_config,
            objectives=self.objectives,
            objective_schema=self.objective_schema,
            optimizer=self.optimizer,
            optimization_id=self._optimization_id,
            optimization_status=self._status,
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

        return build_agent_configuration(
            configuration_space=self.optimizer.config_space,
            explicit_agents=explicit_agents,
            agent_prefixes=agent_prefixes,
            agent_measures=agent_measures,
            global_measures=global_measures,
            parameter_agents=parameter_agents,
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

    def _get_budget_limit(self) -> float | None:
        """Extract and validate budget limit from config."""
        raw_budget_limit = self.config.get("budget_limit")
        if raw_budget_limit is None:
            return None
        budget_limit = float(raw_budget_limit)
        return budget_limit if budget_limit > 0 else None

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
        cost_approved = self.config.get("cost_approved", False)
        cost_config = None
        if cost_limit is not None or cost_approved:
            cost_config = CostEnforcerConfig(
                limit=float(cost_limit) if cost_limit else DEFAULT_COST_LIMIT_USD,
                approved=bool(cost_approved),
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

        self._stop_condition_manager = StopConditionManager(
            max_trials=self._max_trials,
            max_samples=self._max_total_examples,
            samples_include_pruned=samples_include_pruned,
            plateau_window=plateau_window or None,
            plateau_epsilon=plateau_epsilon,
            objective_schema=self.objective_schema,
            budget_limit=self._get_budget_limit(),
            budget_metric=str(self.config.get("budget_metric", "total_cost")),
            include_pruned=bool(self.config.get("budget_include_pruned", True)),
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
        return BackendSessionManager.create_backend_client(self.traigent_config)

    def _initialize_runtime_state(self) -> None:
        self._trials: list[TrialResult] = []
        self._start_time: float | None = None
        self._status = OptimizationStatus.PENDING
        self._optimization_id = str(uuid.uuid4())
        self._stop_reason: StopReason | None = None
        self._logger: OptimizationLogger | None = None
        self._logger_v2: Any | None = None
        self._logger_facade = LoggerFacade()
        self._trials_prevented = 0
        self._examples_capped = 0
        self._cached_results_reused = 0
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

        if not self._trials:
            return None

        # Find trial with best primary objective (assuming first objective is primary)
        if self.optimizer.objectives:
            primary_objective = self.optimizer.objectives[0]
            minimization = is_minimization_objective(primary_objective)
            if minimization:
                best_trial = min(
                    self._trials,
                    key=lambda t: t.metrics.get(primary_objective, float("inf")),
                )
            else:
                best_trial = max(
                    self._trials,
                    key=lambda t: t.metrics.get(primary_objective, float("-inf")),
                )
            self._best_trial_cached = best_trial
            return best_trial

        # If no objectives defined, return last trial
        return self._trials[-1] if self._trials else None

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
        # "no_decision" - fall back to simple comparison
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
            logger.warning(
                "PromotionGate evaluation failed, using simple comparison: %s", e
            )
            return self._simple_is_better(candidate_trial)

    def _simple_is_better(self, trial_result: TrialResult) -> bool:
        """Check if trial_result is better than current best using simple comparison."""
        if self._best_trial_cached is None:
            return True

        if not self.optimizer.objectives:
            return True

        primary_objective = self.optimizer.objectives[0]
        new_score = (
            trial_result.get_metric(primary_objective)
            if hasattr(trial_result, "get_metric")
            else ((trial_result.metrics or {}).get(primary_objective))
        )
        if new_score is None:
            return False

        current_score = (
            self._best_trial_cached.get_metric(primary_objective)
            if hasattr(self._best_trial_cached, "get_metric")
            else (self._best_trial_cached.metrics or {}).get(primary_objective)
        )
        if current_score is None:
            return True

        minimization = is_minimization_objective(primary_objective)
        if minimization:
            return new_score < current_score
        return new_score > current_score

    def _update_best_trial_cache(self, trial_result: TrialResult) -> None:
        if not self.optimizer.objectives:
            self._best_trial_cached = trial_result
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

    async def _cleanup_backend_client(self) -> None:
        """Close backend HTTP session if one was opened."""

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
        self, session_id: str | None, func: Callable[..., Any], dataset: Dataset
    ) -> None:
        if not session_id:
            return

        experiment_name = (
            func.__name__ if hasattr(func, "__name__") else "unknown_function"
        )

        self._logger = OptimizationLogger(
            experiment_name=experiment_name,
            session_id=session_id,
            execution_mode=self.traigent_config.execution_mode or "edge_analytics",
        )
        self._logger_facade.attach(self._logger)

        self._logger_facade.log_session_start(
            config=(
                self.traigent_config.to_dict()
                if hasattr(self.traigent_config, "to_dict")
                else {}
            ),
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

        if self.backend_client and session_id:
            await self.backend_session_manager.submit_trial(
                trial_result=trial_result,
                session_id=session_id,
                dataset_name=getattr(self, "_dataset_name", "dataset"),
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
            configs.append(baseline_config)
            remaining -= 1
            if remaining <= 0:
                return configs, False

        # Try hybrid/async generation if applicable
        used_async_batch = False
        if self.traigent_config.execution_mode_enum is ExecutionMode.HYBRID:
            async_configs = await self._try_hybrid_batch_generation(dataset, remaining)
            if async_configs:
                configs.extend(async_configs)
                used_async_batch = True

        # Fall back to sequential generation
        if not used_async_batch:
            configs.extend(self._generate_sequential_configs(remaining))

        return configs, used_async_batch

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
            if isinstance(cfg_eval, dict) and "__subset_indices__" in cfg_eval:
                try:
                    idxs = list(cfg_eval.pop("__subset_indices__"))
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
                    cfg_eval,
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
                )
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

    async def _check_batch_vendor_failures(self, results: list[Any]) -> str | None:
        """Return ``"break"`` if every trial in the batch hit a vendor error."""
        if not self._prompt_adapter or not results:
            return None

        from traigent.core.exception_handler import classify_vendor_error

        vendor_failures = sum(
            1
            for r in results
            if self._is_vendor_failure(getattr(r, "result", r), classify_vendor_error)
        )
        if vendor_failures == 0 or vendor_failures < len(results):
            return None

        exc = VendorPauseError(
            f"All {vendor_failures} parallel trials failed with vendor errors",
        )
        return await self._handle_vendor_pause(exc)

    @staticmethod
    def _is_vendor_failure(trial: Any, classify_fn: Callable[..., Any]) -> bool:
        """Check if a single trial result represents a vendor error."""
        return (
            isinstance(trial, TrialResult)
            and trial.status == TrialStatus.FAILED
            and bool(trial.error_message)
            and classify_fn(RuntimeError(trial.error_message)) is not None
        )

    async def _submit_usage_analytics(self) -> None:
        """Submit usage analytics if enabled."""

        if not self.traigent_config.enable_usage_analytics:
            return

        if is_backend_offline():
            logger.debug("Skipping analytics submission: backend is offline")
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
        await self._workflow_trace_manager.submit_traces(session_id)

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
        self._start_time = time.time()
        self._successful_trials = 0
        self._failed_trials = 0
        self._best_trial_cached = None

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

        if function_name and function_name != descriptor.identifier:
            logger.debug(
                "Ignoring supplied function_name '%s' in favour of fully-qualified identifier '%s'",
                function_name,
                descriptor.identifier,
            )

        # Create backend session using manager
        session_context = self.backend_session_manager.create_session(
            func=func,
            dataset=dataset,
            function_descriptor=descriptor,
            max_trials=self.max_trials,
            max_total_examples=self.max_total_examples,
            start_time=self._start_time or time.time(),
            agent_configuration=self._agent_configuration,
        )
        session_id: str | None = session_context.session_id

        if session_id:
            self._initialize_logger(session_id, func, dataset)

        self.callback_manager.on_optimization_start(
            config_space=self.optimizer.config_space,
            objectives=self.optimizer.objectives,
            algorithm=self.optimizer.__class__.__name__,
        )

        return session_id

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

        if self.backend_client:
            # Ensure max_trials is not None
            max_trials_value = self.max_trials if self.max_trials is not None else 10
            logger.info(
                f"Creating session with max_trials={max_trials_value} (self.max_trials={self.max_trials})"
            )

            result = self.backend_client.create_session(
                function_name=identifier,
                search_space=getattr(self.optimizer, "config_space", {}),
                optimization_goal="maximize",  # Default assumption
                metadata={
                    "optimization_id": self._optimization_id,
                    "max_trials": max_trials_value,
                    "function_name": identifier,
                    "function_display_name": (
                        descriptor.display_name if descriptor else identifier
                    ),
                    "evaluation_set": dataset_name or "default_evaluation",
                },
            )
            session_id = self.backend_session_manager.handle_session_creation_result(
                result
            )
            return session_id
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
        """Check cost approval before optimization. Delegates to CostEstimator."""
        self._cost_estimator.check_cost_approval(dataset)

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

    async def _maybe_pause_on_cost_limit(self) -> bool:
        """If stopped for cost_limit, prompt user to raise budget.

        Returns True if the user chose to continue (budget raised).
        """
        if self._stop_reason != "cost_limit" or self._prompt_adapter is None:
            return False
        decision = await self._handle_budget_limit_pause()
        if decision == "continue":
            self._stop_reason = None
            return True
        return False

    def _apply_budget_stop(self, budget_stop: StopReason | None) -> bool:
        """Record a budget stop reason and return True if the loop should break."""
        if not budget_stop:
            return False
        if not self._stop_reason:
            self._stop_reason = budget_stop
        return True

    async def _handle_vendor_pause_in_loop(self, exc: VendorPauseError) -> str:
        """Handle VendorPauseError in the optimization loop.

        Returns ``"break"`` to stop or ``"continue"`` to retry.
        """
        decision = await self._handle_vendor_pause(exc)
        if decision == "break":
            self._stop_reason = "vendor_error"
        return decision

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
            return await self._trial_lifecycle.run_sequential_trial(
                func=func,
                dataset=dataset,
                session_id=session_id,
                function_name=function_identifier,
                trial_count=trial_count,
            )
        except VendorPauseError as e:
            decision = await self._handle_vendor_pause(e)
            if decision == "break":
                self._stop_reason = "vendor_error"
                return trial_count, "break"
            return trial_count, "continue"  # User chose to resume

    async def _handle_vendor_pause(self, exc: VendorPauseError) -> str:
        """Prompt user to resume or stop after a vendor error.

        Returns:
            ``"continue"`` to retry the trial, ``"break"`` to stop.
        """
        if self._prompt_adapter is None or exc.category is None:
            return "break"
        decision = await asyncio.to_thread(
            self._prompt_adapter.prompt_vendor_pause,
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
        decision = await asyncio.to_thread(
            self._prompt_adapter.prompt_budget_pause,
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

        Returns:
            True if the user chose to stop, False otherwise.
        """
        if self._prompt_adapter is None or not results:
            return False

        from traigent.core.exception_handler import classify_vendor_error

        vendor_failures = 0
        for r in results:
            trial = getattr(r, "result", r)
            if (
                isinstance(trial, TrialResult)
                and trial.status == TrialStatus.FAILED
                and trial.error_message
                and classify_vendor_error(RuntimeError(trial.error_message)) is not None
            ):
                vendor_failures += 1

        if vendor_failures == 0 or vendor_failures < len(results):
            return False

        exc = VendorPauseError(
            f"All {vendor_failures} parallel trials failed with vendor errors",
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

        await self.backend_session_manager.update_weighted_scores(result, session_id)
        self.backend_session_manager.submit_session_aggregation(result, session_id)

        session_summary = self.backend_session_manager.finalize_session(
            session_id, self._status
        )
        self.backend_session_manager.attach_session_metadata(
            result, session_id, session_summary
        )

        # Populate experiment_id and cloud_url from session metadata
        exp_id = (result.metadata or {}).get("experiment_id")
        if exp_id:
            result.experiment_id = exp_id
            try:
                from traigent.cloud.sync_manager import build_experiment_url
                from traigent.config.backend_config import BackendConfig

                result.cloud_url = build_experiment_url(
                    BackendConfig.get_cloud_backend_url(), exp_id
                )
            except Exception:
                pass  # Cloud URL is best-effort

        await self._submit_usage_analytics()

        # Submit collected workflow traces and graph to backend
        if self.backend_session_manager.backend_tracking_enabled:
            await self._submit_workflow_traces(session_id)

        self.callback_manager.on_optimization_complete(result)

        cost_status = self.cost_enforcer.get_status()
        logger.info(
            f"Optimization {self._optimization_id} completed: "
            f"{len(self._trials)} trials, best score: "
            f"{'N/A' if result.best_score is None else f'{result.best_score:.4f}'}, "
            f"total cost: ${cost_status.accumulated_cost_usd:.4f}"
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

            await self._run_optimization_loop(
                func, dataset, session_id, function_identifier
            )

            # Set final status
            self._status = (
                OptimizationStatus.CANCELLED
                if self._stop_reason == "timeout"
                else OptimizationStatus.COMPLETED
            )

            result = self._create_optimization_result()
            await self._finalize_optimization(result, session_id, session_span)
            return result

        except Exception as e:
            self._status = OptimizationStatus.FAILED
            logger.error(f"Optimization {self._optimization_id} failed: {e}")
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
            if not self._stop_reason:
                # Map stop condition reasons to StopReason literals
                reason_mapping: dict[str | None, StopReason] = {
                    "max_samples": "max_samples_reached",
                    "max_trials": "max_trials_reached",
                    "plateau": "plateau",
                    "cost_limit": "cost_limit",
                    "timeout": "timeout",
                    "budget": "cost_limit",
                }
                self._stop_reason = reason_mapping.get(reason, "condition")
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
        return self._cost_estimator.estimate_optimization_cost(dataset)

    def _extract_trial_cost(self, trial_result: TrialResult) -> float | None:
        """Extract cost from trial result. Delegates to CostEstimator."""
        return CostEstimator.extract_trial_cost(trial_result)

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
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            **({} if session_summary is None else {"session_summary": session_summary}),
            "safeguards": safeguards_telemetry,
        }

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
        if is_backend_offline():
            metadata["offline_mode"] = True

        return metadata

    def _create_optimization_result(self) -> OptimizationResult:
        """Create final optimization result.

        Returns:
            OptimizationResult with all trial data
        """
        selection = select_best_configuration(
            trials=self._trials,
            primary_objective=self.optimizer.objectives[0],
            config_space_keys=set(getattr(self.optimizer, "config_space", {}).keys()),
            aggregate_configs=not self.traigent_config.is_edge_analytics_mode(),
            tie_breakers=self._tie_breakers or None,
            band_target=self._band_target,
            comparability_mode=self.traigent_config.get_comparability_mode(),
        )
        best_config = selection.best_config
        best_score = selection.best_score
        session_summary = selection.session_summary

        # Calculate duration
        duration = time.time() - self._start_time if self._start_time else 0.0

        # Create convergence info
        successful_trials = [t for t in self._trials if t.is_successful]
        convergence_info = {
            "total_trials": len(self._trials),
            "successful_trials": len(successful_trials),
            "success_rate": (
                len(successful_trials) / len(self._trials) if self._trials else 0.0
            ),
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

        # Create optimization result
        optimization_result = OptimizationResult(
            trials=self._trials.copy(),
            best_config=best_config,
            best_score=best_score,
            optimization_id=self._optimization_id,
            duration=duration,
            convergence_info=convergence_info,
            status=self._status,
            objectives=self.optimizer.objectives,
            algorithm=self.optimizer.__class__.__name__,
            timestamp=now,
            total_cost=total_cost if total_cost > 0 else None,
            total_tokens=total_tokens if total_tokens > 0 else None,
            metrics=processed_metrics,
            metadata=self._build_result_metadata(session_summary, safeguards_telemetry),
            stop_reason=self._stop_reason,
            run_label=run_label,
        )

        # Log optimization completion
        if self._logger:
            weighted_results = None
            if len(self.optimizer.objectives) > 1:
                weighted_results = optimization_result.calculate_weighted_scores()

            self._logger_facade.log_session_end(
                optimization_result=optimization_result,
                weighted_results=weighted_results,
            )

        return optimization_result
