"""Optimization orchestration engine."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import copy
import inspect
import math
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

# Type-only import for BackendIntegratedClient (runtime import is lazy)
if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient

from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.cache_policy import CachePolicyHandler
from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig, Permit
from traigent.core.logger_facade import LoggerFacade
from traigent.core.metadata_helpers import (
    merge_run_metrics_into_session_summary,
)
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
    prepare_evaluation_config,
    prepare_objectives,
    validate_constructor_arguments,
)
from traigent.core.parallel_execution_manager import (
    ParallelExecutionManager,
    PermittedTrialResult,
)
from traigent.core.progress_manager import (
    ProgressManager,
)
from traigent.core.result_selection import (
    TieBreaker,
    _is_minimization_objective,
    select_best_configuration,
)
from traigent.core.sample_budget import SampleBudgetManager
from traigent.core.stop_condition_manager import StopConditionManager
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.core.utils import extract_examples_attempted
from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.metrics.registry import clone_registry
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.callbacks import CallbackManager, OptimizationCallback, ProgressInfo
from traigent.utils.exceptions import (
    OptimizationError,
)
from traigent.utils.function_identity import (
    FunctionDescriptor,
    resolve_function_descriptor,
)
from traigent.utils.env_config import should_show_cloud_notice
from traigent.utils.local_analytics import collect_and_submit_analytics
from traigent.utils.logging import get_logger
from traigent.utils.optimization_logger import OptimizationLogger

from .tracing import (
    optimization_session_span,
    record_optimization_complete,
)

logger = get_logger(__name__)

CLOUD_UNAVAILABLE_NOTICE = (
    "⚠️ Full insights unavailable without Traigent Cloud API key.\n"
    "  You're missing: AI recommendations • trade-off analysis • cost tracking • audit features.\n"
    "  Get yours now at https://traigent.ai"
)

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
        use_versioned_logger: bool = False,
        file_version: str = "2",
        **kwargs: Any,
    ) -> None:
        """Initialize optimization orchestrator.

        Keyword Args:
            default_config: Optional baseline configuration evaluated once before
                optimizer-driven suggestions. Counts toward max_trials.
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
        self._constraints_pre_eval: list[Callable[..., bool]] = []
        self._constraints_post_eval: list[Callable[..., bool]] = []
        if raw_constraints:
            for constraint in raw_constraints:
                if constraint_requires_metrics(constraint):
                    self._constraints_post_eval.append(constraint)
                else:
                    self._constraints_pre_eval.append(constraint)

        self.objectives, self.objective_schema = prepare_objectives(
            objectives, objective_schema
        )

        # TVL 0.9 tie-breaker configuration
        self._tie_breakers: dict[str, TieBreaker] = (
            kwargs.pop("tie_breakers", None) or {}
        )

        # Multi-agent configuration
        explicit_agents: dict[str, AgentDefinition] | None = kwargs.pop("agents", None)
        agent_prefixes: list[str] | None = kwargs.pop("agent_prefixes", None)
        agent_measures: dict[str, list[str]] | None = kwargs.pop("agent_measures", None)
        global_measures: list[str] | None = kwargs.pop("global_measures", None)
        # TVL parameter_agents are pre-extracted from TVL spec (since TVL uses raw
        # values in configuration_space, not ParameterRange objects)
        tvl_parameter_agents: dict[str, str] | None = kwargs.pop(
            "tvl_parameter_agents", None
        )

        # Build agent configuration if any multi-agent params provided
        self._agent_configuration: AgentConfiguration | None = None
        has_multi_agent_config = (
            explicit_agents
            or agent_prefixes
            or agent_measures
            or global_measures
            or tvl_parameter_agents
        )
        if has_multi_agent_config:
            # Extract parameter agents from Range(..., agent="x") parameters
            parameter_agents = extract_parameter_agents(self.optimizer.config_space)
            # Merge with TVL parameter_agents (TVL takes lower priority - decorator
            # Range(..., agent="x") can override TVL if needed)
            if tvl_parameter_agents:
                merged_agents = dict(tvl_parameter_agents)
                merged_agents.update(parameter_agents)  # Decorator overrides TVL
                parameter_agents = merged_agents
            self._agent_configuration = build_agent_configuration(
                configuration_space=self.optimizer.config_space,
                explicit_agents=explicit_agents,
                agent_prefixes=agent_prefixes,
                agent_measures=agent_measures,
                global_measures=global_measures,
                parameter_agents=parameter_agents,
            )

        # Derive band_target from objective_schema if available
        self._band_target: float | None = None
        if self.objective_schema is not None and self.objective_schema.objectives:
            primary_obj = self.objective_schema.objectives[0]
            if hasattr(primary_obj, "band") and primary_obj.band is not None:
                band = primary_obj.band
                if band.center is not None:
                    self._band_target = band.center
                elif band.low is not None and band.high is not None:
                    self._band_target = (band.low + band.high) / 2.0

        self._default_config: dict[str, Any] | None = None
        self._default_config_used = False
        if isinstance(default_config, dict) and default_config:
            self._default_config = copy.deepcopy(default_config)
        elif default_config is not None:
            logger.debug(
                "Ignoring default_config with unexpected type: %s",
                type(default_config).__name__,
            )

        self.use_versioned_logger = use_versioned_logger
        self.file_version = file_version

        self._configure_evaluator_execution_mode()

        self.callback_manager = CallbackManager(
            cast(list[OptimizationCallback] | None, callbacks) if callbacks else None
        )
        self.metric_registry = (
            metric_registry.clone() if metric_registry is not None else clone_registry()
        )

        self._backend_client: BackendIntegratedClient | None = None
        self.backend_client = self._initialize_backend_client()
        self._initialize_runtime_state()

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
        self._trial_lifecycle = TrialLifecycle(self)
        self._initialized = True

    def _configure_stop_conditions(self) -> None:
        """Configure stop conditions and sample budget management.

        Extracts configuration from self.config and initializes:
        - StopConditionManager for trial/sample/budget limits
        - SampleBudgetManager for sample budget tracking
        - CostEnforcer for cost limit enforcement (default: $2 USD per run)
        """
        plateau_window = int(self.config.get("plateau_window", 0) or 0)
        plateau_epsilon = (
            float(self.config.get("plateau_epsilon", 1e-6) or 1e-6)
            if plateau_window > 0
            else None
        )

        raw_budget_limit = self.config.get("budget_limit")
        budget_limit = None
        if raw_budget_limit is not None:
            budget_limit = float(raw_budget_limit)
            if budget_limit <= 0:
                budget_limit = None

        budget_metric = str(self.config.get("budget_metric", "total_cost"))
        include_pruned = bool(self.config.get("budget_include_pruned", True))
        samples_include_pruned = bool(self.config.get("samples_include_pruned", True))

        self._stop_condition_manager = StopConditionManager(
            max_trials=self._max_trials,
            max_samples=self._max_total_examples,
            samples_include_pruned=samples_include_pruned,
            plateau_window=plateau_window or None,
            plateau_epsilon=plateau_epsilon,
            objective_schema=self.objective_schema,
            budget_limit=budget_limit,
            budget_metric=budget_metric,
            include_pruned=include_pruned,
        )

        # TVL 0.9 convergence criteria (hypervolume-based early stopping)
        convergence_metric = self.config.get("convergence_metric")
        convergence_window = self.config.get("convergence_window")
        convergence_threshold = self.config.get("convergence_threshold")
        if (
            convergence_metric == "hypervolume_improvement"
            and convergence_window is not None
            and convergence_threshold is not None
            and self.objective_schema is not None
        ):
            # Skip hypervolume convergence if any objective has band orientation
            # Hypervolume requires maximize/minimize directions only
            directions = [obj.orientation for obj in self.objective_schema.objectives]
            if "band" in directions:
                logger.info(
                    "Skipping hypervolume convergence: band objectives are not "
                    "compatible with hypervolume computation"
                )
            else:
                objective_names = [obj.name for obj in self.objective_schema.objectives]
                self._stop_condition_manager.add_convergence_condition(
                    window=int(convergence_window),
                    threshold=float(convergence_threshold),
                    objective_names=objective_names,
                    directions=cast(list[str], directions),
                )

        # Initialize cost enforcer for cost limit enforcement
        # Respects TRAIGENT_RUN_COST_LIMIT, TRAIGENT_COST_APPROVED, TRAIGENT_MOCK_LLM
        cost_limit = self.config.get("cost_limit")
        cost_approved = self.config.get("cost_approved", False)
        cost_config = None
        if cost_limit is not None or cost_approved:
            cost_config = CostEnforcerConfig(
                limit=float(cost_limit) if cost_limit else 2.0,
                approved=bool(cost_approved),
            )
        self.cost_enforcer = CostEnforcer(config=cost_config)

        # Share cost enforcer with parallel execution manager for permit-based control
        self.parallel_execution_manager.set_cost_enforcer(self.cost_enforcer)

        # Register cost limit stop condition using the shared enforcer
        self._stop_condition_manager.register_cost_limit_condition(self.cost_enforcer)

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

    def _initialize_backend_client(self) -> BackendIntegratedClient | None:
        """Initialize backend client if cloud features are available.

        Returns None when cloud plugin is not installed (graceful degradation).
        Raises FeatureNotAvailableError only when cloud mode is explicitly
        requested but plugin is missing.
        """
        # Try to import cloud module - may not be available if cloud plugin not installed
        try:
            from traigent.cloud.backend_client import (
                BackendClientConfig,
                BackendIntegratedClient,
            )
            from traigent.config.backend_config import BackendConfig
        except ModuleNotFoundError as err:
            # Cloud module not installed - check if this was the cloud module itself
            if err.name and err.name.startswith("traigent.cloud"):
                if self.traigent_config.execution_mode == "cloud":
                    # User explicitly requested cloud mode but plugin not installed
                    from traigent.utils.exceptions import FeatureNotAvailableError

                    raise FeatureNotAvailableError(
                        "Cloud execution mode",
                        plugin_name="traigent-cloud",
                        install_hint="pip install traigent[cloud]",
                    ) from err
                # For edge_analytics or other modes, gracefully degrade to local-only
                logger.info(
                    f"Cloud module not available for {self.traigent_config.execution_mode} mode. "
                    "Continuing with local storage only."
                )
                return None
            # Re-raise if it's a different missing module (broken install)
            raise

        backend_url = BackendConfig.get_backend_url()
        api_key = BackendConfig.get_api_key()

        if (
            self.traigent_config.is_edge_analytics_mode()
            or BackendConfig.is_local_backend()
        ):
            logger.info(
                f"Configuring for {self.traigent_config.execution_mode} mode "
                f"with backend at {backend_url} (fallback enabled)"
            )
        else:
            logger.info(
                f"Configuring for {self.traigent_config.execution_mode} mode "
                f"with backend at {backend_url}"
            )

        backend_config = BackendClientConfig(
            backend_base_url=backend_url,
            enable_session_sync=True,
        )
        local_storage_path = self.traigent_config.get_local_storage_path()

        try:
            client = BackendIntegratedClient(
                api_key=api_key,
                backend_config=backend_config,
                enable_fallback=True,
                local_storage_path=local_storage_path,
            )
            logger.info(
                f"Backend client initialized for {self.traigent_config.execution_mode} mode - "
                f"session endpoints at {backend_config.backend_base_url}"
            )
            return client
        except Exception as exc:
            logger.warning(
                "Backend initialization warning. Continuing with local storage only. "
                "Results will not appear in backend UI.",
                exc_info=exc,
            )
            return BackendIntegratedClient(
                api_key=None,
                backend_config=backend_config,
                enable_fallback=True,
                local_storage_path=local_storage_path,
            )

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
        self._cloud_notice_shown = False
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
        if self.max_trials is None or self.max_trials == 0:
            return 1.0 if self.trial_count == 0 else 0.0

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
            best_trial = max(
                self._trials,
                key=lambda t: t.metrics.get(primary_objective, float("-inf")),
                default=None,
            )
            if best_trial is not None:
                self._best_trial_cached = best_trial
            return best_trial

        # If no objectives defined, return last trial
        return self._trials[-1] if self._trials else None

    def _update_best_trial_cache(self, trial_result: TrialResult) -> None:
        if not self.optimizer.objectives:
            self._best_trial_cached = trial_result
            return

        primary_objective = self.optimizer.objectives[0]
        new_score = (
            trial_result.get_metric(primary_objective, 0.0)
            if hasattr(trial_result, "get_metric")
            else ((trial_result.metrics or {}).get(primary_objective, 0.0))
        )
        new_score = new_score or 0.0

        if self._best_trial_cached is None:
            self._best_trial_cached = trial_result
            return

        current_score = (
            self._best_trial_cached.get_metric(primary_objective, 0.0)
            if hasattr(self._best_trial_cached, "get_metric")
            else (self._best_trial_cached.metrics or {}).get(primary_objective, 0.0)
        )
        current_score = current_score or 0.0

        minimization = _is_minimization_objective(primary_objective)
        if minimization:
            if new_score < current_score:
                self._best_trial_cached = trial_result
        else:
            if new_score > current_score:
                self._best_trial_cached = trial_result

    def _remaining_sample_budget(self) -> float:
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

        if self.backend_client and session_id:
            await self.backend_session_manager.submit_trial(
                trial_result=trial_result,
                session_id=session_id,
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

        self._register_examples_attempted(trial_result)

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
        """Generate configs sequentially from the optimizer."""
        configs: list[dict[str, Any]] = []
        for _ in range(count):
            try:
                configs.append(self.optimizer.suggest_next_trial(self._trials))
            except OptimizationError:
                break
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.exception(
                    "Optimizer failed to suggest trial during batch generation: %s", e
                )
                break
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
        scheduled_configs, scheduled_optuna_ids, results = (
            await self._schedule_and_run_parallel_trials(
                func, trial_descriptors, ceilings, session_id, trial_count
            )
        )

        if not results:
            return trial_count, "break"

        # Phase 8: Process results
        trial_count = await self._process_parallel_results(
            scheduled_configs, results, scheduled_optuna_ids, session_id, trial_count
        )

        # Phase 9: Checkpoint logging
        self._maybe_log_checkpoint(trial_count)

        return trial_count, "continue"

    def _submit_usage_analytics(self) -> None:
        """Submit usage analytics if enabled."""

        if not self.traigent_config.enable_usage_analytics:
            return

        try:
            collect_and_submit_analytics(self.traigent_config)
            logger.debug("Analytics submitted after optimization completion")
        except Exception as exc:
            logger.debug("Analytics submission failed: %s", exc)

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

        self._cloud_notice_shown = False
        from traigent.cloud.api_operations import reset_api_key_error_state

        reset_api_key_error_state()
        if should_show_cloud_notice(self.traigent_config):
            print(CLOUD_UNAVAILABLE_NOTICE)
            logger.info(CLOUD_UNAVAILABLE_NOTICE)
            self._cloud_notice_shown = True

        descriptor = resolve_function_descriptor(func)
        self._function_descriptor = descriptor

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

    def _validate_dataset(self, dataset: Dataset) -> None:
        """Ensure dataset is present and non-empty before optimization."""

        if dataset is None:
            raise ValueError("Dataset cannot be None") from None

        if not dataset or len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")

    async def create_session(
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

            session_id: str = self.backend_client.create_session(
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
            logger.info(f"Created session: {session_id}")
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
        """Check cost approval before optimization."""
        if self.cost_enforcer.is_mock_mode:
            return
        estimated_cost = self._estimate_optimization_cost(dataset)
        if not self.cost_enforcer.check_and_approve(estimated_cost):
            from traigent.core.cost_enforcement import OptimizationAborted

            raise OptimizationAborted(
                f"Cost approval declined. Estimated cost: ${estimated_cost:.2f}, "
                f"limit: ${self.cost_enforcer.config.limit:.2f}. "
                f"Set TRAIGENT_COST_APPROVED=true or increase TRAIGENT_RUN_COST_LIMIT."
            )

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
            if budget_stop:
                if not self._stop_reason:
                    self._stop_reason = budget_stop
                break

            if self._should_stop(trial_count):
                break

            if self.parallel_trials > 1:
                trial_count, action = await self._run_parallel_batch(
                    func=func,
                    dataset=dataset,
                    session_id=session_id,
                    function_name=function_identifier,
                    trial_count=trial_count,
                    remaining=remaining,
                    remaining_samples=remaining_samples,
                )
            else:
                trial_count, action = await self._trial_lifecycle.run_sequential_trial(
                    func=func,
                    dataset=dataset,
                    session_id=session_id,
                    function_name=function_identifier,
                    trial_count=trial_count,
                )

            if action == "break":
                break

        return trial_count

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
        await self.backend_session_manager.update_weighted_scores(result, session_id)
        self.backend_session_manager.submit_session_aggregation(result, session_id)

        session_summary = self.backend_session_manager.finalize_session(
            session_id, self._status
        )
        self.backend_session_manager.attach_session_metadata(
            result, session_id, session_summary
        )

        self._submit_usage_analytics()
        self.callback_manager.on_optimization_complete(result)

        cost_status = self.cost_enforcer.get_status()
        logger.info(
            f"Optimization {self._optimization_id} completed: "
            f"{len(self._trials)} trials, best score: {result.best_score:.4f}, "
            f"total cost: ${cost_status.accumulated_cost_usd:.4f}"
        )
        if self._cloud_notice_shown:
            print(f"\n{CLOUD_UNAVAILABLE_NOTICE}")
            logger.info(CLOUD_UNAVAILABLE_NOTICE)

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

        best_score = 0.0
        best_trial = self.best_result
        if best_trial and self.optimizer.objectives:
            primary_objective = self.optimizer.objectives[0]
            best_score = best_trial.get_metric(primary_objective, 0.0) or 0.0

        success_rate = (success_count / total_trials) if total_trials else 0.0

        elapsed = time.time() - self._start_time if self._start_time else 0.0

        logger.info(
            f"Progress: {trial_count} trials, "
            f"best score: {best_score:.4f}, "
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
        """Estimate total optimization cost for pre-approval check.

        Calculation:
        - Estimates total samples based on configuration and dataset size
        - Uses max_total_examples if configured (shared budget across trials)
        - Otherwise estimates samples_per_trial × max_trials
        - Includes retry factor (1.2x) for potential failures
        - Uses conservative estimates for unknown models

        Note: This is an ESTIMATE. Actual costs may vary significantly.
        The estimate is based on typical token usage and current pricing.

        Args:
            dataset: The evaluation dataset.

        Returns:
            Estimated cost in USD.
        """
        # Base cost per example (conservative estimate for GPT-4 class models)
        # Assumes ~2000 tokens input, ~500 tokens output per example
        base_cost_per_example = 0.01  # $0.01 per example (conservative)

        # Get dataset size
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else 100

        # Get max trials (default to 10 if not set)
        max_trials = self._max_trials or 10

        # Determine total samples based on configuration
        if self._max_total_examples is not None:
            # Global sample budget is set - use it directly for cost estimation
            # Don't clip to dataset_size because:
            # 1. With multiple trials, samples can be re-evaluated with different configs
            # 2. The budget represents total API calls, not unique samples
            # 3. Clipping would underestimate cost when budget > dataset_size
            total_samples = self._max_total_examples
            estimation_mode = "total_examples_budget"
        else:
            # No global budget - each trial may evaluate the full dataset
            # This is a worst-case conservative estimate
            samples_per_trial = dataset_size
            total_samples = max_trials * samples_per_trial
            estimation_mode = "per_trial_full_dataset"

        # Total cost with retry factor for failures and potential re-evaluations
        retry_factor = 1.2
        estimated_total = total_samples * base_cost_per_example * retry_factor

        logger.debug(
            f"Cost estimate ({estimation_mode}): {total_samples} total samples "
            f"× ${base_cost_per_example}/sample × {retry_factor} retry = ${estimated_total:.2f}"
        )

        return estimated_total

    def _extract_trial_cost(self, trial_result: TrialResult) -> float | None:
        """Extract cost from trial result for cost enforcement tracking.

        Attempts to find cost from multiple sources:
        1. trial_result.metrics["total_cost"] or ["cost"]
        2. trial_result.metadata["total_example_cost"]
        3. Returns None if cost cannot be determined (triggers fallback mode)

        Args:
            trial_result: The completed trial result.

        Returns:
            Cost in USD, or None if cost cannot be determined.
        """
        # Try metrics first
        metrics = trial_result.metrics or {}
        for key in ("total_cost", "cost"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    pass

        # Try metadata
        metadata = trial_result.metadata or {}
        if "total_example_cost" in metadata:
            try:
                return float(metadata["total_example_cost"])
            except (TypeError, ValueError):
                pass

        # Cost cannot be determined
        return None

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

        return metadata

    def _create_optimization_result(self) -> OptimizationResult:
        """Create final optimization result.

        Returns:
            OptimizationResult with all trial data
        """
        successful_trials = [t for t in self._trials if t.is_successful]
        selection = select_best_configuration(
            trials=successful_trials,
            primary_objective=self.optimizer.objectives[0],
            config_space_keys=set(getattr(self.optimizer, "config_space", {}).keys()),
            aggregate_configs=not self.traigent_config.is_edge_analytics_mode(),
            tie_breakers=self._tie_breakers or None,
            band_target=self._band_target,
        )
        best_config = selection.best_config
        best_score = selection.best_score
        session_summary = selection.session_summary

        # Calculate duration
        duration = time.time() - self._start_time if self._start_time else 0.0

        # Create convergence info
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
            timestamp=datetime.now(UTC),
            total_cost=total_cost if total_cost > 0 else None,
            total_tokens=total_tokens if total_tokens > 0 else None,
            metrics=processed_metrics,
            metadata=self._build_result_metadata(session_summary, safeguards_telemetry),
            stop_reason=self._stop_reason,
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
