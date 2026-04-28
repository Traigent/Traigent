"""Traigent Integration with execution mode support."""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Performance FUNC-CLOUD-HYBRID FUNC-INVOKERS REQ-CLOUD-009 REQ-INV-006 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from traigent.adapters.execution_adapter import LocalExecutionAdapter
from traigent.api.types import TrialResult, TrialStatus
from traigent.config.types import ExecutionMode, TraigentConfig, validate_execution_mode
from traigent.optimizers import get_optimizer
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger

# Cloud imports - required at runtime for this integration module
try:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )
    from traigent.cloud.optimizer_client import OptimizerDirectClient

    _CLOUD_AVAILABLE = True
except (
    ModuleNotFoundError
) as err:  # pragma: no cover - only runs when cloud not installed
    # Check .name to distinguish missing cloud vs broken transitive dependency
    missing_module = getattr(err, "name", "") or ""
    if missing_module == "traigent.cloud" or missing_module.startswith(
        "traigent.cloud."
    ):
        _CLOUD_AVAILABLE = False
    else:
        raise  # Re-raise for broken dependencies like missing boto3
    if TYPE_CHECKING:
        from traigent.cloud.backend_client import (
            BackendClientConfig,
            BackendIntegratedClient,
        )
        from traigent.cloud.optimizer_client import OptimizerDirectClient

logger = get_logger(__name__)
_SAAS_ACTIVE_STATUSES = frozenset(
    {"PENDING", "RUNNING", "IN_PROGRESS", "QUEUED", "STARTED"}
)
_SAAS_TERMINAL_STATUSES = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "TIMED_OUT", "STOPPED"}
)
_DEFAULT_CONFIG_VALUE_KEYS = ("value", "default", "initial", "low", "min")


def _require_cloud() -> None:
    """Raise FeatureNotAvailableError if cloud module is not available."""
    if not _CLOUD_AVAILABLE:
        from traigent.utils.exceptions import FeatureNotAvailableError

        raise FeatureNotAvailableError(
            "Traigent cloud integration",
            plugin_name="traigent-cloud",
            install_hint="pip install traigent[cloud]",
        )


class TraigentClient:
    """Enhanced Traigent client with execution mode support.

    This client automatically detects and uses the appropriate execution mode:
    - Local: Everything runs locally (no backend connection)
    - Hybrid: Local execution with metric submission to backend/portal tracking
    - Cloud: Reserved for future remote execution; not available yet
    """

    def __init__(
        self,
        api_key: str | None = None,
        backend_url: str | None = None,
        execution_mode: str = "auto",
        agent_builder=None,
    ) -> None:
        """Initialize Traigent client with execution mode.

        Args:
            api_key: API key for backend
            backend_url: Backend URL (defaults to centralized config)
            execution_mode: 'auto', 'hybrid', 'cloud', 'edge_analytics'
            agent_builder: Agent builder instance for local execution

        Note:
            Backend integration dependencies are required for 'hybrid' mode.
            Edge analytics mode works without backend/cloud dependencies installed.
        """
        from traigent.config.backend_config import BackendConfig

        # Track whether caller explicitly supplied an API key
        self._explicit_api_key = api_key is not None

        # Use provided values or get from centralized config
        self.api_key = api_key or BackendConfig.get_api_key()
        self.backend_url = backend_url or BackendConfig.get_backend_url()

        # Determine execution mode first (before initializing cloud components)
        self.execution_mode = self._determine_execution_mode(execution_mode)
        logger.info(
            f"Traigent client initialized with execution mode: {self.execution_mode}"
        )

        # Initialize backend client only if needed for non-edge_analytics modes
        # Edge analytics mode runs fully locally without any cloud dependencies
        if self.execution_mode != ExecutionMode.EDGE_ANALYTICS:
            _require_cloud()
            self.backend_client = BackendIntegratedClient(
                api_key=self.api_key,
                backend_config=BackendClientConfig(backend_base_url=self.backend_url),
            )
        else:
            self.backend_client = None  # type: ignore[assignment]

        # Initialize execution adapter
        self.execution_adapter = None
        self.optimizer_client = None
        self.agent_builder = agent_builder

    async def optimize(
        self,
        function: Callable[..., Any],
        dataset: dict[str, Any],
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 50,
        optimization_config: dict[str, Any | None] = None,
    ) -> dict[str, Any]:
        """Optimize function with automatic mode selection.

        Args:
            function: Function or agent to optimize
            dataset: Dataset for evaluation
            configuration_space: Search space for parameters
            objectives: Optimization objectives
            max_trials: Maximum number of trials
            optimization_config: Additional optimization settings

        Returns:
            Optimization results
        """
        logger.info(f"Starting optimization in {self.execution_mode} mode")

        # Normalize configuration space and derive fallback defaults
        fallback_model = (
            "gpt-3.5-turbo"
            if self.execution_mode == ExecutionMode.EDGE_ANALYTICS
            else None
        )
        configuration_space, config_defaults = self._normalise_configuration_space(
            configuration_space, fallback_model=fallback_model
        )

        if not self._has_parameter_value(configuration_space.get("model")):
            raise ValueError(
                "Optimization configuration space is missing required entries: model"
            )

        # Determine execution strategy
        if self.execution_mode == ExecutionMode.HYBRID:
            return await self._optimize_hybrid(
                function,
                dataset,
                configuration_space,
                objectives,
                max_trials,
                optimization_config or {},
                config_defaults,
            )
        elif self.execution_mode == ExecutionMode.CLOUD:
            from traigent.cloud.client import CLOUD_REMOTE_EXECUTION_UNAVAILABLE

            raise OptimizationError(
                f"{CLOUD_REMOTE_EXECUTION_UNAVAILABLE} "
                "Cloud mode will be enabled when remote agent execution is implemented."
            )
        else:
            # Edge Analytics mode
            return await self._optimize_local(
                dataset,
                configuration_space,
                objectives,
                max_trials,
                optimization_config or {},
                config_defaults,
            )

    async def _optimize_hybrid(
        self,
        function: Callable[..., Any],
        dataset: dict[str, Any],
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int,
        optimization_config: dict[str, Any | None],
        config_defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Hybrid mode: local execution with backend orchestration/tracking.

        Data never leaves the client, only metrics are submitted.
        """
        logger.info("Starting hybrid mode optimization")

        async with self.backend_client:
            # Create hybrid session
            (
                session_id,
                token,
                endpoint,
            ) = await self.backend_client.create_hybrid_session(
                problem_statement=(
                    function.__name__
                    if hasattr(function, "__name__")
                    else str(function)
                ),
                search_space=configuration_space,
                optimization_config={
                    "objectives": objectives,
                    "max_trials": max_trials,
                    **(optimization_config or {}),
                },
            )

            logger.info(f"Created hybrid session: {session_id}")

            # Initialize direct optimizer client
            async with OptimizerDirectClient(endpoint, token) as optimizer:
                # Initialize local execution adapter
                if not self.agent_builder:
                    raise ValueError("Agent builder required for hybrid mode")

                adapter = LocalExecutionAdapter(self.agent_builder)

                # Optimization loop
                completed_trials = 0
                best_result = None
                best_metric = -float("inf")

                for trial_num in range(max_trials):
                    # Get next configuration from optimizer
                    response = await optimizer.get_next_configuration(session_id)

                    if not response.get("has_next"):
                        logger.info("No more configurations to evaluate")
                        break

                    config = self._apply_config_defaults(
                        response["configuration"], config_defaults
                    )
                    trial_id = response["trial_id"]

                    logger.info(f"Executing trial {trial_num + 1}/{max_trials}")

                    # Execute locally
                    try:
                        result = await adapter.execute_configuration(
                            agent_spec=config, dataset=dataset, trial_id=trial_id
                        )

                        # Submit metrics directly to optimizer
                        await optimizer.submit_metrics(
                            session_id=session_id,
                            trial_id=trial_id,
                            metrics=result["metrics"],
                            execution_time=result["execution_time"],
                            metadata=result.get("metadata"),
                        )

                        # Track best result
                        primary_metric = result["metrics"].get(objectives[0], 0)
                        if primary_metric > best_metric:
                            best_metric = primary_metric
                            best_result = {
                                "configuration": config,
                                "metrics": result["metrics"],
                                "trial_id": trial_id,
                            }

                        completed_trials += 1

                    except Exception as e:
                        logger.error(f"Trial {trial_id} failed: {str(e)}")
                        # Continue with next trial

                # Finalize and get results
                final_results = await self.backend_client.finalize_hybrid_session(
                    session_id
                )

                # Enhance with local best result if backend doesn't have it
                if best_result and not final_results.get("best_configuration"):
                    final_results["best_configuration"] = best_result

                final_results["execution_mode"] = "hybrid"
                final_results["completed_trials"] = completed_trials

                return final_results

    async def _optimize_saas(
        self,
        function: Callable[..., Any],
        dataset: dict[str, Any],
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int,
        optimization_config: dict[str, Any | None],
        config_defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """SaaS mode: full backend execution.

        Dataset is uploaded and execution happens on backend infrastructure.
        """
        logger.info("Starting SaaS mode optimization")
        _ = config_defaults  # Unused but kept for signature symmetry

        settings = optimization_config or {}
        poll_interval = self._coerce_positive_float(
            settings.get("poll_interval"),
            default=1.0,
            fallback_for_non_positive=0.1,
        )
        max_poll_duration = self._coerce_positive_float(
            settings.get("max_poll_duration"),
            default=1800.0,
        )

        async with self.backend_client:
            upload_dataset = self._require_backend_method("upload_dataset")
            dataset_response = await upload_dataset(
                name=f"{function.__name__}_dataset", data=dataset
            )
            dataset_id = dataset_response["dataset_id"]

            create_session = self._require_backend_method("create_optimization_session")
            session_response = await create_session(
                function_name=function.__name__,
                dataset_id=dataset_id,
                configuration_space=configuration_space,
                objectives=objectives,
                max_trials=max_trials,
                optimization_config=optimization_config,
            )

            session_id = session_response["session_id"]
            logger.info(f"Created SaaS session: {session_id}")

            get_status = self._require_backend_method("get_session_status")
            await self._poll_saas_session(
                session_id,
                get_status,
                max_trials=max_trials,
                poll_interval=poll_interval,
                max_poll_duration=max_poll_duration,
            )

            get_results = self._require_backend_method("get_optimization_results")
            results = await get_results(session_id)
            results["execution_mode"] = "cloud"

            return cast(dict[str, Any], results)

    @staticmethod
    def _coerce_positive_float(
        value: Any,
        *,
        default: float,
        fallback_for_non_positive: float | None = None,
    ) -> float:
        """Parse a positive float, with separate fallbacks for invalid/non-positive."""
        try:
            parsed = float(value) if value is not None else default
        except (TypeError, ValueError):
            return default
        if parsed > 0:
            return parsed
        if fallback_for_non_positive is not None:
            return fallback_for_non_positive
        return default

    def _require_backend_method(self, name: str) -> Callable[..., Any]:
        """Return a required backend-client method or raise a consistent error."""
        method = getattr(self.backend_client, name, None)
        if method is None:
            raise OptimizationError(f"Backend client does not support {name}")
        return cast(Callable[..., Any], method)

    async def _poll_saas_session(
        self,
        session_id: str,
        get_status: Callable[..., Any],
        *,
        max_trials: int,
        poll_interval: float,
        max_poll_duration: float,
    ) -> None:
        """Poll a SaaS optimization session until it reaches a terminal state."""
        poll_start = asyncio.get_running_loop().time()
        while True:
            elapsed = asyncio.get_running_loop().time() - poll_start
            if elapsed > max_poll_duration:
                raise OptimizationError(
                    f"SaaS optimization polling timed out after {elapsed:.1f}s "
                    f"(limit: {max_poll_duration:.1f}s)"
                )

            status = await get_status(session_id)
            if self._is_saas_session_complete(status, session_id, max_trials):
                return
            await asyncio.sleep(poll_interval)

    def _is_saas_session_complete(
        self,
        status: dict[str, Any],
        session_id: str,
        max_trials: int,
    ) -> bool:
        """Log and validate SaaS session status, returning True on terminal states."""
        status_value = str(status.get("status") or "").upper()
        logger.info(
            "Session progress: %s/%s trials",
            status.get("completed_trials", 0),
            max_trials,
        )
        if status_value in _SAAS_TERMINAL_STATUSES:
            return True
        if status_value not in _SAAS_ACTIVE_STATUSES:
            raise OptimizationError(
                f"Unexpected SaaS session status '{status.get('status')}' "
                f"for session {session_id}"
            )
        return False

    async def _optimize_local(
        self,
        dataset: dict[str, Any],
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int,
        optimization_config: dict[str, Any | None],
        config_defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Edge Analytics mode: everything runs locally without backend.

        This is useful for development and testing.
        """
        logger.info("Starting Edge Analytics optimization")

        if not self.agent_builder:
            raise ValueError("Agent builder required for Edge Analytics mode") from None

        # Simple grid search for Edge Analytics mode
        adapter = LocalExecutionAdapter(self.agent_builder)

        if max_trials is not None and max_trials <= 0:
            logger.info("Edge Analytics optimization skipped due to max_trials<=0")
            return {
                "execution_mode": ExecutionMode.EDGE_ANALYTICS.value,
                "best_configuration": None,
                "all_results": [],
                "completed_trials": 0,
                "status": "no_trials",
            }

        optimizer = self._create_local_optimizer(
            configuration_space,
            objectives,
            optimization_config,
        )

        effective_max_trials = max_trials if max_trials is not None else 50
        primary_objective = objectives[0] if objectives else None

        history: list[TrialResult] = []
        results: list[dict[str, Any]] = []
        best_record: dict[str, Any] | None = None
        best_score: float | None = None

        while len(history) < effective_max_trials:
            if optimizer.should_stop(history):
                logger.info(
                    "Edge Analytics optimizer stop condition reached after %d trials",
                    len(history),
                )
                break

            try:
                suggestion = optimizer.suggest_next_trial(history)
            except OptimizationError as exc:
                logger.info("Edge Analytics optimizer exhausted search space: %s", exc)
                break

            config = self._apply_config_defaults(suggestion, config_defaults)
            trial_id = f"local_trial_{len(history)}"
            trial = await self._execute_local_trial(
                adapter,
                dataset=dataset,
                config=config,
                trial_id=trial_id,
            )
            history.append(trial)
            optimizer.update_best(trial)

            record = self._build_local_result_record(trial)
            if record is None:
                continue
            results.append(record)

            score = self._resolve_trial_score(trial, primary_objective)
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_record = record

        return {
            "execution_mode": ExecutionMode.EDGE_ANALYTICS.value,
            "best_configuration": best_record,
            "all_results": results,
            "completed_trials": len(results),
            "status": "completed",
        }

    def _create_local_optimizer(
        self,
        configuration_space: dict[str, Any],
        objectives: list[str],
        optimization_config: dict[str, Any | None],
    ) -> Any:
        """Build the local edge-analytics optimizer from user config."""
        optimizer_config = optimization_config or {}
        algorithm_name = str(optimizer_config.get("algorithm") or "grid")
        optimizer_kwargs = self._extract_optimizer_kwargs(optimizer_config)
        optimizer_context = TraigentConfig.edge_analytics_mode(
            minimal_logging=True,
            auto_sync=False,
        )
        try:
            return get_optimizer(
                algorithm_name,
                configuration_space,
                objectives,
                context=optimizer_context,
                **optimizer_kwargs,
            )
        except OptimizationError as exc:
            raise ValueError(
                f"Failed to initialize optimizer '{algorithm_name}': {exc}"
            ) from exc

    @staticmethod
    def _extract_optimizer_kwargs(
        optimization_config: dict[str, Any | None],
    ) -> dict[str, Any]:
        """Collect supported optimizer kwargs from the local optimization config."""
        optimizer_kwargs: dict[str, Any] = {}
        opt_kwargs = optimization_config.get("optimizer_kwargs")
        if isinstance(opt_kwargs, dict):
            optimizer_kwargs.update(opt_kwargs)
        if "objective_weights" in optimization_config:
            optimizer_kwargs.setdefault(
                "objective_weights", optimization_config["objective_weights"]
            )
        return optimizer_kwargs

    async def _execute_local_trial(
        self,
        adapter: LocalExecutionAdapter,
        *,
        dataset: dict[str, Any],
        config: dict[str, Any],
        trial_id: str,
    ) -> TrialResult:
        """Execute one local trial and normalize it into a TrialResult."""
        try:
            execution = await adapter.execute_configuration(
                agent_spec=config,
                dataset=dataset,
                trial_id=trial_id,
            )
        except Exception as exc:
            logger.error("Local trial %s failed: %s", trial_id, exc, exc_info=True)
            return self._build_local_trial_result(
                trial_id=trial_id,
                config=config,
                metrics={},
                duration=0.0,
                metadata={},
                status=TrialStatus.FAILED,
                error_message=str(exc),
            )

        return self._build_local_trial_result(
            trial_id=trial_id,
            config=config,
            metrics=cast(dict[str, Any], execution.get("metrics", {}) or {}),
            duration=float(execution.get("execution_time", 0.0) or 0.0),
            metadata=cast(dict[str, Any], execution.get("metadata", {}) or {}),
            status=TrialStatus.COMPLETED,
        )

    @staticmethod
    def _build_local_trial_result(
        *,
        trial_id: str,
        config: dict[str, Any],
        metrics: dict[str, Any],
        duration: float,
        metadata: dict[str, Any],
        status: TrialStatus,
        error_message: str | None = None,
    ) -> TrialResult:
        """Construct a TrialResult for a local optimization trial."""
        return TrialResult(
            trial_id=trial_id,
            config=config,
            metrics=metrics,
            status=status,
            duration=duration,
            timestamp=datetime.now(UTC),
            error_message=error_message,
            metadata=metadata,
        )

    @staticmethod
    def _build_local_result_record(trial: TrialResult) -> dict[str, Any] | None:
        """Build the public result record for a completed local trial."""
        if trial.status != TrialStatus.COMPLETED:
            return None
        return {
            "configuration": trial.config,
            "metrics": trial.metrics,
            "execution_time": trial.duration,
        }

    @staticmethod
    def _resolve_trial_score(
        trial: TrialResult,
        primary_objective: str | None,
    ) -> float | int | None:
        """Pick the best comparable score from a completed trial result."""
        if primary_objective:
            score = trial.get_metric(primary_objective)
            if score is not None:
                return score
        metrics = trial.metrics or {}
        score = metrics.get("score")
        if score is not None:
            return cast(float | int, score)
        return next(
            (value for value in metrics.values() if isinstance(value, (int, float))),
            None,
        )

    def _determine_execution_mode(self, requested_mode: str) -> ExecutionMode:
        """Determine actual execution mode based on environment.

        Args:
            requested_mode: User requested mode

        Returns:
            Actual execution mode to use

        Raises:
            ConfigurationError: If the requested mode is invalid or unsupported.
        """
        if requested_mode != "auto":
            return validate_execution_mode(requested_mode)

        # Auto-detection remains conservative; users can request hybrid explicitly
        # when they want backend/portal tracking.
        return ExecutionMode.EDGE_ANALYTICS

    def _check_privacy_requirements(self) -> bool:
        """Check if privacy requirements mandate standard mode.

        Returns:
            True if privacy mode is required
        """
        # Check for sensitive data indicators
        return bool(
            os.environ.get("TRAIGENT_PRIVATE_DATA")
            or os.environ.get("TRAIGENT_COMPLIANCE_MODE")
            or os.path.exists(".traigent-private")
        )

    def _normalise_configuration_space(
        self,
        configuration_space: dict[str, Any] | None,
        *,
        fallback_model: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fill in required configuration defaults and derive fallback values."""

        normalised = dict(configuration_space or {})

        if fallback_model and not self._has_parameter_value(normalised.get("model")):
            normalised["model"] = [fallback_model]

        def ensure(key: str, default_factory: Callable[[], Any]) -> None:
            if not self._has_parameter_value(normalised.get(key)):
                normalised[key] = default_factory()

        ensure("temperature", lambda: [0.7])
        ensure("max_tokens", lambda: [512])
        ensure("top_p", lambda: [1.0])

        defaults: dict[str, Any] = {}
        for key in ("model", "temperature", "max_tokens", "top_p"):
            value = normalised.get(key)
            if self._has_parameter_value(value):
                extracted = self._extract_default_value(value)
                if extracted is not None:
                    defaults[key] = extracted

        defaults.setdefault(
            "agent_platform", normalised.get("agent_platform", "openai")
        )
        return normalised, defaults

    @staticmethod
    def _has_parameter_value(value: Any) -> bool:
        """Return True if the configuration entry contains usable values."""

        if value is None:
            return False
        if isinstance(value, (list, tuple, set)):
            return len(value) > 0
        if isinstance(value, dict):
            return len(value) > 0
        return True

    @staticmethod
    def _extract_default_value(value: Any) -> Any | None:
        """Extract a representative scalar from a configuration definition."""

        if value is None:
            return None
        if isinstance(value, list):
            return value[0] if value else None
        if isinstance(value, tuple):
            return TraigentClient._extract_tuple_default_value(value)
        if isinstance(value, dict):
            return TraigentClient._extract_mapping_default_value(value)
        return value

    @staticmethod
    def _extract_tuple_default_value(value: tuple[Any, ...]) -> Any | None:
        """Extract a representative default from a tuple configuration."""
        if not value:
            return None
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            low, high = value
            return (low + high) / 2
        return value[0]

    @staticmethod
    def _extract_mapping_default_value(value: dict[str, Any]) -> Any | None:
        """Extract a representative default from a mapping configuration."""
        for candidate in _DEFAULT_CONFIG_VALUE_KEYS:
            if candidate in value and value[candidate] is not None:
                return value[candidate]
        return None

    @staticmethod
    def _apply_config_defaults(
        config: dict[str, Any], defaults: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge configuration with default fallback values."""

        merged: dict[str, Any] = {
            key: value for key, value in defaults.items() if value is not None
        }
        merged.update(config)
        return merged
