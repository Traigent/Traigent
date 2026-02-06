"""OptiGen Integration for Traigent SDK with execution mode support."""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Performance FUNC-CLOUD-HYBRID FUNC-INVOKERS REQ-CLOUD-009 REQ-INV-006 SYNC-CloudHybrid

import asyncio
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from traigent.adapters.execution_adapter import (
    LocalExecutionAdapter,
)
from traigent.api.types import TrialResult, TrialStatus
from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient
from traigent.cloud.optimizer_client import OptimizerDirectClient
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.optimizers import get_optimizer
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class OptiGenClient:
    """Enhanced OptiGen client with execution mode support.

    This client automatically detects and uses the appropriate execution mode:
    - Local: Everything runs locally (no backend connection)
    - Hybrid: Local execution with metric submission to backend
    - SaaS: Full backend execution
    """

    def __init__(
        self,
        api_key: str | None = None,
        backend_url: str | None = None,
        execution_mode: str = "auto",
        agent_builder=None,
    ) -> None:
        """Initialize OptiGen client with execution mode.

        Args:
            api_key: API key for backend
            backend_url: Backend URL (defaults to centralized config)
            execution_mode: 'auto' or 'edge_analytics' (only edge_analytics supported)
            agent_builder: Agent builder instance for local execution
        """
        from traigent.config.backend_config import BackendConfig

        # Track whether caller explicitly supplied an API key
        self._explicit_api_key = api_key is not None

        # Use provided values or get from centralized config
        self.api_key = api_key or BackendConfig.get_api_key()
        self.backend_url = backend_url or BackendConfig.get_backend_url()

        # Initialize backend client
        self.backend_client = BackendIntegratedClient(
            api_key=self.api_key,
            backend_config=BackendClientConfig(backend_base_url=self.backend_url),
        )

        # Determine execution mode
        self.execution_mode = self._determine_execution_mode(execution_mode)
        logger.info(
            f"OptiGen client initialized with execution mode: {self.execution_mode}"
        )

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
        if self.execution_mode in {ExecutionMode.HYBRID, ExecutionMode.CLOUD}:
            # Cloud and Hybrid modes not yet supported
            from traigent.utils.exceptions import ConfigurationError

            raise ConfigurationError(
                f"execution_mode='{self.execution_mode.value}' is not yet supported. "
                "Use execution_mode='edge_analytics' for local optimization."
            )
        else:
            # Edge Analytics mode (only supported mode)
            return await self._optimize_local(
                function,
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
        """Standard mode: local execution with backend orchestration.

        Data never leaves the client, only metrics are submitted.
        """
        logger.info("Starting standard mode optimization")

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
                    raise ValueError("Agent builder required for standard mode")

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

                final_results["execution_mode"] = "standard"
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

        poll_interval = float((optimization_config or {}).get("poll_interval") or 1.0)
        if poll_interval <= 0:
            poll_interval = 0.1

        async with self.backend_client:
            # Upload dataset (using dynamic attribute access for optional methods)
            upload_dataset = getattr(self.backend_client, "upload_dataset", None)
            if upload_dataset is None:
                raise OptimizationError(
                    "Backend client does not support upload_dataset"
                )
            dataset_response = await upload_dataset(
                name=f"{function.__name__}_dataset", data=dataset
            )
            dataset_id = dataset_response["dataset_id"]

            # Create optimization session
            create_session = getattr(
                self.backend_client, "create_optimization_session", None
            )
            if create_session is None:
                raise OptimizationError(
                    "Backend client does not support create_optimization_session"
                )
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

            # Wait for completion (with progress updates)
            get_status = getattr(self.backend_client, "get_session_status", None)
            if get_status is None:
                raise OptimizationError(
                    "Backend client does not support get_session_status"
                )
            while True:
                status = await get_status(session_id)

                logger.info(
                    f"Session progress: {status.get('completed_trials', 0)}/{max_trials} trials"
                )

                if status["status"] in ["COMPLETED", "FAILED"]:
                    break

                await asyncio.sleep(poll_interval)

            # Get results
            get_results = getattr(self.backend_client, "get_optimization_results", None)
            if get_results is None:
                raise OptimizationError(
                    "Backend client does not support get_optimization_results"
                )
            results = await get_results(session_id)
            results["execution_mode"] = "cloud"

            return cast(dict[str, Any], results)

    async def _optimize_local(
        self,
        function: Callable[..., Any],
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

        optimizer_config = optimization_config or {}
        algorithm_name = str(optimizer_config.get("algorithm") or "grid")

        optimizer_kwargs: dict[str, Any] = {}
        opt_kwargs = optimizer_config.get("optimizer_kwargs")
        if isinstance(opt_kwargs, dict):
            optimizer_kwargs.update(opt_kwargs)
        if "objective_weights" in optimizer_config:
            optimizer_kwargs.setdefault(
                "objective_weights", optimizer_config["objective_weights"]
            )

        optimizer_context = TraigentConfig.edge_analytics_mode(
            minimal_logging=True,
            auto_sync=False,
        )

        try:
            optimizer = get_optimizer(
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

            try:
                execution = await adapter.execute_configuration(
                    agent_spec=config, dataset=dataset, trial_id=trial_id
                )
                metrics = execution.get("metrics", {}) or {}
                duration = float(execution.get("execution_time", 0.0) or 0.0)
                metadata = execution.get("metadata", {}) or {}
                status = TrialStatus.COMPLETED
                error_message = None
            except Exception as exc:
                logger.error("Local trial %s failed: %s", trial_id, exc, exc_info=True)
                metrics = {}
                duration = 0.0
                metadata = {}
                status = TrialStatus.FAILED
                error_message = str(exc)

            trial = TrialResult(
                trial_id=trial_id,
                config=config,
                metrics=metrics,
                status=status,
                duration=duration,
                timestamp=datetime.now(UTC),
                error_message=error_message,
                metadata=metadata,
            )
            history.append(trial)
            optimizer.update_best(trial)

            if status == TrialStatus.COMPLETED:
                record = {
                    "configuration": config,
                    "metrics": metrics,
                    "execution_time": duration,
                }
                results.append(record)

                score = None
                if primary_objective:
                    score = trial.get_metric(primary_objective)
                if score is None:
                    score = metrics.get("score")
                if score is None and metrics:
                    score = next(
                        (
                            value
                            for value in metrics.values()
                            if isinstance(value, (int, float))
                        ),
                        None,
                    )
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

    def _determine_execution_mode(self, requested_mode: str) -> ExecutionMode:
        """Determine actual execution mode based on environment.

        Args:
            requested_mode: User requested mode

        Returns:
            Actual execution mode to use (only edge_analytics is currently supported)

        Raises:
            ConfigurationError: If mode is cloud/hybrid (not yet supported)
                               or privacy/standard (removed)
        """
        from traigent.config.types import resolve_execution_mode

        if requested_mode != "auto":
            # Use resolve_execution_mode which validates and raises
            # ConfigurationError for unsupported modes
            return resolve_execution_mode(requested_mode)

        # Auto-detection: always default to edge_analytics
        # since cloud and hybrid are not yet supported in the open-source SDK
        # Environment variables TRAIGENT_FORCE_LOCAL/HYBRID/CLOUD are ignored
        # for auto mode - only edge_analytics is available
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
            or os.path.exists(".optigen-private")
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
            if len(value) == 0:
                return None
            if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                low, high = value
                return (low + high) / 2
            return value[0]
        if isinstance(value, dict):
            for candidate in ("value", "default", "initial", "low", "min"):
                if candidate in value and value[candidate] is not None:
                    return value[candidate]
            return None
        return value

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
