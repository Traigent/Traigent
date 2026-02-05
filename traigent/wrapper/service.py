"""TraigentService - Wrapper for external agentic services.

Provides a minimal framework for exposing agentic services via the
Traigent hybrid API protocol. Services can use decorators to define
their configuration space, execute logic, and evaluation logic.

Example:
    from traigent.wrapper import TraigentService

    app = TraigentService(capability_id="my_agent")

    @app.tvars
    def config_space():
        return {
            "model": {"type": "enum", "values": ["gpt-4o", "claude-3"]},
            "temperature": {"type": "float", "range": [0.0, 2.0]},
        }

    @app.execute
    async def run_agent(input_id: str, data: dict, config: dict) -> dict:
        # Your agent logic here
        return {"output": result, "cost_usd": 0.002, "latency_ms": 150}

    @app.evaluate
    async def score(output: dict, target: dict, config: dict) -> dict:
        return {"accuracy": 0.95, "safety": 1.0}

    if __name__ == "__main__":
        app.run(port=8080)
"""

# Traceability: HYBRID-MODE-OPTIMIZATION CLIENT-WRAPPER-SDK

from __future__ import annotations

import inspect
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ServiceConfig:
    """Configuration for TraigentService.

    Attributes:
        capability_id: Unique identifier for this capability.
        version: API version string.
        supports_keep_alive: Whether to enable keep-alive support.
        supports_streaming: Whether to enable streaming support.
        max_batch_size: Maximum inputs per execute request.
        schema_version: TVL schema version.
    """

    capability_id: str = "default"
    version: str = "1.0"
    supports_keep_alive: bool = False
    supports_streaming: bool = False
    max_batch_size: int = 100
    schema_version: str = "0.9"


@dataclass
class Session:
    """Active session state.

    Attributes:
        session_id: Unique session identifier.
        created_at: Timestamp when session was created.
        last_activity: Timestamp of last activity.
        state: Custom state stored by the service.
    """

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    state: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()


class TraigentService:
    """Wrapper framework for Traigent-compatible agentic services.

    Provides decorators for defining configuration space, execution
    logic, and evaluation logic. Can serve via HTTP or be used
    directly as a library.

    Example:
        app = TraigentService(capability_id="qa_agent")

        @app.tvars
        def config_space():
            return {"model": {"type": "enum", "values": ["gpt-4"]}}

        @app.execute
        async def run(input_id, data, config):
            return {"output": "result"}

        app.run(port=8080)
    """

    def __init__(
        self,
        capability_id: str = "default",
        version: str = "1.0",
        supports_keep_alive: bool = False,
        supports_streaming: bool = False,
        max_batch_size: int = 100,
    ) -> None:
        """Initialize TraigentService.

        Args:
            capability_id: Unique identifier for this capability.
            version: API version string.
            supports_keep_alive: Whether to enable keep-alive support.
            supports_streaming: Whether to enable streaming support.
            max_batch_size: Maximum inputs per execute request.
        """
        self.config = ServiceConfig(
            capability_id=capability_id,
            version=version,
            supports_keep_alive=supports_keep_alive,
            supports_streaming=supports_streaming,
            max_batch_size=max_batch_size,
        )

        # Registered handlers
        self._tvars_handler: Callable[[], dict[str, Any]] | None = None
        self._execute_handler: Callable[..., Any] | None = None
        self._evaluate_handler: Callable[..., Any] | None = None

        # Session management
        self._sessions: dict[str, Session] = {}

        # Cached config space
        self._cached_tvars: dict[str, Any] | None = None

    def tvars(self, func: F) -> F:
        """Decorator to register TVAR configuration function.

        The decorated function should return a dictionary of TVAR
        definitions following TVL 0.9 specification.

        Example:
            @app.tvars
            def config_space():
                return {
                    "model": {"type": "enum", "values": ["gpt-4"]},
                    "temperature": {"type": "float", "range": [0.0, 2.0]},
                }
        """
        self._tvars_handler = func
        self._cached_tvars = None  # Invalidate cache
        return func

    def tunables(self, func: F) -> F:
        """Decorator to register tunable configuration function.

        Client-friendly alias for tvars(). The decorated function should
        return a dictionary of tunable definitions.

        Example:
            @app.tunables
            def config_space():
                return {
                    "model": {"type": "enum", "values": ["gpt-4", "claude-3"]},
                    "temperature": {"type": "float", "range": [0.0, 1.0]},
                }
        """
        return self.tvars(func)

    def execute(self, func: F) -> F:
        """Decorator to register execution handler.

        The decorated function receives:
            - input_id: str - Identifier for this input
            - data: dict - Input data
            - config: dict - Configuration parameters

        Should return a dict with:
            - output: Any - The result
            - cost_usd: float (optional) - Cost for this execution
            - latency_ms: float (optional) - Processing latency
            - metrics: dict (optional) - Quality metrics (combined mode)

        Example:
            @app.execute
            async def run_agent(input_id: str, data: dict, config: dict) -> dict:
                result = await my_llm_call(data["query"], **config)
                return {"output": result, "cost_usd": 0.002}
        """
        self._execute_handler = func
        return func

    def evaluate(self, func: F) -> F:
        """Decorator to register evaluation handler.

        The decorated function receives:
            - output: Any - The agent output
            - target: Any - The expected output
            - config: dict - Configuration parameters

        Should return a dict of metric names to values.

        Example:
            @app.evaluate
            async def score(output: dict, target: dict, config: dict) -> dict:
                accuracy = compute_accuracy(output, target)
                return {"accuracy": accuracy, "safety": 1.0}
        """
        self._evaluate_handler = func
        return func

    def get_config_space(self) -> dict[str, Any]:
        """Get TVAR definitions.

        Returns:
            Dictionary with schema_version, capability_id, and tvars.
        """
        if self._cached_tvars is None and self._tvars_handler is not None:
            tvars_dict = self._tvars_handler()
            # Convert to TVL format if needed
            self._cached_tvars = self._normalize_tvars(tvars_dict)

        tvars = self._cached_tvars or {}

        return {
            "schema_version": self.config.schema_version,
            "capability_id": self.config.capability_id,
            "tvars": [{"name": name, **spec} for name, spec in tvars.items()],
            "constraints": {},
        }

    def _normalize_tvars(self, tvars: dict[str, Any]) -> dict[str, Any]:
        """Normalize TVAR definitions to TVL format."""
        normalized = {}
        for name, spec in tvars.items():
            if isinstance(spec, dict):
                normalized[name] = spec
            elif isinstance(spec, list):
                # List of values -> enum type
                normalized[name] = {"type": "enum", "domain": {"values": spec}}
            else:
                # Scalar default value
                normalized[name] = {"type": "str", "default": spec}
        return normalized

    def get_capabilities(self) -> dict[str, Any]:
        """Get service capabilities.

        Returns:
            Dictionary with version and feature flags.
        """
        return {
            "version": self.config.version,
            "supports_evaluate": self._evaluate_handler is not None,
            "supports_keep_alive": self.config.supports_keep_alive,
            "supports_streaming": self.config.supports_streaming,
            "max_batch_size": self.config.max_batch_size,
        }

    def get_health(self) -> dict[str, Any]:
        """Get service health status.

        Returns:
            Dictionary with status and details.
        """
        return {
            "status": "healthy",
            "version": self.config.version,
            "capability_id": self.config.capability_id,
            "active_sessions": len(self._sessions),
        }

    async def handle_execute(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle execute request.

        Args:
            request: Execute request with capability_id, config, inputs.

        Returns:
            Execute response with outputs and metrics.

        Raises:
            ValueError: If no execute handler registered.
        """
        if self._execute_handler is None:
            raise ValueError("No execute handler registered")

        request_id = request.get("request_id", str(uuid.uuid4()))
        # capability_id from request (default to self.config.capability_id)
        _ = request.get("capability_id", self.config.capability_id)
        config = request.get("config", {})
        inputs = request.get("inputs", [])
        session_id = request.get("session_id")

        # Update session if provided
        if session_id and session_id in self._sessions:
            self._sessions[session_id].touch()

        start_time = time.time()
        outputs: list[dict[str, Any]] = []
        total_cost = 0.0
        all_quality_metrics: list[dict[str, float]] = []

        for inp in inputs:
            input_id = inp.get("input_id", str(uuid.uuid4()))
            data = inp.get("data", inp)

            try:
                # Call handler
                result = self._execute_handler(input_id, data, config)
                if inspect.isawaitable(result):
                    result = await result

                # Extract output and metrics
                if isinstance(result, dict):
                    output = result.get("output", result)
                    cost = result.get("cost_usd", 0.0)
                    metrics = result.get("metrics", {})
                else:
                    output = result
                    cost = 0.0
                    metrics = {}

                total_cost += cost
                if metrics:
                    all_quality_metrics.append(metrics)

                outputs.append(
                    {
                        "input_id": input_id,
                        "output": output,
                        "cost_usd": cost,
                        "metrics": metrics,
                    }
                )

            except Exception as e:
                logger.error(f"Execute failed for {input_id}: {e}")
                outputs.append(
                    {
                        "input_id": input_id,
                        "error": str(e),
                    }
                )

        elapsed_ms = (time.time() - start_time) * 1000

        response: dict[str, Any] = {
            "request_id": request_id,
            "execution_id": str(uuid.uuid4()),
            "status": "completed",
            "outputs": outputs,
            "operational_metrics": {
                "total_cost_usd": total_cost,
                "cost_usd": total_cost,
                "latency_ms": elapsed_ms,
            },
        }

        # Include quality metrics if available (combined mode)
        if all_quality_metrics:
            response["quality_metrics"] = self._aggregate_metrics(all_quality_metrics)

        if session_id:
            response["session_id"] = session_id

        return response

    async def handle_evaluate(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle evaluate request.

        Args:
            request: Evaluate request with evaluations.

        Returns:
            Evaluate response with results and aggregate metrics.

        Raises:
            ValueError: If no evaluate handler registered.
        """
        if self._evaluate_handler is None:
            raise ValueError("No evaluate handler registered")

        request_id = request.get("request_id", str(uuid.uuid4()))
        evaluations = request.get("evaluations", [])
        config = request.get("config", {})
        session_id = request.get("session_id")

        # Update session if provided
        if session_id and session_id in self._sessions:
            self._sessions[session_id].touch()

        results: list[dict[str, Any]] = []
        all_metrics: list[dict[str, float]] = []

        for evaluation in evaluations:
            input_id = evaluation.get("input_id", str(uuid.uuid4()))
            output = evaluation.get("output")
            target = evaluation.get("target")

            try:
                # Call handler
                result = self._evaluate_handler(output, target, config)
                if inspect.isawaitable(result):
                    result = await result

                metrics = result if isinstance(result, dict) else {"score": result}
                all_metrics.append(metrics)

                results.append(
                    {
                        "input_id": input_id,
                        "metrics": metrics,
                    }
                )

            except Exception as e:
                logger.error(f"Evaluate failed for {input_id}: {e}")
                results.append(
                    {
                        "input_id": input_id,
                        "error": str(e),
                    }
                )

        response: dict[str, Any] = {
            "request_id": request_id,
            "status": "completed",
            "results": results,
            "aggregate_metrics": self._compute_aggregate_metrics(all_metrics),
        }

        return response

    def _aggregate_metrics(
        self, metrics_list: list[dict[str, float]]
    ) -> dict[str, float]:
        """Aggregate per-example metrics to single values."""
        if not metrics_list:
            return {}

        aggregated: dict[str, float] = {}
        counts: dict[str, int] = {}

        for metrics in metrics_list:
            for name, value in metrics.items():
                if name not in aggregated:
                    aggregated[name] = 0.0
                    counts[name] = 0
                aggregated[name] += value
                counts[name] += 1

        return {
            name: total / counts[name]
            for name, total in aggregated.items()
            if counts[name] > 0
        }

    def _compute_aggregate_metrics(
        self, metrics_list: list[dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Compute aggregate statistics (mean, std, n) for each metric."""
        if not metrics_list:
            return {}

        aggregates: dict[str, dict[str, float]] = {}

        # Collect values by metric name
        values_by_metric: dict[str, list[float]] = {}
        for metrics in metrics_list:
            for name, value in metrics.items():
                if name not in values_by_metric:
                    values_by_metric[name] = []
                values_by_metric[name].append(value)

        # Compute stats
        import statistics

        for name, values in values_by_metric.items():
            n = len(values)
            mean = sum(values) / n
            std = statistics.stdev(values) if n > 1 else 0.0
            aggregates[name] = {"mean": mean, "std": std, "n": float(n)}

        return aggregates

    def handle_keep_alive(self, session_id: str) -> bool:
        """Handle keep-alive request.

        Args:
            session_id: Session to keep alive.

        Returns:
            True if session is alive, False if not found.
        """
        if session_id not in self._sessions:
            return False

        self._sessions[session_id].touch()
        return True

    def create_session(self) -> str:
        """Create a new session.

        Returns:
            New session ID.
        """
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(session_id=session_id)
        return session_id

    def run(
        self,
        host: str = "0.0.0.0",  # nosec B104 - intentional for server binding
        port: int = 8080,
        server: Literal["uvicorn", "hypercorn"] = "uvicorn",
    ) -> None:
        """Start HTTP server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
            server: ASGI server to use (uvicorn or hypercorn).
        """
        from traigent.wrapper.server import create_app, run_server

        app = create_app(self)
        run_server(app, host=host, port=port, server=server)
