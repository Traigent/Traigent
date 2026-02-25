"""TraigentService - Wrapper for external agentic services.

Provides a minimal framework for exposing agentic services via the
Traigent hybrid API protocol. Services can use decorators to define
their configuration space, execute logic, and evaluation logic.

Example:
    from traigent.wrapper import TraigentService

    app = TraigentService(tunable_id="my_agent")

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

import asyncio
import copy
import inspect
import json
import math
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from traigent.utils.logging import get_logger
from traigent.wrapper.errors import HybridAPIError

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# OpenAPI contract: tunable and objective names must be valid Python identifiers.
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_]\w*$")


@dataclass
class ServiceConfig:
    """Configuration for TraigentService.

    Attributes:
        tunable_id: Unique identifier for this tunable.
        version: API version string.
        supports_keep_alive: Whether to enable keep-alive support.
        supports_streaming: Whether to enable streaming support.
        max_batch_size: Maximum inputs per execute request.
        schema_version: TVL schema version.
        constraints: Optional constraints (legacy textual or typed TVL 0.9 format).
        objectives: Optional objective definitions.
        exploration: Optional exploration configuration.
        promotion_policy: Optional promotion policy definition.
        defaults: Optional default configuration values.
        measures: Optional declared measure names.
    """

    tunable_id: str = "default"
    version: str = "1.0"
    supports_keep_alive: bool = False
    supports_streaming: bool = False
    max_batch_size: int = 100
    schema_version: str = "0.9"
    constraints: dict[str, Any] | list[Any] | None = None
    objectives: list[dict[str, Any]] | None = None
    exploration: dict[str, Any] | None = None
    promotion_policy: dict[str, Any] | None = None
    defaults: dict[str, Any] | None = None
    measures: list[str] | None = None


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
        app = TraigentService(tunable_id="qa_agent")

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
        tunable_id: str = "default",
        version: str = "1.0",
        supports_keep_alive: bool = False,
        supports_streaming: bool = False,
        max_batch_size: int = 100,
        constraints: dict[str, Any] | list[Any] | None = None,
        objectives: list[dict[str, Any]] | None = None,
        exploration: dict[str, Any] | None = None,
        promotion_policy: dict[str, Any] | None = None,
        defaults: dict[str, Any] | None = None,
        measures: list[str] | None = None,
    ) -> None:
        """Initialize TraigentService.

        Args:
            tunable_id: Unique identifier for this tunable.
            version: API version string.
            supports_keep_alive: Whether to enable keep-alive support.
            supports_streaming: Whether to enable streaming support.
            max_batch_size: Maximum inputs per execute request.
            constraints: Optional constraints config.
            objectives: Optional objective definitions.
            exploration: Optional exploration configuration.
            promotion_policy: Optional promotion policy.
            defaults: Optional default tunable values.
            measures: Optional declared measure names.
        """
        self.config = ServiceConfig(
            tunable_id=tunable_id,
            version=version,
            supports_keep_alive=supports_keep_alive,
            supports_streaming=supports_streaming,
            max_batch_size=max_batch_size,
            constraints=constraints,
            objectives=objectives,
            exploration=exploration,
            promotion_policy=promotion_policy,
            defaults=defaults,
            measures=measures,
        )

        # Registered handlers
        self._tvars_handler: Callable[[], dict[str, Any]] | None = None
        self._execute_handler: Callable[..., Any] | None = None
        self._evaluate_handler: Callable[..., Any] | None = None
        self._constraints_handler: Callable[[], dict[str, Any] | list[Any]] | None = (
            None
        )
        self._objectives_handler: Callable[[], list[dict[str, Any]]] | None = None
        self._exploration_handler: Callable[[], dict[str, Any]] | None = None
        self._promotion_policy_handler: Callable[[], dict[str, Any]] | None = None
        self._defaults_handler: Callable[[], dict[str, Any]] | None = None
        self._measures_handler: Callable[[], list[str]] | None = None

        # Session management
        self._sessions: dict[str, Session] = {}
        self._started_at: float = time.time()

        # In-process idempotency caches keyed by request_id.
        # Value is (payload fingerprint excluding request_id, response payload).
        # Bounded to prevent unbounded memory growth in long-lived servers.
        self._idempotency_cache_max_size = 1000
        self._execute_idempotency_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._evaluate_idempotency_cache: dict[str, tuple[str, dict[str, Any]]] = {}

        # Cached config space
        self._cached_tvars: dict[str, Any] | None = None

    def _fingerprint_request(self, request: dict[str, Any]) -> str:
        """Build stable request fingerprint for idempotency checks.

        Excludes request_id because request_id is used as the dedupe key itself.
        """
        payload = {k: v for k, v in request.items() if k != "request_id"}
        try:
            return json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        except TypeError:
            # Fallback for non-JSON-native values in custom wrappers.
            return repr(payload)

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

    def constraints(self, func: F) -> F:
        """Decorator to register constraints declaration function."""
        self._constraints_handler = func  # type: ignore[assignment]
        return func

    def objectives(self, func: F) -> F:
        """Decorator to register objectives declaration function."""
        self._objectives_handler = func  # type: ignore[assignment]
        return func

    def exploration(self, func: F) -> F:
        """Decorator to register exploration declaration function."""
        self._exploration_handler = func  # type: ignore[assignment]
        return func

    def promotion_policy(self, func: F) -> F:
        """Decorator to register promotion policy declaration function."""
        self._promotion_policy_handler = func  # type: ignore[assignment]
        return func

    def defaults(self, func: F) -> F:
        """Decorator to register default-config declaration function."""
        self._defaults_handler = func  # type: ignore[assignment]
        return func

    def measures(self, func: F) -> F:
        """Decorator to register declared measure names function."""
        self._measures_handler = func  # type: ignore[assignment]
        return func

    def _resolve_declared_section(
        self,
        handler: Callable[[], Any] | None,
        configured: Any,
    ) -> Any:
        """Resolve section value from decorator handler or constructor config."""
        if handler is None:
            return configured
        if inspect.iscoroutinefunction(handler):
            raise ValueError("declaration handlers must be synchronous functions")
        value = handler()
        if inspect.isawaitable(value):
            # Defensive: close accidental coroutine objects to avoid runtime warnings.
            close = getattr(value, "close", None)
            if callable(close):
                close()
            raise ValueError("declaration handlers must be synchronous functions")
        return value

    def get_config_space(self) -> dict[str, Any]:
        """Get TVAR definitions.

        Returns:
            Dictionary with schema_version, tunable_id, and tunables.
        """
        if self._cached_tvars is None and self._tvars_handler is not None:
            tvars_dict = self._tvars_handler()
            if not isinstance(tvars_dict, dict):
                raise ValueError("tunables handler must return a dict")
            # Convert to TVL format if needed
            self._cached_tvars = self._normalize_tvars(tvars_dict)

        tvars = self._cached_tvars or {}
        tunables = [{"name": name, **spec} for name, spec in tvars.items()]
        constraints = self._resolve_declared_section(
            self._constraints_handler,
            self.config.constraints,
        )
        objectives = self._resolve_declared_section(
            self._objectives_handler,
            self.config.objectives,
        )
        exploration = self._resolve_declared_section(
            self._exploration_handler,
            self.config.exploration,
        )
        promotion_policy = self._resolve_declared_section(
            self._promotion_policy_handler,
            self.config.promotion_policy,
        )
        defaults = self._resolve_declared_section(
            self._defaults_handler,
            self.config.defaults,
        )
        measures = self._resolve_declared_section(
            self._measures_handler,
            self.config.measures,
        )

        if constraints is not None and not isinstance(constraints, (dict, list)):
            raise ValueError("constraints must be a dict or list")
        if objectives is not None and not isinstance(objectives, list):
            raise ValueError("objectives must be a list")
        if exploration is not None and not isinstance(exploration, dict):
            raise ValueError("exploration must be a dict")
        if promotion_policy is not None and not isinstance(promotion_policy, dict):
            raise ValueError("promotion_policy must be a dict")
        if defaults is not None and not isinstance(defaults, dict):
            raise ValueError("defaults must be a dict")
        if measures is not None and (
            not isinstance(measures, list)
            or not all(isinstance(measure, str) for measure in measures)
        ):
            raise ValueError("measures must be a list of strings")

        response: dict[str, Any] = {
            "schema_version": self.config.schema_version,
            "tunable_id": self.config.tunable_id,
            "tunables": tunables,
            "tvars": tunables,  # Backward compatibility alias
            "constraints": constraints or {},
        }
        if objectives is not None:
            response["objectives"] = objectives
        if exploration is not None:
            response["exploration"] = exploration
        if promotion_policy is not None:
            response["promotion_policy"] = promotion_policy
        if defaults is not None:
            response["defaults"] = defaults
        if measures is not None:
            response["measures"] = measures
        return response

    def _normalize_tvars(self, tvars: dict[str, Any]) -> dict[str, Any]:
        """Normalize TVAR definitions to TVL format."""
        normalized: dict[str, dict[str, Any]] = {}
        for name, spec in tvars.items():
            if not _IDENTIFIER_RE.match(name):
                raise ValueError(
                    f"Invalid TVAR name '{name}': must be a valid Python identifier "
                    f"(pattern: ^[a-zA-Z_][a-zA-Z0-9_]*$)"
                )
            if isinstance(spec, dict):
                normalized_spec = dict(spec)
                domain = normalized_spec.get("domain", {})
                if not isinstance(domain, dict):
                    domain = {}

                # Accept wrapper-friendly top-level values/range/resolution and
                # normalize to contract-compliant nested domain shape.
                if "values" in normalized_spec and "values" not in domain:
                    domain["values"] = normalized_spec.pop("values")
                if "range" in normalized_spec and "range" not in domain:
                    domain["range"] = normalized_spec.pop("range")
                if "resolution" in normalized_spec and "resolution" not in domain:
                    domain["resolution"] = normalized_spec.pop("resolution")

                if domain is not None:
                    normalized_spec["domain"] = domain

                normalized[name] = normalized_spec
            elif isinstance(spec, list):
                # List of values -> enum type
                normalized[name] = {"type": "enum", "domain": {"values": spec}}
            else:
                # Scalar default value
                normalized[name] = {
                    "type": "str",
                    "domain": {"values": [spec]},
                    "default": spec,
                }
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
            "max_payload_bytes": None,
            # Explicitly advertise supported capability IDs for multi-capability clients.
            "tunable_ids": [self.config.tunable_id],
        }

    def get_health(self) -> dict[str, Any]:
        """Get service health status.

        Returns:
            Dictionary with status and details.
        """
        uptime_seconds = max(0.0, time.time() - self._started_at)
        return {
            "status": "healthy",
            "version": self.config.version,
            "uptime_seconds": uptime_seconds,
            "details": {
                "tunable_id": self.config.tunable_id,
                "active_sessions": len(self._sessions),
            },
        }

    async def handle_execute(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle execute request.

        Args:
            request: Execute request with tunable_id, config, inputs.

        Returns:
            Execute response with outputs and metrics.

        Raises:
            ValueError: If no execute handler registered.
        """
        if self._execute_handler is None:
            raise ValueError("No execute handler registered")

        request_id = str(request.get("request_id", str(uuid.uuid4())))
        request_fingerprint = self._fingerprint_request(request)
        cached_execute = self._execute_idempotency_cache.get(request_id)
        if cached_execute is not None:
            cached_fingerprint, cached_response = cached_execute
            if cached_fingerprint != request_fingerprint:
                raise ValueError(
                    "request_id reuse with different payload in execute request"
                )
            return copy.deepcopy(cached_response)

        tunable_id = request.get("tunable_id", self.config.tunable_id)
        config = request.get("config", {})
        inputs = request.get("inputs")
        session_id = request.get("session_id")

        if not isinstance(inputs, list) or len(inputs) == 0:
            raise ValueError("inputs must be a non-empty list")

        # Validate tunable_id matches this service
        if tunable_id != self.config.tunable_id:
            raise ValueError(
                f"tunable_id mismatch: request has '{tunable_id}', "
                f"service is '{self.config.tunable_id}'"
            )

        # Enforce max_batch_size from capabilities
        max_batch = self.config.max_batch_size
        if max_batch and len(inputs) > max_batch:
            raise ValueError(
                f"Batch size {len(inputs)} exceeds max_batch_size {max_batch}"
            )

        # Update session if provided
        if session_id:
            self._touch_or_create_session(session_id)

        start_time = time.time()
        outputs: list[dict[str, Any]] = []
        total_cost = 0.0
        all_quality_metrics: list[dict[str, float]] = []
        failed_input_ids: list[str] = []

        for inp in inputs:
            if isinstance(inp, dict):
                input_id = str(inp.get("input_id", str(uuid.uuid4())))
                data = inp.get("data", inp)
            else:
                input_id = str(uuid.uuid4())
                data = inp

            try:
                # Call handler
                result = self._execute_handler(input_id, data, config)
                if inspect.isawaitable(result):
                    result = await result

                # Extract output and metrics
                if isinstance(result, dict):
                    output = result.get("output", result)
                    cost = float(result.get("cost_usd", 0.0) or 0.0)
                    latency_ms = result.get("latency_ms")
                    metrics = (
                        result.get("metrics")
                        if isinstance(result.get("metrics"), dict)
                        else {}
                    )
                else:
                    output = result
                    cost = 0.0
                    latency_ms = None
                    metrics = {}

                total_cost += cost
                if metrics:
                    all_quality_metrics.append(metrics)

                output_item: dict[str, Any] = {
                    "input_id": input_id,
                    "output": output,
                    "cost_usd": cost,
                }
                if isinstance(latency_ms, (int, float)):
                    output_item["latency_ms"] = float(latency_ms)
                if metrics:
                    output_item["metrics"] = metrics

                outputs.append(output_item)

            except HybridAPIError:
                # Bubble up explicit transport-level errors (401/429/503/etc.)
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Execute failed for {input_id}: {e}")
                failed_input_ids.append(input_id)
                outputs.append(
                    {
                        "input_id": input_id,
                        "error": str(e),
                    }
                )

        elapsed_ms = (time.time() - start_time) * 1000
        successful_outputs = len(outputs) - len(failed_input_ids)
        if successful_outputs == 0 and outputs:
            status = "failed"
        elif failed_input_ids:
            status = "partial"
        else:
            status = "completed"

        response: dict[str, Any] = {
            "request_id": request_id,
            "execution_id": str(uuid.uuid4()),
            "status": status,
            "outputs": outputs,
            "operational_metrics": {
                "total_cost_usd": total_cost,
                "cost_usd": total_cost,
                "latency_ms": elapsed_ms,
            },
        }

        if failed_input_ids:
            response["error"] = {
                "code": (
                    "EXECUTION_PARTIAL_FAILURE"
                    if status == "partial"
                    else "EXECUTION_FAILED"
                ),
                "message": "One or more inputs failed during execution",
                "failed_inputs": failed_input_ids,
            }

        # Include quality metrics if available (combined mode)
        if all_quality_metrics:
            response["quality_metrics"] = self._aggregate_metrics(all_quality_metrics)

        if session_id:
            response["session_id"] = session_id

        # Cache terminal response for idempotent retries.
        # Evict oldest entry if cache is full.
        if len(self._execute_idempotency_cache) >= self._idempotency_cache_max_size:
            oldest_key = next(iter(self._execute_idempotency_cache))
            del self._execute_idempotency_cache[oldest_key]
        self._execute_idempotency_cache[request_id] = (
            request_fingerprint,
            copy.deepcopy(response),
        )
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

        request_id = str(request.get("request_id", str(uuid.uuid4())))
        request_fingerprint = self._fingerprint_request(request)
        cached_evaluate = self._evaluate_idempotency_cache.get(request_id)
        if cached_evaluate is not None:
            cached_fingerprint, cached_response = cached_evaluate
            if cached_fingerprint != request_fingerprint:
                raise ValueError(
                    "request_id reuse with different payload in evaluate request"
                )
            return copy.deepcopy(cached_response)

        tunable_id = request.get("tunable_id", self.config.tunable_id)
        evaluations = request.get("evaluations", [])
        config = request.get("config", {})
        session_id = request.get("session_id")

        if not isinstance(evaluations, list):
            raise ValueError("evaluations must be a list")

        # Validate tunable_id matches this service
        if tunable_id != self.config.tunable_id:
            raise ValueError(
                f"tunable_id mismatch: request has '{tunable_id}', "
                f"service is '{self.config.tunable_id}'"
            )

        # Update session if provided
        if session_id:
            self._touch_or_create_session(session_id)

        results: list[dict[str, Any]] = []
        all_metrics: list[dict[str, float]] = []
        failed_input_ids: list[str] = []

        for evaluation in evaluations:
            input_id = evaluation.get("input_id", str(uuid.uuid4()))
            output = evaluation.get("output")
            if output is None and "output_id" in evaluation:
                output = {"output_id": evaluation["output_id"]}
            target = evaluation.get("target")
            if target is None and "target_id" in evaluation:
                target = {"target_id": evaluation["target_id"]}

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

            except HybridAPIError:
                # Bubble up explicit transport-level errors (401/429/503/etc.)
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Evaluate failed for {input_id}: {e}")
                failed_input_ids.append(input_id)
                results.append(
                    {
                        "input_id": input_id,
                        "metrics": {},
                        "error": str(e),
                    }
                )

        successful_results = len(results) - len(failed_input_ids)
        if successful_results == 0 and results:
            status = "failed"
        elif failed_input_ids:
            status = "partial"
        else:
            status = "completed"

        response: dict[str, Any] = {
            "request_id": request_id,
            "status": status,
            "results": results,
            "aggregate_metrics": self._compute_aggregate_metrics(all_metrics),
        }

        if failed_input_ids:
            response["error"] = {
                "code": (
                    "EVALUATION_PARTIAL_FAILURE"
                    if status == "partial"
                    else "EVALUATION_FAILED"
                ),
                "message": "One or more items failed during evaluation",
                "failed_inputs": failed_input_ids,
            }

        # Cache terminal response for idempotent retries.
        # Evict oldest entry if cache is full.
        if len(self._evaluate_idempotency_cache) >= self._idempotency_cache_max_size:
            oldest_key = next(iter(self._evaluate_idempotency_cache))
            del self._evaluate_idempotency_cache[oldest_key]
        self._evaluate_idempotency_cache[request_id] = (
            request_fingerprint,
            copy.deepcopy(response),
        )
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
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if math.isnan(value) or math.isinf(value):
                    continue
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
    ) -> dict[str, dict[str, float | int]]:
        """Compute aggregate statistics (mean, std, n) for each metric."""
        if not metrics_list:
            return {}

        aggregates: dict[str, dict[str, float | int]] = {}

        # Collect values by metric name
        values_by_metric: dict[str, list[float]] = {}
        for metrics in metrics_list:
            for name, value in metrics.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if math.isnan(value) or math.isinf(value):
                    continue
                if name not in values_by_metric:
                    values_by_metric[name] = []
                values_by_metric[name].append(float(value))

        # Compute stats
        import statistics

        for name, values in values_by_metric.items():
            n = len(values)
            mean = sum(values) / n
            std = statistics.stdev(values) if n > 1 else 0.0
            aggregates[name] = {"mean": mean, "std": std, "n": n}

        return aggregates

    def handle_keep_alive(self, session_id: str) -> bool:
        """Handle keep-alive request.

        Args:
            session_id: Session to keep alive.

        Returns:
            True if session is alive, False if not found.
        """
        if not self.config.supports_keep_alive:
            return False

        return self._touch_or_create_session(session_id)

    def _touch_or_create_session(self, session_id: str) -> bool:
        """Touch an existing session or create it when keep-alive is enabled."""
        if not session_id:
            return False

        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
            return True

        if not self.config.supports_keep_alive:
            return False

        self._sessions[session_id] = Session(session_id=session_id)
        return True

    def create_session(self, session_id: str | None = None) -> str:
        """Create a new session.

        Returns:
            New session ID.
        """
        session_id = session_id or str(uuid.uuid4())
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
