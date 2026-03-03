"""Mock server for hybrid mode integration tests.

Provides a simple in-memory implementation of the hybrid API endpoints
for testing without external dependencies.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockServerConfig:
    """Configuration for mock server behavior."""

    # Simulated costs per execution
    cost_per_example: float = 0.001
    latency_per_example_ms: float = 50.0

    # Whether to support optional features
    supports_evaluate: bool = True
    supports_keep_alive: bool = True

    # Simulate failures
    fail_execute_after: int | None = None
    fail_with_auth_error: bool = False

    # Batch limits
    max_batch_size: int = 100

    # Privacy-preserving mode
    privacy_preserving: bool = False


@dataclass
class MockHybridServer:
    """Mock implementation of hybrid API endpoints for testing.

    This class simulates an external agent service that implements
    the Traigent hybrid protocol. Used for integration testing without
    requiring an actual external service.

    Supports both standard mode (full content) and privacy-preserving mode
    (only IDs transmitted, content stored locally).
    """

    config: MockServerConfig = field(default_factory=MockServerConfig)

    # Track state for assertions
    execute_call_count: int = 0
    evaluate_call_count: int = 0
    keep_alive_call_count: int = 0
    received_configs: list[dict[str, Any]] = field(default_factory=list)
    active_sessions: set[str] = field(default_factory=set)

    # Privacy-preserving mode: local data storage
    # Maps example_id -> input data
    _input_storage: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Maps output_id -> output data
    _output_storage: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Maps target_id -> target data
    _target_storage: dict[str, dict[str, Any]] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset server state for test isolation."""
        self.execute_call_count = 0
        self.evaluate_call_count = 0
        self.keep_alive_call_count = 0
        self.received_configs = []
        self.active_sessions = set()
        self._input_storage = {}
        self._output_storage = {}
        self._target_storage = {}

    def store_input(self, example_id: str, data: dict[str, Any]) -> None:
        """Store input data locally (for privacy-preserving mode)."""
        self._input_storage[example_id] = data

    def store_target(self, target_id: str, data: dict[str, Any]) -> None:
        """Store target data locally (for privacy-preserving mode)."""
        self._target_storage[target_id] = data

    def get_input(self, example_id: str) -> dict[str, Any]:
        """Retrieve locally-stored input data by ID."""
        return self._input_storage.get(example_id, {})

    def get_output(self, output_id: str) -> dict[str, Any]:
        """Retrieve locally-stored output data by ID."""
        return self._output_storage.get(output_id, {})

    def get_target(self, target_id: str) -> dict[str, Any]:
        """Retrieve locally-stored target data by ID."""
        return self._target_storage.get(target_id, {})

    def get_capabilities(self) -> dict[str, Any]:
        """Return service capabilities."""
        return {
            "version": "1.0",
            "supports_evaluate": self.config.supports_evaluate,
            "supports_keep_alive": self.config.supports_keep_alive,
            "supports_streaming": False,
            "max_batch_size": self.config.max_batch_size,
            "max_payload_bytes": None,
        }

    def get_config_space(self) -> dict[str, Any]:
        """Return TVAR definitions for the mock agent."""
        tunables = [
            {
                "name": "model",
                "type": "enum",
                "domain": {"values": ["fast", "accurate", "balanced"]},
                "default": "balanced",
            },
            {
                "name": "temperature",
                "type": "float",
                "domain": {"range": [0.0, 1.0], "resolution": 0.1},
                "default": 0.5,
            },
            {
                "name": "max_retries",
                "type": "int",
                "domain": {"range": [0, 5]},
                "default": 2,
            },
            {
                "name": "use_cache",
                "type": "bool",
                "domain": {},
                "default": True,
            },
        ]
        return {
            "schema_version": "0.9",
            "tunable_id": "mock_test_agent",
            "tunables": tunables,
            "tvars": tunables,
        }

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute agent with given config on examples.

        Simulates agent execution by computing mock outputs based on
        the configuration and tracking costs.
        """
        self.execute_call_count += 1

        # Track received config for test assertions
        config = request.get("config", {})
        self.received_configs.append(config)

        request_id = request.get("request_id", str(uuid.uuid4()))

        # Check for simulated auth error
        if self.config.fail_with_auth_error:
            return {
                "request_id": request_id,
                "execution_id": "",
                "status": "failed",
                "outputs": [],
                "operational_metrics": {},
                "error": {
                    "code": "AUTH_ERROR",
                    "message": "Unauthorized",
                },
            }

        # Check for simulated failure after N calls
        if (
            self.config.fail_execute_after is not None
            and self.execute_call_count > self.config.fail_execute_after
        ):
            return {
                "request_id": request_id,
                "execution_id": "",
                "status": "failed",
                "outputs": [],
                "operational_metrics": {},
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Simulated failure for testing",
                },
            }

        inputs = request.get("examples", [])
        session_id = request.get("session_id")

        if session_id:
            self.active_sessions.add(session_id)

        # Generate outputs based on config
        outputs = []
        total_cost = 0.0
        total_latency = 0.0

        for inp in inputs:
            example_id = inp.get("example_id", str(uuid.uuid4()))

            # Privacy-preserving mode: look up data locally by example_id
            # Standard mode: use data from request
            if "data" in inp:
                data = inp["data"]
            else:
                # Privacy-preserving: look up locally-stored input
                data = self.get_input(example_id)

            # Simulate output based on config
            output = self._simulate_output(config, data)

            # Calculate costs
            cost = self.config.cost_per_example
            latency = self.config.latency_per_example_ms

            # Model affects cost
            model = config.get("model", "balanced")
            if model == "accurate":
                cost *= 2.0
                latency *= 1.5
            elif model == "fast":
                cost *= 0.5
                latency *= 0.5

            total_cost += cost
            total_latency += latency

            # Build output item
            output_item: dict[str, Any] = {
                "example_id": example_id,
                "cost_usd": cost,
                "latency_ms": latency,
            }

            if self.config.privacy_preserving:
                # Privacy-preserving mode: store output locally and return only ID
                output_id = f"out_{example_id}_{session_id or 'default'}"
                self._output_storage[output_id] = output
                output_item["output_id"] = output_id
            else:
                # Standard mode: include full output content
                output_item["output"] = output

            outputs.append(output_item)

        response = {
            "request_id": request_id,
            "execution_id": str(uuid.uuid4()),
            "status": "completed",
            "outputs": outputs,
            "operational_metrics": {
                "total_cost_usd": total_cost,
                "total_latency_ms": total_latency,
                "p95_latency_ms": total_latency / len(inputs) if inputs else 0,
            },
        }

        if session_id:
            response["session_id"] = session_id

        return response

    def _simulate_output(
        self, config: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate agent output based on config and input.

        The mock agent's behavior is deterministic based on config:
        - 'accurate' model: higher quality scores
        - 'fast' model: lower quality but faster
        - 'balanced' model: medium quality
        - Higher temperature: more variation in output
        """
        model = config.get("model", "balanced")
        temperature = config.get("temperature", 0.5)
        use_cache = config.get("use_cache", True)

        # Base quality depends on model
        if model == "accurate":
            base_quality = 0.9
        elif model == "fast":
            base_quality = 0.6
        else:
            base_quality = 0.75

        # Temperature affects consistency (not randomness in mock)
        quality_variation = temperature * 0.1

        # Cache affects slightly
        if use_cache:
            base_quality += 0.02

        # Generate deterministic output
        return {
            "response": f"Mock response for {data}",
            "quality_score": min(1.0, base_quality + quality_variation),
            "model_used": model,
            "cached": use_cache,
        }

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Evaluate outputs against targets.

        Computes mock quality metrics for each output-target pair.
        """
        self.evaluate_call_count += 1

        request_id = request.get("request_id", str(uuid.uuid4()))

        if not self.config.supports_evaluate:
            return {
                "request_id": request_id,
                "status": "failed",
                "results": [],
                "aggregate_metrics": {},
                "error": {
                    "code": "NOT_SUPPORTED",
                    "message": "Evaluate not supported",
                },
            }

        evaluations = request.get("evaluations", [])
        results = []
        accuracy_sum = 0.0

        for eval_item in evaluations:
            example_id = eval_item.get("example_id", str(uuid.uuid4()))

            # Privacy-preserving mode: look up output and target by ID
            # Standard mode: use data from request
            if "output" in eval_item:
                output = eval_item["output"]
            elif "output_id" in eval_item:
                output = self.get_output(eval_item["output_id"])
            else:
                output = {}

            if "target" in eval_item:
                target = eval_item["target"]
            elif "target_id" in eval_item:
                target = self.get_target(eval_item["target_id"])
            else:
                target = {}

            # Compute mock accuracy based on output quality
            quality = output.get("quality_score", 0.5)

            # Target matching affects accuracy
            if "expected" in target:
                # Simple string matching simulation
                accuracy = quality * 0.9
            else:
                accuracy = quality

            accuracy_sum += accuracy

            results.append(
                {
                    "example_id": example_id,
                    "metrics": {
                        "accuracy": accuracy,
                        "relevance": quality * 0.95,
                    },
                }
            )

        n = len(results) if results else 1
        mean_accuracy = accuracy_sum / n

        return {
            "request_id": request_id,
            "status": "completed",
            "results": results,
            "aggregate_metrics": {
                "accuracy": {
                    "mean": mean_accuracy,
                    "std": 0.05,
                    "n": n,
                },
                "relevance": {
                    "mean": mean_accuracy * 0.95,
                    "std": 0.03,
                    "n": n,
                },
            },
        }

    def keep_alive(self, session_id: str) -> dict[str, Any]:
        """Handle keep-alive heartbeat."""
        self.keep_alive_call_count += 1

        if not self.config.supports_keep_alive:
            return {
                "status": "unsupported",
                "alive": False,
                "error": "Keep-alive not supported",
            }

        if session_id in self.active_sessions:
            return {"status": "alive", "alive": True, "session_id": session_id}
        else:
            return {"status": "expired", "alive": False, "reason": "Session not found"}

    def health_check(self) -> dict[str, Any]:
        """Return health status."""
        return {
            "status": "healthy",
            "version": "1.0.0-mock",
            "uptime_seconds": 3600.0,
        }


class MockHTTPTransport:
    """Mock HTTP transport that routes to MockHybridServer.

    Implements the HybridTransport protocol using an in-memory
    mock server instead of actual HTTP calls.
    """

    def __init__(self, server: MockHybridServer):
        self._server = server
        self._closed = False

    async def capabilities(self) -> dict[str, Any]:
        """Return service capabilities."""
        from traigent.hybrid.protocol import ServiceCapabilities

        data = self._server.get_capabilities()
        return ServiceCapabilities.from_dict(data)

    async def discover_config_space(
        self, *, tunable_id: str | None = None
    ) -> dict[str, Any]:
        """Fetch config space from mock server."""
        from traigent.hybrid.protocol import ConfigSpaceResponse

        data = self._server.get_config_space()
        return ConfigSpaceResponse.from_dict(data)

    async def execute(self, request: Any) -> Any:
        """Execute via mock server."""
        from traigent.hybrid.protocol import HybridExecuteResponse

        request_dict = request.to_dict()
        response_data = self._server.execute(request_dict)
        return HybridExecuteResponse.from_dict(response_data)

    async def evaluate(self, request: Any) -> Any:
        """Evaluate via mock server."""
        from traigent.hybrid.protocol import HybridEvaluateResponse

        request_dict = request.to_dict()
        response_data = self._server.evaluate(request_dict)
        return HybridEvaluateResponse.from_dict(response_data)

    async def health_check(self) -> Any:
        """Health check via mock server."""
        from traigent.hybrid.protocol import HealthCheckResponse

        data = self._server.health_check()
        return HealthCheckResponse.from_dict(data)

    async def keep_alive(self, session_id: str) -> bool:
        """Keep-alive via mock server."""
        result = self._server.keep_alive(session_id)
        return result.get("alive", False)

    async def close(self) -> None:
        """Close the transport."""
        self._closed = True
