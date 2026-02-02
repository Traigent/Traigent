"""Integration tests for hybrid mode optimization.

Tests the complete hybrid mode workflow including:
- Full optimization loop with mock external service
- Cost limit enforcement
- Batch size control
- Keep-alive heartbeat and cleanup
- Transport abstraction
"""

import asyncio

import pytest

from tests.integration.hybrid.mock_hybrid_server import (
    MockHTTPTransport,
    MockHybridServer,
    MockServerConfig,
)
from traigent.hybrid.discovery import ConfigSpaceDiscovery
from traigent.hybrid.lifecycle import AgentLifecycleManager
from traigent.hybrid.protocol import (
    HybridEvaluateRequest,
    HybridExecuteRequest,
)


class TestHybridModeConfigSpaceDiscovery:
    """Integration tests for config space discovery."""

    @pytest.fixture
    def mock_server(self) -> MockHybridServer:
        """Create mock server for testing."""
        return MockHybridServer()

    @pytest.fixture
    def transport(self, mock_server: MockHybridServer) -> MockHTTPTransport:
        """Create mock transport."""
        return MockHTTPTransport(mock_server)

    @pytest.mark.asyncio
    async def test_fetch_and_normalize_config_space(
        self, transport: MockHTTPTransport
    ) -> None:
        """Test fetching and normalizing config space from service."""
        discovery = ConfigSpaceDiscovery(transport)

        config_space = await discovery.fetch_and_normalize()

        # Verify TVAR normalization
        assert "model" in config_space
        assert config_space["model"] == ["fast", "accurate", "balanced"]

        assert "temperature" in config_space
        assert config_space["temperature"]["low"] == 0.0
        assert config_space["temperature"]["high"] == 1.0
        assert config_space["temperature"]["step"] == 0.1

        assert "max_retries" in config_space
        assert config_space["max_retries"]["low"] == 0
        assert config_space["max_retries"]["high"] == 5
        assert config_space["max_retries"]["type"] == "int"

        assert "use_cache" in config_space
        assert config_space["use_cache"] == [True, False]

    @pytest.mark.asyncio
    async def test_discovery_caching(self, transport: MockHTTPTransport) -> None:
        """Test that discovery caches results."""
        discovery = ConfigSpaceDiscovery(transport)

        # First fetch
        await discovery.fetch()
        capability_id_1 = discovery.get_capability_id()

        # Second fetch should return cached
        await discovery.fetch()
        capability_id_2 = discovery.get_capability_id()

        assert capability_id_1 == capability_id_2 == "mock_test_agent"

    @pytest.mark.asyncio
    async def test_discovery_clear_cache(self, transport: MockHTTPTransport) -> None:
        """Test clearing cache forces re-fetch."""
        discovery = ConfigSpaceDiscovery(transport)

        await discovery.fetch()
        tvars_1 = discovery.get_tvars()

        discovery.clear_cache()

        # After clear, should have no TVARs until fetch
        tvars_empty = discovery.get_tvars()
        assert tvars_empty == []

        # Re-fetch
        await discovery.fetch()
        tvars_2 = discovery.get_tvars()
        assert len(tvars_2) == len(tvars_1)


class TestHybridModeExecution:
    """Integration tests for hybrid mode execution flow."""

    @pytest.fixture
    def mock_server(self) -> MockHybridServer:
        """Create mock server for testing."""
        return MockHybridServer()

    @pytest.fixture
    def transport(self, mock_server: MockHybridServer) -> MockHTTPTransport:
        """Create mock transport."""
        return MockHTTPTransport(mock_server)

    @pytest.mark.asyncio
    async def test_execute_single_example(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test executing a single example."""
        request = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={"model": "accurate", "temperature": 0.3},
            inputs=[{"input_id": "ex_1", "data": {"query": "test question"}}],
        )

        response = await transport.execute(request)

        assert response.status == "completed"
        assert len(response.outputs) == 1
        assert response.outputs[0]["input_id"] == "ex_1"
        assert "output" in response.outputs[0]
        assert response.operational_metrics["total_cost_usd"] > 0

        # Verify server received the config
        assert mock_server.execute_call_count == 1
        assert mock_server.received_configs[0]["model"] == "accurate"

    @pytest.mark.asyncio
    async def test_execute_batch(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test executing a batch of examples."""
        inputs = [
            {"input_id": f"ex_{i}", "data": {"query": f"question {i}"}}
            for i in range(5)
        ]

        request = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={"model": "balanced"},
            inputs=inputs,
        )

        response = await transport.execute(request)

        assert response.status == "completed"
        assert len(response.outputs) == 5

        # Verify costs scale with batch size
        total_cost = response.operational_metrics["total_cost_usd"]
        assert total_cost == 5 * mock_server.config.cost_per_example

    @pytest.mark.asyncio
    async def test_execute_with_session(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test executing with session ID for stateful agents."""
        session_id = "session_123"

        request = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={"model": "fast"},
            inputs=[{"input_id": "ex_1", "data": {}}],
            session_id=session_id,
        )

        response = await transport.execute(request)

        assert response.status == "completed"
        assert response.session_id == session_id
        assert session_id in mock_server.active_sessions

    @pytest.mark.asyncio
    async def test_model_affects_cost(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test that different model configurations affect cost."""
        inputs = [{"input_id": "ex_1", "data": {}}]

        # Fast model - lower cost
        request_fast = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={"model": "fast"},
            inputs=inputs,
        )
        response_fast = await transport.execute(request_fast)
        cost_fast = response_fast.operational_metrics["total_cost_usd"]

        mock_server.reset()

        # Accurate model - higher cost
        request_accurate = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={"model": "accurate"},
            inputs=inputs,
        )
        response_accurate = await transport.execute(request_accurate)
        cost_accurate = response_accurate.operational_metrics["total_cost_usd"]

        # Accurate should cost more than fast
        assert cost_accurate > cost_fast


class TestHybridModeEvaluation:
    """Integration tests for hybrid mode evaluation flow."""

    @pytest.fixture
    def mock_server(self) -> MockHybridServer:
        """Create mock server for testing."""
        return MockHybridServer()

    @pytest.fixture
    def transport(self, mock_server: MockHybridServer) -> MockHTTPTransport:
        """Create mock transport."""
        return MockHTTPTransport(mock_server)

    @pytest.mark.asyncio
    async def test_evaluate_outputs(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test evaluating outputs against targets."""
        request = HybridEvaluateRequest(
            capability_id="mock_test_agent",
            evaluations=[
                {
                    "input_id": "ex_1",
                    "output": {"quality_score": 0.9, "response": "answer 1"},
                    "target": {"expected": "correct answer"},
                },
                {
                    "input_id": "ex_2",
                    "output": {"quality_score": 0.7, "response": "answer 2"},
                    "target": {"expected": "correct answer"},
                },
            ],
        )

        response = await transport.evaluate(request)

        assert response.status == "completed"
        assert len(response.results) == 2
        assert "accuracy" in response.aggregate_metrics
        assert response.aggregate_metrics["accuracy"]["n"] == 2

    @pytest.mark.asyncio
    async def test_execute_then_evaluate(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test two-phase: execute then evaluate workflow."""
        # Phase 1: Execute
        exec_request = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={"model": "accurate", "temperature": 0.2},
            inputs=[
                {"input_id": "ex_1", "data": {"query": "What is 2+2?"}},
                {"input_id": "ex_2", "data": {"query": "Capital of France?"}},
            ],
        )
        exec_response = await transport.execute(exec_request)

        assert exec_response.status == "completed"

        # Phase 2: Evaluate
        evaluations = [
            {
                "input_id": out["input_id"],
                "output": out["output"],
                "target": {"expected": "correct"},
            }
            for out in exec_response.outputs
        ]

        eval_request = HybridEvaluateRequest(
            capability_id="mock_test_agent",
            execution_id=exec_response.execution_id,
            evaluations=evaluations,
        )
        eval_response = await transport.evaluate(eval_request)

        assert eval_response.status == "completed"
        assert mock_server.execute_call_count == 1
        assert mock_server.evaluate_call_count == 1


class TestHybridModeLifecycle:
    """Integration tests for agent lifecycle management."""

    @pytest.fixture
    def mock_server(self) -> MockHybridServer:
        """Create mock server for testing."""
        return MockHybridServer()

    @pytest.fixture
    def transport(self, mock_server: MockHybridServer) -> MockHTTPTransport:
        """Create mock transport."""
        return MockHTTPTransport(mock_server)

    @pytest.mark.asyncio
    async def test_lifecycle_register_and_release(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test registering and releasing sessions."""
        manager = AgentLifecycleManager(
            transport=transport,
            heartbeat_interval=0.1,  # Fast for tests
        )

        session_id = manager.create_session()
        await manager.register(session_id)

        assert manager.session_count == 1
        assert manager.is_session_alive(session_id)

        await manager.release()

        assert manager.session_count == 0

    @pytest.mark.asyncio
    async def test_lifecycle_heartbeat(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test that heartbeat calls keep-alive endpoint."""
        manager = AgentLifecycleManager(
            transport=transport,
            heartbeat_interval=0.05,  # Very fast for test
        )

        session_id = "test_session"

        # Execute a request to register session on server
        request = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={},
            inputs=[{"input_id": "ex_1", "data": {}}],
            session_id=session_id,
        )
        await transport.execute(request)

        # Now register with lifecycle manager
        await manager.register(session_id)

        # Wait for heartbeats
        await asyncio.sleep(0.2)

        # Should have called keep-alive
        assert mock_server.keep_alive_call_count > 0

        await manager.release()

    @pytest.mark.asyncio
    async def test_lifecycle_context_manager(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test using lifecycle manager as context manager."""
        async with AgentLifecycleManager(
            transport=transport,
            heartbeat_interval=0.1,
        ) as manager:
            await manager.register("session_1")
            await manager.register("session_2")
            assert manager.session_count == 2

        # After exit, should be released
        assert manager.session_count == 0


class TestHybridModeCostControl:
    """Integration tests for cost tracking and limits."""

    @pytest.fixture
    def mock_server(self) -> MockHybridServer:
        """Create mock server with specific cost configuration."""
        config = MockServerConfig(cost_per_example=0.01)  # $0.01 per example
        return MockHybridServer(config=config)

    @pytest.fixture
    def transport(self, mock_server: MockHybridServer) -> MockHTTPTransport:
        """Create mock transport."""
        return MockHTTPTransport(mock_server)

    @pytest.mark.asyncio
    async def test_cost_tracking_single_batch(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test that costs are accurately tracked for single batch."""
        inputs = [{"input_id": f"ex_{i}", "data": {}} for i in range(10)]

        request = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={"model": "balanced"},  # 1x cost multiplier
            inputs=inputs,
        )

        response = await transport.execute(request)

        # Should be 10 * $0.01 = $0.10
        assert response.operational_metrics["total_cost_usd"] == pytest.approx(0.10)

    @pytest.mark.asyncio
    async def test_cost_tracking_multiple_batches(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test cost accumulation across multiple batches."""
        total_cost = 0.0

        for batch_idx in range(3):
            inputs = [
                {"input_id": f"batch{batch_idx}_ex_{i}", "data": {}} for i in range(5)
            ]

            request = HybridExecuteRequest(
                capability_id="mock_test_agent",
                config={"model": "balanced"},
                inputs=inputs,
            )

            response = await transport.execute(request)
            total_cost += response.operational_metrics["total_cost_usd"]

        # 3 batches * 5 examples * $0.01 = $0.15
        assert total_cost == pytest.approx(0.15)

    @pytest.mark.asyncio
    async def test_per_example_cost_extraction(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test extracting per-example costs from response."""
        request = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={"model": "accurate"},  # 2x cost
            inputs=[
                {"input_id": "ex_1", "data": {}},
                {"input_id": "ex_2", "data": {}},
            ],
        )

        response = await transport.execute(request)

        # Each output should have its own cost
        for output in response.outputs:
            assert "cost_usd" in output
            assert output["cost_usd"] == pytest.approx(0.02)  # $0.01 * 2x


class TestHybridModeBatchControl:
    """Integration tests for batch size control."""

    @pytest.fixture
    def mock_server(self) -> MockHybridServer:
        """Create mock server."""
        return MockHybridServer()

    @pytest.fixture
    def transport(self, mock_server: MockHybridServer) -> MockHTTPTransport:
        """Create mock transport."""
        return MockHTTPTransport(mock_server)

    @pytest.mark.asyncio
    async def test_batch_size_one(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test batch size of 1 (sequential execution)."""
        for i in range(3):
            request = HybridExecuteRequest(
                capability_id="mock_test_agent",
                config={"model": "fast"},
                inputs=[{"input_id": f"ex_{i}", "data": {}}],
            )
            await transport.execute(request)

        # Should have 3 separate execute calls
        assert mock_server.execute_call_count == 3

    @pytest.mark.asyncio
    async def test_batch_size_all(
        self, transport: MockHTTPTransport, mock_server: MockHybridServer
    ) -> None:
        """Test sending all examples in one batch."""
        inputs = [{"input_id": f"ex_{i}", "data": {}} for i in range(10)]

        request = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={"model": "fast"},
            inputs=inputs,
        )
        response = await transport.execute(request)

        # Should have 1 execute call with 10 outputs
        assert mock_server.execute_call_count == 1
        assert len(response.outputs) == 10


class TestHybridModeErrorHandling:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_handle_execute_failure(self) -> None:
        """Test handling execute failures gracefully."""
        config = MockServerConfig(fail_execute_after=0)  # Fail immediately
        server = MockHybridServer(config=config)
        transport = MockHTTPTransport(server)

        request = HybridExecuteRequest(
            capability_id="mock_test_agent",
            config={},
            inputs=[{"input_id": "ex_1", "data": {}}],
        )

        response = await transport.execute(request)
        assert response.status == "failed"
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_handle_evaluate_not_supported(self) -> None:
        """Test handling when evaluate is not supported."""
        config = MockServerConfig(supports_evaluate=False)
        server = MockHybridServer(config=config)
        transport = MockHTTPTransport(server)

        request = HybridEvaluateRequest(
            capability_id="mock_test_agent",
            evaluations=[{"input_id": "ex_1", "output": {}, "target": {}}],
        )

        response = await transport.evaluate(request)
        assert response.status == "failed"

    @pytest.mark.asyncio
    async def test_transport_close(self) -> None:
        """Test transport cleanup on close."""
        server = MockHybridServer()
        transport = MockHTTPTransport(server)

        await transport.close()
        assert transport._closed is True


class TestHybridModeCapabilities:
    """Integration tests for capability negotiation."""

    @pytest.mark.asyncio
    async def test_capabilities_full_featured(self) -> None:
        """Test discovering full-featured service."""
        config = MockServerConfig(
            supports_evaluate=True,
            supports_keep_alive=True,
        )
        server = MockHybridServer(config=config)
        transport = MockHTTPTransport(server)

        capabilities = await transport.capabilities()

        assert capabilities.supports_evaluate is True
        assert capabilities.supports_keep_alive is True
        assert capabilities.version == "1.0"

    @pytest.mark.asyncio
    async def test_capabilities_minimal(self) -> None:
        """Test discovering minimal service."""
        config = MockServerConfig(
            supports_evaluate=False,
            supports_keep_alive=False,
        )
        server = MockHybridServer(config=config)
        transport = MockHTTPTransport(server)

        capabilities = await transport.capabilities()

        assert capabilities.supports_evaluate is False
        assert capabilities.supports_keep_alive is False

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health check endpoint."""
        server = MockHybridServer()
        transport = MockHTTPTransport(server)

        health = await transport.health_check()

        assert health.status == "healthy"
        assert health.version == "1.0.0-mock"
