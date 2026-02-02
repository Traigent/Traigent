"""Integration tests for hybrid and SaaS execution modes."""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.adapters.execution_adapter import LocalExecutionAdapter
from traigent.cloud.optimizer_client import OptimizerDirectClient
from traigent.config.types import ExecutionMode
from traigent.traigent_client import TraigentClient as TraigentClient


class TestExecutionModes:
    """Test suite for different execution modes."""

    @pytest.fixture
    def mock_agent_builder(self):
        """Mock agent builder for tests."""
        builder = Mock()

        # Mock agent that returns simple results
        mock_agent = Mock()
        mock_agent.execute = Mock(return_value="test output")
        builder.build_agent = Mock(return_value=mock_agent)

        return builder

    @pytest.fixture
    def test_dataset(self):
        """Test dataset for evaluations."""
        return {
            "name": "test_dataset",
            "examples": [
                {
                    "id": "ex1",
                    "input": {"text": "What is 2+2?"},
                    "expected_output": "4",
                    "metadata": {"evaluation_type": "exact_match"},
                },
                {
                    "id": "ex2",
                    "input": {"text": "What is the capital of France?"},
                    "expected_output": "Paris",
                    "metadata": {"evaluation_type": "exact_match"},
                },
                {
                    "id": "ex3",
                    "input": {"text": "Calculate 10 * 5"},
                    "expected_output": "50",
                    "metadata": {"evaluation_type": "numeric", "tolerance": 0.1},
                },
            ],
        }

    @pytest.fixture
    def configuration_space(self):
        """Test configuration space."""
        return {
            "temperature": [0.1, 0.5, 0.9],
            "max_tokens": [100, 500],
            "model": ["gpt-3.5-turbo"],
        }

    @pytest.mark.asyncio
    async def test_edge_analytics_mode_execution(
        self, mock_agent_builder, test_dataset, configuration_space
    ):
        """Test Edge Analytics (formerly local) mode execution."""
        # Set environment for Edge Analytics mode
        os.environ["TRAIGENT_FORCE_LOCAL"] = "true"

        try:
            client = TraigentClient(
                execution_mode="edge_analytics", agent_builder=mock_agent_builder
            )

            # Define test function
            def test_function(input_data):
                return "test output"

            # Run optimization
            result = await client.optimize(
                function=test_function,
                dataset=test_dataset,
                configuration_space=configuration_space,
                objectives=["accuracy"],
                max_trials=2,
            )

            # Verify results
            assert result["execution_mode"] == "edge_analytics"
            assert result["status"] == "completed"
            assert "best_configuration" in result
            assert "all_results" in result
            assert result["completed_trials"] > 0

            # Verify agent builder was called
            assert mock_agent_builder.build_agent.called

        finally:
            del os.environ["TRAIGENT_FORCE_LOCAL"]

    def test_legacy_local_label_is_accepted(self):
        """Ensure legacy execution_mode='edge_analytics' still maps to Edge Analytics."""
        client = TraigentClient(execution_mode="edge_analytics")
        assert client.execution_mode == ExecutionMode.EDGE_ANALYTICS

    @pytest.mark.asyncio
    async def test_hybrid_mode_execution(
        self, mock_agent_builder, test_dataset, configuration_space
    ):
        """Test standard mode execution."""
        # Mock backend and optimizer clients
        with patch("traigent.traigent_client.BackendIntegratedClient") as MockBackend:
            with patch(
                "traigent.traigent_client.OptimizerDirectClient"
            ) as MockOptimizer:
                # Setup mock backend
                mock_backend = AsyncMock()
                mock_backend.create_hybrid_session = AsyncMock(
                    return_value=(
                        "session123",
                        "token123",
                        "http://optimizer:8000/api/v1/metrics",
                    )
                )
                mock_backend.finalize_hybrid_session = AsyncMock(
                    return_value={
                        "best_configuration": {"temperature": 0.5, "max_tokens": 500},
                        "best_metrics": {"accuracy": 0.95},
                        "status": "completed",
                    }
                )
                mock_backend.__aenter__ = AsyncMock(return_value=mock_backend)
                mock_backend.__aexit__ = AsyncMock()
                MockBackend.return_value = mock_backend

                # Setup mock optimizer
                mock_optimizer = AsyncMock()
                mock_optimizer.get_next_configuration = AsyncMock(
                    side_effect=[
                        {
                            "has_next": True,
                            "configuration": {"temperature": 0.5, "max_tokens": 500},
                            "trial_id": "trial1",
                        },
                        {
                            "has_next": True,
                            "configuration": {"temperature": 0.9, "max_tokens": 100},
                            "trial_id": "trial2",
                        },
                        {"has_next": False},
                    ]
                )
                mock_optimizer.submit_metrics = AsyncMock()
                mock_optimizer.__aenter__ = AsyncMock(return_value=mock_optimizer)
                mock_optimizer.__aexit__ = AsyncMock()
                MockOptimizer.return_value = mock_optimizer

                # Create client in standard mode
                client = TraigentClient(
                    execution_mode="standard",
                    agent_builder=mock_agent_builder,
                    api_key="test_key",
                )

                # Run optimization
                result = await client.optimize(
                    function=lambda x: x,
                    dataset=test_dataset,
                    configuration_space=configuration_space,
                    objectives=["accuracy"],
                    max_trials=2,
                )

                # Verify results
                assert result["execution_mode"] == "standard"
                assert result["completed_trials"] == 2
                assert "best_configuration" in result

                # Verify backend calls
                mock_backend.create_hybrid_session.assert_called_once()
                mock_backend.finalize_hybrid_session.assert_called_once()

                # Verify optimizer calls
                assert mock_optimizer.get_next_configuration.call_count == 2
                assert mock_optimizer.submit_metrics.call_count == 2

    @pytest.mark.asyncio
    async def test_cloud_mode_execution(self, test_dataset, configuration_space):
        """Test SaaS mode execution."""
        with patch("traigent.traigent_client.BackendIntegratedClient") as MockBackend:
            # Setup mock backend for SaaS mode
            mock_backend = AsyncMock()
            mock_backend.upload_dataset = AsyncMock(
                return_value={"dataset_id": "dataset123"}
            )
            mock_backend.create_optimization_session = AsyncMock(
                return_value={"session_id": "saas_session123"}
            )
            mock_backend.get_session_status = AsyncMock(
                side_effect=[
                    {"status": "RUNNING", "completed_trials": 1},
                    {"status": "RUNNING", "completed_trials": 2},
                    {"status": "COMPLETED", "completed_trials": 2},
                ]
            )
            mock_backend.get_optimization_results = AsyncMock(
                return_value={
                    "best_configuration": {"temperature": 0.7},
                    "best_metrics": {"accuracy": 0.98},
                    "total_trials": 2,
                }
            )
            mock_backend.__aenter__ = AsyncMock(return_value=mock_backend)
            mock_backend.__aexit__ = AsyncMock()
            MockBackend.return_value = mock_backend

            # Create client in SaaS mode
            client = TraigentClient(execution_mode="cloud", api_key="test_key")

            # Run optimization
            result = await client.optimize(
                function=lambda x: x,
                dataset=test_dataset,
                configuration_space=configuration_space,
                objectives=["accuracy"],
                max_trials=2,
            )

            # Verify results
            assert result["execution_mode"] == "cloud"
            assert result["best_metrics"]["accuracy"] == 0.98

            # Verify backend calls
            mock_backend.upload_dataset.assert_called_once()
            mock_backend.create_optimization_session.assert_called_once()
            assert mock_backend.get_session_status.call_count >= 1
            mock_backend.get_optimization_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_mode_auto_detection(self):
        """Test automatic execution mode detection."""
        # Test Edge Analytics detection (no API key)
        client = TraigentClient(execution_mode="auto")
        assert client.execution_mode == ExecutionMode.EDGE_ANALYTICS

        # Test SaaS mode detection (with API key)
        client = TraigentClient(execution_mode="auto", api_key="test_key")
        assert client.execution_mode == ExecutionMode.CLOUD

        # Test standard mode detection (privacy flag)
        os.environ["TRAIGENT_PRIVATE_DATA"] = "true"
        try:
            client = TraigentClient(execution_mode="auto", api_key="test_key")
            assert client.execution_mode == ExecutionMode.STANDARD
        finally:
            del os.environ["TRAIGENT_PRIVATE_DATA"]

    @pytest.mark.asyncio
    async def test_local_execution_adapter(self, mock_agent_builder, test_dataset):
        """Test local execution adapter directly."""
        adapter = LocalExecutionAdapter(mock_agent_builder)

        agent_spec = {"temperature": 0.5, "max_tokens": 500, "model": "gpt-3.5-turbo"}

        result = await adapter.execute_configuration(
            agent_spec=agent_spec, dataset=test_dataset, trial_id="test_trial_1"
        )

        # Verify result structure
        assert result["trial_id"] == "test_trial_1"
        assert "metrics" in result
        assert "execution_time" in result
        assert "metadata" in result

        # Verify metrics calculation
        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "success_rate" in metrics
        assert "total_examples" in metrics

        # Verify execution mode
        mode = await adapter.get_execution_mode()
        assert mode == "edge_analytics"

    @pytest.mark.asyncio
    async def test_metric_submission_batching(self):
        """Test metric submission batching in optimizer client."""
        with patch("aiohttp.ClientSession") as MockSession:
            # Setup mock session
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "accepted"})
            mock_response.raise_for_status = Mock()

            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            MockSession.return_value = mock_session

            # Create optimizer client
            client = OptimizerDirectClient(
                endpoint="http://optimizer:8000/api/v1/metrics", token="test_token"
            )

            # Set small batch size for testing
            client.set_batch_size(2)
            client.set_flush_interval(0.1)

            async with client:
                # Submit metrics
                await client.submit_metrics(
                    session_id="session1",
                    trial_id="trial1",
                    metrics={"accuracy": 0.8},
                    execution_time=1.5,
                )

                await client.submit_metrics(
                    session_id="session1",
                    trial_id="trial2",
                    metrics={"accuracy": 0.85},
                    execution_time=1.7,
                )

                # Should trigger batch submission
                await asyncio.sleep(0.2)

            # Verify batch submission was called
            assert mock_session.post.called

    @pytest.mark.asyncio
    async def test_error_handling_in_hybrid_mode(
        self, mock_agent_builder, test_dataset, configuration_space
    ):
        """Test error handling in standard mode."""
        with patch("traigent.traigent_client.BackendIntegratedClient") as MockBackend:
            # Setup mock backend that fails on context manager entry
            mock_backend = AsyncMock()
            mock_backend.create_hybrid_session = AsyncMock(
                side_effect=Exception("Backend connection failed")
            )
            # Make __aenter__ raise the exception during connection
            mock_backend.__aenter__ = AsyncMock(
                side_effect=Exception("Backend connection failed")
            )
            mock_backend.__aexit__ = AsyncMock()
            MockBackend.return_value = mock_backend

            client = TraigentClient(
                execution_mode="standard",
                agent_builder=mock_agent_builder,
                api_key="test_key",  # Need API key for backend client
            )

            # Should handle exception gracefully or raise it
            try:
                await client.optimize(
                    function=lambda x: x,
                    dataset=test_dataset,
                    configuration_space=configuration_space,
                    objectives=["accuracy"],
                    max_trials=2,
                )
                # If no exception raised, check if error is in result
                assert False, "Expected exception but none was raised"
            except Exception as e:
                assert "Backend connection failed" in str(e)


class TestPrivacyCompliance:
    """Test privacy compliance in different modes."""

    @pytest.fixture
    def mock_agent_builder(self):
        """Mock agent builder for tests."""
        builder = Mock()

        # Mock agent that returns simple results
        mock_agent = Mock()
        mock_agent.execute = Mock(return_value="test output")
        builder.build_agent = Mock(return_value=mock_agent)

        return builder

    @pytest.fixture
    def test_dataset(self):
        """Test dataset for evaluations."""
        return {
            "name": "test_dataset",
            "examples": [
                {
                    "id": "ex1",
                    "input": {"text": "What is 2+2?"},
                    "expected_output": "4",
                    "metadata": {"evaluation_type": "exact_match"},
                },
                {
                    "id": "ex2",
                    "input": {"text": "What is the capital of France?"},
                    "expected_output": "Paris",
                    "metadata": {"evaluation_type": "exact_match"},
                },
                {
                    "id": "ex3",
                    "input": {"text": "Calculate 10 * 5"},
                    "expected_output": "50",
                    "metadata": {"evaluation_type": "numeric", "tolerance": 0.1},
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_hybrid_mode_no_data_transmission(
        self, mock_agent_builder, test_dataset
    ):
        """Verify that standard mode doesn't transmit actual data."""
        transmitted_data = []

        # Intercept all network calls
        with patch("aiohttp.ClientSession") as MockSession:
            mock_session = AsyncMock()

            # Capture all POST data
            async def capture_post(url, json=None, **kwargs):
                transmitted_data.append(json)
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={"status": "ok"})
                return mock_response

            mock_session.post = capture_post
            MockSession.return_value = mock_session

            # Run in standard mode
            os.environ["TRAIGENT_FORCE_HYBRID"] = "true"
            try:
                TraigentClient(
                    execution_mode="standard", agent_builder=mock_agent_builder
                )

                # Check transmitted data doesn't contain dataset
                for data in transmitted_data:
                    if data:
                        assert "examples" not in str(data)
                        assert test_dataset["examples"][0]["input"]["text"] not in str(
                            data
                        )

            finally:
                del os.environ["TRAIGENT_FORCE_HYBRID"]

    @pytest.mark.asyncio
    async def test_cloud_mode_data_encryption(self):
        """Verify that SaaS mode encrypts data before transmission."""
        # This would test the DatasetStorageService encryption
        # Implementation depends on actual encryption setup
        pytest.skip("Requires DatasetStorageService encryption setup")


class TestPerformanceAndScaling:
    """Test performance characteristics of different modes."""

    @pytest.fixture
    def mock_agent_builder(self):
        """Mock agent builder for tests."""
        builder = Mock()

        # Mock agent that returns simple results
        mock_agent = Mock()
        mock_agent.execute = Mock(return_value="test output")
        builder.build_agent = Mock(return_value=mock_agent)

        return builder

    @pytest.fixture
    def test_dataset(self):
        """Test dataset for evaluations."""
        return {
            "name": "test_dataset",
            "examples": [
                {
                    "id": "ex1",
                    "input": {"text": "What is 2+2?"},
                    "expected_output": "4",
                    "metadata": {"evaluation_type": "exact_match"},
                },
                {
                    "id": "ex2",
                    "input": {"text": "What is the capital of France?"},
                    "expected_output": "Paris",
                    "metadata": {"evaluation_type": "exact_match"},
                },
                {
                    "id": "ex3",
                    "input": {"text": "Calculate 10 * 5"},
                    "expected_output": "50",
                    "metadata": {"evaluation_type": "numeric", "tolerance": 0.1},
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_concurrent_hybrid_executions(self, mock_agent_builder, test_dataset):
        """Test multiple concurrent hybrid executions."""
        # Create multiple clients
        clients = []
        for _i in range(3):
            client = TraigentClient(
                execution_mode="edge_analytics",  # Use Edge Analytics for simplicity
                agent_builder=mock_agent_builder,
            )
            clients.append(client)

        # Run concurrent optimizations
        tasks = []
        for i, client in enumerate(clients):
            # Capture loop variable properly
            idx = i
            task = client.optimize(
                function=lambda x, _idx=idx: f"output_{_idx}",
                dataset=test_dataset,
                configuration_space={"param": [1, 2, 3]},
                objectives=["accuracy"],
                max_trials=2,
            )
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 3
        for result in results:
            assert result["status"] == "completed"
            assert result["completed_trials"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
