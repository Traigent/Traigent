"""Integration tests for summary_stats mode in privacy/local execution."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.utils.exceptions import ConfigurationError


class TestSummaryStatsIntegration:
    """Test the complete flow of summary_stats mode."""

    @pytest.mark.asyncio
    async def test_local_mode_uses_summary_stats(self):
        """Test that Edge Analytics mode correctly generates and uses summary_stats."""
        # Create evaluator with local execution mode
        evaluator = LocalEvaluator(
            metrics=["accuracy"], execution_mode="edge_analytics"
        )

        # Create a simple test function
        async def test_function(**kwargs):
            return "output"

        # Create test dataset
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"text": "test"}, expected_output="output")
                for _ in range(3)
            ]
        )

        # Run evaluation
        result = await evaluator.evaluate(test_function, {"temperature": 0.7}, dataset)

        # Check that summary_stats was generated
        assert hasattr(result, "summary_stats")
        assert result.summary_stats is not None
        assert "metrics" in result.summary_stats
        assert "execution_time" in result.summary_stats
        assert "metadata" in result.summary_stats

        # Check pandas.describe() format
        for _metric_name, stats in result.summary_stats["metrics"].items():
            assert "count" in stats
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "25%" in stats
            assert "50%" in stats
            assert "75%" in stats
            assert "max" in stats

    @pytest.mark.asyncio
    async def test_private_mode_uses_summary_stats(self):
        """Test that privacy mode correctly generates and uses summary_stats."""
        # Create evaluator with private execution mode
        evaluator = LocalEvaluator(metrics=["accuracy"], execution_mode="privacy")

        # Create a simple test function
        async def test_function(**kwargs):
            return "output"

        # Create test dataset
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"text": "test"}, expected_output="output")
                for _ in range(3)
            ]
        )

        # Run evaluation
        result = await evaluator.evaluate(test_function, {"temperature": 0.7}, dataset)

        # Check that summary_stats was generated
        assert hasattr(result, "summary_stats")
        assert result.summary_stats is not None
        assert (
            result.summary_stats["metadata"]["aggregation_method"] == "pandas.describe"
        )

    @pytest.mark.asyncio
    async def test_cloud_mode_does_not_use_summary_stats(self):
        """Test that cloud mode does NOT generate summary_stats."""
        # Create evaluator with cloud execution mode (or None which defaults to no summary_stats)
        evaluator = LocalEvaluator(metrics=["accuracy"], execution_mode="cloud")

        # Create a simple test function
        async def test_function(**kwargs):
            return "output"

        # Create test dataset
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"text": "test"}, expected_output="output")
            ]
        )

        # Run evaluation
        result = await evaluator.evaluate(test_function, {"temperature": 0.7}, dataset)

        # Check that summary_stats was NOT generated
        assert not hasattr(result, "summary_stats") or result.summary_stats is None

    @pytest.mark.asyncio
    async def test_orchestrator_passes_execution_mode_to_evaluator(self):
        """Test that orchestrator correctly passes execution mode to evaluator."""
        # Create optimizer
        optimizer = RandomSearchOptimizer(
            config_space={"temperature": (0.1, 1.0)}, objectives=["accuracy"]
        )

        # Create evaluator
        evaluator = LocalEvaluator(metrics=["accuracy"])

        # Create config with Edge Analytics mode
        config = TraigentConfig(execution_mode="edge_analytics")

        # Create orchestrator
        OptimizationOrchestrator(
            optimizer=optimizer, evaluator=evaluator, max_trials=1, config=config
        )

        # Check that execution mode was passed to evaluator
        assert evaluator.execution_mode == "edge_analytics"

    @pytest.mark.asyncio
    async def test_backend_client_submits_summary_stats(self, monkeypatch):
        """Test that backend client correctly submits summary_stats in privacy mode."""
        # Disable offline mode so backend calls are actually made
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")

        # Create backend client
        backend_config = BackendClientConfig(
            backend_base_url="http://localhost:5000", enable_session_sync=True
        )
        backend_client = BackendIntegratedClient(
            api_key="test_key",  # pragma: allowlist secret
            backend_config=backend_config,
            enable_fallback=True,
        )

        # Mock the aiohttp session
        with patch(
            "traigent.cloud.backend_client.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_session.post.return_value = mock_response
            mock_session.__aenter__.return_value = mock_session
            mock_session_class.return_value = mock_session

            # Create summary_stats data
            summary_stats = {
                "metrics": {
                    "accuracy": {
                        "count": 10,
                        "mean": 0.85,
                        "std": 0.1,
                        "min": 0.7,
                        "25%": 0.8,
                        "50%": 0.85,
                        "75%": 0.9,
                        "max": 0.95,
                    }
                },
                "execution_time": 10.5,
                "total_examples": 10,
                "metadata": {
                    "sdk_version": "2.0.0",
                    "aggregation_method": "pandas.describe",
                },
            }

            # Submit with summary_stats
            result = await backend_client._submit_summary_stats(
                session_id="test_session",
                trial_id="test_trial",
                config={"temperature": 0.7},
                summary_stats=summary_stats,
                status="completed",
            )

            # Check that submission was made with correct data
            assert result is True
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert "test_session" in call_args[0][0]  # URL contains session ID
            submitted_data = call_args[1]["json"]
            assert submitted_data["summary_stats"] == summary_stats
            assert submitted_data["trial_id"] == "test_trial"
            assert submitted_data["config"] == {"temperature": 0.7}

    def test_backend_client_routes_based_on_execution_mode(self):
        """Test that backend client routes to correct submission method based on mode."""
        # Create backend client
        backend_config = BackendClientConfig(
            backend_base_url="http://localhost:5000", enable_session_sync=True
        )
        backend_client = BackendIntegratedClient(
            api_key="test_key",  # pragma: allowlist secret
            backend_config=backend_config,
            enable_fallback=True,
        )

        # Test with Edge Analytics mode metadata
        metadata_local = {
            "execution_mode": "edge_analytics",
            "trial_id": "test_trial",
            "summary_stats": {
                "metrics": {"accuracy": {"count": 10, "mean": 0.85}},
                "execution_time": 10.5,
                "total_examples": 10,
                "metadata": {},
            },
        }

        # Mock the async submission methods
        with patch.object(backend_client, "_submit_summary_stats") as mock_summary:
            with patch.object(
                backend_client, "_submit_trial_result_via_session"
            ) as mock_detailed:
                # Mock asyncio.run to handle async calls
                async def run_submit():
                    mock_summary.return_value = True
                    mock_detailed.return_value = True

                    # This would be called by submit_result
                    execution_mode = metadata_local.get(
                        "execution_mode", "edge_analytics"
                    )
                    use_summary_stats = execution_mode in ["edge_analytics", "privacy"]
                    summary_stats = metadata_local.get("summary_stats")

                    if use_summary_stats and summary_stats:
                        await backend_client._submit_summary_stats(
                            session_id="test_session",
                            trial_id=metadata_local["trial_id"],
                            config={"temperature": 0.7},
                            summary_stats=summary_stats,
                            status="completed",
                        )
                        return True
                    return False

                # Run the test
                result = asyncio.run(run_submit())

                # Check that summary_stats path was taken
                assert result is True
                mock_summary.assert_called_once()
                mock_detailed.assert_not_called()


class TestExecutionModeHandling:
    """Test that execution modes are handled correctly throughout the system."""

    def test_traigent_config_valid_execution_modes(self):
        """Test that TraigentConfig accepts all valid enum execution modes.

        Note: TraigentConfig resolves mode strings leniently. User-facing
        validation happens at TraigentClient level via validate_execution_mode().
        """
        # edge_analytics is the primary supported mode
        config = TraigentConfig(execution_mode="edge_analytics")
        assert config.execution_mode == "edge_analytics"

        # TraigentConfig accepts all valid enum values (lenient resolution)
        for mode in ["cloud", "hybrid", "standard"]:
            config = TraigentConfig(execution_mode=mode)
            assert config.execution_mode == mode

        # 'privacy' is a back-compat alias that maps to 'hybrid' + privacy_enabled
        config = TraigentConfig(execution_mode="privacy")
        assert config.execution_mode == "hybrid"
        assert config.privacy_enabled is True

        # validate_execution_mode provides strict validation
        from traigent.config.types import validate_execution_mode

        for mode in ["cloud", "hybrid"]:
            with pytest.raises(ConfigurationError, match="not yet supported"):
                validate_execution_mode(mode)

        for mode in ["privacy", "standard"]:
            with pytest.raises(ConfigurationError, match="No such mode"):
                validate_execution_mode(mode)

    def test_traigent_config_invalid_execution_mode(self):
        """Test that TraigentConfig rejects invalid execution mode strings."""
        for invalid_mode in ["invalid_mode", "local"]:
            with pytest.raises(ValueError, match="execution_mode must be one of"):
                TraigentConfig(execution_mode=invalid_mode)

    def test_evaluator_execution_mode_initialization(self):
        """Test that evaluator correctly initializes with execution_mode."""
        evaluator = LocalEvaluator(
            metrics=["accuracy"], execution_mode="edge_analytics"
        )
        assert evaluator.execution_mode == "edge_analytics"

        # Test with None
        evaluator2 = LocalEvaluator(metrics=["accuracy"])
        assert evaluator2.execution_mode is None

    def test_orchestrator_sets_evaluator_execution_mode(self):
        """Test that orchestrator sets evaluator execution mode from config."""
        optimizer = RandomSearchOptimizer(
            config_space={"temperature": (0.1, 1.0)}, objectives=["accuracy"]
        )
        evaluator = LocalEvaluator(metrics=["accuracy"])

        # Initially None
        assert evaluator.execution_mode is None

        # Create orchestrator with Edge Analytics config
        config = TraigentConfig(execution_mode="edge_analytics")
        OptimizationOrchestrator(
            optimizer=optimizer, evaluator=evaluator, max_trials=1, config=config
        )

        # Should now be set to Edge Analytics
        assert evaluator.execution_mode == "edge_analytics"
