"""Integration tests for apply_best_config and get_optimization_insights across all modes."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from traigent import get_optimization_insights
from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ConfigurationError


class TestApplyBestConfigIntegration:
    """Integration tests for apply_best_config functionality across all modes."""

    @pytest.fixture(autouse=True)
    def dataset_root(self, monkeypatch, tmp_path_factory):
        """Ensure datasets are written under the trusted root."""
        root = tmp_path_factory.mktemp("integration_datasets")
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(root))
        return Path(root)

    @pytest.fixture
    def sample_dataset_file(self, dataset_root: Path):
        """Create temporary dataset file."""
        dataset_path = dataset_root / "sample_dataset.jsonl"
        dataset_path.write_text(
            "\n".join(
                [
                    '{"input": {"text": "hello world"}, "output": "HELLO WORLD"}',
                    '{"input": {"text": "good morning"}, "output": "GOOD MORNING"}',
                    '{"input": {"text": "test message"}, "output": "TEST MESSAGE"}',
                ]
            ),
            encoding="utf-8",
        )

        yield str(dataset_path)
        dataset_path.unlink(missing_ok=True)

    @pytest.fixture
    def sample_optimization_result(self):
        """Create comprehensive optimization result."""
        trials = [
            TrialResult(
                trial_id="trial_1",
                config={"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 200},
                metrics={"accuracy": 0.82, "cost_per_1k": 0.002, "latency": 0.4},
                status=TrialStatus.COMPLETED,
                duration=1.8,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_2",
                config={"model": "GPT-4o", "temperature": 0.1, "max_tokens": 150},
                metrics={"accuracy": 0.94, "cost_per_1k": 0.008, "latency": 0.7},
                status=TrialStatus.COMPLETED,
                duration=2.3,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_3",
                config={"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 300},
                metrics={"accuracy": 0.78, "cost_per_1k": 0.003, "latency": 0.5},
                status=TrialStatus.COMPLETED,
                duration=2.1,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        return OptimizationResult(
            trials=trials,
            best_config={"model": "GPT-4o", "temperature": 0.1, "max_tokens": 150},
            best_score=0.94,
            optimization_id="integration_test",
            duration=15.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost_per_1k", "latency"],
            algorithm="bayesian",
            timestamp=datetime.now(),
            metadata={},
        )

    def test_local_mode_complete_workflow(
        self, sample_dataset_file, sample_optimization_result
    ):
        """Test complete apply_best_config workflow in Edge Analytics mode."""

        def test_function(
            text: str,
            model: str = "default",
            temperature: float = 0.5,
            max_tokens: int = 100,
        ) -> str:
            # Simulate model behavior
            if model == "GPT-4o":
                pass
            else:
                pass

            1.0 - (temperature * 0.1)  # Lower temp = better performance
            return f"{model}({temperature},{max_tokens}):{text.upper()}"

        # Create optimized function in Edge Analytics mode with parameter injection
        opt_func = OptimizedFunction(
            func=test_function,
            config_space={
                "model": ["gpt-4o-mini", "GPT-4o"],
                "temperature": [0.1, 0.3, 0.7],
                "max_tokens": [100, 150, 200, 300],
            },
            objectives=["accuracy", "cost_per_1k", "latency"],
            use_cloud_service=False,
            eval_dataset=sample_dataset_file,
            injection_mode="seamless",  # This will pass config as kwargs
        )

        # Store optimization results
        opt_func._optimization_results = sample_optimization_result

        # Step 1: Apply best configuration
        apply_result = opt_func.apply_best_config()
        assert apply_result is True
        assert opt_func._current_config == sample_optimization_result.best_config

        # Step 2: Test that function works after config is applied
        result = opt_func("test input")
        # The result format depends on how the config is injected
        # Just verify the function executes successfully
        assert isinstance(result, str)
        assert "TEST INPUT" in result.upper()

        # Step 3: Generate insights
        insights = get_optimization_insights(sample_optimization_result)

        # Verify insights structure
        assert "top_configurations" in insights
        assert "performance_summary" in insights
        assert "parameter_insights" in insights
        assert "recommendations" in insights

        # Verify insights content
        assert len(insights["top_configurations"]) == 3
        assert insights["performance_summary"]["best_score"] == 0.94
        assert "model" in insights["parameter_insights"]
        assert len(insights["recommendations"]) > 0

    def test_hybrid_mode_complete_workflow(
        self, sample_dataset_file, sample_optimization_result
    ):
        """Test complete workflow in standard mode with cloud service mocks."""

        def test_function(
            text: str, model: str = "default", temperature: float = 0.5
        ) -> str:
            return f"hybrid:{model}:{temperature}:{text.upper()}"

        # Mock cloud service
        mock_cloud_client = Mock()
        mock_cloud_client.__aenter__ = Mock(return_value=mock_cloud_client)
        mock_cloud_client.__aexit__ = Mock(return_value=None)

        with patch(
            "traigent.cloud.client.TraiGentCloudClient", return_value=mock_cloud_client
        ):
            # Create optimized function in standard mode
            opt_func = OptimizedFunction(
                func=test_function,
                config_space={
                    "model": ["gpt-4o-mini", "GPT-4o"],
                    "temperature": [0.1, 0.5, 0.9],
                },
                objectives=["accuracy", "latency"],
                execution_mode="cloud",
                use_cloud_service=True,
                eval_dataset=sample_dataset_file,
            )

            # Store optimization results
            opt_func._optimization_results = sample_optimization_result

            # Apply best configuration
            apply_result = opt_func.apply_best_config()
            assert apply_result is True

            # Test insights generation
            insights = get_optimization_insights(sample_optimization_result)
            assert "error" not in insights
            assert insights["performance_summary"]["total_trials"] == 3

    def test_cloud_mode_complete_workflow(
        self, sample_dataset_file, sample_optimization_result
    ):
        """Test complete workflow in SaaS mode with server mocks."""

        def test_function(text: str, model: str = "default") -> str:
            return f"saas:{model}:{text.upper()}"

        # Mock cloud orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.optimize = Mock()

        async def mock_optimize(*args, **kwargs):
            return sample_optimization_result

        mock_orchestrator.optimize = mock_optimize

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator",
            return_value=mock_orchestrator,
        ):
            # Create optimized function in SaaS mode
            opt_func = OptimizedFunction(
                func=test_function,
                config_space={
                    "model": ["gpt-4o-mini", "GPT-4o"],
                },
                objectives=["accuracy"],
                execution_mode="cloud",
                use_cloud_service=False,  # SaaS mode
                eval_dataset=sample_dataset_file,
            )

            # Store optimization results
            opt_func._optimization_results = sample_optimization_result

            # Apply and test
            apply_result = opt_func.apply_best_config()
            assert apply_result is True

            # Generate insights
            insights = get_optimization_insights(sample_optimization_result)
            assert insights["performance_summary"]["successful_trials"] == 3

    def test_error_handling_integration(self, sample_dataset_file):
        """Test error handling across the complete workflow."""

        def test_function(text: str) -> str:
            return text.upper()

        opt_func = OptimizedFunction(
            func=test_function,
            config_space={"temperature": [0.1, 0.5, 0.9]},
            objectives=["accuracy"],
            eval_dataset=sample_dataset_file,
        )

        # Test apply_best_config with no results
        with pytest.raises((ValueError, RuntimeError, ConfigurationError)):
            opt_func.apply_best_config()

        # Test insights with no results
        insights = get_optimization_insights(None)
        assert "error" in insights
        assert insights["error"] == "No optimization results available"

        # Test insights with empty result
        empty_result = OptimizationResult(
            trials=[],
            best_config=None,
            best_score=0.0,
            optimization_id="empty",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        insights = get_optimization_insights(empty_result)
        assert "error" in insights
        assert insights["error"] == "No optimization results available"

    def test_cross_mode_consistency(
        self, sample_dataset_file, sample_optimization_result
    ):
        """Test that apply_best_config behaves consistently across all modes."""

        def test_function(text: str, model: str = "default") -> str:
            return f"{model}:{text.upper()}"

        modes = [
            {
                "execution_mode": "edge_analytics",
                "use_cloud_service": False,
                "name": "edge_analytics",
            },
            {"execution_mode": "cloud", "use_cloud_service": False, "name": "cloud"},
            {"execution_mode": "cloud", "use_cloud_service": True, "name": "hybrid"},
        ]

        results = {}

        for mode_config in modes:
            opt_func = OptimizedFunction(
                func=test_function,
                config_space={"model": ["gpt-4o-mini", "GPT-4o"]},
                objectives=["accuracy"],
                execution_mode=mode_config["execution_mode"],
                use_cloud_service=mode_config["use_cloud_service"],
                eval_dataset=sample_dataset_file,
            )

            # Store same optimization results
            opt_func._optimization_results = sample_optimization_result

            # Apply configuration
            apply_result = opt_func.apply_best_config()
            results[mode_config["name"]] = {
                "apply_result": apply_result,
                "config": opt_func._current_config.copy(),
            }

        # All modes should succeed
        for mode_name, result in results.items():
            assert result["apply_result"] is True, f"Mode {mode_name} failed"
            assert (
                result["config"] == sample_optimization_result.best_config
            ), f"Mode {mode_name} config mismatch"

        # Generate insights (should be identical regardless of mode)
        insights = get_optimization_insights(sample_optimization_result)
        assert "error" not in insights
        assert insights["performance_summary"]["best_score"] == 0.94

    def test_real_world_simulation(
        self, sample_dataset_file, sample_optimization_result
    ):
        """Test realistic usage pattern: optimize -> apply -> use -> insights."""

        def customer_support_agent(
            query: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7
        ) -> str:
            """Simulate a customer support agent."""
            # Simulate different model behaviors
            if model == "GPT-4o":
                quality = "high"
                speed = "medium"
            elif model == "gpt-4o-mini":
                quality = "good"
                speed = "fast"
            else:
                quality = "medium"
                speed = "fast"

            return f"[{quality}|{speed}] Response to: {query}"

        # Step 1: Create optimized function
        opt_func = OptimizedFunction(
            func=customer_support_agent,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini", "GPT-4o"],
                "temperature": [0.1, 0.5, 0.9],
            },
            objectives=["accuracy", "cost_per_1k", "latency"],
            eval_dataset=sample_dataset_file,
        )

        # Step 2: Simulate optimization results
        opt_func._optimization_results = sample_optimization_result

        # Step 3: Apply best configuration
        success = opt_func.apply_best_config()
        assert success is True
        assert opt_func._current_config["model"] == "GPT-4o"
        assert opt_func._current_config["temperature"] == 0.1

        # Step 4: Use optimized function
        response = opt_func("What's your return policy?")
        # Just verify the function works - the response format depends on injection mode
        assert isinstance(response, str)
        assert "Response to:" in response
        assert "return policy" in response.lower()

        # Step 5: Get business insights
        insights = get_optimization_insights(sample_optimization_result)

        # Verify business value insights
        top_configs = insights["top_configurations"]
        assert len(top_configs) == 3

        best_config = top_configs[0]
        assert best_config["rank"] == 1
        assert best_config["score"] == 0.94
        assert "cost_analysis" in best_config

        # Performance summary should show improvement
        perf_summary = insights["performance_summary"]
        assert perf_summary["best_score"] > perf_summary["worst_score"]
        assert perf_summary["improvement"] > 0

        # Should have actionable recommendations
        recommendations = insights["recommendations"]
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    @pytest.mark.asyncio
    async def test_async_workflow_integration(
        self, sample_dataset_file, sample_optimization_result
    ):
        """Test integration with async optimization workflows."""

        def async_compatible_function(text: str, model: str = "default") -> str:
            return f"async:{model}:{text}"

        # Mock async orchestrator
        mock_orchestrator = Mock()

        async def mock_optimize(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate async work
            return sample_optimization_result

        mock_orchestrator.optimize = mock_optimize

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator",
            return_value=mock_orchestrator,
        ):
            opt_func = OptimizedFunction(
                func=async_compatible_function,
                config_space={"model": ["gpt-4o-mini", "GPT-4o"]},
                objectives=["accuracy"],
                eval_dataset=sample_dataset_file,
            )

            # Simulate running optimization
            result = await opt_func.optimize()
            assert result is not None

            # Apply best config from async optimization
            apply_success = opt_func.apply_best_config(result)
            assert apply_success is True

            # Generate insights from async results
            insights = get_optimization_insights(result)
            assert "error" not in insights
            assert insights["performance_summary"]["total_trials"] == 3


class TestModeSpecificBehavior:
    """Test mode-specific behavior differences."""

    @pytest.fixture
    def sample_optimization_result(self):
        """Create comprehensive optimization result."""
        trials = [
            TrialResult(
                trial_id="trial_1",
                config={"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 200},
                metrics={"accuracy": 0.82, "cost_per_1k": 0.002, "latency": 0.4},
                status=TrialStatus.COMPLETED,
                duration=1.8,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_2",
                config={"model": "GPT-4o", "temperature": 0.1, "max_tokens": 150},
                metrics={"accuracy": 0.94, "cost_per_1k": 0.008, "latency": 0.7},
                status=TrialStatus.COMPLETED,
                duration=2.3,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        return OptimizationResult(
            trials=trials,
            best_config={"model": "GPT-4o", "temperature": 0.1, "max_tokens": 150},
            best_score=0.94,
            optimization_id="mode_test",
            duration=15.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost_per_1k", "latency"],
            algorithm="bayesian",
            timestamp=datetime.now(),
            metadata={},
        )

    def test_local_mode_privacy_preservation(self, sample_optimization_result):
        """Test that Edge Analytics mode preserves privacy expectations."""

        def sensitive_function(data: str, model: str = "default") -> str:
            # This would contain sensitive business logic
            return f"processed:{model}:{data}"

        opt_func = OptimizedFunction(
            func=sensitive_function,
            config_space={"model": ["gpt-4o-mini", "GPT-4o"]},
            objectives=["accuracy"],
            use_cloud_service=False,
        )

        opt_func._optimization_results = sample_optimization_result

        # Edge Analytics mode should work without any cloud dependencies
        success = opt_func.apply_best_config()
        assert success is True

        # Function should execute locally
        result = opt_func("sensitive data")
        # Just verify the function works locally without cloud dependencies
        assert isinstance(result, str)
        assert "sensitive data" in result

    def test_cloud_mode_efficiency_features(self, sample_optimization_result):
        """Test that cloud modes enable efficiency features."""

        def cloud_optimized_function(text: str, model: str = "default") -> str:
            return f"cloud:{model}:{text}"

        # Mock cloud client for standard mode
        mock_client = Mock()
        mock_client.__aenter__ = Mock(return_value=mock_client)
        mock_client.__aexit__ = Mock(return_value=None)

        with patch(
            "traigent.cloud.client.TraiGentCloudClient", return_value=mock_client
        ):
            opt_func = OptimizedFunction(
                func=cloud_optimized_function,
                config_space={"model": ["gpt-4o-mini", "GPT-4o"]},
                objectives=["accuracy"],
                execution_mode="cloud",
                use_cloud_service=True,  # Standard mode
            )

            opt_func._optimization_results = sample_optimization_result

            # Should work with cloud optimizations
            success = opt_func.apply_best_config()
            assert success is True

            # Insights should reflect cloud capabilities
            insights = get_optimization_insights(sample_optimization_result)
            assert "error" not in insights

    def test_insights_mode_independence(self, sample_optimization_result):
        """Test that insights generation is independent of execution mode."""

        # Generate insights multiple times
        insights1 = get_optimization_insights(sample_optimization_result)
        insights2 = get_optimization_insights(sample_optimization_result)

        # Should be identical
        assert insights1 == insights2

        # Should contain all expected sections
        for insights in [insights1, insights2]:
            assert "top_configurations" in insights
            assert "performance_summary" in insights
            assert "parameter_insights" in insights
            assert "recommendations" in insights
            assert "error" not in insights
