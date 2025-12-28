"""Tests for cloud execution functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import traigent
from traigent.cloud.client import CloudOptimizationResult
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"text": "Test input 1"}, expected_output="output1"
        ),
        EvaluationExample(
            input_data={"text": "Test input 2"}, expected_output="output2"
        ),
        EvaluationExample(
            input_data={"text": "Test input 3"}, expected_output="output3"
        ),
    ]
    return Dataset(examples=examples, name="test_dataset")


class TestCloudExecutionDecorator:
    """Test cases for cloud execution mode in the decorator."""

    def test_decorator_with_cloud_execution(self):
        """Test decorator with execution_mode="cloud"."""

        @traigent.optimize(
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3]},
            execution_mode="cloud",
        )
        def test_function(input_text: str) -> str:
            return f"processed: {input_text}"

        assert isinstance(test_function, OptimizedFunction)
        assert test_function.execution_mode == "cloud"

    def test_decorator_with_edge_execution(self):
        """Test decorator with execution_mode="edge_analytics" (default)."""

        @traigent.optimize(
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3]},
        )
        def test_function(input_text: str) -> str:
            return f"processed: {input_text}"

        assert isinstance(test_function, OptimizedFunction)
        assert test_function.execution_mode == "edge_analytics"

    def test_decorator_execution_mode_parameter_passing(self):
        """Test that execution_mode parameter is properly passed through."""

        @traigent.optimize(
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3]},
            execution_mode="cloud",
            injection_mode="context",
        )
        def test_function(input_text: str) -> str:
            return f"processed: {input_text}"

        # Check that all parameters are set correctly
        assert test_function.execution_mode == "cloud"
        assert test_function.injection_mode == "context"
        assert test_function.objectives == ["accuracy"]


class TestOptimizedFunctionCloudMode:
    """Test cases for OptimizedFunction cloud execution."""

    def test_optimized_function_initialization_cloud_mode(self):
        """Test OptimizedFunction initialization with cloud execution."""

        def test_func(x: str) -> str:
            return x.upper()

        optimized_func = OptimizedFunction(
            func=test_func,
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3]},
            execution_mode="cloud",
        )

        assert optimized_func.execution_mode == "cloud"
        assert optimized_func._cloud_client is None  # Lazy initialization

    def test_optimized_function_initialization_edge_mode(self):
        """Test OptimizedFunction initialization with edge execution."""

        def test_func(x: str) -> str:
            return x.upper()

        optimized_func = OptimizedFunction(
            func=test_func,
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3]},
            execution_mode="edge_analytics",
        )

        assert optimized_func.execution_mode == "edge_analytics"

    def test_optimize_with_cloud_mode_success(self, sample_dataset):
        """Test optimization with cloud execution (successful cloud optimization)."""

        async def run_test():
            def test_func(input_data: dict) -> str:
                return f"processed: {input_data['text']}"

            optimized_func = OptimizedFunction(
                func=test_func,
                eval_dataset=sample_dataset,
                objectives=["accuracy"],
                configuration_space={"param": [1, 2, 3]},
                execution_mode="cloud",
            )

            # Mock the cloud optimization
            mock_cloud_result = CloudOptimizationResult(
                best_config={"param": 2},
                best_metrics={"accuracy": 0.85},
                trials_count=25,
                cost_reduction=0.65,
                optimization_time=120.0,
                subset_used=True,
                subset_size=2,
            )

            with patch.object(
                optimized_func, "_optimize_with_cloud_service", return_value=AsyncMock()
            ) as mock_cloud_opt:
                mock_cloud_opt.return_value = mock_cloud_result

                # This should call cloud optimization, not local
                result = await optimized_func.optimize(max_trials=50)

                # Verify cloud optimization was called
                mock_cloud_opt.assert_called_once()
                assert result == mock_cloud_result

        asyncio.run(run_test())

    def test_optimize_with_cloud_mode_fallback(self, sample_dataset):
        """Test optimization with cloud execution falling back to local."""

        async def run_test():
            def test_func(input_data: dict) -> str:
                return f"processed: {input_data['text']}"

            optimized_func = OptimizedFunction(
                func=test_func,
                eval_dataset=sample_dataset,
                objectives=["accuracy"],
                configuration_space={"param": [1, 2, 3]},
                execution_mode="cloud",
            )

            # Mock cloud optimization failure
            with patch.object(
                optimized_func,
                "_optimize_with_cloud_service",
                side_effect=Exception("Cloud service unavailable"),
            ):
                # Should fall back to local optimization
                result = await optimized_func.optimize(algorithm="random", max_trials=5)

                # Should still get a result from local optimization
                assert result is not None
                assert result.best_config is not None

        asyncio.run(run_test())

    def test_optimize_with_edge_mode(self, sample_dataset):
        """Test optimization with edge_analytics execution (local optimization)."""

        async def run_test():
            def test_func(input_data: dict) -> str:
                return f"processed: {input_data['text']}"

            optimized_func = OptimizedFunction(
                func=test_func,
                eval_dataset=sample_dataset,
                objectives=["accuracy"],
                configuration_space={"param": [1, 2, 3]},
                execution_mode="edge_analytics",
            )

            # Mock to ensure cloud optimization is not called
            with patch.object(
                optimized_func, "_optimize_with_cloud_service"
            ) as mock_cloud_opt:
                result = await optimized_func.optimize(algorithm="random", max_trials=5)

                # Cloud optimization should NOT be called
                mock_cloud_opt.assert_not_called()
                assert result is not None

        asyncio.run(run_test())

    def test_cloud_optimization_method(self, sample_dataset):
        """Test the cloud optimization method directly."""

        async def run_test():
            def test_func(input_data: dict) -> str:
                return f"processed: {input_data['text']}"

            optimized_func = OptimizedFunction(
                func=test_func,
                eval_dataset=sample_dataset,
                objectives=["accuracy"],
                configuration_space={"param": [1, 2, 3]},
                execution_mode="cloud",
            )

            # Mock the TraigentCloudClient
            mock_cloud_result = CloudOptimizationResult(
                best_config={"param": 2},
                best_metrics={"accuracy": 0.85},
                trials_count=25,
                cost_reduction=0.65,
                optimization_time=120.0,
                subset_used=True,
                subset_size=2,
            )

            mock_client = MagicMock()
            mock_client.optimize_function = AsyncMock(return_value=mock_cloud_result)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "traigent.cloud.client.TraigentCloudClient", return_value=mock_client
            ):
                result = await optimized_func._optimize_with_cloud_service(
                    dataset=sample_dataset, max_trials=50
                )

                # Check that result is properly converted
                assert result.best_config == {"param": 2}
                assert result.best_metrics == {"accuracy": 0.85}
                assert result.metadata["cloud_service"] is True
                assert result.metadata["cost_reduction"] == 0.65
                assert result.metadata["subset_used"] is True

        asyncio.run(run_test())

    def test_cloud_optimization_client_initialization(self, sample_dataset):
        """Test that cloud client is initialized lazily."""

        async def run_test():
            def test_func(input_data: dict) -> str:
                return f"processed: {input_data['text']}"

            optimized_func = OptimizedFunction(
                func=test_func,
                eval_dataset=sample_dataset,
                objectives=["accuracy"],
                configuration_space={"param": [1, 2, 3]},
                execution_mode="cloud",
            )

            # Initially cloud client should be None
            assert optimized_func._cloud_client is None

            # Mock the cloud client
            mock_client = MagicMock()
            mock_client.optimize_function = AsyncMock(
                return_value=CloudOptimizationResult(
                    best_config={},
                    best_metrics={},
                    trials_count=0,
                    cost_reduction=0.0,
                    optimization_time=0.0,
                    subset_used=False,
                )
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "traigent.cloud.client.TraigentCloudClient", return_value=mock_client
            ) as mock_client_class:
                await optimized_func._optimize_with_cloud_service(sample_dataset)

                # Cloud client should be initialized with proper parameters
                mock_client_class.assert_called_once_with(enable_fallback=True)

        asyncio.run(run_test())

        def test_func(x: str) -> str:
            return x.upper()

        # Should accept boolean values
        optimized_func = OptimizedFunction(
            func=test_func,
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3]},
            execution_mode="cloud",
        )
        assert optimized_func.execution_mode == "cloud"

        optimized_func = OptimizedFunction(
            func=test_func,
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3]},
            execution_mode="edge_analytics",
        )
        assert optimized_func.execution_mode == "edge_analytics"


class TestCloudExecutionIntegration:
    """Integration tests for cloud execution functionality."""

    def test_end_to_end_cloud_optimization(self, sample_dataset):
        """Test end-to-end cloud optimization flow."""

        async def run_test():
            @traigent.optimize(
                eval_dataset=sample_dataset,
                objectives=["accuracy"],
                configuration_space={
                    "strategy": ["upper", "lower"],
                    "add_prefix": [True, False],
                },
                execution_mode="cloud",
            )
            def text_processor(input_data: dict) -> str:
                config = traigent.get_config()
                text = input_data["text"]

                if config and config.get("strategy") == "upper":
                    text = text.upper()
                elif config and config.get("strategy") == "lower":
                    text = text.lower()

                if config and config.get("add_prefix"):
                    text = f"PROCESSED: {text}"

                return text

            # Mock cloud optimization to avoid actual network calls
            from datetime import datetime

            from traigent.api.types import (
                OptimizationResult,
                OptimizationStatus,
                TrialResult,
                TrialStatus,
            )

            mock_trial = TrialResult(
                trial_id="cloud_best",
                config={"strategy": "upper", "add_prefix": True},
                metrics={"accuracy": 0.9},
                status=TrialStatus.COMPLETED,
                duration=60.0,
                timestamp=datetime.now(),
                metadata={},
            )

            mock_optimization_result = OptimizationResult(
                trials=[mock_trial],
                best_config={"strategy": "upper", "add_prefix": True},
                best_score=0.9,
                optimization_id="cloud_test",
                duration=60.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=["accuracy"],
                algorithm="cloud_service",
                timestamp=datetime.now(),
                metadata={
                    "cloud_service": True,
                    "cost_reduction": 0.67,
                    "subset_used": True,
                    "subset_size": 2,
                    "trials_count": 15,
                },
            )

            with patch.object(
                text_processor,
                "_optimize_with_cloud_service",
                return_value=mock_optimization_result,
            ) as mock_cloud:
                result = await text_processor.optimize()

                # Verify cloud optimization was used
                mock_cloud.assert_called_once()

                # Check optimization results
                assert result.best_config == {"strategy": "upper", "add_prefix": True}
                assert result.best_metrics == {"accuracy": 0.9}
                assert result.metadata["cloud_service"] is True
                assert result.metadata["cost_reduction"] == 0.67

        asyncio.run(run_test())

    def test_cloud_execution_function_behavior(self, sample_dataset):
        """Test that function behavior is unchanged with cloud execution."""

        @traigent.optimize(
            eval_dataset=sample_dataset,
            objectives=["accuracy"],
            configuration_space={"strategy": ["upper", "lower"]},
            execution_mode="cloud",
            default_config={"strategy": "upper"},
        )
        def text_processor(input_data: dict) -> str:
            config = traigent.get_config()
            text = input_data["text"]

            if config and config.get("strategy") == "upper":
                return text.upper()
            else:
                return text.lower()

        # Function should work normally even with cloud execution enabled
        result = text_processor({"text": "hello world"})
        assert result == "HELLO WORLD"  # Using default config

    def test_cloud_execution_with_different_algorithms(self, sample_dataset):
        """Test cloud execution interaction with different algorithms."""

        async def run_test():
            def test_func(input_data: dict) -> str:
                return input_data["text"].upper()

            optimized_func = OptimizedFunction(
                func=test_func,
                eval_dataset=sample_dataset,
                objectives=["accuracy"],
                configuration_space={"param": [1, 2, 3]},
                execution_mode="cloud",
            )

            # Mock cloud optimization
            mock_cloud_result = CloudOptimizationResult(
                best_config={"param": 2},
                best_metrics={"accuracy": 0.85},
                trials_count=25,
                cost_reduction=0.65,
                optimization_time=120.0,
                subset_used=True,
                subset_size=2,
            )

            with patch.object(
                optimized_func,
                "_optimize_with_cloud_service",
                return_value=mock_cloud_result,
            ) as mock_cloud:

                # Test with different algorithm parameters
                await optimized_func.optimize(
                    algorithm="bayesian",  # This should be ignored in cloud mode
                    max_trials=100,
                )

                # Cloud optimization should be called regardless of algorithm
                mock_cloud.assert_called_once()
                args, kwargs = mock_cloud.call_args
                # Check that max_trials was passed as positional or keyword argument
                if len(args) >= 2:
                    assert args[1] == 100  # max_trials as positional argument
                else:
                    assert (
                        kwargs.get("max_trials") == 100
                    )  # max_trials as keyword argument

        asyncio.run(run_test())

    def test_cloud_execution_configuration_injection(self, sample_dataset):
        """Test that configuration injection still works with cloud execution."""

        @traigent.optimize(
            eval_dataset=sample_dataset,
            objectives=["accuracy"],
            configuration_space={"multiplier": [1, 2, 3]},
            execution_mode="cloud",
            injection_mode="context",
            default_config={"multiplier": 2},
        )
        def multiply_text_length(input_data: dict) -> str:
            config = traigent.get_config()
            multiplier = config.get("multiplier", 1) if config else 1
            text = input_data["text"]
            return text * multiplier

        # Test with default config
        result = multiply_text_length({"text": "hi"})
        assert result == "hihi"  # multiplier = 2

        # Test manual config setting
        multiply_text_length.set_config({"multiplier": 3})
        result = multiply_text_length({"text": "hi"})
        assert result == "hihihi"  # multiplier = 3
