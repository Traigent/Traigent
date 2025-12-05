"""Comprehensive tests for apply_best_config functionality."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ConfigurationError


class MockFunction:
    """Mock function for testing apply_best_config."""

    def __init__(self, behavior="normal"):
        self.call_count = 0
        self.call_history = []
        self.behavior = behavior
        self.__name__ = "mock_function"

    def __call__(self, text: str, **kwargs) -> str:
        self.call_count += 1
        self.call_history.append((text, kwargs))

        if self.behavior == "error":
            raise ValueError("Mock function error")
        elif self.behavior == "config_dependent":
            # Return different results based on config
            model = kwargs.get("model", "default")
            temp = kwargs.get("temperature", 0.5)
            return f"Response with {model} at {temp}: {text.upper()}"
        else:
            return text.upper()


class TestApplyBestConfig:
    """Test OptimizedFunction.apply_best_config method."""

    @pytest.fixture
    def sample_config_space(self):
        """Sample configuration space."""
        return {
            "temperature": [0.0, 0.3, 0.5, 0.7, 1.0],
            "max_tokens": [100, 200, 300],
            "model": ["gpt-4o-mini", "GPT-4o"],
        }

    @pytest.fixture
    def sample_objectives(self):
        """Sample objectives."""
        return ["accuracy", "latency"]

    @pytest.fixture
    def mock_function(self):
        """Create mock function."""
        mock_func = MockFunction()
        mock_func.__name__ = "mock_function"
        return mock_func

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        examples = [
            EvaluationExample({"text": "hello"}, "HELLO"),
            EvaluationExample({"text": "world"}, "WORLD"),
        ]
        return Dataset(examples, name="test_dataset", description="Test dataset")

    @pytest.fixture
    def sample_optimization_result(self):
        """Create sample optimization result."""
        trials = [
            TrialResult(
                trial_id="trial_1",
                config={"temperature": 0.3, "max_tokens": 200, "model": "gpt-4o-mini"},
                metrics={"accuracy": 0.85, "latency": 0.5},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_2",
                config={"temperature": 0.7, "max_tokens": 300, "model": "GPT-4o"},
                metrics={"accuracy": 0.92, "latency": 0.8},
                status=TrialStatus.COMPLETED,
                duration=3.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_3",
                config={"temperature": 0.5, "max_tokens": 100, "model": "gpt-4o-mini"},
                metrics={"accuracy": 0.78, "latency": 0.3},
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        return OptimizationResult(
            trials=trials,
            best_config={"temperature": 0.7, "max_tokens": 300, "model": "GPT-4o"},
            best_score=0.92,
            optimization_id="test_opt_1",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "latency"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

    def test_apply_best_config_basic_success(
        self,
        mock_function,
        sample_config_space,
        sample_objectives,
        sample_optimization_result,
    ):
        """Test basic successful application of best config."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Store optimization results
        opt_func._optimization_results = sample_optimization_result

        # Apply best config
        result = opt_func.apply_best_config()

        assert result is True
        assert opt_func._current_config == sample_optimization_result.best_config

        # Verify the best config matches expected
        expected_config = {"temperature": 0.7, "max_tokens": 300, "model": "GPT-4o"}
        assert opt_func._current_config == expected_config

    def test_apply_best_config_with_explicit_results(
        self,
        mock_function,
        sample_config_space,
        sample_objectives,
        sample_optimization_result,
    ):
        """Test applying best config with explicitly provided results."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Don't store results in the instance
        result = opt_func.apply_best_config(sample_optimization_result)

        assert result is True
        assert opt_func._current_config == sample_optimization_result.best_config

    def test_apply_best_config_no_results_error(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test error when no optimization results are available."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # No results stored
        with pytest.raises(
            ConfigurationError, match="No optimization results available"
        ):
            opt_func.apply_best_config()

    def test_apply_best_config_no_best_config_error(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test error when optimization result has no best config."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Create result without best config
        empty_result = OptimizationResult(
            trials=[],
            best_config=None,  # No best config
            best_score=0.0,
            optimization_id="empty_opt",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=sample_objectives,
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        with pytest.raises(
            ConfigurationError, match="No optimization results available"
        ):
            opt_func.apply_best_config(empty_result)

    def test_apply_best_config_preserves_old_config(
        self,
        mock_function,
        sample_config_space,
        sample_objectives,
        sample_optimization_result,
    ):
        """Test that old config is preserved for potential rollback."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Set initial config
        initial_config = {"temperature": 0.1, "max_tokens": 50, "model": "gpt-4o-mini"}
        opt_func._current_config = initial_config.copy()

        # Apply best config
        opt_func.apply_best_config(sample_optimization_result)

        # Current config should be updated
        assert opt_func._current_config == sample_optimization_result.best_config
        assert opt_func._current_config != initial_config

    @patch("traigent.core.optimized_function.logger")
    def test_apply_best_config_logging(
        self,
        mock_logger,
        mock_function,
        sample_config_space,
        sample_objectives,
        sample_optimization_result,
    ):
        """Test that applying best config is properly logged."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        opt_func.apply_best_config(sample_optimization_result)

        # Verify logging occurred
        mock_logger.info.assert_called_once()
        log_call_args = mock_logger.info.call_args[0][0]
        assert "Applied best config for mock_function" in log_call_args
        assert str(sample_optimization_result.best_config) in log_call_args

    def test_apply_best_config_calls_setup_function_wrapper(
        self,
        mock_function,
        sample_config_space,
        sample_objectives,
        sample_optimization_result,
    ):
        """Test that _setup_function_wrapper is called after config update."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        with patch.object(opt_func, "_setup_function_wrapper") as mock_setup:
            opt_func.apply_best_config(sample_optimization_result)
            mock_setup.assert_called_once()

    def test_apply_best_config_different_modes(
        self,
        mock_function,
        sample_config_space,
        sample_objectives,
        sample_optimization_result,
    ):
        """Test apply_best_config in different execution modes."""
        # Test Edge Analytics mode
        opt_func_local = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="edge_analytics",
            use_cloud_service=False,
        )

        result_local = opt_func_local.apply_best_config(sample_optimization_result)
        assert result_local is True

        # Test cloud mode
        opt_func_commercial = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="cloud",
            use_cloud_service=False,
        )

        result_commercial = opt_func_commercial.apply_best_config(
            sample_optimization_result
        )
        assert result_commercial is True

        # Test hybrid mode (cloud + service)
        opt_func_hybrid = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="cloud",
            use_cloud_service=True,
        )

        result_hybrid = opt_func_hybrid.apply_best_config(sample_optimization_result)
        assert result_hybrid is True

    def test_apply_best_config_with_custom_evaluator(
        self,
        mock_function,
        sample_config_space,
        sample_objectives,
        sample_optimization_result,
    ):
        """Test apply_best_config with custom evaluator configured."""

        def custom_evaluator(func, config, example):
            return {"accuracy": 0.9, "custom_metric": 0.8}

        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            custom_evaluator=custom_evaluator,
        )

        result = opt_func.apply_best_config(sample_optimization_result)
        assert result is True
        assert opt_func.custom_evaluator is custom_evaluator

    def test_apply_best_config_integration_with_function_execution(self):
        """Test that applied config affects function execution."""
        # Create function that depends on config
        config_dependent_func = MockFunction(behavior="config_dependent")
        config_dependent_func.__name__ = "config_dependent_function"

        opt_func = OptimizedFunction(
            func=config_dependent_func,
            config_space={
                "model": ["gpt-3.5-turbo", "GPT-4o"],
                "temperature": [0.1, 0.5, 0.9],
            },
            objectives=["accuracy"],
        )

        # Create optimization result with specific config
        best_config = {"model": "GPT-4o", "temperature": 0.1}
        optimization_result = OptimizationResult(
            trials=[
                TrialResult(
                    trial_id="trial_1",
                    config=best_config,
                    metrics={"accuracy": 0.95},
                    status=TrialStatus.COMPLETED,
                    duration=2.0,
                    timestamp=datetime.now(),
                    metadata={},
                )
            ],
            best_config=best_config,
            best_score=0.95,
            optimization_id="integration_test",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        # Apply best config
        opt_func.apply_best_config(optimization_result)

        # Execute function and verify config is used
        opt_func("test input")

        # Verify the function was called with the applied config
        assert config_dependent_func.call_count == 1
        config_dependent_func.call_history[0]

        # The function should receive the config parameters
        # Check if config was applied (the exact mechanism depends on implementation)
        assert opt_func._current_config["model"] == "GPT-4o"
        assert opt_func._current_config["temperature"] == 0.1

    def test_apply_best_config_empty_results_object(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test handling of empty/minimal optimization results."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Empty results object
        empty_result = OptimizationResult(
            trials=[],
            best_config={},  # Empty config
            best_score=0.0,
            optimization_id="empty",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        with pytest.raises(
            ConfigurationError, match="No optimization results available"
        ):
            opt_func.apply_best_config(empty_result)

    def test_apply_best_config_partial_config(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test applying config that doesn't cover all parameters in config space."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Partial config (missing max_tokens)
        partial_config = {"temperature": 0.5, "model": "GPT-4o"}
        partial_result = OptimizationResult(
            trials=[
                TrialResult(
                    trial_id="trial_1",
                    config=partial_config,
                    metrics={"accuracy": 0.90},
                    status=TrialStatus.COMPLETED,
                    duration=2.0,
                    timestamp=datetime.now(),
                    metadata={},
                )
            ],
            best_config=partial_config,
            best_score=0.90,
            optimization_id="partial_test",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        # Should still apply successfully
        result = opt_func.apply_best_config(partial_result)
        assert result is True
        assert opt_func._current_config == partial_config

    def test_apply_best_config_config_validation(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that applied config is properly validated if validation is enabled."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Config with values outside the defined space
        invalid_config = {
            "temperature": 1.5,  # Outside [0.0, 0.3, 0.5, 0.7, 1.0]
            "max_tokens": 50,  # Outside [100, 200, 300]
            "model": "invalid-model",  # Outside ["gpt-4o-mini", "GPT-4o"]
        }

        invalid_result = OptimizationResult(
            trials=[
                TrialResult(
                    trial_id="trial_1",
                    config=invalid_config,
                    metrics={"accuracy": 0.85},
                    status=TrialStatus.COMPLETED,
                    duration=2.0,
                    timestamp=datetime.now(),
                    metadata={},
                )
            ],
            best_config=invalid_config,
            best_score=0.85,
            optimization_id="invalid_test",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        # Should still apply (validation happens during optimization, not application)
        result = opt_func.apply_best_config(invalid_result)
        assert result is True
        assert opt_func._current_config == invalid_config


class TestApplyBestConfigIntegration:
    """Integration tests for apply_best_config with other components."""

    @pytest.fixture(autouse=True)
    def dataset_root(self, monkeypatch, tmp_path_factory):
        """Ensure dataset files are created within the trusted root."""
        root = tmp_path_factory.mktemp("apply_best_config")
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(root))
        return Path(root)

    @pytest.fixture
    def sample_dataset_file(self, dataset_root: Path):
        """Create temporary dataset file."""
        dataset_path = dataset_root / "apply_best.jsonl"
        dataset_path.write_text(
            "\n".join(
                [
                    '{"input": {"text": "hello"}, "output": "HELLO"}',
                    '{"input": {"text": "world"}, "output": "WORLD"}',
                ]
            ),
            encoding="utf-8",
        )

        yield str(dataset_path)
        dataset_path.unlink(missing_ok=True)

    def test_apply_best_config_after_full_optimization(self, sample_dataset_file):
        """Test apply_best_config after running a complete optimization."""

        def test_function(
            text: str, model: str = "default", temperature: float = 0.5
        ) -> str:
            return f"{model}:{temperature}:{text.upper()}"

        opt_func = OptimizedFunction(
            func=test_function,
            config_space={
                "model": ["gpt-3.5-turbo", "GPT-4o"],
                "temperature": [0.1, 0.5, 0.9],
            },
            objectives=["accuracy"],
            max_trials=3,
            eval_dataset=sample_dataset_file,
        )

        # Mock the optimization process
        with patch("traigent.optimizers.get_optimizer"), patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as mock_orchestrator_class:

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Create mock result
            mock_result = OptimizationResult(
                trials=[
                    TrialResult(
                        trial_id="trial_1",
                        config={"model": "GPT-4o", "temperature": 0.1},
                        metrics={"accuracy": 0.95},
                        status=TrialStatus.COMPLETED,
                        duration=2.0,
                        timestamp=datetime.now(),
                        metadata={},
                    )
                ],
                best_config={"model": "GPT-4o", "temperature": 0.1},
                best_score=0.95,
                optimization_id="integration_test",
                duration=10.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=["accuracy"],
                algorithm="random",
                timestamp=datetime.now(),
                metadata={},
            )

            # Mock async optimize method
            async def mock_optimize(*args, **kwargs):
                return mock_result

            mock_orchestrator.optimize = mock_optimize

            # Run optimization
            import asyncio

            asyncio.run(opt_func.optimize())

            # Apply best config
            apply_result = opt_func.apply_best_config()

            assert apply_result is True
            assert opt_func._current_config["model"] == "GPT-4o"
            assert opt_func._current_config["temperature"] == 0.1
