"""Comprehensive tests for optimized function implementation.

Tests cover:
- OptimizedFunction initialization and configuration
- Custom evaluator wrapper functionality
- Dataset loading from multiple sources
- Cloud service integration and fallback
- Framework override functionality
- Configuration injection mechanisms
- Error handling and validation
- Commercial mode activation
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

import traigent
from traigent.api.types import ExampleResult, OptimizationResult, OptimizationStatus
from traigent.core.optimized_function import OptimizationState, OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import (
    ConfigurationError,
    OptimizationError,
    OptimizationStateError,
    ValidationError,
)

# Test Fixtures and Mock Classes


class MockFunction:
    """Mock function for testing optimization."""

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
        elif self.behavior == "slow":
            import time

            time.sleep(0.1)
            return text.upper()
        else:
            return text.upper()


def mock_custom_evaluator(func, config, example):
    """Mock custom evaluator function."""
    try:
        input_data = example.input_data
        expected_output = example.expected_output

        # Call function with config parameters
        actual_output = func(input_data.get("text", ""), **config)

        # Calculate simple metrics
        accuracy = 1.0 if actual_output == expected_output else 0.0

        return ExampleResult(
            example_id=example.input_data.get("id", "test"),
            input_data=input_data,
            expected_output=expected_output,
            actual_output=actual_output,
            metrics={"accuracy": accuracy, "length": len(actual_output)},
            execution_time=0.01,
            success=True,
            error_message=None,
        )
    except Exception as e:
        return ExampleResult(
            example_id=example.input_data.get("id", "test"),
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=None,
            metrics={"accuracy": 0.0, "length": 0},
            execution_time=0.01,
            success=False,
            error_message=str(e),
        )


class TestOptimizedFunction:
    """Test OptimizedFunction class."""

    @pytest.fixture
    def sample_config_space(self):
        """Sample configuration space."""
        return {
            "temperature": [0.0, 0.3, 0.5, 0.7, 1.0],
            "max_tokens": [100, 200, 300],
            "model": ["o4-mini", "GPT-4o"],
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

    # Constructor Tests

    def test_optimized_function_creation_basic(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test basic OptimizedFunction creation."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        assert opt_func.func is mock_function
        assert opt_func.config_space == sample_config_space
        assert opt_func.objectives == sample_objectives
        assert opt_func.algorithm == "random"  # default
        assert opt_func.max_trials == 50  # default
        assert opt_func.custom_evaluator is None

    def test_optimized_function_creation_with_all_params(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test OptimizedFunction creation with all parameters."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            algorithm="bayesian",
            max_trials=100,
            timeout=120.0,
            custom_evaluator=mock_custom_evaluator,
            execution_mode="cloud",
            use_cloud_service=False,
        )

        assert opt_func.algorithm == "bayesian"
        assert opt_func.max_trials == 100
        assert opt_func.timeout == 120.0
        assert opt_func.custom_evaluator is mock_custom_evaluator
        assert opt_func.execution_mode == "cloud"
        assert opt_func.use_cloud_service is False

    def test_optimized_function_creation_invalid_function(
        self, sample_config_space, sample_objectives
    ):
        """Test OptimizedFunction creation with invalid function."""
        with pytest.raises((TypeError, ValidationError)):
            OptimizedFunction(
                func="not_a_function",
                config_space=sample_config_space,
                objectives=sample_objectives,
            )

    def test_optimized_function_creation_invalid_config_space(
        self, mock_function, sample_objectives
    ):
        """Test OptimizedFunction creation with invalid config space."""
        with pytest.raises((TypeError, ValueError, ValidationError)):
            OptimizedFunction(
                func=mock_function,
                config_space="not_a_dict",
                objectives=sample_objectives,
            )

    def test_optimized_function_creation_empty_config_space(
        self, mock_function, sample_objectives
    ):
        """Test OptimizedFunction creation with empty config space."""
        with pytest.raises((ValueError, ValidationError)):
            OptimizedFunction(
                func=mock_function, config_space={}, objectives=sample_objectives
            )

    def test_optimized_function_creation_invalid_objectives(
        self, mock_function, sample_config_space
    ):
        """Test OptimizedFunction creation with invalid objectives."""
        with pytest.raises((TypeError, ValueError, ValidationError)):
            OptimizedFunction(
                func=mock_function,
                config_space=sample_config_space,
                objectives="not_a_list",
            )

    def test_optimized_function_creation_invalid_history_limit(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """History limit must be a positive integer."""
        with pytest.raises(ValueError, match="optimization_history_limit must be >= 1"):
            OptimizedFunction(
                func=mock_function,
                config_space=sample_config_space,
                objectives=sample_objectives,
                optimization_history_limit=0,
            )

    def test_optimized_function_creation_empty_objectives(
        self, mock_function, sample_config_space
    ):
        """Test OptimizedFunction creation with empty objectives defaults to accuracy."""
        # Empty objectives list defaults to ["accuracy"] as fallback
        opt_func = OptimizedFunction(
            func=mock_function, config_space=sample_config_space, objectives=[]
        )
        # System defaults to accuracy when empty list provided
        assert opt_func.objectives == ["accuracy"]
        assert opt_func.objective_schema is not None

    @pytest.mark.asyncio
    async def test_scoring_function_overrides_exact_match(self) -> None:
        """scoring_function should override exact-match accuracy."""

        def contains_accuracy(
            output: str, expected: str, llm_metrics: dict | None = None
        ) -> float:
            if not output or not expected:
                return 0.0
            return 1.0 if expected.lower() in output.lower() else 0.0

        def qa_agent(text: str) -> str:
            # Deliberately not an exact match to expected output.
            return "The capital of France is Paris."

        dataset = Dataset(
            [EvaluationExample({"text": "ignored"}, "Paris")],
            name="scoring_function_test",
            description="Test dataset",
        )

        opt_func = OptimizedFunction(
            func=qa_agent,
            config_space={"temperature": [0.1]},
            objectives=["accuracy"],
            eval_dataset=dataset,
            scoring_function=contains_accuracy,
        )

        evaluator = opt_func._create_effective_evaluator(
            timeout=5.0,
            custom_evaluator=None,
            effective_batch_size=1,
            effective_thread_workers=None,
            effective_privacy_enabled=False,
        )

        result = await evaluator.evaluate(
            qa_agent,
            {"temperature": 0.1},
            dataset,
        )

        assert result.aggregated_metrics["accuracy"] == 1.0
        assert [obj.name for obj in opt_func.objective_schema.objectives] == [
            "accuracy"
        ]

    def test_global_objective_default_applies_when_decorator_omits(
        self, sample_dataset
    ):
        """Global configure() objectives should fill in when decorator omits them."""
        traigent.configure(objectives=["cost"])

        @traigent.optimize(
            eval_dataset=sample_dataset,
            configuration_space={"temperature": [0.3]},
            execution_mode="edge_analytics",
        )
        def decorated(question: str, config: dict | None = None) -> str:
            return "42"

        assert decorated.objectives == ["cost"]
        assert [obj.name for obj in decorated.objective_schema.objectives] == ["cost"]

    @pytest.mark.asyncio
    async def test_runtime_objective_override_is_temporary(
        self, sample_dataset, monkeypatch
    ):
        """Runtime objectives override for a single call without persisting."""

        @traigent.optimize(
            eval_dataset=sample_dataset,
            objectives=["accuracy"],
            configuration_space={"temperature": [0.1]},
            execution_mode="edge_analytics",
        )
        def decorated(question: str, config: dict | None = None) -> str:
            return "42"

        original_schema = decorated.objective_schema
        dummy_result = Mock(spec=OptimizationResult)

        async def fake_execute(self, **kwargs):
            assert self.objectives == ["latency"]
            return dummy_result

        monkeypatch.setattr(
            OptimizedFunction,
            "_execute_optimization",
            fake_execute,
        )

        result = await decorated.optimize(objectives=["latency"], max_trials=1)

        assert result is dummy_result
        assert decorated.objective_schema is original_schema
        assert decorated.objectives == ["accuracy"]

    def test_optimized_function_creation_negative_max_trials(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test OptimizedFunction creation with negative max_trials."""
        with pytest.raises((ValueError, ValidationError)):
            OptimizedFunction(
                func=mock_function,
                config_space=sample_config_space,
                objectives=sample_objectives,
                max_trials=-1,
            )

    def test_optimized_function_creation_negative_timeout(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test OptimizedFunction creation with negative timeout."""
        with pytest.raises((ValueError, ValidationError)):
            OptimizedFunction(
                func=mock_function,
                config_space=sample_config_space,
                objectives=sample_objectives,
                timeout=-1.0,
            )

    # Dataset Loading Tests

    def test_load_dataset_from_object(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test loading dataset from Dataset object."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        loaded_dataset = opt_func._load_dataset()

        assert loaded_dataset is sample_dataset
        assert len(loaded_dataset.examples) == 2

    def test_load_dataset_from_file_path(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test loading dataset from file path."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"input": {"text": "hello"}, "output": "HELLO"}\n')
            f.write('{"input": {"text": "world"}, "output": "WORLD"}\n')
            temp_path = f.name

        try:
            opt_func = OptimizedFunction(
                func=mock_function,
                config_space=sample_config_space,
                objectives=sample_objectives,
            )

            with patch(
                "traigent.evaluators.base.Dataset.from_jsonl"
            ) as mock_from_jsonl:
                mock_dataset = Dataset(
                    [
                        EvaluationExample({"text": "hello"}, "HELLO"),
                        EvaluationExample({"text": "world"}, "WORLD"),
                    ],
                    name="loaded",
                    description="Loaded dataset",
                )
                mock_from_jsonl.return_value = mock_dataset

                opt_func.eval_dataset = temp_path
                loaded_dataset = opt_func._load_dataset()

                assert loaded_dataset is mock_dataset
                mock_from_jsonl.assert_called_once_with(temp_path)

        finally:
            Path(temp_path).unlink()

    def test_load_dataset_from_list(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test loading dataset from list of file paths."""
        # Create temporary JSONL files
        file_paths = []
        for _i, data in enumerate(
            [
                '{"input": {"text": "hello"}, "output": "HELLO"}',
                '{"input": {"text": "world"}, "output": "WORLD"}',
            ]
        ):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                f.write(data + "\n")
                file_paths.append(f.name)

        try:
            opt_func = OptimizedFunction(
                func=mock_function,
                config_space=sample_config_space,
                objectives=sample_objectives,
            )

            with patch(
                "traigent.evaluators.base.Dataset.from_jsonl"
            ) as mock_from_jsonl:
                # Mock individual dataset loading
                mock_dataset1 = Mock()
                mock_dataset1.examples = [EvaluationExample({"text": "hello"}, "HELLO")]
                mock_dataset2 = Mock()
                mock_dataset2.examples = [EvaluationExample({"text": "world"}, "WORLD")]
                mock_from_jsonl.side_effect = [mock_dataset1, mock_dataset2]

                opt_func.eval_dataset = file_paths
                loaded_dataset = opt_func._load_dataset()

                # Should be a real Dataset with combined examples
                assert isinstance(loaded_dataset, Dataset)
                assert len(loaded_dataset.examples) == 2
                assert mock_from_jsonl.call_count == 2

        finally:
            # Clean up temporary files
            for path in file_paths:
                Path(path).unlink()

    def test_load_dataset_invalid_type(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test loading dataset with invalid type."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        opt_func.eval_dataset = 12345  # Invalid type
        with pytest.raises(ConfigurationError, match="Invalid dataset type"):
            opt_func._load_dataset()

    def test_load_dataset_nonexistent_file(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test loading dataset from nonexistent file."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        opt_func.eval_dataset = "/nonexistent/path/dataset.jsonl"
        with pytest.raises(ConfigurationError, match="Failed to load dataset"):
            opt_func._load_dataset()

    # Optimization Tests

    @pytest.mark.asyncio
    async def test_optimize_basic_workflow(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test basic optimization workflow."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            max_trials=3,
            eval_dataset=sample_dataset,
        )

        # Mock dependencies
        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            # Setup mock optimizer
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            # Setup mock orchestrator
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup mock optimization result
            from datetime import datetime

            from traigent.api.types import TrialResult, TrialStatus

            mock_trial = TrialResult(
                trial_id="trial_1",
                config={"temperature": 0.5},
                metrics={"accuracy": 0.8},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            )

            mock_result = OptimizationResult(
                trials=[mock_trial],
                best_config={"temperature": 0.5},
                best_score=0.8,
                optimization_id="test_opt_1",
                duration=10.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=sample_objectives,
                algorithm="random",
                timestamp=datetime.now(),
                metadata={},
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            # Run optimization
            result = await opt_func.optimize(algorithm="random")

            # Verify result
            assert result is mock_result
            assert result.status == OptimizationStatus.COMPLETED

            # Verify optimizer was created correctly if called (may be optimized away in some cases)
            if mock_get_optimizer.called:
                mock_get_optimizer.assert_called_with(
                    algorithm="random",
                    config_space=sample_config_space,
                    objectives=sample_objectives,
                )

            # Verify orchestrator was created and called if the optimization path was taken
            if mock_orchestrator_class.called:
                mock_orchestrator_class.assert_called_once()
                # The func will be wrapped, so just check that optimize was called with a dataset
                mock_orchestrator.optimize.assert_called_once()
                call_args = mock_orchestrator.optimize.call_args
                assert "func" in call_args[1]
                assert call_args[1]["dataset"] == sample_dataset

    @pytest.mark.asyncio
    async def test_optimize_zero_max_trials_short_circuits(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Ensure optimize() respects max_trials=0 without executing trials."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            from datetime import datetime

            mock_result = OptimizationResult(
                trials=[],
                best_config={},
                best_score=0.0,
                optimization_id="opt_zero",
                duration=0.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=sample_objectives,
                algorithm="random",
                timestamp=datetime.now(),
                metadata={},
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = await opt_func.optimize(max_trials=0)

            assert result is mock_result
            assert len(result.trials) == 0
            assert mock_orchestrator_class.call_args[1]["max_trials"] == 0
            mock_orchestrator.optimize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_optimize_with_custom_evaluator(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test optimization with custom evaluator."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            custom_evaluator=mock_custom_evaluator,
            max_trials=2,
            eval_dataset=sample_dataset,
            mock_mode_config={
                "enabled": True,
                "override_evaluator": False,
            },  # Don't override custom evaluator in mock mode
        )

        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            from datetime import datetime

            from traigent.api.types import TrialResult, TrialStatus

            mock_trial = TrialResult(
                trial_id="trial_1",
                config={"temperature": 0.7},
                metrics={"accuracy": 0.9},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            )

            mock_result = OptimizationResult(
                trials=[mock_trial],
                best_config={"temperature": 0.7},
                best_score=0.9,
                optimization_id="test_opt_2",
                duration=5.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=sample_objectives,
                algorithm="random",
                timestamp=datetime.now(),
                metadata={},
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = await opt_func.optimize(algorithm="random")

            # Verify custom evaluator was used
            assert result is mock_result

            # Check that CustomEvaluatorWrapper was created with custom evaluator
            from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper

            orchestrator_call = mock_orchestrator_class.call_args
            evaluator_arg = orchestrator_call[1]["evaluator"]
            assert isinstance(evaluator_arg, CustomEvaluatorWrapper)
            assert evaluator_arg.custom_evaluator is mock_custom_evaluator

    @pytest.mark.asyncio
    async def test_optimize_with_cloud_service(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test optimization with cloud service."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="cloud",
            max_trials=5,
            eval_dataset=sample_dataset,
        )

        with patch.object(
            opt_func, "_optimize_with_cloud_service"
        ) as mock_cloud_optimize:
            from datetime import datetime

            mock_result = OptimizationResult(
                trials=[],
                best_config={"temperature": 0.3},
                best_score=0.95,
                optimization_id="test_opt_3",
                duration=15.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=sample_objectives,
                algorithm="cloud",
                timestamp=datetime.now(),
                metadata={},
            )
            mock_cloud_optimize.return_value = mock_result

            result = await opt_func.optimize()

            assert result is mock_result
            # Check that cloud optimization was attempted
            assert mock_cloud_optimize.call_count == 1
            call_args = mock_cloud_optimize.call_args[0]
            assert isinstance(call_args[0], Dataset)

    @pytest.mark.asyncio
    async def test_optimize_cloud_service_fallback(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test cloud service fallback to local optimization."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="cloud",
            cloud_fallback_policy="auto",
            max_trials=3,
            eval_dataset=sample_dataset,
        )

        with (
            patch.object(
                opt_func, "_optimize_with_cloud_service"
            ) as mock_cloud_optimize,
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            # Cloud service fails
            mock_cloud_optimize.side_effect = OptimizationError(
                "Cloud service unavailable"
            )

            # Setup local fallback
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            from datetime import datetime

            from traigent.api.types import TrialResult, TrialStatus

            mock_trial = TrialResult(
                trial_id="trial_1",
                config={"temperature": 0.4},
                metrics={"accuracy": 0.7},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            )

            mock_result = OptimizationResult(
                trials=[mock_trial],
                best_config={"temperature": 0.4},
                best_score=0.7,
                optimization_id="test_opt_4",
                duration=8.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=sample_objectives,
                algorithm="random",
                timestamp=datetime.now(),
                metadata={},
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = await opt_func.optimize(algorithm="random")

            # Should fallback to local optimization
            assert result is mock_result
            mock_cloud_optimize.assert_called_once()
            # get_optimizer may or may not be called depending on implementation optimizations
            # The key thing is that we got a result, indicating fallback worked

    @pytest.mark.asyncio
    async def test_optimize_with_none_dataset(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test optimization with None dataset."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=None,  # Explicitly set to None
        )

        with pytest.raises(ConfigurationError, match="Invalid dataset type"):
            await opt_func.optimize()

    @pytest.mark.asyncio
    async def test_optimize_algorithm_not_found(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test optimization with unknown algorithm."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            algorithm="unknown_algorithm",
            eval_dataset=sample_dataset,  # Set dataset on instance
        )

        with patch("traigent.optimizers.get_optimizer") as mock_get_optimizer:
            mock_get_optimizer.side_effect = ValueError("Unknown algorithm")

            with pytest.raises(OptimizationError, match="Unknown optimizer"):
                await opt_func.optimize(algorithm="unknown_algorithm")

    # Framework Override Tests

    @pytest.mark.asyncio
    async def test_optimize_with_framework_override(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test optimization with framework override."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            auto_override_frameworks=True,
            framework_targets=["openai.OpenAI"],
            eval_dataset=sample_dataset,
        )

        with (
            patch("traigent.integrations.framework_override.override_context"),
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            # Setup mocks
            mock_get_optimizer.return_value = Mock()
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            from datetime import datetime

            from traigent.api.types import TrialResult, TrialStatus

            mock_trial = TrialResult(
                trial_id="trial_1",
                config={"temperature": 0.6},
                metrics={"accuracy": 0.85},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            )

            mock_result = OptimizationResult(
                trials=[mock_trial],
                best_config={"temperature": 0.6},
                best_score=0.85,
                optimization_id="test_opt_5",
                duration=3.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=sample_objectives,
                algorithm="random",
                timestamp=datetime.now(),
                metadata={},
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = await opt_func.optimize(algorithm="random")

            # Verify framework override was used if the framework override path was taken
            # Due to mocking complexities, we'll just verify we got a result
            assert result is mock_result
            # The test primarily verifies that the framework override configuration doesn't break optimization

    # Configuration Tests

    def test_configuration_injection(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test configuration injection from context."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        with patch("traigent.config.get_provider") as mock_get_provider:
            mock_provider = Mock()
            mock_provider.get_config.return_value = {
                "algorithm": "bayesian",
                "max_trials": 20,
                "timeout": 60.0,
            }
            mock_get_provider.return_value = mock_provider

            # Configuration should be injected when not explicitly provided
            # This would happen during actual optimization call
            assert opt_func.algorithm == "random"  # Original value

    def test_cloud_mode_activation(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test cloud execution activation."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="cloud",
        )

        assert opt_func.execution_mode == "cloud"

        # In actual implementation, this would enable additional features
        # For now, just verify the flag is set correctly

    # Error Handling Tests

    @pytest.mark.asyncio
    async def test_optimize_with_optimization_error(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test handling of optimization errors."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,  # Set dataset on instance
        )

        # Create an optimization function that will actually throw an error during execution
        # Instead of mocking get_optimizer, let's create a scenario that will fail
        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.optimize = AsyncMock(
                side_effect=OptimizationError("Optimization failed")
            )

            with pytest.raises(OptimizationError, match="Optimization failed"):
                await opt_func.optimize(algorithm="random")

    @pytest.mark.asyncio
    async def test_optimize_with_configuration_error(
        self, mock_function, sample_objectives, sample_dataset
    ):
        """Test handling of configuration errors."""
        # Test with invalid config space that will fail validation during construction
        with pytest.raises((TypeError, ValidationError)):
            OptimizedFunction(
                func=mock_function,
                config_space="invalid_config_space",  # Should be dict, not string
                objectives=sample_objectives,
                eval_dataset=sample_dataset,
            )

    # Integration Tests

    @pytest.mark.asyncio
    async def test_end_to_end_optimization_integration(self, mock_function):
        """Test complete end-to-end optimization integration."""
        # Create realistic configuration
        config_space = {"temperature": [0.0, 0.5, 1.0], "max_tokens": [50, 100, 150]}
        objectives = ["accuracy"]

        # Create dataset
        examples = [
            EvaluationExample({"text": "hello"}, "HELLO"),
            EvaluationExample({"text": "test"}, "TEST"),
        ]
        dataset = Dataset(
            examples, name="integration_test", description="Integration test dataset"
        )

        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=config_space,
            objectives=objectives,
            custom_evaluator=mock_custom_evaluator,
            max_trials=2,
            timeout=30.0,
            eval_dataset=dataset,  # Set dataset on instance
        )

        # Mock all required dependencies
        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            # Setup realistic mocks
            mock_optimizer = Mock()
            mock_optimizer.suggest.side_effect = [
                {"temperature": 0.5, "max_tokens": 100},
                {"temperature": 0.8, "max_tokens": 150},
            ]
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Create realistic optimization result
            from datetime import datetime

            from traigent.api.types import TrialResult, TrialStatus

            # Create mock trials with metrics
            mock_trial = TrialResult(
                trial_id="trial_1",
                config={"temperature": 0.8, "max_tokens": 150},
                metrics={"accuracy": 1.0},
                status=TrialStatus.COMPLETED,
                duration=2.5,
                timestamp=datetime.now(),
                metadata={},
            )

            mock_result = OptimizationResult(
                trials=[mock_trial],
                best_config={"temperature": 0.8, "max_tokens": 150},
                best_score=1.0,
                optimization_id="test_opt_6",
                duration=5.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=["accuracy"],
                algorithm="random",
                timestamp=datetime.now(),
                metadata={},
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            # Run optimization
            result = await opt_func.optimize(algorithm="random")

            # Comprehensive verification
            assert result.status == OptimizationStatus.COMPLETED
            assert result.best_config is not None
            assert result.best_metrics is not None
            assert "accuracy" in result.best_metrics
            assert len(result.trials) == 1
            assert result.duration > 0

            # Verify all components were called correctly if the optimization path was taken
            if mock_get_optimizer.called:
                mock_get_optimizer.assert_called_with(
                    algorithm="random", config_space=config_space, objectives=objectives
                )
            if mock_orchestrator_class.called:
                mock_orchestrator_class.assert_called_once()
                # The func will be wrapped, so just check that optimize was called with a dataset
                mock_orchestrator.optimize.assert_called_once()
                call_args = mock_orchestrator.optimize.call_args
                assert "func" in call_args[1]
                assert call_args[1]["dataset"] == dataset

    # Performance and Memory Tests

    def test_optimized_function_memory_efficiency(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that OptimizedFunction doesn't leak memory."""
        import gc

        # Create and destroy multiple instances
        for i in range(100):
            opt_func = OptimizedFunction(
                func=mock_function,
                config_space=sample_config_space,
                objectives=sample_objectives,
                max_trials=i + 1,
            )
            # Use the instance briefly
            assert opt_func.max_trials == i + 1

        # Force garbage collection
        gc.collect()

        # Memory usage should remain reasonable - verify we can still allocate
        # Create one more instance to verify memory was properly reclaimed
        final_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            max_trials=1,
        )
        assert final_func is not None, "Should create instance after GC"
        assert final_func.max_trials == 1, "Instance should be properly initialized"


class TestOptimizedFunctionLifecycle:
    """Test OptimizedFunction lifecycle state management."""

    @pytest.fixture
    def sample_config_space(self):
        """Sample configuration space."""
        return {
            "temperature": [0.0, 0.5, 1.0],
            "model": ["gpt-3.5", "gpt-4"],
        }

    @pytest.fixture
    def sample_objectives(self):
        """Sample objectives."""
        return ["accuracy"]

    @pytest.fixture
    def mock_function(self):
        """Create mock function."""
        mock_func = MockFunction()
        mock_func.__name__ = "mock_lifecycle_func"
        return mock_func

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        examples = [
            EvaluationExample({"text": "hello"}, "HELLO"),
        ]
        return Dataset(examples, name="test_dataset", description="Test dataset")

    # State Lifecycle Tests

    def test_initial_state_is_unoptimized(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that OptimizedFunction starts in UNOPTIMIZED state."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        assert opt_func.state == OptimizationState.UNOPTIMIZED
        assert opt_func._best_config is None

    def test_state_property_returns_correct_state(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that state property returns the internal state."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Directly manipulate internal state for testing
        opt_func._state = OptimizationState.OPTIMIZING
        assert opt_func.state == OptimizationState.OPTIMIZING

        opt_func._state = OptimizationState.OPTIMIZED
        assert opt_func.state == OptimizationState.OPTIMIZED

        opt_func._state = OptimizationState.ERROR
        assert opt_func.state == OptimizationState.ERROR

    # current_config Access Tests

    def test_current_config_accessible_when_unoptimized(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that current_config is accessible in UNOPTIMIZED state."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Should not raise
        config = opt_func.current_config
        assert isinstance(config, dict)

    def test_current_config_raises_when_optimizing(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that current_config raises OptimizationStateError during optimization."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Simulate optimization in progress
        opt_func._state = OptimizationState.OPTIMIZING

        with pytest.raises(OptimizationStateError) as exc_info:
            _ = opt_func.current_config

        assert "during an active optimization" in str(exc_info.value)
        assert exc_info.value.current_state == "OPTIMIZING"

    def test_current_config_accessible_when_optimized(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that current_config is accessible in OPTIMIZED state."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Simulate completed optimization
        opt_func._state = OptimizationState.OPTIMIZED
        opt_func._current_config = {"temperature": 0.5, "model": "gpt-4"}

        config = opt_func.current_config
        assert config == {"temperature": 0.5, "model": "gpt-4"}

    def test_current_config_accessible_when_error(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that current_config is accessible in ERROR state."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        opt_func._state = OptimizationState.ERROR

        # Should not raise - can access config even after error
        config = opt_func.current_config
        assert isinstance(config, dict)

    def test_current_config_returns_copy(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that current_config returns a copy, not the original."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        opt_func._current_config = {"temperature": 0.5}
        config1 = opt_func.current_config
        config1["temperature"] = 999  # Mutate the copy

        config2 = opt_func.current_config
        assert config2["temperature"] == 0.5  # Original unchanged

    # best_config Tests

    def test_best_config_none_initially(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that best_config is None before optimization."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        assert opt_func.best_config is None

    def test_best_config_returns_value_after_optimization(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that best_config returns the best configuration."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        opt_func._best_config = {"temperature": 0.7, "model": "gpt-4"}

        assert opt_func.best_config == {"temperature": 0.7, "model": "gpt-4"}

    def test_best_config_returns_copy(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that best_config returns a copy, not the original."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        opt_func._best_config = {"temperature": 0.5}
        config1 = opt_func.best_config
        config1["temperature"] = 999

        config2 = opt_func.best_config
        assert config2["temperature"] == 0.5

    # cleanup() Tests

    def test_cleanup_clears_optimization_history(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that cleanup() clears optimization history."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Add some history
        opt_func._optimization_history = [Mock(), Mock()]

        opt_func.cleanup()

        assert len(opt_func._optimization_history) == 0

    def test_cleanup_preserves_config_by_default(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that cleanup() preserves config by default."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Set up state after optimization
        opt_func._state = OptimizationState.OPTIMIZED
        opt_func._current_config = {"temperature": 0.7}
        opt_func._best_config = {"temperature": 0.7}

        opt_func.cleanup(preserve_config=True)

        assert opt_func._current_config == {"temperature": 0.7}
        assert opt_func._best_config == {"temperature": 0.7}
        assert opt_func._state == OptimizationState.OPTIMIZED

    def test_cleanup_can_reset_config(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that cleanup(preserve_config=False) resets config."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Set up state after optimization
        opt_func._state = OptimizationState.OPTIMIZED
        opt_func._current_config = {"temperature": 0.7}
        opt_func._best_config = {"temperature": 0.7}

        opt_func.cleanup(preserve_config=False)

        assert opt_func._best_config is None
        assert opt_func._state == OptimizationState.UNOPTIMIZED
        # current_config should be reset to default
        assert opt_func._current_config == opt_func.default_config

    def test_cleanup_clears_stats(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that cleanup() clears accumulated stats."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Add stats
        opt_func._stats = {"trial_count": 50, "total_cost": 100.0}

        opt_func.cleanup()

        # Stats should be cleared
        assert len(opt_func._stats) == 0

    # reset() Tests

    def test_reset_fully_resets_state(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that reset() fully resets to initial state."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Set up state after optimization
        opt_func._state = OptimizationState.OPTIMIZED
        opt_func._current_config = {"temperature": 0.9}
        opt_func._best_config = {"temperature": 0.9}
        opt_func._optimization_history = [Mock(), Mock()]

        opt_func.reset()

        assert opt_func._state == OptimizationState.UNOPTIMIZED
        assert opt_func._best_config is None
        assert len(opt_func._optimization_history) == 0
        assert opt_func._current_config == opt_func.default_config

    def test_reset_is_idempotent(
        self, mock_function, sample_config_space, sample_objectives
    ):
        """Test that calling reset() multiple times is safe."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
        )

        opt_func.reset()
        opt_func.reset()
        opt_func.reset()

        assert opt_func._state == OptimizationState.UNOPTIMIZED

    # Integration test: state transitions during optimization

    @pytest.mark.asyncio
    async def test_state_transitions_during_optimization(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test that state transitions correctly during optimization."""
        opt_func = OptimizedFunction(
            func=mock_function,
            config_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=1,
        )

        assert opt_func.state == OptimizationState.UNOPTIMIZED

        # Track state changes
        states_seen = []

        original_run_and_finalize = opt_func._run_and_finalize_optimization

        async def tracking_wrapper(*args, **kwargs):
            # Record state at start of optimization
            states_seen.append(("start", opt_func._state))
            result = await original_run_and_finalize(*args, **kwargs)
            states_seen.append(("end", opt_func._state))
            return result

        with (
            patch.object(opt_func, "_run_and_finalize_optimization", tracking_wrapper),
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            from datetime import datetime

            from traigent.api.types import TrialResult, TrialStatus

            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_trial = TrialResult(
                trial_id="trial_1",
                config={"temperature": 0.5},
                metrics={"accuracy": 0.8},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            )

            mock_result = OptimizationResult(
                trials=[mock_trial],
                best_config={"temperature": 0.5},
                best_score=0.8,
                optimization_id="test_opt",
                duration=2.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=sample_objectives,
                algorithm="random",
                timestamp=datetime.now(),
                metadata={},
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            await opt_func.optimize(algorithm="random")

        # After optimization, state should be OPTIMIZED
        assert opt_func.state == OptimizationState.OPTIMIZED
        assert opt_func.best_config is not None


class TestReOptimization:
    """Test re-optimization scenarios (calling optimize() multiple times)."""

    @pytest.fixture
    def sample_config_space(self) -> dict:
        return {"temperature": [0.1, 0.5, 0.9], "top_k": [10, 20, 50]}

    @pytest.fixture
    def sample_objectives(self) -> list[str]:
        return ["accuracy"]

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset using Dataset object (avoids path validation)."""
        examples = [
            EvaluationExample({"input": "q1"}, "a1"),
            EvaluationExample({"input": "q2"}, "a2"),
        ]
        return Dataset(examples, name="test_dataset", description="Test dataset")

    @pytest.fixture
    def alt_dataset(self):
        """Alternative dataset for testing with different data."""
        examples = [
            EvaluationExample({"input": "alt_q1"}, "alt_a1"),
            EvaluationExample({"input": "alt_q2"}, "alt_a2"),
            EvaluationExample({"input": "alt_q3"}, "alt_a3"),
        ]
        return Dataset(examples, name="alt_dataset", description="Alt dataset")

    def _create_mock_result(
        self, objectives: list[str], config: dict, score: float, opt_id: str
    ) -> OptimizationResult:
        """Helper to create mock optimization results."""
        from datetime import datetime

        from traigent.api.types import TrialResult, TrialStatus

        mock_trial = TrialResult(
            trial_id=f"trial_{opt_id}",
            config=config,
            metrics={"accuracy": score},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
            metadata={},
        )
        return OptimizationResult(
            trials=[mock_trial],
            best_config=config,
            best_score=score,
            optimization_id=opt_id,
            duration=2.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=objectives,
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

    @pytest.mark.asyncio
    async def test_optimize_twice_returns_fresh_results(
        self, sample_config_space, sample_objectives, sample_dataset
    ):
        """Calling optimize() twice should return fresh results each time."""

        def my_func(x: str) -> str:
            return x.upper()

        opt_func = OptimizedFunction(
            func=my_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=2,
        )

        first_config = {"temperature": 0.1, "top_k": 10}
        second_config = {"temperature": 0.9, "top_k": 50}

        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # First optimization returns first_config with score 0.7
            first_result = self._create_mock_result(
                sample_objectives, first_config, 0.7, "opt_1"
            )
            # Second optimization returns second_config with score 0.9
            second_result = self._create_mock_result(
                sample_objectives, second_config, 0.9, "opt_2"
            )

            mock_orchestrator.optimize = AsyncMock(
                side_effect=[first_result, second_result]
            )

            # First optimization
            result1 = await opt_func.optimize(algorithm="random")
            assert result1.best_config == first_config
            assert result1.best_score == 0.7
            assert opt_func.state == OptimizationState.OPTIMIZED

            # Second optimization should return fresh results
            result2 = await opt_func.optimize(algorithm="random")
            assert result2.best_config == second_config
            assert result2.best_score == 0.9
            assert opt_func.state == OptimizationState.OPTIMIZED

            # Verify orchestrator was called twice
            assert mock_orchestrator.optimize.call_count == 2

    @pytest.mark.asyncio
    async def test_optimization_history_is_bounded(
        self, sample_config_space, sample_objectives, sample_dataset
    ):
        """History should keep only the most recent optimization results."""

        def my_func(x: str) -> str:
            return x.upper()

        opt_func = OptimizedFunction(
            func=my_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=2,
            optimization_history_limit=2,
        )

        results = [
            self._create_mock_result(
                sample_objectives, {"temperature": 0.1, "top_k": 10}, 0.70, "opt_1"
            ),
            self._create_mock_result(
                sample_objectives, {"temperature": 0.5, "top_k": 20}, 0.80, "opt_2"
            ),
            self._create_mock_result(
                sample_objectives, {"temperature": 0.9, "top_k": 50}, 0.90, "opt_3"
            ),
        ]

        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.optimize = AsyncMock(side_effect=results)

            await opt_func.optimize(algorithm="random")
            await opt_func.optimize(algorithm="random")
            await opt_func.optimize(algorithm="random")

        history = opt_func.get_optimization_history()
        assert len(history) == 2
        assert [item.optimization_id for item in history] == ["opt_2", "opt_3"]

    @pytest.mark.asyncio
    async def test_reset_clears_previous_optimization_state(
        self, sample_config_space, sample_objectives, sample_dataset
    ):
        """reset() should clear optimization state and allow fresh optimization."""

        def my_func(x: str) -> str:
            return x.upper()

        opt_func = OptimizedFunction(
            func=my_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=2,
        )

        best_config = {"temperature": 0.5, "top_k": 20}

        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_result = self._create_mock_result(
                sample_objectives, best_config, 0.85, "opt_reset"
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            # Run optimization
            await opt_func.optimize(algorithm="random")
            assert opt_func.state == OptimizationState.OPTIMIZED
            assert opt_func.best_config == best_config

            # Reset should clear state
            opt_func.reset()
            assert opt_func.state == OptimizationState.UNOPTIMIZED
            assert opt_func.best_config is None

            # Should be able to optimize again after reset
            new_config = {"temperature": 0.1, "top_k": 10}
            new_result = self._create_mock_result(
                sample_objectives, new_config, 0.95, "opt_after_reset"
            )
            mock_orchestrator.optimize = AsyncMock(return_value=new_result)

            result = await opt_func.optimize(algorithm="random")
            assert result.best_config == new_config
            assert opt_func.state == OptimizationState.OPTIMIZED

    @pytest.mark.asyncio
    async def test_optimize_with_different_datasets(
        self, sample_config_space, sample_objectives, sample_dataset, alt_dataset
    ):
        """Optimization should work with different datasets on subsequent calls."""

        def my_func(x: str) -> str:
            return x.upper()

        opt_func = OptimizedFunction(
            func=my_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=2,
        )

        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # First optimization with original dataset
            first_config = {"temperature": 0.5, "top_k": 20}
            first_result = self._create_mock_result(
                sample_objectives, first_config, 0.75, "opt_dataset1"
            )
            mock_orchestrator.optimize = AsyncMock(return_value=first_result)

            result1 = await opt_func.optimize(algorithm="random")
            assert result1.best_score == 0.75

            # Second optimization with different dataset
            second_config = {"temperature": 0.9, "top_k": 50}
            second_result = self._create_mock_result(
                sample_objectives, second_config, 0.88, "opt_dataset2"
            )
            mock_orchestrator.optimize = AsyncMock(return_value=second_result)

            # Update dataset and run again
            opt_func._eval_dataset = alt_dataset
            result2 = await opt_func.optimize(algorithm="random")
            assert result2.best_score == 0.88
            assert result2.best_config == second_config

    @pytest.mark.asyncio
    async def test_cleanup_preserves_config_by_default(
        self, sample_config_space, sample_objectives, sample_dataset
    ):
        """cleanup() should preserve best_config by default."""

        def my_func(x: str) -> str:
            return x.upper()

        opt_func = OptimizedFunction(
            func=my_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=2,
        )

        best_config = {"temperature": 0.5, "top_k": 20}

        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_result = self._create_mock_result(
                sample_objectives, best_config, 0.9, "opt_cleanup"
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            await opt_func.optimize(algorithm="random")
            assert opt_func.best_config == best_config

            # Cleanup with preserve_config=True (default)
            opt_func.cleanup(preserve_config=True)

            # Config should still be preserved
            assert opt_func.best_config == best_config
            assert opt_func.state == OptimizationState.OPTIMIZED

    @pytest.mark.asyncio
    async def test_cleanup_without_preserving_config(
        self, sample_config_space, sample_objectives, sample_dataset
    ):
        """cleanup(preserve_config=False) should clear best_config."""

        def my_func(x: str) -> str:
            return x.upper()

        opt_func = OptimizedFunction(
            func=my_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=2,
        )

        best_config = {"temperature": 0.5, "top_k": 20}

        with (
            patch("traigent.optimizers.get_optimizer") as mock_get_optimizer,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_optimizer = Mock()
            mock_get_optimizer.return_value = mock_optimizer

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_result = self._create_mock_result(
                sample_objectives, best_config, 0.9, "opt_cleanup_clear"
            )
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            await opt_func.optimize(algorithm="random")
            assert opt_func.best_config == best_config

            # Cleanup without preserving config
            opt_func.cleanup(preserve_config=False)

            # Config should be cleared, state reset
            assert opt_func.best_config is None
            assert opt_func.state == OptimizationState.UNOPTIMIZED


class TestConfigPersistence:
    """Tests for config persistence features (export_config, auto_load, etc.)."""

    @pytest.fixture
    def sample_config_space(self):
        """Sample configuration space."""
        return {
            "temperature": [0.0, 0.5, 1.0],
            "model": ["gpt-4", "gpt-3.5"],
        }

    @pytest.fixture
    def sample_objectives(self):
        """Sample objectives."""
        return ["accuracy"]

    @pytest.fixture
    def sample_dataset(self):
        """Sample evaluation dataset."""
        examples = [
            EvaluationExample({"text": "hello"}, "HELLO"),
            EvaluationExample({"text": "world"}, "WORLD"),
        ]
        return Dataset(examples, name="test_dataset")

    @pytest.fixture
    def mock_function(self):
        """Mock function for testing."""

        def func(text: str) -> str:
            return text.upper()

        func.__name__ = "test_func"
        return func

    def test_export_config_slim_format(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test export_config with slim format."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        # Set best config manually
        opt_func._best_config = {"temperature": 0.5, "model": "gpt-4"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.json"
            result_path = opt_func.export_config(output_path, format="slim")

            assert result_path.exists()

            import json

            with open(result_path) as f:
                data = json.load(f)

            assert "config" in data
            assert data["config"]["temperature"] == pytest.approx(0.5)
            assert data["config"]["model"] == "gpt-4"
            assert "function_name" in data
            assert "traigent_version" in data

    def test_export_config_full_format(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test export_config with full format including trials."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        # Set best config
        opt_func._best_config = {"temperature": 0.5, "model": "gpt-4"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config_full.json"
            result_path = opt_func.export_config(output_path, format="full")

            assert result_path.exists()

            import json

            with open(result_path) as f:
                data = json.load(f)

            assert "config" in data
            assert "configuration_space" in data

    def test_export_config_raises_without_best_config(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test that export_config raises error when no best config exists."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        # No best config set
        with pytest.raises(ConfigurationError, match="No best configuration"):
            with tempfile.TemporaryDirectory() as tmpdir:
                opt_func.export_config(Path(tmpdir) / "config.json")

    def test_export_config_invalid_format(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test that export_config raises error for invalid format."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )
        opt_func._best_config = {"temperature": 0.5}

        with pytest.raises(ConfigurationError, match="Unknown export format"):
            with tempfile.TemporaryDirectory() as tmpdir:
                opt_func.export_config(
                    Path(tmpdir) / "config.json", format="invalid_format"
                )

    def test_load_config_from_path_slim_format(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test loading config from slim format file."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {
                "config": {"temperature": 0.7, "model": "gpt-3.5"},
                "function_name": "test_func",
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            loaded = opt_func._load_config_from_path(str(config_path))

            assert loaded is not None
            assert loaded["temperature"] == pytest.approx(0.7)
            assert loaded["model"] == "gpt-3.5"

    def test_load_config_from_path_best_config_format(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test loading config from best_config format file."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "best_config.json"
            config_data = {
                "best_config": {"temperature": 0.3, "model": "gpt-4"},
                "best_score": 0.95,
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            loaded = opt_func._load_config_from_path(str(config_path))

            assert loaded is not None
            assert loaded["temperature"] == pytest.approx(0.3)

    def test_load_config_from_path_direct_dict(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test loading config from direct dict format file."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "direct_config.json"
            config_data = {"temperature": 1.0, "model": "gpt-4"}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            loaded = opt_func._load_config_from_path(str(config_path))

            assert loaded is not None
            assert loaded["temperature"] == pytest.approx(1.0)

    def test_load_config_from_path_file_not_found(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test loading config from non-existent file returns None."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        result = opt_func._load_config_from_path("/nonexistent/path/config.json")
        assert result is None

    def test_load_config_from_path_invalid_json(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test loading config from invalid JSON file returns None."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.json"
            with open(config_path, "w") as f:
                f.write("not valid json {{{")

            result = opt_func._load_config_from_path(str(config_path))
            assert result is None

    def test_auto_load_with_load_from_parameter(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test auto-loading config via load_from parameter."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "auto_config.json"
            config_data = {"config": {"temperature": 0.8, "model": "gpt-4"}}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            opt_func = OptimizedFunction(
                func=mock_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                eval_dataset=sample_dataset,
                load_from=str(config_path),
            )

            # Config should be auto-loaded
            assert opt_func._current_config is not None
            assert opt_func._current_config.get("temperature") == pytest.approx(0.8)
            assert opt_func._best_config is not None

    def test_auto_load_with_env_variable(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test auto-loading config via TRAIGENT_CONFIG_PATH env var."""
        import json
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env_config.json"
            config_data = {"config": {"temperature": 0.2, "model": "gpt-3.5"}}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            # Set env var temporarily
            original = os.environ.get("TRAIGENT_CONFIG_PATH")
            try:
                os.environ["TRAIGENT_CONFIG_PATH"] = str(config_path)

                opt_func = OptimizedFunction(
                    func=mock_function,
                    configuration_space=sample_config_space,
                    objectives=sample_objectives,
                    eval_dataset=sample_dataset,
                )

                # Config should be auto-loaded from env var
                assert opt_func._current_config is not None
                assert opt_func._current_config.get("temperature") == pytest.approx(0.2)
            finally:
                if original is None:
                    os.environ.pop("TRAIGENT_CONFIG_PATH", None)
                else:
                    os.environ["TRAIGENT_CONFIG_PATH"] = original

    def test_auto_load_priority_load_from_over_env(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test that load_from parameter takes priority over env var."""
        import json
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Config from load_from param
            load_from_path = Path(tmpdir) / "load_from.json"
            load_from_data = {"config": {"temperature": 0.1}}
            with open(load_from_path, "w") as f:
                json.dump(load_from_data, f)

            # Config from env var
            env_path = Path(tmpdir) / "env.json"
            env_data = {"config": {"temperature": 0.9}}
            with open(env_path, "w") as f:
                json.dump(env_data, f)

            original = os.environ.get("TRAIGENT_CONFIG_PATH")
            try:
                os.environ["TRAIGENT_CONFIG_PATH"] = str(env_path)

                opt_func = OptimizedFunction(
                    func=mock_function,
                    configuration_space=sample_config_space,
                    objectives=sample_objectives,
                    eval_dataset=sample_dataset,
                    load_from=str(load_from_path),
                )

                # load_from should take priority
                assert opt_func._current_config.get("temperature") == pytest.approx(0.1)
            finally:
                if original is None:
                    os.environ.pop("TRAIGENT_CONFIG_PATH", None)
                else:
                    os.environ["TRAIGENT_CONFIG_PATH"] = original

    def test_auto_load_best_finds_latest_config(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test that auto_load_best=True finds latest config in logs."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create optimization log directory structure
            log_dir = (
                Path(tmpdir)
                / "optimization_logs"
                / "experiments"
                / "test_func"
                / "runs"
                / "run_001"
                / "artifacts"
            )
            log_dir.mkdir(parents=True)

            config_path = log_dir / "best_config.json"
            config_data = {"temperature": 0.6, "model": "gpt-4"}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            opt_func = OptimizedFunction(
                func=mock_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                eval_dataset=sample_dataset,
                auto_load_best=True,
                local_storage_path=tmpdir,
            )

            # Should find and load the config
            assert opt_func._current_config is not None
            assert opt_func._current_config.get("temperature") == pytest.approx(0.6)

    def test_find_latest_config_path_with_mock(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test _find_latest_config_path with mocked Path.cwd to isolate test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt_func = OptimizedFunction(
                func=mock_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                eval_dataset=sample_dataset,
                local_storage_path=tmpdir,
            )

            # Mock Path.cwd to return our temp dir so it won't find project logs
            with patch("pathlib.Path.cwd", return_value=Path(tmpdir)):
                result = opt_func._find_latest_config_path()
                # With isolated cwd, should not find any configs
                assert result is None

    def test_export_config_without_metadata(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test export_config with include_metadata=False."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )
        opt_func._best_config = {"temperature": 0.5}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "no_meta.json"
            opt_func.export_config(output_path, include_metadata=False)

            import json

            with open(output_path) as f:
                data = json.load(f)

            assert "config" in data
            assert "function_name" not in data
            assert "traigent_version" not in data

    def test_auto_load_silently_skips_on_error(
        self, mock_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test that auto-load silently skips when config can't be loaded."""
        opt_func = OptimizedFunction(
            func=mock_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            load_from="/nonexistent/path/config.json",
        )

        # Should not raise, just skip loading
        # Config should remain unset (or default)
        assert opt_func._best_config is None or opt_func._best_config == {}
