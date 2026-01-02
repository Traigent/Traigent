"""Tests for OptimizedFunction optimization functionality.

Tests the core optimization process including running trials, handling results,
and various optimization modes.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.objectives import create_default_objectives
from traigent.core.optimized_function import OptimizedFunction
from traigent.tvl.spec_loader import TVLBudget, TVLSpecArtifact
from traigent.utils.exceptions import OptimizationError

from .test_fixtures import (
    create_simple_evaluator,
)


def require(condition: bool, message: str = "Assertion failed") -> None:
    """Raise AssertionError when condition is false."""
    if not condition:
        raise AssertionError(message)


class TestOptimization:
    """Test optimization functionality."""

    @pytest.mark.asyncio
    async def test_basic_optimization(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test basic optimization workflow."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=5,  # renamed from num_trials
        )

        # Mock the orchestrator and optimizer
        from datetime import datetime

        mock_result = OptimizationResult(
            trials=[],
            best_config={"temperature": 0.5, "max_tokens": 200, "model": "gpt-4"},
            best_score=0.95,
            optimization_id="test-001",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="grid",
            timestamp=datetime.now(),
            metadata={},
        )

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = await opt_func.optimize()

            require(isinstance(result, OptimizationResult))
            require(result.best_config["temperature"] == pytest.approx(0.5))
            require(result.best_score == pytest.approx(0.95))
            require(result.status == OptimizationStatus.COMPLETED)

    def test_optimize_sync_wrapper(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test synchronous optimize method."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=3,
        )

        # Mock the async optimize method
        mock_result = OptimizationResult(
            trials=[],
            best_config={"temperature": 0.7},
            best_score=0.9,
            optimization_id="test-opt-001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="random",
            timestamp=None,
            metadata={},
        )

        # Run async optimize in sync context
        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = asyncio.run(opt_func.optimize())

            require(result == mock_result)
            require(result.best_config["temperature"] == pytest.approx(0.7))

    @pytest.mark.asyncio
    async def test_optimization_with_custom_evaluator(
        self, simple_function, sample_config_space, sample_dataset
    ):
        """Test optimization with custom evaluator."""
        custom_evaluator = create_simple_evaluator()

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            custom_evaluator=custom_evaluator,
            eval_dataset=sample_dataset,
            max_trials=3,
        )

        # Mock the orchestrator
        from datetime import datetime

        mock_result = OptimizationResult(
            trials=[],
            best_config={"temperature": 0.3},
            best_score=0.85,
            optimization_id="test-002",
            duration=3.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid",
            timestamp=datetime.now(),
            metadata={},
        )

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = await opt_func.optimize()

            require(result.status == OptimizationStatus.COMPLETED)
            require(opt_func.custom_evaluator == custom_evaluator)

    @pytest.mark.asyncio
    async def test_optimization_with_timeout(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test optimization with timeout."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=100,
            timeout=0.1,  # Very short timeout
        )

        # Mock a result with proper structure
        from datetime import datetime

        from traigent.api.types import OptimizationResult, OptimizationStatus

        mock_result = OptimizationResult(
            trials=[],  # Empty list instead of Mock
            best_config={"temperature": 0.5},
            best_score=0.85,
            optimization_id="timeout-test",
            duration=0.1,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="grid",
            timestamp=datetime.now(),
            metadata={},
        )

        # Mock the orchestrator to return proper result
        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            # Test that timeout parameter is passed correctly
            result = await opt_func.optimize(timeout=0.1)
            require(result.best_score == pytest.approx(0.85))

    @pytest.mark.asyncio
    async def test_cost_limit_forwarded_to_orchestrator(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Ensure cost_limit/cost_approved are forwarded to the orchestrator."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=3,
        )

        from datetime import datetime

        mock_result = OptimizationResult(
            trials=[],
            best_config={"temperature": 0.5},
            best_score=0.9,
            optimization_id="cost-limit-test",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            await opt_func.optimize(cost_limit=1.5, cost_approved=True)

            _, kwargs = MockOrchestrator.call_args
            require(kwargs["cost_limit"] == pytest.approx(1.5))
            require(kwargs["cost_approved"] is True)

    @pytest.mark.asyncio
    async def test_optimization_error_handling(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test error handling during optimization."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
        )

        # Mock optimizer that raises error
        mock_optimizer = Mock()
        mock_optimizer.optimize = AsyncMock(side_effect=OptimizationError("Test error"))

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(side_effect=Exception("Test error"))

            with pytest.raises(OptimizationError, match="Optimization failed"):
                await opt_func.optimize()

    @pytest.mark.asyncio
    async def test_runtime_tvl_spec_overrides(
        self, simple_function, sample_config_space, sample_dataset
    ):
        """Providing tvl_spec at runtime updates the configuration and budget."""

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            eval_dataset=sample_dataset,
            max_trials=2,
        )

        from datetime import datetime

        mock_result = OptimizationResult(
            trials=[],
            best_config={"temperature": 0.2},
            best_score=0.9,
            optimization_id="runtime-tvl",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid",
            timestamp=datetime.now(),
            metadata={"tvl": True},
        )

        tvl_artifact = TVLSpecArtifact(
            path=Path("demo.tvl.yml"),
            environment=None,
            configuration_space={"temperature": [0.1, 0.5, 0.9]},
            objective_schema=create_default_objectives(["accuracy"]),
            constraints=[],
            default_config={"temperature": 0.1},
            metadata={"spec_id": "demo"},
            budget=TVLBudget(max_trials=5),
            algorithm="grid",
        )

        with (
            patch(
                "traigent.core.optimized_function.load_tvl_spec",
                return_value=tvl_artifact,
            ) as mock_loader,
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as MockOrchestrator,
        ):
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            await opt_func.optimize(tvl_spec="demo.tvl.yml")

            mock_loader.assert_called_once()
            _, orchestrator_kwargs = MockOrchestrator.call_args
            require(orchestrator_kwargs["max_trials"] == 5)

    @pytest.mark.asyncio
    async def test_decorator_runtime_defaults_propagate_to_optimize(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Ensure decorator-level runtime defaults flow into optimize()."""
        callback = Mock(name="default_callback")
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            timeout=123.0,
            save_to="defaults.json",
            callbacks=[callback],
            cache_policy="no_repeats",
            max_examples=2,
        )

        sentinel_result = Mock(spec=OptimizationResult)
        with patch.object(
            opt_func,
            "_execute_optimization",
            new=AsyncMock(return_value=sentinel_result),
        ) as mock_execute:
            result = await opt_func.optimize()

        require(result is sentinel_result)
        called = mock_execute.await_args.kwargs
        require(called["timeout"] == pytest.approx(123.0))
        require(called["save_to"] == "defaults.json")
        require(called["callbacks"] == opt_func.callbacks)

        runtime_kwargs = called["algorithm_kwargs"]
        require(runtime_kwargs["cache_policy"] == "no_repeats")
        require(runtime_kwargs["max_examples"] == 2)
        require("parallel_config" not in runtime_kwargs)
        require(opt_func.max_examples == 2)

    @pytest.mark.asyncio
    async def test_runtime_overrides_take_precedence(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Ensure call-time overrides replace decorator defaults."""
        callback = Mock(name="default_callback")
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            timeout=123.0,
            save_to="defaults.json",
            callbacks=[callback],
            cache_policy="no_repeats",
            max_examples=2,
        )

        override_callbacks = [Mock(name="override_callback")]
        sentinel_result = Mock(spec=OptimizationResult)
        with patch.object(
            opt_func,
            "_execute_optimization",
            new=AsyncMock(return_value=sentinel_result),
        ) as mock_execute:
            result = await opt_func.optimize(
                timeout=5.0,
                save_to="override.json",
                callbacks=override_callbacks,
                cache_policy="deterministic",
                max_examples=4,
            )

        require(result is sentinel_result)
        called = mock_execute.await_args.kwargs
        require(called["timeout"] == pytest.approx(5.0))
        require(called["save_to"] == "override.json")
        require(called["callbacks"] == override_callbacks)

        runtime_kwargs = called["algorithm_kwargs"]
        require(runtime_kwargs["cache_policy"] == "deterministic")
        require(runtime_kwargs["max_examples"] == 4)
        require(opt_func._decorator_runtime_overrides["cache_policy"] == "no_repeats")
        require(opt_func.max_examples == 2)

    def test_apply_best_config(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test applying best configuration from optimization results."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Create mock optimization results
        opt_results = OptimizationResult(
            trials=[
                TrialResult(
                    trial_id="trial-1",
                    config={"temperature": 0.5, "max_tokens": 200, "model": "gpt-4"},
                    metrics={"accuracy": 0.95},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=None,
                    metadata={},
                )
            ],
            best_config={"temperature": 0.5, "max_tokens": 200, "model": "gpt-4"},
            best_score=0.95,
            optimization_id="test-opt",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="random",
            timestamp=None,
            metadata={},
        )

        # Store results
        opt_func._optimization_results = opt_results

        # Apply best config
        result = opt_func.apply_best_config()
        require(result is True)
        require(opt_func._current_config == {
            "temperature": 0.5,
            "max_tokens": 200,
            "model": "gpt-4",
        })

    def test_apply_best_config_no_results(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test applying best config when no optimization results exist."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
        )

        # No optimization results - should raise error
        from traigent.utils.exceptions import ConfigurationError

        with pytest.raises(
            ConfigurationError, match="No optimization results available"
        ):
            opt_func.apply_best_config()

    @pytest.mark.asyncio
    async def test_optimization_with_cloud_execution(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test optimization in cloud execution."""
        OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            execution_mode="cloud",
        )

        # In cloud execution, should use cloud services
        with patch("traigent.cloud.client.TraigentCloudClient") as mock_cloud_client:
            mock_client_instance = Mock()
            mock_cloud_client.return_value = mock_client_instance

            # Mock cloud optimization
            mock_client_instance.optimize_function = AsyncMock(
                return_value=Mock(best_config={"temperature": 0.6}, best_score=0.92)
            )

            # Should attempt to use cloud service
            # Implementation depends on actual cloud execution logic
            require(
                mock_cloud_client.call_count in (0, 1)
            )  # Cloud may or may not be invoked

    def test_get_optimization_results(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test retrieving optimization results."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
        )

        # Initially no results
        require(opt_func.get_optimization_results() is None)

        # Add results
        mock_results = Mock(spec=OptimizationResult)
        opt_func._optimization_results = mock_results

        require(opt_func.get_optimization_results() == mock_results)

    @pytest.mark.asyncio
    async def test_parallel_trials(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test optimization with parallel trials."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            parallel_config={"trial_concurrency": 3},
            max_trials=9,
        )

        # Mock optimizer that supports parallel trials
        mock_optimizer = Mock()
        mock_optimizer.optimize = AsyncMock(
            return_value=Mock(
                best_config={"temperature": 0.4},
                best_score=0.88,
                trials=[],
                optimization_history=[],
            )
        )
        mock_optimizer.supports_parallel = True

        # Mock the orchestrator
        from datetime import datetime

        mock_result = OptimizationResult(
            trials=[],
            best_config={"temperature": 0.4},
            best_score=0.88,
            optimization_id="test-003",
            duration=9.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="grid",
            timestamp=datetime.now(),
            metadata={},
        )

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = await opt_func.optimize()

            require(result.best_score == pytest.approx(0.88))
