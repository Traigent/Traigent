"""End-to-end tests for Model 1: Interactive Optimization (Client-side execution)."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from traigent.cloud.client import TraiGentCloudClient
from traigent.cloud.models import (
    DatasetSubsetIndices,
    NextTrialResponse,
    OptimizationFinalizationResponse,
    OptimizationSessionStatus,
    SessionCreationResponse,
    TrialStatus,
    TrialSuggestion,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.optimizers.interactive_optimizer import InteractiveOptimizer


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset(
        [
            EvaluationExample(
                input_data={"text": "What is machine learning?"},
                expected_output="Machine learning is a type of AI that enables systems to learn from data.",
            ),
            EvaluationExample(
                input_data={"text": "Explain neural networks"},
                expected_output="Neural networks are computing systems inspired by biological neural networks.",
            ),
            EvaluationExample(
                input_data={"text": "What is deep learning?"},
                expected_output="Deep learning is a subset of machine learning using neural networks with multiple layers.",
            ),
            EvaluationExample(
                input_data={"text": "Define artificial intelligence"},
                expected_output="AI is the simulation of human intelligence by machines.",
            ),
            EvaluationExample(
                input_data={"text": "What is NLP?"},
                expected_output="Natural Language Processing is AI that helps computers understand human language.",
            ),
        ]
    )


@pytest.fixture
def mock_cloud_client():
    """Create a mock cloud client that acts as remote service."""
    client = AsyncMock(spec=TraiGentCloudClient)

    # Track state
    client._trial_count = 0
    client._results = []

    # Mock session creation
    async def create_session(request):
        return SessionCreationResponse(
            session_id="test-session-123",
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={"exploration_ratio": 0.3, "adaptive_sampling": True},
        )

    # Mock trial suggestions
    async def get_next_trial(request):
        if client._trial_count >= 3:  # Complete after 3 trials
            return NextTrialResponse(
                suggestion=None,
                should_continue=False,
                reason="Optimization converged",
                session_status=OptimizationSessionStatus.ACTIVE,
            )

        client._trial_count += 1

        # Generate different configs for each trial
        configs = [
            {"temperature": 0.3, "model": "o4-mini", "max_tokens": 100},
            {"temperature": 0.7, "model": "GPT-4o", "max_tokens": 200},
            {"temperature": 0.5, "model": "GPT-4o", "max_tokens": 150},
        ]

        config = configs[client._trial_count - 1]

        # Simulate dataset subset selection
        subset_size = 2 if client._trial_count == 1 else 3  # Start small

        suggestion = TrialSuggestion(
            trial_id=f"trial-{client._trial_count}",
            session_id="test-session-123",
            trial_number=client._trial_count,
            config=config,
            dataset_subset=DatasetSubsetIndices(
                indices=list(range(subset_size)),
                selection_strategy="diverse_sampling",
                confidence_level=0.8,
                estimated_representativeness=0.85,
            ),
            exploration_type=(
                "exploration" if client._trial_count == 1 else "exploitation"
            ),
        )

        return NextTrialResponse(
            suggestion=suggestion,
            should_continue=True,
            session_status=OptimizationSessionStatus.ACTIVE,
        )

    # Mock result submission
    async def submit_result(result):
        client._results.append(result)

    # Mock finalization
    async def finalize_session(request):
        # Find best result
        best_metrics = {"accuracy": 0.92, "latency": 0.15}
        best_config = {"temperature": 0.5, "model": "GPT-4o", "max_tokens": 150}

        return OptimizationFinalizationResponse(
            session_id="test-session-123",
            best_config=best_config,
            best_metrics=best_metrics,
            total_trials=3,
            successful_trials=3,
            total_duration=45.6,
            cost_savings=0.68,
            convergence_history=[
                {"trial": 1, "accuracy": 0.75},
                {"trial": 2, "accuracy": 0.88},
                {"trial": 3, "accuracy": 0.92},
            ],
        )

    # Mock status check
    async def get_session_status(session_id):
        return {
            "status": "active",
            "completed_trials": client._trial_count,
            "progress": client._trial_count / 10,  # Assuming max 10 trials
            "session_id": session_id,
        }

    client.create_session = create_session
    client.get_next_trial = get_next_trial
    client.submit_result = submit_result
    client.finalize_session = finalize_session
    client.get_session_status = get_session_status

    return client


@pytest.fixture
def mock_local_executor():
    """Create a mock local function executor."""

    async def execute_function(
        text: str, temperature: float, model: str, max_tokens: int
    ):
        # Simulate execution
        await asyncio.sleep(0.01)

        # Simulate quality based on config
        quality = 0.5
        if model == "GPT-4o":
            quality += 0.3
        if 0.3 <= temperature <= 0.7:
            quality += 0.2

        return {"output": f"Response for: {text}", "quality": quality}

    return execute_function


class TestInteractiveOptimizationE2E:
    """End-to-end tests for interactive optimization workflow."""

    @pytest.mark.asyncio
    async def test_complete_optimization_workflow(
        self, mock_cloud_client, mock_local_executor, sample_dataset
    ):
        """Test complete interactive optimization workflow from start to finish."""

        # Setup optimizer
        optimizer = InteractiveOptimizer(
            config_space={
                "temperature": (0.0, 1.0),
                "model": ["o4-mini", "GPT-4o"],
                "max_tokens": [100, 150, 200],
            },
            objectives=["accuracy", "latency"],
            remote_service=mock_cloud_client,
            dataset_metadata={"size": len(sample_dataset.examples), "type": "qa_pairs"},
        )

        # Phase 1: Initialize session
        session = await optimizer.initialize_session(
            function_name="test_llm_function", max_trials=10
        )

        assert session.session_id == "test-session-123"
        assert session.status == OptimizationSessionStatus.ACTIVE
        assert session.max_trials == 10

        # Phase 2: Run optimization trials
        trial_results = []

        while True:
            # Get suggestion
            suggestion = await optimizer.get_next_suggestion(
                dataset_size=len(sample_dataset.examples)
            )

            if not suggestion:
                break

            assert suggestion.trial_id.startswith("trial-")
            assert suggestion.config is not None
            assert len(suggestion.dataset_subset.indices) > 0

            # Execute locally on subset
            correct = 0
            total_time = 0

            for idx in suggestion.dataset_subset.indices:
                example = sample_dataset.examples[idx]
                start_time = asyncio.get_event_loop().time()

                result = await mock_local_executor(
                    text=example.input_data["text"], **suggestion.config
                )

                exec_time = asyncio.get_event_loop().time() - start_time
                total_time += exec_time

                # Simple evaluation
                if result["quality"] > 0.7:
                    correct += 1

            # Calculate metrics
            accuracy = correct / len(suggestion.dataset_subset.indices)
            avg_latency = total_time / len(suggestion.dataset_subset.indices)

            metrics = {"accuracy": accuracy, "latency": avg_latency}

            # Report results
            await optimizer.report_results(
                trial_id=suggestion.trial_id,
                metrics=metrics,
                duration=total_time,
                status=TrialStatus.COMPLETED,
            )

            trial_results.append(
                {
                    "trial_id": suggestion.trial_id,
                    "config": suggestion.config,
                    "metrics": metrics,
                    "subset_size": len(suggestion.dataset_subset.indices),
                }
            )

        # Verify trials executed
        assert len(trial_results) == 3
        assert mock_cloud_client._trial_count == 3

        # Phase 3: Get optimization status
        status = await optimizer.get_optimization_status()

        assert status["status"] == "active"
        assert status["completed_trials"] == 3
        assert status["progress"] == 0.3  # 3/10 trials

        # Phase 4: Finalize optimization
        final_results = await optimizer.finalize_optimization()

        assert final_results.best_config == {
            "temperature": 0.5,
            "model": "GPT-4o",
            "max_tokens": 150,
        }
        assert final_results.best_metrics["accuracy"] == 0.92
        assert final_results.cost_savings == 0.68
        assert final_results.successful_trials == 3

        # Verify convergence history
        assert len(final_results.convergence_history) == 3
        assert final_results.convergence_history[-1]["accuracy"] == 0.92

    @pytest.mark.asyncio
    async def test_optimization_with_failures(self, mock_cloud_client, sample_dataset):
        """Test optimization handling trial failures gracefully."""

        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            remote_service=mock_cloud_client,
        )

        await optimizer.initialize_session("test_function", max_trials=5)

        # Get first suggestion
        suggestion = await optimizer.get_next_suggestion(len(sample_dataset.examples))

        # Report a failed trial
        await optimizer.report_results(
            trial_id=suggestion.trial_id,
            metrics={},
            duration=0.1,
            status=TrialStatus.FAILED,
            error_message="Execution timeout",
        )

        # Should still be able to continue
        next_suggestion = await optimizer.get_next_suggestion(
            len(sample_dataset.examples)
        )
        assert next_suggestion is not None

        # Report successful trial
        await optimizer.report_results(
            trial_id=next_suggestion.trial_id,
            metrics={"accuracy": 0.85},
            duration=1.5,
            status=TrialStatus.COMPLETED,
        )

        # Verify session continues after failure
        status = await optimizer.get_optimization_status()
        assert status["completed_trials"] == 1  # Only successful trials count

    @pytest.mark.asyncio
    async def test_adaptive_subset_selection(self, mock_cloud_client, sample_dataset):
        """Test that dataset subset sizes adapt during optimization."""

        # Track subset sizes
        subset_sizes = []

        # Override get_next_trial to track subset sizes
        original_get_next_trial = mock_cloud_client.get_next_trial

        async def tracking_get_next_trial(request):
            response = await original_get_next_trial(request)
            if response.suggestion:
                subset_sizes.append(len(response.suggestion.dataset_subset.indices))
            return response

        mock_cloud_client.get_next_trial = tracking_get_next_trial

        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            remote_service=mock_cloud_client,
            optimization_strategy={"adaptive_sample_size": True},
        )

        await optimizer.initialize_session("test_function", max_trials=10)

        # Run all trials
        while True:
            suggestion = await optimizer.get_next_suggestion(
                len(sample_dataset.examples)
            )
            if not suggestion:
                break

            await optimizer.report_results(
                trial_id=suggestion.trial_id, metrics={"accuracy": 0.8}, duration=1.0
            )

        # Verify adaptive behavior (sizes should increase)
        assert len(subset_sizes) >= 2
        assert subset_sizes[0] <= subset_sizes[-1]  # Later trials use more data

    @pytest.mark.asyncio
    async def test_optimization_strategy_propagation(
        self, mock_cloud_client, sample_dataset
    ):
        """Test that optimization strategy is properly propagated."""

        strategy = {
            "exploration_ratio": 0.4,
            "min_examples_per_trial": 3,
            "adaptive_sample_size": True,
            "early_stopping": True,
            "patience": 3,
        }

        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy", "cost"],
            remote_service=mock_cloud_client,
            optimization_strategy=strategy,
        )

        # Capture session creation request
        called_request = None
        original_create = mock_cloud_client.create_session

        async def capture_create(request):
            nonlocal called_request
            called_request = request
            return await original_create(request)

        mock_cloud_client.create_session = capture_create

        await optimizer.initialize_session("test_function", max_trials=20)

        # Verify strategy was passed
        assert called_request is not None
        assert called_request.optimization_strategy == strategy

    @pytest.mark.asyncio
    async def test_concurrent_optimization_sessions(self, sample_dataset):
        """Test running multiple optimization sessions concurrently."""

        # Create separate mock clients for each session
        sessions_data = []

        async def run_optimization(session_id: str, config_space: dict[str, Any]):
            # Create independent mock client
            mock_client = AsyncMock()
            mock_client._trial_count = 0

            async def create_session(request):
                return SessionCreationResponse(
                    session_id=session_id,
                    status=OptimizationSessionStatus.ACTIVE,
                    optimization_strategy={},
                )

            async def get_next_trial(request):
                mock_client._trial_count += 1
                if mock_client._trial_count > 2:
                    return NextTrialResponse(
                        suggestion=None, should_continue=False, reason="Complete"
                    )

                return NextTrialResponse(
                    suggestion=TrialSuggestion(
                        trial_id=f"{session_id}-trial-{mock_client._trial_count}",
                        session_id=session_id,
                        trial_number=mock_client._trial_count,
                        config={"temperature": 0.5},
                        dataset_subset=DatasetSubsetIndices(
                            indices=[0, 1],
                            selection_strategy="random",
                            confidence_level=0.8,
                            estimated_representativeness=0.8,
                        ),
                        exploration_type="exploration",
                    ),
                    should_continue=True,
                )

            async def submit_result(result):
                pass

            async def finalize_session(request):
                return OptimizationFinalizationResponse(
                    session_id=session_id,
                    best_config={"temperature": 0.5},
                    best_metrics={"accuracy": 0.85},
                    total_trials=2,
                    successful_trials=2,
                    total_duration=10.0,
                    cost_savings=0.5,
                )

            mock_client.create_session = create_session
            mock_client.get_next_trial = get_next_trial
            mock_client.submit_result = submit_result
            mock_client.finalize_session = finalize_session

            # Run optimization
            optimizer = InteractiveOptimizer(
                config_space=config_space,
                objectives=["accuracy"],
                remote_service=mock_client,
            )

            await optimizer.initialize_session(f"function_{session_id}", max_trials=5)

            trials_run = 0
            while True:
                suggestion = await optimizer.get_next_suggestion(
                    len(sample_dataset.examples)
                )
                if not suggestion:
                    break

                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"accuracy": 0.8},
                    duration=0.5,
                )
                trials_run += 1

            final = await optimizer.finalize_optimization()

            sessions_data.append(
                {
                    "session_id": session_id,
                    "trials_run": trials_run,
                    "best_accuracy": final.best_metrics["accuracy"],
                }
            )

        # Run 3 optimizations concurrently
        await asyncio.gather(
            run_optimization("session-1", {"temperature": (0.0, 1.0)}),
            run_optimization("session-2", {"max_tokens": [100, 200, 300]}),
            run_optimization("session-3", {"top_p": (0.5, 1.0)}),
        )

        # Verify all sessions completed
        assert len(sessions_data) == 3
        assert all(s["trials_run"] == 2 for s in sessions_data)
        assert all(s["best_accuracy"] == 0.85 for s in sessions_data)
