"""Comprehensive tests for privacy-first mode (Model 1).

These tests verify that:
1. No actual data leaves the client
2. Only metadata and aggregated metrics are sent
3. Subset selection works correctly with indices only
4. The optimization process maintains data privacy throughout
"""

import asyncio
import random
from typing import Any
from unittest.mock import patch

import pytest

# Import our dummy server
from tests.mocks.dummy_privacy_server import DummyPrivacyServer
from traigent.cloud.models import (
    OptimizationSessionStatus,
    SessionCreationRequest,
    TrialStatus,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.optimizers.interactive_optimizer import InteractiveOptimizer


@pytest.fixture
def dummy_server():
    """Create a fresh dummy server for each test."""
    return DummyPrivacyServer()


@pytest.fixture
def sensitive_dataset():
    """Create a dataset with sensitive information that should never be sent."""
    examples = []
    for i in range(50):
        examples.append(
            EvaluationExample(
                input_data={
                    "customer_id": f"CONFIDENTIAL-{i}",
                    "ssn": f"123-45-{i:04d}",  # Mock sensitive data
                    "query": f"Private query {i}",
                    "internal_data": {"secret": f"proprietary-{i}"},
                },
                expected_output=f"Expected response {i}",
            )
        )
    return Dataset(examples)


@pytest.fixture
def privacy_optimizer(dummy_server):
    """Create an optimizer connected to the dummy privacy server."""
    return InteractiveOptimizer(
        config_space={
            "temperature": (0.0, 1.0),
            "model": ["o4-mini", "GPT-4o"],
            "max_tokens": [100, 200, 300],
        },
        objectives=["accuracy", "privacy_score", "latency"],
        remote_service=dummy_server,
        dataset_metadata={
            "size": 50,
            "type": "customer_support",
            "contains_pii": True,
            "sensitivity_level": "high",
        },
        optimization_strategy={
            "exploration_ratio": 0.3,
            "min_examples_per_trial": 5,
            "max_subset_size": 20,
        },
    )


class TestPrivacyCompliance:
    """Test privacy compliance in Model 1."""

    @pytest.mark.asyncio
    async def test_no_data_sent_on_session_creation(
        self, privacy_optimizer, dummy_server
    ):
        """Verify no actual data is sent when creating a session."""
        # Create session
        session = await privacy_optimizer.initialize_session(
            function_name="process_sensitive_data", max_trials=10, user_id="test_user"
        )

        # Verify session created
        assert session.session_id.startswith("mock-session-")

        # Check privacy report
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True
        assert privacy_report["violation_count"] == 0

        # Verify only allowed fields were sent
        allowed_fields = {
            "function_name",
            "configuration_space",
            "objectives",
            "dataset_metadata",
            "max_trials",
            "optimization_strategy",
            "user_id",
            "billing_tier",
            "metadata",
            "problem_type",  # Added as this is a valid non-sensitive field
        }

        for req in dummy_server.received_data:
            if req["request_type"] == "SessionCreationRequest":
                assert set(req["fields"]).issubset(allowed_fields)

    @pytest.mark.asyncio
    async def test_only_indices_in_suggestions(self, privacy_optimizer, dummy_server):
        """Verify suggestions contain only indices, not actual data."""
        # Initialize session
        await privacy_optimizer.initialize_session("test_function", max_trials=5)

        # Get suggestions
        suggestions = []
        for _ in range(3):
            suggestion = await privacy_optimizer.get_next_suggestion(dataset_size=50)
            if suggestion:
                suggestions.append(suggestion)

        # Verify suggestions
        assert len(suggestions) > 0

        for suggestion in suggestions:
            # Check dataset subset contains only indices
            assert hasattr(suggestion.dataset_subset, "indices")
            assert isinstance(suggestion.dataset_subset.indices, list)
            assert all(
                isinstance(idx, int) for idx in suggestion.dataset_subset.indices
            )
            assert all(0 <= idx < 50 for idx in suggestion.dataset_subset.indices)

            # Verify no actual data in suggestion
            assert not hasattr(suggestion, "data")
            assert not hasattr(suggestion, "examples")
            assert not hasattr(suggestion.dataset_subset, "examples")

        # Verify privacy compliance
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True

    @pytest.mark.asyncio
    async def test_only_metrics_sent_in_results(self, privacy_optimizer, dummy_server):
        """Verify only aggregated metrics are sent, not raw outputs."""
        # Initialize and get suggestion
        await privacy_optimizer.initialize_session("test_function", max_trials=5)
        suggestion = await privacy_optimizer.get_next_suggestion(dataset_size=50)

        # Report results with only metrics
        await privacy_optimizer.report_results(
            trial_id=suggestion.trial_id,
            metrics={"accuracy": 0.89, "privacy_score": 0.95, "latency": 0.234},
            duration=12.5,
            status=TrialStatus.COMPLETED,
            metadata={
                "subset_size": len(suggestion.dataset_subset.indices),
                "errors": 0,
            },
        )

        # Verify privacy compliance
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True

        # Verify what was sent
        session = dummy_server.sessions[privacy_optimizer.session_id]
        last_trial = session.trials[-1]

        assert "metrics" in last_trial
        assert all(isinstance(v, (int, float)) for v in last_trial["metrics"].values())
        assert "raw_outputs" not in last_trial
        assert "examples" not in last_trial

    @pytest.mark.asyncio
    async def test_sensitive_data_never_leaves_client(
        self, privacy_optimizer, dummy_server, sensitive_dataset
    ):
        """Test complete workflow ensuring sensitive data never leaves client."""
        # Initialize session
        await privacy_optimizer.initialize_session("analyze_customers", max_trials=10)

        # Mock local evaluation function
        async def evaluate_locally(
            config: dict[str, Any], indices: list[int]
        ) -> dict[str, float]:
            """Simulate local evaluation on sensitive data."""
            # Access sensitive data locally
            [sensitive_dataset.examples[i] for i in indices]

            # Simulate processing
            accuracy = random.uniform(0.7, 0.95)
            privacy_score = 1.0  # Perfect privacy since data stays local

            # Return only metrics
            return {
                "accuracy": accuracy,
                "privacy_score": privacy_score,
                "latency": random.uniform(0.1, 0.5),
            }

        # Run optimization loop
        for _trial in range(5):
            # Get suggestion
            suggestion = await privacy_optimizer.get_next_suggestion(
                dataset_size=len(sensitive_dataset.examples)
            )

            if not suggestion:
                break

            # Evaluate locally
            metrics = await evaluate_locally(
                suggestion.config, suggestion.dataset_subset.indices
            )

            # Report only metrics
            await privacy_optimizer.report_results(
                trial_id=suggestion.trial_id, metrics=metrics, duration=1.0
            )

        # Finalize
        await privacy_optimizer.finalize_optimization()

        # Verify complete privacy compliance
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True
        assert privacy_report["violation_count"] == 0

        # Verify no sensitive fields were ever sent
        all_fields = privacy_report["data_fields_seen"]
        sensitive_fields = {
            "ssn",
            "customer_id",
            "internal_data",
            "secret",
            "data",
            "examples",
        }
        assert len(all_fields.intersection(sensitive_fields)) == 0

    @pytest.mark.asyncio
    async def test_privacy_violation_detection(self, dummy_server):
        """Test that privacy violations are properly detected."""
        # Create optimizer that might violate privacy
        InteractiveOptimizer(
            config_space={"param": [1, 2, 3]},
            objectives=["metric"],
            remote_service=dummy_server,
        )

        # Try to send actual data (should be caught by server)
        with patch.object(dummy_server, "received_data", []):
            # Simulate a bad request with actual data
            bad_request = SessionCreationRequest(
                function_name="bad_function",
                configuration_space={},
                objectives=["metric"],
                dataset_metadata={"size": 10},
                max_trials=5,
            )
            # Inject actual data (privacy violation)
            bad_request.dataset = Dataset(
                [
                    EvaluationExample(
                        input_data={"secret": "data"}, expected_output="output"
                    )
                ]
            )

            await dummy_server.create_session(bad_request)

            # Check privacy report
            privacy_report = dummy_server.get_privacy_report()
            assert privacy_report["compliant"] is False
            assert privacy_report["violation_count"] > 0
            assert any("Dataset found" in v for v in privacy_report["violations"])


class TestSubsetSelection:
    """Test smart subset selection in privacy mode."""

    @pytest.mark.asyncio
    async def test_adaptive_subset_sizing(self, privacy_optimizer, dummy_server):
        """Test that subset sizes adapt during optimization."""
        await privacy_optimizer.initialize_session("test_function", max_trials=15)

        subset_sizes = []
        strategies = []

        # Collect subset information across trials
        for trial in range(12):
            suggestion = await privacy_optimizer.get_next_suggestion(dataset_size=100)
            if not suggestion:
                break

            subset_sizes.append(len(suggestion.dataset_subset.indices))
            strategies.append(suggestion.dataset_subset.selection_strategy)

            # Report mock results
            await privacy_optimizer.report_results(
                trial_id=suggestion.trial_id,
                metrics={"accuracy": 0.8 + trial * 0.01},
                duration=1.0,
            )

        # Verify adaptive behavior
        # Early trials should have smaller subsets
        early_sizes = subset_sizes[:3]
        # Take trials that are actually in late phase (trials 11, 12 which are > 10)
        actual_late_trials = [
            subset_sizes[i] for i in range(len(subset_sizes)) if i >= 10
        ]

        assert all(
            s <= 10 for s in early_sizes
        ), "Early trials should use small subsets"
        if actual_late_trials:  # Only check if we have late trials
            assert all(
                s >= 15 for s in actual_late_trials
            ), "Late trials should use larger subsets"
        assert max(early_sizes) <= min(
            subset_sizes[3:]
        ), "Subset size should increase over time"

        # Verify different strategies are used
        assert "diverse_sampling" in strategies
        assert len(set(strategies)) > 1, "Multiple selection strategies should be used"

    @pytest.mark.asyncio
    async def test_subset_indices_validity(self, privacy_optimizer, dummy_server):
        """Test that subset indices are always valid."""
        await privacy_optimizer.initialize_session("test_function", max_trials=10)

        dataset_sizes = [10, 50, 100, 500]

        for dataset_size in dataset_sizes:
            suggestion = await privacy_optimizer.get_next_suggestion(
                dataset_size=dataset_size
            )

            if suggestion:
                indices = suggestion.dataset_subset.indices

                # Verify indices are valid
                assert len(indices) > 0
                assert len(indices) <= dataset_size
                assert all(0 <= idx < dataset_size for idx in indices)
                assert len(set(indices)) == len(indices), "Indices should be unique"

                # Verify subset size respects strategy
                strategy = privacy_optimizer.optimization_strategy
                min_size = strategy.get("min_examples_per_trial", 5)
                max_size = strategy.get("max_subset_size", 50)

                assert len(indices) >= min(min_size, dataset_size)
                assert len(indices) <= min(max_size, dataset_size)

    @pytest.mark.asyncio
    async def test_confidence_levels(self, privacy_optimizer, dummy_server):
        """Test that confidence levels increase during optimization."""
        await privacy_optimizer.initialize_session("test_function", max_trials=10)

        confidence_levels = []

        for _ in range(8):
            suggestion = await privacy_optimizer.get_next_suggestion(dataset_size=100)
            if not suggestion:
                break

            confidence_levels.append(suggestion.dataset_subset.confidence_level)

            await privacy_optimizer.report_results(
                trial_id=suggestion.trial_id, metrics={"accuracy": 0.85}, duration=1.0
            )

        # Verify confidence increases
        assert confidence_levels[0] < confidence_levels[-1]
        assert all(0 <= c <= 1 for c in confidence_levels)


class TestOptimizationFlow:
    """Test complete optimization flow in privacy mode."""

    @pytest.mark.asyncio
    async def test_complete_optimization_cycle(self, privacy_optimizer, dummy_server):
        """Test a complete optimization cycle with privacy preservation."""
        # Initialize
        session = await privacy_optimizer.initialize_session(
            function_name="optimize_private_llm", max_trials=20
        )

        assert session.status == OptimizationSessionStatus.ACTIVE

        # Track optimization progress
        trial_results = []
        best_accuracy = 0.0

        # Run optimization
        while True:
            # Get suggestion
            suggestion = await privacy_optimizer.get_next_suggestion(dataset_size=100)
            if not suggestion:
                break

            # Simulate local evaluation
            # In real use, this would call the actual function with the subset
            accuracy = 0.6 + random.random() * 0.3
            if suggestion.config.get("model") == "GPT-4o":
                accuracy += 0.1

            metrics = {
                "accuracy": accuracy,
                "latency": random.uniform(0.1, 0.5),
                "cost": 0.1 if suggestion.config.get("model") == "GPT-4o" else 0.05,
            }

            best_accuracy = max(best_accuracy, accuracy)

            # Report results
            await privacy_optimizer.report_results(
                trial_id=suggestion.trial_id, metrics=metrics, duration=2.0
            )

            trial_results.append(
                {
                    "trial": len(trial_results) + 1,
                    "config": suggestion.config,
                    "subset_size": len(suggestion.dataset_subset.indices),
                    "metrics": metrics,
                }
            )

            # Check status
            status = await privacy_optimizer.get_optimization_status()
            if status["completed_trials"] >= 10:
                break

        # Finalize
        final_results = await privacy_optimizer.finalize_optimization()

        # Verify results
        assert final_results.total_trials >= 10
        assert final_results.successful_trials > 0
        assert final_results.best_config is not None
        assert final_results.best_metrics is not None
        assert final_results.cost_savings > 0  # From subset selection

        # Verify complete privacy compliance
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True

        # Verify optimization improved
        assert final_results.best_metrics.get("accuracy", 0) >= 0.7

    @pytest.mark.asyncio
    async def test_early_stopping(self, privacy_optimizer, dummy_server):
        """Test early stopping when no improvement."""
        await privacy_optimizer.initialize_session("test_function", max_trials=50)

        # Report same metrics repeatedly
        for _i in range(10):
            suggestion = await privacy_optimizer.get_next_suggestion(dataset_size=100)
            if not suggestion:
                break

            await privacy_optimizer.report_results(
                trial_id=suggestion.trial_id,
                metrics={"accuracy": 0.80, "latency": 0.3},
                duration=1.0,
            )

        # Server might decide to stop early
        await privacy_optimizer.get_optimization_status()

        # Even with early stopping, privacy should be maintained
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True

    @pytest.mark.asyncio
    async def test_failed_trials_handling(self, privacy_optimizer, dummy_server):
        """Test handling of failed trials while maintaining privacy."""
        await privacy_optimizer.initialize_session("test_function", max_trials=10)

        successful_trials = 0
        failed_trials = 0

        for i in range(8):
            suggestion = await privacy_optimizer.get_next_suggestion(dataset_size=100)
            if not suggestion:
                break

            # Simulate some failures
            if i % 3 == 0:
                # Failed trial
                await privacy_optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={},
                    duration=0.1,
                    status=TrialStatus.FAILED,
                    error_message="Simulated error - no sensitive info",
                )
                failed_trials += 1
            else:
                # Successful trial
                await privacy_optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"accuracy": 0.75 + i * 0.02},
                    duration=2.0,
                    status=TrialStatus.COMPLETED,
                )
                successful_trials += 1

        # Finalize
        results = await privacy_optimizer.finalize_optimization()

        assert results.successful_trials == successful_trials
        assert results.total_trials == successful_trials + failed_trials

        # Privacy maintained even with failures
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True


class TestPrivacyEdgeCases:
    """Test edge cases and error conditions in privacy mode."""

    @pytest.mark.asyncio
    async def test_empty_dataset_metadata(self, dummy_server):
        """Test handling of missing dataset metadata."""
        optimizer = InteractiveOptimizer(
            config_space={"param": [1, 2, 3]},
            objectives=["metric"],
            remote_service=dummy_server,
            dataset_metadata={},  # Empty metadata
        )

        await optimizer.initialize_session("test", max_trials=5)
        suggestion = await optimizer.get_next_suggestion(dataset_size=10)

        assert suggestion is not None
        assert len(suggestion.dataset_subset.indices) > 0

        # Privacy still maintained
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, dummy_server):
        """Test multiple concurrent optimization sessions."""
        # Create multiple optimizers
        optimizers = []
        for i in range(3):
            opt = InteractiveOptimizer(
                config_space={"param": [1, 2, 3]},
                objectives=["metric"],
                remote_service=dummy_server,
                dataset_metadata={"size": 50, "session": i},
            )
            optimizers.append(opt)

        # Initialize all sessions
        sessions = []
        for i, opt in enumerate(optimizers):
            session = await opt.initialize_session(f"function_{i}", max_trials=5)
            sessions.append(session)

        # Run trials concurrently
        async def run_trial(optimizer, trial_num):
            suggestion = await optimizer.get_next_suggestion(dataset_size=50)
            if suggestion:
                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"metric": random.random()},
                    duration=1.0,
                )
                return True
            return False

        # Run 3 trials for each optimizer
        tasks = []
        for trial in range(3):
            for opt in optimizers:
                tasks.append(run_trial(opt, trial))

        await asyncio.gather(*tasks)

        # Finalize all
        final_results = []
        for opt in optimizers:
            result = await opt.finalize_optimization()
            final_results.append(result)

        # Verify all sessions completed
        assert len(final_results) == 3
        assert all(r.total_trials > 0 for r in final_results)

        # Privacy maintained across all sessions
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True
