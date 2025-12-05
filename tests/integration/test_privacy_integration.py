"""Integration tests for privacy-first mode client-server interaction.

These tests verify the complete flow between the interactive optimizer
and remote service, ensuring data privacy is maintained at all levels.
"""

from typing import Any, Dict, List

import pytest

from tests.mocks.dummy_privacy_server import DummyPrivacyServer
from traigent.cloud.models import TrialStatus
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.optimizers.interactive_optimizer import InteractiveOptimizer


class MockPrivacyCloudClient:
    """Mock cloud client that wraps the dummy privacy server."""

    def __init__(self, dummy_server: DummyPrivacyServer):
        self.server = dummy_server
        self.api_key = "test-key"
        self.base_url = "https://mock.api"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def create_session(self, request):
        return await self.server.create_session(request)

    async def get_next_trial(self, request):
        return await self.server.get_next_trial(request)

    async def submit_result(self, result):
        return await self.server.submit_result(result)

    async def finalize_session(self, request):
        return await self.server.finalize_session(request)


@pytest.fixture
def privacy_dataset():
    """Dataset with various types of sensitive information."""
    examples = []

    # Customer support scenarios with PII
    for i in range(30):
        examples.append(
            EvaluationExample(
                input_data={
                    "customer_query": f"I need help with my account ending in {1000 + i}",
                    "customer_email": f"customer{i}@private.com",
                    "account_number": f"ACC-{i:06d}",
                    "payment_info": {"last_four": f"{i:04d}", "type": "visa"},
                    "support_notes": f"Previous issue resolved on 2024-{i % 12 + 1:02d}-15",
                    "priority": "high" if i % 3 == 0 else "normal",
                    "category": ["billing", "technical", "account"][i % 3],
                },
                expected_output=f"Resolved customer issue #{i} with professional response",
            )
        )

    return Dataset(examples)


@pytest.fixture
def mock_privacy_service():
    """Create a mock privacy service for testing."""
    server = DummyPrivacyServer()
    return MockPrivacyCloudClient(server), server


class TestPrivacyIntegration:
    """Integration tests for privacy-preserving optimization."""

    @pytest.mark.asyncio
    async def test_end_to_end_privacy_optimization(
        self, privacy_dataset, mock_privacy_service
    ):
        """Test complete end-to-end optimization while preserving privacy."""
        mock_client, dummy_server = mock_privacy_service

        # Create optimizer with privacy configuration
        optimizer = InteractiveOptimizer(
            config_space={
                "model": ["o4-mini", "GPT-4o"],
                "temperature": (0.1, 0.9),
                "max_tokens": [150, 250, 350],
                "system_prompt_style": ["professional", "friendly", "concise"],
            },
            objectives=["accuracy", "customer_satisfaction", "resolution_rate"],
            remote_service=mock_client,
            dataset_metadata={
                "size": len(privacy_dataset.examples),
                "domain": "customer_support",
                "contains_pii": True,
                "languages": ["english"],
                "avg_query_length": 150,
                "categories": ["billing", "technical", "account"],
            },
            optimization_strategy={
                "exploration_ratio": 0.4,
                "min_examples_per_trial": 3,
                "max_subset_size": 15,
                "early_stopping": True,
                "privacy_mode": True,
            },
        )

        # Initialize session
        session = await optimizer.initialize_session(
            function_name="handle_customer_support",
            max_trials=25,
            user_id="privacy_test_user",
            billing_tier="enterprise",
        )

        assert session.session_id is not None

        # Track what happens during optimization
        trial_configs = []
        subset_information = []
        metrics_sent = []

        # Mock customer support function
        async def handle_customer_support(
            config: Dict[str, Any], subset_indices: List[int]
        ) -> Dict[str, float]:
            """Simulate handling customer support with privacy."""
            # Only access local data
            [privacy_dataset.examples[i] for i in subset_indices]

            # Simulate evaluation without exposing data
            base_accuracy = 0.75

            # Model quality adjustment
            if config.get("model") == "GPT-4o":
                base_accuracy += 0.12

            # Temperature adjustment
            temp = config.get("temperature", 0.7)
            if 0.3 <= temp <= 0.7:
                base_accuracy += 0.08

            # Style adjustment
            if config.get("system_prompt_style") == "professional":
                base_accuracy += 0.05

            # Add some variance
            import random

            variance = random.uniform(-0.05, 0.05)

            return {
                "accuracy": min(0.98, base_accuracy + variance),
                "customer_satisfaction": min(0.95, base_accuracy * 1.1 + variance),
                "resolution_rate": min(0.92, base_accuracy * 0.9 + variance),
            }

        # Run optimization loop
        trial_count = 0
        while trial_count < 20:
            # Get next suggestion
            suggestion = await optimizer.get_next_suggestion(
                dataset_size=len(privacy_dataset.examples)
            )

            if not suggestion:
                break

            trial_count += 1
            trial_configs.append(suggestion.config)
            subset_information.append(
                {
                    "size": len(suggestion.dataset_subset.indices),
                    "strategy": suggestion.dataset_subset.selection_strategy,
                    "confidence": suggestion.dataset_subset.confidence_level,
                    "indices_range": (
                        min(suggestion.dataset_subset.indices),
                        max(suggestion.dataset_subset.indices),
                    ),
                }
            )

            # Execute locally (no data sent to server)
            metrics = await handle_customer_support(
                suggestion.config, suggestion.dataset_subset.indices
            )

            metrics_sent.append(metrics)

            # Report only aggregated metrics
            await optimizer.report_results(
                trial_id=suggestion.trial_id,
                metrics=metrics,
                duration=3.5,
                status=TrialStatus.COMPLETED,
                metadata={
                    "subset_processed": len(suggestion.dataset_subset.indices),
                    "category_distribution": "anonymized_stats",
                },
            )

            # Check progress
            status = await optimizer.get_optimization_status()
            print(f"Trial {trial_count}: {status['progress']:.1%} complete")

        # Finalize optimization
        results = await optimizer.finalize_optimization(include_full_history=False)

        # Verify results quality
        assert results.total_trials >= 15
        assert results.successful_trials == results.total_trials  # All succeeded
        assert results.best_metrics["accuracy"] >= 0.75
        assert results.cost_savings > 0.5  # Significant savings from subset selection

        # CRITICAL: Verify complete privacy compliance
        privacy_report = dummy_server.get_privacy_report()
        assert (
            privacy_report["compliant"] is True
        ), f"Privacy violations: {privacy_report['violations']}"
        assert privacy_report["violation_count"] == 0

        # Verify no sensitive data fields were ever sent
        forbidden_fields = {
            "customer_email",
            "account_number",
            "payment_info",
            "customer_query",
            "support_notes",
            "dataset",
            "examples",
            "input_data",
            "expected_output",
            "raw_data",
        }
        sent_fields = privacy_report["data_fields_seen"]
        assert len(sent_fields.intersection(forbidden_fields)) == 0

        # Verify optimization behavior
        assert len({config["model"] for config in trial_configs}) > 1  # Explored models
        assert len(subset_information) > 0

        # Subset sizes should increase over time
        early_sizes = [s["size"] for s in subset_information[:5]]
        late_sizes = [s["size"] for s in subset_information[-5:]]
        assert sum(late_sizes) > sum(early_sizes)

    @pytest.mark.asyncio
    async def test_privacy_with_failures(self, privacy_dataset, mock_privacy_service):
        """Test privacy preservation even when some trials fail."""
        mock_client, dummy_server = mock_privacy_service

        optimizer = InteractiveOptimizer(
            config_space={"param": [1, 2, 3, 4, 5]},
            objectives=["success_rate"],
            remote_service=mock_client,
            dataset_metadata={"size": len(privacy_dataset.examples), "type": "test"},
        )

        await optimizer.initialize_session("test_with_failures", max_trials=15)

        successful = 0
        failed = 0

        for trial in range(12):
            suggestion = await optimizer.get_next_suggestion(
                dataset_size=len(privacy_dataset.examples)
            )
            if not suggestion:
                break

            # Simulate failures for some configurations
            if suggestion.config["param"] == 1:
                # Report failure
                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={},
                    duration=0.5,
                    status=TrialStatus.FAILED,
                    error_message="Configuration failed - no sensitive data here",
                )
                failed += 1
            else:
                # Report success
                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"success_rate": 0.8 + trial * 0.01},
                    duration=2.0,
                    status=TrialStatus.COMPLETED,
                )
                successful += 1

        results = await optimizer.finalize_optimization()

        assert results.successful_trials == successful
        assert results.total_trials == successful + failed

        # Privacy maintained even with failures
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True

    @pytest.mark.asyncio
    async def test_privacy_with_large_dataset(self, mock_privacy_service):
        """Test privacy preservation with large dataset."""
        mock_client, dummy_server = mock_privacy_service

        # Create large dataset with sensitive info
        Dataset(
            [
                EvaluationExample(
                    input_data={
                        "id": f"SENSITIVE-{i}",
                        "data": f"Private information {i}",
                        "metadata": {"secret": f"value_{i}"},
                    },
                    expected_output=f"Output {i}",
                )
                for i in range(500)
            ]
        )

        optimizer = InteractiveOptimizer(
            config_space={"param1": (0.0, 1.0), "param2": [1, 2, 3, 4, 5]},
            objectives=["metric1", "metric2"],
            remote_service=mock_client,
            dataset_metadata={
                "size": 500,
                "type": "large_scale_test",
                "memory_footprint": "large",
            },
        )

        await optimizer.initialize_session("large_scale_test", max_trials=30)

        # Track subset utilization
        total_examples_used = 0
        max_subset_size = 0

        for _ in range(25):
            suggestion = await optimizer.get_next_suggestion(dataset_size=500)
            if not suggestion:
                break

            subset_size = len(suggestion.dataset_subset.indices)
            total_examples_used += subset_size
            max_subset_size = max(max_subset_size, subset_size)

            # Verify indices are valid
            assert all(0 <= idx < 500 for idx in suggestion.dataset_subset.indices)

            await optimizer.report_results(
                trial_id=suggestion.trial_id,
                metrics={"metric1": 0.8, "metric2": 0.75},
                duration=1.0,
            )

        results = await optimizer.finalize_optimization()

        # Verify efficiency
        efficiency = 1.0 - (total_examples_used / (500 * results.total_trials))
        assert efficiency > 0.6  # At least 60% reduction in examples processed
        assert max_subset_size < 100  # Never used more than 20% of dataset

        # Privacy maintained
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True

    @pytest.mark.asyncio
    async def test_privacy_across_multiple_optimizations(self, mock_privacy_service):
        """Test privacy preservation across multiple optimization sessions."""
        mock_client, dummy_server = mock_privacy_service

        # Run multiple optimizations
        results_list = []

        for session_num in range(3):
            optimizer = InteractiveOptimizer(
                config_space={"param": [1, 2, 3]},
                objectives=["metric"],
                remote_service=mock_client,
                dataset_metadata={
                    "size": 50,
                    "session": session_num,
                    "contains_secrets": True,
                },
            )

            await optimizer.initialize_session(
                f"multi_test_{session_num}", max_trials=8
            )

            # Run a few trials
            for _trial in range(6):
                suggestion = await optimizer.get_next_suggestion(dataset_size=50)
                if not suggestion:
                    break

                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"metric": 0.8 + session_num * 0.05},
                    duration=1.0,
                )

            results = await optimizer.finalize_optimization()
            results_list.append(results)

        # Verify all optimizations completed
        assert len(results_list) == 3
        assert all(r.total_trials > 0 for r in results_list)

        # Privacy maintained across all sessions
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True
        assert privacy_report["requests_received"] >= 3  # Multiple sessions


class TestDataSubsetStrategy:
    """Test dataset subset selection strategies in privacy mode."""

    @pytest.mark.asyncio
    async def test_diverse_sampling_strategy(self, mock_privacy_service):
        """Test diverse sampling strategy preserves privacy."""
        mock_client, dummy_server = mock_privacy_service

        optimizer = InteractiveOptimizer(
            config_space={"param": [1, 2]},
            objectives=["diversity"],
            remote_service=mock_client,
            dataset_metadata={"size": 100, "diversity": "high"},
        )

        await optimizer.initialize_session("diversity_test", max_trials=10)

        # Get early suggestions (should use diverse sampling)
        diverse_suggestions = []
        for _ in range(3):
            suggestion = await optimizer.get_next_suggestion(dataset_size=100)
            if suggestion:
                diverse_suggestions.append(suggestion)
                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"diversity": 0.7},
                    duration=1.0,
                )

        # Verify diverse sampling was used
        assert any(
            s.dataset_subset.selection_strategy == "diverse_sampling"
            for s in diverse_suggestions
        )

        # Privacy maintained
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True

    @pytest.mark.asyncio
    async def test_representative_sampling_strategy(self, mock_privacy_service):
        """Test representative sampling strategy preserves privacy."""
        mock_client, dummy_server = mock_privacy_service

        optimizer = InteractiveOptimizer(
            config_space={"param": [1, 2, 3]},
            objectives=["representativeness"],
            remote_service=mock_client,
            dataset_metadata={"size": 200, "distribution": "normal"},
        )

        await optimizer.initialize_session("representative_test", max_trials=15)

        # Run several trials to get to representative sampling phase
        for trial in range(8):
            suggestion = await optimizer.get_next_suggestion(dataset_size=200)
            if not suggestion:
                break

            await optimizer.report_results(
                trial_id=suggestion.trial_id,
                metrics={"representativeness": 0.75 + trial * 0.01},
                duration=1.0,
            )

            # Later trials should use representative sampling
            if trial >= 4:
                assert suggestion.dataset_subset.selection_strategy in [
                    "representative_sampling",
                    "high_confidence_sampling",
                ]

        # Privacy maintained throughout
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True


@pytest.mark.asyncio
async def test_privacy_with_real_client_simulation(privacy_dataset):
    """Test privacy preservation with simulated real client behavior."""
    # This test simulates how the real client would behave
    dummy_server = DummyPrivacyServer()

    class SimulatedRealClient:
        """Simulates the actual TraiGentCloudClient behavior."""

        def __init__(self, server):
            self.server = server

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        # Methods that would make HTTP requests in real implementation
        async def create_session(self, request):
            # In real implementation, this would serialize request to JSON
            # and send via HTTP - no actual data should be in the payload
            return await self.server.create_session(request)

        async def get_next_trial(self, request):
            return await self.server.get_next_trial(request)

        async def submit_result(self, result):
            return await self.server.submit_result(result)

        async def finalize_session(self, request):
            return await self.server.finalize_session(request)

    async with SimulatedRealClient(dummy_server) as client:
        # Real usage pattern
        from traigent.optimizers.interactive_optimizer import InteractiveOptimizer

        # This is how a user would actually use the privacy mode
        optimizer = InteractiveOptimizer(
            config_space={
                "model": ["o4-mini", "GPT-4o"],
                "temperature": (0.0, 1.0),
                "max_tokens": [100, 200, 300],
            },
            objectives=["accuracy", "cost_efficiency"],
            remote_service=client,
            dataset_metadata={
                "size": len(privacy_dataset.examples),
                "domain": "customer_support",
                "sensitive": True,
            },
        )

        # User's function that processes sensitive data locally
        async def my_sensitive_function(input_data, **config):
            """User's function that must stay local due to privacy."""
            # This would contain proprietary business logic
            # and handle sensitive customer data
            model = config.get("model", "o4-mini")
            temp = config.get("temperature", 0.7)

            # Simulate processing without exposing details
            base_score = 0.8 if model == "GPT-4o" else 0.7
            temp_adjustment = 0.1 * (1 - abs(temp - 0.6))

            return base_score + temp_adjustment

        # Complete optimization workflow
        await optimizer.initialize_session("my_sensitive_function", max_trials=15)

        optimization_results = []

        while len(optimization_results) < 12:
            suggestion = await optimizer.get_next_suggestion(
                dataset_size=len(privacy_dataset.examples)
            )

            if not suggestion:
                break

            # User processes only the suggested subset locally
            subset_examples = [
                privacy_dataset.examples[i] for i in suggestion.dataset_subset.indices
            ]

            # Evaluate locally - no data sent
            scores = []
            for example in subset_examples:
                score = await my_sensitive_function(
                    example.input_data, **suggestion.config
                )
                scores.append(score)

            # Only aggregated metrics are reported
            metrics = {
                "accuracy": sum(scores) / len(scores),
                "cost_efficiency": 1.0 / suggestion.config.get("max_tokens", 100),
            }

            await optimizer.report_results(
                trial_id=suggestion.trial_id, metrics=metrics, duration=2.0
            )

            optimization_results.append(
                {
                    "config": suggestion.config,
                    "metrics": metrics,
                    "subset_size": len(subset_examples),
                }
            )

        final_results = await optimizer.finalize_optimization()

        # Verify optimization succeeded
        assert final_results.total_trials >= 10
        assert final_results.best_metrics["accuracy"] > 0.75

        # MOST IMPORTANT: Verify no privacy violations
        privacy_report = dummy_server.get_privacy_report()
        assert privacy_report["compliant"] is True
        assert privacy_report["violation_count"] == 0

        print(f"Privacy Report: {privacy_report}")
        print(f"Optimization completed with {final_results.total_trials} trials")
        print(f"Best accuracy: {final_results.best_metrics['accuracy']:.3f}")
        print(f"Cost savings: {final_results.cost_savings:.1%}")
