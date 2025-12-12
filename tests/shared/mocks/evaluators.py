"""Mock evaluator implementations for testing.

This module provides mock implementations of evaluation strategies
for testing evaluation workflows without running actual evaluations.
"""

import random
from collections.abc import Callable
from typing import Any

from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing evaluation workflows."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_count = 0
        self.fixed_metrics = {}
        self.metric_noise = 0.1

    def set_metrics(self, metrics: dict[str, float]):
        """Set fixed metrics to return (for predictable testing)."""
        self.fixed_metrics = metrics

    def set_metric_noise(self, noise_level: float):
        """Set noise level for metrics (0.0 = no noise, 1.0 = high noise)."""
        self.metric_noise = noise_level

    def evaluate(
        self,
        function: Callable,
        dataset: Dataset,
        config: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate function with mock metrics."""
        self.evaluation_count += 1

        # Generate base metrics
        if self.fixed_metrics:
            metrics = self.fixed_metrics.copy()
        else:
            # Generate realistic base metrics
            metrics = {
                "accuracy": 0.75 + random.uniform(-0.1, 0.1),
                "precision": 0.73 + random.uniform(-0.1, 0.1),
                "recall": 0.77 + random.uniform(-0.1, 0.1),
                "f1_score": 0.75 + random.uniform(-0.1, 0.1),
            }

        # Add config-based adjustments
        if config:
            # Better models get higher accuracy
            if config.get("model") == "gpt-4":
                for metric in metrics:
                    metrics[metric] = min(1.0, metrics[metric] + 0.05)
            elif config.get("model") == "gpt-4o-mini":
                for metric in metrics:
                    metrics[metric] = min(1.0, metrics[metric] + 0.03)

            # Lower temperature generally improves consistency
            temp = config.get("temperature", 0.5)
            if temp < 0.3:
                for metric in metrics:
                    metrics[metric] = min(1.0, metrics[metric] + 0.02)

        # Add noise if configured
        if self.metric_noise > 0:
            for metric in metrics:
                noise = random.uniform(-self.metric_noise, self.metric_noise)
                metrics[metric] = max(0.0, min(1.0, metrics[metric] + noise))

        # Round to reasonable precision
        for metric in metrics:
            metrics[metric] = round(metrics[metric], 3)

        # Add evaluation metadata
        metadata = {
            "evaluation_id": f"eval_{self.evaluation_count}",
            "dataset_size": len(dataset.examples),
            "config_used": config or {},
            "mock_evaluation": True,
        }

        return EvaluationResult(metrics=metrics, metadata=metadata)


class MockAsyncEvaluator(MockEvaluator):
    """Mock async evaluator for testing async evaluation workflows."""

    async def evaluate_async(
        self,
        function: Callable,
        dataset: Dataset,
        config: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Async version of evaluate."""
        return self.evaluate(function, dataset, config)


class MockDetailedEvaluator(MockEvaluator):
    """Mock evaluator that provides detailed per-example results."""

    def evaluate(
        self,
        function: Callable,
        dataset: Dataset,
        config: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate with detailed per-example metrics."""
        result = super().evaluate(function, dataset, config)

        # Add per-example details
        example_results = []
        for i, example in enumerate(dataset.examples):
            # Simulate function call and evaluation
            predicted_output = f"Mock prediction {i+1}"

            # Generate per-example metrics
            example_accuracy = random.uniform(0.6, 1.0)
            example_confidence = random.uniform(0.7, 1.0)

            example_results.append(
                {
                    "example_id": i,
                    "input": example.input_data,
                    "expected": example.expected_output,
                    "predicted": predicted_output,
                    "accuracy": round(example_accuracy, 3),
                    "confidence": round(example_confidence, 3),
                    "processing_time": round(random.uniform(0.5, 2.0), 3),
                }
            )

        result.metadata["example_results"] = example_results
        result.metadata["detailed_evaluation"] = True

        return result


class MockStreamingEvaluator(MockEvaluator):
    """Mock evaluator that simulates streaming evaluation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.streaming_enabled = True
        self.batch_size = 5

    def evaluate_streaming(
        self,
        function: Callable,
        dataset: Dataset,
        config: dict[str, Any] | None = None,
    ):
        """Generator that yields evaluation results in batches."""
        examples = dataset.examples

        for i in range(0, len(examples), self.batch_size):
            batch_examples = examples[i : i + self.batch_size]
            batch_dataset = Dataset(
                name=f"{dataset.name}_batch_{i//self.batch_size}",
                examples=batch_examples,
            )

            batch_result = self.evaluate(function, batch_dataset, config)
            batch_result.metadata["batch_info"] = {
                "batch_index": i // self.batch_size,
                "batch_size": len(batch_examples),
                "examples_processed": i + len(batch_examples),
                "total_examples": len(examples),
            }

            yield batch_result


class MockMultiObjectiveEvaluator(MockEvaluator):
    """Mock evaluator that supports multiple objectives."""

    def __init__(self, objectives: list[str], **kwargs):
        super().__init__(**kwargs)
        self.objectives = objectives

    def evaluate(
        self,
        function: Callable,
        dataset: Dataset,
        config: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate with multiple objectives."""
        # Generate metrics for all objectives
        metrics = {}

        for objective in self.objectives:
            if objective == "accuracy":
                metrics[objective] = 0.75 + random.uniform(-0.1, 0.1)
            elif objective == "cost":
                base_cost = 0.01
                if config and config.get("model") == "gpt-4":
                    base_cost *= 3
                metrics[objective] = base_cost + random.uniform(-0.005, 0.005)
            elif objective == "latency":
                base_latency = 1.0
                if config and config.get("max_tokens", 100) > 500:
                    base_latency *= 1.5
                metrics[objective] = base_latency + random.uniform(-0.2, 0.2)
            elif objective == "safety":
                metrics[objective] = 0.9 + random.uniform(-0.1, 0.05)
            elif objective == "coherence":
                metrics[objective] = 0.8 + random.uniform(-0.1, 0.1)
            else:
                # Generic objective
                metrics[objective] = random.uniform(0.5, 1.0)

        # Apply config adjustments
        if config:
            model = config.get("model", "")
            if "gpt-4" in model:
                # Better model improves quality metrics but increases cost
                for obj in ["accuracy", "safety", "coherence"]:
                    if obj in metrics:
                        metrics[obj] = min(1.0, metrics[obj] + 0.05)

        # Round metrics
        for objective in metrics:
            if objective == "cost":
                metrics[objective] = round(metrics[objective], 4)
            else:
                metrics[objective] = round(metrics[objective], 3)

        metadata = {
            "objectives": self.objectives,
            "multi_objective": True,
            "pareto_optimal": random.choice([True, False]),
        }

        return EvaluationResult(metrics=metrics, metadata=metadata)


def create_mock_evaluator(
    evaluator_type: str = "standard", objectives: list[str] | None = None, **kwargs
) -> MockEvaluator:
    """Factory function to create mock evaluators."""
    if evaluator_type == "standard":
        return MockEvaluator(**kwargs)
    elif evaluator_type == "async":
        return MockAsyncEvaluator(**kwargs)
    elif evaluator_type == "detailed":
        return MockDetailedEvaluator(**kwargs)
    elif evaluator_type == "streaming":
        return MockStreamingEvaluator(**kwargs)
    elif evaluator_type == "multi_objective":
        if not objectives:
            objectives = ["accuracy", "cost", "latency"]
        return MockMultiObjectiveEvaluator(objectives, **kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


def create_realistic_evaluation_sequence(
    evaluator: MockEvaluator, config_sequence: list[dict[str, Any]], dataset: Dataset
) -> list[EvaluationResult]:
    """Create a sequence of realistic evaluation results."""
    results = []

    for i, config in enumerate(config_sequence):
        # Add some variation based on trial number
        if hasattr(evaluator, "set_metric_noise"):
            # Reduce noise as we get more trials (simulate learning)
            noise_level = max(0.05, 0.2 - (i * 0.02))
            evaluator.set_metric_noise(noise_level)

        # Capture loop variable properly
        idx = i
        result = evaluator.evaluate(
            function=lambda x, _idx=idx: f"Mock output {_idx+1}",
            dataset=dataset,
            config=config,
        )

        # Add trial context
        result.metadata["trial_number"] = i + 1
        result.metadata["config_hash"] = hash(str(sorted(config.items())))

        results.append(result)

    return results
