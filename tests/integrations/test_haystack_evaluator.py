"""Tests for HaystackEvaluator (Story 3.4).

This module tests the HaystackEvaluator integration with the core
Traigent evaluation infrastructure.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.evaluators.base import Dataset as CoreDataset
from traigent.integrations.haystack.evaluation import EvaluationDataset
from traigent.integrations.haystack.evaluator import HaystackEvaluator


class TestHaystackEvaluatorInit:
    """Tests for HaystackEvaluator initialization."""

    def test_init_with_basic_params(self):
        """Test basic initialization."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            metrics=["accuracy"],
        )

        assert evaluator.pipeline is pipeline
        assert evaluator.haystack_dataset is dataset
        assert evaluator.metrics == ["accuracy"]
        assert evaluator.output_key is None

    def test_init_with_output_key(self):
        """Test initialization with output_key."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            metrics=["accuracy"],
            output_key="llm.replies",
        )

        assert evaluator.output_key == "llm.replies"

    def test_init_with_metric_functions(self):
        """Test initialization with custom metric functions."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        def custom_metric(output, expected):
            return 1.0 if output == expected else 0.0

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            metric_functions={"custom": custom_metric},
        )

        # Should infer metrics from metric_functions
        assert evaluator.metrics == ["custom"]
        assert "custom" in evaluator._metric_registry

    def test_init_with_timeout_and_workers(self):
        """Test initialization with custom timeout and workers."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            timeout=120.0,
            max_workers=4,
        )

        assert evaluator.timeout == 120.0
        assert evaluator.max_workers == 4


class TestHaystackEvaluatorEvaluate:
    """Tests for HaystackEvaluator.evaluate() method."""

    @pytest.mark.asyncio
    async def test_evaluate_runs_pipeline(self):
        """Test that evaluate runs the pipeline with config."""
        # Create mock pipeline
        pipeline = MagicMock()
        component = MagicMock()
        component.temperature = 0.5
        pipeline.get_component.return_value = component
        pipeline.run.return_value = {"answer": "test answer"}

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "Q1"}, "expected": "A1"},
                {"input": {"query": "Q2"}, "expected": "A2"},
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            metrics=["accuracy"],
        )

        config = {"generator.temperature": 0.8}
        core_dataset = dataset.to_core_dataset()

        result = await evaluator.evaluate(
            func=pipeline.run,
            config=config,
            dataset=core_dataset,
        )

        assert result is not None
        assert result.duration > 0
        assert len(result.outputs) == 2

    @pytest.mark.asyncio
    async def test_evaluate_handles_errors(self):
        """Test that evaluate handles pipeline errors gracefully."""
        pipeline = MagicMock()
        component = MagicMock()
        component.temperature = 0.5
        pipeline.get_component.return_value = component
        pipeline.run.side_effect = RuntimeError("Pipeline failed")

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "Q1"}, "expected": "A1"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            metrics=["accuracy"],
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={"generator.temperature": 0.8},
            dataset=dataset.to_core_dataset(),
        )

        assert result is not None
        assert result.errors[0] is not None
        assert "Pipeline failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_evaluate_with_empty_dataset_raises(self):
        """Test that empty dataset raises EvaluationError."""
        pipeline = MagicMock()
        dataset = EvaluationDataset(examples=[])

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        with pytest.raises(Exception, match="empty"):
            await evaluator.evaluate(
                func=pipeline.run,
                config={},
                dataset=CoreDataset(examples=[]),
            )

    @pytest.mark.asyncio
    async def test_evaluate_with_progress_callback(self):
        """Test that progress callback is invoked."""
        pipeline = MagicMock()
        component = MagicMock()
        component.temperature = 0.5
        pipeline.get_component.return_value = component
        pipeline.run.return_value = {"answer": "test"}

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "Q1"}, "expected": "A1"},
                {"input": {"query": "Q2"}, "expected": "A2"},
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        progress_calls = []

        def progress_callback(index: int, payload: dict):
            progress_calls.append((index, payload))

        await evaluator.evaluate(
            func=pipeline.run,
            config={"generator.temperature": 0.8},
            dataset=dataset.to_core_dataset(),
            progress_callback=progress_callback,
        )

        assert len(progress_calls) == 2
        assert progress_calls[0][0] == 0
        assert progress_calls[1][0] == 1
        assert "success" in progress_calls[0][1]

    @pytest.mark.asyncio
    async def test_evaluate_with_sample_lease(self):
        """Test that sample lease is consumed."""
        pipeline = MagicMock()
        component = MagicMock()
        component.temperature = 0.5
        pipeline.get_component.return_value = component
        pipeline.run.return_value = {"answer": "test"}

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "Q1"}, "expected": "A1"},
                {"input": {"query": "Q2"}, "expected": "A2"},
                {"input": {"query": "Q3"}, "expected": "A3"},
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        # Mock sample lease that limits to 2 examples
        sample_lease = MagicMock()
        sample_lease.remaining.return_value = 2
        sample_lease.try_take = MagicMock(return_value=True)

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={"generator.temperature": 0.8},
            dataset=dataset.to_core_dataset(),
            sample_lease=sample_lease,
        )

        # Should only run 2 examples due to budget
        assert result.examples_consumed == 2
        sample_lease.try_take.assert_called_once_with(2)


class TestHaystackEvaluatorOutputExtraction:
    """Tests for output extraction from pipeline results."""

    def test_extract_output_no_key(self):
        """Test output extraction with no key (full output)."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        output = {"llm": {"replies": ["answer"]}, "retriever": {"docs": []}}
        result = evaluator._extract_output(output)

        assert result == output

    def test_extract_output_with_simple_key(self):
        """Test output extraction with simple key."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm",
        )

        output = {"llm": {"replies": ["answer"]}, "retriever": {"docs": []}}
        result = evaluator._extract_output(output)

        assert result == {"replies": ["answer"]}

    def test_extract_output_with_nested_key(self):
        """Test output extraction with nested key."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
        )

        output = {"llm": {"replies": ["answer"]}, "retriever": {"docs": []}}
        result = evaluator._extract_output(output)

        assert result == ["answer"]

    def test_extract_output_missing_key_returns_none(self):
        """Test that missing key returns None."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="nonexistent.key",
        )

        output = {"llm": {"replies": ["answer"]}}
        result = evaluator._extract_output(output)

        assert result is None

    def test_extract_output_none_input(self):
        """Test that None input returns None."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm",
        )

        result = evaluator._extract_output(None)

        assert result is None


class TestHaystackEvaluatorHelpers:
    """Tests for HaystackEvaluator helper methods."""

    def test_create_pipeline_wrapper(self):
        """Test create_pipeline_wrapper returns callable."""
        pipeline = MagicMock()
        pipeline.run.return_value = {"answer": "test"}

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "answer"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        wrapper = evaluator.create_pipeline_wrapper()
        result = wrapper(query="test")

        pipeline.run.assert_called_once_with(query="test")
        assert result == {"answer": "test"}

    def test_get_core_dataset(self):
        """Test get_core_dataset returns core Dataset."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "Q1"}, "expected": "A1"},
                {"input": {"query": "Q2"}, "expected": "A2"},
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        core_dataset = evaluator.get_core_dataset()

        assert isinstance(core_dataset, CoreDataset)
        assert len(core_dataset.examples) == 2
        assert core_dataset.examples[0].input_data == {"query": "Q1"}
        assert core_dataset.examples[0].expected_output == "A1"


class TestEvaluationDatasetConversion:
    """Tests for EvaluationDataset.to_core_dataset() method."""

    def test_to_core_dataset_basic(self):
        """Test basic conversion to core dataset."""
        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "Q1"}, "expected": "A1"},
                {"input": {"query": "Q2"}, "expected": "A2"},
            ]
        )

        core_dataset = dataset.to_core_dataset()

        assert isinstance(core_dataset, CoreDataset)
        assert len(core_dataset.examples) == 2

    def test_to_core_dataset_preserves_input(self):
        """Test that input data is preserved in conversion."""
        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "test", "context": "ctx"}, "expected": "answer"},
            ]
        )

        core_dataset = dataset.to_core_dataset()

        assert core_dataset.examples[0].input_data == {
            "query": "test",
            "context": "ctx",
        }

    def test_to_core_dataset_preserves_expected(self):
        """Test that expected output is preserved."""
        dataset = EvaluationDataset.from_dicts(
            [
                {
                    "input": {"query": "Q1"},
                    "expected": {"answer": "A1", "confidence": 0.9},
                },
            ]
        )

        core_dataset = dataset.to_core_dataset()

        assert core_dataset.examples[0].expected_output == {
            "answer": "A1",
            "confidence": 0.9,
        }

    def test_to_core_dataset_preserves_metadata(self):
        """Test that metadata is preserved."""
        dataset = EvaluationDataset.from_dicts(
            [
                {
                    "input": {"query": "Q1"},
                    "expected": "A1",
                    "metadata": {"source": "test", "difficulty": "easy"},
                },
            ]
        )

        core_dataset = dataset.to_core_dataset()

        assert core_dataset.examples[0].metadata == {
            "source": "test",
            "difficulty": "easy",
        }

    def test_to_core_dataset_empty(self):
        """Test conversion of empty dataset."""
        dataset = EvaluationDataset(examples=[])

        core_dataset = dataset.to_core_dataset()

        assert isinstance(core_dataset, CoreDataset)
        assert len(core_dataset.examples) == 0


class TestHaystackEvaluatorIntegration:
    """Integration tests for HaystackEvaluator."""

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self):
        """Test complete evaluation flow with metrics."""
        # Create mock pipeline
        pipeline = MagicMock()
        generator = MagicMock()
        generator.temperature = 0.5
        retriever = MagicMock()
        retriever.top_k = 5

        def get_component(name):
            if name == "generator":
                return generator
            elif name == "retriever":
                return retriever
            return None

        pipeline.get_component.side_effect = get_component

        # Return matching outputs for accuracy
        outputs = [
            {"answer": "A1"},
            {"answer": "A2"},
        ]
        pipeline.run.side_effect = outputs

        # Create dataset
        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "Q1"}, "expected": "A1"},
                {"input": {"query": "Q2"}, "expected": "A2"},
            ]
        )

        # Create evaluator
        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            metrics=["accuracy"],
            output_key="answer",
        )

        # Run evaluation
        config = {
            "generator.temperature": 0.8,
            "retriever.top_k": 10,
        }

        result = await evaluator.evaluate(
            func=pipeline.run,
            config=config,
            dataset=dataset.to_core_dataset(),
        )

        # Verify results
        assert result is not None
        assert result.duration > 0
        assert len(result.outputs) == 2
        assert result.outputs[0] == "A1"
        assert result.outputs[1] == "A2"
        # Config should have been applied (verified by get_component calls)
        assert generator.temperature == 0.8  # noqa: S101
        assert retriever.top_k == 10
