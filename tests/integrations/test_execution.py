"""Tests for pipeline execution with configuration (Story 3.2).

This module tests the apply_config and execute_with_config functionality
for running Haystack pipelines with optimization configurations.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.integrations.haystack.evaluation import EvaluationDataset
from traigent.integrations.haystack.execution import (
    ExampleResult,
    RunResult,
    apply_config,
    execute_with_config,
)


class TestApplyConfig:
    """Tests for apply_config function."""

    def test_applies_single_parameter(self):
        """Test applying a single parameter."""
        # Create mock pipeline with component
        pipeline = MagicMock()
        component = MagicMock()
        component.temperature = 0.5
        pipeline.get_component.return_value = component

        config = {"generator.temperature": 0.8}
        apply_config(pipeline, config)

        assert component.temperature == 0.8
        pipeline.get_component.assert_called_with("generator")

    def test_applies_multiple_parameters(self):
        """Test applying multiple parameters."""
        pipeline = MagicMock()
        gen_component = MagicMock()
        gen_component.temperature = 0.5
        gen_component.model = "old-model"
        ret_component = MagicMock()
        ret_component.top_k = 5

        def get_component(name):
            if name == "generator":
                return gen_component
            elif name == "retriever":
                return ret_component
            return None

        pipeline.get_component.side_effect = get_component

        config = {
            "generator.temperature": 0.8,
            "generator.model": "new-model",
            "retriever.top_k": 10,
        }
        apply_config(pipeline, config)

        assert gen_component.temperature == 0.8
        assert gen_component.model == "new-model"
        assert ret_component.top_k == 10

    def test_raises_for_missing_component(self):
        """Test that missing component raises KeyError."""
        pipeline = MagicMock()
        pipeline.get_component.return_value = None

        config = {"missing.temperature": 0.8}
        with pytest.raises(KeyError, match="not found in pipeline"):
            apply_config(pipeline, config)

    def test_raises_for_missing_parameter(self):
        """Test that missing parameter raises KeyError."""
        pipeline = MagicMock()
        component = MagicMock(spec=["temperature"])  # Only has temperature
        pipeline.get_component.return_value = component

        config = {"generator.nonexistent": 0.8}
        with pytest.raises(KeyError, match="not found on component"):
            apply_config(pipeline, config)

    def test_raises_for_invalid_qualified_name_no_dot(self):
        """Test that invalid format (no dot) raises KeyError."""
        pipeline = MagicMock()

        config = {"invalid_name": 0.8}  # Missing dot
        with pytest.raises(KeyError, match="expected 'component.parameter'"):
            apply_config(pipeline, config)

    def test_empty_config_is_valid(self):
        """Test that empty config is valid (no-op)."""
        pipeline = MagicMock()

        result = apply_config(pipeline, {})

        assert result is pipeline
        pipeline.get_component.assert_not_called()

    def test_returns_same_pipeline(self):
        """Test that apply_config returns the same pipeline instance."""
        pipeline = MagicMock()
        component = MagicMock()
        component.temperature = 0.5
        pipeline.get_component.return_value = component

        result = apply_config(pipeline, {"generator.temperature": 0.8})

        assert result is pipeline

    def test_applies_none_value(self):
        """Test applying None as a value."""
        pipeline = MagicMock()
        component = MagicMock()
        component.optional_param = "some_value"
        pipeline.get_component.return_value = component

        config = {"generator.optional_param": None}
        apply_config(pipeline, config)

        assert component.optional_param is None

    def test_applies_various_types(self):
        """Test applying various value types."""
        pipeline = MagicMock()
        component = MagicMock()
        component.int_param = 0
        component.float_param = 0.0
        component.str_param = ""
        component.bool_param = False
        component.list_param = []
        pipeline.get_component.return_value = component

        config = {
            "generator.int_param": 42,
            "generator.float_param": 3.14,
            "generator.str_param": "hello",
            "generator.bool_param": True,
            "generator.list_param": [1, 2, 3],
        }
        apply_config(pipeline, config)

        assert component.int_param == 42
        assert component.float_param == 3.14
        assert component.str_param == "hello"
        assert component.bool_param is True
        assert component.list_param == [1, 2, 3]


class TestExecuteWithConfig:
    """Tests for execute_with_config function."""

    def _create_mock_pipeline(self, run_results=None, run_error=None):
        """Helper to create a mock pipeline."""
        pipeline = MagicMock()
        component = MagicMock()
        component.temperature = 0.5
        pipeline.get_component.return_value = component

        if run_error:
            pipeline.run.side_effect = run_error
        elif run_results:
            pipeline.run.side_effect = run_results
        else:
            pipeline.run.return_value = {"output": "result"}

        return pipeline

    def test_executes_all_examples(self):
        """Test that all examples are executed."""
        pipeline = self._create_mock_pipeline(
            run_results=[
                {"output": "result1"},
                {"output": "result2"},
            ]
        )

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "q1"}, "expected": "e1"},
                {"input": {"query": "q2"}, "expected": "e2"},
            ]
        )

        # Use copy_pipeline=False to test on original mock
        result = execute_with_config(
            pipeline, {"generator.temperature": 0.8}, dataset, copy_pipeline=False
        )

        assert result.success is True
        assert len(result.example_results) == 2
        assert result.outputs == [{"output": "result1"}, {"output": "result2"}]
        assert pipeline.run.call_count == 2

    def test_passes_correct_inputs(self):
        """Test that correct inputs are passed to pipeline.run()."""
        pipeline = self._create_mock_pipeline()

        dataset = EvaluationDataset.from_dicts(
            [
                {
                    "input": {"query": "test_query", "context": "test_context"},
                    "expected": "e1",
                },
            ]
        )

        # Use copy_pipeline=False to test on original mock
        execute_with_config(
            pipeline, {"generator.temperature": 0.8}, dataset, copy_pipeline=False
        )

        pipeline.run.assert_called_once_with(query="test_query", context="test_context")

    def test_handles_pipeline_error(self):
        """Test that pipeline errors are caught."""
        pipeline = self._create_mock_pipeline(run_error=RuntimeError("Pipeline failed"))

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "q1"}, "expected": "e1"},
            ]
        )

        result = execute_with_config(pipeline, {"generator.temperature": 0.8}, dataset)

        assert result.success is False
        assert result.failed_count == 1
        assert "Pipeline failed" in result.example_results[0].error

    def test_handles_config_error(self):
        """Test that configuration errors are caught."""
        pipeline = MagicMock()
        pipeline.get_component.return_value = None  # Component not found

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "q1"}, "expected": "e1"},
            ]
        )

        result = execute_with_config(pipeline, {"missing.temperature": 0.8}, dataset)

        assert result.success is False
        assert "Configuration error" in result.error

    def test_records_execution_time(self):
        """Test that execution time is recorded."""
        pipeline = self._create_mock_pipeline()

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "q1"}, "expected": "e1"},
            ]
        )

        result = execute_with_config(pipeline, {"generator.temperature": 0.8}, dataset)

        assert result.total_execution_time > 0
        assert result.example_results[0].execution_time >= 0

    def test_continues_on_error_by_default(self):
        """Test that execution continues after error by default."""
        pipeline = MagicMock()
        component = MagicMock()
        component.temperature = 0.5
        pipeline.get_component.return_value = component

        call_counter = {"count": 0}

        def run_side_effect(**kwargs):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                raise RuntimeError("First failed")
            return {"output": "second succeeded"}

        pipeline.run.side_effect = run_side_effect

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "q1"}, "expected": "e1"},
                {"input": {"query": "q2"}, "expected": "e2"},
            ]
        )

        result = execute_with_config(
            pipeline, {"generator.temperature": 0.8}, dataset, copy_pipeline=False
        )

        assert result.success is False
        assert len(result.example_results) == 2
        assert result.failed_count == 1
        assert result.success_count == 1

    def test_aborts_on_error_when_configured(self):
        """Test that execution aborts on error when abort_on_error=True."""
        call_count = 0

        def run_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First failed")
            return {"output": "should not reach"}

        pipeline = self._create_mock_pipeline()
        pipeline.run.side_effect = run_side_effect

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "q1"}, "expected": "e1"},
                {"input": {"query": "q2"}, "expected": "e2"},
            ]
        )

        result = execute_with_config(
            pipeline,
            {"generator.temperature": 0.8},
            dataset,
            abort_on_error=True,
        )

        assert result.success is False
        assert len(result.example_results) == 1  # Only first example
        assert call_count == 1

    def test_empty_config(self):
        """Test execution with empty config."""
        pipeline = self._create_mock_pipeline()

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "q1"}, "expected": "e1"},
            ]
        )

        result = execute_with_config(pipeline, {}, dataset)

        assert result.success is True
        assert len(result.outputs) == 1


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_outputs_returns_successful_only(self):
        """Test that outputs property returns only successful outputs."""
        result = RunResult(
            config={},
            example_results=[
                ExampleResult(0, {}, {"out": 1}, True),
                ExampleResult(1, {}, None, False, "error"),
                ExampleResult(2, {}, {"out": 3}, True),
            ],
        )

        assert result.outputs == [{"out": 1}, {"out": 3}]

    def test_all_outputs_includes_none(self):
        """Test that all_outputs includes None for failed examples."""
        result = RunResult(
            config={},
            example_results=[
                ExampleResult(0, {}, {"out": 1}, True),
                ExampleResult(1, {}, None, False, "error"),
                ExampleResult(2, {}, {"out": 3}, True),
            ],
        )

        assert result.all_outputs == [{"out": 1}, None, {"out": 3}]

    def test_failed_count(self):
        """Test failed_count property."""
        result = RunResult(
            config={},
            example_results=[
                ExampleResult(0, {}, {"out": 1}, True),
                ExampleResult(1, {}, None, False, "error"),
                ExampleResult(2, {}, None, False, "error2"),
            ],
        )

        assert result.failed_count == 2

    def test_success_count(self):
        """Test success_count property."""
        result = RunResult(
            config={},
            example_results=[
                ExampleResult(0, {}, {"out": 1}, True),
                ExampleResult(1, {}, None, False, "error"),
                ExampleResult(2, {}, {"out": 3}, True),
            ],
        )

        assert result.success_count == 2

    def test_success_rate(self):
        """Test success_rate property."""
        result = RunResult(
            config={},
            example_results=[
                ExampleResult(0, {}, {"out": 1}, True),
                ExampleResult(1, {}, None, False, "error"),
                ExampleResult(2, {}, {"out": 3}, True),
                ExampleResult(3, {}, {"out": 4}, True),
            ],
        )

        assert result.success_rate == 0.75

    def test_success_rate_empty(self):
        """Test success_rate with no examples."""
        result = RunResult(config={}, example_results=[])

        assert result.success_rate == 0.0

    def test_len(self):
        """Test __len__ returns example count."""
        result = RunResult(
            config={},
            example_results=[
                ExampleResult(0, {}, {}, True),
                ExampleResult(1, {}, {}, True),
            ],
        )

        assert len(result) == 2

    def test_repr_success(self):
        """Test repr for successful run."""
        result = RunResult(
            config={"a": 1},
            example_results=[ExampleResult(0, {}, {}, True)],
            success=True,
            total_execution_time=1.5,
        )

        repr_str = repr(result)
        assert "success" in repr_str
        assert "1.5" in repr_str or "1.50" in repr_str

    def test_repr_failed(self):
        """Test repr for failed run."""
        result = RunResult(
            config={"a": 1},
            example_results=[ExampleResult(0, {}, None, False)],
            success=False,
            total_execution_time=0.5,
        )

        repr_str = repr(result)
        assert "failed" in repr_str


class TestExampleResult:
    """Tests for ExampleResult dataclass."""

    def test_repr_success(self):
        """Test repr for successful example."""
        result = ExampleResult(
            example_index=0,
            input={"query": "test"},
            output={"answer": "result"},
            success=True,
        )

        repr_str = repr(result)
        assert "index=0" in repr_str
        assert "success" in repr_str

    def test_repr_failed(self):
        """Test repr for failed example."""
        result = ExampleResult(
            example_index=1,
            input={"query": "test"},
            output=None,
            success=False,
            error="Something went wrong",
        )

        repr_str = repr(result)
        assert "index=1" in repr_str
        assert "failed" in repr_str
        assert "Something went wrong" in repr_str


class TestIntegration:
    """Integration tests for execution module."""

    def test_full_execution_flow(self):
        """Test complete execution flow."""
        # Create mock pipeline
        pipeline = MagicMock()
        generator = MagicMock()
        generator.temperature = 0.5
        generator.model = "gpt-4"
        retriever = MagicMock()
        retriever.top_k = 5

        def get_component(name):
            if name == "generator":
                return generator
            elif name == "retriever":
                return retriever
            return None

        pipeline.get_component.side_effect = get_component

        outputs = [
            {"answer": "AI is artificial intelligence"},
            {"answer": "ML is machine learning"},
        ]
        pipeline.run.side_effect = outputs

        # Create dataset
        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "What is AI?"}, "expected": "AI is..."},
                {"input": {"query": "What is ML?"}, "expected": "ML is..."},
            ]
        )

        # Create config
        config = {
            "generator.temperature": 0.8,
            "retriever.top_k": 10,
        }

        # Execute
        result = execute_with_config(pipeline, config, dataset)

        # Verify
        assert result.success is True
        assert len(result.outputs) == 2
        assert result.outputs[0]["answer"] == "AI is artificial intelligence"
        assert result.outputs[1]["answer"] == "ML is machine learning"
        assert generator.temperature == 0.8
        assert retriever.top_k == 10
        assert result.config == config
