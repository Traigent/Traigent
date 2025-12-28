"""Comprehensive tests for evaluators base module.

Tests cover:
- Dataset creation and validation
- EvaluationExample functionality
- Dataset loading from various sources
- BaseEvaluator interface
- File format handling (JSONL, JSON)
- Error handling and validation
- Memory efficiency with large datasets
"""

import json
from typing import Any

import pytest

from traigent.api.types import ExampleResult
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
    load_dataset_from_file,
)
from traigent.utils.exceptions import ValidationError


@pytest.fixture(autouse=True)
def dataset_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    return tmp_path


class TestEvaluationExample:
    """Test EvaluationExample class."""

    def test_evaluation_example_creation_basic(self):
        """Test creating EvaluationExample with basic data."""
        input_data = {"text": "Hello world"}
        expected_output = "Hello, world!"

        example = EvaluationExample(input_data, expected_output)

        assert example.input_data == input_data
        assert example.expected_output == expected_output
        assert example.metadata == {}

    def test_evaluation_example_creation_with_metadata(self):
        """Test creating EvaluationExample with metadata."""
        input_data = {"query": "What is AI?"}
        expected_output = "Artificial Intelligence explanation"
        metadata = {"difficulty": "easy", "category": "definition"}

        example = EvaluationExample(input_data, expected_output, metadata)

        assert example.input_data == input_data
        assert example.expected_output == expected_output
        assert example.metadata == metadata
        assert example.metadata["difficulty"] == "easy"
        assert example.metadata["category"] == "definition"

    def test_evaluation_example_creation_empty_input(self):
        """Test creating EvaluationExample with empty input."""
        input_data = {}
        expected_output = "Some output"

        example = EvaluationExample(input_data, expected_output)

        assert example.input_data == {}
        assert example.expected_output == expected_output

    def test_evaluation_example_creation_none_values(self):
        """Test creating EvaluationExample with None values."""
        # None input should be allowed
        example1 = EvaluationExample(None, "output")
        assert example1.input_data is None
        assert example1.expected_output == "output"

        # None output should be allowed
        example2 = EvaluationExample({"input": "test"}, None)
        assert example2.input_data == {"input": "test"}
        assert example2.expected_output is None

        # None metadata should default to empty dict
        example3 = EvaluationExample({"input": "test"}, "output", None)
        assert example3.metadata == {}

    def test_evaluation_example_complex_data_types(self):
        """Test EvaluationExample with complex data types."""
        complex_input = {
            "text": "Complex example",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "context": ["previous", "messages"],
            "flags": {"use_cache": True, "debug": False},
        }

        complex_output = {
            "response": "Generated text",
            "metadata": {"tokens_used": 50, "confidence": 0.95},
        }

        example = EvaluationExample(complex_input, complex_output)

        assert example.input_data["parameters"]["temperature"] == 0.7
        assert example.input_data["context"] == ["previous", "messages"]
        assert example.expected_output["metadata"]["confidence"] == 0.95

    def test_evaluation_example_string_representation(self):
        """Test string representation of EvaluationExample."""
        example = EvaluationExample(
            {"text": "Hello"}, "Hi there!", {"type": "greeting"}
        )

        str_repr = str(example)
        assert "Hello" in str_repr
        assert "Hi there!" in str_repr

        # Check that repr is valid
        repr_str = repr(example)
        assert "EvaluationExample" in repr_str

    def test_evaluation_example_equality(self):
        """Test equality comparison of EvaluationExample objects."""
        example1 = EvaluationExample({"text": "test"}, "output", {"meta": "data"})
        example2 = EvaluationExample({"text": "test"}, "output", {"meta": "data"})
        example3 = EvaluationExample({"text": "different"}, "output", {"meta": "data"})

        assert example1 == example2
        assert example1 != example3
        assert example1 != "not_an_example"


class TestDataset:
    """Test Dataset class."""

    def test_dataset_creation_basic(self):
        """Test creating Dataset with basic examples."""
        examples = [
            EvaluationExample({"text": "Hello"}, "Hi"),
            EvaluationExample({"text": "Goodbye"}, "Bye"),
        ]

        dataset = Dataset(examples, name="test_dataset", description="Test dataset")

        assert len(dataset.examples) == 2
        assert dataset.name == "test_dataset"
        assert dataset.description == "Test dataset"
        assert dataset.size == 2

    def test_dataset_creation_empty(self):
        """Test creating Dataset with empty examples list."""
        dataset = Dataset([], name="empty", description="Empty dataset")

        assert len(dataset.examples) == 0
        assert dataset.size == 0
        assert dataset.name == "empty"

    def test_dataset_creation_default_values(self):
        """Test creating Dataset with default name and description."""
        examples = [EvaluationExample({"test": "data"}, "output")]

        dataset = Dataset(examples)

        assert dataset.name == "dataset"
        assert dataset.description == "Traigent evaluation dataset"
        assert len(dataset.examples) == 1

    def test_dataset_creation_none_examples(self):
        """Test creating Dataset with None examples."""
        with pytest.raises(TypeError):
            Dataset(None)

    def test_dataset_creation_invalid_examples_type(self):
        """Test creating Dataset with invalid examples type."""
        with pytest.raises(TypeError):
            Dataset("not_a_list")

    def test_dataset_creation_invalid_example_objects(self):
        """Test creating Dataset with invalid example objects in list."""
        invalid_examples = [
            EvaluationExample({"valid": "example"}, "output"),
            "not_an_evaluation_example",
            EvaluationExample({"another": "valid"}, "output2"),
        ]

        with pytest.raises(TypeError):
            Dataset(invalid_examples)

    def test_dataset_indexing(self):
        """Test Dataset indexing operations."""
        examples = [
            EvaluationExample({"text": "First"}, "1st"),
            EvaluationExample({"text": "Second"}, "2nd"),
            EvaluationExample({"text": "Third"}, "3rd"),
        ]

        dataset = Dataset(examples, name="indexed_dataset")

        # Test positive indexing
        assert dataset[0].input_data["text"] == "First"
        assert dataset[1].expected_output == "2nd"
        assert dataset[2].input_data["text"] == "Third"

        # Test negative indexing
        assert dataset[-1].input_data["text"] == "Third"
        assert dataset[-2].expected_output == "2nd"

        # Test index out of range
        with pytest.raises(IndexError):
            _ = dataset[10]

        with pytest.raises(IndexError):
            _ = dataset[-10]

    def test_dataset_iteration(self):
        """Test Dataset iteration."""
        examples = [
            EvaluationExample({"id": 1}, "output1"),
            EvaluationExample({"id": 2}, "output2"),
            EvaluationExample({"id": 3}, "output3"),
        ]

        dataset = Dataset(examples)

        # Test iteration
        iterated_examples = []
        for example in dataset:
            iterated_examples.append(example)

        assert len(iterated_examples) == 3
        assert iterated_examples[0].input_data["id"] == 1
        assert iterated_examples[1].expected_output == "output2"
        assert iterated_examples[2].input_data["id"] == 3

    def test_dataset_slicing(self):
        """Test Dataset slicing operations."""
        examples = [EvaluationExample({"id": i}, f"output{i}") for i in range(10)]

        dataset = Dataset(examples)

        # Test slice operations
        first_three = dataset[:3]
        assert len(first_three) == 3
        assert first_three[0].input_data["id"] == 0
        assert first_three[2].input_data["id"] == 2

        middle_slice = dataset[3:7]
        assert len(middle_slice) == 4
        assert middle_slice[0].input_data["id"] == 3
        assert middle_slice[-1].input_data["id"] == 6

        step_slice = dataset[::2]
        assert len(step_slice) == 5
        assert step_slice[0].input_data["id"] == 0
        assert step_slice[1].input_data["id"] == 2

    def test_dataset_len(self):
        """Test Dataset length function."""
        # Empty dataset
        empty_dataset = Dataset([])
        assert len(empty_dataset) == 0

        # Dataset with examples
        examples = [EvaluationExample({"id": i}, f"out{i}") for i in range(5)]
        dataset = Dataset(examples)
        assert len(dataset) == 5

    def test_dataset_bool(self):
        """Test Dataset boolean evaluation."""
        # Empty dataset should be falsy
        empty_dataset = Dataset([])
        assert not empty_dataset
        assert bool(empty_dataset) is False

        # Non-empty dataset should be truthy
        dataset = Dataset([EvaluationExample({"test": "data"}, "output")])
        assert dataset
        assert bool(dataset) is True

    def test_dataset_add_example(self):
        """Test adding examples to dataset."""
        dataset = Dataset([])

        # Add first example
        example1 = EvaluationExample({"first": "example"}, "output1")
        dataset.add_example(example1)

        assert len(dataset) == 1
        assert dataset[0] is example1

        # Add second example
        example2 = EvaluationExample({"second": "example"}, "output2")
        dataset.add_example(example2)

        assert len(dataset) == 2
        assert dataset[1] is example2

    def test_dataset_add_example_invalid_type(self):
        """Test adding invalid example type to dataset."""
        dataset = Dataset([])

        with pytest.raises(TypeError):
            dataset.add_example("not_an_example")

        with pytest.raises(TypeError):
            dataset.add_example({"dict": "not_example"})

    def test_dataset_string_representation(self):
        """Test Dataset string representation."""
        examples = [
            EvaluationExample({"text": "Hello"}, "Hi"),
            EvaluationExample({"text": "Goodbye"}, "Bye"),
        ]
        dataset = Dataset(examples, name="test_dataset", description="Test description")

        str_repr = str(dataset)
        # Dataset class uses default __str__ which shows the object representation
        assert "Dataset" in str_repr or "test_dataset" in str_repr

        repr_str = repr(dataset)
        assert "Dataset" in repr_str
        assert "test_dataset" in repr_str


class TestLoadDatasetFromFile:
    """Test load_dataset_from_file function."""

    def test_load_jsonl_file_basic(self, tmp_path):
        """Test loading basic JSONL file."""
        jsonl_content = [
            '{"input": {"text": "Hello"}, "output": "Hi"}',
            '{"input": {"text": "Goodbye"}, "output": "Bye"}',
        ]

        temp_path = tmp_path / "sample.jsonl"
        temp_path.write_text("\n".join(jsonl_content), encoding="utf-8")

        dataset = load_dataset_from_file(str(temp_path))

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2
        assert dataset[0].input_data["text"] == "Hello"
        assert dataset[0].expected_output == "Hi"
        assert dataset[1].input_data["text"] == "Goodbye"
        assert dataset[1].expected_output == "Bye"

    def test_load_jsonl_file_with_metadata(self, tmp_path):
        """Test loading JSONL file with metadata."""
        jsonl_content = [
            '{"input": {"text": "Test"}, "output": "Result", "type": "simple"}',
            '{"input": {"text": "Complex"}, "output": "Answer", "type": "complex", "difficulty": 5}',
        ]

        temp_path = tmp_path / "metadata.jsonl"
        temp_path.write_text("\n".join(jsonl_content), encoding="utf-8")

        dataset = load_dataset_from_file(str(temp_path))

        assert len(dataset) == 2
        # Metadata is extracted from fields other than input/output
        assert dataset[0].metadata["type"] == "simple"
        assert dataset[1].metadata["type"] == "complex"
        assert dataset[1].metadata["difficulty"] == 5

    def test_load_jsonl_file_missing_fields(self, tmp_path):
        """Test loading JSONL file with missing required fields."""
        # Missing output field
        jsonl_content = [
            '{"input": {"text": "Hello"}}',  # Missing output
            '{"output": "Hi"}',  # Missing input
        ]

        temp_path = tmp_path / "missing_fields.jsonl"
        temp_path.write_text("\n".join(jsonl_content), encoding="utf-8")

        with pytest.raises(ValidationError):
            load_dataset_from_file(str(temp_path))

    def test_load_jsonl_file_invalid_json(self, tmp_path):
        """Test loading JSONL file with invalid JSON."""
        invalid_content = [
            '{"input": {"text": "Hello"}, "output": "Hi"}',  # Valid
            "invalid json line",  # Invalid
            '{"input": {"text": "Goodbye"}, "output": "Bye"}',  # Valid
        ]

        temp_path = tmp_path / "invalid.jsonl"
        temp_path.write_text("\n".join(invalid_content), encoding="utf-8")

        with pytest.raises(ValidationError):
            load_dataset_from_file(str(temp_path))

    def test_load_json_array_file(self, tmp_path):
        """Test loading JSON array file."""
        json_data = [
            {"input": {"text": "First"}, "output": "1st"},
            {"input": {"text": "Second"}, "output": "2nd"},
        ]

        temp_path = tmp_path / "array.json"
        temp_path.write_text(json.dumps(json_data), encoding="utf-8")

        dataset = load_dataset_from_file(str(temp_path))

        assert len(dataset) == 2
        assert dataset[0].input_data["text"] == "First"
        assert dataset[1].expected_output == "2nd"

    def test_load_file_nonexistent(self):
        """Test loading non-existent file."""
        with pytest.raises(ValidationError, match="Dataset file not found"):
            load_dataset_from_file("/nonexistent/path/dataset.jsonl")

    def test_load_file_unsupported_extension(self, tmp_path):
        """Test loading file with unsupported extension."""
        temp_path = tmp_path / "dataset.txt"
        temp_path.write_text("some text content", encoding="utf-8")

        with pytest.raises(ValidationError, match="Unsupported file format"):
            load_dataset_from_file(str(temp_path))

    def test_load_empty_file(self, tmp_path):
        """Test loading empty file."""
        temp_path = tmp_path / "empty.jsonl"
        temp_path.touch()

        with pytest.raises(ValidationError, match="No valid examples found"):
            load_dataset_from_file(str(temp_path))

    def test_load_jsonl_file_with_empty_lines(self, tmp_path):
        """Test loading JSONL file with empty lines."""
        jsonl_content = [
            '{"input": {"text": "Hello"}, "output": "Hi"}',
            "",  # Empty line
            '{"input": {"text": "Goodbye"}, "output": "Bye"}',
            "   ",  # Whitespace only
            '{"input": {"text": "Test"}, "output": "Result"}',
        ]

        temp_path = tmp_path / "with_empty_lines.jsonl"
        temp_path.write_text("\n".join(jsonl_content), encoding="utf-8")

        dataset = load_dataset_from_file(str(temp_path))

        # Should skip empty lines and load valid examples
        assert len(dataset) == 3
        assert dataset[0].input_data["text"] == "Hello"
        assert dataset[1].input_data["text"] == "Goodbye"
        assert dataset[2].input_data["text"] == "Test"


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing BaseEvaluator interface."""

    def __init__(self):
        self.evaluate_call_count = 0
        self.last_config = None
        self.last_dataset = None
        self.should_fail = False
        self.result_to_return = None

    async def evaluate(
        self, func: Any, config: dict[str, Any], dataset: Dataset
    ) -> EvaluationResult:
        """Mock evaluate method."""
        self.evaluate_call_count += 1
        self.last_config = config
        self.last_dataset = dataset

        if self.should_fail:
            raise ValueError("Mock evaluation failure")

        if self.result_to_return:
            return self.result_to_return

        # Create mock example results
        example_results = []
        for i, example in enumerate(dataset):
            example_result = ExampleResult(
                example_id=f"example_{i}",
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=f"mock_output_{i}",
                metrics={"accuracy": 0.8, "score": i * 0.1},
                execution_time=0.1,
                success=True,
                error_message=None,
            )
            example_results.append(example_result)

        # Create aggregated metrics
        aggregated_metrics = {"accuracy": 0.8}

        if example_results:
            aggregated_metrics["average_score"] = sum(
                r.metrics["score"] for r in example_results
            ) / len(example_results)
        else:
            aggregated_metrics["average_score"] = 0.0

        return EvaluationResult(
            config=config,
            example_results=example_results,
            aggregated_metrics=aggregated_metrics,
            total_examples=len(example_results),
            successful_examples=len(example_results),
            duration=0.1 * len(example_results),
        )


class TestBaseEvaluator:
    """Test BaseEvaluator abstract base class."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator."""
        return MockEvaluator()

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        examples = [
            EvaluationExample({"text": "Hello"}, "Hi"),
            EvaluationExample({"text": "Goodbye"}, "Bye"),
        ]
        return Dataset(examples, name="test_dataset")

    def test_base_evaluator_cannot_be_instantiated(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()

    @pytest.mark.asyncio
    async def test_mock_evaluator_basic_evaluation(
        self, mock_evaluator, sample_dataset
    ):
        """Test basic evaluation with mock evaluator."""
        config = {"temperature": 0.7, "max_tokens": 100}

        # Mock function for testing
        def mock_func(input_data):
            return f"output_for_{input_data}"

        result = await mock_evaluator.evaluate(mock_func, config, sample_dataset)

        # Verify evaluator was called correctly
        assert mock_evaluator.evaluate_call_count == 1
        assert mock_evaluator.last_config == config
        assert mock_evaluator.last_dataset is sample_dataset

        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.config == config
        assert len(result.example_results) == 2
        assert result.total_examples == 2
        assert result.successful_examples == 2
        assert "accuracy" in result.aggregated_metrics

    @pytest.mark.asyncio
    async def test_mock_evaluator_with_failure(self, mock_evaluator, sample_dataset):
        """Test mock evaluator with simulated failure."""
        mock_evaluator.should_fail = True
        config = {"temperature": 0.5}

        def mock_func(input_data):
            return f"output_for_{input_data}"

        with pytest.raises(ValueError, match="Mock evaluation failure"):
            await mock_evaluator.evaluate(mock_func, config, sample_dataset)

        assert mock_evaluator.evaluate_call_count == 1

    @pytest.mark.asyncio
    async def test_mock_evaluator_empty_dataset(self, mock_evaluator):
        """Test mock evaluator with empty dataset."""
        empty_dataset = Dataset([])
        config = {"test": "config"}

        def mock_func(input_data):
            return f"output_for_{input_data}"

        result = await mock_evaluator.evaluate(mock_func, config, empty_dataset)

        assert len(result.example_results) == 0
        assert result.total_examples == 0
        assert result.successful_examples == 0

    @pytest.mark.asyncio
    async def test_mock_evaluator_large_dataset(self, mock_evaluator):
        """Test mock evaluator with large dataset."""
        # Create large dataset
        examples = [
            EvaluationExample({"id": i, "text": f"text_{i}"}, f"output_{i}")
            for i in range(1000)
        ]
        large_dataset = Dataset(examples)

        config = {"batch_size": 100}

        def mock_func(input_data):
            return f"output_for_{input_data}"

        result = await mock_evaluator.evaluate(mock_func, config, large_dataset)

        assert len(result.example_results) == 1000
        assert result.total_examples == 1000
        assert result.successful_examples == 1000

        # Verify metrics were computed correctly
        expected_avg_score = sum(i * 0.1 for i in range(1000)) / 1000
        assert (
            abs(result.aggregated_metrics["average_score"] - expected_avg_score) < 0.001
        )


class TestEvaluationResult:
    """Test EvaluationResult class."""

    def test_evaluation_result_creation(self):
        """Test creating EvaluationResult."""
        config = {"temperature": 0.7}
        example_results = [
            ExampleResult(
                example_id="1",
                input_data={"text": "Hello"},
                expected_output="Hi",
                actual_output="Hello there",
                metrics={"accuracy": 0.9},
                execution_time=0.1,
                success=True,
                error_message=None,
            )
        ]
        aggregated_metrics = {"accuracy": 0.9, "avg_time": 0.1}

        result = EvaluationResult(
            config=config,
            example_results=example_results,
            aggregated_metrics=aggregated_metrics,
            total_examples=1,
            successful_examples=1,
            duration=0.1,
        )

        assert result.config == config
        assert len(result.example_results) == 1
        assert result.aggregated_metrics == aggregated_metrics
        assert result.total_examples == 1
        assert result.successful_examples == 1
        assert result.duration == 0.1

    def test_evaluation_result_string_representation(self):
        """Test EvaluationResult string representation."""
        config = {"test": "config"}
        result = EvaluationResult(
            config=config,
            example_results=[],
            aggregated_metrics={"score": 0.8},
            total_examples=10,
            successful_examples=8,
            duration=5.0,
        )

        str_repr = str(result)
        assert "10" in str_repr  # total examples
        assert "8" in str_repr  # successful examples
        assert "0.8" in str_repr  # score


class TestExampleResult:
    """Test ExampleResult class."""

    def test_example_result_creation_success(self):
        """Test creating successful ExampleResult."""
        result = ExampleResult(
            example_id="test_1",
            input_data={"text": "Hello"},
            expected_output="Hi",
            actual_output="Hello there",
            metrics={"accuracy": 0.8, "length": 11},
            execution_time=0.05,
            success=True,
            error_message=None,
        )

        assert result.example_id == "test_1"
        assert result.input_data == {"text": "Hello"}
        assert result.expected_output == "Hi"
        assert result.actual_output == "Hello there"
        assert result.metrics["accuracy"] == 0.8
        assert result.execution_time == 0.05
        assert result.success is True
        assert result.error_message is None

    def test_example_result_creation_failure(self):
        """Test creating failed ExampleResult."""
        result = ExampleResult(
            example_id="test_2",
            input_data={"text": "Error case"},
            expected_output="Should work",
            actual_output=None,
            metrics={"accuracy": 0.0},
            execution_time=0.01,
            success=False,
            error_message="Function threw exception",
        )

        assert result.success is False
        assert result.actual_output is None
        assert result.error_message == "Function threw exception"
        assert result.metrics["accuracy"] == 0.0


class TestIntegration:
    """Test integration scenarios between evaluator components."""

    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self, tmp_path):
        """Test complete evaluation workflow from file to result."""
        # Create test data file
        test_data = [
            {"input": {"text": "Hello world"}, "output": "HELLO WORLD"},
            {"input": {"text": "Goodbye"}, "output": "GOODBYE"},
        ]

        temp_path = tmp_path / "workflow.json"
        temp_path.write_text(json.dumps(test_data), encoding="utf-8")

        # Load dataset from file
        dataset = load_dataset_from_file(str(temp_path))

        # Create and use evaluator
        evaluator = MockEvaluator()
        config = {"algorithm": "test"}

        def mock_func(input_data):
            return input_data["text"].upper()

        result = await evaluator.evaluate(mock_func, config, dataset)

        # Verify complete workflow
        assert len(dataset) == 2
        assert dataset[0].input_data["text"] == "Hello world"
        assert dataset[1].expected_output == "GOODBYE"

        assert result.total_examples == 2
        assert result.config == config
        assert len(result.example_results) == 2

    def test_dataset_memory_efficiency_large_scale(self):
        """Test memory efficiency with large datasets."""
        import gc

        # Create large dataset
        large_examples = [
            EvaluationExample(
                {"text": f"Example {i} with some longer text content"}, f"Output {i}"
            )
            for i in range(10000)
        ]

        dataset = Dataset(large_examples, name="large_dataset")

        # Verify dataset works correctly
        assert len(dataset) == 10000
        assert dataset[0].input_data["text"].startswith("Example 0")
        assert dataset[-1].expected_output == "Output 9999"

        # Test iteration doesn't cause memory issues
        count = 0
        for _example in dataset:
            count += 1
            if count > 100:  # Just test first 100 to avoid long test
                break

        assert count == 101

        # Force garbage collection
        del large_examples
        gc.collect()

        # Dataset should still work
        assert len(dataset) == 10000
