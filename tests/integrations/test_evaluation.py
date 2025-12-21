"""Tests for evaluation dataset format (Story 3.1).

This module tests the EvaluationDataset and validate_dataset functionality
for the Haystack pipeline optimization evaluation dataset format.
"""

from __future__ import annotations

import pytest

from traigent.integrations.haystack.evaluation import (
    EvaluationDataset,
    EvaluationExample,
    validate_dataset,
)
from traigent.utils.exceptions import DatasetValidationError


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_valid_dataset_passes(self):
        """Test that valid dataset passes validation."""
        data = [
            {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence"},
            {"input": {"query": "Define ML"}, "expected": "Machine Learning"},
        ]
        validate_dataset(data)  # Should not raise

    def test_dataset_with_extra_keys_passes(self):
        """Test that extra keys are allowed."""
        data = [
            {
                "input": {"query": "Q1"},
                "expected": "A1",
                "id": "test-001",
                "metadata": {"source": "unit-test"},
                "custom_field": "ignored",
            },
        ]
        validate_dataset(data)  # Should not raise

    def test_empty_dataset_raises_valueerror(self):
        """Test that empty dataset raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_dataset([])

    def test_missing_input_raises_error(self):
        """Test that missing 'input' key raises DatasetValidationError."""
        data = [
            {"expected": "answer"},
        ]
        with pytest.raises(
            DatasetValidationError, match="missing required key 'input'"
        ) as exc:
            validate_dataset(data)
        assert exc.value.details["index"] == 0
        assert exc.value.details["missing_key"] == "input"

    def test_missing_expected_raises_error(self):
        """Test that missing 'expected' key raises DatasetValidationError."""
        data = [
            {"input": {"query": "test"}},
        ]
        with pytest.raises(
            DatasetValidationError, match="missing required key 'expected'"
        ) as exc:
            validate_dataset(data)
        assert exc.value.details["index"] == 0
        assert exc.value.details["missing_key"] == "expected"

    def test_non_dict_input_raises_error(self):
        """Test that non-dict 'input' raises DatasetValidationError."""
        data = [
            {"input": "not a dict", "expected": "answer"},
        ]
        with pytest.raises(DatasetValidationError, match="'input' must be a dict"):
            validate_dataset(data)

    def test_non_dict_entry_raises_error(self):
        """Test that non-dict entry raises DatasetValidationError."""
        data = ["not a dict"]  # type: ignore[list-item]
        with pytest.raises(DatasetValidationError, match="must be a dict"):
            validate_dataset(data)

    def test_error_includes_entry_index(self):
        """Test that error includes the problematic entry index."""
        data = [
            {"input": {"q": "1"}, "expected": "a"},
            {"input": {"q": "2"}},  # Missing expected at index 1
        ]
        with pytest.raises(DatasetValidationError) as exc:
            validate_dataset(data)
        assert exc.value.details["index"] == 1

    def test_error_at_later_index(self):
        """Test that error correctly reports index for later entries."""
        data = [
            {"input": {"q": "1"}, "expected": "a"},
            {"input": {"q": "2"}, "expected": "b"},
            {"input": {"q": "3"}, "expected": "c"},
            {"expected": "d"},  # Missing input at index 3
        ]
        with pytest.raises(DatasetValidationError) as exc:
            validate_dataset(data)
        assert exc.value.details["index"] == 3
        assert exc.value.details["missing_key"] == "input"

    def test_list_as_input_raises_error(self):
        """Test that list as 'input' raises DatasetValidationError."""
        data = [
            {"input": ["query1", "query2"], "expected": "answer"},
        ]
        with pytest.raises(DatasetValidationError, match="'input' must be a dict"):
            validate_dataset(data)

    def test_none_as_input_raises_error(self):
        """Test that None as 'input' raises DatasetValidationError."""
        data = [
            {"input": None, "expected": "answer"},
        ]
        with pytest.raises(DatasetValidationError, match="'input' must be a dict"):
            validate_dataset(data)

    def test_empty_input_dict_allowed(self):
        """Test that empty dict as 'input' is allowed."""
        data = [
            {"input": {}, "expected": "answer"},
        ]
        validate_dataset(data)  # Should not raise

    def test_none_as_expected_allowed(self):
        """Test that None as 'expected' is allowed."""
        data = [
            {"input": {"query": "test"}, "expected": None},
        ]
        validate_dataset(data)  # Should not raise


class TestEvaluationExample:
    """Tests for EvaluationExample dataclass."""

    def test_create_example(self):
        """Test creating an EvaluationExample."""
        example = EvaluationExample(
            input={"query": "test"},
            expected="answer",
        )
        assert example.input == {"query": "test"}
        assert example.expected == "answer"
        assert example.metadata == {}
        assert example.id is None

    def test_create_example_with_metadata(self):
        """Test creating an EvaluationExample with metadata."""
        example = EvaluationExample(
            input={"query": "test"},
            expected="answer",
            metadata={"source": "unit-test"},
            id="test-001",
        )
        assert example.metadata == {"source": "unit-test"}
        assert example.id == "test-001"

    def test_repr_without_id(self):
        """Test repr without id."""
        example = EvaluationExample(
            input={"query": "test"},
            expected="answer",
        )
        repr_str = repr(example)
        assert "input=" in repr_str
        assert "expected=" in repr_str
        assert "id=" not in repr_str

    def test_repr_with_id(self):
        """Test repr with id."""
        example = EvaluationExample(
            input={"query": "test"},
            expected="answer",
            id="test-001",
        )
        repr_str = repr(example)
        assert "id='test-001'" in repr_str


class TestEvaluationDataset:
    """Tests for EvaluationDataset class."""

    def test_from_dicts_creates_dataset(self):
        """Test from_dicts factory method."""
        data = [
            {"input": {"query": "Q1"}, "expected": "A1"},
            {"input": {"query": "Q2"}, "expected": "A2"},
        ]
        dataset = EvaluationDataset.from_dicts(data)

        assert len(dataset) == 2
        assert dataset[0].input == {"query": "Q1"}
        assert dataset[0].expected == "A1"
        assert dataset[1].input == {"query": "Q2"}
        assert dataset[1].expected == "A2"

    def test_iteration(self):
        """Test dataset iteration."""
        data = [
            {"input": {"q": "1"}, "expected": "a"},
            {"input": {"q": "2"}, "expected": "b"},
        ]
        dataset = EvaluationDataset.from_dicts(data)

        inputs = [ex.input for ex in dataset]
        assert inputs == [{"q": "1"}, {"q": "2"}]

    def test_length(self):
        """Test dataset length."""
        data = [
            {"input": {"q": "1"}, "expected": "a"},
            {"input": {"q": "2"}, "expected": "b"},
            {"input": {"q": "3"}, "expected": "c"},
        ]
        dataset = EvaluationDataset.from_dicts(data)
        assert len(dataset) == 3

    def test_indexing(self):
        """Test dataset indexing."""
        data = [
            {"input": {"q": "1"}, "expected": "a"},
            {"input": {"q": "2"}, "expected": "b"},
        ]
        dataset = EvaluationDataset.from_dicts(data)

        assert dataset[0].expected == "a"
        assert dataset[1].expected == "b"

    def test_negative_indexing(self):
        """Test dataset negative indexing."""
        data = [
            {"input": {"q": "1"}, "expected": "a"},
            {"input": {"q": "2"}, "expected": "b"},
        ]
        dataset = EvaluationDataset.from_dicts(data)

        assert dataset[-1].expected == "b"
        assert dataset[-2].expected == "a"

    def test_metadata_preserved(self):
        """Test that metadata is preserved."""
        data = [
            {
                "input": {"q": "test"},
                "expected": "answer",
                "metadata": {"source": "unit-test", "difficulty": "easy"},
                "id": "test-001",
            },
        ]
        dataset = EvaluationDataset.from_dicts(data)

        assert dataset[0].metadata == {"source": "unit-test", "difficulty": "easy"}
        assert dataset[0].id == "test-001"

    def test_missing_metadata_defaults_to_empty_dict(self):
        """Test that missing metadata defaults to empty dict."""
        data = [
            {"input": {"q": "test"}, "expected": "answer"},
        ]
        dataset = EvaluationDataset.from_dicts(data)

        assert dataset[0].metadata == {}
        assert dataset[0].id is None

    def test_empty_dataset_direct_construction(self):
        """Test creating empty dataset directly."""
        dataset = EvaluationDataset(examples=[])
        assert len(dataset) == 0

    def test_from_dicts_empty_raises_valueerror(self):
        """Test from_dicts with empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            EvaluationDataset.from_dicts([])

    def test_from_dicts_invalid_entry_raises_error(self):
        """Test from_dicts with invalid entry raises DatasetValidationError."""
        data = [
            {"input": {"q": "1"}, "expected": "a"},
            {"input": {"q": "2"}},  # Missing expected
        ]
        with pytest.raises(DatasetValidationError):
            EvaluationDataset.from_dicts(data)

    def test_repr(self):
        """Test dataset repr."""
        data = [
            {"input": {"q": "1"}, "expected": "a"},
            {"input": {"q": "2"}, "expected": "b"},
        ]
        dataset = EvaluationDataset.from_dicts(data)

        repr_str = repr(dataset)
        assert "EvaluationDataset" in repr_str
        assert "2" in repr_str

    def test_complex_expected_types(self):
        """Test that complex expected types are preserved."""
        data = [
            {"input": {"q": "1"}, "expected": ["answer1", "answer2"]},  # List
            {"input": {"q": "2"}, "expected": {"key": "value"}},  # Dict
            {"input": {"q": "3"}, "expected": 42},  # Int
            {"input": {"q": "4"}, "expected": 3.14},  # Float
        ]
        dataset = EvaluationDataset.from_dicts(data)

        assert dataset[0].expected == ["answer1", "answer2"]
        assert dataset[1].expected == {"key": "value"}
        assert dataset[2].expected == 42
        assert dataset[3].expected == 3.14

    def test_complex_input_types(self):
        """Test that complex input types are preserved."""
        data = [
            {
                "input": {
                    "query": "test",
                    "documents": [{"content": "doc1"}, {"content": "doc2"}],
                    "params": {"top_k": 5},
                },
                "expected": "answer",
            },
        ]
        dataset = EvaluationDataset.from_dicts(data)

        assert dataset[0].input["query"] == "test"
        assert len(dataset[0].input["documents"]) == 2
        assert dataset[0].input["params"]["top_k"] == 5


class TestIntegration:
    """Integration tests for evaluation dataset."""

    def test_full_workflow(self):
        """Test complete workflow from creation to iteration."""
        # Create dataset
        data = [
            {
                "input": {"query": "What is Python?"},
                "expected": "Python is a programming language",
                "id": "q1",
            },
            {
                "input": {"query": "What is AI?"},
                "expected": "AI is artificial intelligence",
                "id": "q2",
                "metadata": {"category": "tech"},
            },
        ]
        dataset = EvaluationDataset.from_dicts(data)

        # Verify length
        assert len(dataset) == 2

        # Iterate and collect
        results = []
        for example in dataset:
            results.append(
                {
                    "id": example.id,
                    "input_query": example.input.get("query"),
                    "expected": example.expected,
                }
            )

        assert len(results) == 2
        assert results[0]["id"] == "q1"
        assert results[1]["id"] == "q2"
        assert results[1]["input_query"] == "What is AI?"

    def test_unicode_content(self):
        """Test that unicode content is handled correctly."""
        data = [
            {"input": {"query": "什么是人工智能？"}, "expected": "人工智能是..."},
            {"input": {"query": "Qu'est-ce que l'IA?"}, "expected": "L'IA est..."},
            {"input": {"query": "🤖 AI?"}, "expected": "🧠 Intelligence!"},
        ]
        dataset = EvaluationDataset.from_dicts(data)

        assert len(dataset) == 3
        assert dataset[0].input["query"] == "什么是人工智能？"
        assert dataset[2].expected == "🧠 Intelligence!"
