"""Tests for OptimizedFunction dataset loading functionality.

Tests loading datasets from various sources including objects, files, and lists.
"""

import json
from pathlib import Path

import pytest

from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ConfigurationError

from .test_fixtures import (
    create_dataset_file,
    create_test_dataset,
)


class TestDatasetLoading:
    """Test dataset loading functionality."""

    @pytest.fixture(autouse=True)
    def dataset_root(self, monkeypatch, tmp_path_factory):
        root = tmp_path_factory.mktemp("dataset_root")
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(root))
        return root

    def test_load_dataset_from_object(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test loading dataset from Dataset object."""
        dataset = create_test_dataset(5)

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=dataset,
        )

        loaded_dataset = opt_func._load_dataset()
        assert isinstance(loaded_dataset, Dataset)
        assert len(loaded_dataset) == 5
        assert loaded_dataset[0].input_data["text"] == "input_0"

    def test_load_dataset_from_json_file(
        self,
        simple_function,
        sample_config_space,
        sample_objectives,
        dataset_root,
    ):
        """Test loading dataset from JSON file."""
        dataset_path = create_dataset_file(dataset_root, "test_dataset.json", size=3)

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=dataset_path,
        )

        loaded_dataset = opt_func._load_dataset()
        assert isinstance(loaded_dataset, Dataset)
        assert len(loaded_dataset) == 3

    def test_load_dataset_from_jsonl_file(
        self,
        simple_function,
        sample_config_space,
        sample_objectives,
        dataset_root,
    ):
        """Test loading dataset from JSONL file."""
        dataset_path = Path(dataset_root) / "test_dataset.jsonl"

        examples = []
        for i in range(4):
            examples.append(
                json.dumps({"input": {"text": f"input_{i}"}, "output": f"output_{i}"})
            )

        dataset_path.write_text("\n".join(examples), encoding="utf-8")

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=str(dataset_path),
        )

        loaded_dataset = opt_func._load_dataset()
        assert isinstance(loaded_dataset, Dataset)
        assert len(loaded_dataset) == 4

    def test_load_dataset_from_list(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test error when trying to load dataset from list of dicts (unsupported)."""
        examples = [
            {"input_data": {"text": "hello"}, "expected_output": "HELLO"},
            {"input_data": {"text": "world"}, "expected_output": "WORLD"},
        ]

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=examples,
        )

        # Should raise ConfigurationError for unsupported list of dicts
        with pytest.raises(ConfigurationError, match="Failed to load dataset"):
            opt_func._load_dataset()

    def test_load_dataset_from_evaluation_examples(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test error when trying to load dataset from EvaluationExample list (unsupported)."""
        examples = [
            EvaluationExample(input_data={"text": "test1"}, expected_output="TEST1"),
            EvaluationExample(input_data={"text": "test2"}, expected_output="TEST2"),
        ]

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=examples,
        )

        # Should raise ConfigurationError for unsupported EvaluationExample list
        with pytest.raises(ConfigurationError, match="Failed to load dataset"):
            opt_func._load_dataset()

    def test_load_dataset_invalid_type(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test error with invalid dataset type."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=12345,  # Invalid type
        )

        with pytest.raises(ConfigurationError, match="Invalid dataset type"):
            opt_func._load_dataset()

    def test_load_dataset_nonexistent_file(
        self,
        simple_function,
        sample_config_space,
        sample_objectives,
        dataset_root,
    ):
        """Test error with nonexistent file."""
        nonexistent_file = Path(dataset_root) / "missing" / "dataset.json"

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=str(nonexistent_file),
        )

        with pytest.raises(ConfigurationError, match="Failed to load dataset"):
            opt_func._load_dataset()

    def test_load_dataset_invalid_json(
        self,
        simple_function,
        sample_config_space,
        sample_objectives,
        dataset_root,
    ):
        """Test error with invalid JSON file."""
        dataset_path = Path(dataset_root) / "invalid.json"
        dataset_path.write_text("{ invalid json }", encoding="utf-8")

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=str(dataset_path),
        )

        with pytest.raises(Exception):  # noqa: B017 - JSON decode error
            opt_func._load_dataset()

    def test_load_dataset_empty_list(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test loading empty dataset from empty list of files."""
        # Empty list means list of empty JSONL files
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=[],  # Empty list of dataset files
        )

        loaded_dataset = opt_func._load_dataset()
        assert isinstance(loaded_dataset, Dataset)
        assert len(loaded_dataset) == 0

    def test_dataset_lazy_loading(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test that dataset is loaded lazily when needed."""
        dataset = create_test_dataset(3)

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=dataset,
        )

        # Load dataset multiple times - should return same object
        loaded1 = opt_func._load_dataset()
        loaded2 = opt_func._load_dataset()

        # Both should be the same dataset object
        assert loaded1 is dataset
        assert loaded2 is dataset

    def test_dataset_with_custom_structure(
        self,
        simple_function,
        sample_config_space,
        sample_objectives,
        dataset_root,
    ):
        """Test loading dataset with custom structure."""
        dataset_path = Path(dataset_root) / "custom.json"
        dataset_payload = {
            "examples": [
                {
                    "input": {
                        "question": "What is AI?",
                        "context": "AI is...",
                    },
                    "output": "Artificial Intelligence",
                }
            ],
            "metadata": {"source": "unit-test"},
        }
        dataset_path.write_text(json.dumps(dataset_payload), encoding="utf-8")

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=str(dataset_path),
        )

        loaded_dataset = opt_func._load_dataset()
        assert len(loaded_dataset) == 1
        assert "question" in loaded_dataset[0].input_data
        assert "context" in loaded_dataset[0].input_data
