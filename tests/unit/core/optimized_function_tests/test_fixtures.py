"""Shared test fixtures and utilities for OptimizedFunction tests.

This module provides simplified mock implementations and fixtures
for testing OptimizedFunction without complex mock infrastructure.
"""

import json
import tempfile
from pathlib import Path

import pytest

from traigent.api.types import ExampleResult
from traigent.evaluators.base import Dataset, EvaluationExample


class SimpleMockFunction:
    """Simplified mock function for testing optimization."""

    def __init__(self, return_value="MOCK_OUTPUT"):
        self.call_count = 0
        self.last_args = None
        self.last_kwargs = None
        self.return_value = return_value
        self.__name__ = "mock_function"

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return self.return_value


def create_simple_evaluator():
    """Create a simple evaluator function for testing."""

    def evaluator(func, config, example):
        try:
            # Extract input text
            text = example.input_data.get("text", "")

            # Call function with config
            result = func(text, **config)

            # Simple accuracy metric
            accuracy = 1.0 if result == example.expected_output else 0.5

            return ExampleResult(
                example_id="test",
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=result,
                metrics={"accuracy": accuracy},
                execution_time=0.01,
                success=True,
                error_message=None,
            )
        except Exception as e:
            return ExampleResult(
                example_id="test",
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=None,
                metrics={"accuracy": 0.0},
                execution_time=0.01,
                success=False,
                error_message=str(e),
            )

    return evaluator


def create_test_dataset(size=3):
    """Create a simple test dataset."""
    examples = []
    for i in range(size):
        examples.append(
            EvaluationExample(
                input_data={"text": f"input_{i}"}, expected_output=f"OUTPUT_{i}"
            )
        )
    return Dataset(examples)


def create_dataset_file(directory, filename="test_dataset.json", size=3):
    """Create a test dataset file in JSONL format."""
    dataset_path = Path(directory) / filename

    examples = [
        {"input": {"text": f"input_{i}"}, "output": f"output_{i}"} for i in range(size)
    ]

    if dataset_path.suffix.lower() == ".json":
        payload = {
            "name": "test_dataset",
            "description": "Test dataset",
            "examples": examples,
        }
        dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        lines = [json.dumps(example) for example in examples]
        dataset_path.write_text("\n".join(lines), encoding="utf-8")

    return str(dataset_path)


@pytest.fixture
def simple_function():
    """Provide a simple mock function."""
    return SimpleMockFunction()


@pytest.fixture
def sample_config_space():
    """Standard configuration space for tests."""
    return {
        "temperature": [0.0, 0.5, 1.0],
        "max_tokens": [100, 200],
        "model": ["gpt-3.5", "gpt-4"],
    }


@pytest.fixture
def sample_objectives():
    """Standard objectives for tests."""
    return ["accuracy", "cost"]


@pytest.fixture
def sample_dataset():
    """Standard test dataset."""
    return create_test_dataset(3)


@pytest.fixture
def temp_dataset_file():
    """Create a temporary dataset file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_dataset_file(tmpdir)
        yield dataset_path
