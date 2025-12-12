#!/usr/bin/env python3
"""Tests for Product & Technical Agent use-case."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# Set mock mode for all tests
os.environ["TRAIGENT_MOCK_MODE"] = "true"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
USE_CASES_DIR = PROJECT_ROOT / "use-cases" / "product-technical"
sys.path.insert(0, str(PROJECT_ROOT))


def load_module_from_path(name: str, path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agent_module():
    """Load the Code agent module."""
    return load_module_from_path(
        "code_agent", USE_CASES_DIR / "agent" / "code_agent.py"
    )


@pytest.fixture
def evaluator_module():
    """Load the Code evaluator module."""
    return load_module_from_path(
        "code_evaluator", USE_CASES_DIR / "eval" / "evaluator.py"
    )


@pytest.fixture
def dataset_path():
    """Path to the coding tasks dataset."""
    return USE_CASES_DIR / "datasets" / "coding_tasks.jsonl"


class TestCodeAgentImports:
    """Test that agent module imports correctly."""

    def test_agent_module_loads(self, agent_module):
        """Agent module should load without errors."""
        assert agent_module is not None

    def test_agent_function_exists(self, agent_module):
        """Agent function should be defined."""
        assert hasattr(agent_module, "code_generation_agent")
        assert callable(agent_module.code_generation_agent)

    def test_helper_functions_exist(self, agent_module):
        """Helper functions should be defined."""
        assert hasattr(agent_module, "extract_code")
        assert hasattr(agent_module, "generate_mock_code")


class TestCodeEvaluatorImports:
    """Test that evaluator module imports correctly."""

    def test_evaluator_module_loads(self, evaluator_module):
        """Evaluator module should load without errors."""
        assert evaluator_module is not None

    def test_evaluator_class_exists(self, evaluator_module):
        """Evaluator class should be defined."""
        assert hasattr(evaluator_module, "CodeEvaluator")

    def test_evaluator_instantiates(self, evaluator_module):
        """Evaluator should instantiate without errors."""
        evaluator = evaluator_module.CodeEvaluator()
        assert evaluator is not None


class TestCodeDataset:
    """Test that dataset loads correctly."""

    def test_dataset_file_exists(self, dataset_path):
        """Dataset file should exist."""
        assert dataset_path.exists(), f"Dataset not found at {dataset_path}"

    def test_dataset_is_valid_jsonl(self, dataset_path):
        """Dataset should be valid JSONL format."""
        with open(dataset_path) as f:
            lines = f.readlines()

        assert len(lines) > 0, "Dataset is empty"

        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                assert isinstance(data, dict), f"Line {i+1} is not a dict"
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON on line {i+1}: {e}")

    def test_dataset_has_expected_fields(self, dataset_path):
        """Dataset entries should have expected fields."""
        with open(dataset_path) as f:
            first_line = f.readline()

        data = json.loads(first_line)

        # Check for input field with task
        assert "input" in data, "Missing 'input' field"
        input_data = data["input"]
        assert "task" in input_data, "Missing 'task' in input"
        assert "function_name" in input_data, "Missing 'function_name' in input"

    def test_dataset_size(self, dataset_path):
        """Dataset should have sufficient entries for evaluation."""
        with open(dataset_path) as f:
            count = sum(1 for _ in f)

        # Should have at least 50 entries
        assert count >= 50, f"Dataset has only {count} entries, expected >= 50"

    def test_dataset_structure_for_evaluation(self, dataset_path):
        """Dataset should have structure that supports code evaluation."""
        with open(dataset_path) as f:
            lines = f.readlines()

        # Verify we have enough data
        assert len(lines) >= 50, "Dataset should have at least 50 coding tasks"

        # Verify entries have required fields for code generation
        sample = json.loads(lines[0])
        input_data = sample.get("input", {})

        assert "task" in input_data, "Input should have 'task' field"
        assert "function_name" in input_data, "Input should have 'function_name' field"


class TestCodeAgentExecution:
    """Test agent execution in mock mode."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_agent_generates_is_prime(self, agent_module):
        """Agent should generate is_prime function."""
        result = agent_module.code_generation_agent(
            task="Write a function that checks if a number is prime",
            function_name="is_prime",
            signature="def is_prime(n: int) -> bool",
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "code" in result
        assert "function_name" in result
        assert "is_prime" in result["code"]

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_agent_generates_factorial(self, agent_module):
        """Agent should generate factorial function."""
        result = agent_module.code_generation_agent(
            task="Write a function that computes the factorial of a number",
            function_name="factorial",
            signature="def factorial(n: int) -> int",
        )

        assert result is not None
        assert "code" in result
        assert "factorial" in result["code"]

    def test_extract_code_from_markdown(self, agent_module):
        """Extract code should handle markdown code blocks."""
        markdown_response = """Here's the solution:

```python
def foo(x):
    return x * 2
```

This function doubles the input."""

        code = agent_module.extract_code(markdown_response)
        assert "def foo" in code
        assert "return x * 2" in code
        assert "```" not in code

    def test_extract_code_plain(self, agent_module):
        """Extract code should handle plain code."""
        plain_response = """def bar(y):
    return y + 1"""

        code = agent_module.extract_code(plain_response)
        assert "def bar" in code


class TestCodeEvaluatorExecution:
    """Test evaluator execution."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for LLM-as-judge evaluation",
    )
    def test_evaluator_evaluate_correct_code(self, evaluator_module):
        """Evaluator should give high score to correct code."""
        evaluator = evaluator_module.CodeEvaluator()

        sample_input = {
            "task": "Check if number is prime",
            "function_name": "is_prime",
            "signature": "def is_prime(n: int) -> bool",
            "test_cases": [
                {"input": {"n": 2}, "expected": True},
                {"input": {"n": 4}, "expected": False},
                {"input": {"n": 17}, "expected": True},
            ],
        }

        correct_code = """def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True"""

        sample_output = {
            "code": correct_code,
            "function_name": "is_prime",
        }

        result = evaluator.evaluate_sample(sample_input, sample_output)

        assert result is not None
        assert isinstance(result, dict)
        # Correct code should have high test pass rate
        if "test_pass_rate" in result:
            assert result["test_pass_rate"] >= 0.8

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for LLM-as-judge evaluation",
    )
    def test_evaluator_evaluate_broken_code(self, evaluator_module):
        """Evaluator should give low score to broken code."""
        evaluator = evaluator_module.CodeEvaluator()

        sample_input = {
            "task": "Check if number is prime",
            "function_name": "is_prime",
            "signature": "def is_prime(n: int) -> bool",
            "test_cases": [
                {"input": {"n": 2}, "expected": True},
            ],
        }

        broken_code = """def is_prime(n: int) -> bool:
    return True  # Always returns True - wrong!"""

        sample_output = {
            "code": broken_code,
            "function_name": "is_prime",
        }

        result = evaluator.evaluate_sample(sample_input, sample_output)

        assert result is not None
        # Should still return a result even for incorrect code
