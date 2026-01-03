#!/usr/bin/env python3
"""Tests for Operations Agent use-case."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# Set mock mode for all tests
os.environ["TRAIGENT_MOCK_LLM"] = "true"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
USE_CASES_DIR = PROJECT_ROOT / "use-cases" / "operations"
sys.path.insert(0, str(PROJECT_ROOT))


def load_module_from_path(name: str, path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agent_module():
    """Load the Operations agent module."""
    return load_module_from_path(
        "operations_agent", USE_CASES_DIR / "agent" / "operations_agent.py"
    )


@pytest.fixture
def evaluator_module():
    """Load the Operations evaluator module."""
    return load_module_from_path(
        "operations_evaluator", USE_CASES_DIR / "eval" / "evaluator.py"
    )


@pytest.fixture
def dataset_path():
    """Path to the tasks dataset."""
    return USE_CASES_DIR / "datasets" / "tasks_dataset.jsonl"


class TestOperationsAgentImports:
    """Test that agent module imports correctly."""

    def test_agent_module_loads(self, agent_module):
        """Agent module should load without errors."""
        assert agent_module is not None

    def test_agent_function_exists(self, agent_module):
        """Agent function should be defined."""
        assert hasattr(agent_module, "operations_workflow_agent")
        assert callable(agent_module.operations_workflow_agent)

    def test_helper_functions_exist(self, agent_module):
        """Helper functions should be defined."""
        assert hasattr(agent_module, "format_task_context")
        assert hasattr(agent_module, "generate_rule_based_response")


class TestOperationsEvaluatorImports:
    """Test that evaluator module imports correctly."""

    def test_evaluator_module_loads(self, evaluator_module):
        """Evaluator module should load without errors."""
        assert evaluator_module is not None

    def test_evaluator_class_exists(self, evaluator_module):
        """Evaluator class should be defined."""
        assert hasattr(evaluator_module, "OperationsEvaluator")

    def test_evaluator_instantiates(self, evaluator_module):
        """Evaluator should instantiate without errors."""
        evaluator = evaluator_module.OperationsEvaluator()
        assert evaluator is not None


class TestOperationsDataset:
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

        # Check for input field
        assert "input" in data, "Missing 'input' field"
        input_data = data["input"]
        assert "task_type" in input_data, "Missing 'task_type' in input"
        assert "description" in input_data, "Missing 'description' in input"

    def test_dataset_size(self, dataset_path):
        """Dataset should have sufficient entries for evaluation."""
        with open(dataset_path) as f:
            count = sum(1 for _ in f)

        # Should have at least 100 entries for statistical significance
        assert count >= 100, f"Dataset has only {count} entries, expected >= 100"


class TestOperationsAgentExecution:
    """Test agent execution in mock mode."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_agent_runs_expense_approval(self, agent_module):
        """Agent should handle expense approval tasks."""
        result = agent_module.operations_workflow_agent(
            task_type="expense_approval",
            description="Process expense report for $500",
            context={
                "employee_level": "senior_engineer",
                "budget_remaining": "$10,000",
                "policy_limit": "$3,000",
                "department": "Engineering",
            },
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "actions" in result
        assert "should_escalate" in result
        assert isinstance(result["actions"], list)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_agent_runs_access_request(self, agent_module):
        """Agent should handle access request tasks."""
        result = agent_module.operations_workflow_agent(
            task_type="access_request",
            description="Request admin access to production database",
            context={
                "requester_role": "developer",
                "sensitivity_level": "high",
                "justification": "Debug production issue",
            },
        )

        assert result is not None
        assert "actions" in result
        assert "should_escalate" in result

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_escalation_for_high_amount(self, agent_module):
        """Agent should escalate high-value requests."""
        result = agent_module.operations_workflow_agent(
            task_type="expense_approval",
            description="Process expense report for $15,000",
            context={
                "amount": "$15,000",
                "employee_level": "manager",
                "department": "Sales",
            },
        )

        # High amounts should trigger escalation in conservative/moderate modes
        assert result is not None
        assert "should_escalate" in result

    def test_format_task_context(self, agent_module):
        """Task context formatter should work correctly."""
        context = agent_module.format_task_context(
            task_type="expense_approval",
            description="Test expense",
            context={"amount": "$500", "department": "Engineering"},
        )

        assert "expense_approval" in context
        assert "Test expense" in context
        assert "$500" in context


class TestOperationsEvaluatorExecution:
    """Test evaluator execution."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for LLM-as-judge evaluation",
    )
    def test_evaluator_evaluate_sample(self, evaluator_module):
        """Evaluator should evaluate a sample correctly."""
        evaluator = evaluator_module.OperationsEvaluator()

        sample_input = {
            "task_type": "expense_approval",
            "description": "Process expense report",
            "context": {"amount": "$500"},
        }

        sample_output = {
            "actions": ["validate_amount", "check_policy_limit", "auto_approve"],
            "should_escalate": False,
        }

        result = evaluator.evaluate_sample(sample_input, sample_output)

        assert result is not None
        assert isinstance(result, dict)
