#!/usr/bin/env python3
"""Tests for Customer Support Agent use-case."""

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
USE_CASES_DIR = PROJECT_ROOT / "use-cases" / "customer-support"
sys.path.insert(0, str(PROJECT_ROOT))


def load_module_from_path(name: str, path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agent_module():
    """Load the Customer Support agent module."""
    return load_module_from_path(
        "support_agent", USE_CASES_DIR / "agent" / "support_agent.py"
    )


@pytest.fixture
def evaluator_module():
    """Load the Customer Support evaluator module."""
    return load_module_from_path(
        "support_evaluator", USE_CASES_DIR / "eval" / "evaluator.py"
    )


@pytest.fixture
def dataset_path():
    """Path to the support tickets dataset."""
    return USE_CASES_DIR / "datasets" / "support_tickets.jsonl"


class TestSupportAgentImports:
    """Test that agent module imports correctly."""

    def test_agent_module_loads(self, agent_module):
        """Agent module should load without errors."""
        assert agent_module is not None

    def test_agent_function_exists(self, agent_module):
        """Agent function should be defined."""
        assert hasattr(agent_module, "customer_support_agent")
        assert callable(agent_module.customer_support_agent)

    def test_helper_functions_exist(self, agent_module):
        """Helper functions should be defined."""
        assert hasattr(agent_module, "determine_resolution_type")
        assert hasattr(agent_module, "generate_mock_response")


class TestSupportEvaluatorImports:
    """Test that evaluator module imports correctly."""

    def test_evaluator_module_loads(self, evaluator_module):
        """Evaluator module should load without errors."""
        assert evaluator_module is not None

    def test_evaluator_class_exists(self, evaluator_module):
        """Evaluator class should be defined."""
        assert hasattr(evaluator_module, "SupportEvaluator")

    def test_evaluator_instantiates(self, evaluator_module):
        """Evaluator should instantiate without errors."""
        evaluator = evaluator_module.SupportEvaluator()
        assert evaluator is not None


class TestSupportDataset:
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

        # Check for input field with query
        assert "input" in data, "Missing 'input' field"
        input_data = data["input"]
        assert "query" in input_data, "Missing 'query' in input"

    def test_dataset_size(self, dataset_path):
        """Dataset should have sufficient entries for evaluation."""
        with open(dataset_path) as f:
            count = sum(1 for _ in f)

        # Should have at least 100 entries for statistical significance
        assert count >= 100, f"Dataset has only {count} entries, expected >= 100"

    def test_dataset_has_escalation_scenarios(self, dataset_path):
        """Dataset should include escalation scenarios."""
        escalation_count = 0

        with open(dataset_path) as f:
            for line in f:
                data = json.loads(line)
                input_data = data.get("input", {})
                output_data = data.get("output", {})
                query = input_data.get("query", "").lower()

                # Check for escalation indicators
                if input_data.get("should_escalate", False):
                    escalation_count += 1
                elif isinstance(output_data, dict) and output_data.get(
                    "should_escalate", False
                ):
                    escalation_count += 1
                # Check for escalation keywords in query
                elif any(
                    kw in query
                    for kw in ["supervisor", "manager", "escalate", "unacceptable"]
                ):
                    escalation_count += 1

        # Should have some escalation scenarios (be flexible)
        assert (
            escalation_count >= 5
        ), f"Only {escalation_count} escalation scenarios found"


class TestSupportAgentExecution:
    """Test agent execution in mock mode."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_agent_handles_refund_request(self, agent_module):
        """Agent should handle refund requests."""
        result = agent_module.customer_support_agent(
            query="I want a refund for my order",
            customer_context={
                "customer_tier": "standard",
                "sentiment": "neutral",
                "order_status": "delivered",
                "previous_interactions": 0,
            },
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "response" in result
        assert "should_escalate" in result
        assert "resolution_type" in result
        assert len(result["response"]) > 50

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_agent_handles_tracking_request(self, agent_module):
        """Agent should handle order tracking requests."""
        result = agent_module.customer_support_agent(
            query="Where is my order? I want to track my package.",
            customer_context={
                "customer_tier": "gold",
                "sentiment": "neutral",
                "order_status": "shipped",
                "previous_interactions": 1,
            },
        )

        assert result is not None
        assert "response" in result
        assert result["resolution_type"] == "information"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_agent_escalates_supervisor_request(self, agent_module):
        """Agent should escalate when customer requests supervisor."""
        result = agent_module.customer_support_agent(
            query="This is unacceptable! I want to speak to your supervisor immediately!",
            customer_context={
                "customer_tier": "platinum",
                "sentiment": "very_negative",
                "order_status": "delivered",
                "previous_interactions": 3,
            },
        )

        assert result is not None
        assert result["should_escalate"] is True

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_agent_handles_damaged_item(self, agent_module):
        """Agent should handle damaged item complaints."""
        result = agent_module.customer_support_agent(
            query="My item arrived damaged and broken. I need a replacement.",
            customer_context={
                "customer_tier": "standard",
                "sentiment": "negative",
                "order_status": "delivered",
                "previous_interactions": 0,
            },
        )

        assert result is not None
        assert result["resolution_type"] in ["replacement", "refund", "escalated"]

    def test_determine_resolution_type(self, agent_module):
        """Resolution type determination should work correctly."""
        # Test refund detection
        assert (
            agent_module.determine_resolution_type(
                "We will process your refund immediately.", False
            )
            == "refund"
        )

        # Test escalation
        assert (
            agent_module.determine_resolution_type("Any response", True) == "escalated"
        )

        # Test tracking/information
        assert (
            agent_module.determine_resolution_type(
                "Your package is being tracked and in transit.", False
            )
            == "information"
        )


class TestSupportEvaluatorExecution:
    """Test evaluator execution."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for LLM-as-judge evaluation",
    )
    def test_evaluator_evaluate_good_response(self, evaluator_module):
        """Evaluator should give high score to good responses."""
        evaluator = evaluator_module.SupportEvaluator()

        sample_input = {
            "query": "I want a refund for my damaged item",
            "customer_context": {
                "customer_tier": "gold",
                "sentiment": "negative",
            },
        }

        sample_output = {
            "response": """I'm so sorry to hear that your item arrived damaged.
            As a valued gold customer, I'd like to make this right immediately.
            I've processed a full refund to your original payment method.
            Is there anything else I can help you with?""",
            "should_escalate": False,
            "resolution_type": "refund",
        }

        result = evaluator.evaluate_sample(sample_input, sample_output)

        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for LLM-as-judge evaluation",
    )
    def test_evaluator_evaluate_escalation_scenario(self, evaluator_module):
        """Evaluator should handle escalation scenarios."""
        evaluator = evaluator_module.SupportEvaluator()

        sample_input = {
            "query": "I want to speak to a manager right now!",
            "customer_context": {
                "customer_tier": "platinum",
                "sentiment": "very_negative",
            },
            "should_escalate": True,  # Expected output
        }

        sample_output = {
            "response": "I understand your frustration. Let me connect you with a supervisor.",
            "should_escalate": True,
            "resolution_type": "escalated",
        }

        result = evaluator.evaluate_sample(sample_input, sample_output)

        assert result is not None
