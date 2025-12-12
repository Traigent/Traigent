#!/usr/bin/env python3
"""Tests for GTM & Acquisition Agent use-case."""

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
USE_CASES_DIR = PROJECT_ROOT / "use-cases" / "gtm-acquisition"
sys.path.insert(0, str(PROJECT_ROOT))


def load_module_from_path(name: str, path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agent_module():
    """Load the GTM agent module."""
    return load_module_from_path("gtm_agent", USE_CASES_DIR / "agent" / "gtm_agent.py")


@pytest.fixture
def evaluator_module():
    """Load the GTM evaluator module."""
    return load_module_from_path(
        "gtm_evaluator", USE_CASES_DIR / "eval" / "evaluator.py"
    )


@pytest.fixture
def dataset_path():
    """Path to the leads dataset."""
    return USE_CASES_DIR / "datasets" / "leads_dataset.jsonl"


class TestGTMAgentImports:
    """Test that agent module imports correctly."""

    def test_agent_module_loads(self, agent_module):
        """Agent module should load without errors."""
        assert agent_module is not None

    def test_agent_function_exists(self, agent_module):
        """Agent function should be defined."""
        assert hasattr(agent_module, "gtm_outreach_agent")
        assert callable(agent_module.gtm_outreach_agent)

    def test_format_lead_context_exists(self, agent_module):
        """Helper function should be defined."""
        assert hasattr(agent_module, "format_lead_context")
        assert callable(agent_module.format_lead_context)


class TestGTMEvaluatorImports:
    """Test that evaluator module imports correctly."""

    def test_evaluator_module_loads(self, evaluator_module):
        """Evaluator module should load without errors."""
        assert evaluator_module is not None

    def test_evaluator_class_exists(self, evaluator_module):
        """Evaluator class should be defined."""
        assert hasattr(evaluator_module, "MessageQualityEvaluator")

    def test_evaluator_instantiates(self, evaluator_module):
        """Evaluator should instantiate without errors."""
        evaluator = evaluator_module.MessageQualityEvaluator()
        assert evaluator is not None


class TestGTMDataset:
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

        # Check for input field with lead data
        assert "input" in data, "Missing 'input' field"
        input_data = data["input"]
        assert "lead" in input_data, "Missing 'lead' in input"

    def test_dataset_size(self, dataset_path):
        """Dataset should have sufficient entries for evaluation."""
        with open(dataset_path) as f:
            count = sum(1 for _ in f)

        # Should have at least 100 entries for statistical significance
        assert count >= 100, f"Dataset has only {count} entries, expected >= 100"


class TestGTMAgentExecution:
    """Test agent execution in mock mode."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for full agent execution",
    )
    def test_agent_runs_with_sample_lead(self, agent_module):
        """Agent should run and return a message for a sample lead."""
        sample_lead = {
            "name": "Test User",
            "title": "VP of Engineering",
            "company": "TestCorp",
            "industry": "SaaS",
            "company_size": "100-200",
            "recent_news": "Just raised Series B",
            "pain_points": ["scaling infrastructure", "developer productivity"],
        }

        result = agent_module.gtm_outreach_agent(
            lead=sample_lead,
            product="AI DevOps Platform",
            sender_name="Test SDR",
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 50, "Response too short"

    def test_format_lead_context(self, agent_module):
        """Lead context formatter should work correctly."""
        lead = {
            "name": "John Doe",
            "title": "CTO",
            "company": "Acme Inc",
            "industry": "FinTech",
        }

        context = agent_module.format_lead_context(lead)

        assert "John Doe" in context
        assert "CTO" in context
        assert "Acme Inc" in context
        assert "FinTech" in context


class TestGTMEvaluatorExecution:
    """Test evaluator execution."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY for LLM-as-judge evaluation",
    )
    def test_evaluator_evaluate_sample(self, evaluator_module):
        """Evaluator should evaluate a sample correctly."""
        evaluator = evaluator_module.MessageQualityEvaluator()

        # Create sample input/output
        sample_input = {
            "lead": {
                "name": "Test User",
                "title": "VP Engineering",
                "company": "TestCorp",
                "industry": "SaaS",
            },
            "product": "AI Platform",
            "sender_name": "Test SDR",
        }

        sample_output = """Hi Test,

Congrats on the recent growth at TestCorp! As VP Engineering, I imagine scaling your AI infrastructure is top of mind.

Our AI Platform helps SaaS companies like yours accelerate development by 40%. Would you be open to a quick chat?

Best,
Test"""

        result = evaluator.evaluate_sample(sample_input, sample_output)

        assert result is not None
        assert "message_quality" in result or "overall" in result
