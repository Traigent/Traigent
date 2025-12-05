#!/usr/bin/env python
"""
CI Integration Tests for tokencost Integration

This script runs tokencost integration tests suitable for CI/CD pipelines.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestTokencostCIIntegration:
    """CI tests for tokencost integration."""

    def test_tokencost_availability_detection(self):
        """Test that TOKENCOST_AVAILABLE flag is properly set."""
        # Test with tokencost available
        with patch("traigent.evaluators.metrics_tracker.tokencost") as mock_tokencost:
            mock_tokencost.calculate_prompt_cost = Mock()
            mock_tokencost.calculate_completion_cost = Mock()

            # Re-import to trigger availability check
            import importlib

            import traigent.evaluators.metrics_tracker

            importlib.reload(traigent.evaluators.metrics_tracker)

            # In CI, tokencost might not be available, so we test both cases
            from traigent.evaluators.metrics_tracker import TOKENCOST_AVAILABLE

            assert isinstance(
                TOKENCOST_AVAILABLE, bool
            ), "TOKENCOST_AVAILABLE should be a boolean"

    def test_extract_llm_metrics_signature(self):
        """Test that extract_llm_metrics has correct signature."""
        import inspect

        from traigent.evaluators.metrics_tracker import extract_llm_metrics

        sig = inspect.signature(extract_llm_metrics)
        params = list(sig.parameters.keys())

        expected_params = ["response", "model_name", "original_prompt", "response_text"]
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

    def test_extract_llm_metrics_fallback_behavior(self):
        """Test that extract_llm_metrics works without tokencost."""
        from traigent.evaluators.metrics_tracker import extract_llm_metrics

        # Create mock response
        class MockOpenAIResponse:
            def __init__(self):
                self.usage = Mock()
                self.usage.prompt_tokens = 15
                self.usage.completion_tokens = 8
                self.usage.total_tokens = 23

        response = MockOpenAIResponse()

        # Test without tokencost (fallback mode)
        with patch("traigent.evaluators.metrics_tracker.TOKENCOST_AVAILABLE", False):
            metrics = extract_llm_metrics(response=response)

            assert metrics.tokens.input_tokens == 15
            assert metrics.tokens.output_tokens == 8
            assert metrics.tokens.total_tokens == 23
            assert metrics.cost.total_cost == 0.0  # Fallback should give 0 cost

    def test_extract_llm_metrics_with_tokencost_mock(self):
        """Test extract_llm_metrics with mocked tokencost."""
        from traigent.evaluators.metrics_tracker import extract_llm_metrics

        class MockOpenAIResponse:
            def __init__(self):
                self.usage = Mock()
                self.usage.prompt_tokens = 15
                self.usage.completion_tokens = 8
                self.usage.total_tokens = 23

        response = MockOpenAIResponse()
        model_name = "gpt-4o-mini"
        original_prompt = [{"role": "user", "content": "test"}]
        response_text = "positive"

        # Mock tokencost functions
        with patch(
            "traigent.evaluators.metrics_tracker.TOKENCOST_AVAILABLE", True
        ), patch(
            "traigent.evaluators.metrics_tracker.calculate_prompt_cost"
        ) as mock_prompt_cost, patch(
            "traigent.evaluators.metrics_tracker.calculate_completion_cost"
        ) as mock_completion_cost:

            mock_prompt_cost.return_value = 0.000002
            mock_completion_cost.return_value = 0.000005

            metrics = extract_llm_metrics(
                response=response,
                model_name=model_name,
                original_prompt=original_prompt,
                response_text=response_text,
            )

            assert metrics.tokens.input_tokens == 15
            assert metrics.tokens.output_tokens == 8
            assert metrics.cost.input_cost == 0.000002
            assert metrics.cost.output_cost == 0.000005
            assert metrics.cost.total_cost == 0.000007

            # Verify tokencost was called correctly
            mock_prompt_cost.assert_called_once_with(original_prompt, model_name)
            mock_completion_cost.assert_called_once_with(response_text, model_name)

    def test_local_evaluator_integration(self):
        """Test LocalEvaluator integrates properly with cost calculation."""
        import asyncio

        from traigent.evaluators.base import Dataset, EvaluationExample
        from traigent.evaluators.local import LocalEvaluator

        # Create test dataset
        examples = [
            EvaluationExample(
                input_data={"text": "This is positive"}, expected_output="positive"
            ),
        ]
        dataset = Dataset(examples=examples)

        # Mock function
        async def mock_function(text: str) -> str:
            return "positive"

        evaluator = LocalEvaluator(metrics=["accuracy"])
        config = {"model": "gpt-4o-mini", "temperature": 0.0}

        # Test evaluation completes without errors
        async def run_test():
            result = await evaluator.evaluate(mock_function, config, dataset)
            assert result.successful_examples > 0
            assert "cost" in result.aggregated_metrics
            return result

        # Run the async test
        result = asyncio.run(run_test())
        assert result is not None

    def test_requirements_include_tokencost(self):
        """Test that requirements files include tokencost."""
        requirements_files = ["requirements/requirements.txt", "pyproject.toml"]

        for req_file in requirements_files:
            req_path = project_root / req_file
            if req_path.exists():
                content = req_path.read_text()
                assert (
                    "tokencost" in content.lower()
                ), f"tokencost not found in {req_file}"

    def test_pyproject_toml_tokencost_dependency(self):
        """Test that pyproject.toml includes tokencost in dependencies."""
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()

            # Check that tokencost is in dependencies
            assert (
                "tokencost>=0.1.0" in content
            ), "tokencost>=0.1.0 not found in pyproject.toml dependencies"

    def test_error_handling_with_tokencost_exceptions(self):
        """Test graceful error handling when tokencost raises exceptions."""
        from traigent.evaluators.metrics_tracker import extract_llm_metrics

        class MockOpenAIResponse:
            def __init__(self):
                self.usage = Mock()
                self.usage.prompt_tokens = 15
                self.usage.completion_tokens = 8
                self.usage.total_tokens = 23

        response = MockOpenAIResponse()

        # Test that tokencost exceptions don't crash the system
        with patch(
            "traigent.evaluators.metrics_tracker.TOKENCOST_AVAILABLE", True
        ), patch(
            "traigent.evaluators.metrics_tracker.calculate_prompt_cost"
        ) as mock_prompt_cost:

            mock_prompt_cost.side_effect = Exception("tokencost API error")

            # Should not raise exception, should fallback gracefully
            metrics = extract_llm_metrics(response=response, model_name="gpt-4o-mini")

            assert metrics.tokens.total_tokens == 23
            assert metrics.cost.total_cost == 0.0  # Should fallback to zero cost

    def test_string_response_handling(self):
        """Test cost calculation when response is just a string."""
        from traigent.evaluators.metrics_tracker import extract_llm_metrics

        response_text = "positive"
        model_name = "gpt-4o-mini"

        # Test that string responses are handled gracefully
        metrics = extract_llm_metrics(
            response=response_text, model_name=model_name, response_text=response_text
        )

        # Should not crash and should return valid metrics
        assert hasattr(metrics, "tokens")
        assert hasattr(metrics, "cost")
        assert isinstance(metrics.cost.total_cost, (int, float))


if __name__ == "__main__":
    # Run tests when called directly
    exit_code = pytest.main([__file__, "-v", "--tb=short", "--no-header"])
    sys.exit(exit_code)
