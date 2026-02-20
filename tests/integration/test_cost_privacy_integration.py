"""Integration tests for cost calculation with privacy mode support."""

import os
from typing import Any
from unittest.mock import Mock

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import ExampleMetrics, extract_llm_metrics


# Mock response classes
class MockAnthropicMessage:
    """Mock Anthropic Message response."""

    def __init__(self, content: str, input_tokens: int = 10, output_tokens: int = 20):
        self.model = "claude-3-haiku-20240307"
        self.content = [Mock(text=content)]
        # Create usage object with proper attributes
        self.usage = type(
            "Usage", (), {"input_tokens": input_tokens, "output_tokens": output_tokens}
        )()
        self.type = "message"
        self.role = "assistant"
        # Make the class name match what Anthropic uses
        self.__class__.__name__ = "Message"


class MockOpenAICompletion:
    """Mock OpenAI ChatCompletion response."""

    def __init__(
        self, content: str, prompt_tokens: int = 10, completion_tokens: int = 20
    ):
        self.model = "gpt-4o-mini"
        self.choices = [Mock(message=Mock(content=content), finish_reason="stop")]
        # Create usage object with proper attributes
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )()
        self.id = "chatcmpl-test"


@pytest.mark.asyncio
async def test_privacy_mode_with_length_storage():
    """Test that privacy mode stores and uses lengths for cost calculation."""
    # Create evaluator with privacy mode enabled
    evaluator = LocalEvaluator(
        metrics=["accuracy"], privacy_enabled=True, mock_mode_config={"enabled": False}
    )

    # Create test dataset
    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": "What is the capital of France?"},
                expected_output="Paris",
            ),
            EvaluationExample(
                input_data={
                    "messages": [
                        {"role": "user", "content": "Tell me a joke about programming"}
                    ]
                },
                expected_output="A funny joke",
            ),
        ]
    )

    # Mock function that returns standardized format
    async def mock_function(**kwargs) -> dict[str, Any]:
        input_data = kwargs
        if "text" in input_data:
            response = MockAnthropicMessage("Paris is the capital of France")
            return {"text": "Paris", "raw_response": response}
        else:
            response = MockOpenAICompletion("Why do programmers prefer dark mode?")
            return {"text": "A funny joke", "raw_response": response}

    # Run evaluation
    config = {"model": "claude-3-haiku-20240307"}
    result = await evaluator.evaluate(mock_function, config, dataset)

    # Verify that metrics were calculated despite privacy mode
    assert result.successful_examples > 0
    assert "accuracy" in result.aggregated_metrics

    # Check that token metrics were extracted
    for example_result in result.example_results:
        assert example_result.metrics.get("total_tokens", 0) > 0
        # In privacy mode, we should still have token counts from the response
        assert example_result.metrics.get("input_tokens", 0) > 0
        assert example_result.metrics.get("output_tokens", 0) > 0


@pytest.mark.asyncio
async def test_standardized_response_format():
    """Test that all integration functions return standardized format."""
    # Test response format from different integrations
    anthropic_response = MockAnthropicMessage("Test response")
    openai_response = MockOpenAICompletion("Test response")

    # Extract metrics from both response types
    anthropic_metrics = extract_llm_metrics(
        response=anthropic_response,
        model_name="claude-3-haiku-20240307",
        response_text="Test response",
    )

    openai_metrics = extract_llm_metrics(
        response=openai_response,
        model_name="gpt-4o-mini",
        response_text="Test response",
    )

    # Verify both responses produce valid metrics
    assert anthropic_metrics.tokens.input_tokens == 10
    assert anthropic_metrics.tokens.output_tokens == 20
    assert anthropic_metrics.tokens.total_tokens == 30

    assert openai_metrics.tokens.input_tokens == 10
    assert openai_metrics.tokens.output_tokens == 20
    assert openai_metrics.tokens.total_tokens == 30


def test_anthropic_detection_logic():
    """Test improved Anthropic response detection."""
    from traigent.evaluators.metrics_tracker import ResponseHandlerFactory

    handler_chain = ResponseHandlerFactory.create_handler_chain()

    # Test Anthropic response
    anthropic_response = MockAnthropicMessage("Test")
    metrics = handler_chain.handle(anthropic_response)
    assert metrics is not None
    assert metrics.tokens.input_tokens == 10
    assert metrics.tokens.output_tokens == 20

    # Test OpenAI response (should not be handled by Anthropic handler)
    openai_response = MockOpenAICompletion("Test")
    metrics = handler_chain.handle(openai_response)
    assert metrics is not None
    assert metrics.tokens.input_tokens == 10  # from prompt_tokens
    assert metrics.tokens.output_tokens == 20  # from completion_tokens

    # Test that responses are handled by correct handlers
    # Create a response that looks like Anthropic but has OpenAI structure
    ambiguous_response = Mock(
        model="claude-3-haiku",
        choices=[Mock()],  # OpenAI structure
        usage=Mock(prompt_tokens=5, completion_tokens=10),  # OpenAI tokens
    )
    metrics = handler_chain.handle(ambiguous_response)
    # Should be handled as OpenAI, not Anthropic
    assert metrics.tokens.input_tokens == 5
    assert metrics.tokens.output_tokens == 10


def test_error_visibility_in_cost_calculation():
    """Test that errors in cost calculation are visible."""
    import logging

    from traigent.evaluators.metrics_tracker import CostCalculator

    # Set up logging capture
    logging.getLogger("traigent.evaluators.metrics_tracker")

    # Create a calculator
    calculator = CostCalculator()

    # Create metrics with invalid model name
    metrics = ExampleMetrics()
    metrics.tokens.input_tokens = 100
    metrics.tokens.output_tokens = 200

    # Enable debug mode for error visibility
    os.environ["TRAIGENT_DEBUG"] = "true"

    # In mock mode, costs should be 0 regardless of model
    # But we should see error logging for invalid models
    calculator.calculate_cost(
        metrics,
        model_name="invalid-model-xyz-123",
        original_prompt="Test prompt",
        response_text="Test response",
    )

    # In mock mode, cost should be 0
    assert metrics.cost.total_cost == 0.0

    # Clean up
    del os.environ["TRAIGENT_DEBUG"]


def test_token_estimation_with_tiktoken():
    """No post-call token estimation should occur without usage metadata."""
    from traigent.evaluators.metrics_tracker import CostCalculator

    calculator = CostCalculator()
    metrics = ExampleMetrics()

    # Test with privacy mode lengths
    prompt_length = 100  # characters
    response_length = 200  # characters

    # With no token usage metadata and no raw prompt/response text provided,
    # post-call tracking must not invent token counts heuristically.
    calculator._try_unified_cost_calculation(
        metrics,
        model_name="gpt-4o-mini",
        original_prompt=None,
        response_text=None,
        prompt_length=prompt_length,
        response_length=response_length,
    )

    assert metrics.tokens.input_tokens == 0
    assert metrics.tokens.output_tokens == 0
    assert metrics.tokens.total_tokens == 0
    assert metrics.cost.total_cost == 0.0


@pytest.mark.asyncio
async def test_end_to_end_flow_with_mock_mode():
    """Test complete flow from function call to metrics extraction in mock mode."""
    os.environ["TRAIGENT_MOCK_LLM"] = "true"

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        privacy_enabled=False,
        detailed=True,
        mock_mode_config={"enabled": True, "override_evaluator": False},
    )

    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": "Test question"}, expected_output="Test answer"
            )
        ]
    )

    # Mock function simulating SDK response format
    async def mock_target(**kwargs) -> dict[str, Any]:
        return {
            "text": "Test answer",
            "raw_response": MockOpenAICompletion("Test answer", 15, 25),
        }

    config = {"model": "gpt-4o-mini"}
    result = await evaluator.evaluate(mock_target, config, dataset)

    # Verify end-to-end flow
    assert result.successful_examples == 1
    assert result.total_examples == 1
    assert result.example_results[0].success
    assert result.example_results[0].actual_output["text"] == "Test answer"

    # Check metrics extraction
    example_metrics = result.example_results[0].metrics
    assert example_metrics["accuracy"] == 1.0  # Exact match
    assert example_metrics["total_tokens"] == 40  # 15 + 25
    assert example_metrics["total_cost"] == 0.0  # Mock mode

    # Clean up
    del os.environ["TRAIGENT_MOCK_LLM"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
