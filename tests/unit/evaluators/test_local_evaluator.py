"""Unit tests for LocalEvaluator with focus on token estimation and metrics flow."""

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


class MockLLMResponse:
    """Mock LLM response with usage metadata."""

    def __init__(self, text: str, input_tokens: int = 100, output_tokens: int = 50):
        self.text = text
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )()

    def __str__(self):
        return self.text


class TestLocalEvaluatorTokenEstimation:
    """Test LocalEvaluator token estimation functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        examples = [
            EvaluationExample(
                input_data={"text": "This is a test input with some words"},
                expected_output="positive",
            ),
            EvaluationExample(
                input_data={
                    "text": "Another longer test input with more words to check"
                },
                expected_output="negative",
            ),
            EvaluationExample(input_data={"text": "Short"}, expected_output="neutral"),
        ]
        return Dataset(examples=examples, name="test_dataset")

    @pytest.fixture
    def evaluator(self):
        """Create a LocalEvaluator instance."""
        return LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

    @pytest.mark.asyncio
    async def test_token_estimation_for_string_outputs(self, evaluator, sample_dataset):
        """Test that string outputs get proper token estimates."""

        async def string_function(text: str) -> str:
            """Function that returns plain strings."""
            return "positive" if "test" in text else "negative"

        config = {}
        result = await evaluator.evaluate(string_function, config, sample_dataset)

        # Check that we have example results
        assert len(result.example_results) == 3

        # Check each example has token metrics
        for _i, example_result in enumerate(result.example_results):
            # Verify token metrics exist and are > 0
            assert "input_tokens" in example_result.metrics
            assert "output_tokens" in example_result.metrics
            assert "total_tokens" in example_result.metrics

            assert example_result.metrics["input_tokens"] > 0
            assert example_result.metrics["output_tokens"] > 0
            assert example_result.metrics["total_tokens"] > 0

            # Verify total = input + output
            assert (
                example_result.metrics["total_tokens"]
                == example_result.metrics["input_tokens"]
                + example_result.metrics["output_tokens"]
            )

    @pytest.mark.asyncio
    async def test_token_estimation_calculation_accuracy(
        self, evaluator, sample_dataset
    ):
        """Test that token estimation uses correct formula (1 token per 4 chars)."""

        async def known_output_function(text: str) -> str:
            """Function that returns known output for testing."""
            return "test_output"  # 11 characters = 2-3 tokens (11//4 = 2, max(1,2) = 2)

        config = {}
        result = await evaluator.evaluate(known_output_function, config, sample_dataset)

        # Check first example (input: "This is a test input with some words" = 37 chars = 9 tokens)
        first_result = result.example_results[0]
        expected_input_tokens = max(
            1, len(str(sample_dataset.examples[0].input_data)) // 4
        )
        expected_output_tokens = max(1, len("test_output") // 4)

        assert first_result.metrics["input_tokens"] == expected_input_tokens
        assert first_result.metrics["output_tokens"] == expected_output_tokens

    @pytest.mark.asyncio
    async def test_llm_response_metrics_extraction(self, evaluator, sample_dataset):
        """Test that LLM response objects with metadata are handled correctly."""

        async def llm_response_function(text: str) -> MockLLMResponse:
            """Function that returns mock LLM response objects."""
            return MockLLMResponse("positive", input_tokens=150, output_tokens=75)

        config = {}
        result = await evaluator.evaluate(llm_response_function, config, sample_dataset)

        # Check that real LLM metrics are used (not estimated)
        for example_result in result.example_results:
            assert example_result.metrics["input_tokens"] == 150
            assert example_result.metrics["output_tokens"] == 75
            assert example_result.metrics["total_tokens"] == 225

    @pytest.mark.asyncio
    async def test_mixed_outputs_handling(self, evaluator, sample_dataset):
        """Test handling mix of string and LLM response outputs."""

        call_count = 0

        async def mixed_function(text: str):
            """Function that alternates between strings and LLM responses."""
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return "string_output"  # String on odd calls
            else:
                return MockLLMResponse(
                    "llm_output", 200, 100
                )  # LLM response on even calls

        config = {}
        result = await evaluator.evaluate(mixed_function, config, sample_dataset)

        # First and third should have estimated tokens (strings)
        assert result.example_results[0].metrics["output_tokens"] == max(
            1, len("string_output") // 4
        )
        assert result.example_results[2].metrics["output_tokens"] == max(
            1, len("string_output") // 4
        )

        # Second should have real LLM tokens
        assert result.example_results[1].metrics["input_tokens"] == 200
        assert result.example_results[1].metrics["output_tokens"] == 100

    @pytest.mark.asyncio
    async def test_empty_string_handling(self, evaluator, sample_dataset):
        """Test edge cases with empty or None outputs."""

        async def empty_function(text: str):
            """Function that returns empty strings."""
            return ""

        config = {}
        result = await evaluator.evaluate(empty_function, config, sample_dataset)

        # Empty strings should still get minimum 1 token
        for example_result in result.example_results:
            assert example_result.metrics["output_tokens"] >= 1

    @pytest.mark.asyncio
    async def test_large_text_token_estimation(self, evaluator):
        """Test handling of very large strings."""

        # Create dataset with large text
        large_text = "This is a very long text. " * 100  # ~2700 characters
        large_dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data={"text": large_text}, expected_output="test"
                )
            ]
        )

        async def large_text_function(text: str) -> str:
            return "response " * 50  # ~500 characters

        config = {}
        result = await evaluator.evaluate(large_text_function, config, large_dataset)

        example_result = result.example_results[0]

        # Check reasonable token counts for large text
        assert example_result.metrics["input_tokens"] > 100
        assert example_result.metrics["output_tokens"] > 100
        assert example_result.metrics["total_tokens"] > 200

    @pytest.mark.asyncio
    async def test_token_metrics_in_custom_metrics(self, evaluator, sample_dataset):
        """Test that token metrics flow to custom_metrics for measures array."""

        async def simple_function(text: str) -> str:
            return "output"

        config = {}
        result = await evaluator.evaluate(simple_function, config, sample_dataset)

        # Check that example_results have token metrics in both places
        for example_result in result.example_results:
            # In metrics dict (for measures array)
            assert "input_tokens" in example_result.metrics
            assert "output_tokens" in example_result.metrics
            assert "total_tokens" in example_result.metrics

            # Values should be consistent
            assert example_result.metrics["input_tokens"] > 0
            assert example_result.metrics["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_token_metrics_aggregation(self, evaluator, sample_dataset):
        """Test that token metrics are properly aggregated."""

        async def consistent_function(text: str) -> str:
            return "test"  # Same output for all = 1 token each

        config = {}
        result = await evaluator.evaluate(consistent_function, config, sample_dataset)

        # Check aggregated metrics
        assert "input_tokens" in result.aggregated_metrics
        assert "output_tokens" in result.aggregated_metrics
        assert "total_tokens" in result.aggregated_metrics

        # Should be averages across all examples
        assert result.aggregated_metrics["output_tokens"] > 0
        assert result.aggregated_metrics["input_tokens"] > 0
        assert result.aggregated_metrics["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_cost_metrics_initialization(self, evaluator, sample_dataset):
        """Test that cost metrics are initialized to 0 for string outputs."""

        async def string_function(text: str) -> str:
            return "output"

        config = {}
        result = await evaluator.evaluate(string_function, config, sample_dataset)

        # Cost metrics should exist and be 0
        for example_result in result.example_results:
            assert "input_cost" in example_result.metrics
            assert "output_cost" in example_result.metrics
            assert "total_cost" in example_result.metrics

            assert example_result.metrics["input_cost"] == 0.0
            assert example_result.metrics["output_cost"] == 0.0
            assert example_result.metrics["total_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_execution_modes_token_handling(self, sample_dataset):
        """Test token metrics handling with edge_analytics execution mode.

        Note: Only edge_analytics is currently supported. privacy/standard
        were removed, cloud/hybrid are not yet supported.
        """

        async def test_function(text: str) -> str:
            return "output"

        config = {}

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        result = await evaluator.evaluate(test_function, config, sample_dataset)

        # Should have token metrics
        assert "input_tokens" in result.aggregated_metrics
        assert result.aggregated_metrics["input_tokens"] > 0

        # Check example results have token data
        for example_result in result.example_results:
            assert "input_tokens" in example_result.metrics
            assert example_result.metrics["input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_metrics_with_errors(self, evaluator, sample_dataset):
        """Test token metrics when some function calls fail."""

        call_count = 0

        async def error_function(text: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Simulated error")
            return "success"

        config = {}
        result = await evaluator.evaluate(error_function, config, sample_dataset)

        # Should have 2 successful and 1 failed
        assert result.successful_examples == 2
        assert result.total_examples == 3

        # Successful examples should have token metrics
        successful_results = [r for r in result.example_results if r.success]
        assert len(successful_results) == 2

        for success_result in successful_results:
            assert success_result.metrics["input_tokens"] > 0
            assert success_result.metrics["output_tokens"] > 0


class TestLocalEvaluatorIntegration:
    """Integration tests for LocalEvaluator with real-world scenarios."""

    @pytest.mark.asyncio
    async def test_sentiment_analysis_simulation(self, monkeypatch):
        """Test with realistic sentiment analysis function."""
        # Ensure we're not in mock mode for exact accuracy testing
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")

        # Create realistic dataset
        examples = [
            EvaluationExample(
                input_data={"text": "This product is excellent and works great!"},
                expected_output="positive",
            ),
            EvaluationExample(
                input_data={"text": "Terrible quality, completely broken on arrival."},
                expected_output="negative",
            ),
            EvaluationExample(
                input_data={"text": "It's okay, nothing special."},
                expected_output="neutral",
            ),
        ]
        dataset = Dataset(examples=examples, name="sentiment_test")

        async def mock_sentiment_analysis(text: str, temperature: float = 0.5) -> str:
            """Mock sentiment analysis function."""
            text_lower = text.lower()
            if "excellent" in text_lower or "great" in text_lower:
                return "positive"
            elif "terrible" in text_lower or "broken" in text_lower:
                return "negative"
            else:
                return "neutral"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {"temperature": 0.3}
        result = await evaluator.evaluate(mock_sentiment_analysis, config, dataset)

        # Should achieve perfect accuracy
        assert result.aggregated_metrics["accuracy"] == 1.0

        # Should have token estimates for all examples
        assert result.aggregated_metrics["total_tokens"] > 0

        # Each example should have realistic token counts
        for example_result in result.example_results:
            assert example_result.metrics["input_tokens"] > 5  # Longer inputs
            assert (
                example_result.metrics["output_tokens"] >= 1
            )  # At least 1 token output
            assert example_result.success


class TestPromptTemplateFallbackLength:
    """Test prompt template-aware fallback token estimation."""

    @staticmethod
    def _single_example_dataset(input_data: dict[str, str]) -> Dataset:
        return Dataset(
            examples=[EvaluationExample(input_data=input_data, expected_output="4")],
            name="prompt_template_test_dataset",
        )

    @pytest.mark.asyncio
    async def test_non_privacy_uses_rendered_prompt_length_for_input_tokens(
        self, monkeypatch
    ):
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        input_data = {"question": "2+2?"}
        prompt = "You are a math tutor. Think step by step. Question: {question}"
        rendered_prompt = prompt.format(**input_data)
        expected_input_tokens = max(1, len(rendered_prompt) // 4)

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            detailed=True,
            privacy_enabled=False,
            execution_mode="edge_analytics",
        )
        dataset = self._single_example_dataset(input_data)

        async def returns_plain_string(question: str) -> str:
            del question
            return "4"

        result = await evaluator.evaluate(
            returns_plain_string, {"model": "gpt-4o-mini", "prompt": prompt}, dataset
        )

        assert result.example_results[0].metrics["input_tokens"] == expected_input_tokens

    @pytest.mark.asyncio
    async def test_privacy_uses_rendered_prompt_length_for_prompt_length_based_tokens(
        self, monkeypatch
    ):
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        input_data = {"question": "2+2?"}
        prompt = "You are a math tutor. Think step by step. Question: {question}"
        rendered_prompt = prompt.format(**input_data)
        expected_input_tokens = max(1, len(rendered_prompt) // 4)

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            detailed=True,
            privacy_enabled=True,
            execution_mode="edge_analytics",
        )
        dataset = self._single_example_dataset(input_data)

        async def returns_plain_string(question: str) -> str:
            del question
            return "4"

        result = await evaluator.evaluate(
            returns_plain_string, {"model": "gpt-4o-mini", "prompt": prompt}, dataset
        )

        assert result.example_results[0].metrics["input_tokens"] == expected_input_tokens

    @pytest.mark.asyncio
    async def test_non_privacy_format_failure_falls_back_to_additive_length(
        self, monkeypatch
    ):
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        input_data = {"question": "2+2?"}
        prompt = "You are a math tutor. Missing key: {missing_key}"
        expected_input_tokens = max(1, (len(prompt) + len(str(input_data))) // 4)

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            detailed=True,
            privacy_enabled=False,
            execution_mode="edge_analytics",
        )
        dataset = self._single_example_dataset(input_data)

        async def returns_plain_string(question: str) -> str:
            del question
            return "4"

        result = await evaluator.evaluate(
            returns_plain_string, {"model": "gpt-4o-mini", "prompt": prompt}, dataset
        )

        assert result.example_results[0].metrics["input_tokens"] == expected_input_tokens

    @pytest.mark.asyncio
    async def test_privacy_format_failure_falls_back_to_additive_length(
        self, monkeypatch
    ):
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        input_data = {"question": "2+2?"}
        prompt = "You are a math tutor. Missing key: {missing_key}"
        expected_input_tokens = max(1, (len(prompt) + len(str(input_data))) // 4)

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            detailed=True,
            privacy_enabled=True,
            execution_mode="edge_analytics",
        )
        dataset = self._single_example_dataset(input_data)

        async def returns_plain_string(question: str) -> str:
            del question
            return "4"

        result = await evaluator.evaluate(
            returns_plain_string, {"model": "gpt-4o-mini", "prompt": prompt}, dataset
        )

        assert result.example_results[0].metrics["input_tokens"] == expected_input_tokens

    @pytest.mark.asyncio
    async def test_non_privacy_without_prompt_preserves_legacy_input_estimation(
        self, monkeypatch
    ):
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        input_data = {"question": "2+2?"}
        expected_input_tokens = max(1, len(str(input_data)) // 4)

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            detailed=True,
            privacy_enabled=False,
            execution_mode="edge_analytics",
        )
        dataset = self._single_example_dataset(input_data)

        async def returns_plain_string(question: str) -> str:
            del question
            return "4"

        result = await evaluator.evaluate(
            returns_plain_string, {"model": "gpt-4o-mini"}, dataset
        )

        assert result.example_results[0].metrics["input_tokens"] == expected_input_tokens

    @pytest.mark.asyncio
    async def test_privacy_without_prompt_preserves_legacy_input_estimation(
        self, monkeypatch
    ):
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        input_data = {"question": "2+2?"}
        expected_input_tokens = max(1, len(str(input_data)) // 4)

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            detailed=True,
            privacy_enabled=True,
            execution_mode="edge_analytics",
        )
        dataset = self._single_example_dataset(input_data)

        async def returns_plain_string(question: str) -> str:
            del question
            return "4"

        result = await evaluator.evaluate(
            returns_plain_string, {"model": "gpt-4o-mini"}, dataset
        )

        assert result.example_results[0].metrics["input_tokens"] == expected_input_tokens
