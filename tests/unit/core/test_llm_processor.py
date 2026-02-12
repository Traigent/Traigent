"""Unit tests for traigent.core.llm_processor.

Tests for LLM response processing utilities, metrics extraction,
and response parsing logic.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance
# Traceability: FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from traigent.core.llm_processor import (
    LLMResponseProcessor,
    extract_model_name,
    extract_response_text,
    get_default_processor,
)
from traigent.core.types_ext import LLMMetrics


class TestLLMResponseProcessorInit:
    """Tests for LLMResponseProcessor initialization."""

    def test_init_creates_empty_lists(self) -> None:
        """Test processor initializes with empty extractor and parser lists."""
        processor = LLMResponseProcessor()
        assert processor._metrics_extractors == []
        assert processor._response_parsers == []


class TestExtractModelName:
    """Tests for extract_model_name method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_extract_model_name_from_config(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test model name extracted from config parameter."""
        response = Mock()
        result = processor.extract_model_name(response, config_model="gpt-4")
        assert result == "gpt-4"

    def test_extract_model_name_from_response_metadata(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test model name extracted from response_metadata.model."""
        response = Mock()
        response.response_metadata = {"model": "claude-3-sonnet"}
        result = processor.extract_model_name(response, config_model=None)
        assert result == "claude-3-sonnet"

    def test_extract_model_name_from_response_metadata_model_name(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test model name extracted from response_metadata.model_name."""
        response = Mock()
        response.response_metadata = {"model_name": "gpt-3.5-turbo"}
        result = processor.extract_model_name(response, config_model=None)
        assert result == "gpt-3.5-turbo"

    def test_extract_model_name_from_model_attribute(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test model name extracted from response.model attribute."""
        response = Mock()
        response.response_metadata = {}
        response.model = "gemini-pro"
        result = processor.extract_model_name(response, config_model=None)
        assert result == "gemini-pro"

    def test_extract_model_name_from_model_name_attribute(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test model name extracted from response.model_name attribute."""
        response = Mock(spec=["model_name"])
        response.model_name = "llama-2-70b"
        result = processor.extract_model_name(response, config_model=None)
        assert result == "llama-2-70b"

    def test_extract_model_name_config_takes_precedence(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test config_model parameter takes precedence over response."""
        response = Mock()
        response.response_metadata = {"model": "claude-3-sonnet"}
        result = processor.extract_model_name(response, config_model="gpt-4")
        assert result == "gpt-4"

    def test_extract_model_name_none_response(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test model name extraction with None response."""
        result = processor.extract_model_name(None, config_model=None)
        assert result is None

    def test_extract_model_name_no_metadata_attribute(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test model name extraction when response has no metadata."""
        response = Mock(spec=[])
        result = processor.extract_model_name(response, config_model=None)
        assert result is None


class TestReconstructPrompt:
    """Tests for reconstruct_prompt method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_reconstruct_prompt_from_messages(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test prompt reconstruction from messages key."""
        input_data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = processor.reconstruct_prompt(input_data)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_reconstruct_prompt_from_text(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test prompt reconstruction from text key."""
        input_data = {"text": "What is the capital of France?"}
        result = processor.reconstruct_prompt(input_data)
        assert result == [{"role": "user", "content": "What is the capital of France?"}]

    def test_reconstruct_prompt_from_dict_fallback(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test prompt reconstruction from arbitrary dict."""
        input_data = {"question": "What is AI?"}
        result = processor.reconstruct_prompt(input_data)
        assert result == [{"role": "user", "content": str(input_data)}]

    def test_reconstruct_prompt_from_string(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test prompt reconstruction from plain string."""
        input_data = "Simple prompt string"
        result = processor.reconstruct_prompt(input_data)
        assert result == [{"role": "user", "content": "Simple prompt string"}]

    def test_reconstruct_prompt_from_none(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test prompt reconstruction from None."""
        result = processor.reconstruct_prompt(None)
        assert result == [{"role": "user", "content": "None"}]


class TestExtractResponseText:
    """Tests for extract_response_text method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_extract_response_text_from_string(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test text extraction from string output."""
        output = "This is a response"
        result = processor.extract_response_text(output)
        assert result == "This is a response"

    def test_extract_response_text_from_none(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test text extraction from None output."""
        result = processor.extract_response_text(None)
        assert result is None

    def test_extract_response_text_from_text_attribute(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test text extraction from object with text attribute."""
        output = Mock()
        output.text = "Response text"
        result = processor.extract_response_text(output)
        assert result == "Response text"

    def test_extract_response_text_from_content_string(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test text extraction from object with content string."""
        output = Mock(spec=["content"])
        output.content = "Content text"
        result = processor.extract_response_text(output)
        assert result == "Content text"

    def test_extract_response_text_from_content_list(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test text extraction from object with content list."""
        content_item = Mock(spec=["text"])
        content_item.text = "List content text"
        output = Mock(spec=["content"])
        output.content = [content_item]
        result = processor.extract_response_text(output)
        assert result == "List content text"

    def test_extract_response_text_from_empty_content_list(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test text extraction from object with empty content list."""
        output = Mock(spec=["content"])
        output.content = []
        result = processor.extract_response_text(output)
        assert result is None

    def test_extract_response_text_no_attributes(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test text extraction from object with no text/content attributes."""
        output = Mock(spec=[])
        result = processor.extract_response_text(output)
        assert result is None


class TestExtractLLMMetrics:
    """Tests for extract_llm_metrics method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_extract_llm_metrics_with_custom_extractor(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test metrics extraction using custom extractor function."""
        mock_metrics: LLMMetrics = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_cost": 0.001,
            "output_cost": 0.002,
            "total_cost": 0.003,
            "response_time_ms": 150.0,
            "model_name": "gpt-4",
        }

        def custom_extractor(
            response=None, model_name=None, original_prompt=None, response_text=None
        ) -> LLMMetrics:
            return mock_metrics

        processor.set_metrics_extractor(custom_extractor)
        response = Mock()
        result = processor.extract_llm_metrics(response, model_name="gpt-4")
        assert result == mock_metrics

    def test_extract_llm_metrics_fallback_to_response(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test metrics extraction fallback to response parsing."""
        response = Mock()
        response.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        response.cost = Mock(input_cost=0.01, output_cost=0.02, total_cost=0.03)
        response.response_time_ms = 200.0

        result = processor.extract_llm_metrics(response, model_name="claude-3")
        assert result is not None
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["model_name"] == "claude-3"

    def test_extract_llm_metrics_handles_exception(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test metrics extraction handles exceptions gracefully."""

        def failing_extractor(**kwargs) -> LLMMetrics:
            raise ValueError("Extractor failed")

        processor.set_metrics_extractor(failing_extractor)
        response = Mock()
        result = processor.extract_llm_metrics(response, model_name="gpt-4")
        assert result is None

    def test_extract_llm_metrics_with_unknown_model(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test metrics extraction defaults to 'unknown' model name."""
        response = Mock()
        response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        response.cost = Mock(input_cost=0.0, output_cost=0.0, total_cost=0.0)
        response.response_time_ms = 100.0

        result = processor.extract_llm_metrics(response, model_name=None)
        assert result is not None
        assert result["model_name"] == "unknown"


class TestExtractMetricsFromResponse:
    """Tests for _extract_metrics_from_response method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_extract_metrics_from_complete_response(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test metrics extraction from complete response object."""
        response = Mock()
        response.usage = Mock(
            prompt_tokens=200, completion_tokens=100, total_tokens=300
        )
        response.cost = Mock(input_cost=0.02, output_cost=0.04, total_cost=0.06)
        response.response_time_ms = 300.0

        result = processor._extract_metrics_from_response(response, "gpt-4-turbo")
        assert result is not None
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 100
        assert result["total_tokens"] == 300
        assert result["input_cost"] == 0.02
        assert result["output_cost"] == 0.04
        assert result["total_cost"] == 0.06
        assert result["response_time_ms"] == 300.0
        assert result["model_name"] == "gpt-4-turbo"

    def test_extract_metrics_calculates_total_tokens(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test total_tokens calculation when not provided."""
        response = Mock()
        response.usage = Mock(prompt_tokens=50, completion_tokens=25)
        # No total_tokens attribute
        delattr(response.usage, "total_tokens")
        response.cost = Mock(input_cost=0.0, output_cost=0.0, total_cost=0.0)

        result = processor._extract_metrics_from_response(response, "test-model")
        assert result is not None
        assert result["total_tokens"] == 75  # 50 + 25

    def test_extract_metrics_calculates_total_cost(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test total_cost calculation when not provided."""
        response = Mock()
        response.usage = Mock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        response.cost = Mock(input_cost=0.005, output_cost=0.010)
        # No total_cost attribute
        delattr(response.cost, "total_cost")

        result = processor._extract_metrics_from_response(response, "test-model")
        assert result is not None
        assert result["total_cost"] == 0.015  # 0.005 + 0.010

    def test_extract_metrics_with_nested_response_time(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test response time extraction from nested metrics attribute."""
        response = Mock()
        response.usage = Mock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        response.cost = Mock(input_cost=0.0, output_cost=0.0, total_cost=0.0)
        response.metrics = Mock(response_time_ms=500.0)
        # No response_time_ms at top level
        delattr(response, "response_time_ms")

        result = processor._extract_metrics_from_response(response, "test-model")
        assert result is not None
        assert result["response_time_ms"] == 500.0

    def test_extract_metrics_handles_missing_attributes(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test metrics extraction with missing attributes defaults to zero."""
        response = Mock(spec=[])  # Empty spec, no attributes
        result = processor._extract_metrics_from_response(response, "test-model")
        assert result is not None
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["input_cost"] == 0.0
        assert result["output_cost"] == 0.0
        assert result["total_cost"] == 0.0
        assert result["response_time_ms"] == 0

    def test_extract_metrics_handles_exception(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test metrics extraction returns None on exception."""
        response = Mock()
        # Simulate exception when accessing usage
        type(response).usage = property(
            lambda self: (_ for _ in ()).throw(Exception("Error"))
        )

        result = processor._extract_metrics_from_response(response, "test-model")
        assert result is None


class TestEnhanceExampleResult:
    """Tests for enhance_example_result method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    @pytest.fixture
    def example_result(self) -> Mock:
        """Create mock example result object."""
        result = Mock()
        result.metrics = {}
        result.execution_time = 0.0
        return result

    @pytest.fixture
    def llm_metrics(self) -> LLMMetrics:
        """Create test LLM metrics."""
        return {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_cost": 0.01,
            "output_cost": 0.02,
            "total_cost": 0.03,
            "response_time_ms": 500.0,
            "model_name": "gpt-4",
        }

    def test_enhance_example_result_adds_metrics(
        self,
        processor: LLMResponseProcessor,
        example_result: Mock,
        llm_metrics: LLMMetrics,
    ) -> None:
        """Test enhancing result adds all LLM metrics."""
        processor.enhance_example_result(
            example_result, llm_metrics, execution_time=1.0
        )

        assert example_result.metrics["input_tokens"] == 100
        assert example_result.metrics["output_tokens"] == 50
        assert example_result.metrics["total_tokens"] == 150
        assert example_result.metrics["input_cost"] == 0.01
        assert example_result.metrics["output_cost"] == 0.02
        assert example_result.metrics["total_cost"] == 0.03

    def test_enhance_example_result_updates_execution_time(
        self,
        processor: LLMResponseProcessor,
        example_result: Mock,
        llm_metrics: LLMMetrics,
    ) -> None:
        """Test enhancing result updates execution time from metrics."""
        processor.enhance_example_result(
            example_result, llm_metrics, execution_time=1.0
        )
        assert example_result.execution_time == 0.5  # 500ms / 1000

    def test_enhance_example_result_creates_metrics_dict(
        self, processor: LLMResponseProcessor, llm_metrics: LLMMetrics
    ) -> None:
        """Test enhancing result creates metrics dict if None."""
        example_result = Mock()
        example_result.metrics = None
        example_result.execution_time = 0.0

        processor.enhance_example_result(
            example_result, llm_metrics, execution_time=1.0
        )
        assert example_result.metrics is not None
        assert isinstance(example_result.metrics, dict)
        assert "input_tokens" in example_result.metrics

    def test_enhance_example_result_uses_measured_time_fallback(
        self, processor: LLMResponseProcessor, example_result: Mock
    ) -> None:
        """Test using measured execution time when metrics have no response_time_ms."""
        llm_metrics: LLMMetrics = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "response_time_ms": 0,  # Zero response time
            "model_name": "test",
        }

        processor.enhance_example_result(
            example_result, llm_metrics, execution_time=2.5
        )
        assert example_result.execution_time == 2.5

    def test_enhance_example_result_with_none_metrics(
        self, processor: LLMResponseProcessor, example_result: Mock
    ) -> None:
        """Test enhancing result with None metrics."""
        processor.enhance_example_result(example_result, None, execution_time=1.5)
        assert example_result.execution_time == 1.5

    def test_enhance_example_result_handles_missing_execution_time_attr(
        self, processor: LLMResponseProcessor, llm_metrics: LLMMetrics
    ) -> None:
        """Test enhancing result when execution_time attribute doesn't exist."""
        example_result = Mock(spec=["metrics"])
        example_result.metrics = {}

        processor.enhance_example_result(
            example_result, llm_metrics, execution_time=3.0
        )
        assert example_result.execution_time == 0.5  # From metrics response_time_ms


class TestValidateResponseFormat:
    """Tests for validate_response_format method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_validate_response_format_valid(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test validation passes for valid response format."""
        response = Mock()
        response.content = "Response content"
        response.usage = Mock(prompt_tokens=10)

        result = processor.validate_response_format(response)
        assert result is True

    def test_validate_response_format_none(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test validation fails for None response."""
        result = processor.validate_response_format(None)
        assert result is False

    def test_validate_response_format_missing_content(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test validation fails when content attribute missing."""
        response = Mock(spec=["usage"])
        response.usage = Mock(prompt_tokens=10)

        result = processor.validate_response_format(response)
        assert result is False

    def test_validate_response_format_missing_usage(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test validation fails when usage attribute missing."""
        response = Mock(spec=["content"])
        response.content = "Content"

        result = processor.validate_response_format(response)
        assert result is False


class TestSetMetricsExtractor:
    """Tests for set_metrics_extractor method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_set_metrics_extractor(self, processor: LLMResponseProcessor) -> None:
        """Test setting custom metrics extractor function."""

        def custom_extractor(**kwargs) -> LLMMetrics:
            return {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "response_time_ms": 1.0,
                "model_name": "custom",
            }

        processor.set_metrics_extractor(custom_extractor)
        assert hasattr(processor, "_extract_llm_metrics_func")
        assert processor._extract_llm_metrics_func == custom_extractor


class TestAddResponseParser:
    """Tests for add_response_parser method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_add_response_parser(self, processor: LLMResponseProcessor) -> None:
        """Test adding response parser function."""

        def parser1(response):
            return {"key1": "value1"}

        def parser2(response):
            return {"key2": "value2"}

        processor.add_response_parser(parser1)
        processor.add_response_parser(parser2)

        assert len(processor._response_parsers) == 2
        assert parser1 in processor._response_parsers
        assert parser2 in processor._response_parsers


class TestParseResponseContent:
    """Tests for parse_response_content method."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_parse_response_content_with_parsers(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test parsing response with multiple parsers."""

        def parser1(response):
            return {"field1": "value1", "field2": "value2"}

        def parser2(response):
            return {"field3": "value3"}

        processor.add_response_parser(parser1)
        processor.add_response_parser(parser2)

        response = Mock()
        result = processor.parse_response_content(response)

        assert result == {"field1": "value1", "field2": "value2", "field3": "value3"}

    def test_parse_response_content_no_parsers(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test parsing response with no parsers returns empty dict."""
        response = Mock()
        result = processor.parse_response_content(response)
        assert result == {}

    def test_parse_response_content_handles_parser_exception(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test parsing handles parser exceptions gracefully."""

        def failing_parser(response):
            raise ValueError("Parser failed")

        def working_parser(response):
            return {"key": "value"}

        processor.add_response_parser(failing_parser)
        processor.add_response_parser(working_parser)

        response = Mock()
        result = processor.parse_response_content(response)

        # Should still get result from working parser
        assert result == {"key": "value"}

    def test_parse_response_content_parser_returns_none(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test parsing when parser returns None."""

        def none_parser(response):
            return None

        def valid_parser(response):
            return {"key": "value"}

        processor.add_response_parser(none_parser)
        processor.add_response_parser(valid_parser)

        response = Mock()
        result = processor.parse_response_content(response)
        assert result == {"key": "value"}


class TestGetDefaultProcessor:
    """Tests for get_default_processor function."""

    def test_get_default_processor_returns_instance(self) -> None:
        """Test get_default_processor returns LLMResponseProcessor instance."""
        processor = get_default_processor()
        assert isinstance(processor, LLMResponseProcessor)

    def test_get_default_processor_returns_same_instance(self) -> None:
        """Test get_default_processor returns same instance on multiple calls."""
        processor1 = get_default_processor()
        processor2 = get_default_processor()
        assert processor1 is processor2


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_extract_model_name_convenience_function(self) -> None:
        """Test extract_model_name convenience function."""
        response = Mock()
        response.response_metadata = {"model": "gpt-4"}

        result = extract_model_name(response)
        assert result == "gpt-4"

    def test_extract_response_text_convenience_function(self) -> None:
        """Test extract_response_text convenience function."""
        output = "Test response text"
        result = extract_response_text(output)
        assert result == "Test response text"

    def test_convenience_functions_use_default_processor(self) -> None:
        """Test convenience functions use the default processor."""
        default_processor = get_default_processor()

        # Patch the default processor's methods
        with patch.object(
            default_processor, "extract_model_name", return_value="mocked-model"
        ) as mock_extract_model:
            extract_model_name(Mock(), "test-model")
            mock_extract_model.assert_called_once()

        with patch.object(
            default_processor, "extract_response_text", return_value="mocked-text"
        ) as mock_extract_text:
            extract_response_text("test")
            mock_extract_text.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def processor(self) -> LLMResponseProcessor:
        """Create test processor instance."""
        return LLMResponseProcessor()

    def test_extract_model_name_with_non_dict_metadata(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test extract_model_name when response_metadata is not a dict."""
        response = Mock(spec=["response_metadata"])
        response.response_metadata = "not a dict"
        result = processor.extract_model_name(response, config_model=None)
        assert result is None

    def test_extract_response_text_with_non_list_content(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test extract_response_text when content is not string or list."""
        output = Mock(spec=["content"])
        output.content = 12345  # Not string or list
        result = processor.extract_response_text(output)
        assert result is None

    def test_enhance_example_result_with_missing_metric_keys(
        self, processor: LLMResponseProcessor
    ) -> None:
        """Test enhance_example_result with incomplete metrics dict."""
        example_result = Mock()
        example_result.metrics = {}
        example_result.execution_time = 0.0

        incomplete_metrics: dict = {
            "input_tokens": 10,
            "output_tokens": 5,
            # Missing other required keys
        }

        # incomplete_metrics is intentionally partial for testing
        processor.enhance_example_result(
            example_result, incomplete_metrics, execution_time=1.0  # type: ignore[arg-type]
        )
        # Should handle gracefully with .get() default to 0
        assert example_result.metrics["input_tokens"] == 10
        assert example_result.metrics["output_tokens"] == 5
        assert example_result.metrics["total_tokens"] == 0  # Default

    @pytest.mark.parametrize(
        "input_data,expected_length",
        [
            ({"messages": []}, 0),
            ({"text": ""}, 1),
            ("", 1),
            ([], 1),
            ({}, 1),
        ],
    )
    def test_reconstruct_prompt_with_empty_inputs(
        self, processor: LLMResponseProcessor, input_data, expected_length
    ) -> None:
        """Test reconstruct_prompt handles empty/minimal inputs."""
        result = processor.reconstruct_prompt(input_data)
        assert result is not None
        if input_data == {"messages": []}:
            assert len(result) == expected_length
        else:
            assert len(result) >= expected_length
