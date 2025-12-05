"""Unit tests for agent specification generator."""

from unittest.mock import Mock, patch

import pytest

from traigent.agents.specification_generator import (
    FunctionAnalysis,
    SpecificationGenerator,
    generator,
)
from traigent.cloud.models import AgentSpecification
from traigent.config.types import TraigentConfig


# Sample functions for testing
def sample_customer_support_function(query: str, context: str = "") -> str:
    """Handle customer support queries with appropriate responses.

    This function processes customer queries and provides helpful responses
    based on the context and query type.
    """
    return f"Response to: {query}"


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of given text and return classification results."""
    return {"sentiment": "positive", "confidence": 0.85}


def generate_blog_post(topic: str, length: int = 500) -> str:
    """Generate engaging blog post content about the specified topic."""
    return f"Blog post about {topic}"


def process_data_extraction(document: str, fields: list) -> dict:
    """Extract structured information from unstructured documents."""
    return {"extracted_fields": fields}


def complex_reasoning_task(problem: str, steps: bool = True) -> dict:
    """Solve complex problems using step-by-step reasoning approach.

    This function breaks down complex problems into manageable steps,
    applies logical reasoning, and provides detailed solutions.
    """
    if steps:
        return {"solution": "answer", "steps": ["step1", "step2"]}
    return {"solution": "answer"}


@pytest.fixture
def spec_generator():
    """Create a specification generator instance."""
    return SpecificationGenerator()


@pytest.fixture
def sample_traigent_config():
    """Create a sample TraiGent configuration."""
    return TraigentConfig(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500,
        top_p=0.9,
        custom_params={
            "configuration_space": {
                "model": ["gpt-4o-mini", "gpt-4o"],
                "temperature": (0.0, 1.0),
                "max_tokens": [100, 500, 1000],
            },
            "objectives": ["accuracy", "cost"],
            "algorithm": "bayesian",
            "max_iterations": 50,
        },
    )


@pytest.fixture
def mock_optimized_function():
    """Create a mock optimized function with TraiGent attributes."""
    func = Mock()
    func.__name__ = "customer_support_agent"
    func.__doc__ = "Handle customer support queries efficiently"

    # Mock signature
    param1 = Mock()
    param1.name = "query"
    param2 = Mock()
    param2.name = "context"

    sig = Mock()
    sig.parameters = {"query": param1, "context": param2}
    sig.return_annotation = str

    with patch("inspect.signature", return_value=sig):
        with patch("inspect.getdoc", return_value=func.__doc__):
            yield func


class TestFunctionAnalysis:
    """Test FunctionAnalysis dataclass."""

    def test_function_analysis_creation(self):
        """Test creating FunctionAnalysis instance."""
        analysis = FunctionAnalysis(
            name="test_function",
            signature="test_function(x: str) -> str",
            docstring="Test function",
            parameters=["x"],
            return_annotation="str",
            imports=["os", "sys"],
            dependencies=["os", "sys"],
            complexity_score=0.5,
            inferred_domain="conversational",
            suggested_agent_type="conversational",
        )

        assert analysis.name == "test_function"
        assert analysis.signature == "test_function(x: str) -> str"
        assert analysis.docstring == "Test function"
        assert analysis.parameters == ["x"]
        assert analysis.return_annotation == "str"
        assert analysis.imports == ["os", "sys"]
        assert analysis.dependencies == ["os", "sys"]
        assert analysis.complexity_score == 0.5
        assert analysis.inferred_domain == "conversational"
        assert analysis.suggested_agent_type == "conversational"


class TestSpecificationGenerator:
    """Test SpecificationGenerator class."""

    def test_initialization(self, spec_generator):
        """Test generator initialization."""
        assert isinstance(spec_generator._domain_patterns, dict)
        assert "conversational" in spec_generator._domain_patterns
        assert "analytical" in spec_generator._domain_patterns
        assert "content_generation" in spec_generator._domain_patterns
        assert "task_automation" in spec_generator._domain_patterns

        assert isinstance(spec_generator._platform_mappings, dict)
        assert "openai" in spec_generator._platform_mappings
        assert "langchain" in spec_generator._platform_mappings

    def test_analyze_function_customer_support(self, spec_generator):
        """Test analyzing customer support function."""
        analysis = spec_generator._analyze_function(sample_customer_support_function)

        assert analysis.name == "sample_customer_support_function"
        assert "query" in analysis.parameters
        assert "context" in analysis.parameters
        assert "customer" in analysis.docstring.lower()
        assert analysis.inferred_domain == "conversational"
        assert analysis.suggested_agent_type == "conversational"
        assert 0.0 <= analysis.complexity_score <= 1.0

    def test_analyze_function_sentiment_analysis(self, spec_generator):
        """Test analyzing sentiment analysis function."""
        analysis = spec_generator._analyze_function(analyze_sentiment)

        assert analysis.name == "analyze_sentiment"
        assert "text" in analysis.parameters
        assert analysis.inferred_domain == "analytical"
        assert analysis.suggested_agent_type == "analytical"

    def test_analyze_function_content_generation(self, spec_generator):
        """Test analyzing content generation function."""
        analysis = spec_generator._analyze_function(generate_blog_post)

        assert analysis.name == "generate_blog_post"
        assert "topic" in analysis.parameters
        assert analysis.inferred_domain == "content_generation"
        assert analysis.suggested_agent_type == "content_generation"

    def test_analyze_function_data_processing(self, spec_generator):
        """Test analyzing data processing function."""
        analysis = spec_generator._analyze_function(process_data_extraction)

        assert analysis.name == "process_data_extraction"
        assert analysis.inferred_domain == "task_automation"
        assert analysis.suggested_agent_type == "task_automation"

    def test_analyze_function_complex_reasoning(self, spec_generator):
        """Test analyzing complex reasoning function."""
        analysis = spec_generator._analyze_function(complex_reasoning_task)

        assert analysis.name == "complex_reasoning_task"
        assert analysis.complexity_score > 0.1  # Should be reasonably complex
        # Could be analytical or reasoning domain
        assert analysis.inferred_domain in ["analytical", "conversational"]

    def test_infer_domain(self, spec_generator):
        """Test domain inference from name and docstring."""
        # Conversational domain
        domain = spec_generator._infer_domain(
            "customer_chat", "handle customer conversations"
        )
        assert domain == "conversational"

        # Analytical domain
        domain = spec_generator._infer_domain(
            "analyze_data", "classify and analyze input data"
        )
        assert domain == "analytical"

        # Content generation domain
        domain = spec_generator._infer_domain(
            "generate_content", "create new written content"
        )
        assert domain == "content_generation"

        # Task automation domain
        domain = spec_generator._infer_domain(
            "process_documents", "extract and transform data"
        )
        assert domain == "task_automation"

        # Default to conversational
        domain = spec_generator._infer_domain("unknown_function", "does something")
        assert domain == "conversational"

    def test_suggest_agent_type_from_domain(self, spec_generator):
        """Test agent type suggestion from domain."""
        assert (
            spec_generator._suggest_agent_type_from_domain("conversational")
            == "conversational"
        )
        assert (
            spec_generator._suggest_agent_type_from_domain("analytical") == "analytical"
        )
        assert (
            spec_generator._suggest_agent_type_from_domain("content_generation")
            == "content_generation"
        )
        assert (
            spec_generator._suggest_agent_type_from_domain("task_automation")
            == "task_automation"
        )
        assert (
            spec_generator._suggest_agent_type_from_domain("unknown")
            == "conversational"
        )

    def test_generate_agent_name(self, spec_generator):
        """Test agent name generation."""
        # Snake case function
        analysis = FunctionAnalysis(
            name="customer_support_agent",
            signature="",
            docstring=None,
            parameters=[],
            return_annotation=None,
            imports=[],
            dependencies=[],
            complexity_score=0.5,
            inferred_domain="conversational",
            suggested_agent_type="conversational",
        )

        name = spec_generator._generate_agent_name(analysis)
        assert name == "Customer Support Agent Assistant"

        # CamelCase function
        analysis.name = "dataAnalyzer"
        analysis.inferred_domain = "analytical"
        name = spec_generator._generate_agent_name(analysis)
        assert name == "Data Analyzer"

        # Content generation
        analysis.name = "blog_writer"
        analysis.inferred_domain = "content_generation"
        name = spec_generator._generate_agent_name(analysis)
        assert name == "Blog Writer Generator"

    def test_infer_platform(self, spec_generator):
        """Test platform inference from configuration."""
        # OpenAI models
        config = {"configuration_space": {"model": ["gpt-4o-mini", "gpt-4o"]}}
        platform = spec_generator._infer_platform(config)
        assert platform == "openai"

        # Claude models
        config = {"configuration_space": {"model": ["claude-3-sonnet"]}}
        platform = spec_generator._infer_platform(config)
        assert platform == "anthropic"

        # LLaMA models
        config = {"configuration_space": {"model": ["llama-2-7b"]}}
        platform = spec_generator._infer_platform(config)
        assert platform == "huggingface"

        # Framework override
        config = {"auto_override_frameworks": True}
        platform = spec_generator._infer_platform(config)
        assert platform == "langchain"

        # Default
        config = {}
        platform = spec_generator._infer_platform(config)
        assert platform == "openai"

    def test_generate_prompt_template(self, spec_generator):
        """Test prompt template generation."""
        analysis = FunctionAnalysis(
            name="customer_support",
            signature="customer_support(query: str) -> str",
            docstring="Handle customer support queries",
            parameters=["query"],
            return_annotation="str",
            imports=[],
            dependencies=[],
            complexity_score=0.5,
            inferred_domain="conversational",
            suggested_agent_type="conversational",
        )

        config = {}
        template = spec_generator._generate_prompt_template(analysis, config)

        assert "helpful AI assistant" in template
        assert "customer_support" in template
        assert "{input}" in template
        assert "Handle customer support queries" in template
        assert "str" in template

    def test_extract_model_parameters(self, spec_generator):
        """Test model parameter extraction."""
        config = {
            "configuration_space": {
                "model": ["gpt-4o-mini", "gpt-4o"],
                "temperature": (0.0, 1.0),
                "max_tokens": [100, 500],
                "top_p": 0.9,
            }
        }

        params = spec_generator._extract_model_parameters(config)

        assert params["model"] == "gpt-4o-mini"  # First in list
        assert params["temperature"] == 0.5  # Midpoint of range
        assert params["max_tokens"] == 100  # First in list
        assert params["top_p"] == 0.9  # Single value
        assert "frequency_penalty" in params  # Default value

    def test_generate_reasoning_instructions(self, spec_generator):
        """Test reasoning instruction generation."""
        # High complexity
        analysis = FunctionAnalysis(
            name="complex_task",
            signature="",
            docstring=None,
            parameters=[],
            return_annotation=None,
            imports=[],
            dependencies=[],
            complexity_score=0.8,
            inferred_domain="conversational",
            suggested_agent_type="conversational",
        )

        instructions = spec_generator._generate_reasoning_instructions(analysis)
        assert "step by step" in instructions.lower()

        # Analytical domain
        analysis.complexity_score = 0.5
        analysis.inferred_domain = "analytical"
        instructions = spec_generator._generate_reasoning_instructions(analysis)
        assert "analyze" in instructions.lower()

        # Low complexity, non-analytical
        analysis.complexity_score = 0.3
        analysis.inferred_domain = "conversational"
        instructions = spec_generator._generate_reasoning_instructions(analysis)
        assert instructions is None

    def test_infer_style_tone_format(self, spec_generator):
        """Test style, tone, and format inference."""
        analysis = FunctionAnalysis(
            name="customer_support",
            signature="() -> dict",
            docstring=None,
            parameters=[],
            return_annotation="dict",
            imports=[],
            dependencies=[],
            complexity_score=0.5,
            inferred_domain="conversational",
            suggested_agent_type="conversational",
        )

        style = spec_generator._infer_style(analysis)
        assert style == "conversational and helpful"

        tone = spec_generator._infer_tone(analysis)
        assert "professional" in tone and "empathetic" in tone

        format_result = spec_generator._infer_format(analysis)
        assert "structured" in format_result and "json" in format_result.lower()

    def test_extract_custom_tools_from_source(self, spec_generator):
        """Test custom tool extraction from function source."""
        # Mock function with requests usage
        func = Mock()
        func.__name__ = "web_scraper"

        # Mock hasattr to return False for _traigent_tools
        def mock_hasattr(obj, name):
            if name == "_traigent_tools":
                return False
            return hasattr(obj, name)

        source_code = """
def web_scraper(url):
    import requests
    import json
    response = requests.get(url)
    return json.loads(response.text)
"""

        with patch("inspect.getsource", return_value=source_code):
            with patch("builtins.hasattr", side_effect=mock_hasattr):
                tools = spec_generator._extract_custom_tools(func)
                assert "web_search" in tools
                assert "json_parser" in tools

    def test_extract_custom_tools_from_attributes(self, spec_generator):
        """Test custom tool extraction from function attributes."""
        func = Mock()
        func.__name__ = "test_function"
        func._traigent_tools = ["calculator", "database"]

        tools = spec_generator._extract_custom_tools(func)
        assert tools == ["calculator", "database"]

    def test_calculate_complexity(self, spec_generator):
        """Test complexity calculation."""
        # Simple function
        simple_source = """
def simple_func():
    return "hello"
"""
        complexity = spec_generator._calculate_complexity(simple_source)
        assert 0.0 <= complexity <= 0.3

        # Complex function
        complex_source = """
def complex_func(data):
    result = []
    for item in data:
        if item.get('valid'):
            try:
                processed = process_item(item)
                if processed:
                    for sub_item in processed:
                        if validate_sub_item(sub_item):
                            result.append(transform(sub_item))
            except Exception as e:
                handle_error(e)
                continue
        else:
            skip_invalid(item)
    return result
"""
        complexity = spec_generator._calculate_complexity(complex_source)
        assert complexity > 0.5

    def test_extract_dependencies(self, spec_generator):
        """Test dependency extraction from source code."""
        source = """
import os
import sys
from typing import Dict, List
from traigent.core import optimizer
import requests.auth
"""
        imports, dependencies = spec_generator._extract_dependencies(source)

        assert "os" in imports
        assert "sys" in imports
        assert "typing" in imports
        assert "traigent.core" in imports
        assert "requests.auth" in imports

        assert "os" in dependencies
        assert "sys" in dependencies
        assert "typing" in dependencies
        assert "traigent" in dependencies
        assert "requests" in dependencies

    def test_extract_parameters_from_signature(self, spec_generator):
        """Test parameter extraction from signature string."""
        # Simple signature
        params = spec_generator._extract_parameters_from_signature("func(a, b)")
        assert params == ["a", "b"]

        # Signature with types and defaults
        params = spec_generator._extract_parameters_from_signature(
            "func(query: str, context: str = '', max_length: int = 100)"
        )
        assert params == ["query", "context", "max_length"]

        # Signature with self (should be excluded)
        params = spec_generator._extract_parameters_from_signature("method(self, x, y)")
        assert params == ["x", "y"]

        # Empty signature
        params = spec_generator._extract_parameters_from_signature("func()")
        assert params == []

        # Signature with nested generics should not split inner commas
        params = spec_generator._extract_parameters_from_signature(
            "func(mapping: dict[str, tuple[int, int]], callback: Callable[[int, int], bool])"
        )
        assert params == ["mapping", "callback"]

        # Signature with positional-only, keyword-only, and varargs
        params = spec_generator._extract_parameters_from_signature(
            "func(pos_only: int, /, value: str, *args, flag: bool = False, **kwargs)"
        )
        assert params == ["pos_only", "value", "args", "flag", "kwargs"]

    def test_extract_return_annotation(self, spec_generator):
        """Test return annotation extraction."""
        # With return annotation
        annotation = spec_generator._extract_return_annotation("func() -> str")
        assert annotation == "str"

        # With complex return annotation
        annotation = spec_generator._extract_return_annotation(
            "func() -> Dict[str, Any]"
        )
        assert annotation == "Dict[str, Any]"

        # Without return annotation
        annotation = spec_generator._extract_return_annotation("func()")
        assert annotation is None

    def test_from_optimized_function(self, spec_generator):
        """Test generating specification from optimized function."""
        # Add TraiGent config to function
        sample_customer_support_function._traigent_config = {
            "configuration_space": {"model": ["gpt-4o-mini"], "temperature": 0.7},
            "objectives": ["accuracy"],
        }

        spec = spec_generator.from_optimized_function(
            sample_customer_support_function,
            agent_name="Custom Support Agent",
            agent_platform="openai",
        )

        assert isinstance(spec, AgentSpecification)
        assert spec.name == "Custom Support Agent"
        assert spec.agent_platform == "openai"
        assert spec.agent_type == "conversational"
        assert "customer support" in spec.prompt_template.lower()
        assert spec.model_parameters["model"] == "gpt-4o-mini"
        assert spec.model_parameters["temperature"] == 0.7
        assert spec.persona == "a helpful and knowledgeable assistant"
        assert "Be empathetic and understanding" in spec.guidelines

    def test_from_function_signature(self, spec_generator):
        """Test generating specification from function signature."""
        spec = spec_generator.from_function_signature(
            function_name="sentiment_analyzer",
            function_signature="sentiment_analyzer(text: str) -> dict",
            docstring="Analyze the sentiment of given text",
            configuration_space={"model": ["GPT-4o"], "temperature": 0.5},
            objectives=["accuracy", "speed"],
        )

        assert isinstance(spec, AgentSpecification)
        assert spec.name == "Sentiment Analyzer"
        assert spec.agent_type == "analytical"
        assert spec.model_parameters["model"] == "GPT-4o"
        assert spec.model_parameters["temperature"] == 0.5
        assert "generated_from" in spec.metadata
        assert spec.metadata["generated_from"] == "sdk_function_signature"

    def test_update_agent_specification(self, spec_generator):
        """Test updating agent specification with optimization results."""
        # Create initial spec
        spec = AgentSpecification(
            id="test-agent",
            name="Test Agent",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="Test",
            model_parameters={"model": "gpt-4o-mini", "temperature": 0.7},
            metadata={},
        )

        # Optimization results
        optimization_results = {
            "best_config": {
                "model": "GPT-4o",
                "temperature": 0.9,
                "max_tokens": 500,
                "unknown_param": "ignored",  # Should be ignored
            },
            "best_score": 0.95,
            "iterations": 25,
        }

        updated_spec = spec_generator.update_agent_specification(
            spec, optimization_results
        )

        assert updated_spec.model_parameters["model"] == "GPT-4o"
        assert updated_spec.model_parameters["temperature"] == 0.9
        assert updated_spec.model_parameters["max_tokens"] == 500
        assert "unknown_param" not in updated_spec.model_parameters

        assert updated_spec.metadata["optimization_results"] == optimization_results
        assert updated_spec.metadata["optimized"] is True
        assert "last_optimization" in updated_spec.metadata

    def test_extract_optimization_config_from_traigent_config(
        self, spec_generator, sample_traigent_config
    ):
        """Test extracting optimization config from TraiGent config object."""
        func = Mock()
        func._traigent_config = sample_traigent_config

        config = spec_generator._extract_optimization_config(func, None)

        assert "model" in config
        assert "temperature" in config
        assert "max_tokens" in config
        assert config["model"] == "gpt-4o-mini"
        assert config["temperature"] == 0.7

    def test_extract_optimization_config_from_dict(self, spec_generator):
        """Test extracting optimization config from dictionary."""

        # Create a real object instead of mock to avoid recursion
        class MockFunc:
            def __init__(self):
                self._traigent_config = {
                    "configuration_space": {"model": ["gpt-4o-mini"]},
                    "objectives": ["accuracy"],
                }

        func = MockFunc()
        config = spec_generator._extract_optimization_config(func, None)

        assert config["configuration_space"]["model"] == ["gpt-4o-mini"]
        assert config["objectives"] == ["accuracy"]

    def test_extract_optimization_config_from_attributes(self, spec_generator):
        """Test extracting optimization config from function attributes."""
        func = Mock()
        func._traigent_configuration_space = {"temperature": (0.0, 1.0)}
        func._traigent_objectives = ["cost", "latency"]
        func._traigent_algorithm = "random"

        config = spec_generator._extract_optimization_config(func, None)

        assert config["configuration_space"]["temperature"] == (0.0, 1.0)
        assert config["objectives"] == ["cost", "latency"]
        assert config["algorithm"] == "random"

    def test_extract_optimization_config_with_override(self, spec_generator):
        """Test optimization config extraction with provided override."""

        # Create a real object instead of mock to avoid recursion
        class MockFunc:
            def __init__(self):
                self._traigent_config = {"objectives": ["accuracy"]}

        func = MockFunc()
        override_config = {
            "configuration_space": {"model": ["GPT-4o"]},
            "objectives": ["speed"],  # Provided but function config takes precedence
        }

        config = spec_generator._extract_optimization_config(func, override_config)

        assert config["configuration_space"]["model"] == ["GPT-4o"]
        # Function config takes precedence over override config in this implementation
        assert config["objectives"] == ["accuracy"]

    def test_get_return_annotation(self, spec_generator):
        """Test getting return annotation from function."""

        # Function with annotation
        def annotated_func() -> str:
            return "test"

        annotation = spec_generator._get_return_annotation(annotated_func)
        assert annotation == "<class 'str'>"

        # Function without annotation
        def unannotated_func():
            return "test"

        annotation = spec_generator._get_return_annotation(unannotated_func)
        assert annotation is None


class TestModuleLevelGenerator:
    """Test module-level generator instance."""

    def test_global_generator_instance(self):
        """Test that global generator instance exists and works."""
        assert isinstance(generator, SpecificationGenerator)

        # Test it can analyze a function
        analysis = generator._analyze_function(sample_customer_support_function)
        assert analysis.name == "sample_customer_support_function"


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_analyze_function_no_source(self, spec_generator):
        """Test analyzing function when source is not available."""
        # Create a built-in function that has no source
        with patch(
            "inspect.getsource", side_effect=OSError("could not get source code")
        ):
            analysis = spec_generator._analyze_function(len)  # built-in function

            assert analysis.name == "len"
            assert analysis.imports == []
            assert analysis.dependencies == []
            assert analysis.complexity_score == 0.5  # Default value

    def test_calculate_complexity_invalid_source(self, spec_generator):
        """Test complexity calculation with invalid source."""
        invalid_source = "invalid python code {"
        complexity = spec_generator._calculate_complexity(invalid_source)
        assert complexity >= 0.0  # Should return some default value

    def test_extract_dependencies_invalid_source(self, spec_generator):
        """Test dependency extraction with invalid source."""
        invalid_source = "not valid python"
        imports, dependencies = spec_generator._extract_dependencies(invalid_source)
        assert imports == []
        assert dependencies == []

    def test_extract_custom_tools_no_source(self, spec_generator):
        """Test custom tool extraction when source is not available."""
        func = Mock()
        func.__name__ = "test_func"

        # Mock hasattr to return False for _traigent_tools
        def mock_hasattr(obj, name):
            if name == "_traigent_tools":
                return False
            return hasattr(obj, name)

        with patch("inspect.getsource", side_effect=OSError("no source")):
            with patch("builtins.hasattr", side_effect=mock_hasattr):
                tools = spec_generator._extract_custom_tools(func)
                assert tools is None

    def test_generate_prompt_template_no_docstring(self, spec_generator):
        """Test prompt template generation with no docstring."""
        analysis = FunctionAnalysis(
            name="test_function",
            signature="test_function(x: str) -> str",
            docstring=None,  # No docstring
            parameters=["x"],
            return_annotation="str",
            imports=[],
            dependencies=[],
            complexity_score=0.5,
            inferred_domain="conversational",
            suggested_agent_type="conversational",
        )

        template = spec_generator._generate_prompt_template(analysis, {})

        # Should still generate a valid template
        assert "AI assistant" in template
        assert "{input}" in template
        assert "str" in template

    def test_generate_prompt_template_no_parameters(self, spec_generator):
        """Test prompt template generation with no parameters."""
        analysis = FunctionAnalysis(
            name="simple_func",
            signature="simple_func() -> str",
            docstring="Simple function",
            parameters=[],  # No parameters
            return_annotation="str",
            imports=[],
            dependencies=[],
            complexity_score=0.5,
            inferred_domain="conversational",
            suggested_agent_type="conversational",
        )

        template = spec_generator._generate_prompt_template(analysis, {})

        # Should still work
        assert "AI assistant" in template
        assert "{input}" in template


class TestInferenceEdgeCases:
    """Test edge cases in inference methods."""

    def test_infer_domain_empty_inputs(self, spec_generator):
        """Test domain inference with empty inputs."""
        domain = spec_generator._infer_domain("", "")
        assert domain == "conversational"  # Default

        domain = spec_generator._infer_domain("", None)
        assert domain == "conversational"  # Default

    def test_infer_platform_empty_config(self, spec_generator):
        """Test platform inference with empty config."""
        platform = spec_generator._infer_platform({})
        assert platform == "openai"  # Default

        platform = spec_generator._infer_platform({"configuration_space": {}})
        assert platform == "openai"  # Default

    def test_extract_model_parameters_empty_config(self, spec_generator):
        """Test model parameter extraction with empty config."""
        params = spec_generator._extract_model_parameters({})

        # Should return defaults
        assert params["model"] == "gpt-4o-mini"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 150

    def test_generate_agent_name_edge_cases(self, spec_generator):
        """Test agent name generation edge cases."""
        # Single character name
        analysis = FunctionAnalysis(
            name="a",
            signature="",
            docstring=None,
            parameters=[],
            return_annotation=None,
            imports=[],
            dependencies=[],
            complexity_score=0.5,
            inferred_domain="conversational",
            suggested_agent_type="conversational",
        )

        name = spec_generator._generate_agent_name(analysis)
        assert name == "A Assistant"

        # Already has suffix
        analysis.name = "customer_assistant"
        name = spec_generator._generate_agent_name(analysis)
        assert name == "Customer Assistant"  # Should not double suffix


class TestIntegrationScenarios:
    """Test integration scenarios with realistic function examples."""

    def test_full_workflow_customer_support(self, spec_generator):
        """Test full workflow for customer support function."""

        # Simulate decorated function
        def customer_support_bot(query: str, urgency: str = "normal") -> dict:
            """Intelligent customer support bot that handles queries efficiently.

            Processes customer queries with context awareness and provides
            appropriate responses based on urgency level.
            """
            return {"response": "helpful answer", "urgency": urgency}

        # Add TraiGent configuration
        customer_support_bot._traigent_config = {
            "configuration_space": {
                "model": ["gpt-4o-mini", "gpt-4o"],
                "temperature": (0.3, 0.8),
                "max_tokens": [150, 300],
            },
            "objectives": ["accuracy", "response_time"],
        }

        # Generate specification
        spec = spec_generator.from_optimized_function(customer_support_bot)

        # Verify key aspects
        assert spec.agent_type == "conversational"
        assert spec.agent_platform == "openai"
        assert "customer support" in spec.name.lower()
        assert "helpful" in spec.persona
        assert "empathetic" in spec.tone
        assert "query" in spec.prompt_template
        assert "urgency" in spec.prompt_template

    def test_full_workflow_data_analysis(self, spec_generator):
        """Test full workflow for data analysis function."""

        def analyze_financial_data(data: list, metrics: list = None) -> dict:
            """Advanced financial data analysis with statistical insights.

            Performs comprehensive analysis of financial datasets,
            calculates key metrics, and identifies trends.
            """
            return {"analysis": "results", "trends": []}

        # Add configuration
        analyze_financial_data._traigent_config = {
            "configuration_space": {
                "model": ["GPT-4o"],
                "temperature": 0.1,  # Low temperature for analytical tasks
                "max_tokens": 1000,
            },
            "objectives": ["accuracy"],
        }

        spec = spec_generator.from_optimized_function(analyze_financial_data)

        assert spec.agent_type == "analytical"
        assert "analyze" in spec.reasoning.lower()
        assert "analytical" in spec.style
        assert "friendly" in spec.tone
        assert spec.model_parameters["temperature"] == 0.1

    def test_full_workflow_content_generation(self, spec_generator):
        """Test full workflow for content generation function."""

        def write_marketing_copy(
            product: str, audience: str, style: str = "professional"
        ) -> str:
            """Creative marketing copy writer for various products and audiences.

            Generates compelling marketing content tailored to specific
            products, target audiences, and brand styles.
            """
            return f"Marketing copy for {product}"

        spec = spec_generator.from_optimized_function(write_marketing_copy)

        assert spec.agent_type == "content_generation"
        assert "creative" in spec.style
        assert "Generator" in spec.name
        assert "creative" in spec.guidelines[2]  # Should have creativity guideline
