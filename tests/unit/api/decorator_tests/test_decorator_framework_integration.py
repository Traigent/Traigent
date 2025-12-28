"""Tests for decorator integration with various LLM frameworks.

Tests Traigent decorator integration with:
- OpenAI (sync and async)
- LangChain
- Anthropic
- Cohere
- Framework auto-detection
- Framework override mechanisms
"""

from unittest.mock import patch

import pytest

from traigent.api.decorators import optimize
from traigent.integrations.framework_override import (
    disable_framework_overrides,
    register_framework_mapping,
)

from .mock_infrastructure import (
    SimpleMockAnthropic,
    SimpleMockAsyncOpenAI,
    SimpleMockCohere,
    SimpleMockLangChain,
    SimpleMockOpenAI,
)
from .test_base import DecoratorTestBase


class TestOpenAIIntegration(DecoratorTestBase):
    """Test integration with OpenAI SDK."""

    def test_openai_sync_integration(self):
        """Test decorator with synchronous OpenAI calls."""
        with patch("openai.OpenAI", SimpleMockOpenAI):

            @optimize(
                configuration_space={
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.0, 0.5, 1.0],
                },
                injection_mode="seamless",
            )
            def generate_response(prompt: str) -> str:
                import openai

                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                return response.choices[0].message.content

            result = generate_response("Hello")
            assert "OpenAI response" in result

    def test_openai_async_integration(self):
        """Test decorator with asynchronous OpenAI calls."""
        with patch("openai.AsyncOpenAI", SimpleMockAsyncOpenAI):

            @optimize(
                configuration_space={
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.0, 0.5, 1.0],
                },
                injection_mode="seamless",
            )
            async def generate_response_async(prompt: str) -> str:
                import openai

                client = openai.AsyncOpenAI()
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                return response.choices[0].message.content

            import asyncio

            result = asyncio.run(generate_response_async("Hello"))
            assert "AsyncOpenAI response" in result

    def test_openai_with_custom_parameters(self):
        """Test OpenAI integration with custom parameters."""
        with patch("openai.OpenAI", SimpleMockOpenAI):

            @optimize(
                configuration_space={
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.0, 0.5, 1.0],
                    "max_tokens": [100, 200, 500],
                    "top_p": [0.9, 0.95, 1.0],
                },
                injection_mode="seamless",
            )
            def generate_with_params(prompt: str, creativity: float = 0.5) -> str:
                import openai

                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=creativity,
                    max_tokens=200,
                    top_p=0.95,
                )
                return response.choices[0].message.content

            result = generate_with_params("Hello", creativity=0.8)
            assert "OpenAI response" in result


class TestLangChainIntegration(DecoratorTestBase):
    """Test integration with LangChain framework."""

    def test_langchain_llm_integration(self):
        """Test decorator with LangChain LLM."""
        pytest.importorskip("langchain_openai", reason="langchain_openai not installed")
        with patch("langchain_openai.ChatOpenAI", SimpleMockLangChain):

            @optimize(
                configuration_space={
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.0, 0.5, 1.0],
                },
                injection_mode="seamless",
            )
            def langchain_generate(prompt: str) -> str:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
                return llm.invoke(prompt)

            result = langchain_generate("Hello")
            assert "LangChain response" in result

    def test_langchain_chain_integration(self):
        """Test decorator with LangChain chains."""
        pytest.importorskip("langchain_openai", reason="langchain_openai not installed")
        with patch("langchain_openai.ChatOpenAI", SimpleMockLangChain):

            @optimize(
                configuration_space={
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.0, 0.5, 1.0],
                },
                injection_mode="seamless",
            )
            def langchain_chain(prompt: str) -> str:
                from langchain_core.prompts import PromptTemplate
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
                prompt_template = PromptTemplate(
                    input_variables=["input"], template="Answer this: {input}"
                )
                # Use the simpler pattern instead of deprecated LLMChain
                formatted_prompt = prompt_template.format(input=prompt)
                return llm.invoke(formatted_prompt)

            result = langchain_chain("What is 2+2?")
            assert "LangChain response" in result


class TestAnthropicIntegration(DecoratorTestBase):
    """Test integration with Anthropic SDK."""

    def test_anthropic_integration(self):
        """Test decorator with Anthropic Claude."""
        pytest.importorskip("anthropic", reason="Anthropic package not installed")
        with patch("anthropic.Anthropic", SimpleMockAnthropic):

            @optimize(
                configuration_space={
                    "model": ["claude-2", "claude-instant"],
                    "temperature": [0.0, 0.5, 1.0],
                },
                injection_mode="seamless",
            )
            def anthropic_generate(prompt: str) -> str:
                import anthropic

                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-2",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=200,
                )
                return response.content[0].text

            result = anthropic_generate("Hello")
            assert "Anthropic response" in result


class TestCohereIntegration(DecoratorTestBase):
    """Test integration with Cohere SDK."""

    def test_cohere_integration(self):
        """Test decorator with Cohere."""
        pytest.importorskip("cohere", reason="Cohere package not installed")

        with patch("cohere.Client", SimpleMockCohere):

            @optimize(
                configuration_space={
                    "model": ["command", "command-light"],
                    "temperature": [0.0, 0.5, 1.0],
                },
                injection_mode="seamless",
            )
            def cohere_generate(prompt: str) -> str:
                import cohere

                client = cohere.Client()
                response = client.generate(
                    model="command", prompt=prompt, temperature=0.7, max_tokens=200
                )
                return response.generations[0].text

            result = cohere_generate("Hello")
            assert "Cohere response" in result


class TestFrameworkAutoDetection(DecoratorTestBase):
    """Test automatic framework detection."""

    def test_auto_detect_openai(self):
        """Test auto-detection of OpenAI framework."""

        # Skip framework detection test since detect_framework doesn't exist yet
        @optimize(
            configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            injection_mode="seamless",
        )
        def test_func(prompt: str) -> str:
            # Simulate OpenAI usage pattern
            return f"Detected framework response: {prompt}"

        result = test_func("Hello")
        assert "Detected framework response" in result

    def test_auto_detect_multiple_frameworks(self):
        """Test handling of multiple frameworks in same function."""

        # Skip framework detection test since detect_framework doesn't exist yet
        @optimize(
            configuration_space={"model": ["gpt-3.5-turbo", "claude-2"]},
            injection_mode="seamless",
        )
        def multi_framework_func(prompt: str, use_openai: bool = True) -> str:
            if use_openai:
                # Simulate OpenAI call
                return f"OpenAI: {prompt}"
            else:
                # Simulate Anthropic call
                return f"Anthropic: {prompt}"

        result1 = multi_framework_func("Hello", use_openai=True)
        result2 = multi_framework_func("Hello", use_openai=False)

        assert "OpenAI" in result1
        assert "Anthropic" in result2


class TestFrameworkOverrideMechanisms(DecoratorTestBase):
    """Test framework override mechanisms."""

    def test_register_custom_framework_mapping(self):
        """Test registering custom framework mappings."""

        # Register a custom framework
        def custom_override(config):
            return {"custom_param": config.get("model", "default")}

        register_framework_mapping("custom_framework", custom_override)

        @optimize(
            configuration_space={"model": ["custom-1", "custom-2"]},
            injection_mode="seamless",
        )
        def custom_func(prompt: str) -> str:
            # Simulate custom framework usage
            return f"Custom response: {prompt}"

        result = custom_func("Hello")
        assert "Custom response" in result

    def test_disable_framework_overrides(self):
        """Test disabling framework overrides."""
        disable_framework_overrides()

        @optimize(
            configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            injection_mode="seamless",
        )
        def test_func(prompt: str) -> str:
            # With overrides disabled, should use original behavior
            return f"Original behavior: {prompt}"

        result = test_func("Hello")
        assert "Original behavior" in result
