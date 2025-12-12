"""Mock LLM provider implementations for testing.

This module provides consistent mock implementations of all major LLM providers
used in the TraiGent SDK, with configuration logging capabilities for test validation.
"""

from typing import Any
from unittest.mock import Mock


class ConfigurationLogger:
    """Global configuration logger for test verification."""

    def __init__(self):
        self.logs: list[dict[str, Any]] = []

    def log(self, framework: str, method: str, config: dict[str, Any], source: str):
        """Log a configuration event."""
        self.logs.append(
            {
                "framework": framework,
                "method": method,
                "config": config,
                "source": source,
                "timestamp": len(self.logs),
            }
        )

    def count_config_occurrences(self, key: str, value: Any) -> int:
        """Count how many times a specific config key-value pair appears."""
        count = 0
        for log in self.logs:
            if key in log["config"] and log["config"][key] == value:
                count += 1
        return count

    def has_config(self, key: str, value: Any) -> bool:
        """Check if a specific config key-value pair exists in logs."""
        return self.count_config_occurrences(key, value) > 0

    def get_framework_configs(self, framework: str) -> list[dict[str, Any]]:
        """Get all configurations for a specific framework."""
        return [log["config"] for log in self.logs if log["framework"] == framework]

    def clear(self):
        """Clear all logs."""
        self.logs = []


# Global logger instance
config_logger = ConfigurationLogger()


class MockOpenAI:
    """Mock OpenAI client that logs configuration and provides realistic responses."""

    def __init__(self, **kwargs):
        self.init_config = kwargs
        config_logger.log("openai", "__init__", kwargs, "init")
        self.chat = self.Chat()
        self.completions = self.Completions()

    class Chat:
        def __init__(self):
            self.completions = self.ChatCompletions()

        class ChatCompletions:
            def create(self, **kwargs):
                config_logger.log("openai", "chat.completions.create", kwargs, "call")
                response = Mock()
                response.choices = [Mock()]
                response.choices[0].message.content = (
                    f"OpenAI response with {kwargs.get('temperature', 'default')} temperature"
                )
                return response

    class Completions:
        def create(self, **kwargs):
            config_logger.log("openai", "completions.create", kwargs, "call")
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].text = (
                f"OpenAI completion with {kwargs.get('temperature', 'default')} temperature"
            )
            return response


class MockAsyncOpenAI:
    """Mock async OpenAI client."""

    def __init__(self, **kwargs):
        self.init_config = kwargs
        config_logger.log("openai_async", "__init__", kwargs, "init")
        self.chat = self.Chat()
        self.completions = self.Completions()

    class Chat:
        def __init__(self):
            self.completions = self.ChatCompletions()

        class ChatCompletions:
            async def create(self, **kwargs):
                config_logger.log(
                    "openai_async", "chat.completions.create", kwargs, "call"
                )
                response = Mock()
                response.choices = [Mock()]
                response.choices[0].message.content = (
                    f"Async OpenAI response with {kwargs.get('temperature', 'default')} temperature"
                )
                return response

    class Completions:
        async def create(self, **kwargs):
            config_logger.log("openai_async", "completions.create", kwargs, "call")
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].text = (
                f"Async OpenAI completion with {kwargs.get('temperature', 'default')} temperature"
            )
            return response


class MockAnthropic:
    """Mock Anthropic client that logs configuration."""

    def __init__(self, **kwargs):
        self.init_config = kwargs
        config_logger.log("anthropic", "__init__", kwargs, "init")
        self.messages = self.Messages()

    class Messages:
        def create(self, **kwargs):
            config_logger.log("anthropic", "messages.create", kwargs, "call")
            response = Mock()
            response.content = [Mock()]
            response.content[0].text = (
                f"Anthropic response with {kwargs.get('temperature', 'default')} temperature"
            )
            return response

        def stream(self, **kwargs):
            config_logger.log("anthropic", "messages.stream", kwargs, "call")

            # Return an async generator-like mock
            async def mock_stream():
                yield Mock()

            return mock_stream()


class MockAsyncAnthropic:
    """Mock async Anthropic client."""

    def __init__(self, **kwargs):
        self.init_config = kwargs
        config_logger.log("anthropic_async", "__init__", kwargs, "init")
        self.messages = self.Messages()

    class Messages:
        async def create(self, **kwargs):
            config_logger.log("anthropic_async", "messages.create", kwargs, "call")
            response = Mock()
            response.content = [Mock()]
            response.content[0].text = (
                f"Async Anthropic response with {kwargs.get('temperature', 'default')} temperature"
            )
            return response


class MockLangChainLLM:
    """Mock LangChain LLM that logs configuration."""

    def __init__(self, **kwargs):
        self.init_config = kwargs
        config_logger.log("langchain", "__init__", kwargs, "init")

    def invoke(self, prompt: str, **kwargs):
        config_logger.log("langchain", "invoke", kwargs, "call")
        return f"LangChain response to: {prompt} with {kwargs.get('temperature', 'default')} temperature"

    def stream(self, prompt: str, **kwargs):
        config_logger.log("langchain", "stream", kwargs, "call")
        # Return a generator-like mock
        yield f"LangChain stream response to: {prompt}"

    async def astream(self, prompt: str, **kwargs):
        config_logger.log("langchain", "astream", kwargs, "call")
        # Return an async generator-like mock
        yield f"LangChain async stream response to: {prompt}"


class MockCohere:
    """Mock Cohere client that logs configuration."""

    def __init__(self, **kwargs):
        self.init_config = kwargs
        config_logger.log("cohere", "__init__", kwargs, "init")

    def generate(self, **kwargs):
        config_logger.log("cohere", "generate", kwargs, "call")
        response = Mock()
        response.generations = [Mock()]
        response.generations[0].text = (
            f"Cohere response with {kwargs.get('temperature', 'default')} temperature"
        )
        return response

    def chat(self, **kwargs):
        config_logger.log("cohere", "chat", kwargs, "call")
        response = Mock()
        response.text = f"Cohere chat response with {kwargs.get('temperature', 'default')} temperature"
        return response


class MockHuggingFacePipeline:
    """Mock HuggingFace pipeline that logs configuration."""

    def __init__(self, **kwargs):
        self.init_config = kwargs
        config_logger.log("huggingface", "__init__", kwargs, "init")

    def __call__(self, inputs, **kwargs):
        config_logger.log("huggingface", "__call__", kwargs, "call")
        return [
            {
                "generated_text": f"HuggingFace response with {kwargs.get('temperature', 'default')} temperature"
            }
        ]


# Mock framework classes for integration testing
class MockOpenAIClient(MockOpenAI):
    """Alias for MockOpenAI for consistency across test files."""

    pass


class MockAnthropicClient(MockAnthropic):
    """Alias for MockAnthropic for consistency across test files."""

    pass


class MockLangChainOpenAI(MockLangChainLLM):
    """Specific LangChain OpenAI implementation mock."""

    pass


# Provider factory for easy mock creation
def create_mock_provider(provider_type: str, **kwargs):
    """Factory function to create mock providers."""
    providers = {
        "openai": MockOpenAI,
        "openai_async": MockAsyncOpenAI,
        "anthropic": MockAnthropic,
        "anthropic_async": MockAsyncAnthropic,
        "langchain": MockLangChainLLM,
        "cohere": MockCohere,
        "huggingface": MockHuggingFacePipeline,
    }

    if provider_type not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return providers[provider_type](**kwargs)
