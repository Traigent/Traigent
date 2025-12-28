"""Shared mock infrastructure for decorator tests.

This module provides lightweight mock implementations of various LLM frameworks
for testing Traigent decorator functionality. These mocks focus on configuration
tracking rather than realistic behavior simulation.
"""

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, Mock


@dataclass
class ConfigurationLog:
    """Log entry for configuration changes."""

    timestamp: float
    framework: str
    method: str
    config: dict[str, Any]
    source: str  # 'init' or 'call'


class ConfigurationLogger:
    """Simple configuration logger for test verification."""

    def __init__(self):
        self.logs: list[ConfigurationLog] = []

    def log(self, framework: str, method: str, config: dict[str, Any], source: str):
        """Log a configuration event."""
        self.logs.append(
            ConfigurationLog(
                timestamp=time.time(),
                framework=framework,
                method=method,
                config=config,
                source=source,
            )
        )

    def get_logs_for_framework(self, framework: str) -> list[ConfigurationLog]:
        """Get all logs for a specific framework."""
        return [log for log in self.logs if log.framework == framework]

    def has_config(self, key: str, value: Any) -> bool:
        """Check if a specific config key-value pair exists in logs."""
        for log in self.logs:
            if key in log.config and log.config[key] == value:
                return True
        return False

    def clear(self):
        """Clear all logs."""
        self.logs.clear()


# Simplified mock classes that return predictable responses
class SimpleMockOpenAI:
    """Minimal OpenAI mock that returns predictable responses."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.chat = Mock()
        self.chat.completions.create = Mock(
            return_value=Mock(choices=[Mock(message=Mock(content="OpenAI response"))])
        )


class SimpleMockAsyncOpenAI:
    """Minimal async OpenAI mock."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.chat = Mock()
        self.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content="AsyncOpenAI response"))]
            )
        )


class SimpleMockLangChain:
    """Minimal LangChain LLM mock."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def invoke(self, prompt: str, **kwargs):
        return "LangChain response"

    async def ainvoke(self, prompt: str, **kwargs):
        return "AsyncLangChain response"


class SimpleMockAnthropic:
    """Minimal Anthropic mock."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.messages = Mock()
        self.messages.create = Mock(
            return_value=Mock(content=[Mock(text="Anthropic response")])
        )


class SimpleMockCohere:
    """Minimal Cohere mock."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def generate(self, prompt: str, **kwargs):
        return Mock(generations=[Mock(text="Cohere response")])


def create_mock_framework(framework: str, async_mode: bool = False):
    """Factory function to create appropriate mock based on framework."""
    if framework == "openai":
        return SimpleMockAsyncOpenAI if async_mode else SimpleMockOpenAI
    elif framework == "langchain":
        return SimpleMockLangChain
    elif framework == "anthropic":
        return SimpleMockAnthropic
    elif framework == "cohere":
        return SimpleMockCohere
    else:
        raise ValueError(f"Unknown framework: {framework}")


def create_simple_dataset(size: int = 3):
    """Create a simple test dataset."""
    from traigent.evaluators.base import Dataset, EvaluationExample

    examples = []
    for i in range(size):
        examples.append(
            EvaluationExample(
                input_data={"text": f"Input {i}"}, expected_output=f"Output {i}"
            )
        )
    return Dataset(examples)
