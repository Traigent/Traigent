"""Unit tests for cloud provider integration plugins."""

import pytest

from traigent.config.types import TraigentConfig
from traigent.integrations.llms.azure_openai_plugin import AzureOpenAIPlugin
from traigent.integrations.llms.bedrock_plugin import BedrockPlugin
from traigent.integrations.llms.gemini_plugin import GeminiPlugin


@pytest.fixture
def bedrock_plugin():
    return BedrockPlugin()


@pytest.fixture
def azure_plugin():
    return AzureOpenAIPlugin()


@pytest.fixture
def gemini_plugin():
    return GeminiPlugin()


def test_bedrock_metadata(bedrock_plugin):
    assert bedrock_plugin.metadata.name == "bedrock"
    assert "boto3" in bedrock_plugin.metadata.supported_packages
    assert bedrock_plugin.metadata.priority.name == "NORMAL"


def test_bedrock_mappings(bedrock_plugin):
    mappings = bedrock_plugin.get_parameter_mappings()
    assert mappings["model"] == "model_id"
    assert mappings["stop_sequences"] == "stop_sequences"


def test_bedrock_overrides(bedrock_plugin):
    config = TraigentConfig(
        model="anthropic.claude-3-sonnet",
        temperature=0.7,
        custom_params={"stop": ["END"], "region_name": "us-east-1"},
    )
    kwargs = {"messages": []}

    overridden = bedrock_plugin.apply_overrides(kwargs, config)

    assert overridden["model_id"] == "anthropic.claude-3-sonnet"
    assert overridden["temperature"] == 0.7
    # stop_sequences is moved to extra_params because BedrockChatClient doesn't accept it directly
    assert overridden["extra_params"]["stop_sequences"] == ["END"]
    # region_name is a client construction param, NOT a per-request param
    # It should be filtered out, not passed to invoke()
    assert "region_name" not in overridden
    assert "region_name" not in overridden.get("extra_params", {})


def test_azure_metadata(azure_plugin):
    assert azure_plugin.metadata.name == "azure_openai"
    assert "openai" in azure_plugin.metadata.supported_packages


def test_azure_overrides(azure_plugin):
    config = TraigentConfig(
        model="gpt-4",
        custom_params={
            "deployment": "my-deployment",
            "api_version": "2023-05-15",
            "response_format": {"type": "json_object"},
        },
    )
    kwargs = {}

    overridden = azure_plugin.apply_overrides(kwargs, config)

    # Deployment should override model if present in custom_params
    assert overridden["model"] == "my-deployment"
    assert overridden["api_version"] == "2023-05-15"
    assert overridden["response_format"] == {"type": "json_object"}


def test_gemini_metadata(gemini_plugin):
    assert gemini_plugin.metadata.name == "gemini"
    assert "google-generativeai" in gemini_plugin.metadata.supported_packages


def test_gemini_overrides(gemini_plugin):
    config = TraigentConfig(
        model="gemini-pro", max_tokens=100, custom_params={"candidate_count": 1}
    )
    kwargs = {}

    overridden = gemini_plugin.apply_overrides(kwargs, config)

    assert overridden["model_name"] == "gemini-pro"
    # Generation params should be wrapped in generation_config
    assert "generation_config" in overridden
    assert overridden["generation_config"]["max_output_tokens"] == 100
    assert overridden["generation_config"]["candidate_count"] == 1
