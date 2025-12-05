"""Tests for LLMPlugin base class."""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.llms import LLMPlugin
from traigent.integrations.utils import Framework


class ConcreteOpenAIPlugin(LLMPlugin):
    """Concrete test plugin for OpenAI."""

    FRAMEWORK = Framework.OPENAI

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_openai",
            version="1.0.0",
            supported_packages=["openai"],
            priority=IntegrationPriority.HIGH,
        )

    def get_target_classes(self) -> list[str]:
        return ["openai.OpenAI"]

    def get_target_methods(self) -> dict[str, list[str]]:
        return {"openai.OpenAI": ["chat.completions.create"]}


class ConcreteLangChainPlugin(LLMPlugin):
    """Concrete test plugin for LangChain."""

    FRAMEWORK = Framework.LANGCHAIN

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_langchain",
            version="1.0.0",
            supported_packages=["langchain"],
            priority=IntegrationPriority.NORMAL,
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Add LangChain-specific mappings."""
        return {
            "verbose": "verbose",
            "callbacks": "callbacks",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Add LangChain-specific validation."""
        return {
            "verbose": ValidationRule(allowed_values=[True, False]),
        }

    def get_target_classes(self) -> list[str]:
        return ["langchain_openai.ChatOpenAI"]

    def get_target_methods(self) -> dict[str, list[str]]:
        return {"langchain_openai.ChatOpenAI": ["invoke"]}


class ConcreteBypassPlugin(LLMPlugin):
    """Plugin that bypasses normalizer for custom mappings."""

    FRAMEWORK = Framework.HUGGINGFACE

    def _should_use_normalizer(self) -> bool:
        return False

    def _get_extra_mappings(self) -> dict[str, str]:
        """Completely custom mappings."""
        return {
            "model": "model_id",
            "temperature": "temp",
            "max_tokens": "max_length",
        }

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_bypass",
            version="1.0.0",
            supported_packages=["transformers"],
            priority=IntegrationPriority.LOW,
        )

    def get_target_classes(self) -> list[str]:
        return ["transformers.Pipeline"]

    def get_target_methods(self) -> dict[str, list[str]]:
        return {}


class TestLLMPluginMappings:
    """Test parameter mapping functionality."""

    def test_openai_mappings_from_normalizer(self):
        """OpenAI plugin gets mappings from normalizer."""
        plugin = ConcreteOpenAIPlugin()
        mappings = plugin._get_default_mappings()

        # These should come from ParameterNormalizer
        assert mappings["model"] == "model"
        assert mappings["max_tokens"] == "max_tokens"
        assert mappings["temperature"] == "temperature"
        assert mappings["stream"] == "stream"
        assert mappings["top_p"] == "top_p"

    def test_langchain_mappings_with_extras(self):
        """LangChain plugin gets normalizer + extra mappings."""
        plugin = ConcreteLangChainPlugin()
        mappings = plugin._get_default_mappings()

        # From normalizer (note: LangChain has different names)
        assert mappings["model"] == "model_name"
        assert mappings["stream"] == "streaming"
        assert mappings["max_tokens"] == "max_tokens"

        # From _get_extra_mappings
        assert mappings["verbose"] == "verbose"
        assert mappings["callbacks"] == "callbacks"

    def test_bypass_normalizer(self):
        """Plugin can bypass normalizer for custom mappings."""
        plugin = ConcreteBypassPlugin()
        mappings = plugin._get_default_mappings()

        # Should only have custom mappings, not from normalizer
        assert mappings["model"] == "model_id"
        assert mappings["temperature"] == "temp"
        assert mappings["max_tokens"] == "max_length"

        # Should NOT have other normalizer params
        assert "top_p" not in mappings
        assert "stop" not in mappings

    def test_extra_mappings_override_normalizer(self):
        """Extra mappings take precedence over normalizer."""

        class OverridePlugin(LLMPlugin):
            FRAMEWORK = Framework.OPENAI

            def _get_extra_mappings(self) -> dict[str, str]:
                # Override normalizer's "model" mapping
                return {"model": "custom_model_field"}

            def _get_metadata(self):
                return PluginMetadata(
                    name="test", version="1.0.0", supported_packages=["openai"]
                )

            def get_target_classes(self):
                return []

            def get_target_methods(self):
                return {}

        plugin = OverridePlugin()
        mappings = plugin._get_default_mappings()

        # Extra mapping should override normalizer
        assert mappings["model"] == "custom_model_field"
        # Other normalizer mappings should still work
        assert mappings["temperature"] == "temperature"


class TestLLMPluginValidation:
    """Test validation rule functionality."""

    def test_common_validation_rules(self):
        """All LLM plugins have common validation rules."""
        plugin = ConcreteOpenAIPlugin()
        rules = plugin._get_common_validation_rules()

        assert "temperature" in rules
        assert rules["temperature"].min_value == 0.0
        assert rules["temperature"].max_value == 2.0

        assert "top_p" in rules
        assert rules["top_p"].min_value == 0.0
        assert rules["top_p"].max_value == 1.0

        assert "max_tokens" in rules
        assert rules["max_tokens"].min_value == 1

    def test_provider_specific_rules_merged(self):
        """Provider-specific rules are merged with common rules."""
        plugin = ConcreteLangChainPlugin()
        rules = plugin._get_validation_rules()

        # Common rules should be present
        assert "temperature" in rules
        assert "top_p" in rules

        # Provider-specific rules should be added
        assert "verbose" in rules
        assert rules["verbose"].allowed_values == [True, False]

    def test_provider_rules_override_common(self):
        """Provider-specific rules override common rules."""

        class CustomRulesPlugin(LLMPlugin):
            FRAMEWORK = Framework.OPENAI

            def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
                # Override temperature range
                return {
                    "temperature": ValidationRule(min_value=0.0, max_value=1.0),
                }

            def _get_metadata(self):
                return PluginMetadata(
                    name="test", version="1.0.0", supported_packages=["openai"]
                )

            def get_target_classes(self):
                return []

            def get_target_methods(self):
                return {}

        plugin = CustomRulesPlugin()
        rules = plugin._get_validation_rules()

        # Provider-specific rule should override common
        assert rules["temperature"].max_value == 1.0  # Not 2.0


class TestLLMPluginFramework:
    """Test framework configuration."""

    def test_framework_required(self):
        """Plugin without FRAMEWORK still works but gets no normalizer mappings."""

        class NoFrameworkPlugin(LLMPlugin):
            # FRAMEWORK not set

            def _get_metadata(self):
                return PluginMetadata(
                    name="test", version="1.0.0", supported_packages=["custom"]
                )

            def get_target_classes(self):
                return []

            def get_target_methods(self):
                return {}

        plugin = NoFrameworkPlugin()
        mappings = plugin._get_default_mappings()

        # Should be empty (no normalizer mappings without FRAMEWORK)
        assert mappings == {}

    def test_different_frameworks_different_mappings(self):
        """Different frameworks get different parameter names."""
        openai_plugin = ConcreteOpenAIPlugin()
        langchain_plugin = ConcreteLangChainPlugin()

        openai_mappings = openai_plugin._get_default_mappings()
        langchain_mappings = langchain_plugin._get_default_mappings()

        # Same canonical param, different framework names
        assert openai_mappings["model"] == "model"
        assert langchain_mappings["model"] == "model_name"

        assert openai_mappings["stream"] == "stream"
        assert langchain_mappings["stream"] == "streaming"


class TestLLMPluginSupportedParams:
    """Test supported canonical params filtering."""

    def test_default_supports_all_params(self):
        """By default, all canonical params are supported."""
        plugin = ConcreteOpenAIPlugin()
        supported = plugin._get_supported_canonical_params()

        # Default returns None (meaning all supported)
        assert supported is None

    def test_filtered_mappings_with_supported_params(self):
        """Plugin with supported params only maps those params."""

        class FilteredPlugin(LLMPlugin):
            FRAMEWORK = Framework.BEDROCK

            def _get_supported_canonical_params(self) -> set[str]:
                return {"model", "temperature", "max_tokens"}

            def _get_metadata(self):
                return PluginMetadata(
                    name="test", version="1.0.0", supported_packages=["boto3"]
                )

            def get_target_classes(self):
                return []

            def get_target_methods(self):
                return {}

        plugin = FilteredPlugin()
        mappings = plugin._get_default_mappings()

        # Should only have the supported params
        assert "model" in mappings
        assert "temperature" in mappings
        assert "max_tokens" in mappings

        # Should NOT have unsupported params
        assert "frequency_penalty" not in mappings
        assert "presence_penalty" not in mappings
        assert "top_k" not in mappings

    def test_extra_mappings_bypass_filter(self):
        """Extra mappings are added regardless of supported params filter."""

        class PluginWithExtras(LLMPlugin):
            FRAMEWORK = Framework.BEDROCK

            def _get_supported_canonical_params(self) -> set[str]:
                return {"model"}  # Only support model from normalizer

            def _get_extra_mappings(self) -> dict[str, str]:
                return {"custom_param": "bedrock_custom_param"}

            def _get_metadata(self):
                return PluginMetadata(
                    name="test", version="1.0.0", supported_packages=["boto3"]
                )

            def get_target_classes(self):
                return []

            def get_target_methods(self):
                return {}

        plugin = PluginWithExtras()
        mappings = plugin._get_default_mappings()

        # Should have model from normalizer (filtered)
        assert "model" in mappings

        # Should have custom_param from extras (not filtered)
        assert "custom_param" in mappings
        assert mappings["custom_param"] == "bedrock_custom_param"

        # Should NOT have temperature (filtered out)
        assert "temperature" not in mappings

    def test_empty_supported_params_means_no_normalizer_mappings(self):
        """Empty supported params set means no params from normalizer."""

        class NoParamsPlugin(LLMPlugin):
            FRAMEWORK = Framework.OPENAI

            def _get_supported_canonical_params(self) -> set[str]:
                return set()  # Empty - no normalizer params

            def _get_extra_mappings(self) -> dict[str, str]:
                return {"only_extra": "only_extra_value"}

            def _get_metadata(self):
                return PluginMetadata(
                    name="test", version="1.0.0", supported_packages=["openai"]
                )

            def get_target_classes(self):
                return []

            def get_target_methods(self):
                return {}

        plugin = NoParamsPlugin()
        mappings = plugin._get_default_mappings()

        # Should only have extras, no normalizer mappings
        assert len(mappings) == 1
        assert mappings["only_extra"] == "only_extra_value"


class TestLLMPluginInheritance:
    """Test that LLMPlugin properly inherits from IntegrationPlugin."""

    def test_is_integration_plugin(self):
        """LLMPlugin is an IntegrationPlugin."""
        from traigent.integrations.base_plugin import IntegrationPlugin

        plugin = ConcreteOpenAIPlugin()
        assert isinstance(plugin, IntegrationPlugin)
        assert isinstance(plugin, LLMPlugin)

    def test_has_required_methods(self):
        """LLMPlugin has all required IntegrationPlugin methods."""
        plugin = ConcreteOpenAIPlugin()

        # Required by IntegrationPlugin
        assert hasattr(plugin, "_get_metadata")
        assert hasattr(plugin, "_get_default_mappings")
        assert hasattr(plugin, "_get_validation_rules")
        assert hasattr(plugin, "get_target_classes")
        assert hasattr(plugin, "get_target_methods")
        assert hasattr(plugin, "apply_overrides")

    def test_metadata_accessible(self):
        """Plugin metadata is accessible."""
        plugin = ConcreteOpenAIPlugin()
        metadata = plugin._get_metadata()

        assert metadata.name == "test_openai"
        assert metadata.version == "1.0.0"
        assert "openai" in metadata.supported_packages
