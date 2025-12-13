"""Tests for the LangChain integration plugin."""

import pytest

from traigent.integrations.llms.langchain_plugin import LangChainPlugin
from traigent.integrations.utils import Framework


class TestLangChainPlugin:
    """LangChain plugin behaviours."""

    def setup_method(self) -> None:
        self.plugin = LangChainPlugin()

    def test_apply_overrides_with_dict_config(self) -> None:
        """Plugin should handle raw dict payloads without attribute errors."""
        config_payload = {
            "model": "gpt-4",
            "stream": True,
            "search_type": "mmr",
            "lambda_mult": 0.35,
            "k": 4,
        }

        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["model_name"] == "gpt-4"
        assert "model" not in overridden
        assert overridden["streaming"] is True
        assert overridden["k"] == 4
        assert overridden["search_kwargs"]["lambda_mult"] == pytest.approx(0.35)


class TestLangChainParameterMappings:
    """Test parameter mapping via ParameterNormalizer (declarative approach)."""

    def setup_method(self) -> None:
        self.plugin = LangChainPlugin()

    def test_model_to_model_name_mapping(self) -> None:
        """Test that 'model' is correctly mapped to 'model_name' via normalizer."""
        config_payload = {"model": "gpt-4"}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert "model_name" in overridden
        assert overridden["model_name"] == "gpt-4"
        # Original 'model' key should not be in output (was mapped)
        assert "model" not in overridden

    def test_model_name_direct_passthrough(self) -> None:
        """Test that 'model_name' provided directly passes through unchanged."""
        config_payload = {"model_name": "claude-3-opus"}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["model_name"] == "claude-3-opus"

    def test_stream_to_streaming_mapping(self) -> None:
        """Test that 'stream' is correctly mapped to 'streaming' via normalizer."""
        config_payload = {"stream": True}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert "streaming" in overridden
        assert overridden["streaming"] is True
        # Original 'stream' key should not be in output (was mapped)
        assert "stream" not in overridden

    def test_streaming_direct_passthrough(self) -> None:
        """Test that 'streaming' provided directly passes through unchanged."""
        config_payload = {"streaming": False}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["streaming"] is False

    def test_max_tokens_preserved(self) -> None:
        """Test that max_tokens is preserved (same name in LangChain)."""
        config_payload = {"max_tokens": 1000}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["max_tokens"] == 1000

    def test_temperature_preserved(self) -> None:
        """Test that temperature is preserved."""
        config_payload = {"temperature": 0.7}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["temperature"] == pytest.approx(0.7)

    def test_top_p_preserved(self) -> None:
        """Test that top_p is preserved."""
        config_payload = {"top_p": 0.95}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["top_p"] == pytest.approx(0.95)

    def test_combined_mappings(self) -> None:
        """Test multiple mappings work together."""
        config_payload = {
            "model": "gpt-4-turbo",
            "stream": True,
            "temperature": 0.5,
            "max_tokens": 2000,
        }
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["model_name"] == "gpt-4-turbo"
        assert overridden["streaming"] is True
        assert overridden["temperature"] == pytest.approx(0.5)
        assert overridden["max_tokens"] == 2000

    def test_user_provided_kwarg_not_overwritten(self) -> None:
        """Test that user-provided kwargs take precedence over config mappings."""
        # User already provided model_name in kwargs
        kwargs = {"model_name": "user-specified-model"}
        config_payload = {"model": "config-model"}

        overridden = self.plugin.apply_overrides(kwargs, config_payload)

        # User's value should win
        assert overridden["model_name"] == "user-specified-model"


class TestLangChainRetrieverParameters:
    """Test LangChain-specific retriever parameter handling."""

    def setup_method(self) -> None:
        self.plugin = LangChainPlugin()

    def test_mmr_search_with_lambda_mult(self) -> None:
        """Test MMR search type sets lambda_mult in search_kwargs."""
        config_payload = {
            "search_type": "mmr",
            "lambda_mult": 0.5,
        }
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert "search_kwargs" in overridden
        assert overridden["search_kwargs"]["lambda_mult"] == pytest.approx(0.5)

    def test_similarity_score_threshold_search(self) -> None:
        """Test similarity_score_threshold search type sets score_threshold."""
        config_payload = {
            "search_type": "similarity_score_threshold",
            "score_threshold": 0.8,
        }
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert "search_kwargs" in overridden
        assert overridden["search_kwargs"]["score_threshold"] == pytest.approx(0.8)

    def test_k_parameter_passthrough(self) -> None:
        """Test that k parameter is passed through for general use."""
        config_payload = {"k": 10}
        overridden = self.plugin.apply_overrides({}, config_payload)

        # k should be present (either direct or in search_kwargs depending on context)
        assert (
            overridden.get("k") == 10
            or overridden.get("search_kwargs", {}).get("k") == 10
        )

    def test_k_routed_to_search_kwargs_for_retriever(self) -> None:
        """Test that k is routed to search_kwargs when __self__ is Retriever-like.

        The plugin routes k into search_kwargs based on checking if __self__
        contains 'Retriever' or 'VectorStore' in its type string.
        """

        class FakeRetriever:
            """Fake retriever class to simulate LangChain retriever context."""

            pass

        config_payload = {"k": 5}
        # Simulate being called in retriever context via __self__
        kwargs = {"__self__": FakeRetriever()}
        overridden = self.plugin.apply_overrides(kwargs, config_payload)

        # k should be routed into search_kwargs for retriever-like objects
        assert "search_kwargs" in overridden
        assert overridden["search_kwargs"]["k"] == 5

    def test_k_routed_to_search_kwargs_for_vectorstore(self) -> None:
        """Test that k is routed to search_kwargs when __self__ is VectorStore-like."""

        class FakeVectorStore:
            """Fake vector store class to simulate LangChain VectorStore context."""

            pass

        config_payload = {"k": 8}
        kwargs = {"__self__": FakeVectorStore()}
        overridden = self.plugin.apply_overrides(kwargs, config_payload)

        # k should be routed into search_kwargs for VectorStore-like objects
        assert "search_kwargs" in overridden
        assert overridden["search_kwargs"]["k"] == 8


class TestLangChainExtraMappings:
    """Test LangChain-specific extra mappings from _get_extra_mappings()."""

    def setup_method(self) -> None:
        self.plugin = LangChainPlugin()

    def test_extra_mappings_include_model(self) -> None:
        """Verify model→model_name is in extra mappings."""
        mappings = self.plugin._get_extra_mappings()
        assert "model" in mappings
        assert mappings["model"] == "model_name"

    def test_extra_mappings_include_langchain_specific(self) -> None:
        """Verify LangChain-specific params are in extra mappings."""
        mappings = self.plugin._get_extra_mappings()

        # Check for LangChain-specific params
        assert "verbose" in mappings
        assert "callbacks" in mappings
        assert "memory" in mappings
        assert "agent_type" in mappings
        assert "chain_type" in mappings
        assert "search_type" in mappings

    def test_retrieval_params_in_mappings(self) -> None:
        """Verify retrieval/RAG params are in mappings."""
        mappings = self.plugin._get_extra_mappings()

        assert "k" in mappings
        assert "fetch_k" in mappings
        assert "lambda_mult" in mappings
        assert "score_threshold" in mappings


class TestLangChainValidationRules:
    """Test LangChain-specific validation rules."""

    def setup_method(self) -> None:
        self.plugin = LangChainPlugin()

    def test_k_validation_range(self) -> None:
        """Test k parameter has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "k" in rules
        assert rules["k"].min_value == 1
        assert rules["k"].max_value == 100

    def test_lambda_mult_validation_range(self) -> None:
        """Test lambda_mult parameter has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "lambda_mult" in rules
        assert rules["lambda_mult"].min_value == 0.0
        assert rules["lambda_mult"].max_value == 1.0

    def test_search_type_allowed_values(self) -> None:
        """Test search_type has correct allowed values."""
        rules = self.plugin._get_provider_specific_rules()
        assert "search_type" in rules
        assert set(rules["search_type"].allowed_values) == {
            "similarity",
            "mmr",
            "similarity_score_threshold",
        }

    def test_streaming_allowed_values(self) -> None:
        """Test streaming parameter accepts boolean."""
        rules = self.plugin._get_provider_specific_rules()
        assert "streaming" in rules
        assert rules["streaming"].allowed_values == [True, False]


class TestLangChainPluginMetadata:
    """Test LangChain plugin metadata and framework."""

    def setup_method(self) -> None:
        self.plugin = LangChainPlugin()

    def test_framework_is_langchain(self) -> None:
        """Test plugin identifies as LangChain framework."""
        assert self.plugin.FRAMEWORK == Framework.LANGCHAIN

    def test_metadata_name(self) -> None:
        """Test plugin metadata has correct name."""
        assert self.plugin.metadata.name == "langchain"

    def test_supported_packages(self) -> None:
        """Test plugin lists supported LangChain packages."""
        packages = self.plugin.metadata.supported_packages
        assert "langchain" in packages
        assert "langchain-core" in packages
        assert "langchain-openai" in packages
        assert "langchain-anthropic" in packages

    def test_target_classes_include_core_llm(self) -> None:
        """Test target classes include core LLM classes."""
        classes = self.plugin.get_target_classes()
        assert "langchain_openai.ChatOpenAI" in classes
        assert "langchain_anthropic.ChatAnthropic" in classes

    def test_target_methods_include_invoke(self) -> None:
        """Test target methods include invoke methods."""
        methods = self.plugin.get_target_methods()
        openai_methods = methods.get("langchain_openai.ChatOpenAI", [])
        assert "invoke" in openai_methods
        assert "ainvoke" in openai_methods
