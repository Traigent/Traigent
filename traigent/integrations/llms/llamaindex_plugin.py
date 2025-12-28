"""LlamaIndex integration plugin for Traigent.

This plugin provides automatic parameter override functionality for LlamaIndex,
supporting various LLM providers, embedding models, and RAG components.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from pathlib import Path
from typing import Any

from traigent.config.types import TraigentConfig
from traigent.integrations.base_plugin import PluginMetadata, ValidationRule
from traigent.integrations.llms import LLMPlugin
from traigent.integrations.utils import Framework
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class LlamaIndexPlugin(LLMPlugin):
    """Plugin for LlamaIndex framework integration.

    Supports:
    - OpenAI LLM integration
    - Anthropic LLM integration
    - HuggingFace LLM integration
    - Cohere LLM integration
    - Various embedding models
    - Vector stores and indices
    - Query engines and retrievers
    """

    FRAMEWORK = Framework.LLAMAINDEX

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize LlamaIndex plugin."""
        super().__init__(config_path)

    def _get_supported_canonical_params(self) -> set[str]:
        """LlamaIndex supports common LLM params across wrapped providers.

        Since LlamaIndex wraps multiple providers (OpenAI, Anthropic, etc.),
        we support the common subset plus OpenAI-specific params that are
        widely used. The extra_mappings handle LlamaIndex-specific transforms.
        """
        return {
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "stream",
            "frequency_penalty",
            "presence_penalty",
        }

    def get_target_classes(self) -> list[str]:
        """Return list of LlamaIndex classes to override."""
        return [
            # Core LLM classes
            "llama_index.llms.openai.OpenAI",
            "llama_index.llms.anthropic.Anthropic",
            "llama_index.llms.cohere.Cohere",
            "llama_index.llms.huggingface.HuggingFaceLLM",
            "llama_index.llms.replicate.Replicate",
            "llama_index.llms.palm.PaLM",
            # Legacy imports (for backward compatibility)
            "llama_index.llms.OpenAI",
            "llama_index.llms.Anthropic",
            "llama_index.llms.Cohere",
            # Embedding models
            "llama_index.embeddings.openai.OpenAIEmbedding",
            "llama_index.embeddings.huggingface.HuggingFaceEmbedding",
            "llama_index.embeddings.cohere.CohereEmbedding",
            "llama_index.embeddings.langchain.LangchainEmbedding",
            # Legacy embedding imports
            "llama_index.embeddings.OpenAIEmbedding",
            "llama_index.embeddings.HuggingFaceEmbedding",
            # Service context components
            "llama_index.core.ServiceContext",
            "llama_index.ServiceContext",
            # LLM Predictor (deprecated but still used)
            "llama_index.llm_predictor.LLMPredictor",
            "llama_index.core.llm_predictor.LLMPredictor",
            # Chat engines
            "llama_index.chat_engine.SimpleChatEngine",
            "llama_index.chat_engine.CondenseQuestionChatEngine",
            "llama_index.chat_engine.ReActChatEngine",
            # Retrievers
            "llama_index.core.retrievers.BaseRetriever",
            "llama_index.retrievers.BaseRetriever",
            "llama_index.indices.vector_store.retrievers.VectorIndexRetriever",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to their methods that should be overridden."""
        return {
            # OpenAI LLM methods
            "llama_index.llms.openai.OpenAI": [
                "complete",
                "acomplete",
                "chat",
                "achat",
                "stream_complete",
                "stream_chat",
            ],
            "llama_index.llms.OpenAI": [
                "complete",
                "acomplete",
                "chat",
                "achat",
                "stream_complete",
                "stream_chat",
            ],
            # Anthropic LLM methods
            "llama_index.llms.anthropic.Anthropic": [
                "complete",
                "acomplete",
                "chat",
                "achat",
                "stream_complete",
                "stream_chat",
            ],
            "llama_index.llms.Anthropic": [
                "complete",
                "acomplete",
                "chat",
                "achat",
                "stream_complete",
                "stream_chat",
            ],
            # Cohere LLM methods
            "llama_index.llms.cohere.Cohere": [
                "complete",
                "acomplete",
                "chat",
                "achat",
                "stream_complete",
                "stream_chat",
            ],
            "llama_index.llms.Cohere": [
                "complete",
                "acomplete",
                "chat",
                "achat",
                "stream_complete",
                "stream_chat",
            ],
            # HuggingFace LLM methods
            "llama_index.llms.huggingface.HuggingFaceLLM": [
                "complete",
                "acomplete",
                "stream_complete",
            ],
            # Embedding methods
            "llama_index.embeddings.openai.OpenAIEmbedding": [
                "get_embedding",
                "aget_embedding",
                "get_query_embedding",
                "get_text_embedding",
            ],
            "llama_index.embeddings.OpenAIEmbedding": [
                "get_embedding",
                "aget_embedding",
                "get_query_embedding",
                "get_text_embedding",
            ],
            # Chat engine methods
            "llama_index.chat_engine.SimpleChatEngine": [
                "chat",
                "achat",
                "stream_chat",
                "astream_chat",
            ],
            # Retrievers
            "llama_index.core.retrievers.BaseRetriever": ["retrieve", "aretrieve"],
            "llama_index.retrievers.BaseRetriever": ["retrieve", "aretrieve"],
            "llama_index.indices.vector_store.retrievers.VectorIndexRetriever": [
                "retrieve",
                "aretrieve",
            ],
            # Service context methods
            "llama_index.core.ServiceContext": ["from_defaults"],
            "llama_index.ServiceContext": ["from_defaults"],
        }

    def _get_metadata(self) -> PluginMetadata:
        """Return metadata about this plugin."""
        return PluginMetadata(
            name="llamaindex",
            version="1.0.0",
            description="LlamaIndex framework integration for Traigent",
            author="Traigent Team",
            supported_packages=["llama_index", "llama-index"],
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return parameter mappings from Traigent to LlamaIndex."""
        return {
            # Core LLM parameters
            "model": "model",  # Most LlamaIndex LLMs use 'model'
            # Additional OpenAI-specific
            "n": "n",
            "logprobs": "logprobs",
            # Anthropic-specific mappings
            "max_tokens_to_sample": "max_tokens",  # Anthropic uses different name
            "top_k": "top_k",
            # HuggingFace-specific
            "model_name": "model_name",
            "device_map": "device_map",
            "model_kwargs": "model_kwargs",
            "generate_kwargs": "generate_kwargs",
            "tokenizer_name": "tokenizer_name",
            "tokenizer_kwargs": "tokenizer_kwargs",
            # Cohere-specific
            "cohere_api_key": "api_key",
            # Embedding parameters
            "embedding_model": "model_name",
            "embed_batch_size": "embed_batch_size",
            "chunk_size": "chunk_size",
            "chunk_overlap": "chunk_overlap",
            # Service context parameters
            "llm_predictor": "llm_predictor",
            "prompt_helper": "prompt_helper",
            "embed_model": "embed_model",
            "node_parser": "node_parser",
            "callback_manager": "callback_manager",
            # System/Context parameters
            "system_prompt": "system_prompt",
            "query_wrapper_prompt": "query_wrapper_prompt",
            "context_window": "context_window",
            "num_output": "num_output",
            # Retrieval parameters
            "similarity_top_k": "similarity_top_k",
            "similarity_cutoff": "similarity_cutoff",
            "alpha": "alpha",  # For hybrid search
            # Response synthesis parameters
            "response_mode": "response_mode",
            "use_async": "use_async",
            "streaming": "streaming",
            "response_format": "response_format",
            # API Keys (provider-specific)
            "openai_api_key": "api_key",
            "anthropic_api_key": "api_key",
            "replicate_api_key": "api_key",
            # Timeout and retry
            "timeout": "timeout",
            "max_retries": "max_retries",
            "retry_min_seconds": "retry_min_seconds",
            "retry_max_seconds": "retry_max_seconds",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return validation rules for LlamaIndex parameters."""
        return {
            "max_tokens": ValidationRule(
                min_value=1,
                max_value=128000,  # Support for larger context models
            ),
            "top_k": ValidationRule(min_value=1, max_value=100),
            "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "presence_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "similarity_top_k": ValidationRule(min_value=1, max_value=100),
            "similarity_cutoff": ValidationRule(min_value=0.0, max_value=1.0),
            "alpha": ValidationRule(min_value=0.0, max_value=1.0),
            "embed_batch_size": ValidationRule(min_value=1, max_value=512),
            "chunk_size": ValidationRule(min_value=1, max_value=8192),
            "chunk_overlap": ValidationRule(min_value=0, max_value=1024),
            "context_window": ValidationRule(min_value=512, max_value=128000),
            "num_output": ValidationRule(min_value=1, max_value=8192),
            "n": ValidationRule(min_value=1, max_value=10),
            "logprobs": ValidationRule(min_value=0, max_value=5),
            "timeout": ValidationRule(min_value=1, max_value=300),
            "max_retries": ValidationRule(min_value=0, max_value=10),
            "response_mode": ValidationRule(
                allowed_values=[
                    "refine",
                    "compact",
                    "tree_summarize",
                    "simple_summarize",
                    "no_text",
                    "accumulate",
                ]
            ),
            "model": ValidationRule(
                allowed_values=None,  # Allow any model name
                custom_validator="_validate_model_name",
            ),
            "streaming": ValidationRule(allowed_values=[True, False]),
            "stream": ValidationRule(allowed_values=[True, False]),
            "use_async": ValidationRule(allowed_values=[True, False]),
        }

    def _validate_model_name(self, param_name: str, value: Any) -> list[str]:
        """Custom validator for model names.

        Args:
            param_name: Parameter name
            value: Parameter value

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Allow None for optional model
        if value is None:
            return errors

        # Check if it's a string
        if not isinstance(value, str):
            errors.append(f"Model name must be a string, got {type(value).__name__}")
            return errors

        # Check for empty string
        if value == "":
            errors.append("Model name cannot be empty")

        # Check for common invalid patterns
        if (
            value
            and not value.replace("-", "")
            .replace("_", "")
            .replace(".", "")
            .replace("/", "")
            .isalnum()
        ):
            errors.append(f"Model name contains invalid characters: {value}")

        return errors

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply parameter overrides with LlamaIndex-specific handling.

        Args:
            kwargs: Original keyword arguments
            config: TraigentConfig or dict with override parameters

        Returns:
            Modified kwargs with overrides applied
        """
        config_obj = self._normalize_config(config)

        # First apply base overrides
        overridden = super().apply_overrides(kwargs, config_obj)

        # Handle LlamaIndex-specific parameter transformations

        # Handle model name variations
        if "model" in overridden and "model_name" not in overridden:
            # Some LlamaIndex classes use model_name instead of model
            overridden["model_name"] = overridden["model"]

        # Handle max_tokens variations
        if "max_tokens" in overridden and "max_tokens_to_sample" not in overridden:
            # Anthropic in LlamaIndex uses max_tokens_to_sample
            overridden["max_tokens_to_sample"] = overridden["max_tokens"]

        # Handle streaming flags
        if "stream" in overridden and "streaming" not in overridden:
            overridden["streaming"] = overridden["stream"]

        # Remove None values that might cause issues
        overridden = {k: v for k, v in overridden.items() if v is not None}

        return overridden
