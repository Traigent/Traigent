"""LangChain integration plugin for Traigent.

This module provides the LangChain-specific plugin implementation for
parameter mappings and framework overrides.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import importlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.llms.base_llm_plugin import LLMPlugin
from traigent.integrations.utils import Framework

if TYPE_CHECKING:
    from traigent.config.types import TraigentConfig

logger = logging.getLogger(__name__)


class LangChainPlugin(LLMPlugin):
    """Plugin for LangChain framework integration."""

    FRAMEWORK = Framework.LANGCHAIN

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize LangChain plugin with dynamic discovery."""
        super().__init__(config_path)
        self._discovered_classes: dict[str, Any] = {}
        self._discover_langchain_components()

    def _get_metadata(self) -> PluginMetadata:
        """Return LangChain plugin metadata."""
        return PluginMetadata(
            name="langchain",
            version="1.0.0",
            supported_packages=[
                "langchain",
                "langchain-core",
                "langchain-community",
                "langchain-openai",
                "langchain-anthropic",
                "langchain-google",
                "langchain-aws",
                "langchain-huggingface",
            ],
            priority=IntegrationPriority.HIGH,
            description="LangChain framework integration for LLM chains and agents",
            author="Traigent Team",
            requires_packages=["langchain>=0.1.0"],
            supports_versions={"langchain": "0.1", "langchain-core": "0.1"},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return LangChain-specific parameter mappings not in ParameterNormalizer."""
        return {
            # LangChain uses model_name in many places
            "model": "model_name",
            "model_name": "model_name",
            # LangChain-specific
            "verbose": "verbose",
            "callbacks": "callbacks",
            "tags": "tags",
            "metadata": "metadata",
            "run_name": "run_name",
            # Memory and context
            "memory": "memory",
            "return_messages": "return_messages",
            "max_tokens_limit": "max_tokens_limit",
            # Agent-specific
            "agent_type": "agent_type",
            "handle_parsing_errors": "handle_parsing_errors",
            "max_iterations": "max_iterations",
            "max_execution_time": "max_execution_time",
            "early_stopping_method": "early_stopping_method",
            # Chain-specific
            "chain_type": "chain_type",
            "return_intermediate_steps": "return_intermediate_steps",
            "include_run_info": "include_run_info",
            # Retrieval/RAG parameters
            "k": "k",
            "fetch_k": "fetch_k",
            "lambda_mult": "lambda_mult",
            "filter": "filter",
            "search_type": "search_type",
            "score_threshold": "score_threshold",
            # Embedding-specific
            "embedding_model": "model_name",
            "chunk_size": "chunk_size",
            "chunk_overlap": "chunk_overlap",
            # Provider-specific API keys
            "openai_api_key": "openai_api_key",
            "anthropic_api_key": "anthropic_api_key",
            "google_api_key": "google_api_key",
            "aws_access_key_id": "aws_access_key_id",
            "aws_secret_access_key": "aws_secret_access_key",
            "aws_region": "region_name",
            # Timeout and retry
            "request_timeout": "request_timeout",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return LangChain-specific validation rules."""
        return {
            "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "presence_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "k": ValidationRule(min_value=1, max_value=100),
            "fetch_k": ValidationRule(min_value=1, max_value=1000),
            "lambda_mult": ValidationRule(min_value=0.0, max_value=1.0),
            "score_threshold": ValidationRule(min_value=0.0, max_value=1.0),
            "search_type": ValidationRule(
                allowed_values=["similarity", "mmr", "similarity_score_threshold"]
            ),
            "chunk_size": ValidationRule(min_value=1, max_value=10000),
            "chunk_overlap": ValidationRule(min_value=0, max_value=5000),
            "max_iterations": ValidationRule(min_value=1, max_value=100),
            "max_execution_time": ValidationRule(min_value=1.0, max_value=3600.0),
            "streaming": ValidationRule(allowed_values=[True, False]),
            "verbose": ValidationRule(allowed_values=[True, False]),
            "return_messages": ValidationRule(allowed_values=[True, False]),
            "return_intermediate_steps": ValidationRule(allowed_values=[True, False]),
            "handle_parsing_errors": ValidationRule(
                allowed_values=[True, False, "fix"]
            ),
            "timeout": ValidationRule(min_value=1, max_value=600),
            "max_retries": ValidationRule(min_value=0, max_value=10),
        }

    def get_target_classes(self) -> list[str]:
        """Return list of LangChain classes to override.

        This list is dynamically updated based on discovered components.
        """
        base_classes = [
            # Core LLM classes
            "langchain.llms.base.BaseLLM",
            "langchain.chat_models.base.BaseChatModel",
            "langchain_core.language_models.llms.BaseLLM",
            "langchain_core.language_models.chat_models.BaseChatModel",
            # Provider-specific classes
            "langchain_openai.ChatOpenAI",
            "langchain_openai.OpenAI",
            "langchain_anthropic.ChatAnthropic",
            "langchain_google_genai.ChatGoogleGenerativeAI",
            "langchain_aws.ChatBedrock",
            "langchain_community.llms.huggingface_hub.HuggingFaceHub",
            # Chain classes
            "langchain.chains.base.Chain",
            "langchain.chains.llm.LLMChain",
            "langchain.chains.conversational_retrieval.ConversationalRetrievalChain",
            "langchain.chains.retrieval_qa.RetrievalQA",
            # Agent classes
            "langchain.agents.agent.BaseSingleActionAgent",
            "langchain.agents.agent.BaseMultiActionAgent",
            # Retriever classes
            "langchain.schema.retriever.BaseRetriever",
            "langchain.vectorstores.base.VectorStore",
            # Embedding classes
            "langchain.embeddings.base.Embeddings",
            "langchain_openai.OpenAIEmbeddings",
        ]

        # Add discovered classes
        base_classes.extend(list(self._discovered_classes.keys()))

        return base_classes

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of LangChain classes to methods to override."""
        methods = {
            # LLM methods
            "langchain_openai.ChatOpenAI": [
                "invoke",
                "ainvoke",
                "stream",
                "astream",
                "batch",
                "abatch",
            ],
            "langchain_openai.OpenAI": [
                "invoke",
                "ainvoke",
                "stream",
                "astream",
                "batch",
                "abatch",
            ],
            "langchain_anthropic.ChatAnthropic": [
                "invoke",
                "ainvoke",
                "stream",
                "astream",
                "batch",
                "abatch",
            ],
            # Chain methods
            "langchain.chains.llm.LLMChain": [
                "run",
                "arun",
                "__call__",
                "invoke",
                "ainvoke",
            ],
            "langchain.chains.conversational_retrieval.ConversationalRetrievalChain": [
                "run",
                "arun",
                "__call__",
                "invoke",
                "ainvoke",
            ],
            "langchain.chains.retrieval_qa.RetrievalQA": [
                "run",
                "arun",
                "__call__",
                "invoke",
                "ainvoke",
            ],
            # Agent methods
            "langchain.agents.agent.BaseSingleActionAgent": ["plan", "aplan"],
            "langchain.agents.agent.BaseMultiActionAgent": ["plan", "aplan"],
            # Retriever methods
            "langchain.schema.retriever.BaseRetriever": [
                "get_relevant_documents",
                "aget_relevant_documents",
            ],
            "langchain.vectorstores.base.VectorStore": [
                "similarity_search",
                "asimilarity_search",
                "similarity_search_with_score",
                "asimilarity_search_with_score",
                "max_marginal_relevance_search",
                "amax_marginal_relevance_search",
            ],
            # Embedding methods
            "langchain.embeddings.base.Embeddings": [
                "embed_documents",
                "aembed_documents",
                "embed_query",
                "aembed_query",
            ],
            "langchain_openai.OpenAIEmbeddings": [
                "embed_documents",
                "aembed_documents",
                "embed_query",
                "aembed_query",
            ],
        }

        # Add methods for discovered classes
        for class_name in self._discovered_classes:
            if class_name not in methods:
                # Default methods for discovered LLM classes
                if "LLM" in class_name or "Chat" in class_name:
                    methods[class_name] = [
                        "invoke",
                        "ainvoke",
                        "stream",
                        "astream",
                        "batch",
                        "abatch",
                    ]
                elif "Chain" in class_name:
                    methods[class_name] = [
                        "run",
                        "arun",
                        "__call__",
                        "invoke",
                        "ainvoke",
                    ]
                elif "Retriever" in class_name or "VectorStore" in class_name:
                    methods[class_name] = [
                        "get_relevant_documents",
                        "aget_relevant_documents",
                    ]

        return methods

    def _discover_langchain_components(self) -> None:
        """Dynamically discover LangChain components.

        This method attempts to discover installed LangChain packages and their
        LLM/Chain/Agent classes to ensure comprehensive coverage.
        """
        packages_to_scan = [
            ("langchain_openai", ["ChatOpenAI", "OpenAI", "OpenAIEmbeddings"]),
            ("langchain_anthropic", ["ChatAnthropic", "Anthropic"]),
            (
                "langchain_google_genai",
                ["ChatGoogleGenerativeAI", "GoogleGenerativeAI"],
            ),
            ("langchain_google_vertexai", ["ChatVertexAI", "VertexAI"]),
            ("langchain_aws", ["ChatBedrock", "Bedrock"]),
            ("langchain_cohere", ["ChatCohere", "Cohere"]),
            ("langchain_huggingface", ["HuggingFaceHub", "HuggingFaceEndpoint"]),
            ("langchain_together", ["Together"]),
            ("langchain_nvidia", ["ChatNVIDIA"]),
        ]

        for package_name, class_names in packages_to_scan:
            try:
                module = importlib.import_module(package_name)
                for class_name in class_names:
                    if hasattr(module, class_name):
                        full_class_name = f"{package_name}.{class_name}"
                        self._discovered_classes[full_class_name] = getattr(
                            module, class_name
                        )
                        logger.debug(
                            f"Discovered LangChain component: {full_class_name}"
                        )
            except ImportError:
                # Package not installed, skip
                continue
            except Exception as e:
                logger.debug(f"Error discovering {package_name}: {e}")

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply LangChain-specific overrides.

        This method extends the base implementation to handle LangChain-specific
        logic like chain construction and retriever configuration.
        """
        config_obj = self._normalize_config(config)

        # Apply base overrides
        overridden = super().apply_overrides(kwargs, config_obj)

        custom_params_raw = getattr(config_obj, "custom_params", {}) or {}
        if isinstance(custom_params_raw, Mapping):
            custom_params = dict(custom_params_raw)
        else:
            try:
                custom_params = dict(custom_params_raw)
            except Exception:
                custom_params = {}

        # Note: model→model_name and stream→streaming mappings are handled
        # declaratively by the plugin's mapping pipeline (ParameterNormalizer +
        # LLMPlugin._get_default_mappings). No manual handling needed here.

        # Handle retriever-specific parameters
        if "search_type" in custom_params:
            search_type = custom_params["search_type"]
            if (
                search_type == "similarity_score_threshold"
                and "score_threshold" in custom_params
            ):
                # Ensure score_threshold is set for this search type
                overridden["search_kwargs"] = overridden.get("search_kwargs", {})
                overridden["search_kwargs"]["score_threshold"] = custom_params[
                    "score_threshold"
                ]
            elif search_type == "mmr" and "lambda_mult" in custom_params:
                # Ensure lambda_mult is set for MMR
                overridden["search_kwargs"] = overridden.get("search_kwargs", {})
                overridden["search_kwargs"]["lambda_mult"] = custom_params[
                    "lambda_mult"
                ]

        # Handle k parameter for retrievers
        if "k" in custom_params:
            # Some retrievers use search_kwargs
            if any(
                key in str(type(kwargs.get("__self__", "")))
                for key in ["Retriever", "VectorStore"]
            ):
                overridden["search_kwargs"] = overridden.get("search_kwargs", {})
                overridden["search_kwargs"]["k"] = custom_params["k"]
            else:
                overridden["k"] = custom_params["k"]

        return overridden
