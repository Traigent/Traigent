"""Base class for vector store integration plugins.

This module provides the base class for vector store plugins, defining common
parameters and validation rules for retrieval operations.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from abc import abstractmethod

from traigent.integrations.base_plugin import (
    IntegrationPlugin,
    ValidationRule,
)


class VectorStorePlugin(IntegrationPlugin):
    """Abstract base class for vector store integration plugins."""

    def _get_default_mappings(self) -> dict[str, str]:
        """Return default parameter mappings for vector stores.

        Subclasses should extend this with provider-specific mappings.

        Note: search_type is intentionally excluded as it's a LangChain abstraction,
        not a native parameter for vector store APIs (Chroma, Pinecone, Weaviate).
        For search type selection, use provider-specific methods or the LangChain plugin.
        """
        return {
            "k": "k",  # Number of results
            "score_threshold": "score_threshold",
            "filter": "filter",
        }

    def _get_validation_rules(self) -> dict[str, ValidationRule]:
        """Return common validation rules for vector stores."""
        return {
            "k": ValidationRule(min_value=1, max_value=1000),
            "score_threshold": ValidationRule(min_value=0.0, max_value=1.0),
        }

    @abstractmethod
    def get_target_classes(self) -> list[str]:
        """Return list of vector store classes to override."""
        raise NotImplementedError

    @abstractmethod
    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override."""
        raise NotImplementedError
