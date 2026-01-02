"""Weaviate integration plugin for Traigent.

This module provides the Weaviate-specific plugin implementation for
parameter mappings and framework overrides.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from typing import TYPE_CHECKING

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
)
from traigent.integrations.vector_stores.base import VectorStorePlugin

if TYPE_CHECKING:
    pass


class WeaviatePlugin(VectorStorePlugin):
    """Plugin for Weaviate integration."""

    def _get_metadata(self) -> PluginMetadata:
        """Return Weaviate plugin metadata."""
        return PluginMetadata(
            name="weaviate",
            version="1.0.0",
            supported_packages=["weaviate-client"],
            priority=IntegrationPriority.NORMAL,
            description="Weaviate vector store integration",
            author="Traigent Team",
            requires_packages=["weaviate-client>=4.0.0"],
            supports_versions={"weaviate-client": "4."},
        )

    def _get_default_mappings(self) -> dict[str, str]:
        """Return default parameter mappings for Weaviate."""
        mappings = super()._get_default_mappings()
        mappings.update(
            {
                "k": "limit",
                "score_threshold": "certainty",  # Or distance, depending on metric
                "filter": "filters",
            }
        )
        return mappings

    def get_target_classes(self) -> list[str]:
        """Return list of Weaviate classes to override.

        Weaviate v4 uses a fluent API where collection.query returns a Query object.
        We target the Query class methods for vector search operations.
        """
        return [
            "weaviate.collections.query._QueryGRPC",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override.

        Weaviate v4 Query object has various search methods:
        - near_text: semantic text search
        - near_vector: vector similarity search
        - bm25: keyword search
        - hybrid: hybrid semantic + keyword search
        - fetch_objects: retrieve objects by filter
        """
        return {
            "weaviate.collections.query._QueryGRPC": [
                "near_text",
                "near_vector",
                "bm25",
                "hybrid",
                "fetch_objects",
            ],
        }
