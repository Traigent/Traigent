"""Pinecone integration plugin for TraiGent.

This module provides the Pinecone-specific plugin implementation for
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


class PineconePlugin(VectorStorePlugin):
    """Plugin for Pinecone integration."""

    def _get_metadata(self) -> PluginMetadata:
        """Return Pinecone plugin metadata."""
        return PluginMetadata(
            name="pinecone",
            version="1.0.0",
            supported_packages=["pinecone", "pinecone-client"],
            priority=IntegrationPriority.NORMAL,
            description="Pinecone vector store integration",
            author="TraiGent Team",
            requires_packages=["pinecone-client>=3.0.0"],
            supports_versions={"pinecone-client": "3."},
        )

    def _get_default_mappings(self) -> dict[str, str]:
        """Return default parameter mappings for Pinecone."""
        mappings = super()._get_default_mappings()
        mappings.update(
            {
                "k": "top_k",
                "filter": "filter",
                "include_values": "include_values",
                "include_metadata": "include_metadata",
                "namespace": "namespace",  # Pinecone namespace for query isolation
            }
        )
        return mappings

    def get_target_classes(self) -> list[str]:
        """Return list of Pinecone classes to override."""
        return [
            "pinecone.Index",
            "pinecone.grpc.IndexGRPC",
            "pinecone.data.index.Index",  # v3 structure
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override."""
        return {
            "pinecone.Index": ["query"],
            "pinecone.grpc.IndexGRPC": ["query"],
            "pinecone.data.index.Index": ["query"],
        }
