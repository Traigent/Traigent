"""ChromaDB integration plugin for TraiGent.

This module provides the ChromaDB-specific plugin implementation for
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


class ChromaDBPlugin(VectorStorePlugin):
    """Plugin for ChromaDB integration."""

    def _get_metadata(self) -> PluginMetadata:
        """Return ChromaDB plugin metadata."""
        return PluginMetadata(
            name="chromadb",
            version="1.0.0",
            supported_packages=["chromadb"],
            priority=IntegrationPriority.NORMAL,
            description="ChromaDB vector store integration",
            author="TraiGent Team",
            requires_packages=["chromadb>=0.4.0"],
            supports_versions={"chromadb": "0."},
        )

    def _get_default_mappings(self) -> dict[str, str]:
        """Return default parameter mappings for ChromaDB."""
        mappings = super()._get_default_mappings()
        mappings.update(
            {
                "k": "n_results",
                "filter": "where",
                "include": "include",
            }
        )
        return mappings

    def get_target_classes(self) -> list[str]:
        """Return list of ChromaDB classes to override."""
        return [
            "chromadb.api.models.Collection.Collection",
            "chromadb.Collection",  # Alias often used
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override."""
        return {
            "chromadb.api.models.Collection.Collection": ["query", "get"],
            "chromadb.Collection": ["query", "get"],
        }
