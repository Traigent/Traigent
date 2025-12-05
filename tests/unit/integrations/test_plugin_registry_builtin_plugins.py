"""Ensure plugin registry loads built-in integration plugins."""

from __future__ import annotations

import pytest

from traigent.integrations.plugin_registry import PluginRegistry


@pytest.fixture
def fresh_registry(monkeypatch) -> PluginRegistry:
    """Provide a fresh registry instance with built-ins discovered."""
    import traigent.integrations.plugin_registry as registry_module

    monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)
    registry = registry_module.PluginRegistry()
    yield registry

    # Reset singleton for other tests
    monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)


@pytest.mark.unit
def test_builtin_plugins_registered(fresh_registry: PluginRegistry) -> None:
    """Built-in LLM and vector store plugins should auto-register."""
    expected = {
        "bedrock",
        "azure_openai",
        "gemini",
        "chromadb",
        "pinecone",
        "weaviate",
    }

    registered = set(fresh_registry._plugins.keys())
    missing = expected - registered

    assert not missing, f"Missing built-in plugins: {sorted(missing)}"

    # Spot-check class mappings for vector stores to ensure overrides are discoverable
    chroma_plugin = fresh_registry.get_plugin_for_class("chromadb.Collection")
    assert chroma_plugin is not None
    assert chroma_plugin.metadata.name == "chromadb"

    weaviate_plugin = fresh_registry.get_plugin_for_class(
        "weaviate.collections.query._QueryGRPC"
    )
    assert weaviate_plugin is not None
    assert weaviate_plugin.metadata.name == "weaviate"
