import pytest

from traigent.integrations.llms.cohere_plugin import CoherePlugin
from traigent.integrations.llms.huggingface_plugin import HuggingFacePlugin
from traigent.integrations.plugin_registry import PluginRegistry


@pytest.fixture
def fresh_registry(monkeypatch) -> PluginRegistry:
    """Provide a fresh registry instance isolated from singleton state."""
    import traigent.integrations.plugin_registry as registry_module

    monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)
    registry = registry_module.PluginRegistry()
    yield registry
    monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)


@pytest.mark.unit
def test_registry_discovers_new_plugins(fresh_registry: PluginRegistry) -> None:
    """Built-in discovery should include newly added plugins."""
    # Auto-discovery occurs in __init__
    assert "cohere" in fresh_registry._plugins
    assert "huggingface" in fresh_registry._plugins

    assert isinstance(fresh_registry.get_plugin("cohere"), CoherePlugin)
    assert isinstance(fresh_registry.get_plugin("huggingface"), HuggingFacePlugin)


@pytest.mark.unit
def test_registry_get_plugin_keys(fresh_registry: PluginRegistry) -> None:
    """Registry should list newly added plugins with lowercase keys."""
    plugin_names = fresh_registry.list_plugins()
    assert "cohere" in plugin_names
    assert "huggingface" in plugin_names
