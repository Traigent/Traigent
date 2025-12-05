"""Tests for integration plugin registry safeguards."""

from __future__ import annotations

import pytest

from traigent.integrations.base_plugin import (
    IntegrationPlugin,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.plugin_registry import PluginRegistry
from traigent.utils.exceptions import TraigentError


class _StubPlugin(IntegrationPlugin):
    """Minimal IntegrationPlugin stub for registry tests."""

    def __init__(self, metadata: PluginMetadata):
        self._provided_metadata = metadata
        super().__init__()

    def _get_metadata(self) -> PluginMetadata:
        return self._provided_metadata

    def _get_default_mappings(self) -> dict[str, str]:
        return {}

    def _get_validation_rules(self) -> dict[str, ValidationRule]:
        return {}

    def get_target_classes(self) -> list[str]:
        return []

    def get_target_methods(self) -> dict[str, list[str]]:
        return {}


@pytest.fixture
def isolated_registry(monkeypatch) -> PluginRegistry:
    """Provide a fresh registry instance without discovering builtins."""
    import traigent.integrations.plugin_registry as registry_module

    monkeypatch.setattr(
        registry_module.PluginRegistry, "_discover_builtin_plugins", lambda self: None
    )
    monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)

    registry = registry_module.PluginRegistry()
    yield registry

    # Ensure singleton is reset for other tests
    monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)


def test_register_rejects_missing_metadata(isolated_registry: PluginRegistry) -> None:
    """Registry should reject plugins that unset metadata."""

    class MissingMetadataPlugin(_StubPlugin):
        def __init__(self) -> None:
            super().__init__(
                PluginMetadata(
                    name="missing_metadata",
                    version="1.0.0",
                    supported_packages=["pkg"],
                )
            )
            self.metadata = None  # type: ignore[assignment]

    with pytest.raises(TraigentError):
        isolated_registry.register(MissingMetadataPlugin())


def test_register_rejects_blank_name(isolated_registry: PluginRegistry) -> None:
    """Registry should reject blank plugin names."""
    plugin = _StubPlugin(
        PluginMetadata(name="   ", version="1.0.0", supported_packages=["pkg"])
    )

    with pytest.raises(TraigentError):
        isolated_registry.register(plugin)


def test_register_normalizes_metadata_fields(isolated_registry: PluginRegistry) -> None:
    """Registry should trim metadata fields before registration."""
    plugin = _StubPlugin(
        PluginMetadata(
            name="  DemoPlugin ",
            version=" 0.1.0 ",
            supported_packages=[" pkg-one ", "pkg-two"],
        )
    )

    isolated_registry.register(plugin)

    registered = isolated_registry.get_plugin("DemoPlugin")
    assert registered is plugin
    assert registered.metadata.name == "DemoPlugin"
    assert registered.metadata.version == "0.1.0"
    assert registered.metadata.supported_packages == ["pkg-one", "pkg-two"]
