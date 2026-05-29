"""Unit tests for traigent.integrations.plugin_registry.

Tests for plugin registry core functionality including registration,
lookup, priority handling, and lifecycle management.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility
# Traceability (cont): CONC-Quality-Maintainability FUNC-INTEGRATIONS
# Traceability (cont): FUNC-INVOKERS REQ-INT-008 REQ-INJ-002
# Traceability (cont): SYNC-IntegrationHook

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from traigent.integrations.base_plugin import (
    IntegrationPlugin,
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.plugin_registry import PluginRegistry, get_registry
from traigent.utils.exceptions import TraigentError


class _MinimalPlugin(IntegrationPlugin):
    """Minimal plugin for testing."""

    def __init__(self, metadata: PluginMetadata, config_path: Path | None = None):
        self._provided_metadata = metadata
        super().__init__(config_path=config_path)

    def _get_metadata(self) -> PluginMetadata:
        return self._provided_metadata

    def _get_default_mappings(self) -> dict[str, str]:
        return {"temperature": "temp", "max_tokens": "max_length"}

    def _get_validation_rules(self) -> dict[str, ValidationRule]:
        return {}

    def get_target_classes(self) -> list[str]:
        return ["test.TestClass"]

    def get_target_methods(self) -> dict[str, list[str]]:
        return {"test.TestClass": ["method1", "method2"]}


@pytest.fixture
def isolated_registry(monkeypatch) -> PluginRegistry:
    """Provide a fresh registry instance without built-in discovery."""
    import traigent.integrations.plugin_registry as registry_module

    # Prevent auto-discovery of built-in plugins
    monkeypatch.setattr(
        registry_module.PluginRegistry, "_discover_builtin_plugins", lambda self: None
    )
    # Reset singleton
    monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)

    registry = registry_module.PluginRegistry()
    yield registry

    # Clean up singleton for other tests
    monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)


@pytest.fixture
def sample_metadata() -> PluginMetadata:
    """Create sample plugin metadata."""
    return PluginMetadata(
        name="test_plugin",
        version="1.0.0",
        supported_packages=["test-package"],
        priority=IntegrationPriority.NORMAL,
        description="Test plugin",
    )


@pytest.fixture
def sample_plugin(sample_metadata: PluginMetadata) -> _MinimalPlugin:
    """Create a sample plugin for testing."""
    return _MinimalPlugin(sample_metadata)


def _allowed_plugins_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create the cwd-local plugin directory allowed by PluginRegistry."""
    monkeypatch.chdir(tmp_path)
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    return plugins_dir


class TestPluginRegistrySingleton:
    """Tests for singleton pattern behavior."""

    def test_singleton_returns_same_instance(self, monkeypatch) -> None:
        """Test that multiple instantiations return the same singleton instance."""
        import traigent.integrations.plugin_registry as registry_module

        # Reset singleton
        monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)

        registry1 = PluginRegistry()
        registry2 = PluginRegistry()

        assert registry1 is registry2

    def test_singleton_initialization_happens_once(self, monkeypatch) -> None:
        """Test that initialization code runs only once for singleton."""
        import traigent.integrations.plugin_registry as registry_module

        monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)
        monkeypatch.setattr(
            registry_module.PluginRegistry,
            "_discover_builtin_plugins",
            lambda self: None,
        )

        registry1 = PluginRegistry()
        initial_plugins = registry1._plugins.copy()

        # Create another reference - should not re-initialize
        registry2 = PluginRegistry()

        assert registry1 is registry2
        assert registry2._plugins == initial_plugins


class TestPluginRegistration:
    """Tests for plugin registration functionality."""

    def test_register_plugin_success(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test successful plugin registration."""
        isolated_registry.register(sample_plugin)

        assert "test_plugin" in isolated_registry._plugins
        assert isolated_registry.get_plugin("test_plugin") is sample_plugin

    def test_register_plugin_updates_class_mappings(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test that registration updates class to plugin mappings."""
        isolated_registry.register(sample_plugin)

        assert "test.TestClass" in isolated_registry._class_to_plugin
        assert isolated_registry._class_to_plugin["test.TestClass"] == "test_plugin"

    def test_register_plugin_updates_package_mappings(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test that registration updates package to plugin mappings."""
        isolated_registry.register(sample_plugin)

        assert "test-package" in isolated_registry._package_to_plugin
        assert "test_plugin" in isolated_registry._package_to_plugin["test-package"]

    def test_register_duplicate_name_raises_error(
        self, isolated_registry: PluginRegistry, sample_metadata: PluginMetadata
    ) -> None:
        """Test that registering duplicate plugin name raises error."""
        plugin1 = _MinimalPlugin(sample_metadata)
        plugin2 = _MinimalPlugin(sample_metadata)

        isolated_registry.register(plugin1)

        with pytest.raises(TraigentError, match="already registered"):
            isolated_registry.register(plugin2)

    def test_register_higher_priority_replaces_existing(
        self, isolated_registry: PluginRegistry, sample_metadata: PluginMetadata
    ) -> None:
        """Test that higher priority plugin replaces lower priority."""
        low_priority_meta = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            supported_packages=["test-package"],
            priority=IntegrationPriority.LOW,
        )
        high_priority_meta = PluginMetadata(
            name="test_plugin",
            version="2.0.0",
            supported_packages=["test-package"],
            priority=IntegrationPriority.HIGH,
        )

        low_plugin = _MinimalPlugin(low_priority_meta)
        high_plugin = _MinimalPlugin(high_priority_meta)

        isolated_registry.register(low_plugin)
        isolated_registry.register(high_plugin)

        registered = isolated_registry.get_plugin("test_plugin")
        assert registered is high_plugin
        assert registered.metadata.version == "2.0.0"

    def test_register_class_mapping_respects_priority(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that class mappings are updated based on plugin priority."""
        low_priority_meta = PluginMetadata(
            name="low_plugin",
            version="1.0.0",
            supported_packages=["pkg"],
            priority=IntegrationPriority.LOW,
        )
        high_priority_meta = PluginMetadata(
            name="high_plugin",
            version="1.0.0",
            supported_packages=["pkg"],
            priority=IntegrationPriority.HIGH,
        )

        low_plugin = _MinimalPlugin(low_priority_meta)
        high_plugin = _MinimalPlugin(high_priority_meta)

        # Register low priority first
        isolated_registry.register(low_plugin)
        assert isolated_registry._class_to_plugin["test.TestClass"] == "low_plugin"

        # Register high priority - should take over class mapping
        isolated_registry.register(high_plugin)
        assert isolated_registry._class_to_plugin["test.TestClass"] == "high_plugin"

    def test_register_multiple_packages_for_same_plugin(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test registering plugin with multiple supported packages."""
        metadata = PluginMetadata(
            name="multi_pkg_plugin",
            version="1.0.0",
            supported_packages=["pkg1", "pkg2", "pkg3"],
        )
        plugin = _MinimalPlugin(metadata)

        isolated_registry.register(plugin)

        assert "multi_pkg_plugin" in isolated_registry._package_to_plugin["pkg1"]
        assert "multi_pkg_plugin" in isolated_registry._package_to_plugin["pkg2"]
        assert "multi_pkg_plugin" in isolated_registry._package_to_plugin["pkg3"]


class TestPluginUnregistration:
    """Tests for plugin unregistration functionality."""

    def test_unregister_plugin_removes_from_registry(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test that unregister removes plugin from registry."""
        isolated_registry.register(sample_plugin)
        isolated_registry.unregister("test_plugin")

        assert "test_plugin" not in isolated_registry._plugins
        assert isolated_registry.get_plugin("test_plugin") is None

    def test_unregister_plugin_removes_class_mappings(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test that unregister removes class mappings."""
        isolated_registry.register(sample_plugin)
        isolated_registry.unregister("test_plugin")

        assert "test.TestClass" not in isolated_registry._class_to_plugin

    def test_unregister_plugin_removes_package_mappings(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test that unregister removes package mappings."""
        isolated_registry.register(sample_plugin)
        isolated_registry.unregister("test_plugin")

        assert (
            "test-package" not in isolated_registry._package_to_plugin
            or "test_plugin" not in isolated_registry._package_to_plugin["test-package"]
        )

    def test_unregister_nonexistent_plugin_does_nothing(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that unregistering non-existent plugin doesn't raise error."""
        # Should not raise - verify it completes and registry is unchanged
        plugins_before = len(isolated_registry._plugins)
        isolated_registry.unregister("nonexistent_plugin")
        assert len(isolated_registry._plugins) == plugins_before

    def test_unregister_cleans_empty_package_lists(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test that empty package lists are removed after unregistration."""
        isolated_registry.register(sample_plugin)
        isolated_registry.unregister("test_plugin")

        # Package key should be removed if no plugins left
        assert "test-package" not in isolated_registry._package_to_plugin


class TestPluginLookup:
    """Tests for plugin lookup methods."""

    def test_get_plugin_returns_registered_plugin(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test get_plugin returns the correct plugin."""
        isolated_registry.register(sample_plugin)

        result = isolated_registry.get_plugin("test_plugin")
        assert result is sample_plugin

    def test_get_plugin_returns_none_for_missing(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test get_plugin returns None for non-existent plugin."""
        result = isolated_registry.get_plugin("nonexistent")
        assert result is None

    def test_get_plugin_for_class_exact_match(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test get_plugin_for_class with exact class name match."""
        isolated_registry.register(sample_plugin)

        result = isolated_registry.get_plugin_for_class("test.TestClass")
        assert result is sample_plugin

    def test_get_plugin_for_class_partial_match(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test get_plugin_for_class with partial class name match."""
        isolated_registry.register(sample_plugin)

        # Should match via partial matching
        result = isolated_registry.get_plugin_for_class("test.TestClass.NestedClass")
        assert result is sample_plugin

    def test_get_plugin_for_class_returns_none_when_not_found(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test get_plugin_for_class returns None when no match found."""
        result = isolated_registry.get_plugin_for_class("unknown.Class")
        assert result is None

    def test_get_plugins_for_package_returns_all_matching(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test get_plugins_for_package returns all plugins supporting package."""
        meta1 = PluginMetadata(
            name="plugin1", version="1.0.0", supported_packages=["shared-pkg"]
        )
        meta2 = PluginMetadata(
            name="plugin2", version="1.0.0", supported_packages=["shared-pkg"]
        )
        plugin1 = _MinimalPlugin(meta1)
        plugin2 = _MinimalPlugin(meta2)

        isolated_registry.register(plugin1)
        isolated_registry.register(plugin2)

        result = isolated_registry.get_plugins_for_package("shared-pkg")
        assert len(result) == 2
        assert plugin1 in result
        assert plugin2 in result

    def test_get_plugins_for_package_returns_empty_for_unknown(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test get_plugins_for_package returns empty list for unknown package."""
        result = isolated_registry.get_plugins_for_package("unknown-pkg")
        assert result == []

    def test_list_plugins_returns_all_names(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test list_plugins returns all registered plugin names."""
        meta1 = PluginMetadata(
            name="plugin1", version="1.0.0", supported_packages=["pkg"]
        )
        meta2 = PluginMetadata(
            name="plugin2", version="1.0.0", supported_packages=["pkg"]
        )
        plugin1 = _MinimalPlugin(meta1)
        plugin2 = _MinimalPlugin(meta2)

        isolated_registry.register(plugin1)
        isolated_registry.register(plugin2)

        result = isolated_registry.list_plugins()
        assert len(result) == 2
        assert "plugin1" in result
        assert "plugin2" in result

    def test_get_all_plugins_returns_copy_of_plugins_dict(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test get_all_plugins returns a copy, not the internal dict."""
        isolated_registry.register(sample_plugin)

        result = isolated_registry.get_all_plugins()
        assert result == isolated_registry._plugins
        assert result is not isolated_registry._plugins  # Should be a copy


class TestPluginEnableDisable:
    """Tests for plugin enable/disable functionality."""

    def test_enable_plugin_calls_plugin_enable(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test enable_plugin calls the plugin's enable method."""
        isolated_registry.register(sample_plugin)
        sample_plugin._enabled = False

        isolated_registry.enable_plugin("test_plugin")
        assert sample_plugin.enabled is True

    def test_disable_plugin_calls_plugin_disable(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test disable_plugin calls the plugin's disable method."""
        isolated_registry.register(sample_plugin)

        isolated_registry.disable_plugin("test_plugin")
        assert sample_plugin.enabled is False

    def test_enable_nonexistent_plugin_does_nothing(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test enabling non-existent plugin doesn't raise error."""
        # Should not raise - verify it completes and registry is unchanged
        plugins_before = len(isolated_registry._plugins)
        isolated_registry.enable_plugin("nonexistent")
        assert len(isolated_registry._plugins) == plugins_before

    def test_disable_nonexistent_plugin_does_nothing(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test disabling non-existent plugin doesn't raise error."""
        # Should not raise - verify it completes and registry is unchanged
        plugins_before = len(isolated_registry._plugins)
        isolated_registry.disable_plugin("nonexistent")
        assert len(isolated_registry._plugins) == plugins_before


class TestPluginValidation:
    """Tests for plugin metadata validation."""

    def test_validate_rejects_missing_metadata(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that plugins without metadata are rejected."""

        class NoMetadataPlugin(_MinimalPlugin):
            def __init__(self) -> None:
                metadata = PluginMetadata(
                    name="test", version="1.0.0", supported_packages=["pkg"]
                )
                super().__init__(metadata)
                self.metadata = None  # type: ignore[assignment]

        plugin = NoMetadataPlugin()

        with pytest.raises(TraigentError, match="must define metadata"):
            isolated_registry.register(plugin)

    def test_validate_rejects_wrong_metadata_type(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that plugins with wrong metadata type are rejected."""

        class WrongMetadataPlugin(_MinimalPlugin):
            def __init__(self) -> None:
                metadata = PluginMetadata(
                    name="test", version="1.0.0", supported_packages=["pkg"]
                )
                super().__init__(metadata)
                self.metadata = {"name": "test"}  # type: ignore[assignment]

        plugin = WrongMetadataPlugin()

        with pytest.raises(TraigentError, match="must be PluginMetadata instance"):
            isolated_registry.register(plugin)

    def test_validate_rejects_empty_name(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that plugins with empty name are rejected."""
        metadata = PluginMetadata(
            name="   ", version="1.0.0", supported_packages=["pkg"]
        )
        plugin = _MinimalPlugin(metadata)

        with pytest.raises(TraigentError, match="non-empty string name"):
            isolated_registry.register(plugin)

    def test_validate_rejects_empty_version(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that plugins with empty version are rejected."""
        metadata = PluginMetadata(
            name="test", version="   ", supported_packages=["pkg"]
        )
        plugin = _MinimalPlugin(metadata)

        with pytest.raises(TraigentError, match="non-empty version string"):
            isolated_registry.register(plugin)

    def test_validate_rejects_invalid_supported_packages(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that plugins with invalid supported_packages are rejected."""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            supported_packages=[""],  # Empty string in list
        )
        plugin = _MinimalPlugin(metadata)

        with pytest.raises(TraigentError, match="supported_packages"):
            isolated_registry.register(plugin)

    def test_validate_trims_metadata_fields(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that metadata fields are trimmed during validation."""
        metadata = PluginMetadata(
            name="  test_plugin  ",
            version="  1.0.0  ",
            supported_packages=["  pkg1  ", "  pkg2  "],
        )
        plugin = _MinimalPlugin(metadata)

        isolated_registry.register(plugin)

        registered = isolated_registry.get_plugin("test_plugin")
        assert registered is not None
        assert registered.metadata.name == "test_plugin"
        assert registered.metadata.version == "1.0.0"
        assert registered.metadata.supported_packages == ["pkg1", "pkg2"]


class TestLoadPluginFromModule:
    """Tests for loading plugins from module paths."""

    @patch("traigent.integrations.plugin_registry.importlib.import_module")
    def test_load_plugin_from_module_success(
        self, mock_import: MagicMock, isolated_registry: PluginRegistry
    ) -> None:
        """Test successful plugin loading from module."""
        # Create metadata for the plugin
        metadata = PluginMetadata(
            name="loaded_plugin", version="1.0.0", supported_packages=["pkg"]
        )

        # Create an actual plugin class that can pass issubclass check
        class TestLoadedPlugin(_MinimalPlugin):
            def __init__(self, config_path: Path | None = None):
                super().__init__(metadata, config_path)

        # Mock module
        mock_module = Mock()
        mock_module.TestPlugin = TestLoadedPlugin
        mock_import.return_value = mock_module

        isolated_registry.load_plugin_from_module(
            "traigent_plugins.test_module", "TestPlugin"
        )

        mock_import.assert_called_once_with("traigent_plugins.test_module")
        assert "loaded_plugin" in isolated_registry._plugins

    @patch("traigent.integrations.plugin_registry.importlib.import_module")
    def test_load_plugin_from_module_with_config(
        self,
        mock_import: MagicMock,
        isolated_registry: PluginRegistry,
        tmp_path: Path,
    ) -> None:
        """Test loading plugin with config file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("test: config")

        # Create metadata for the plugin
        metadata = PluginMetadata(
            name="loaded_plugin", version="1.0.0", supported_packages=["pkg"]
        )

        # Create an actual plugin class that can pass issubclass check
        class TestLoadedPlugin(_MinimalPlugin):
            def __init__(self, config_path: Path | None = None):
                super().__init__(metadata, config_path)

        # Mock module
        mock_module = Mock()
        mock_module.TestPlugin = TestLoadedPlugin
        mock_import.return_value = mock_module

        isolated_registry.load_plugin_from_module(
            "traigent_plugins.test_module", "TestPlugin", config_path=config_path
        )

        mock_import.assert_called_once_with("traigent_plugins.test_module")
        # Verify plugin was registered with config
        assert "loaded_plugin" in isolated_registry._plugins

    @patch("traigent.integrations.plugin_registry.importlib.import_module")
    def test_load_plugin_from_module_import_error(
        self, mock_import: MagicMock, isolated_registry: PluginRegistry
    ) -> None:
        """Test that ImportError is handled when loading plugin."""
        mock_import.side_effect = ImportError("Module not found")

        with pytest.raises(TraigentError, match="Failed to import module"):
            isolated_registry.load_plugin_from_module(
                "traigent_plugins.missing", "TestPlugin"
            )

    @patch("traigent.integrations.plugin_registry.importlib.import_module")
    def test_load_plugin_from_module_attribute_error(
        self, mock_import: MagicMock, isolated_registry: PluginRegistry
    ) -> None:
        """Test that AttributeError is handled when class not found."""
        mock_module = Mock(spec=[])  # Empty module
        mock_import.return_value = mock_module

        with pytest.raises(TraigentError, match="Class .* not found"):
            isolated_registry.load_plugin_from_module(
                "traigent_plugins.test_module", "MissingClass"
            )

    @patch("traigent.integrations.plugin_registry.importlib.import_module")
    def test_load_plugin_from_module_not_subclass(
        self, mock_import: MagicMock, isolated_registry: PluginRegistry
    ) -> None:
        """Test that non-IntegrationPlugin classes are rejected."""
        mock_module = Mock()
        mock_module.NotAPlugin = str  # Not an IntegrationPlugin
        mock_import.return_value = mock_module

        with pytest.raises(TraigentError, match="not a valid IntegrationPlugin"):
            isolated_registry.load_plugin_from_module(
                "traigent_plugins.test_module", "NotAPlugin"
            )

    @patch("traigent.integrations.plugin_registry.importlib.import_module")
    def test_load_plugin_from_module_rejects_untrusted_module(
        self, mock_import: MagicMock, isolated_registry: PluginRegistry
    ) -> None:
        """Untrusted module paths must fail before import side effects can run."""
        with pytest.raises(TraigentError, match="not allowed"):
            isolated_registry.load_plugin_from_module("os", "TestPlugin")

        mock_import.assert_not_called()

    @patch("traigent.integrations.plugin_registry.importlib.import_module")
    def test_load_plugin_from_module_rejects_invalid_class_name(
        self, mock_import: MagicMock, isolated_registry: PluginRegistry
    ) -> None:
        """Invalid class names must fail before module import."""
        with pytest.raises(TraigentError, match="class name"):
            isolated_registry.load_plugin_from_module(
                "traigent_plugins.test_module", "not-a-class"
            )

        mock_import.assert_not_called()


class TestDiscoverPluginsInDirectory:
    """Tests for plugin discovery in directories."""

    def test_discover_plugins_in_directory_empty_dir(
        self,
        isolated_registry: PluginRegistry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test discovery in empty directory returns empty list."""
        empty_dir = _allowed_plugins_dir(tmp_path, monkeypatch)

        result = isolated_registry.discover_plugins_in_directory(empty_dir)
        assert result == []

    def test_discover_plugins_in_directory_nonexistent(
        self,
        isolated_registry: PluginRegistry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test discovery in non-existent directory returns empty list."""
        monkeypatch.chdir(tmp_path)
        nonexistent = tmp_path / "plugins" / "nonexistent"

        result = isolated_registry.discover_plugins_in_directory(nonexistent)
        assert result == []

    def test_discover_plugins_skips_private_files(
        self,
        isolated_registry: PluginRegistry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that files starting with underscore are skipped."""
        plugins_dir = _allowed_plugins_dir(tmp_path, monkeypatch)

        # Create private file
        private_file = plugins_dir / "_private.py"
        private_file.write_text("# Private file")

        result = isolated_registry.discover_plugins_in_directory(plugins_dir)
        assert result == []

    @patch(
        "traigent.integrations.plugin_registry.importlib.util.spec_from_file_location"
    )
    def test_discover_plugins_handles_import_errors(
        self,
        mock_spec_from_file: MagicMock,
        isolated_registry: PluginRegistry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that import errors during discovery are handled gracefully."""
        plugins_dir = _allowed_plugins_dir(tmp_path, monkeypatch)

        plugin_file = plugins_dir / "broken_plugin.py"
        plugin_file.write_text("# Broken plugin")

        mock_spec_from_file.return_value = None  # Simulate failure

        result = isolated_registry.discover_plugins_in_directory(plugins_dir)
        assert result == []

    def test_discover_plugins_loads_valid_plugin(
        self,
        isolated_registry: PluginRegistry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test successful plugin discovery from directory."""
        plugins_dir = _allowed_plugins_dir(tmp_path, monkeypatch)

        # Create a valid plugin file
        plugin_code = '''
from traigent.integrations.base_plugin import (
    IntegrationPlugin,
    PluginMetadata,
    ValidationRule,
)
from pathlib import Path

class DiscoveredPlugin(IntegrationPlugin):
    """Test plugin for discovery."""

    def _get_metadata(self):
        return PluginMetadata(
            name="discovered",
            version="1.0.0",
            supported_packages=["test-pkg"]
        )

    def _get_default_mappings(self):
        return {}

    def _get_validation_rules(self):
        return {}

    def get_target_classes(self):
        return ["test.Class"]

    def get_target_methods(self):
        return {"test.Class": ["method"]}
'''
        plugin_file = plugins_dir / "discovered_plugin.py"
        plugin_file.write_text(plugin_code)

        result = isolated_registry.discover_plugins_in_directory(plugins_dir)

        assert len(result) == 1
        assert "discovered" in result
        assert "discovered" in isolated_registry._plugins

    def test_discover_plugins_with_config_file(
        self,
        isolated_registry: PluginRegistry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test plugin discovery finds and uses config files."""
        plugins_dir = _allowed_plugins_dir(tmp_path, monkeypatch)

        # Create a config file
        config_file = plugins_dir / "discoveredplugin.yaml"
        config_file.write_text("mappings:\n  test: value\n")

        # Create a valid plugin file
        plugin_code = '''
from traigent.integrations.base_plugin import (
    IntegrationPlugin,
    PluginMetadata,
    ValidationRule,
)
from pathlib import Path

class DiscoveredPlugin(IntegrationPlugin):
    """Test plugin for discovery."""

    def _get_metadata(self):
        return PluginMetadata(
            name="discovered",
            version="1.0.0",
            supported_packages=["test-pkg"]
        )

    def _get_default_mappings(self):
        return {}

    def _get_validation_rules(self):
        return {}

    def get_target_classes(self):
        return []

    def get_target_methods(self):
        return {}
'''
        plugin_file = plugins_dir / "discovered_plugin.py"
        plugin_file.write_text(plugin_code)

        result = isolated_registry.discover_plugins_in_directory(plugins_dir)

        assert len(result) == 1
        assert "discovered" in result

    def test_discover_plugins_handles_exceptions(
        self,
        isolated_registry: PluginRegistry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that exceptions during plugin loading are logged."""
        plugins_dir = _allowed_plugins_dir(tmp_path, monkeypatch)

        # Create a plugin file that will raise an exception
        plugin_code = """
raise RuntimeError("Test error during module load")
"""
        plugin_file = plugins_dir / "error_plugin.py"
        plugin_file.write_text(plugin_code)

        # Should not raise, just log error
        result = isolated_registry.discover_plugins_in_directory(plugins_dir)
        assert result == []

    def test_discover_plugins_rejects_untrusted_directory(
        self, isolated_registry: PluginRegistry, tmp_path: Path
    ) -> None:
        """Plugin discovery must not execute Python from arbitrary directories."""
        untrusted_dir = tmp_path / "not_plugins"
        untrusted_dir.mkdir()

        with pytest.raises(TraigentError, match="allowed plugin root"):
            isolated_registry.discover_plugins_in_directory(untrusted_dir)

    def test_discover_plugins_skips_symlink_escape(
        self,
        isolated_registry: PluginRegistry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Symlinked plugin files must not escape the trusted plugin directory."""
        plugins_dir = _allowed_plugins_dir(tmp_path, monkeypatch)
        outside_file = tmp_path / "outside_plugin.py"
        outside_file.write_text('raise RuntimeError("should not load")\n')
        link = plugins_dir / "outside_plugin.py"
        try:
            link.symlink_to(outside_file)
        except OSError:
            pytest.skip("symlink creation is not supported on this filesystem")

        result = isolated_registry.discover_plugins_in_directory(plugins_dir)
        assert result == []


class TestConfigReload:
    """Tests for config reloading functionality."""

    def test_reload_configs_calls_load_config_on_plugins(
        self, isolated_registry: PluginRegistry, tmp_path: Path
    ) -> None:
        """Test that reload_configs calls _load_config_overrides on plugins."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("test: config")

        metadata = PluginMetadata(
            name="test_plugin", version="1.0.0", supported_packages=["pkg"]
        )
        plugin = _MinimalPlugin(metadata, config_path=config_path)
        plugin._load_config_overrides = Mock()  # type: ignore[method-assign]

        isolated_registry.register(plugin)
        isolated_registry.reload_configs()

        plugin._load_config_overrides.assert_called_once_with(config_path)

    def test_reload_configs_skips_plugins_without_config(
        self, isolated_registry: PluginRegistry, sample_plugin: _MinimalPlugin
    ) -> None:
        """Test that reload_configs skips plugins without config files."""
        sample_plugin._load_config_overrides = Mock()  # type: ignore[method-assign]

        isolated_registry.register(sample_plugin)
        isolated_registry.reload_configs()

        # Should not be called since no config path
        sample_plugin._load_config_overrides.assert_not_called()

    def test_reload_configs_skips_plugins_with_missing_config(
        self, isolated_registry: PluginRegistry, tmp_path: Path
    ) -> None:
        """Test that reload_configs skips plugins with non-existent config."""
        nonexistent_config = tmp_path / "missing.yaml"

        metadata = PluginMetadata(
            name="test_plugin", version="1.0.0", supported_packages=["pkg"]
        )
        plugin = _MinimalPlugin(metadata, config_path=nonexistent_config)
        plugin._load_config_overrides = Mock()  # type: ignore[method-assign]

        isolated_registry.register(plugin)
        isolated_registry.reload_configs()

        # Should not be called since config doesn't exist
        plugin._load_config_overrides.assert_not_called()


class TestGetRegistry:
    """Tests for global registry accessor."""

    def test_get_registry_returns_singleton(self) -> None:
        """Test that get_registry returns the singleton instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2
        assert isinstance(registry1, PluginRegistry)


class TestBuiltinPluginDiscovery:
    """Tests for built-in plugin discovery."""

    @patch("traigent.integrations.plugin_registry.importlib.import_module")
    def test_discover_builtin_plugins_handles_import_errors(
        self, mock_import: MagicMock, monkeypatch
    ) -> None:
        """Test that ImportError during built-in discovery is logged and skipped."""
        import traigent.integrations.plugin_registry as registry_module

        monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)
        mock_import.side_effect = ImportError("Package not installed")

        # Should not raise
        registry = PluginRegistry()
        assert isinstance(registry, PluginRegistry)

    @patch("traigent.integrations.plugin_registry.importlib.import_module")
    def test_discover_builtin_plugins_handles_general_errors(
        self, mock_import: MagicMock, monkeypatch
    ) -> None:
        """Test that general errors during built-in discovery are logged."""
        import traigent.integrations.plugin_registry as registry_module

        monkeypatch.setattr(registry_module.PluginRegistry, "_instance", None)
        mock_import.side_effect = Exception("Unexpected error")

        # Should not raise
        registry = PluginRegistry()
        assert isinstance(registry, PluginRegistry)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_register_plugin_with_empty_target_classes(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test registering plugin with no target classes."""
        metadata = PluginMetadata(
            name="no_classes", version="1.0.0", supported_packages=["pkg"]
        )

        class NoClassesPlugin(_MinimalPlugin):
            def get_target_classes(self) -> list[str]:
                return []

        plugin = NoClassesPlugin(metadata)
        isolated_registry.register(plugin)

        assert "no_classes" in isolated_registry._plugins

    def test_get_plugin_for_class_with_containing_match(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test partial matching when class name contains registered class."""
        metadata = PluginMetadata(
            name="test_plugin", version="1.0.0", supported_packages=["pkg"]
        )
        plugin = _MinimalPlugin(metadata)
        isolated_registry.register(plugin)

        # Search for a class that contains the registered class name
        result = isolated_registry.get_plugin_for_class("prefix.test.TestClass.suffix")
        assert result is plugin

    def test_unregister_preserves_other_plugins_in_package_mapping(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that unregistering one plugin doesn't affect others in package mapping."""
        meta1 = PluginMetadata(
            name="plugin1", version="1.0.0", supported_packages=["shared"]
        )
        meta2 = PluginMetadata(
            name="plugin2", version="1.0.0", supported_packages=["shared"]
        )
        plugin1 = _MinimalPlugin(meta1)
        plugin2 = _MinimalPlugin(meta2)

        isolated_registry.register(plugin1)
        isolated_registry.register(plugin2)
        isolated_registry.unregister("plugin1")

        # plugin2 should still be mapped to "shared" package
        assert "plugin2" in isolated_registry._package_to_plugin["shared"]
        assert "plugin1" not in isolated_registry._package_to_plugin["shared"]

    def test_register_plugin_with_duplicate_packages(
        self, isolated_registry: PluginRegistry
    ) -> None:
        """Test that duplicate packages in metadata don't cause issues."""
        metadata = PluginMetadata(
            name="dup_pkg", version="1.0.0", supported_packages=["pkg", "pkg", "pkg"]
        )
        plugin = _MinimalPlugin(metadata)

        isolated_registry.register(plugin)

        # Should only appear once in the mapping
        assert isolated_registry._package_to_plugin["pkg"].count("dup_pkg") == 1
