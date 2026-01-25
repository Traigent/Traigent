"""Unit tests for plugin system."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from traigent.plugins.registry import (
    EvaluatorPlugin,
    IntegrationPlugin,
    MetricPlugin,
    OptimizerPlugin,
    PluginRegistry,
    TraigentPlugin,
    discover_plugins,
    get_plugin_registry,
    load_plugin,
    register_plugin,
)


class MockOptimizerPlugin(OptimizerPlugin):
    """Mock optimizer plugin for testing."""

    @property
    def name(self) -> str:
        return "mock_optimizer"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Mock optimizer for testing"

    @property
    def dependencies(self) -> list:
        return ["numpy"]

    def initialize(self) -> None:
        pass

    def create_optimizer(self, **kwargs):
        return Mock()


class MockEvaluatorPlugin(EvaluatorPlugin):
    """Mock evaluator plugin for testing."""

    @property
    def name(self) -> str:
        return "mock_evaluator"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Mock evaluator for testing"

    def initialize(self) -> None:
        pass

    def create_evaluator(self, **kwargs):
        return Mock()


class MockMetricPlugin(MetricPlugin):
    """Mock metric plugin for testing."""

    @property
    def name(self) -> str:
        return "mock_metric"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Mock metric for testing"

    def initialize(self) -> None:
        pass

    def calculate_metric(self, actual, expected, context=None) -> float:
        return 0.95


class MockIntegrationPlugin(IntegrationPlugin):
    """Mock integration plugin for testing."""

    @property
    def name(self) -> str:
        return "mock_integration"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Mock integration for testing"

    def initialize(self) -> None:
        pass

    def get_integration_functions(self):
        return {"test_function": lambda x: x}


class TestPluginRegistry:
    """Test plugin registry functionality."""

    def setup_method(self):
        """Set up test registry."""
        self.registry = PluginRegistry()

    def test_register_optimizer_plugin(self):
        """Test registering optimizer plugin."""
        plugin = MockOptimizerPlugin()

        with patch("importlib.import_module"):
            self.registry.register_plugin(plugin)

        assert "mock_optimizer" in self.registry._plugins
        assert "mock_optimizer" in self.registry._optimizers

        # Test retrieval
        retrieved_plugin = self.registry.get_plugin("mock_optimizer")
        assert retrieved_plugin == plugin

    def test_register_evaluator_plugin(self):
        """Test registering evaluator plugin."""
        plugin = MockEvaluatorPlugin()

        with patch("importlib.import_module"):
            self.registry.register_plugin(plugin)

        assert "mock_evaluator" in self.registry._plugins
        assert "mock_evaluator" in self.registry._evaluators

    def test_register_metric_plugin(self):
        """Test registering metric plugin."""
        plugin = MockMetricPlugin()

        with patch("importlib.import_module"):
            self.registry.register_plugin(plugin)

        assert "mock_metric" in self.registry._plugins
        assert "mock_metric" in self.registry._metrics

    def test_register_integration_plugin(self):
        """Test registering integration plugin."""
        plugin = MockIntegrationPlugin()

        with patch("importlib.import_module"):
            self.registry.register_plugin(plugin)

        assert "mock_integration" in self.registry._plugins
        assert "mock_integration" in self.registry._integrations

    def test_unregister_plugin(self):
        """Test unregistering plugin."""
        plugin = MockOptimizerPlugin()

        with patch("importlib.import_module"):
            self.registry.register_plugin(plugin)

        assert "mock_optimizer" in self.registry._plugins

        self.registry.unregister_plugin("mock_optimizer")

        assert "mock_optimizer" not in self.registry._plugins
        assert "mock_optimizer" not in self.registry._optimizers

    def test_list_plugins(self):
        """Test listing plugins."""
        plugin1 = MockOptimizerPlugin()
        plugin2 = MockEvaluatorPlugin()

        with patch("importlib.import_module"):
            self.registry.register_plugin(plugin1)
            self.registry.register_plugin(plugin2)

        plugins_info = self.registry.list_plugins()

        assert len(plugins_info) == 2
        plugin_names = [p["name"] for p in plugins_info]
        assert "mock_optimizer" in plugin_names
        assert "mock_evaluator" in plugin_names

    def test_dependency_check_failure(self):
        """Test plugin registration with missing dependencies."""
        plugin = MockOptimizerPlugin()

        with patch(
            "importlib.import_module",
            side_effect=ImportError("No module named 'numpy'"),
        ):
            with pytest.raises(ImportError, match="missing dependencies"):
                self.registry.register_plugin(plugin)

    def test_load_plugin_from_module(self):
        """Test loading plugin from module."""
        mock_module = Mock()
        mock_module.MockOptimizerPlugin = MockOptimizerPlugin

        with patch("importlib.import_module", return_value=mock_module):
            with patch("importlib.import_module"):  # For dependency check
                self.registry.load_plugin_from_module(
                    "traigent_plugins.test.module", "MockOptimizerPlugin"
                )

        assert len(self.registry._plugins) == 1

    def test_auto_discover_plugins(self):
        """Test automatic plugin discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock plugin file
            plugin_file = Path(temp_dir) / "test_plugin.py"
            plugin_file.write_text("""
from traigent.plugins.registry import OptimizerPlugin

class TestOptimizer(OptimizerPlugin):
    @property
    def name(self):
        return "test_optimizer"

    @property
    def version(self):
        return "1.0.0"

    @property
    def description(self):
        return "Test optimizer"

    def initialize(self):
        pass

    def create_optimizer(self, **kwargs):
        return None
""")

            # Mock Path.exists() to return True for plugin directories
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.__truediv__ = lambda self, other: mock_path  # For path joining

            with patch("pathlib.Path.home", return_value=mock_path):
                with patch("pathlib.Path.cwd", return_value=mock_path):
                    with patch.object(
                        self.registry, "load_plugins_from_directory"
                    ) as mock_load:
                        self.registry.auto_discover_plugins()
                        # Should try to load from standard directories
                        assert mock_load.call_count >= 1


class TestGlobalPluginFunctions:
    """Test global plugin functions."""

    def test_get_plugin_registry(self):
        """Test getting global registry."""
        registry = get_plugin_registry()
        assert isinstance(registry, PluginRegistry)

    def test_register_plugin_global(self):
        """Test global plugin registration."""
        plugin = MockOptimizerPlugin()

        with patch("importlib.import_module"):
            register_plugin(plugin)

        registry = get_plugin_registry()
        assert "mock_optimizer" in registry._plugins

        # Clean up
        registry.unregister_plugin("mock_optimizer")

    def test_discover_plugins_global(self):
        """Test global plugin discovery."""
        with patch.object(
            get_plugin_registry(), "auto_discover_plugins"
        ) as mock_discover:
            discover_plugins()
            mock_discover.assert_called_once()

    def test_load_plugin_global(self):
        """Test global plugin loading."""
        with patch.object(
            get_plugin_registry(), "load_plugin_from_module"
        ) as mock_load:
            load_plugin("traigent_plugins.test.module", "TestPlugin")
            mock_load.assert_called_once_with(
                "traigent_plugins.test.module", "TestPlugin"
            )


class TestPluginInheritance:
    """Test plugin inheritance and abstract methods."""

    def test_traigent_plugin_abstract(self):
        """Test that TraigentPlugin is properly abstract."""
        with pytest.raises(TypeError):
            TraigentPlugin()

    def test_optimizer_plugin_abstract(self):
        """Test that OptimizerPlugin requires implementation."""

        class IncompleteOptimizer(OptimizerPlugin):
            @property
            def name(self):
                return "incomplete"

            @property
            def version(self):
                return "1.0.0"

            @property
            def description(self):
                return "incomplete"

            def initialize(self):
                pass

            # Missing create_optimizer method

        with pytest.raises(TypeError):
            IncompleteOptimizer()

    def test_evaluator_plugin_abstract(self):
        """Test that EvaluatorPlugin requires implementation."""

        class IncompleteEvaluator(EvaluatorPlugin):
            @property
            def name(self):
                return "incomplete"

            @property
            def version(self):
                return "1.0.0"

            @property
            def description(self):
                return "incomplete"

            def initialize(self):
                pass

            # Missing create_evaluator method

        with pytest.raises(TypeError):
            IncompleteEvaluator()

    def test_metric_plugin_abstract(self):
        """Test that MetricPlugin requires implementation."""

        class IncompleteMetric(MetricPlugin):
            @property
            def name(self):
                return "incomplete"

            @property
            def version(self):
                return "1.0.0"

            @property
            def description(self):
                return "incomplete"

            def initialize(self):
                pass

            # Missing calculate_metric method

        with pytest.raises(TypeError):
            IncompleteMetric()

    def test_integration_plugin_abstract(self):
        """Test that IntegrationPlugin requires implementation."""

        class IncompleteIntegration(IntegrationPlugin):
            @property
            def name(self):
                return "incomplete"

            @property
            def version(self):
                return "1.0.0"

            @property
            def description(self):
                return "incomplete"

            def initialize(self):
                pass

            # Missing get_integration_functions method

        with pytest.raises(TypeError):
            IncompleteIntegration()


class TestPluginNaming:
    """Test plugin naming conventions."""

    def test_optimizer_naming(self):
        """Test optimizer plugin naming."""
        plugin = MockOptimizerPlugin()
        assert plugin.optimizer_name == "mock_optimizer"

    def test_evaluator_naming(self):
        """Test evaluator plugin naming."""
        plugin = MockEvaluatorPlugin()
        assert plugin.evaluator_name == "mock_evaluator"

    def test_metric_naming(self):
        """Test metric plugin naming."""
        plugin = MockMetricPlugin()
        assert plugin.metric_name == "mock_metric"

    def test_integration_naming(self):
        """Test integration plugin naming."""
        plugin = MockIntegrationPlugin()
        assert plugin.integration_name == "mock_integration"

    def test_name_with_spaces(self):
        """Test name conversion with spaces."""

        class SpacedNamePlugin(OptimizerPlugin):
            @property
            def name(self):
                return "My Spaced Plugin"

            @property
            def version(self):
                return "1.0.0"

            @property
            def description(self):
                return "test"

            def initialize(self):
                pass

            def create_optimizer(self, **kwargs):
                return None

        plugin = SpacedNamePlugin()
        assert plugin.optimizer_name == "my_spaced_plugin"
