"""Plugin registry and discovery system for Traigent integrations.

This module provides the central registry for managing integration plugins,
including discovery, registration, and lifecycle management.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility CONC-Quality-Maintainability FUNC-INTEGRATIONS FUNC-INVOKERS REQ-INT-008 REQ-INJ-002 SYNC-IntegrationHook

import importlib
import importlib.util
import inspect
import keyword
import logging
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Optional

from traigent.utils.exceptions import TraigentError

if TYPE_CHECKING:
    from traigent.integrations.base_plugin import IntegrationPlugin, PluginMetadata

logger = logging.getLogger(__name__)

_ALLOWED_PLUGIN_MODULE_PREFIXES = (
    "traigent.integrations",
    "traigent_plugins",
    "custom_plugins",
)
_UNSAFE_PLUGIN_MODULE_PARTS = {
    "__builtin__",
    "__import__",
    "builtins",
    "compile",
    "eval",
    "exec",
    "file",
    "importlib",
    "input",
    "open",
    "os",
    "raw_input",
    "subprocess",
    "sys",
}


def _is_valid_module_identifier(value: str) -> bool:
    return value.isidentifier() and not keyword.iskeyword(value)


def _is_allowed_plugin_module_path(module_path: str) -> bool:
    parts = module_path.split(".")
    if not parts or not all(_is_valid_module_identifier(part) for part in parts):
        return False
    if any(part.lower() in _UNSAFE_PLUGIN_MODULE_PARTS for part in parts):
        return False
    return any(
        module_path == prefix or module_path.startswith(f"{prefix}.")
        for prefix in _ALLOWED_PLUGIN_MODULE_PREFIXES
    )


class PluginRegistry:
    """Central registry for managing integration plugins.

    This class provides a singleton registry for discovering, registering,
    and managing integration plugins throughout the application lifecycle.
    """

    _instance = None
    _lock = RLock()
    _initialized: bool

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        if self._initialized:
            return

        self._plugins: dict[str, IntegrationPlugin] = {}
        self._class_to_plugin: dict[str, str] = {}  # Maps class names to plugin names
        self._package_to_plugin: dict[str, list[str]] = (
            {}
        )  # Maps package names to plugin names
        self._config_dir = Path.home() / ".traigent" / "plugins"
        self._initialized = True

        # Auto-discover and register built-in plugins
        self._discover_builtin_plugins()

    def _discover_builtin_plugins(self) -> None:
        """Discover and register built-in plugins."""
        builtin_plugins = [
            ("traigent.integrations.llms.openai_plugin", "OpenAIPlugin"),
            ("traigent.integrations.llms.langchain_plugin", "LangChainPlugin"),
            ("traigent.integrations.llms.anthropic_plugin", "AnthropicPlugin"),
            ("traigent.integrations.llms.llamaindex_plugin", "LlamaIndexPlugin"),
            ("traigent.integrations.llms.bedrock_plugin", "BedrockPlugin"),
            ("traigent.integrations.llms.azure_openai_plugin", "AzureOpenAIPlugin"),
            ("traigent.integrations.llms.gemini_plugin", "GeminiPlugin"),
            ("traigent.integrations.llms.cohere_plugin", "CoherePlugin"),
            ("traigent.integrations.llms.huggingface_plugin", "HuggingFacePlugin"),
            ("traigent.integrations.llms.litellm_plugin", "LiteLLMPlugin"),
            ("traigent.integrations.pydantic_ai.plugin", "PydanticAIPlugin"),
            ("traigent.integrations.vector_stores.chromadb_plugin", "ChromaDBPlugin"),
            ("traigent.integrations.vector_stores.pinecone_plugin", "PineconePlugin"),
            ("traigent.integrations.vector_stores.weaviate_plugin", "WeaviatePlugin"),
        ]

        for module_path, class_name in builtin_plugins:
            try:
                module = importlib.import_module(module_path)
                plugin_class = getattr(module, class_name)

                # Check for config file
                plugin_name = class_name.replace("Plugin", "").lower()
                config_candidate = self._config_dir / f"{plugin_name}.yaml"
                config_path: Path | None = (
                    config_candidate if config_candidate.exists() else None
                )

                # Instantiate and register
                plugin = plugin_class(config_path=config_path)
                self.register(plugin)
                logger.info(f"Registered built-in plugin: {plugin.metadata.name}")

            except ImportError as e:
                logger.debug(f"Plugin {class_name} not available: {e}")
            except Exception as e:
                logger.error(f"Failed to load plugin {class_name}: {e}")

    def register(self, plugin: "IntegrationPlugin") -> None:
        """Register a plugin with the registry.

        Args:
            plugin: The plugin instance to register

        Raises:
            TraigentError: If a plugin with the same name is already registered
        """
        metadata = self._validate_plugin_metadata(plugin)
        name = metadata.name

        if name in self._plugins:
            existing = self._plugins[name]
            if existing.metadata.priority.value < plugin.metadata.priority.value:
                logger.info(f"Replacing plugin {name} with higher priority version")
            else:
                raise TraigentError(f"Plugin '{name}' is already registered") from None

        self._plugins[name] = plugin

        # Update class mappings
        for class_name in plugin.get_target_classes():
            if class_name not in self._class_to_plugin:
                self._class_to_plugin[class_name] = name
            else:
                # Check priority
                existing_plugin = self._plugins[self._class_to_plugin[class_name]]
                if (
                    plugin.metadata.priority.value
                    > existing_plugin.metadata.priority.value
                ):
                    self._class_to_plugin[class_name] = name

        # Update package mappings
        for package in plugin.metadata.supported_packages:
            if package not in self._package_to_plugin:
                self._package_to_plugin[package] = []
            if name not in self._package_to_plugin[package]:
                self._package_to_plugin[package].append(name)

    def _validate_plugin_metadata(
        self, plugin: "IntegrationPlugin"
    ) -> "PluginMetadata":
        """Validate plugin metadata before registration."""
        metadata = getattr(plugin, "metadata", None)
        if metadata is None:
            raise TraigentError(
                f"Plugin {plugin.__class__.__name__} must define metadata before registration"
            )

        # Import lazily to avoid circular import during module import
        from traigent.integrations.base_plugin import PluginMetadata

        if not isinstance(metadata, PluginMetadata):
            raise TraigentError(
                f"Plugin {plugin.__class__.__name__} metadata must be PluginMetadata instance"
            )

        name = getattr(metadata, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise TraigentError(
                f"Plugin {plugin.__class__.__name__} metadata must provide a non-empty string name"
            )
        metadata.name = name.strip()

        version = getattr(metadata, "version", None)
        if not isinstance(version, str) or not version.strip():
            raise TraigentError(
                f"Plugin {metadata.name} metadata must provide a non-empty version string"
            )
        metadata.version = version.strip()

        supported_packages = getattr(metadata, "supported_packages", None)
        if not isinstance(supported_packages, list) or not all(
            isinstance(pkg, str) and pkg.strip() for pkg in supported_packages
        ):
            raise TraigentError(
                f"Plugin {metadata.name} metadata must declare supported_packages as a list of non-empty strings"
            )
        metadata.supported_packages = [pkg.strip() for pkg in supported_packages]

        return metadata

    def unregister(self, name: str) -> None:
        """Unregister a plugin from the registry.

        Args:
            name: The name of the plugin to unregister
        """
        if name not in self._plugins:
            return

        plugin = self._plugins[name]

        # Remove class mappings
        for class_name in plugin.get_target_classes():
            if self._class_to_plugin.get(class_name) == name:
                del self._class_to_plugin[class_name]

        # Remove package mappings
        for package in plugin.metadata.supported_packages:
            if package in self._package_to_plugin:
                if name in self._package_to_plugin[package]:
                    self._package_to_plugin[package].remove(name)
                if not self._package_to_plugin[package]:
                    del self._package_to_plugin[package]

        # Remove plugin
        del self._plugins[name]
        logger.info(f"Unregistered plugin: {name}")

    def get_plugin(self, name: str) -> Optional["IntegrationPlugin"]:
        """Get a plugin by name.

        Args:
            name: The name of the plugin

        Returns:
            The plugin instance or None if not found
        """
        return self._plugins.get(name)

    def get_plugin_for_class(self, class_name: str) -> Optional["IntegrationPlugin"]:
        """Get the plugin responsible for a given class.

        Args:
            class_name: Fully qualified class name

        Returns:
            The plugin instance or None if no plugin handles this class
        """
        plugin_name = self._class_to_plugin.get(class_name)
        if plugin_name:
            return self._plugins.get(plugin_name)

        # Try partial matching for nested classes
        for registered_class, plugin_name in self._class_to_plugin.items():
            if (
                class_name.startswith(registered_class)
                or registered_class in class_name
            ):
                return self._plugins.get(plugin_name)

        return None

    def get_plugins_for_package(self, package_name: str) -> list["IntegrationPlugin"]:
        """Get all plugins that support a given package.

        Args:
            package_name: Name of the package

        Returns:
            List of plugin instances that support this package
        """
        plugin_names = self._package_to_plugin.get(package_name, [])
        return [self._plugins[name] for name in plugin_names if name in self._plugins]

    def list_plugins(self) -> list[str]:
        """List all registered plugin names.

        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())

    def get_all_plugins(self) -> dict[str, "IntegrationPlugin"]:
        """Get all registered plugins.

        Returns:
            Dictionary of plugin name to plugin instance
        """
        return self._plugins.copy()

    def enable_plugin(self, name: str) -> None:
        """Enable a plugin.

        Args:
            name: The name of the plugin to enable
        """
        plugin = self.get_plugin(name)
        if plugin:
            plugin.enable()
            logger.info(f"Enabled plugin: {name}")

    def disable_plugin(self, name: str) -> None:
        """Disable a plugin.

        Args:
            name: The name of the plugin to disable
        """
        plugin = self.get_plugin(name)
        if plugin:
            plugin.disable()
            logger.info(f"Disabled plugin: {name}")

    def load_plugin_from_module(
        self, module_path: str, class_name: str, config_path: Path | None = None
    ) -> None:
        """Load and register a plugin from a module.

        Args:
            module_path: Dotted path to the module
            class_name: Name of the plugin class
            config_path: Optional path to configuration file
        """
        if not _is_allowed_plugin_module_path(module_path):
            raise TraigentError(f"Plugin module path '{module_path}' is not allowed")
        if not _is_valid_module_identifier(class_name):
            raise TraigentError(f"Plugin class name '{class_name}' is invalid")

        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)

            # Import IntegrationPlugin at runtime to avoid circular import
            from traigent.integrations.base_plugin import IntegrationPlugin

            if not issubclass(plugin_class, IntegrationPlugin):
                raise TraigentError(
                    f"{class_name} is not a valid IntegrationPlugin subclass"
                )

            plugin = plugin_class(config_path=config_path)
            self.register(plugin)
            logger.info(f"Loaded plugin from module: {module_path}.{class_name}")

        except ImportError as e:
            raise TraigentError(f"Failed to import module {module_path}: {e}") from None
        except AttributeError as e:
            raise TraigentError(
                f"Class {class_name} not found in {module_path}: {e}"
            ) from None
        except Exception as e:
            raise TraigentError(f"Failed to load plugin {class_name}: {e}") from None

    def _allowed_plugin_directories(self) -> tuple[Path, ...]:
        """Return roots trusted for filesystem plugin discovery."""
        return (
            self._config_dir,
            Path.cwd() / "traigent_plugins",
            Path.cwd() / "plugins",
            Path(__file__).parent / "contrib",
        )

    def _resolve_trusted_plugin_directory(self, directory: Path) -> Path:
        """Resolve and verify that a plugin directory stays inside a trusted root."""
        resolved = directory.expanduser().resolve()
        for allowed_dir in self._allowed_plugin_directories():
            trusted_root = allowed_dir.expanduser().resolve()
            try:
                resolved.relative_to(trusted_root)
                return resolved
            except (ValueError, RuntimeError):
                continue

        allowed = ", ".join(str(path) for path in self._allowed_plugin_directories())
        raise TraigentError(
            f"Plugin directory '{directory}' is not under an allowed plugin root: {allowed}"
        )

    @staticmethod
    def _resolve_plugin_file(py_file: Path, directory: Path) -> Path | None:
        """Return a plugin file only when it does not escape the trusted directory."""
        try:
            resolved_file = py_file.resolve(strict=True)
            resolved_file.relative_to(directory)
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "Skipping plugin file outside trusted directory: %s", py_file
            )
            return None
        return resolved_file

    @staticmethod
    def _safe_config_path(directory: Path, class_name: str) -> Path | None:
        """Return an adjacent config file only if it remains inside the plugin root."""
        candidate = directory / f"{class_name.lower()}.yaml"
        if not candidate.exists():
            return None
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(directory)
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "Ignoring plugin config outside trusted directory: %s", candidate
            )
            return None
        return resolved

    def discover_plugins_in_directory(self, directory: Path) -> list[str]:
        """Discover and load plugins from a directory.

        Args:
            directory: Path to directory containing plugin modules

        Returns:
            List of loaded plugin names
        """
        loaded: list[str] = []
        trusted_directory = self._resolve_trusted_plugin_directory(directory)

        if not trusted_directory.exists():
            return loaded
        if not trusted_directory.is_dir():
            raise TraigentError(f"Plugin path '{directory}' is not a directory")

        for py_file in sorted(trusted_directory.glob("*.py")):
            if py_file.stem.startswith("_"):
                continue

            resolved_plugin_file = self._resolve_plugin_file(py_file, trusted_directory)
            if resolved_plugin_file is None:
                continue

            module_name = resolved_plugin_file.stem
            spec = importlib.util.spec_from_file_location(
                module_name, resolved_plugin_file
            )
            if spec and spec.loader:
                try:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Import IntegrationPlugin at runtime to avoid circular import
                    from traigent.integrations.base_plugin import IntegrationPlugin

                    # Find IntegrationPlugin subclasses
                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, IntegrationPlugin)
                            and obj != IntegrationPlugin
                        ):
                            # Check for config file
                            config_path = self._safe_config_path(
                                trusted_directory, name
                            )

                            plugin = obj(config_path=config_path)
                            self.register(plugin)
                            loaded.append(plugin.metadata.name)
                            logger.info(f"Discovered plugin: {plugin.metadata.name}")

                except Exception as e:
                    logger.error(f"Failed to load plugin from {py_file}: {e}")

        return loaded

    def reload_configs(self) -> None:
        """Reload configuration files for all plugins."""
        for plugin in self._plugins.values():
            if plugin._config_path and plugin._config_path.exists():
                plugin._load_config_overrides(plugin._config_path)
                logger.info(f"Reloaded config for plugin: {plugin.metadata.name}")


# Global registry instance
registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry instance.

    Returns:
        The singleton PluginRegistry instance
    """
    return registry
