#!/usr/bin/env python3
"""Scaffold a new LLM integration plugin for Traigent.

This script generates all the boilerplate code needed to add a new LLM provider
to Traigent, including:
- Plugin implementation file
- Test file
- Model discovery implementation (optional)
- Documentation updates

Usage:
    python scripts/scaffold_llm_plugin.py <provider_name>

Example:
    python scripts/scaffold_llm_plugin.py groq

This will create:
- traigent/integrations/llms/groq_plugin.py
- tests/unit/integrations/test_groq_plugin.py
- traigent/integrations/model_discovery/groq_discovery.py (optional)
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in snake_str.split("_"))


def to_snake_case(name: str) -> str:
    """Convert any case to snake_case."""
    # Insert underscores before uppercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscores before uppercase letters that follow lowercase
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def validate_provider_name(name: str) -> bool:
    """Validate that provider name is a valid Python identifier."""
    if not name:
        logger.error("Provider name cannot be empty")
        return False
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        logger.error(
            "Provider name must be lowercase alphanumeric with underscores, "
            f"starting with a letter. Got: {name}"
        )
        return False
    return True


def generate_plugin_file(provider_name: str, provider_sdk: str) -> str:
    """Generate the plugin implementation file content."""
    class_name = to_pascal_case(provider_name)
    framework_enum = provider_name.upper()

    return f'''"""
{class_name} integration plugin for Traigent.

This module provides the {class_name}-specific plugin implementation for
parameter mappings and framework overrides.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import logging
from typing import TYPE_CHECKING, Any

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.llms.base_llm_plugin import LLMPlugin
from traigent.integrations.utils import Framework

if TYPE_CHECKING:
    from traigent.config.types import TraigentConfig

logger = logging.getLogger(__name__)


class {class_name}Plugin(LLMPlugin):
    """Plugin for {class_name} SDK integration.

    Supports the official {class_name} Python SDK.
    Handles parameter mapping for chat completions including streaming,
    tool use, and provider-specific parameters.

    Examples:
        >>> from traigent.integrations.llms.{provider_name}_plugin import {class_name}Plugin
        >>> plugin = {class_name}Plugin()
        >>> config = {{"model": "{provider_name}-model", "temperature": 0.7}}
        >>> overridden = plugin.apply_overrides({{}}, config)
    """

    FRAMEWORK = Framework.{framework_enum}

    def _get_metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns:
            PluginMetadata with provider information.
        """
        return PluginMetadata(
            name="{provider_name}",
            version="1.0.0",
            supported_packages=["{provider_sdk}"],
            priority=IntegrationPriority.HIGH,
            description="{class_name} SDK integration",
            author="Traigent Contributors",
            requires_packages=["{provider_sdk}>=1.0.0"],
            supports_versions={{"{provider_sdk}": "1."}},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return provider-specific parameter mappings.

        These mappings handle parameters unique to {class_name}
        that aren't in the standard ParameterNormalizer.

        Returns:
            Dictionary mapping canonical parameter names to {class_name} SDK names.

        Examples:
            >>> plugin = {class_name}Plugin()
            >>> mappings = plugin._get_extra_mappings()
            >>> # Add provider-specific mappings like:
            >>> # {{"custom_param": "sdk_param_name"}}
        """
        return {{
            # TODO: Add provider-specific parameter mappings
            # Example:
            # "seed": "random_seed",
            # "{provider_name}_api_key": "api_key",
        }}

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return provider-specific validation rules.

        Define valid ranges, allowed values, and required parameters.

        Returns:
            Dictionary mapping parameter names to validation rules.

        Examples:
            >>> plugin = {class_name}Plugin()
            >>> rules = plugin._get_provider_specific_rules()
            >>> assert "model" in rules
        """
        return {{
            "model": ValidationRule(
                required=True,
                custom_validator="_validate_model",
            ),
            "temperature": ValidationRule(min_value=0.0, max_value=2.0),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
            "max_tokens": ValidationRule(min_value=1, max_value=100000),
            "stream": ValidationRule(allowed_values=[True, False]),
            # TODO: Add provider-specific validation rules
        }}

    def _validate_model(self, param_name: str, value: Any) -> list[str]:
        """Validate model ID using dynamic discovery.

        Args:
            param_name: Name of the parameter being validated.
            value: Value to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        if not isinstance(value, str):
            errors.append(f"Parameter '{{param_name}}' must be a string")
            return errors

        if not value:
            errors.append(f"Parameter '{{param_name}}' cannot be empty")
            return errors

        try:
            from traigent.integrations.model_discovery import get_model_discovery

            discovery = get_model_discovery(self.FRAMEWORK)
            if discovery and not discovery.is_valid_model(value):
                # Warn but don't block - model might be new
                logger.warning(f"Unrecognized model: {{value}}. Proceeding anyway.")
        except ImportError:
            logger.debug("Model discovery not available")

        return errors

    def get_target_classes(self) -> list[str]:
        """Return list of SDK classes to override.

        Returns:
            List of fully qualified class names to instrument.

        Examples:
            >>> plugin = {class_name}Plugin()
            >>> classes = plugin.get_target_classes()
            >>> assert "{provider_sdk}.Client" in classes
        """
        return [
            "{provider_sdk}.Client",
            "{provider_sdk}.AsyncClient",
            # TODO: Add more SDK classes if needed
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override.

        Returns:
            Dictionary mapping class names to lists of method names.

        Examples:
            >>> plugin = {class_name}Plugin()
            >>> methods = plugin.get_target_methods()
            >>> assert "chat.completions.create" in methods["{provider_sdk}.Client"]
        """
        return {{
            "{provider_sdk}.Client": [
                "chat.completions.create",
                # TODO: Add more methods if needed
                # "completions.create",
            ],
            "{provider_sdk}.AsyncClient": [
                "chat.completions.create",
                # TODO: Add more methods if needed
            ],
        }}

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply provider-specific overrides.

        Extend base implementation for provider-specific logic like
        message formatting or special parameter handling.

        Args:
            kwargs: Original keyword arguments passed to SDK method.
            config: Traigent configuration or dictionary.

        Returns:
            Dictionary with overridden parameters.

        Examples:
            >>> plugin = {class_name}Plugin()
            >>> config = {{"model": "test-model", "temperature": 0.7}}
            >>> result = plugin.apply_overrides({{}}, config)
            >>> assert result["model"] == "test-model"
        """
        config_obj = self._normalize_config(config)
        overridden = super().apply_overrides(kwargs, config_obj)

        # TODO: Add provider-specific override logic here
        # Example: Message formatting, special parameter handling

        return overridden
'''


def generate_test_file(provider_name: str) -> str:
    """Generate the test file content."""
    class_name = to_pascal_case(provider_name)
    framework_enum = provider_name.upper()

    return f'''"""Tests for the {class_name} integration plugin."""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import pytest

from traigent.integrations.llms.{provider_name}_plugin import {class_name}Plugin
from traigent.integrations.utils import Framework


class Test{class_name}Plugin:
    """Basic plugin behavior tests."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.plugin = {class_name}Plugin()

    def test_framework_is_correct(self) -> None:
        """Test plugin identifies correct framework."""
        assert self.plugin.FRAMEWORK == Framework.{framework_enum}

    def test_metadata_name(self) -> None:
        """Test plugin metadata has correct name."""
        assert self.plugin.metadata.name == "{provider_name}"

    def test_apply_overrides_with_dict_config(self) -> None:
        """Plugin should handle raw dict payloads."""
        config_payload = {{
            "model": "test-model",
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 1000,
        }}

        overridden = self.plugin.apply_overrides({{}}, config_payload)

        assert overridden["model"] == "test-model"
        assert overridden["stream"] is True
        assert overridden["temperature"] == pytest.approx(0.7)
        assert overridden["max_tokens"] == 1000


class Test{class_name}ParameterMappings:
    """Test parameter mapping via ParameterNormalizer."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.plugin = {class_name}Plugin()

    def test_model_preserved(self) -> None:
        """Test that model parameter is preserved."""
        config = {{"model": "test-model"}}
        overridden = self.plugin.apply_overrides({{}}, config)
        assert overridden["model"] == "test-model"

    def test_temperature_preserved(self) -> None:
        """Test that temperature is preserved."""
        config = {{"model": "test-model", "temperature": 0.5}}
        overridden = self.plugin.apply_overrides({{}}, config)
        assert overridden["temperature"] == pytest.approx(0.5)

    def test_max_tokens_preserved(self) -> None:
        """Test that max_tokens is preserved."""
        config = {{"model": "test-model", "max_tokens": 500}}
        overridden = self.plugin.apply_overrides({{}}, config)
        assert overridden["max_tokens"] == 500

    def test_user_kwarg_not_overwritten(self) -> None:
        """Test that user-provided kwargs take precedence."""
        kwargs = {{"model": "user-specified-model"}}
        config = {{"model": "config-model"}}
        overridden = self.plugin.apply_overrides(kwargs, config)
        assert overridden["model"] == "user-specified-model"


class Test{class_name}ValidationRules:
    """Test validation rules."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.plugin = {class_name}Plugin()

    def test_temperature_validation_range(self) -> None:
        """Test temperature has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "temperature" in rules
        assert rules["temperature"].min_value == 0.0
        assert rules["temperature"].max_value == 2.0

    def test_top_p_validation_range(self) -> None:
        """Test top_p has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "top_p" in rules
        assert rules["top_p"].min_value == 0.0
        assert rules["top_p"].max_value == 1.0

    def test_max_tokens_validation_minimum(self) -> None:
        """Test max_tokens has minimum value."""
        rules = self.plugin._get_provider_specific_rules()
        assert "max_tokens" in rules
        assert rules["max_tokens"].min_value == 1


class Test{class_name}PluginMetadata:
    """Test plugin metadata."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.plugin = {class_name}Plugin()

    def test_supported_packages(self) -> None:
        """Test plugin lists supported packages."""
        packages = self.plugin.metadata.supported_packages
        assert len(packages) > 0

    def test_metadata_version(self) -> None:
        """Test plugin has version."""
        assert self.plugin.metadata.version is not None

    def test_metadata_description(self) -> None:
        """Test plugin has description."""
        assert self.plugin.metadata.description is not None
        assert "{class_name}" in self.plugin.metadata.description


# TODO: Add integration tests with real SDK if needed
# @pytest.mark.integration
# class Test{class_name}Integration:
#     """Integration tests with real {class_name} SDK."""
#     pass
'''


def generate_model_discovery_file(provider_name: str, provider_sdk: str) -> str:
    """Generate the model discovery implementation file content."""
    class_name = to_pascal_case(provider_name)
    framework_enum = provider_name.upper()
    env_var = f"{provider_name.upper()}_API_KEY"

    return f'''"""Model discovery for {class_name}."""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import logging
import os
import re

from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Pattern for validating model names
{provider_name.upper()}_MODEL_PATTERN = r"^{provider_name}-"


class {class_name}Discovery(ModelDiscovery):
    """Model discovery service for {class_name}.

    Fetches available models from the {class_name} API and validates
    model names against known patterns.

    Examples:
        >>> from traigent.integrations.model_discovery.{provider_name}_discovery import {class_name}Discovery
        >>> discovery = {class_name}Discovery()
        >>> models = discovery.get_available_models()
    """

    PROVIDER = "{provider_name}"
    FRAMEWORK = Framework.{framework_enum}

    def _fetch_models_from_sdk(self) -> list[str]:
        """Fetch models directly from SDK if API key available.

        Returns:
            List of available model IDs, or empty list if fetch fails.
        """
        api_key = os.getenv("{env_var}")
        if not api_key:
            logger.debug("No API key found for {class_name}, skipping model fetch")
            return []

        try:
            # TODO: Import and use the actual SDK client
            # from {provider_sdk} import Client
            # client = Client(api_key=api_key)
            # models_response = client.models.list()
            # return sorted([m.id for m in models_response.data])
            logger.warning(
                "Model fetching not implemented yet for {class_name}. "
                "Update _fetch_models_from_sdk() in "
                "traigent/integrations/model_discovery/{provider_name}_discovery.py"
            )
            return []
        except Exception as e:
            logger.debug(f"Failed to fetch {class_name} models: {{e}}")
            return []

    def _get_config_key(self) -> str:
        """Return config key for loading known models.

        Returns:
            Key to use in models.yaml config file.
        """
        return "{provider_name}"

    def _get_model_pattern(self) -> re.Pattern[str]:
        """Return regex pattern for model validation.

        Returns:
            Compiled regex pattern for validating model names.
        """
        return re.compile({provider_name.upper()}_MODEL_PATTERN)
'''


def update_framework_enum(provider_name: str, repo_root: Path) -> bool:
    """Add the new framework to the Framework enum."""
    normalizer_path = (
        repo_root / "traigent" / "integrations" / "utils" / "parameter_normalizer.py"
    )

    if not normalizer_path.exists():
        logger.error(f"Could not find {normalizer_path}")
        return False

    content = normalizer_path.read_text()
    framework_enum = provider_name.upper()

    # Check if already exists
    if f'{framework_enum} = "{provider_name}"' in content:
        logger.info(f"Framework.{framework_enum} already exists in Framework enum")
        return True

    # Find the Framework enum and add the new value
    enum_pattern = r"(class Framework\(Enum\):.*?)(    [A-Z_]+ = .*?\n)"
    match = re.search(enum_pattern, content, re.DOTALL)

    if not match:
        logger.error("Could not find Framework enum in parameter_normalizer.py")
        return False

    # Insert before the last enum value (or at the end)
    insertion_point = match.end(2)
    new_line = f'    {framework_enum} = "{provider_name}"\n'

    new_content = content[:insertion_point] + new_line + content[insertion_point:]
    normalizer_path.write_text(new_content)

    logger.info(f"Added Framework.{framework_enum} to Framework enum")
    return True


def update_plugin_init(provider_name: str, repo_root: Path) -> bool:
    """Add the new plugin to __init__.py exports."""
    init_path = repo_root / "traigent" / "integrations" / "llms" / "__init__.py"

    if not init_path.exists():
        logger.error(f"Could not find {init_path}")
        return False

    content = init_path.read_text()
    class_name = to_pascal_case(provider_name)

    # Check if already exists
    if f"from traigent.integrations.llms.{provider_name}_plugin import" in content:
        logger.info(f"{class_name}Plugin already exported in __init__.py")
        return True

    # Add import
    import_line = (
        f"from traigent.integrations.llms.{provider_name}_plugin "
        f"import {class_name}Plugin\n"
    )

    # Find the last import statement
    import_pattern = r"(from traigent\.integrations\.llms\..+? import .+?\n)"
    matches = list(re.finditer(import_pattern, content))

    if matches:
        # Insert after the last import
        last_match = matches[-1]
        insertion_point = last_match.end()
        new_content = (
            content[:insertion_point] + import_line + content[insertion_point:]
        )
    else:
        # Insert at the beginning
        new_content = import_line + content

    # Add to __all__
    if "__all__" in content:
        all_pattern = r"(__all__ = \[)(.*?)(\])"
        match = re.search(all_pattern, content, re.DOTALL)
        if match:
            export_line = f'    "{class_name}Plugin",\n'
            new_content = re.sub(
                all_pattern,
                r"\1\2" + export_line + r"\3",
                new_content,
                flags=re.DOTALL,
            )

    init_path.write_text(new_content)
    logger.info(f"Added {class_name}Plugin to __init__.py exports")
    return True


def scaffold_llm_plugin(
    provider_name: str,
    provider_sdk: str | None = None,
    include_model_discovery: bool = False,
) -> bool:
    """Scaffold a new LLM plugin with all necessary files.

    Args:
        provider_name: Name of the provider (e.g., 'groq', 'perplexity').
        provider_sdk: Name of the provider's SDK package (defaults to provider_name).
        include_model_discovery: Whether to generate model discovery implementation.

    Returns:
        True if successful, False otherwise.
    """
    if not validate_provider_name(provider_name):
        return False

    if provider_sdk is None:
        provider_sdk = provider_name

    # Find repository root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    logger.info(f"Scaffolding {provider_name} plugin...")

    # Generate plugin file
    plugin_path = (
        repo_root / "traigent" / "integrations" / "llms" / f"{provider_name}_plugin.py"
    )
    if plugin_path.exists():
        logger.warning(f"Plugin file already exists: {plugin_path}")
        overwrite = input("Overwrite? (y/N): ").lower().strip()
        if overwrite != "y":
            logger.info("Skipping plugin file generation")
        else:
            plugin_content = generate_plugin_file(provider_name, provider_sdk)
            plugin_path.write_text(plugin_content)
            logger.info(f"Created plugin file: {plugin_path}")
    else:
        plugin_content = generate_plugin_file(provider_name, provider_sdk)
        plugin_path.write_text(plugin_content)
        logger.info(f"Created plugin file: {plugin_path}")

    # Generate test file
    test_path = (
        repo_root
        / "tests"
        / "unit"
        / "integrations"
        / f"test_{provider_name}_plugin.py"
    )
    if test_path.exists():
        logger.warning(f"Test file already exists: {test_path}")
        overwrite = input("Overwrite? (y/N): ").lower().strip()
        if overwrite != "y":
            logger.info("Skipping test file generation")
        else:
            test_content = generate_test_file(provider_name)
            test_path.write_text(test_content)
            logger.info(f"Created test file: {test_path}")
    else:
        test_content = generate_test_file(provider_name)
        test_path.write_text(test_content)
        logger.info(f"Created test file: {test_path}")

    # Generate model discovery file (optional)
    if include_model_discovery:
        discovery_path = (
            repo_root
            / "traigent"
            / "integrations"
            / "model_discovery"
            / f"{provider_name}_discovery.py"
        )
        if discovery_path.exists():
            logger.warning(f"Model discovery file already exists: {discovery_path}")
            overwrite = input("Overwrite? (y/N): ").lower().strip()
            if overwrite != "y":
                logger.info("Skipping model discovery file generation")
            else:
                discovery_content = generate_model_discovery_file(
                    provider_name, provider_sdk
                )
                discovery_path.write_text(discovery_content)
                logger.info(f"Created model discovery file: {discovery_path}")
        else:
            discovery_content = generate_model_discovery_file(
                provider_name, provider_sdk
            )
            discovery_path.write_text(discovery_content)
            logger.info(f"Created model discovery file: {discovery_path}")

    # Update Framework enum
    if not update_framework_enum(provider_name, repo_root):
        logger.error("Failed to update Framework enum")
        return False

    # Update __init__.py
    if not update_plugin_init(provider_name, repo_root):
        logger.error("Failed to update __init__.py")
        return False

    logger.info("\n" + "=" * 70)
    logger.info("SUCCESS! Plugin scaffolding complete.")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info(f"1. Review and customize the generated plugin: {plugin_path}")
    logger.info("2. Update the TODO comments in the plugin implementation")
    logger.info(f"3. Run tests: TRAIGENT_MOCK_LLM=true pytest {test_path} -v")
    logger.info("4. Format code: make format")
    logger.info("5. Lint code: make lint")
    logger.info(
        "6. Add known models to traigent/config/models.yaml (if using model discovery)"
    )
    logger.info(
        "7. See docs/contributing/ADDING_NEW_INTEGRATIONS.md for detailed guidance"
    )
    logger.info("=" * 70)

    return True


def main() -> int:
    """Main entry point for the scaffold script."""
    parser = argparse.ArgumentParser(
        description="Scaffold a new LLM integration plugin for Traigent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scaffold a plugin for Groq
  python scripts/scaffold_llm_plugin.py groq

  # Scaffold with custom SDK name
  python scripts/scaffold_llm_plugin.py perplexity --sdk perplexity-python

  # Scaffold with model discovery
  python scripts/scaffold_llm_plugin.py groq --model-discovery

For detailed documentation, see:
  docs/contributing/ADDING_NEW_INTEGRATIONS.md
        """,
    )

    parser.add_argument(
        "provider_name",
        help="Name of the LLM provider (e.g., groq, perplexity, together)",
    )

    parser.add_argument(
        "--sdk",
        help="Name of the provider's SDK package (defaults to provider_name)",
        default=None,
    )

    parser.add_argument(
        "--model-discovery",
        action="store_true",
        help="Generate model discovery implementation",
    )

    args = parser.parse_args()

    provider_name = to_snake_case(args.provider_name)

    success = scaffold_llm_plugin(
        provider_name=provider_name,
        provider_sdk=args.sdk,
        include_model_discovery=args.model_discovery,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
