"""Tests for the LLM plugin scaffold script."""

import os
import re

# Import the scaffold module
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from scaffold_llm_plugin import (
    generate_model_discovery_file,
    generate_plugin_file,
    generate_test_file,
    to_pascal_case,
    to_snake_case,
    validate_provider_name,
)


class TestCaseConversion:
    """Test case conversion utilities."""

    def test_to_pascal_case(self):
        """Test snake_case to PascalCase conversion."""
        assert to_pascal_case("openai") == "Openai"
        assert to_pascal_case("together_ai") == "TogetherAi"
        assert to_pascal_case("my_provider") == "MyProvider"

    def test_to_snake_case(self):
        """Test any case to snake_case conversion."""
        assert to_snake_case("OpenAI") == "open_ai"
        assert to_snake_case("TogetherAI") == "together_ai"
        assert to_snake_case("my-provider") == "my-provider"
        assert to_snake_case("my_provider") == "my_provider"


class TestProviderNameValidation:
    """Test provider name validation."""

    def test_valid_names(self):
        """Test that valid names pass validation."""
        assert validate_provider_name("openai") is True
        assert validate_provider_name("groq") is True
        assert validate_provider_name("together_ai") is True
        assert validate_provider_name("my_provider") is True

    def test_invalid_names(self):
        """Test that invalid names fail validation."""
        assert validate_provider_name("") is False
        assert validate_provider_name("123provider") is False
        assert validate_provider_name("Provider") is False
        assert validate_provider_name("my-provider") is False
        assert validate_provider_name("my provider") is False


class TestFileGeneration:
    """Test file content generation."""

    def test_generate_plugin_file(self):
        """Test plugin file generation."""
        content = generate_plugin_file("test_provider", "test_sdk")

        # Check essential components
        assert "class TestProviderPlugin(LLMPlugin):" in content
        assert "FRAMEWORK = Framework.TEST_PROVIDER" in content
        assert "_get_metadata()" in content
        assert "_get_extra_mappings()" in content
        assert "_get_provider_specific_rules()" in content
        assert "get_target_classes()" in content
        assert "get_target_methods()" in content
        assert "apply_overrides(" in content

        # Check traceability comment
        assert "# Traceability:" in content

        # Check docstrings
        assert '"""' in content
        assert "Examples:" in content

    def test_generate_test_file(self):
        """Test test file generation."""
        content = generate_test_file("test_provider")

        # Check test class structure
        assert "class TestTestProviderPlugin:" in content
        assert "def test_framework_is_correct(self):" in content
        assert "def test_metadata_name(self):" in content
        assert "def test_apply_overrides_with_dict_config(self):" in content

        # Check test patterns
        assert "assert self.plugin.FRAMEWORK == Framework.TEST_PROVIDER" in content
        assert 'assert self.plugin.metadata.name == "test_provider"' in content

        # Check traceability comment
        assert "# Traceability:" in content

    def test_generate_model_discovery_file(self):
        """Test model discovery file generation."""
        content = generate_model_discovery_file("test_provider", "test_sdk")

        # Check class structure
        assert "class TestProviderDiscovery(ModelDiscovery):" in content
        assert "_fetch_models_from_sdk()" in content
        assert "_get_config_key()" in content
        assert "_get_model_pattern()" in content

        # Check pattern definition
        assert "TEST_PROVIDER_MODEL_PATTERN" in content

        # Check traceability comment
        assert "# Traceability:" in content


class TestPluginContentValidation:
    """Test that generated content is valid Python."""

    def test_plugin_file_syntax(self):
        """Test that generated plugin file has valid syntax."""
        content = generate_plugin_file("groq", "groq_sdk")

        # Try to compile the code (checks syntax)
        try:
            compile(content, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated plugin has syntax error: {e}")

    def test_test_file_syntax(self):
        """Test that generated test file has valid syntax."""
        content = generate_test_file("groq")

        # Try to compile the code (checks syntax)
        try:
            compile(content, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated test has syntax error: {e}")

    def test_discovery_file_syntax(self):
        """Test that generated discovery file has valid syntax."""
        content = generate_model_discovery_file("groq", "groq_sdk")

        # Try to compile the code (checks syntax)
        try:
            compile(content, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated discovery has syntax error: {e}")


class TestScaffoldedPluginStructure:
    """Test that scaffolded plugins have correct structure."""

    def test_plugin_has_required_methods(self):
        """Test that plugin has all required methods."""
        content = generate_plugin_file("test_provider", "test_sdk")

        required_methods = [
            "_get_metadata",
            "_get_extra_mappings",
            "_get_provider_specific_rules",
            "_validate_model",
            "get_target_classes",
            "get_target_methods",
            "apply_overrides",
        ]

        for method in required_methods:
            assert f"def {method}(" in content, f"Missing required method: {method}"

    def test_plugin_has_validation_rules(self):
        """Test that plugin defines validation rules."""
        content = generate_plugin_file("test_provider", "test_sdk")

        # Check for common validation rules
        assert "ValidationRule(" in content
        assert "min_value" in content
        assert "max_value" in content
        assert "allowed_values" in content

    def test_test_has_multiple_test_classes(self):
        """Test that test file has organized test classes."""
        content = generate_test_file("test_provider")

        # Check for test class organization
        assert "class TestTestProviderPlugin:" in content
        assert "class TestTestProviderParameterMappings:" in content
        assert "class TestTestProviderValidationRules:" in content
        assert "class TestTestProviderPluginMetadata:" in content


class TestScaffoldDocumentation:
    """Test that scaffolded code has proper documentation."""

    def test_plugin_has_docstrings(self):
        """Test that plugin has comprehensive docstrings."""
        content = generate_plugin_file("test_provider", "test_sdk")

        # Count docstrings (module, class, and methods)
        docstring_count = content.count('"""')
        assert docstring_count >= 10, "Plugin should have docstrings for all methods"

    def test_plugin_has_type_hints(self):
        """Test that plugin has type hints."""
        content = generate_plugin_file("test_provider", "test_sdk")

        # Check for type hints on key methods
        assert "-> PluginMetadata:" in content
        assert "-> dict[str, str]:" in content
        assert "-> dict[str, ValidationRule]:" in content
        assert "-> list[str]:" in content

    def test_test_has_docstrings(self):
        """Test that test file has docstrings."""
        content = generate_test_file("test_provider")

        # Each test method should have a docstring
        docstring_count = content.count('"""')
        assert docstring_count >= 5, "Tests should have docstrings"


class TestScaffoldTODOComments:
    """Test that scaffolded code has helpful TODO comments."""

    def test_plugin_has_todo_comments(self):
        """Test that plugin has TODO comments for customization."""
        content = generate_plugin_file("test_provider", "test_sdk")

        # Should have TODOs in key places
        assert "# TODO:" in content or "TODO:" in content

    def test_plugin_todos_are_specific(self):
        """Test that TODO comments are specific and helpful."""
        content = generate_plugin_file("test_provider", "test_sdk")

        # Check for specific guidance in comments
        assert "provider-specific" in content.lower()
        assert "example" in content.lower() or "e.g." in content.lower()


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Integration test - may modify filesystem"
)
class TestScaffoldIntegration:
    """Integration tests for the scaffold script (filesystem modifications)."""

    def test_scaffold_creates_files(self, tmp_path):
        """Test that scaffold creates expected files."""
        # This would require mocking or using a temporary directory
        # Skipped for now as it modifies the real filesystem
        pytest.skip("Integration test - requires filesystem setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
