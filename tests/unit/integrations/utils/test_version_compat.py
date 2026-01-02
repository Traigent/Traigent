"""Unit tests for traigent.integrations.utils.version_compat.

Tests for version compatibility management for framework integrations.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

from unittest.mock import patch

import pytest
from packaging.version import Version

from traigent.integrations.utils.version_compat import (
    VersionCompatibilityManager,
    VersionMapping,
)


class TestVersionMapping:
    """Tests for VersionMapping class."""

    @pytest.fixture
    def mapping_min_only(self) -> VersionMapping:
        """Create a mapping with only minimum version."""
        return VersionMapping(
            min_version="1.0.0",
            parameter_mapping={"model": "model", "temperature": "temperature"},
        )

    @pytest.fixture
    def mapping_max_only(self) -> VersionMapping:
        """Create a mapping with only maximum version."""
        return VersionMapping(
            max_version="2.0.0",
            parameter_mapping={"model": "model_name"},
        )

    @pytest.fixture
    def mapping_range(self) -> VersionMapping:
        """Create a mapping with both min and max versions."""
        return VersionMapping(
            min_version="1.0.0",
            max_version="2.0.0",
            parameter_mapping={"model": "model", "temperature": "temp"},
            deprecated_params={"old_param"},
            new_params={"new_param"},
        )

    @pytest.fixture
    def mapping_no_bounds(self) -> VersionMapping:
        """Create a mapping with no version bounds."""
        return VersionMapping(
            parameter_mapping={"model": "model"},
        )

    # ========== applies_to_version tests ==========

    def test_applies_to_version_within_range(
        self, mapping_range: VersionMapping
    ) -> None:
        """Test mapping applies to version within range."""
        assert mapping_range.applies_to_version("1.5.0") is True

    def test_applies_to_version_at_min_boundary(
        self, mapping_range: VersionMapping
    ) -> None:
        """Test mapping applies to version at minimum boundary."""
        assert mapping_range.applies_to_version("1.0.0") is True

    def test_applies_to_version_at_max_boundary(
        self, mapping_range: VersionMapping
    ) -> None:
        """Test mapping applies to version at maximum boundary."""
        assert mapping_range.applies_to_version("2.0.0") is True

    def test_applies_to_version_below_min(self, mapping_range: VersionMapping) -> None:
        """Test mapping does not apply to version below minimum."""
        assert mapping_range.applies_to_version("0.9.0") is False

    def test_applies_to_version_above_max(self, mapping_range: VersionMapping) -> None:
        """Test mapping does not apply to version above maximum."""
        assert mapping_range.applies_to_version("2.1.0") is False

    def test_applies_to_version_min_only_above(
        self, mapping_min_only: VersionMapping
    ) -> None:
        """Test min-only mapping applies to versions above minimum."""
        assert mapping_min_only.applies_to_version("5.0.0") is True

    def test_applies_to_version_min_only_below(
        self, mapping_min_only: VersionMapping
    ) -> None:
        """Test min-only mapping does not apply to versions below minimum."""
        assert mapping_min_only.applies_to_version("0.5.0") is False

    def test_applies_to_version_max_only_below(
        self, mapping_max_only: VersionMapping
    ) -> None:
        """Test max-only mapping applies to versions below maximum."""
        assert mapping_max_only.applies_to_version("1.0.0") is True

    def test_applies_to_version_max_only_above(
        self, mapping_max_only: VersionMapping
    ) -> None:
        """Test max-only mapping does not apply to versions above maximum."""
        assert mapping_max_only.applies_to_version("3.0.0") is False

    def test_applies_to_version_no_bounds(
        self, mapping_no_bounds: VersionMapping
    ) -> None:
        """Test mapping with no bounds applies to any version."""
        assert mapping_no_bounds.applies_to_version("0.1.0") is True
        assert mapping_no_bounds.applies_to_version("100.0.0") is True

    def test_applies_to_version_with_invalid_version(
        self, mapping_range: VersionMapping
    ) -> None:
        """Test mapping does not apply to invalid version string."""
        assert mapping_range.applies_to_version("not-a-version") is False

    def test_applies_to_version_with_invalid_min_version(self) -> None:
        """Test mapping does not apply when min_version is invalid."""
        mapping = VersionMapping(
            min_version="invalid",
            parameter_mapping={"model": "model"},
        )
        assert mapping.applies_to_version("1.0.0") is False

    def test_applies_to_version_with_invalid_max_version(self) -> None:
        """Test mapping does not apply when max_version is invalid."""
        mapping = VersionMapping(
            max_version="invalid",
            parameter_mapping={"model": "model"},
        )
        assert mapping.applies_to_version("1.0.0") is False

    def test_applies_to_version_with_prerelease(
        self, mapping_range: VersionMapping
    ) -> None:
        """Test mapping applies to prerelease versions."""
        assert mapping_range.applies_to_version("1.5.0-beta.1") is True

    def test_applies_to_version_with_build_metadata(
        self, mapping_range: VersionMapping
    ) -> None:
        """Test mapping applies to versions with build metadata."""
        assert mapping_range.applies_to_version("1.5.0+build.123") is True

    # ========== _safe_parse_version tests ==========

    def test_safe_parse_version_with_valid_version(self) -> None:
        """Test parsing valid version string."""
        result = VersionMapping._safe_parse_version("1.2.3")
        assert result == Version("1.2.3")

    def test_safe_parse_version_with_none(self) -> None:
        """Test parsing None returns None."""
        result = VersionMapping._safe_parse_version(None)
        assert result is None

    def test_safe_parse_version_with_invalid_version(self) -> None:
        """Test parsing invalid version string returns None."""
        result = VersionMapping._safe_parse_version("not-a-version")
        assert result is None

    def test_safe_parse_version_with_empty_string(self) -> None:
        """Test parsing empty string returns None."""
        result = VersionMapping._safe_parse_version("")
        assert result is None

    def test_safe_parse_version_with_prerelease(self) -> None:
        """Test parsing prerelease version."""
        result = VersionMapping._safe_parse_version("1.0.0-alpha.1")
        assert result == Version("1.0.0-alpha.1")

    def test_safe_parse_version_with_build_metadata(self) -> None:
        """Test parsing version with build metadata."""
        result = VersionMapping._safe_parse_version("1.0.0+20240101")
        assert result == Version("1.0.0+20240101")


class TestVersionCompatibilityManager:
    """Tests for VersionCompatibilityManager class."""

    @pytest.fixture
    def manager(self) -> VersionCompatibilityManager:
        """Create a version compatibility manager instance."""
        return VersionCompatibilityManager()

    # ========== get_package_version tests ==========

    def test_get_package_version_for_installed_package(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test retrieving version of installed package."""
        with patch("importlib.metadata.version") as mock_version:
            mock_version.return_value = "1.2.3"
            result = manager.get_package_version("openai")
            assert result == "1.2.3"
            mock_version.assert_called_once_with("openai")

    def test_get_package_version_for_missing_package(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test retrieving version of missing package returns None."""
        with patch("importlib.metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError()
            result = manager.get_package_version("nonexistent-package")
            assert result is None

    def test_get_package_version_uses_cache(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test package version is cached after first retrieval."""
        with patch("importlib.metadata.version") as mock_version:
            mock_version.return_value = "1.2.3"

            # First call
            result1 = manager.get_package_version("openai")
            assert result1 == "1.2.3"

            # Second call should use cache
            result2 = manager.get_package_version("openai")
            assert result2 == "1.2.3"

            # Should only call metadata once
            mock_version.assert_called_once()

    def test_get_package_version_does_not_cache_none_for_missing_package(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test None is not cached for missing packages, allowing retry."""
        with patch("importlib.metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError()

            # First call
            result1 = manager.get_package_version("missing-pkg")
            assert result1 is None

            # Second call will try again (not cached)
            result2 = manager.get_package_version("missing-pkg")
            assert result2 is None

            # Should attempt to retrieve twice (no caching of None)
            assert mock_version.call_count == 2

    # ========== get_compatible_mapping tests ==========

    def test_get_compatible_mapping_with_explicit_version(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test getting mapping with explicitly provided version."""
        mapping = manager.get_compatible_mapping("openai", "1.0.0")
        assert "model" in mapping
        assert mapping["model"] == "model"

    def test_get_compatible_mapping_with_auto_detected_version(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test getting mapping with auto-detected package version."""
        with patch.object(manager, "get_package_version") as mock_get_version:
            mock_get_version.return_value = "1.0.0"
            mapping = manager.get_compatible_mapping("openai")
            assert "model" in mapping
            mock_get_version.assert_called_once_with("openai")

    def test_get_compatible_mapping_for_missing_package(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test getting mapping for package that is not installed."""
        with patch.object(manager, "get_package_version") as mock_get_version:
            mock_get_version.return_value = None
            mapping = manager.get_compatible_mapping("nonexistent")
            assert mapping == {}

    def test_get_compatible_mapping_with_invalid_version(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test getting mapping with invalid version string."""
        mapping = manager.get_compatible_mapping("openai", "not-a-version")
        assert mapping == {}

    def test_get_compatible_mapping_for_unknown_package(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test getting mapping for package with no known mappings."""
        mapping = manager.get_compatible_mapping("unknown-package", "1.0.0")
        assert mapping == {}

    def test_get_compatible_mapping_selects_newest_applicable_mapping(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test that newest applicable mapping is selected."""
        # For openai 2.0.0, should get the 2.0+ mapping
        mapping = manager.get_compatible_mapping("openai", "2.0.0")
        assert mapping["model"] == "model_id"  # Hypothetical change in 2.0

    def test_get_compatible_mapping_selects_older_mapping(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test that older mapping is selected for older version."""
        # For openai 1.5.0, should get the 1.0+ mapping
        mapping = manager.get_compatible_mapping("openai", "1.5.0")
        assert mapping["model"] == "model"  # Original mapping

    def test_get_compatible_mapping_for_anthropic(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test getting mapping for anthropic package."""
        mapping = manager.get_compatible_mapping("anthropic", "0.5.0")
        assert "max_tokens" in mapping
        assert mapping["max_tokens"] == "max_tokens_to_sample"

    def test_get_compatible_mapping_for_langchain_old(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test getting mapping for old LangChain version."""
        mapping = manager.get_compatible_mapping("langchain", "0.0.5")
        assert mapping["model"] == "model_name"

    def test_get_compatible_mapping_for_langchain_new(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test getting mapping for new LangChain version."""
        mapping = manager.get_compatible_mapping("langchain", "0.2.0")
        assert mapping["model"] == "model"

    # ========== validate_parameters tests ==========

    def test_validate_parameters_with_no_deprecated_params(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test validation with no deprecated parameters."""
        params = {"model": "gpt-4", "temperature": 0.7}
        issues = manager.validate_parameters("openai", "1.0.0", params)
        assert len(issues) == 0

    def test_validate_parameters_with_deprecated_params(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test validation detects deprecated parameters.

        Note: The openai 2.0.0 mapping has deprecated_params={"max_tokens"},
        which means the Traigent parameter "max_tokens" is deprecated in favor
        of "max_completion_tokens" in the parameter_mapping.
        """
        # Create a custom test mapping with clear deprecated parameter
        custom_mapping = VersionMapping(
            min_version="2.0.0",
            parameter_mapping={"new_param": "new_param"},
            deprecated_params={"old_param"},
        )
        manager.version_mappings["test-pkg"] = [custom_mapping]

        params = {"old_param": "value"}
        issues = manager.validate_parameters("test-pkg", "2.0.0", params)
        assert len(issues) == 1
        assert "old_param" in issues[0]
        assert "deprecated" in issues[0].lower()

    def test_validate_parameters_for_unknown_package(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test validation for unknown package returns no issues."""
        params = {"any_param": "value"}
        issues = manager.validate_parameters("unknown-package", "1.0.0", params)
        assert len(issues) == 0

    def test_validate_parameters_with_multiple_deprecated_params(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test validation detects multiple deprecated parameters."""
        # Create a custom mapping with multiple deprecated params
        custom_mapping = VersionMapping(
            min_version="2.0.0",
            parameter_mapping={},
            deprecated_params={"param1", "param2", "param3"},
        )
        manager.version_mappings["test-pkg"] = [custom_mapping]

        params = {"param1": "a", "param2": "b", "param3": "c", "valid_param": "d"}
        issues = manager.validate_parameters("test-pkg", "2.0.0", params)
        assert len(issues) == 3

    def test_validate_parameters_with_version_not_matching_mapping(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test validation when version doesn't match any mapping."""
        params = {"any_param": "value"}
        # Version too old to match any openai mapping
        issues = manager.validate_parameters("openai", "0.5.0", params)
        assert len(issues) == 0

    # ========== migrate_parameters tests ==========

    def test_migrate_parameters_no_changes(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test migration when parameters don't change between versions."""
        params = {"model": "gpt-4", "temperature": 0.7}
        migrated = manager.migrate_parameters("openai", "1.0.0", "1.5.0", params)
        assert migrated == params

    def test_migrate_parameters_with_renamed_param(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test migration when parameter is renamed."""
        params = {"max_tokens": 100}
        migrated = manager.migrate_parameters("openai", "1.0.0", "2.0.0", params)
        # Should migrate max_tokens -> max_completion_tokens
        assert "max_completion_tokens" in migrated
        assert migrated["max_completion_tokens"] == 100

    def test_migrate_parameters_keeps_unmapped_params(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test migration keeps parameters without mapping."""
        params = {"custom_param": "value", "model": "gpt-4"}
        migrated = manager.migrate_parameters("openai", "1.0.0", "1.5.0", params)
        assert "custom_param" in migrated
        assert migrated["custom_param"] == "value"

    def test_migrate_parameters_langchain_model_name_change(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test migration of LangChain model_name to model."""
        params = {"model_name": "gpt-4"}
        migrated = manager.migrate_parameters("langchain", "0.0.5", "0.2.0", params)
        # Old version uses model_name, new version uses model
        assert "model" in migrated
        assert migrated["model"] == "gpt-4"

    def test_migrate_parameters_with_no_mappings(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test migration when package has no mappings."""
        params = {"param1": "value1", "param2": "value2"}
        migrated = manager.migrate_parameters("unknown-pkg", "1.0.0", "2.0.0", params)
        # Should return unchanged
        assert migrated == params

    def test_migrate_parameters_complex_migration(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test complex migration with multiple parameter changes."""
        params = {"model": "gpt-4", "max_tokens": 100, "temperature": 0.7}
        migrated = manager.migrate_parameters("openai", "1.0.0", "2.0.0", params)

        # model should map to model_id
        assert "model_id" in migrated
        assert migrated["model_id"] == "gpt-4"

        # max_tokens should map to max_completion_tokens
        assert "max_completion_tokens" in migrated
        assert migrated["max_completion_tokens"] == 100

        # temperature should remain (it's in both mappings)
        assert "temperature" in migrated
        assert migrated["temperature"] == 0.7

    def test_migrate_parameters_preserves_values(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test migration preserves parameter values correctly."""
        params = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.95,
        }
        migrated = manager.migrate_parameters("openai", "1.0.0", "1.5.0", params)

        # All values should be preserved
        for _key, value in params.items():
            # Check the value exists in migrated dict (might have different key)
            assert value in migrated.values()

    # ========== get_deprecation_warnings tests ==========

    def test_get_deprecation_warnings_no_warnings(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test no warnings for version without deprecated parameters."""
        warnings = manager.get_deprecation_warnings("openai", "1.0.0")
        assert len(warnings) == 0

    def test_get_deprecation_warnings_with_deprecated_params(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test warnings are generated for deprecated parameters."""
        warnings = manager.get_deprecation_warnings("openai", "2.0.0")
        assert len(warnings) == 1
        assert "max_tokens" in warnings[0]
        assert "deprecated" in warnings[0].lower()

    def test_get_deprecation_warnings_for_unknown_package(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test no warnings for unknown package."""
        warnings = manager.get_deprecation_warnings("unknown-package", "1.0.0")
        assert len(warnings) == 0

    def test_get_deprecation_warnings_version_not_matching(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test no warnings when version doesn't match any mapping."""
        # Version too old to match deprecation mapping
        warnings = manager.get_deprecation_warnings("openai", "0.5.0")
        assert len(warnings) == 0

    def test_get_deprecation_warnings_multiple_params(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test warnings for multiple deprecated parameters."""
        # Create a custom mapping with multiple deprecated params
        custom_mapping = VersionMapping(
            min_version="3.0.0",
            parameter_mapping={},
            deprecated_params={"old_param1", "old_param2"},
        )
        manager.version_mappings["test-pkg"] = [custom_mapping]

        warnings = manager.get_deprecation_warnings("test-pkg", "3.0.0")
        assert len(warnings) == 2
        assert any("old_param1" in w for w in warnings)
        assert any("old_param2" in w for w in warnings)

    # ========== Integration tests ==========

    def test_load_version_mappings_creates_all_packages(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test that version mappings are loaded for all known packages."""
        assert "openai" in manager.version_mappings
        assert "anthropic" in manager.version_mappings
        assert "langchain" in manager.version_mappings

    def test_load_version_mappings_creates_multiple_mappings_per_package(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test that multiple version mappings are created per package."""
        assert len(manager.version_mappings["openai"]) >= 2
        assert len(manager.version_mappings["langchain"]) >= 2

    def test_version_cache_is_initially_empty(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test that version cache starts empty."""
        assert len(manager._version_cache) == 0

    def test_full_workflow_custom_package_upgrade(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test complete workflow of package version upgrade."""
        # Create custom mappings for a test package
        old_mapping = VersionMapping(
            min_version="1.0.0",
            max_version="2.0.0",
            parameter_mapping={"param1": "param1", "param2": "param2"},
        )
        new_mapping = VersionMapping(
            min_version="2.0.0",
            parameter_mapping={"param1": "param1_v2", "param2": "param2"},
            deprecated_params={"old_param"},
        )
        manager.version_mappings["test-sdk"] = [old_mapping, new_mapping]

        # Start with parameters in old format
        old_params = {"param1": "value1", "param2": "value2"}

        # Validate parameters against old version (should be fine)
        issues_old = manager.validate_parameters("test-sdk", "1.0.0", old_params)
        assert len(issues_old) == 0

        # Get deprecation warnings for new version
        warnings = manager.get_deprecation_warnings("test-sdk", "2.0.0")
        assert len(warnings) == 1
        assert "old_param" in warnings[0]

        # Migrate parameters
        new_params = manager.migrate_parameters(
            "test-sdk", "1.0.0", "2.0.0", old_params
        )
        assert "param1_v2" in new_params
        assert new_params["param1_v2"] == "value1"

    def test_full_workflow_langchain_upgrade(
        self, manager: VersionCompatibilityManager
    ) -> None:
        """Test complete workflow of upgrading LangChain."""
        # Old LangChain used model_name
        old_params = {"model_name": "gpt-4"}

        # Migrate to new version
        new_params = manager.migrate_parameters(
            "langchain", "0.0.5", "0.2.0", old_params
        )

        # Should now use model instead of model_name
        assert "model" in new_params
        assert new_params["model"] == "gpt-4"
