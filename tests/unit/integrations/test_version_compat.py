"""Tests for version compatibility utilities."""

from __future__ import annotations

from traigent.integrations.utils.version_compat import (
    VersionCompatibilityManager,
    VersionMapping,
)


def test_mapping_rejects_unparsable_version() -> None:
    """applies_to_version should safely handle invalid version strings."""
    mapping = VersionMapping(min_version="1.0.0", max_version="2.0.0")

    assert mapping.applies_to_version("invalid-version") is False


def test_mapping_with_invalid_bounds_is_ignored() -> None:
    """Mappings with invalid bounds should never match."""
    mapping = VersionMapping(min_version="not_a_version")

    assert mapping.applies_to_version("1.2.3") is False


def test_get_compatible_mapping_handles_invalid_version() -> None:
    """Manager should return empty mapping when supplied a bad version."""
    manager = VersionCompatibilityManager()

    assert manager.get_compatible_mapping("openai", version="bad-version") == {}
