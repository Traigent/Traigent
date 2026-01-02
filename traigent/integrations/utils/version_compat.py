"""Version compatibility management for framework integrations.

This module handles parameter mapping across different SDK versions,
ensuring Traigent works with various framework versions.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, cast

from packaging.version import InvalidVersion, Version

from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VersionMapping:
    """Mapping configuration for a specific version range."""

    min_version: str | None = None
    max_version: str | None = None
    parameter_mapping: dict[str, str] = field(default_factory=dict)
    deprecated_params: set[str] = field(default_factory=set)
    new_params: set[str] = field(default_factory=set)

    def applies_to_version(self, ver: str) -> bool:
        """Check if this mapping applies to a given version."""
        parsed_ver = self._safe_parse_version(ver)
        if parsed_ver is None:
            logger.debug("Skipping version mapping due to unparsable version '%s'", ver)
            return False

        min_ver = self._safe_parse_version(self.min_version)
        if self.min_version and min_ver is None:
            logger.debug(
                "Version mapping min_version '%s' is invalid; ignoring mapping",
                self.min_version,
            )
            return False

        max_ver = self._safe_parse_version(self.max_version)
        if self.max_version and max_ver is None:
            logger.debug(
                "Version mapping max_version '%s' is invalid; ignoring mapping",
                self.max_version,
            )
            return False

        if min_ver and parsed_ver < min_ver:
            return False

        if max_ver and parsed_ver > max_ver:
            return False

        return True

    @staticmethod
    def _safe_parse_version(value: str | None) -> Version | None:
        """Safely parse version strings."""
        if value is None:
            return None

        try:
            return Version(value)
        except (InvalidVersion, TypeError):
            return None


class VersionCompatibilityManager:
    """Manage parameter mappings across different SDK versions."""

    def __init__(self) -> None:
        """Initialize the version compatibility manager."""
        self.version_mappings = self._load_version_mappings()
        self._version_cache: dict[str, Any] = {}

    def _load_version_mappings(self) -> dict[str, list[VersionMapping]]:
        """Load known version mappings for different packages."""
        return {
            "openai": [
                VersionMapping(
                    min_version="1.0.0",
                    parameter_mapping={
                        "model": "model",
                        "temperature": "temperature",
                        "max_tokens": "max_tokens",
                        "top_p": "top_p",
                        "stream": "stream",
                        "tools": "tools",
                    },
                ),
                # Future version example
                VersionMapping(
                    min_version="2.0.0",
                    parameter_mapping={
                        "model": "model_id",  # Hypothetical change
                        "temperature": "temperature",
                        "max_tokens": "max_completion_tokens",  # Hypothetical change
                    },
                    deprecated_params={"max_tokens"},
                    new_params={"max_completion_tokens"},
                ),
            ],
            "anthropic": [
                VersionMapping(
                    min_version="0.1.0",
                    parameter_mapping={
                        "model": "model",
                        "max_tokens": "max_tokens_to_sample",
                        "temperature": "temperature",
                    },
                ),
            ],
            "langchain": [
                VersionMapping(
                    max_version="0.1.0",
                    parameter_mapping={
                        "model": "model_name",  # Old LangChain
                    },
                ),
                VersionMapping(
                    min_version="0.1.0",
                    parameter_mapping={
                        "model": "model",  # New LangChain
                    },
                ),
            ],
        }

    def get_package_version(self, package: str) -> str | None:
        """Get the installed version of a package."""
        if package in self._version_cache:
            return cast(str | None, self._version_cache[package])

        try:
            ver = importlib.metadata.version(package)
            self._version_cache[package] = ver
            return ver
        except importlib.metadata.PackageNotFoundError:
            logger.debug(f"Package {package} not found")
            return None

    def get_compatible_mapping(
        self, package: str, version: str | None = None
    ) -> dict[str, str]:
        """Get parameter mapping for specific package version.

        Args:
            package: Package name
            version: Optional version string. If None, will try to detect.

        Returns:
            Parameter mapping dictionary
        """
        if version is None:
            resolved_version = self.get_package_version(package)
            if resolved_version is None:
                return {}
        else:
            resolved_version = version

        if VersionMapping._safe_parse_version(resolved_version) is None:
            logger.debug(
                "Unable to determine compatible mapping: version '%s' is invalid",
                resolved_version,
            )
            return {}

        mappings = self.version_mappings.get(package, [])

        # Find the best matching mapping
        for mapping in reversed(mappings):  # Check newer versions first
            if mapping.applies_to_version(resolved_version):
                return mapping.parameter_mapping

        return {}

    def validate_parameters(
        self, package: str, version: str, params: dict[str, Any]
    ) -> list[str]:
        """Validate parameters against known version schema.

        Args:
            package: Package name
            version: Version string
            params: Parameters to validate

        Returns:
            List of validation warnings/errors
        """
        issues = []
        mappings = self.version_mappings.get(package, [])

        for mapping in mappings:
            if mapping.applies_to_version(version):
                # Check for deprecated parameters
                for param in params:
                    if param in mapping.deprecated_params:
                        issues.append(
                            f"Parameter '{param}' is deprecated in {package} {version}"
                        )

                break

        return issues

    def migrate_parameters(
        self, package: str, old_version: str, new_version: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Migrate parameters from old to new version format.

        Args:
            package: Package name
            old_version: Old version string
            new_version: New version string
            params: Parameters in old format

        Returns:
            Parameters in new format
        """
        old_mapping = self.get_compatible_mapping(package, old_version)
        new_mapping = self.get_compatible_mapping(package, new_version)

        # Create reverse mappings
        old_reverse = {v: k for k, v in old_mapping.items()}

        migrated = {}
        for param_name, param_value in params.items():
            # Find the Traigent parameter name
            traigent_param = old_reverse.get(param_name, param_name)

            # Map to new format
            if traigent_param in new_mapping:
                new_param_name = new_mapping[traigent_param]
                migrated[new_param_name] = param_value
            else:
                # Keep as is if no mapping found
                migrated[param_name] = param_value

        return migrated

    def get_deprecation_warnings(self, package: str, version: str) -> list[str]:
        """Get deprecation warnings for a specific package version."""
        warnings = []
        mappings = self.version_mappings.get(package, [])

        for mapping in mappings:
            if mapping.applies_to_version(version) and mapping.deprecated_params:
                for param in mapping.deprecated_params:
                    warnings.append(
                        f"Parameter '{param}' is deprecated in {package} {version}"
                    )

        return warnings
