# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""Version information for Traigent SDK."""

# Traceability: CONC-Layer-Data CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path


def _read_pyproject_version() -> str | None:
    """Read version from pyproject.toml without external dependencies."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    except Exception:
        return None
    try:
        with open(pyproject, "rb") as f:
            pyproject_data = tomllib.load(f)
            project_data = pyproject_data.get("project")
            if not isinstance(project_data, dict):
                return None
            version = project_data.get("version")
            return version if isinstance(version, str) else None
    except Exception:
        return None


def get_version() -> str:
    """Get the current version of the Traigent SDK.

    Resolution order:
    1. TRAIGENT_FORCE_VERSION env var (override for testing)
    2. pyproject.toml (development mode - single source of truth)
    3. Installed package metadata (pip-installed mode)
    4. Loud failure if neither source is available

    Returns:
        Version string

    Raises:
        RuntimeError: If neither pyproject.toml nor installed package metadata
            can provide a version. Returning a hand-maintained fallback risks
            silently reporting a stale SDK version.
    """
    if override := os.getenv("TRAIGENT_FORCE_VERSION"):
        return override

    # Development mode: read from pyproject.toml (single source of truth)
    pyproject_version = _read_pyproject_version()
    if pyproject_version:
        return pyproject_version

    # Installed package mode: read from package metadata
    try:
        return importlib.metadata.version("traigent")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            "Unable to resolve Traigent SDK version from pyproject.toml or "
            "installed package metadata"
        ) from None

    raise RuntimeError(
        "Unable to resolve Traigent SDK version from pyproject.toml or "
        "installed package metadata"
    )


def get_version_info() -> dict[str, str]:
    """Get detailed version information.

    Returns:
        Dictionary containing version details
    """
    version = get_version()
    parts = version.split(".")

    return {
        "version": version,
        "major": parts[0] if len(parts) > 0 else "0",
        "minor": parts[1] if len(parts) > 1 else "0",
        "patch": parts[2] if len(parts) > 2 else "0",
    }


# Expose version at module level for easy access
__version__ = get_version()
