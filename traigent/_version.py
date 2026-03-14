"""Version information for Traigent SDK."""

# Traceability: CONC-Layer-Data CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path


def get_version() -> str:
    """Get the current version of the Traigent SDK.

    Returns:
        Version string from package metadata or fallback version
    """
    if override := os.getenv("TRAIGENT_FORCE_VERSION"):
        return override

    project_root = Path(__file__).resolve().parent.parent
    if (project_root / "pyproject.toml").exists():
        # In workspace/development mode we rely on the development version string
        return "0.10.0"

    if os.getenv("TRAIGENT_USE_PACKAGE_METADATA", "0") == "1":
        try:
            # Try to get version from installed package metadata
            return importlib.metadata.version("traigent")
        except importlib.metadata.PackageNotFoundError:
            pass

    # Fallback to hardcoded version for development/testing
    return "0.10.0"


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
