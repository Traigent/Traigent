"""Version information for Traigent SDK."""

# Traceability: CONC-Layer-Data CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

_FALLBACK_VERSION = "0.11.2"


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
            return tomllib.load(f)["project"]["version"]  # type: ignore[no-any-return]
    except Exception:
        return None


def get_version() -> str:
    """Get the current version of the Traigent SDK.

    Resolution order:
    1. TRAIGENT_FORCE_VERSION env var (override for testing)
    2. pyproject.toml (development mode - single source of truth)
    3. Installed package metadata (pip-installed mode)
    4. Hardcoded fallback

    Returns:
        Version string
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
        pass

    return _FALLBACK_VERSION


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
