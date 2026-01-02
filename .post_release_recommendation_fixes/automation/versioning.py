#!/usr/bin/env python3
"""Version and path helpers for post-release workflow automation."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_version(explicit: str | None = None) -> str | None:
    """Resolve release version from explicit input or environment."""
    if explicit:
        return explicit
    return os.environ.get("RR_VERSION")


def infer_version_from_source(source: Path) -> str | None:
    """Infer version from a .release_review/<version>/POST_RELEASE_TODO.md path."""
    parts = source.parts
    for idx, part in enumerate(parts[:-1]):
        if part == ".release_review" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def resolve_base_path(base_path: str | Path, version: str | None = None) -> Path:
    """Resolve versioned base path for post-release workflow."""
    base = Path(base_path)
    resolved = resolve_version(version)
    if resolved:
        return base / resolved
    return base


def read_tracking_version(tracking_path: Path) -> str | None:
    """Read Release Version from a tracking file header."""
    if not tracking_path.exists():
        return None
    for line in tracking_path.read_text().splitlines():
        if line.startswith("**Release Version**:"):
            return line.split(":", 1)[1].strip().strip("`")
    return None
