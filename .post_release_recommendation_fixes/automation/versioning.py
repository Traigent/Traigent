#!/usr/bin/env python3
"""Version and path helpers for post-release workflow automation."""

from __future__ import annotations

import os
import re
from pathlib import Path

_VERSION_RE = re.compile(r"^[A-Za-z0-9._+-]+$")


def validate_version(version: str) -> str:
    """Validate version to prevent path traversal."""
    if not _VERSION_RE.fullmatch(version):
        raise ValueError(
            "Invalid version. Use only letters, numbers, '.', '-', '_', and '+'"
        )
    return version


def ensure_within_base(base_path: Path, target_path: Path) -> Path:
    """Ensure target_path resolves within base_path."""
    base_resolved = base_path.resolve()
    target_resolved = target_path.resolve()
    if base_resolved == target_resolved or base_resolved in target_resolved.parents:
        return target_path
    raise ValueError(f"Resolved path {target_resolved} escapes {base_resolved}")


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
        resolved = validate_version(resolved)
        candidate = base / resolved
    else:
        candidate = base
    ensure_within_base(base, candidate)
    return candidate


def read_tracking_version(
    tracking_path: Path, base_path: Path | None = None
) -> str | None:
    """Read Release Version from a tracking file header."""
    if base_path is not None:
        ensure_within_base(base_path, tracking_path)
    if not tracking_path.exists():
        return None
    for line in tracking_path.read_text().splitlines():
        if line.startswith("**Release Version**:"):
            return line.split(":", 1)[1].strip().strip("`")
    return None
