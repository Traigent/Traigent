# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""Canonical SDK User-Agent construction for outbound HTTP clients."""

from __future__ import annotations

from importlib import metadata as importlib_metadata


def get_sdk_user_agent() -> str:
    """Return the canonical Traigent SDK User-Agent value."""
    try:
        version = importlib_metadata.version("traigent")
    except importlib_metadata.PackageNotFoundError:
        from traigent._version import get_version

        version = get_version()
    return f"traigent-sdk/{version}"
