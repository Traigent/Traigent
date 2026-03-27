"""Shared tenant-header constants and env helpers for SDK clients."""

from __future__ import annotations

import os

TENANT_HEADER_NAME = "X-Tenant-Id"
TENANT_ENV_VAR = "TRAIGENT_TENANT_ID"


def read_optional_env(env_var: str) -> str | None:
    """Return a stripped env var value or ``None`` for missing/blank values."""
    value = os.getenv(env_var)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
