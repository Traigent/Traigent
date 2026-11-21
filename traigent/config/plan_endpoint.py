"""Backend endpoint resolution shared by ``traigent plan`` and the MCP plan tool.

Kept click-free so ``traigent.mcp`` can import it without pulling in the CLI.
"""

from __future__ import annotations

import os

from traigent.config.backend_config import (
    DEFAULT_LOCAL_URL,
    BackendConfig,
    _endpoint_required_error,
    is_strict_endpoint_mode,
)

# Order matches the click ``envvar`` list on ``traigent plan --backend-url``.
PLAN_BACKEND_URL_ENV_VARS: tuple[str, ...] = (
    "TRAIGENT_BACKEND_URL",
    "TRAIGENT_API_URL",
)


def backend_url_from_env() -> str | None:
    """Return the first non-empty plan backend URL env var, as click would."""
    for name in PLAN_BACKEND_URL_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def resolve_backend_url(explicit: str | None) -> tuple[str, bool]:
    """Resolve the backend URL.

    Precedence: explicit flag/env value, then env-var / stored-CLI-credential
    configuration (previously ignored; Traigent#1721), then the local default.
    We fall back to ``DEFAULT_LOCAL_URL`` rather than the prod cloud so an
    unconfigured local-dev user keeps hitting localhost as before.

    Returns (resolved_url, used_explicit_source).
    """
    if explicit:
        return explicit, True
    configured = BackendConfig.get_configured_backend_url()
    if not configured and is_strict_endpoint_mode():
        # Strict (self-hosted) mode: no implicit localhost fallback either.
        raise _endpoint_required_error()
    return (configured or DEFAULT_LOCAL_URL), False


def resolve_api_key(explicit: str | None) -> str | None:
    """Resolve the API key: explicit value, ``TRAIGENT_API_KEY``, then stored
    CLI credentials from ``traigent auth login`` (Traigent#1721)."""
    if explicit:
        return explicit
    return BackendConfig.get_api_key()
