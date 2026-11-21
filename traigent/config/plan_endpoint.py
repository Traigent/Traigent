"""Backend endpoint resolution shared by ``traigent plan`` and the MCP plan tool.

Kept click-free so ``traigent.mcp`` can import it without pulling in the CLI.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from traigent.cloud.url_security import (
    _reject_cloud_path_traversal,
    _reject_unsafe_whitespace,
    _validate_cloud_url_parts,
)
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


_PURPOSE = "backend"


def _check_raw_backend_url(raw: str) -> None:
    """Refuse a raw explicit value the origin normalization would hide.

    Reuses the guard helpers of ``traigent.cloud.url_security`` (no DNS here; the
    client still runs the full validation). Messages never echo the value.
    """
    try:
        _reject_unsafe_whitespace(raw, purpose=_PURPOSE)
    except ValueError:
        raise ValueError(
            "backend URL contains control characters or whitespace"
        ) from None
    if not all("\x21" <= ch <= "\x7e" for ch in raw):
        raise ValueError(
            "backend URL must contain only printable ASCII characters; "
            "use the punycode form for an international host name"
        )
    if "\\" in raw:
        raise ValueError("backend URL must not contain a backslash")
    if "?" in raw or "#" in raw:
        raise ValueError("backend URL must not include a query or fragment")
    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parsed = urlparse(candidate)
        # Reading the port and hostname can also raise; never let the value out.
        _ = parsed.port, parsed.hostname
    except ValueError:
        raise ValueError("backend URL is not a valid URL") from None
    _validate_cloud_url_parts(parsed, purpose=_PURPOSE)
    _reject_cloud_path_traversal(parsed.path, purpose=_PURPOSE)


def resolve_backend_url(explicit: str | None) -> tuple[str, bool]:
    """Resolve the backend URL.

    Precedence: explicit flag/env value, then env-var / stored-CLI-credential
    configuration (previously ignored; Traigent#1721), then the local default.
    We fall back to ``DEFAULT_LOCAL_URL`` rather than the prod cloud so an
    unconfigured local-dev user keeps hitting localhost as before.

    The result is always a backend ORIGIN: ``OptimizationPlanClient`` appends its own
    ``/api/v1/optimization/plan``, so an explicit value or env var that carries the
    API path (``https://host/api/v1``, a normal ``TRAIGENT_API_URL``) is reduced with
    :meth:`BackendConfig.normalize_backend_origin`, the same normalization the env /
    stored-credential branch already gets. The RAW explicit value is checked first,
    because normalization would hide what the URL guard refuses: control characters
    or any whitespace (surrounding spaces included), a query, fragment or params,
    userinfo, a backslash, or a ``..`` path segment (plain or percent-encoded) raise
    ``ValueError`` (the message never echoes the value). A value that still cannot
    be normalized is passed on unchanged and refused by the client's own URL
    validation.

    Returns (resolved_url, used_explicit_source).
    """
    if explicit:
        _check_raw_backend_url(explicit)
        return BackendConfig.normalize_backend_origin(explicit) or explicit, True
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
