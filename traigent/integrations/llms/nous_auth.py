"""Nous Portal (Hermes) authentication helper for the Traigent SDK.

Nous Portal is OpenAI-compatible (chat completions via a persistent client),
but unlike every other provider its credential is a **short-lived JWT minted
from a long-lived refresh token**, not a static API key. This module resolves
and caches that JWT — refreshing before expiry — and exposes
:func:`get_nous_api_key` so model discovery, the runnable example, and a user's
own ``@traigent.optimize`` function can obtain a *current* bearer token the same
way the OpenRouter example reads ``os.environ["OPENROUTER_API_KEY"]`` today.

Because the OpenRouter pattern re-executes the decorated function per trial, the
client is (re)constructed per trial and ``get_nous_api_key()`` is called per
trial; the ``min_ttl_seconds`` skew re-mints before expiry, so a long run
survives with no mid-flight retry machinery. Constructing the client **once
outside** the decorated function is unsupported — the SDK has no 401-retry hook.

Design invariants (non-negotiable):

* **Fail loud, always.** Every failure path — missing credentials, malformed
  ``auth.json``, a 4xx/5xx from the token endpoint, a response with no usable
  token or expiry — raises :class:`NousAuthError` naming the source. The helper
  **never** serves a stale, empty, or mock token, and **never** logs the token.
* **Cheap credential probe.** :func:`has_nous_credentials` does env reads + a
  file-existence check only (no network, no mint) so discovery can decide
  whether to attempt SDK discovery or fall back to the ``models.yaml`` list.

OWNER — Phase-0 unknowns (see the ``OWNER:`` comments below): the token-endpoint
URL and the ``~/.hermes/auth.json`` schema are unverifiable offline. They are
encoded as env-overridable constants + a single parse location so the OWNER can
confirm them against real ``hermes-agent`` output before merge.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Security FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from traigent.utils.exceptions import TraigentError

logger = logging.getLogger(__name__)


# OWNER: confirm against real hermes-agent output in Phase-0 before merge.
# The Nous Portal inference base URL (OpenAI-compatible ``/v1`` surface). It is
# passed EXPLICITLY as ``base_url`` to every OpenAI client so calls never fall
# through to ``api.openai.com``. Override with ``TRAIGENT_NOUS_BASE_URL``.
_DEFAULT_NOUS_BASE_URL = "https://inference-api.nousresearch.com/v1"

# OWNER: confirm against real hermes-agent output in Phase-0 before merge.
# The endpoint that exchanges a refresh token for a short-lived access JWT.
# Override with ``TRAIGENT_NOUS_TOKEN_URL``.
_DEFAULT_NOUS_TOKEN_URL = "https://portal.nousresearch.com/api/auth/refresh"

# OWNER: confirm the ~/.hermes/auth.json schema against real hermes-agent output
# in Phase-0 before merge. Assumed shape is a JSON object carrying the refresh
# token under one of these keys. If the real file nests it (e.g. under a
# ``tokens`` object), update the parse in :func:`_read_refresh_token_from_file`.
_AUTH_FILE_REFRESH_KEYS = ("refresh_token", "refreshToken")

# Effective inference base URL (env-overridable at import). Exported so discovery
# and the example pass the SAME value — the single source of the base_url.
NOUS_BASE_URL: str = os.environ.get("TRAIGENT_NOUS_BASE_URL", _DEFAULT_NOUS_BASE_URL)

# Env var names, in resolution order.
_STATIC_KEY_ENV = "NOUS_API_KEY"
_REFRESH_TOKEN_ENVS = ("NOUS_REFRESH_TOKEN", "NOUS_PORTAL_REFRESH_TOKEN")
_AUTH_FILE_ENV = "TRAIGENT_NOUS_AUTH_FILE"
_TOKEN_URL_ENV = "TRAIGENT_NOUS_TOKEN_URL"

# Minimum life a freshly-minted token must have to be worth caching. A mint that
# resolves to an expiry already past (or within this epsilon of) ``now`` yields a
# useless token; we fail loud rather than cache/serve an already-expired one.
_MIN_TOKEN_LIFETIME_SECONDS = 1.0


class NousAuthError(TraigentError):
    """Raised on any Nous Portal credential-resolution or token-refresh failure.

    A subclass of :class:`~traigent.utils.exceptions.TraigentError` so callers
    can catch it alongside other SDK errors; it always names the credential
    source that failed and never contains a token value.
    """


@dataclass
class _TokenState:
    """In-process cache of a minted access JWT and its absolute expiry."""

    jwt: str
    expires_at: float


_state_lock = threading.Lock()
_token_state: _TokenState | None = None


def _now() -> float:
    """Current wall-clock time (seconds). Indirected so tests can inject a clock."""
    return time.time()


def _auth_file_path() -> Path:
    """Resolve the Hermes auth-file path (``TRAIGENT_NOUS_AUTH_FILE`` override)."""
    override = os.environ.get(_AUTH_FILE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hermes" / "auth.json"


def _token_url() -> str:
    """Resolve the token endpoint (``TRAIGENT_NOUS_TOKEN_URL`` override)."""
    return os.environ.get(_TOKEN_URL_ENV, _DEFAULT_NOUS_TOKEN_URL)


def has_nous_credentials() -> bool:
    """Return ``True`` if any Nous credential source is present.

    Cheap and side-effect-free: env reads + a file-existence check, no network
    and no mint. Discovery uses this to decide between attempting SDK discovery
    and falling back to the ``models.yaml`` known-model list.
    """
    if os.environ.get(_STATIC_KEY_ENV):
        return True
    if any(os.environ.get(name) for name in _REFRESH_TOKEN_ENVS):
        return True
    return _auth_file_path().exists()


def clear_nous_auth_cache() -> None:
    """Drop the in-process minted-JWT cache (test hook / manual re-auth)."""
    global _token_state
    with _state_lock:
        _token_state = None


def get_nous_api_key(
    *,
    min_ttl_seconds: int = 300,
    force_refresh: bool = False,
) -> str:
    """Return a current Nous Portal bearer token, minting/refreshing as needed.

    Credential resolution order (first hit wins, then stop — no silent
    cascading fallback):

    1. ``NOUS_API_KEY`` — a manually-minted JWT / static escape hatch. Returned
       **as-is, with no refresh** (it will expire without auto-refresh). This is
       also the offline/mock path: the demo scaffolding seeds this env var, so
       the example runs with zero network.
    2. ``NOUS_REFRESH_TOKEN`` / ``NOUS_PORTAL_REFRESH_TOKEN`` — a headless/CI
       refresh token, exchanged for a short-lived JWT and cached.
    3. ``~/.hermes/auth.json`` (override ``TRAIGENT_NOUS_AUTH_FILE``) — read-only;
       the refresh token is parsed out and used as in (2).

    Args:
        min_ttl_seconds: Re-mint if the cached JWT has less than this many
            seconds of life left, so a per-trial call stays ahead of expiry.
        force_refresh: Ignore the cache and mint a fresh token (only meaningful
            for the refresh-token flow; the ``NOUS_API_KEY`` escape hatch is
            always returned as-is).

    Returns:
        A non-empty bearer token string.

    Raises:
        NousAuthError: If no credential source is present, ``auth.json`` is
            malformed, the token endpoint fails, or the response carries no
            usable token/expiry. Never returns a stale, empty, or mock token.
    """
    static = os.environ.get(_STATIC_KEY_ENV)
    if static:
        # Manually-minted JWT / static escape hatch: returned verbatim with no
        # network and no refresh. It will expire without auto-refresh.
        return static

    with _state_lock:
        global _token_state
        if (
            not force_refresh
            and _token_state is not None
            and _token_state.expires_at - _now() >= min_ttl_seconds
        ):
            return _token_state.jwt

        refresh_token, source = _resolve_refresh_token()
        state = _mint_token(refresh_token, source)
        _token_state = state
        return state.jwt


def _resolve_refresh_token() -> tuple[str, str]:
    """Return ``(refresh_token, human_source)`` or raise naming all sources."""
    for env_name in _REFRESH_TOKEN_ENVS:
        value = os.environ.get(env_name)
        if value:
            return value, f"${env_name}"

    auth_path = _auth_file_path()
    if auth_path.exists():
        return _read_refresh_token_from_file(auth_path), str(auth_path)

    raise NousAuthError(
        "No Nous Portal credentials found. Set NOUS_API_KEY (a pre-minted "
        "JWT), or NOUS_REFRESH_TOKEN / NOUS_PORTAL_REFRESH_TOKEN (a refresh "
        f"token), or log in with the Hermes CLI so {auth_path} exists."
    )


def _read_refresh_token_from_file(path: Path) -> str:
    """Parse the refresh token out of the Hermes auth file, or raise loudly."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NousAuthError(f"Could not read Nous auth file {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NousAuthError(f"Nous auth file {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise NousAuthError(
            f"Nous auth file {path} must be a JSON object carrying a refresh "
            f"token under one of {list(_AUTH_FILE_REFRESH_KEYS)}."
        )

    for key in _AUTH_FILE_REFRESH_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value

    raise NousAuthError(
        f"Nous auth file {path} has no refresh token under any of "
        f"{list(_AUTH_FILE_REFRESH_KEYS)}."
    )


def _mint_token(refresh_token: str, source: str) -> _TokenState:
    """Exchange a refresh token for a JWT + expiry, or raise :class:`NousAuthError`."""
    token_url = _token_url()
    try:
        payload = _request_new_token(refresh_token, token_url=token_url)
    except NousAuthError:
        raise
    except Exception as exc:  # network / requests errors
        raise NousAuthError(
            f"Nous token refresh request to {token_url} failed "
            f"(refresh token from {source}): {type(exc).__name__}: {exc}"
        ) from exc

    jwt_token = payload.get("access_token") or payload.get("accessToken")
    if not isinstance(jwt_token, str) or not jwt_token:
        raise NousAuthError(
            f"Nous token endpoint {token_url} returned no access token "
            f"(response keys: {sorted(payload)})."
        )

    expires_at = _expires_at_from_payload(payload, jwt_token)
    return _TokenState(jwt=jwt_token, expires_at=expires_at)


def _request_new_token(
    refresh_token: str,
    *,
    token_url: str,
    timeout: float = 30.0,
) -> dict:
    """POST the refresh token to the Nous token endpoint; return the parsed JSON.

    This is the single network seam — offline tests monkeypatch it so no HTTP is
    performed. The response body is never echoed into an exception message (it
    may contain the token). ``requests`` is a core SDK dependency, imported
    lazily so importing this module stays network- and heavy-dep-free.
    """
    import requests

    # OWNER: confirm in Phase-0 — Nous may expect grant_type=refresh_token
    # form-encoding instead of this JSON body. The request shape (JSON body +
    # Content-Type header, and the "refresh_token" field name) is an UNVERIFIED
    # assumption on par with the URL / auth.json schema / model-id spellings the
    # other OWNER: markers flag, so it must be confirmed against real
    # hermes-agent output before merge.
    response = requests.post(
        token_url,
        json={"refresh_token": refresh_token},
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise NousAuthError(
            f"Nous token endpoint {token_url} returned HTTP {response.status_code}."
        )
    data = response.json()
    if not isinstance(data, dict):
        raise NousAuthError(
            f"Nous token endpoint {token_url} returned a non-object JSON body."
        )
    return data


def _expires_at_from_payload(payload: dict, jwt_token: str) -> float:
    """Absolute expiry, honoring the SHORTER of ``expires_in`` and the JWT ``exp``.

    Both signals are consulted. When both are present the **shorter** lifetime
    wins (``min(now + expires_in, jwt_exp)``) so an ``expires_in`` that overstates
    the JWT's own ``exp`` can never push the cached expiry past the token's real
    deadline. If neither signal is present — or the resulting expiry is already
    past (or within :data:`_MIN_TOKEN_LIFETIME_SECONDS` of now) — this raises
    :class:`NousAuthError` so the caller never caches or serves a stale token.
    """
    now = _now()

    candidates: list[float] = []

    expires_in = payload.get("expires_in")
    if expires_in is None:
        expires_in = payload.get("expiresIn")
    if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
        if expires_in > 0:
            candidates.append(now + float(expires_in))

    exp = _decode_jwt_exp(jwt_token)
    if exp is not None:
        candidates.append(float(exp))

    if not candidates:
        raise NousAuthError(
            "Nous token response has neither an 'expires_in' field nor a decodable "
            "'exp' claim; refusing to serve a token with unknown expiry."
        )

    # Trust the shorter deadline when both are available.
    expires_at = min(candidates)

    # Never cache/serve an already-expired (or about-to-expire) mint: a freshly
    # minted token with no usable life left is a hard failure, not a fallback.
    if expires_at - now <= _MIN_TOKEN_LIFETIME_SECONDS:
        raise NousAuthError(
            "Nous token endpoint returned an already-expired token "
            f"(computed lifetime {expires_at - now:.3f}s <= "
            f"{_MIN_TOKEN_LIFETIME_SECONDS:.3f}s); refusing to cache or serve it."
        )

    return expires_at


def _decode_jwt_exp(token: str) -> int | None:
    """Read the ``exp`` claim from a JWT payload (stdlib base64, no signature check).

    Returns ``None`` if the token is not a decodable three-segment JWT or has no
    integer ``exp`` — the caller then fails loud rather than guessing a lifetime.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload_segment = parts[1]
    padding = "=" * (-len(payload_segment) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        claims = json.loads(decoded)
    except (binascii.Error, ValueError, json.JSONDecodeError):
        # Malformed base64 (``binascii.Error``), non-UTF-8 bytes / non-JSON
        # payload (``ValueError`` / ``json.JSONDecodeError``): treat as "no
        # decodable exp" and return None so the caller fails loud with a clean
        # NousAuthError instead of leaking a raw base64/JSON exception.
        return None
    if not isinstance(claims, dict):
        return None
    exp = claims.get("exp")
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        return int(exp)
    return None


__all__ = [
    "NOUS_BASE_URL",
    "NousAuthError",
    "get_nous_api_key",
    "has_nous_credentials",
    "clear_nous_auth_cache",
]
