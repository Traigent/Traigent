"""Shared credential resolution for analytics read clients."""

from __future__ import annotations

import os
from typing import Any

from traigent.cloud.auth import AuthManager, AuthMode, InvalidCredentialsError


def analytics_credential_kwargs_from_auth_headers(
    auth_headers: dict[str, str],
) -> dict[str, str]:
    """Translate canonical auth headers into BackendAnalyticsClient kwargs."""
    api_key = auth_headers.get("X-API-Key")
    if api_key:
        return {"api_key": api_key}

    auth_header = auth_headers.get("Authorization", "")
    scheme, separator, token = auth_header.partition(" ")
    if separator and scheme.lower() == "bearer":
        jwt_token = token.strip()
        if jwt_token:
            return {"jwt_token": jwt_token}

    return {}


def _configured_jwt_without_api_key(api_key_fallback: str | None = None) -> bool:
    if api_key_fallback or os.getenv("TRAIGENT_API_KEY"):
        return False
    if os.getenv("TRAIGENT_JWT_TOKEN"):
        return True

    from traigent.cloud.credential_manager import CredentialManager

    credentials = CredentialManager.get_credentials()
    if credentials.get("api_key"):
        return False
    jwt_token = credentials.get("jwt_token")
    return isinstance(jwt_token, str) and bool(jwt_token.strip())


async def resolve_analytics_read_client_credentials(
    auth: AuthManager | Any | None = None,
    *,
    api_key_fallback: str | None = None,
) -> dict[str, str]:
    """Resolve analytics read-client credentials through AuthManager.

    Returns kwargs for :class:`traigent.cloud.analytics_client.BackendAnalyticsClient`.
    The auth manager remains the authority for credential precedence and header
    construction; this helper only converts its emitted headers back into the
    explicit constructor shape required by the low-level analytics HTTP client.
    """
    if auth is None:
        auth = AuthManager(api_key=api_key_fallback)

    if _configured_jwt_without_api_key(api_key_fallback):
        auth_result = await auth.authenticate(mode=AuthMode.JWT_TOKEN)
        if not auth_result.success:
            raise InvalidCredentialsError(
                auth_result.error_message
                or "JWT authentication failed; refusing to create analytics client."
            )

    auth_headers = await auth.get_headers(target="backend")
    credential_kwargs = analytics_credential_kwargs_from_auth_headers(auth_headers)
    if not credential_kwargs:
        raise InvalidCredentialsError(
            "No usable analytics credential resolved; refusing to create analytics client."
        )
    return credential_kwargs
