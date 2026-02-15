"""OpenID Connect authentication provider implementation."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from traigent.utils.exceptions import AuthenticationError
from traigent.utils.logging import get_logger

from .helpers import sanitize_email, sanitize_roles, sanitize_string
from .models import User

logger = get_logger(__name__)

# Optional JWT dependency
try:
    import jwt
    from jwt import PyJWKClient

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None
    PyJWKClient = None


class OIDCAuthProvider:
    """OpenID Connect authentication provider with full implementation."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Initialize OIDC provider with settings.

        Args:
            settings: OIDC configuration including client credentials and endpoints
        """
        if not JWT_AVAILABLE:
            raise ImportError(
                "PyJWT is required for OIDC authentication. "
                "Install it with: pip install pyjwt[crypto]"
            )

        self.client_id = settings.get("client_id")
        self.client_secret = settings.get("client_secret")
        self.issuer = settings.get("issuer")
        self.jwks_uri = settings.get("jwks_uri")
        self.authorization_endpoint = settings.get("authorization_endpoint")
        self.token_endpoint = settings.get("token_endpoint")
        self.userinfo_endpoint = settings.get("userinfo_endpoint")

        self.allowed_algorithms = settings.get("allowed_algorithms", ["RS256"])
        self.max_token_age = settings.get("max_token_age", 3600)
        self.require_https = settings.get("require_https", True)

        if not all([self.client_id, self.issuer, self.jwks_uri]):
            raise ValueError("OIDC settings missing required configuration")

        if self.require_https:
            for url in [
                self.jwks_uri,
                self.authorization_endpoint,
                self.token_endpoint,
                self.userinfo_endpoint,
            ]:
                if url and not url.startswith("https://"):
                    raise ValueError(f"OIDC endpoint must use HTTPS: {url}")

        self.jwks_client = PyJWKClient(self.jwks_uri)
        self._revoked_tokens: set[str] = set()
        self._token_cache: dict[str, dict[str, Any]] = {}

    def authenticate_oidc(self, id_token: str) -> User | None:
        """Authenticate user via OIDC ID token."""
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(id_token)
            token_hash = hashlib.sha256(id_token.encode()).hexdigest()

            if token_hash in self._revoked_tokens:
                logger.warning("Attempted to use revoked token")
                return None

            if token_hash in self._token_cache:
                cached = self._token_cache[token_hash]
                if datetime.now(UTC) < cached["expires_at"]:
                    claims = cached["claims"]
                else:
                    del self._token_cache[token_hash]
            else:
                claims = jwt.decode(
                    id_token,
                    signing_key.key,
                    algorithms=self.allowed_algorithms,
                    audience=self.client_id,
                    issuer=self.issuer,
                    options={
                        "verify_exp": True,
                        "verify_aud": True,
                        "verify_iss": True,
                        "verify_iat": True,
                        "require": ["exp", "iat", "iss", "sub"],
                    },
                )

                iat = claims.get("iat", 0)
                if time.time() - iat > self.max_token_age:
                    logger.warning("Token too old")
                    return None

                self._token_cache[token_hash] = {
                    "claims": claims,
                    "expires_at": datetime.now(UTC) + timedelta(seconds=300),
                }

                if len(self._token_cache) > 1000:
                    oldest = sorted(
                        self._token_cache.items(), key=lambda x: x[1]["expires_at"]
                    )[:500]
                    for key, _ in oldest:
                        del self._token_cache[key]

            if "sub" not in claims:
                logger.warning("OIDC token missing 'sub' claim")
                return None

            user_id = sanitize_string(claims["sub"], max_length=255)
            username = sanitize_string(
                claims.get("preferred_username", claims.get("name", claims["sub"])),
                max_length=32,
            )
            email = sanitize_email(
                claims.get("email", f"{claims['sub']}@oidc"),
                default_domain="oidc.local",
            )
            roles = sanitize_roles(claims.get("roles", claims.get("groups", ["user"])))

            user = User(
                user_id=user_id,
                username=username,
                email=email,
                roles=roles,
                metadata={
                    "oidc_claims": self._sanitize_claims(claims),
                    "authenticated_at": datetime.now(UTC).isoformat(),
                    "token_issued_at": datetime.fromtimestamp(
                        claims.get("iat", 0), tz=UTC
                    ).isoformat(),
                    "token_expires_at": datetime.fromtimestamp(
                        claims.get("exp", 0), tz=UTC
                    ).isoformat(),
                    "auth_method": "oidc",
                    "token_hash": token_hash[:16],
                },
            )

            logger.info(f"OIDC authentication successful for user: {user.username}")
            return user

        except jwt.ExpiredSignatureError:
            logger.warning("OIDC token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid OIDC token: {e}")
            return None
        except Exception as e:
            logger.error(f"OIDC authentication error: {e}")
            raise AuthenticationError(f"OIDC authentication failed: {e}") from None

    def verify_access_token(self, access_token: str) -> dict[str, Any] | None:
        """Verify an OIDC access token."""
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(access_token)
            claims = jwt.decode(
                access_token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.client_id,
                options={"verify_exp": True},
            )
            return cast(dict[str, Any] | None, claims)
        except Exception as e:
            logger.warning(f"Access token verification failed: {e}")
            return None

    def revoke_token(self, token: str) -> None:
        """Revoke a token to prevent further use."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self._revoked_tokens.add(token_hash)
        if len(self._revoked_tokens) > 10000:
            self._revoked_tokens = set(list(self._revoked_tokens)[-5000:])

    def _sanitize_claims(self, claims: dict[str, Any]) -> dict[str, Any]:
        """Sanitize OIDC claims dictionary."""
        if not isinstance(claims, dict):
            return {}
        safe_claims = {}
        safe_keys = [
            "sub",
            "iss",
            "aud",
            "exp",
            "iat",
            "nbf",
            "email",
            "name",
            "given_name",
            "family_name",
            "preferred_username",
            "locale",
        ]
        for key in safe_keys:
            if key in claims:
                value = claims[key]
                if isinstance(value, (str, int, float, bool)):
                    safe_claims[key] = value
        return safe_claims
