"""Unit tests for traigent.security.auth.oidc.

Tests for OpenID Connect authentication provider including JWT validation,
token caching, claims sanitization, and token revocation.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from traigent.security.auth.models import User
from traigent.utils.exceptions import AuthenticationError


class TestOIDCAuthProviderImport:
    """Tests for OIDC provider import behavior."""

    def test_import_without_jwt_raises_error(self) -> None:
        """Test that importing without PyJWT raises ImportError."""
        with patch.dict("sys.modules", {"jwt": None}):
            with patch("traigent.security.auth.oidc.JWT_AVAILABLE", False):
                from traigent.security.auth.oidc import OIDCAuthProvider

                with pytest.raises(ImportError, match="PyJWT is required"):
                    OIDCAuthProvider({})


class TestOIDCAuthProviderInitialization:
    """Tests for OIDC provider initialization and configuration."""

    @pytest.fixture
    def valid_settings(self) -> dict[str, str]:
        """Create valid OIDC settings."""
        return {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "issuer": "https://issuer.example.com",
            "jwks_uri": "https://issuer.example.com/.well-known/jwks.json",
            "authorization_endpoint": "https://issuer.example.com/authorize",
            "token_endpoint": "https://issuer.example.com/token",
            "userinfo_endpoint": "https://issuer.example.com/userinfo",
        }

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_with_valid_settings(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test successful initialization with valid settings."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        provider = OIDCAuthProvider(valid_settings)

        assert provider.client_id == "test-client-id"
        assert provider.client_secret == "test-client-secret"
        assert provider.issuer == "https://issuer.example.com"
        assert provider.jwks_uri == "https://issuer.example.com/.well-known/jwks.json"
        assert provider.allowed_algorithms == ["RS256"]
        assert provider.max_token_age == 3600
        assert provider.require_https is True
        mock_jwks_client.assert_called_once_with(provider.jwks_uri)

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_with_custom_algorithms(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization with custom allowed algorithms."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        valid_settings["allowed_algorithms"] = ["RS256", "RS384"]
        provider = OIDCAuthProvider(valid_settings)

        assert provider.allowed_algorithms == ["RS256", "RS384"]

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_with_custom_token_age(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization with custom max token age."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        valid_settings["max_token_age"] = 7200
        provider = OIDCAuthProvider(valid_settings)

        assert provider.max_token_age == 7200

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_with_http_disabled(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization with HTTPS requirement disabled."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        valid_settings["require_https"] = False
        valid_settings["jwks_uri"] = "http://issuer.example.com/jwks.json"
        provider = OIDCAuthProvider(valid_settings)

        assert provider.require_https is False
        assert provider.jwks_uri == "http://issuer.example.com/jwks.json"

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_missing_client_id(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization fails without client_id."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        del valid_settings["client_id"]

        with pytest.raises(ValueError, match="missing required configuration"):
            OIDCAuthProvider(valid_settings)

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_missing_issuer(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization fails without issuer."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        del valid_settings["issuer"]

        with pytest.raises(ValueError, match="missing required configuration"):
            OIDCAuthProvider(valid_settings)

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_missing_jwks_uri(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization fails without jwks_uri."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        del valid_settings["jwks_uri"]

        with pytest.raises(ValueError, match="missing required configuration"):
            OIDCAuthProvider(valid_settings)

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_http_jwks_uri_fails_with_https_required(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization fails with HTTP jwks_uri when HTTPS required."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        valid_settings["jwks_uri"] = "http://issuer.example.com/jwks.json"

        with pytest.raises(ValueError, match="must use HTTPS"):
            OIDCAuthProvider(valid_settings)

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_http_authorization_endpoint_fails(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization fails with HTTP authorization endpoint."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        valid_settings["authorization_endpoint"] = "http://issuer.example.com/authorize"

        with pytest.raises(ValueError, match="must use HTTPS"):
            OIDCAuthProvider(valid_settings)

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_http_token_endpoint_fails(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization fails with HTTP token endpoint."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        valid_settings["token_endpoint"] = "http://issuer.example.com/token"

        with pytest.raises(ValueError, match="must use HTTPS"):
            OIDCAuthProvider(valid_settings)

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_http_userinfo_endpoint_fails(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization fails with HTTP userinfo endpoint."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        valid_settings["userinfo_endpoint"] = "http://issuer.example.com/userinfo"

        with pytest.raises(ValueError, match="must use HTTPS"):
            OIDCAuthProvider(valid_settings)

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_initialization_allows_none_optional_endpoints(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test initialization allows None for optional endpoints."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        valid_settings["authorization_endpoint"] = None
        valid_settings["token_endpoint"] = None
        valid_settings["userinfo_endpoint"] = None

        provider = OIDCAuthProvider(valid_settings)

        assert provider.authorization_endpoint is None
        assert provider.token_endpoint is None
        assert provider.userinfo_endpoint is None


class TestOIDCAuthProviderAuthentication:
    """Tests for OIDC authentication methods."""

    @pytest.fixture
    def valid_settings(self) -> dict[str, str]:
        """Create valid OIDC settings."""
        return {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "issuer": "https://issuer.example.com",
            "jwks_uri": "https://issuer.example.com/.well-known/jwks.json",
        }

    @pytest.fixture
    def mock_signing_key(self) -> MagicMock:
        """Create mock signing key."""
        key = MagicMock()
        key.key = "mock-key"
        return key

    @pytest.fixture
    def valid_claims(self) -> dict[str, any]:
        """Create valid JWT claims."""
        now = int(time.time())
        return {
            "sub": "user-12345",
            "iss": "https://issuer.example.com",
            "aud": "test-client-id",
            "exp": now + 3600,
            "iat": now,
            "email": "user@example.com",
            "name": "Test User",
            "preferred_username": "testuser",
            "roles": ["admin", "user"],
        }

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_success(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test successful OIDC authentication."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client
        mock_jwt.decode.return_value = valid_claims

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("valid.jwt.token")

        assert user is not None
        assert isinstance(user, User)
        assert user.user_id == "user-12345"
        assert user.username == "testuser"
        assert user.email == "user@example.com"
        assert "admin" in user.roles
        assert "user" in user.roles
        assert user.metadata["auth_method"] == "oidc"

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_with_cached_token(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test authentication uses cached token claims."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client
        mock_jwt.decode.return_value = valid_claims

        provider = OIDCAuthProvider(valid_settings)

        # First authentication
        user1 = provider.authenticate_oidc("valid.jwt.token")
        assert user1 is not None

        # Second authentication with same token (should use cache)
        user2 = provider.authenticate_oidc("valid.jwt.token")
        assert user2 is not None

        # JWT decode should be called only once
        assert mock_jwt.decode.call_count == 1

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_cache_not_used_when_expired(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test authentication removes expired cache but doesn't re-decode with same token."""
        import jwt as real_jwt

        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client
        mock_jwt.decode.return_value = valid_claims
        # Set exception classes for proper handling
        mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
        mock_jwt.InvalidTokenError = real_jwt.InvalidTokenError

        provider = OIDCAuthProvider(valid_settings)

        # First authentication
        token = "valid.jwt.token"
        user1 = provider.authenticate_oidc(token)
        assert user1 is not None
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        assert token_hash in provider._token_cache

        # Expire the cache entry
        provider._token_cache[token_hash]["expires_at"] = datetime.now(UTC) - timedelta(
            seconds=1
        )

        # Authenticate with a different token to verify cache expiry logic
        token2 = "another.jwt.token"
        user2 = provider.authenticate_oidc(token2)
        assert user2 is not None
        # Expired entry should be removed
        # (Note: entry gets deleted but since token2 is different, it creates new cache entry)
        assert mock_jwt.decode.call_count == 2

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_revoked_token(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
    ) -> None:
        """Test authentication fails for revoked token."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client

        provider = OIDCAuthProvider(valid_settings)

        token = "valid.jwt.token"
        provider.revoke_token(token)

        user = provider.authenticate_oidc(token)
        assert user is None

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_missing_sub_claim(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test authentication fails when sub claim is missing."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client

        del valid_claims["sub"]
        mock_jwt.decode.return_value = valid_claims

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("valid.jwt.token")

        assert user is None

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_token_too_old(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test authentication fails when token is too old."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client

        # Set iat to be older than max_token_age
        old_time = int(time.time()) - 7200
        valid_claims["iat"] = old_time
        mock_jwt.decode.return_value = valid_claims

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("valid.jwt.token")

        assert user is None

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_expired_signature(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
    ) -> None:
        """Test authentication handles expired signature gracefully."""
        import jwt as real_jwt

        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client
        mock_jwt.decode.side_effect = real_jwt.ExpiredSignatureError("Token expired")
        mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("expired.jwt.token")

        assert user is None

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_invalid_token(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
    ) -> None:
        """Test authentication handles invalid token gracefully."""
        import jwt as real_jwt

        # Setup exception classes before they're used
        import traigent.security.auth.oidc as oidc_module
        from traigent.security.auth.oidc import OIDCAuthProvider

        oidc_module.jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
        oidc_module.jwt.InvalidTokenError = real_jwt.InvalidTokenError

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client
        mock_jwt.decode.side_effect = real_jwt.InvalidTokenError("Invalid token")

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("invalid.jwt.token")

        assert user is None

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_generic_exception_raises_auth_error(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
    ) -> None:
        """Test authentication raises AuthenticationError on unexpected exceptions."""
        import jwt as real_jwt

        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.side_effect = RuntimeError(
            "Unexpected error"
        )
        mock_jwks_client_class.return_value = mock_jwks_client
        # Need to set these for the exception handler
        mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
        mock_jwt.InvalidTokenError = real_jwt.InvalidTokenError

        provider = OIDCAuthProvider(valid_settings)

        with pytest.raises(AuthenticationError, match="OIDC authentication failed"):
            provider.authenticate_oidc("valid.jwt.token")

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_cache_eviction(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test token cache evicts oldest entries when limit exceeded."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client
        mock_jwt.decode.return_value = valid_claims

        provider = OIDCAuthProvider(valid_settings)

        # Add 1001 tokens to trigger eviction
        for i in range(1001):
            token = f"token-{i}"
            provider.authenticate_oidc(token)

        # Cache should be reduced to 500 entries after eviction
        assert len(provider._token_cache) <= 501

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_uses_name_when_no_preferred_username(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test authentication uses name claim when preferred_username missing."""
        import jwt as real_jwt

        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client

        del valid_claims["preferred_username"]
        # Use a valid username format in the name field
        valid_claims["name"] = "testuser123"
        mock_jwt.decode.return_value = valid_claims
        # Set exception classes
        mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
        mock_jwt.InvalidTokenError = real_jwt.InvalidTokenError

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("valid.jwt.token")

        assert user is not None
        assert user.username is not None

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_uses_sub_when_no_username_or_name(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test authentication uses sub claim when no username or name available."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client

        del valid_claims["preferred_username"]
        del valid_claims["name"]
        mock_jwt.decode.return_value = valid_claims

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("valid.jwt.token")

        assert user is not None
        assert user.username is not None

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_default_email_when_missing(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test authentication generates default email when not provided."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client

        del valid_claims["email"]
        mock_jwt.decode.return_value = valid_claims

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("valid.jwt.token")

        assert user is not None
        assert "@oidc" in user.email

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_uses_groups_when_no_roles(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test authentication uses groups claim when roles not available."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client

        del valid_claims["roles"]
        valid_claims["groups"] = ["developers", "admins"]
        mock_jwt.decode.return_value = valid_claims

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("valid.jwt.token")

        assert user is not None
        assert "developers" in user.roles or "admins" in user.roles

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_authenticate_oidc_default_user_role_when_no_roles_or_groups(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_claims: dict[str, any],
    ) -> None:
        """Test authentication defaults to user role when no roles or groups."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client

        del valid_claims["roles"]
        mock_jwt.decode.return_value = valid_claims

        provider = OIDCAuthProvider(valid_settings)
        user = provider.authenticate_oidc("valid.jwt.token")

        assert user is not None
        assert "user" in user.roles


class TestOIDCAuthProviderAccessToken:
    """Tests for OIDC access token verification."""

    @pytest.fixture
    def valid_settings(self) -> dict[str, str]:
        """Create valid OIDC settings."""
        return {
            "client_id": "test-client-id",
            "issuer": "https://issuer.example.com",
            "jwks_uri": "https://issuer.example.com/.well-known/jwks.json",
        }

    @pytest.fixture
    def mock_signing_key(self) -> MagicMock:
        """Create mock signing key."""
        key = MagicMock()
        key.key = "mock-key"
        return key

    @pytest.fixture
    def valid_access_claims(self) -> dict[str, any]:
        """Create valid access token claims."""
        now = int(time.time())
        return {
            "sub": "user-12345",
            "aud": "test-client-id",
            "exp": now + 3600,
            "scope": "read write",
        }

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_verify_access_token_success(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
        valid_access_claims: dict[str, any],
    ) -> None:
        """Test successful access token verification."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client
        mock_jwt.decode.return_value = valid_access_claims

        provider = OIDCAuthProvider(valid_settings)
        claims = provider.verify_access_token("valid.access.token")

        assert claims is not None
        assert claims["sub"] == "user-12345"
        assert claims["scope"] == "read write"

    @patch("traigent.security.auth.oidc.PyJWKClient")
    @patch("traigent.security.auth.oidc.jwt")
    def test_verify_access_token_failure(
        self,
        mock_jwt: MagicMock,
        mock_jwks_client_class: MagicMock,
        valid_settings: dict[str, str],
        mock_signing_key: MagicMock,
    ) -> None:
        """Test access token verification failure."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client_class.return_value = mock_jwks_client
        mock_jwt.decode.side_effect = Exception("Invalid token")

        provider = OIDCAuthProvider(valid_settings)
        claims = provider.verify_access_token("invalid.access.token")

        assert claims is None


class TestOIDCAuthProviderTokenRevocation:
    """Tests for OIDC token revocation."""

    @pytest.fixture
    def valid_settings(self) -> dict[str, str]:
        """Create valid OIDC settings."""
        return {
            "client_id": "test-client-id",
            "issuer": "https://issuer.example.com",
            "jwks_uri": "https://issuer.example.com/.well-known/jwks.json",
        }

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_revoke_token_adds_to_revoked_set(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test revoking token adds it to revoked set."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        provider = OIDCAuthProvider(valid_settings)
        token = "token-to-revoke"
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        provider.revoke_token(token)

        assert token_hash in provider._revoked_tokens

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_revoke_token_evicts_old_entries(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test token revocation evicts old entries when limit exceeded."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        provider = OIDCAuthProvider(valid_settings)

        # Add 10001 revoked tokens to trigger eviction
        for i in range(10001):
            provider.revoke_token(f"token-{i}")

        # Set should be reduced to 5000 entries
        assert len(provider._revoked_tokens) == 5000


class TestOIDCAuthProviderClaimsSanitization:
    """Tests for OIDC claims sanitization."""

    @pytest.fixture
    def valid_settings(self) -> dict[str, str]:
        """Create valid OIDC settings."""
        return {
            "client_id": "test-client-id",
            "issuer": "https://issuer.example.com",
            "jwks_uri": "https://issuer.example.com/.well-known/jwks.json",
        }

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_sanitize_claims_keeps_safe_keys(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test claims sanitization keeps safe keys."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        provider = OIDCAuthProvider(valid_settings)
        claims = {
            "sub": "user-123",
            "email": "user@example.com",
            "name": "Test User",
            "unsafe_key": "should-be-removed",
        }

        sanitized = provider._sanitize_claims(claims)

        assert sanitized["sub"] == "user-123"
        assert sanitized["email"] == "user@example.com"
        assert sanitized["name"] == "Test User"
        assert "unsafe_key" not in sanitized

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_sanitize_claims_filters_non_scalar_values(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test claims sanitization filters non-scalar values."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        provider = OIDCAuthProvider(valid_settings)
        claims = {
            "sub": "user-123",
            "email": {"nested": "object"},
            "name": ["list", "value"],
        }

        sanitized = provider._sanitize_claims(claims)

        assert sanitized["sub"] == "user-123"
        assert "email" not in sanitized
        assert "name" not in sanitized

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_sanitize_claims_handles_non_dict_input(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test claims sanitization handles non-dict input."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        provider = OIDCAuthProvider(valid_settings)

        sanitized = provider._sanitize_claims("not-a-dict")
        assert sanitized == {}

        sanitized = provider._sanitize_claims(None)
        assert sanitized == {}

        sanitized = provider._sanitize_claims([])
        assert sanitized == {}

    @patch("traigent.security.auth.oidc.PyJWKClient")
    def test_sanitize_claims_keeps_all_safe_claim_types(
        self, mock_jwks_client: MagicMock, valid_settings: dict[str, str]
    ) -> None:
        """Test claims sanitization preserves all scalar types."""
        from traigent.security.auth.oidc import OIDCAuthProvider

        provider = OIDCAuthProvider(valid_settings)
        claims = {
            "sub": "user-123",
            "iat": 1234567890,
            "exp": 1234571490,
            "nbf": 1234567890,
            "email": "user@example.com",
            "locale": "en-US",
        }

        sanitized = provider._sanitize_claims(claims)

        assert sanitized["sub"] == "user-123"
        assert sanitized["iat"] == 1234567890
        assert sanitized["exp"] == 1234571490
        assert sanitized["nbf"] == 1234567890
        assert sanitized["email"] == "user@example.com"
        assert sanitized["locale"] == "en-US"
