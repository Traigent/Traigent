"""Unit tests for CredentialResolver class.

Tests for credential resolution, loading, caching, and encryption.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY REQ-SEC-010

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.auth import AuthCredentials, AuthMode, UnifiedAuthConfig
from traigent.cloud.credential_resolver import CredentialResolver


@pytest.fixture
def config() -> UnifiedAuthConfig:
    """Create test configuration."""
    return UnifiedAuthConfig(
        default_mode=AuthMode.API_KEY,
        cache_credentials=True,
        credentials_file="/tmp/test_credentials.json",
    )


@pytest.fixture
def resolver(config: UnifiedAuthConfig) -> CredentialResolver:
    """Create CredentialResolver instance."""
    return CredentialResolver(config)


@pytest.fixture
def valid_api_key() -> str:
    """Create a valid API key for testing."""
    # tg_ prefix + 61 alphanumeric chars = 64 total
    return "tg_" + "a" * 61


@pytest.fixture
def sample_credentials(valid_api_key: str) -> AuthCredentials:
    """Create sample credentials for testing."""
    return AuthCredentials(
        mode=AuthMode.API_KEY,
        api_key=valid_api_key,
        metadata={"source": "test"},
    )


class TestResolve:
    """Tests for resolve method."""

    @pytest.mark.asyncio
    async def test_resolve_returns_provided_credentials(
        self, resolver: CredentialResolver, sample_credentials: AuthCredentials
    ) -> None:
        """Test resolve returns provided credentials directly."""
        result = await resolver.resolve(sample_credentials, AuthMode.API_KEY)
        assert result is sample_credentials

    @pytest.mark.asyncio
    async def test_resolve_returns_dict_credentials(
        self, resolver: CredentialResolver
    ) -> None:
        """Test resolve returns dict credentials directly."""
        creds_dict = {"api_key": "test-key", "mode": "api_key"}
        result = await resolver.resolve(creds_dict, AuthMode.API_KEY)
        assert result is creds_dict

    @pytest.mark.asyncio
    async def test_resolve_loads_when_none_provided(
        self, resolver: CredentialResolver, sample_credentials: AuthCredentials
    ) -> None:
        """Test resolve loads credentials when none provided."""
        resolver._get_provided_credentials_fn = lambda: sample_credentials

        result = await resolver.resolve(None, AuthMode.API_KEY)
        assert result is sample_credentials


class TestLoadCredentials:
    """Tests for load_credentials method."""

    @pytest.mark.asyncio
    async def test_load_provided_credentials_first(
        self, resolver: CredentialResolver, sample_credentials: AuthCredentials
    ) -> None:
        """Test load_credentials returns provided credentials first."""
        resolver._get_provided_credentials_fn = lambda: sample_credentials
        resolver._get_api_key_token_fn = lambda: None
        set_token_mock = MagicMock()
        resolver._set_api_key_token_fn = set_token_mock

        result = await resolver.load_credentials(AuthMode.API_KEY)

        assert result is sample_credentials
        set_token_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_provided_credentials_with_existing_token(
        self, resolver: CredentialResolver, sample_credentials: AuthCredentials
    ) -> None:
        """Test load_credentials doesn't re-set token if already set."""
        resolver._get_provided_credentials_fn = lambda: sample_credentials
        resolver._get_api_key_token_fn = lambda: MagicMock()  # Token exists
        set_token_mock = MagicMock()
        resolver._set_api_key_token_fn = set_token_mock

        result = await resolver.load_credentials(AuthMode.API_KEY)

        assert result is sample_credentials
        set_token_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_uses_default_mode(
        self, resolver: CredentialResolver, config: UnifiedAuthConfig
    ) -> None:
        """Test load_credentials uses default mode when none provided."""
        resolver._get_provided_credentials_fn = lambda: None

        with patch.object(
            resolver, "load_cached", new_callable=AsyncMock
        ) as mock_cached:
            mock_cached.return_value = None
            with patch.object(
                resolver, "load_from_env", new_callable=AsyncMock
            ) as mock_env:
                mock_env.return_value = None

                await resolver.load_credentials(None)

                mock_env.assert_called_once_with(config.default_mode)

    @pytest.mark.asyncio
    async def test_load_increments_cache_hits(
        self, resolver: CredentialResolver, sample_credentials: AuthCredentials
    ) -> None:
        """Test load_credentials increments cache hits stat."""
        resolver._get_provided_credentials_fn = lambda: None
        increment_mock = MagicMock()
        resolver._increment_cache_hits_fn = increment_mock

        with patch.object(
            resolver, "load_cached", new_callable=AsyncMock
        ) as mock_cached:
            mock_cached.return_value = sample_credentials

            result = await resolver.load_credentials(AuthMode.API_KEY)

            assert result is sample_credentials
            increment_mock.assert_called_once()


class TestLoadCached:
    """Tests for load_cached method."""

    @pytest.mark.asyncio
    async def test_load_cached_returns_none_without_file(
        self, config: UnifiedAuthConfig
    ) -> None:
        """Test load_cached returns None when no credentials file configured."""
        config.credentials_file = None
        resolver = CredentialResolver(config)

        result = await resolver.load_cached()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_cached_handles_file_not_found(
        self, resolver: CredentialResolver
    ) -> None:
        """Test load_cached handles FileNotFoundError gracefully."""
        with patch(
            "traigent.security.crypto_utils.SecureFileManager.read_secure_file",
            side_effect=FileNotFoundError(),
        ):
            result = await resolver.load_cached()
            assert result is None

    @pytest.mark.asyncio
    async def test_load_cached_handles_other_errors(
        self, resolver: CredentialResolver
    ) -> None:
        """Test load_cached handles other exceptions gracefully."""
        with patch(
            "traigent.security.crypto_utils.SecureFileManager.read_secure_file",
            side_effect=ValueError("test error"),
        ):
            result = await resolver.load_cached()
            assert result is None


class TestLoadFromEnv:
    """Tests for load_from_env method."""

    @pytest.mark.asyncio
    async def test_load_api_key_from_credential_manager(
        self, resolver: CredentialResolver, valid_api_key: str
    ) -> None:
        """Test loading API key from credential manager."""
        set_token_mock = MagicMock()
        resolver._set_api_key_token_fn = set_token_mock

        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials"
        ) as mock_get:
            mock_get.return_value = {
                "api_key": valid_api_key,
                "source": "cli",
                "backend_url": "http://localhost:5000",
            }

            result = await resolver.load_from_env(AuthMode.API_KEY)

            assert result is not None
            assert result.api_key == valid_api_key
            assert result.metadata.get("source") == "cli"
            set_token_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_api_key_from_environment(
        self, resolver: CredentialResolver, valid_api_key: str
    ) -> None:
        """Test loading API key from environment variable."""
        set_token_mock = MagicMock()
        resolver._set_api_key_token_fn = set_token_mock

        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials",
            return_value=None,
        ):
            with patch(
                "traigent.cloud.credential_manager.CredentialManager.get_api_key",
                return_value=None,
            ):
                with patch.dict(os.environ, {"TRAIGENT_API_KEY": valid_api_key}):
                    result = await resolver.load_from_env(AuthMode.API_KEY)

                    assert result is not None
                    assert result.api_key == valid_api_key
                    assert result.metadata.get("source") == "environment"

    @pytest.mark.asyncio
    async def test_load_jwt_token_from_credential_manager(
        self, resolver: CredentialResolver
    ) -> None:
        """Test loading JWT token from credential manager."""
        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials"
        ) as mock_get:
            mock_get.return_value = {
                "jwt_token": "test-jwt-token",
                "refresh_token": "test-refresh-token",
                "source": "cli",
            }

            result = await resolver.load_from_env(AuthMode.JWT_TOKEN)

            assert result is not None
            assert result.jwt_token == "test-jwt-token"
            assert result.refresh_token == "test-refresh-token"

    @pytest.mark.asyncio
    async def test_load_oauth2_credentials(self, resolver: CredentialResolver) -> None:
        """Test loading OAuth2 credentials from environment."""
        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials",
            return_value=None,
        ):
            with patch.dict(
                os.environ,
                {
                    "TRAIGENT_CLIENT_ID": "test-client-id",
                    "TRAIGENT_CLIENT_SECRET": "test-secret",
                },
            ):
                result = await resolver.load_from_env(AuthMode.OAUTH2)

                assert result is not None
                assert result.client_id == "test-client-id"
                assert result.client_secret == "test-secret"

    @pytest.mark.asyncio
    async def test_load_service_to_service_credentials(
        self, resolver: CredentialResolver
    ) -> None:
        """Test loading service-to-service credentials."""
        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials",
            return_value=None,
        ):
            with patch.dict(os.environ, {"TRAIGENT_SERVICE_KEY": "test-service-key"}):
                result = await resolver.load_from_env(AuthMode.SERVICE_TO_SERVICE)

                assert result is not None
                assert result.service_key == "test-service-key"

    @pytest.mark.asyncio
    async def test_load_development_credentials(
        self, resolver: CredentialResolver
    ) -> None:
        """Test loading development credentials."""
        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials",
            return_value=None,
        ):
            result = await resolver.load_from_env(AuthMode.DEVELOPMENT)

            assert result is not None
            assert result.mode == AuthMode.DEVELOPMENT
            assert "dev_user" in result.metadata

    @pytest.mark.asyncio
    async def test_load_returns_none_when_no_credentials(
        self, resolver: CredentialResolver
    ) -> None:
        """Test load_from_env returns None when no credentials available."""
        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials",
            return_value=None,
        ):
            with patch(
                "traigent.cloud.credential_manager.CredentialManager.get_api_key",
                return_value=None,
            ):
                with patch.dict(os.environ, {}, clear=True):
                    # Clear relevant env vars
                    os.environ.pop("TRAIGENT_API_KEY", None)
                    os.environ.pop("OPTIGEN_API_KEY", None)

                    result = await resolver.load_from_env(AuthMode.API_KEY)
                    assert result is None

    @pytest.mark.asyncio
    async def test_load_handles_exceptions(self, resolver: CredentialResolver) -> None:
        """Test load_from_env handles exceptions gracefully."""
        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials",
            side_effect=Exception("test error"),
        ):
            result = await resolver.load_from_env(AuthMode.API_KEY)
            assert result is None


class TestCache:
    """Tests for cache method."""

    @pytest.mark.asyncio
    async def test_cache_does_nothing_without_file(
        self, config: UnifiedAuthConfig, sample_credentials: AuthCredentials
    ) -> None:
        """Test cache does nothing when no credentials file configured."""
        config.credentials_file = None
        resolver = CredentialResolver(config)

        # Should not raise
        await resolver.cache(sample_credentials)

    @pytest.mark.asyncio
    async def test_cache_writes_encrypted_credentials(
        self, resolver: CredentialResolver, sample_credentials: AuthCredentials
    ) -> None:
        """Test cache writes encrypted credentials to file."""
        with patch(
            "traigent.security.crypto_utils.SecureFileManager.write_secure_file"
        ) as mock_write:
            with patch.object(
                resolver, "encrypt", return_value={"encrypted": True}
            ) as mock_encrypt:
                await resolver.cache(sample_credentials)

                mock_encrypt.assert_called_once_with(sample_credentials)
                mock_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_handles_errors(
        self, resolver: CredentialResolver, sample_credentials: AuthCredentials
    ) -> None:
        """Test cache handles errors gracefully."""
        with patch(
            "traigent.security.crypto_utils.SecureFileManager.write_secure_file",
            side_effect=Exception("write error"),
        ):
            # Should not raise
            await resolver.cache(sample_credentials)


class TestEncryptDecrypt:
    """Tests for encrypt and decrypt methods."""

    def test_encrypt_creates_data_dict(
        self, resolver: CredentialResolver, sample_credentials: AuthCredentials
    ) -> None:
        """Test encrypt creates proper data dictionary."""
        with patch(
            "traigent.security.crypto_utils.get_credential_storage"
        ) as mock_storage:
            mock_crypto = MagicMock()
            mock_crypto.encrypt_credentials.return_value = {"encrypted": True}
            mock_storage.return_value = mock_crypto

            result = resolver.encrypt(sample_credentials)

            assert result == {"encrypted": True}
            mock_crypto.encrypt_credentials.assert_called_once()
            call_args = mock_crypto.encrypt_credentials.call_args[0][0]
            assert call_args["mode"] == sample_credentials.mode.value
            assert call_args["api_key"] == sample_credentials.api_key

    def test_decrypt_returns_decrypted_data(self, resolver: CredentialResolver) -> None:
        """Test decrypt returns decrypted data."""
        with patch(
            "traigent.security.crypto_utils.get_credential_storage"
        ) as mock_storage:
            mock_crypto = MagicMock()
            mock_crypto.decrypt_credentials.return_value = {
                "mode": "api_key",
                "api_key": "test-key",
            }
            mock_storage.return_value = mock_crypto

            result = resolver.decrypt({"encrypted": True})

            assert result["mode"] == "api_key"
            assert result["api_key"] == "test-key"


class TestCallbacks:
    """Tests for callback functionality."""

    def test_set_callbacks(self, resolver: CredentialResolver) -> None:
        """Test setting callback functions."""
        get_provided = MagicMock(return_value=None)
        set_token = MagicMock()
        get_token = MagicMock(return_value=None)
        increment_hits = MagicMock()

        resolver.set_callbacks(
            get_provided_credentials=get_provided,
            set_api_key_token=set_token,
            get_api_key_token=get_token,
            increment_cache_hits=increment_hits,
        )

        assert resolver._get_provided_credentials_fn is get_provided
        assert resolver._set_api_key_token_fn is set_token
        assert resolver._get_api_key_token_fn is get_token
        assert resolver._increment_cache_hits_fn is increment_hits

    def test_callbacks_only_set_when_provided(
        self, resolver: CredentialResolver
    ) -> None:
        """Test that only provided callbacks are set."""
        original_fn = resolver._get_provided_credentials_fn
        get_token = MagicMock()

        resolver.set_callbacks(get_api_key_token=get_token)

        assert resolver._get_provided_credentials_fn is original_fn
        assert resolver._get_api_key_token_fn is get_token


class TestExpiresAtParsing:
    """Tests for expires_at datetime parsing."""

    @pytest.mark.asyncio
    async def test_parses_iso_format_expires_at(
        self, resolver: CredentialResolver, valid_api_key: str
    ) -> None:
        """Test parsing ISO format expires_at from provided credentials."""
        expires_iso = "2025-12-31T23:59:59+00:00"
        creds = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key=valid_api_key,
            metadata={"source": "test", "expires_at": expires_iso},
        )
        resolver._get_provided_credentials_fn = lambda: creds
        resolver._get_api_key_token_fn = lambda: None
        set_token_mock = MagicMock()
        resolver._set_api_key_token_fn = set_token_mock

        await resolver.load_credentials(AuthMode.API_KEY)

        set_token_mock.assert_called_once()
        call_kwargs = set_token_mock.call_args
        assert call_kwargs[1]["expires_at"] is not None

    @pytest.mark.asyncio
    async def test_handles_invalid_expires_at_format(
        self, resolver: CredentialResolver, valid_api_key: str
    ) -> None:
        """Test handling invalid expires_at format."""
        creds = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key=valid_api_key,
            metadata={"source": "test", "expires_at": "invalid-date"},
        )
        resolver._get_provided_credentials_fn = lambda: creds
        resolver._get_api_key_token_fn = lambda: None
        set_token_mock = MagicMock()
        resolver._set_api_key_token_fn = set_token_mock

        await resolver.load_credentials(AuthMode.API_KEY)

        set_token_mock.assert_called_once()
        call_kwargs = set_token_mock.call_args
        assert call_kwargs[1]["expires_at"] is None
