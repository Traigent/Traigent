"""
Tests for authentication and authorization systems

This test file covers the actual implemented authentication components:
- APIKey management (traigent.cloud.auth)
- APIKeyManager for provider keys (traigent.config.api_keys)
- SSO providers (traigent.security.auth)
"""

import warnings
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from traigent.cloud.auth import APIKey
from traigent.config.api_keys import APIKeyManager
from traigent.security.auth import OIDCAuthProvider, SAMLAuthProvider

try:
    from tests.utils.isolation import TestIsolationMixin
except ImportError:
    # Fallback if running from different directory
    class TestIsolationMixin:
        def setup_method(self, method):
            """No-op fallback when isolation module unavailable."""


class TestAPIKey:
    """Test APIKey class for cloud authentication"""

    def test_api_key_creation(self):
        """Test API key creation with timezone-aware datetime"""
        # Use timezone-aware datetime to match the implementation
        api_key = APIKey(
            key="key-test-123",
            name="Test Key",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
        )

        assert api_key.key == "key-test-123"
        assert api_key.name == "Test Key"
        assert api_key.is_valid()
        assert api_key.has_permission("optimize")

    def test_api_key_expiration(self):
        """Test API key expiration with timezone-aware datetime"""
        # Create an expired key using timezone-aware datetime
        api_key = APIKey(
            key="key-expired-123",
            name="Expired Key",
            created_at=datetime.now(UTC) - timedelta(days=2),
            expires_at=datetime.now(UTC) - timedelta(days=1),
        )

        assert not api_key.is_valid()  # Should be invalid due to expiration

    def test_api_key_permissions(self):
        """Test API key permissions"""
        api_key = APIKey(
            key="key-test-123",
            name="Test Key",
            created_at=datetime.now(UTC),
            permissions={"optimize": True, "analytics": False},
        )

        assert api_key.has_permission("optimize")
        assert not api_key.has_permission("analytics")
        assert not api_key.has_permission("billing")  # default false

    def test_api_key_usage_limits(self):
        """Test usage limits"""
        api_key = APIKey(
            key="key-limited-123",
            name="Limited Key",
            created_at=datetime.now(UTC),
            usage_limit=100,
        )

        assert api_key.usage_limit == 100
        assert api_key.is_valid()

    def test_api_key_no_expiration(self):
        """Test API key without expiration date"""
        api_key = APIKey(
            key="key-noexpire-123",
            name="No Expire Key",
            created_at=datetime.now(UTC),
            expires_at=None,  # No expiration
        )

        assert api_key.is_valid()  # Should be valid without expiration


class TestAPIKeyManager(TestIsolationMixin):
    """Test APIKeyManager class for managing provider API keys"""

    def setup_method(self, method):
        """Set up test method."""
        # Call parent setup for isolation
        super().setup_method(method)
        # Create a fresh instance for each test
        self.key_manager = APIKeyManager()
        with self.key_manager._lock:
            self.key_manager._keys.clear()
            self.key_manager._warned = False

    def test_set_and_get_api_key(self):
        """Test setting and getting API keys"""
        from unittest.mock import patch

        # Mock environment to ensure test isolation
        with patch.dict("os.environ", {}, clear=True):
            # Set API keys for different providers
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress warnings for test
                self.key_manager.set_api_key("openai", "key-test-123", "code")
                self.key_manager.set_api_key(
                    "anthropic", "key-anthropic-456", "env"
                )  # noqa: S106 - test credential

            # Get API keys
            openai_key = self.key_manager.get_api_key("openai")
            anthropic_key = self.key_manager.get_api_key("anthropic")
            missing_key = self.key_manager.get_api_key("nonexistent")

            assert openai_key == "key-test-123"
            assert anthropic_key == "key-anthropic-456"
            assert missing_key is None

    def test_environment_priority(self):
        """Test that environment variables take priority"""
        from unittest.mock import patch

        # Set a key via code (suppress warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.key_manager.set_api_key("openai", "code-key", "code")

        # Mock environment variable - should take priority
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            # Environment variable should take priority over stored key
            key = self.key_manager.get_api_key("openai")
            assert key == "env-key"  # Environment takes priority

        # Without env var, should return the stored key
        with patch.dict("os.environ", {}, clear=True):
            key = self.key_manager.get_api_key("openai")
            assert key == "code-key"

    def test_security_warnings(self):
        """Test that security warnings are issued for hardcoded keys"""
        # Should trigger warning for hardcoded keys
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.key_manager.set_api_key("test", "hardcoded-key", "code")

            assert len(w) == 1
            assert "API keys detected in code" in str(w[0].message)

        # Should not trigger warning for env keys
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.key_manager.set_api_key("test2", "env-key", "env")

            assert len(w) == 0

    def test_repr_security(self):
        """Test that API keys are not exposed in string representation"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.key_manager.set_api_key("openai", "secret-key-123", "code")

        repr_str = repr(self.key_manager)
        str_str = str(self.key_manager)

        # Keys should not be exposed
        assert "secret-key-123" not in repr_str
        assert "secret-key-123" not in str_str
        # Should show count
        assert "1 keys" in repr_str


class TestSSOProviders:
    """Test SSO provider implementations"""

    def test_saml_provider_creation(self):
        """Test SAML provider creation"""
        settings = {
            "sp": {
                "entityId": "test_entity",
                "assertionConsumerService": {
                    "url": "https://example.com/sso/acs",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
            },
            "idp": {
                "entityId": "https://idp.example.com",
                "singleSignOnService": {
                    "url": "https://idp.example.com/sso",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "x509cert": "cert_data",
            },
        }

        # This will skip if python3-saml is not installed
        try:
            provider = SAMLAuthProvider(settings)
            assert provider.settings == settings
        except ImportError:
            pytest.skip("python3-saml not available")

    def test_oidc_provider_creation(self):
        """Test OIDC provider creation"""
        settings = {
            "client_id": "test_client",
            "client_secret": "test_secret",
            "issuer": "https://accounts.google.com",
            "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
            "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
        }

        # This will skip if PyJWT is not installed
        try:
            provider = OIDCAuthProvider(settings)
            assert provider.client_id == "test_client"
            assert provider.client_secret == "test_secret"
            assert provider.issuer == "https://accounts.google.com"
        except ImportError:
            pytest.skip("PyJWT not available")

    def test_oidc_provider_validation(self):
        """Test OIDC provider validates required settings"""
        # Missing required fields should raise error
        invalid_settings = {
            "client_id": "test_client",
            # Missing client_secret and other required fields
        }

        try:
            with pytest.raises((ValueError, KeyError, AttributeError)):
                OIDCAuthProvider(invalid_settings)
        except ImportError:
            pytest.skip("PyJWT not available")

    def test_saml_response_injected_into_onelogin(self, monkeypatch):
        """Ensure the SAML response is forwarded to the underlying toolkit."""
        import traigent.security.auth as security_auth

        class FakeSamlAuth:
            last_response = None

            def __init__(self, request, settings):
                assert request["post_data"]["SAMLResponse"] == "encoded-assertion"
                self._request = request

            def process_response(self):
                FakeSamlAuth.last_response = self._request["post_data"]["SAMLResponse"]

            def is_authenticated(self):
                return True

            def get_attributes(self):
                return {"email": ["alice@example.com"], "roles": ["admin"]}

            def get_nameid(self):
                return "user-123"

            def get_nameid_format(self):
                return "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified"

            def get_session_index(self):
                return "session-xyz"

        settings = {
            "sp": {
                "entityId": "test_entity",
                "assertionConsumerService": {
                    "url": "https://example.com/sso/acs",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
            },
            "idp": {
                "entityId": "https://idp.example.com",
                "singleSignOnService": {
                    "url": "https://idp.example.com/sso",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "x509cert": "-----BEGIN CERTIFICATE-----FAKE-----END CERTIFICATE-----",
            },
        }

        import traigent.security.auth.saml as saml_module

        monkeypatch.setattr(saml_module, "SAML_AVAILABLE", True)
        monkeypatch.setattr(saml_module, "OneLogin_Saml2_Auth", FakeSamlAuth)

        provider = security_auth.SAMLAuthProvider(settings)
        request = {
            "https": "on",
            "http_host": "example.com",
            "script_name": "/acs",
            "post_data": {},
        }

        user = provider.authenticate_saml(request, "encoded-assertion")

        assert user is not None
        assert user.email == "alice@example.com"
        assert FakeSamlAuth.last_response == "encoded-assertion"


class TestSMSAuthProvider:
    """Tests for SMS-based MFA provider."""

    def test_sms_codes_not_stored_in_plaintext(self, monkeypatch):
        import traigent.security.auth as security_auth

        class DummyMessages:
            def __init__(self):
                self.last_kwargs = None

            def create(self, **kwargs):
                self.last_kwargs = kwargs
                return SimpleNamespace(sid="SM123")

        class DummyTwilioClient:
            def __init__(self, account_sid, auth_token):
                self.account_sid = account_sid
                self.auth_token = auth_token
                self.messages = DummyMessages()

        import traigent.security.auth.sms as sms_module

        monkeypatch.setattr(sms_module, "TWILIO_AVAILABLE", True)
        monkeypatch.setattr(sms_module, "TwilioClient", DummyTwilioClient)

        provider = security_auth.SMSAuthProvider(
            {
                "account_sid": "AC123",
                "auth_token": "secret-token",
                "from_number": "+15555550100",
            }
        )

        sid = provider.send_verification_code("+15555550101", "user-1")

        assert sid == "SM123"
        stored = provider._verification_codes["user-1"]
        assert "code" not in stored
        assert "code_hash" in stored
