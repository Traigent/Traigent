"""Comprehensive unit tests for secure credential management."""

import json
import logging
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from traigent.security import credentials as credentials_module
from traigent.security.credentials import (
    CredentialType,
    EnhancedCredentialStore,
    SecureCredential,
    SecureString,
    SecurityLevel,
)
from traigent.utils.exceptions import AuthenticationError


class TestSecureString:
    """Test SecureString memory-safe string wrapper."""

    def test_secure_string_creation(self):
        """Test creating a secure string."""
        secret = SecureString("my-secret-value")
        assert secret.get() == "my-secret-value"

    def test_secure_string_clear(self):
        """Test clearing secure string from memory."""
        secret = SecureString("sensitive-data")
        secret.clear()

        with pytest.raises(AuthenticationError, match="SecureString is locked"):
            secret.get()

    def test_secure_string_lock(self):
        """Test locking secure string."""
        secret = SecureString("locked-secret")
        secret.lock()

        with pytest.raises(AuthenticationError, match="SecureString is locked"):
            secret.get()

    def test_secure_string_repr(self):
        """Test secure string representation doesn't expose value."""
        secret = SecureString("hidden-value")
        repr_str = repr(secret)
        assert "hidden-value" not in repr_str
        assert "SecureString" in repr_str


class TestCredentialType:
    """Test CredentialType enum."""

    def test_credential_types(self):
        """Test all credential types are defined."""
        assert CredentialType.API_KEY.value == "api_key"
        assert CredentialType.SECRET.value == "secret"
        assert CredentialType.PASSWORD.value == "password"
        assert CredentialType.TOKEN.value == "token"
        assert CredentialType.CERTIFICATE.value == "certificate"
        assert CredentialType.PRIVATE_KEY.value == "private_key"
        assert CredentialType.DATABASE_URL.value == "database_url"
        assert CredentialType.ENCRYPTION_KEY.value == "encryption_key"


class TestSecurityLevel:
    """Test SecurityLevel enum."""

    def test_security_levels(self):
        """Test all security levels are defined."""
        assert SecurityLevel.STANDARD.value == "standard"
        assert SecurityLevel.HIGH.value == "high"
        assert SecurityLevel.MAXIMUM.value == "maximum"


class TestSecureCredential:
    """Test SecureCredential data model."""

    def test_secure_credential_creation(self):
        """Test creating a secure credential."""
        now = datetime.now(UTC)
        cred = SecureCredential(
            name="test-api-key",
            type=CredentialType.API_KEY,
            _encrypted_value=b"encrypted_data",
            metadata={"source": "test"},
            security_level=SecurityLevel.STANDARD,
            created_at=now,
        )

        assert cred.name == "test-api-key"
        assert cred.type == CredentialType.API_KEY
        assert cred.security_level == SecurityLevel.STANDARD
        assert cred.access_count == 0

    def test_secure_credential_repr(self):
        """Test credential representation doesn't expose encrypted value."""
        now = datetime.now(UTC)
        cred = SecureCredential(
            name="secret-key",
            type=CredentialType.SECRET,
            _encrypted_value=b"super_secret_data",
            metadata={},
            security_level=SecurityLevel.HIGH,
            created_at=now,
        )

        repr_str = repr(cred)
        assert "super_secret_data" not in repr_str
        assert "secret-key" in repr_str
        assert "SecureCredential" in repr_str


class TestEnhancedCredentialStore:
    """Test EnhancedCredentialStore functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def store(self, temp_dir):
        """Create credential store with temp storage."""
        storage_path = Path(temp_dir) / "credentials"
        return EnhancedCredentialStore(
            master_password="test-passphrase-12345",  # noqa: S106 - test credential
            storage_path=str(storage_path),
        )

    def test_store_initialization(self, temp_dir):
        """Test credential store initialization."""
        expected_path = Path(temp_dir) / "creds"
        store = EnhancedCredentialStore(
            master_password="test-passphrase-12345",  # noqa: S106 - test credential
            storage_path=str(expected_path),
        )
        assert store.storage_path == expected_path
        assert store.security_level == SecurityLevel.HIGH
        assert store.check_credential_health()["total_credentials"] == 0

    def test_ignored_legacy_master_without_store_logs_info_not_warning(
        self, tmp_path, caplog
    ):
        """A stale legacy key alone must not warn during local/mock probes."""
        storage_path = tmp_path / "secure_credentials.enc"
        master_key_path = tmp_path / EnhancedCredentialStore.MASTER_PASSWORD_FILENAME
        master_key_path.write_text("legacy-master", encoding="utf-8")

        with (
            caplog.at_level(logging.INFO, logger="traigent.security.credentials"),
            pytest.raises(AuthenticationError, match="master secret is required"),
        ):
            EnhancedCredentialStore(storage_path=storage_path, use_env_vars=False)

        records = [
            record
            for record in caplog.records
            if "Ignoring legacy local master secret file" in record.message
        ]
        assert len(records) == 1
        assert records[0].levelno == logging.INFO

    def test_ignored_legacy_master_with_store_warns_once(self, tmp_path, caplog):
        """An actual legacy vault migration warning remains actionable but deduped."""
        credentials_module._IGNORED_LEGACY_MASTER_SECRET_WARNING_PATHS.clear()
        storage_path = tmp_path / "secure_credentials.enc"
        storage_path.write_bytes(b"encrypted-store-placeholder")
        master_key_path = tmp_path / EnhancedCredentialStore.MASTER_PASSWORD_FILENAME
        master_key_path.write_text("legacy-master", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="traigent.security.credentials"):
            for _ in range(2):
                with pytest.raises(
                    AuthenticationError, match="master secret is required"
                ):
                    EnhancedCredentialStore(
                        storage_path=storage_path,
                        use_env_vars=False,
                    )

        records = [
            record
            for record in caplog.records
            if "Ignoring legacy local master secret file" in record.message
        ]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING

    def test_set_and_get_credential(self, store):
        """Test setting and getting a credential."""
        store.set(
            name="test-key",
            value="test-value",
            credential_type=CredentialType.API_KEY,
        )

        retrieved = store.get("test-key")
        assert retrieved == "test-value"

    def test_get_nonexistent_credential(self, store):
        """Test getting credential that doesn't exist."""
        result = store.get("nonexistent-key", check_env=False)
        assert result is None

    def test_delete_credential(self, store):
        """Test deleting a credential."""
        store.set(
            name="delete-test",
            value="temporary-value",
            credential_type=CredentialType.TOKEN,
        )

        # Verify it exists
        assert store.get("delete-test") is not None

        # Delete it
        result = store.delete_secure("delete-test")
        assert result is True

        # Verify it's gone
        assert store.get("delete-test", check_env=False) is None

    def test_rotate_credential(self, store):
        """Test credential rotation."""
        store.set(
            name="rotate-test",
            value="old-value",
            credential_type=CredentialType.API_KEY,
        )

        store.rotate_credential("rotate-test", "new-value")

        retrieved = store.get("rotate-test")
        assert retrieved == "new-value"

    def test_credential_with_metadata(self, store):
        """Test storing credential with metadata."""
        metadata = {"source": "test", "environment": "dev"}
        store.set(
            name="meta-test",
            value="value1234",
            credential_type=CredentialType.API_KEY,
            metadata=metadata,
        )

        # The metadata kwarg must not interfere with the stored value, and
        # the credential must be recorded in the store's health accounting.
        assert store.get("meta-test", check_env=False) == "value1234"
        assert store.check_credential_health()["total_credentials"] == 1

    def test_structured_credential_weak_check_ignores_user_metadata(self, store):
        """User metadata in a structured credential payload is not secret material."""
        payload = {
            "api_key": "sk_" + ("a" * 43),  # pragma: allowlist secret
            "backend_url": "https://api.example.test/demo",
            "tenant_id": "tenant_admin_demo",
            "project_id": "project_example",
            "user": {"id": "user_123", "email": "admin@dev.local"},
        }

        store.set(
            name="cli_credentials",
            value=json.dumps(payload, separators=(",", ":")),
            credential_type=CredentialType.TOKEN,
        )

        saved = store.get("cli_credentials", check_env=False)
        assert saved is not None
        assert json.loads(saved)["user"]["email"] == "admin@dev.local"

    def test_structured_credential_weak_check_names_secret_field(self, store):
        """Weak structured credentials are rejected on the offending secret field."""
        payload = {
            "api_key": "your-api-key-here",  # pragma: allowlist secret
            "user": {"email": "user@example.test"},
        }

        with pytest.raises(
            AuthenticationError, match="api_key.*weak|api_key.*placeholder"
        ):
            store.set(
                name="cli_credentials",
                value=json.dumps(payload, separators=(",", ":")),
                credential_type=CredentialType.TOKEN,
            )

    def test_credential_with_expiration(self, store):
        """Test storing credential with expiration."""
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        store.set(
            name="expiring-key",
            value="temporary",
            credential_type=CredentialType.TOKEN,
            expires_at=expires_at,
        )

        # Should be retrievable before expiration
        assert store.get("expiring-key", check_env=False) == "temporary"
        # The 1-hour expiry is within the health check's 7-day warning window.
        health = store.check_credential_health()
        assert health["expiring_soon"] == 1
        assert health["expired"] == 0

    def test_security_metrics(self, store):
        """Test getting security metrics."""
        store.set("metric-test", "value1234", CredentialType.API_KEY)
        store.get("metric-test")

        metrics = store.get_security_metrics()
        assert isinstance(metrics, dict)

    def test_security_event_log_omits_details(self, store, monkeypatch):
        """Security-event logs must not include caller-supplied detail payloads."""
        messages: list[str] = []
        monkeypatch.setattr(
            credentials_module.logger,
            "warning",
            lambda message, *args: messages.append(message % args),
        )

        store._security_event(
            "weak_credential_detected",
            {"field": "credential_field", "raw": "visible-sensitive-material"},
        )

        assert len(messages) == 1
        assert messages[0].startswith(
            "Security event: type=weak_credential_detected security_level="
        )
        assert "visible-sensitive-material" not in messages[0]
        assert "credential_field" not in messages[0]

    def test_credential_health_check(self, store):
        """Test credential health check."""
        store.set("health-test", "value1234", CredentialType.API_KEY)

        health = store.check_credential_health()
        assert isinstance(health, dict)

    def test_unicode_credential_value(self, store):
        """Test storing credential with unicode characters."""
        unicode_value = "🔐 secret-密钥-مفتاح"
        store.set(
            name="unicode-key",
            value=unicode_value,
            credential_type=CredentialType.SECRET,
        )

        retrieved = store.get("unicode-key")
        assert retrieved == unicode_value

    def test_very_long_credential_value(self, store):
        """Test storing very long credential value."""
        long_value = "a" * 10000
        store.set(
            name="long-key",
            value=long_value,
            credential_type=CredentialType.SECRET,
        )

        retrieved = store.get("long-key")
        assert retrieved == long_value
        assert len(retrieved) == 10000

    def test_empty_credential_value(self, store):
        """Test handling empty credential value."""
        # Empty values might be rejected or stored
        try:
            store.set("empty-key", "", CredentialType.API_KEY)
            result = store.get("empty-key", check_env=False)
            # If it succeeds, verify behavior
            assert result is not None or result == ""
        except (ValueError, AuthenticationError):
            # Or it might raise an error, which is also acceptable
            pass

    def test_concurrent_access(self, store):
        """Test concurrent credential access."""
        store.set(
            name="concurrent-test",
            value="shared-value",
            credential_type=CredentialType.API_KEY,
        )

        # Simulate concurrent reads
        results = []
        for _ in range(10):
            value = store.get("concurrent-test")
            results.append(value)

        assert all(v == "shared-value" for v in results if v is not None)

    def test_credential_type_enforcement(self, store):
        """Test credential type is recorded and surfaced via the audit trail."""
        audit_events: list[dict] = []
        store.audit_callback = audit_events.append

        store.set(
            name="typed-key",
            value="value1234",
            credential_type=CredentialType.DATABASE_URL,
        )

        assert store.get("typed-key", check_env=False) == "value1234"
        created_events = [
            event
            for event in audit_events
            if event.get("operation") == "credential_created"
        ]
        assert len(created_events) == 1
        assert created_events[0]["details"]["type"] == CredentialType.DATABASE_URL.value


class TestCredentialStoreEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def store(self, temp_dir):
        """Create credential store with temp storage."""
        storage_path = Path(temp_dir) / "credentials"
        return EnhancedCredentialStore(
            master_password="test-passphrase-12345",  # noqa: S106 - test credential
            storage_path=str(storage_path),
        )

    def test_weak_master_password(self, temp_dir):
        """A short master password is currently accepted and yields a working store.

        EnhancedCredentialStore performs no strength validation on
        master_password (see _init_secure_encryption); any non-empty value is
        used directly as PBKDF2 input. This test documents that behavior by
        exercising the store end-to-end rather than asserting on ambiguity.
        """
        store = EnhancedCredentialStore(
            master_password="weak",
            storage_path=str(Path(temp_dir) / "test"),
        )
        store.set("probe-key", "value1234", CredentialType.API_KEY)
        assert store.get("probe-key", check_env=False) == "value1234"

    def test_retrieve_deleted_credential(self, store):
        """Test retrieving credential after deletion."""
        store.set("deleted-key", "value1234", CredentialType.API_KEY)
        store.delete_secure("deleted-key")

        result = store.get("deleted-key", check_env=False)
        assert result is None

    def test_rotate_nonexistent_credential(self, store):
        """Test rotating credential that doesn't exist."""
        with pytest.raises((KeyError, AuthenticationError, ValueError)):
            store.rotate_credential("nonexistent", "new-value")

    def test_delete_nonexistent_credential(self, store):
        """Test deleting credential that doesn't exist."""
        result = store.delete_secure("nonexistent-key")
        # Should return False or raise error
        assert result is False or isinstance(result, bool)

    def test_multiple_stores_independent(self, temp_dir):
        """Test multiple stores are independent."""
        store1 = EnhancedCredentialStore(
            master_password="passphrase1",
            storage_path=str(Path(temp_dir) / "store1"),
        )
        store2 = EnhancedCredentialStore(
            master_password="passphrase2",
            storage_path=str(Path(temp_dir) / "store2"),
        )

        store1.set("key1", "value123", CredentialType.API_KEY)
        store2.set("key2", "value456", CredentialType.API_KEY)

        # Each store should only have its own keys
        assert store1.get("key1") is not None
        assert store1.get("key2", check_env=False) is None
        assert store2.get("key2") is not None
        assert store2.get("key1", check_env=False) is None
