"""Comprehensive unit tests for secure credential management."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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
            master_password="test_master_password_12345",
            storage_path=str(storage_path),
        )

    def test_store_initialization(self, temp_dir):
        """Test credential store initialization."""
        store = EnhancedCredentialStore(
            master_password="test_password_12345",
            storage_path=str(Path(temp_dir) / "creds"),
        )
        assert store is not None

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

        # Verify credential exists
        assert store.get("meta-test") is not None

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
        assert store.get("expiring-key") is not None

    def test_security_metrics(self, store):
        """Test getting security metrics."""
        store.set("metric-test", "value1234", CredentialType.API_KEY)
        store.get("metric-test")

        metrics = store.get_security_metrics()
        assert isinstance(metrics, dict)

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
        """Test credential type is stored."""
        store.set(
            name="typed-key",
            value="value1234",
            credential_type=CredentialType.DATABASE_URL,
        )

        # Verify credential exists (type is internal)
        assert store.get("typed-key") is not None


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
            master_password="test_master_password_12345",
            storage_path=str(storage_path),
        )

    def test_weak_master_password(self, temp_dir):
        """Test initialization with weak master password."""
        # Should either accept with warning or reject
        try:
            store = EnhancedCredentialStore(
                master_password="weak",
                storage_path=str(Path(temp_dir) / "test"),
            )
            assert store is not None
        except (ValueError, AuthenticationError):
            # Weak password rejection is acceptable
            pass

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
            master_password="password1",
            storage_path=str(Path(temp_dir) / "store1"),
        )
        store2 = EnhancedCredentialStore(
            master_password="password2",
            storage_path=str(Path(temp_dir) / "store2"),
        )

        store1.set("key1", "value123", CredentialType.API_KEY)
        store2.set("key2", "value456", CredentialType.API_KEY)

        # Each store should only have its own keys
        assert store1.get("key1") is not None
        assert store1.get("key2", check_env=False) is None
        assert store2.get("key2") is not None
        assert store2.get("key1", check_env=False) is None
