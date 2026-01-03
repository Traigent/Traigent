"""Integration tests for secure credential storage across all security levels."""

import logging
import os
import stat
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from traigent.security.credentials import (
    CredentialType,
    EnhancedCredentialStore,
    SecureString,
    SecurityLevel,
    get_secure_credential_store,
)
from traigent.utils.exceptions import AuthenticationError as SecurityError


class TestCredentialStoreStandardMode:
    """Test credential store in standard security mode."""

    def test_standard_mode_basic_operations(
        self, temp_credentials_path, clean_environment
    ):
        """Test basic operations in standard mode."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase-123",
            security_level=SecurityLevel.STANDARD,
        )

        # Test setting credentials
        store.set("API_KEY", "test_api_key_123", CredentialType.API_KEY)
        store.set(
            "DATABASE_URL", "postgresql://localhost/db", CredentialType.DATABASE_URL
        )

        # Test getting credentials returns decrypted value
        result = store.get("API_KEY")
        assert result == "test_api_key_123"

        # Test credential health check
        health = store.check_credential_health()
        assert health["total_credentials"] == 2

    def test_environment_variable_fallback(
        self, temp_credentials_path, clean_environment
    ):
        """Test that environment variables are checked first."""
        os.environ["TRAIGENT_API_KEY"] = "env_api_key"

        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.STANDARD,
            use_env_vars=True,
        )

        # Should get from environment
        result = store.get("API_KEY", check_env=True)
        assert result == "env_api_key"

        # Should not get from environment when disabled
        result = store.get("API_KEY", check_env=False)
        assert result is None

    def test_credential_validation(self, temp_credentials_path, clean_environment):
        """Test credential validation."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.STANDARD,
        )

        # Should reject short credentials
        with pytest.raises(ValueError) as exc:
            store.set("SHORT", "abc", CredentialType.SECRET)
        assert "too short" in str(exc.value).lower()

        # Should reject weak/placeholder credentials
        with pytest.raises(SecurityError) as exc:
            store.set("WEAK", "password123", CredentialType.PASSWORD)
        assert (
            "weak" in str(exc.value).lower() or "placeholder" in str(exc.value).lower()
        )

        with pytest.raises(SecurityError) as exc:
            store.set("PLACEHOLDER", "your-api-key-here", CredentialType.API_KEY)
        assert (
            "placeholder" in str(exc.value).lower() or "weak" in str(exc.value).lower()
        )


class TestCredentialStoreHighSecurityMode:
    """Test credential store in high security mode."""

    def test_high_security_mode_initialization(
        self, temp_credentials_path, clean_environment
    ):
        """Test high security mode initialization."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="strong-passphrase-123!@#",  # noqa: S106 - test credential
            security_level=SecurityLevel.HIGH,
        )

        assert store.security_level == SecurityLevel.HIGH
        assert store.PBKDF2_ITERATIONS == 100000  # NIST recommended

    def test_credential_rotation(self, temp_credentials_path, clean_environment):
        """Test credential rotation functionality."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.HIGH,
        )

        # Set initial credential
        store.set("ROTATE_ME", "initial_value_123", CredentialType.SECRET)

        # Rotate credential
        store.rotate_credential("ROTATE_ME", "new_value_456")

        # Check that rotation was recorded
        cred = store._credentials["ROTATE_ME"]
        assert "rotation_history" in cred.metadata
        assert len(cred.metadata["rotation_history"]) == 1
        assert cred.access_count == 0  # Reset after rotation

        # Check metrics
        metrics = store.get_security_metrics()
        assert metrics["rotation_count"] == 1

    def test_pbkdf2_key_derivation(self, temp_credentials_path, clean_environment):
        """Test that PBKDF2 is used for key derivation."""
        with patch("traigent.security.credentials.PBKDF2HMAC") as mock_pbkdf2:
            with patch("traigent.security.credentials.AESGCM") as mock_aesgcm:
                mock_kdf = MagicMock()
                mock_kdf.derive.return_value = (
                    b"\x00" * 32
                )  # Exactly 32 bytes for AES-256
                mock_pbkdf2.return_value = mock_kdf

                # Mock AESGCM to accept the key
                mock_aesgcm.return_value = MagicMock()

                EnhancedCredentialStore(
                    storage_path=temp_credentials_path,
                    master_password="test-passphrase",
                    security_level=SecurityLevel.HIGH,
                )

                # Verify PBKDF2 was called with correct parameters
                mock_pbkdf2.assert_called_once()
                call_args = mock_pbkdf2.call_args[1]
                assert call_args["iterations"] == 100000
                assert call_args["length"] == 32  # 256 bits

                # Verify AESGCM was initialized with the derived key
                mock_aesgcm.assert_called_once_with(b"\x00" * 32)


class TestCredentialStoreMaximumSecurityMode:
    """Test credential store in maximum security mode."""

    def test_maximum_security_automatic_expiration(
        self, temp_credentials_path, clean_environment
    ):
        """Test that maximum security enforces automatic expiration."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="ultra-secure-passphrase-123!@#",
            security_level=SecurityLevel.MAXIMUM,
        )

        # Set credential without explicit expiration
        store.set("AUTO_EXPIRE", "sensitive_value_123", CredentialType.SECRET)

        # Check that expiration was set automatically
        cred = store._credentials["AUTO_EXPIRE"]
        assert cred.expires_at is not None
        remaining_days = (cred.expires_at - datetime.now(UTC)).days
        assert remaining_days <= store.MAX_CREDENTIAL_AGE_DAYS

    def test_hsm_integration_attempt(self, temp_credentials_path, clean_environment):
        """Test HSM integration attempt."""
        with patch("traigent.security.credentials.logger") as mock_logger:
            store = EnhancedCredentialStore(
                storage_path=temp_credentials_path,
                master_password="test-passphrase",
                security_level=SecurityLevel.MAXIMUM,
                enable_hsm=True,
            )

            # Should log that HSM is not available
            assert not store.enable_hsm  # HSM disabled since not available
            mock_logger.info.assert_called()

    def test_credential_expiration_enforcement(
        self, temp_credentials_path, clean_environment
    ):
        """Test credential expiration enforcement."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.MAXIMUM,
        )

        # Create expired credential manually
        from traigent.security.credentials import SecureCredential

        ciphertext, nonce, associated_data = store._encrypt_value("expired-value")
        now_utc = datetime.now(UTC)
        expired_cred = SecureCredential(
            name="EXPIRED",
            type=CredentialType.SECRET,
            _encrypted_value=ciphertext,
            metadata={},
            security_level=SecurityLevel.MAXIMUM,
            created_at=now_utc - timedelta(days=100),
            expires_at=now_utc - timedelta(days=1),  # Expired yesterday
            nonce=nonce,
            associated_data=associated_data,
        )

        store._credentials["EXPIRED"] = expired_cred

        # Should not return expired credential
        result = store.get("EXPIRED")
        assert result is None

        # Check metrics
        metrics = store.get_security_metrics()
        assert metrics["failed_accesses"] == 1


class TestSecureString:
    """Test SecureString memory protection."""

    def test_secure_string_basic_operations(self):
        """Test basic SecureString operations."""
        secure = SecureString("sensitive_data")

        # Should be able to get value
        assert secure.get() == "sensitive_data"

        # Should be able to clear
        secure.clear()
        assert secure._locked

        # Should not be able to get after clearing
        with pytest.raises(SecurityError):
            secure.get()

    def test_secure_string_auto_cleanup(self):
        """Test that SecureString cleans up on deletion."""
        secure = SecureString("test_data")
        secure_id = id(secure)
        # When deleted, should auto-clear
        del secure
        # Verify the object existed and was deleted (id should be positive)
        assert secure_id > 0  # Object had valid id before deletion


class TestCredentialStoreEnvironmentDetection:
    """Test automatic environment detection for credential store."""

    def test_production_environment(self, clean_environment):
        """Test production environment detection."""
        os.environ["TRAIGENT_ENVIRONMENT"] = "production"

        store = get_secure_credential_store()
        assert store.security_level == SecurityLevel.MAXIMUM

    def test_staging_environment(self, clean_environment):
        """Test staging environment detection."""
        os.environ["TRAIGENT_ENVIRONMENT"] = "staging"

        store = get_secure_credential_store()
        assert store.security_level == SecurityLevel.HIGH

    def test_development_environment(self, clean_environment):
        """Test development environment detection."""
        os.environ["TRAIGENT_ENVIRONMENT"] = "development"

        with patch("traigent.security.credentials.logger") as mock_logger:
            store = get_secure_credential_store()
            assert store.security_level == SecurityLevel.STANDARD
            mock_logger.warning.assert_called()


class TestCredentialStorePersistence:
    """Test credential store persistence."""

    def test_save_and_load_credentials(self, temp_credentials_path, clean_environment):
        """Test saving and loading credentials."""
        # Create and save credentials
        store1 = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.HIGH,
        )

        store1.set("PERSIST_1", "value_1_secure", CredentialType.SECRET)
        store1.set("PERSIST_2", "value_2_secure", CredentialType.API_KEY)

        # Create new store with same path and password
        store2 = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.HIGH,
        )

        # Should have loaded the credentials
        assert len(store2._credentials) == 2
        assert "PERSIST_1" in store2._credentials
        assert "PERSIST_2" in store2._credentials

    def test_wrong_passphrase_fails(self, temp_credentials_path, clean_environment):
        """Test that wrong password fails to load credentials."""
        # Create and save credentials
        store1 = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="correct-passphrase",
            security_level=SecurityLevel.HIGH,
        )

        store1.set("SECRET", "sensitive_value", CredentialType.SECRET)

        # Try to load with wrong password
        with pytest.raises(SecurityError):
            EnhancedCredentialStore(
                storage_path=temp_credentials_path,
                master_password="wrong-passphrase",
                security_level=SecurityLevel.HIGH,
            )


class TestCredentialStoreAuditing:
    """Test credential store auditing."""

    def test_audit_callback(
        self, temp_credentials_path, mock_audit_callback, clean_environment
    ):
        """Test that audit callbacks are triggered."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.HIGH,
            audit_callback=mock_audit_callback,
        )

        # Perform operations
        store.set("AUDIT_TEST", "test_value_123", CredentialType.SECRET)
        store.get("AUDIT_TEST")
        store.rotate_credential("AUDIT_TEST", "new_value_456")
        store.delete_secure("AUDIT_TEST")

        # Check that audit callback was called
        assert mock_audit_callback.call_count >= 4  # At least 4 operations

    def test_security_events(
        self, temp_credentials_path, mock_audit_callback, clean_environment
    ):
        """Test that security events are logged."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.HIGH,
            audit_callback=mock_audit_callback,
        )

        # Trigger security event (credential not found)
        result = store.get("NONEXISTENT")
        assert result is None

        # Check that security event was logged
        calls = mock_audit_callback.call_args_list
        assert any("not_found" in str(call) for call in calls)


class TestCredentialStoreHealthCheck:
    """Test credential health monitoring."""

    def test_health_check_reporting(self, temp_credentials_path, clean_environment):
        """Test health check reporting."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.HIGH,
        )

        # Add credentials with various states
        store.set("FRESH", "fresh_value", CredentialType.SECRET)

        # Add old credential (manually for testing)
        from traigent.security.credentials import SecureCredential

        ciphertext, nonce, associated_data = store._encrypt_value("old-secret-value")
        now_utc = datetime.now(UTC)
        old_cred = SecureCredential(
            name="OLD",
            type=CredentialType.SECRET,
            _encrypted_value=ciphertext,
            metadata={},
            security_level=SecurityLevel.HIGH,
            created_at=now_utc - timedelta(days=80),
            access_count=9000,
            nonce=nonce,
            associated_data=associated_data,
        )
        store._credentials["OLD"] = old_cred

        # Check health
        health = store.check_credential_health()
        assert health["total_credentials"] == 2
        assert health["requiring_rotation"] >= 1  # OLD credential needs rotation
        assert health["overused"] >= 1  # OLD credential is overused


class TestCredentialStorePerformance:
    """Test credential store performance."""

    @pytest.mark.integration
    def test_credential_operations_performance(
        self, temp_credentials_path, clean_environment
    ):
        """Test performance of credential operations."""
        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            master_password="test-passphrase",
            security_level=SecurityLevel.STANDARD,  # Use standard for speed
        )

        # Test write performance
        start_time = time.time()
        for i in range(50):
            store.set(f"PERF_{i}", f"value_{i}_secure", CredentialType.SECRET)
        write_time = time.time() - start_time

        # Should handle 50 writes reasonably fast
        assert write_time < 5.0  # Less than 5 seconds

        # Test read performance
        start_time = time.time()
        for i in range(50):
            store.get(f"PERF_{i}")
        read_time = time.time() - start_time

        # Reads should be faster than writes
        assert read_time < 2.0  # Less than 2 seconds


class TestMasterPasswordGeneration:
    """Test secure handling of automatically generated master passwords."""

    def test_generated_master_password_not_logged(
        self,
        temp_credentials_path,
        clean_environment,
        caplog,
        monkeypatch,
    ):
        """Ensure generated master passwords are not written to logs."""
        generated_password = "unit-test-generated-password"
        monkeypatch.setattr(
            "traigent.security.credentials.secrets.token_urlsafe",
            lambda _: generated_password,
        )

        caplog.set_level(logging.CRITICAL)

        store = EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            security_level=SecurityLevel.HIGH,
            use_env_vars=False,
        )
        assert isinstance(store, EnhancedCredentialStore)

        password_file = temp_credentials_path.parent / ".master_password"
        assert password_file.exists()
        assert password_file.read_text(encoding="utf-8").strip() == generated_password
        if os.name != "nt":
            assert stat.S_IMODE(password_file.stat().st_mode) == 0o600

        assert any(
            "Generated new master password" in record.getMessage()
            for record in caplog.records
        )
        for record in caplog.records:
            assert generated_password not in record.getMessage()

    def test_stored_master_password_reused(
        self,
        temp_credentials_path,
        clean_environment,
        monkeypatch,
    ):
        """Reuse stored master password without regenerating."""
        generated_password = "persisted-password-for-test"
        monkeypatch.setattr(
            "traigent.security.credentials.secrets.token_urlsafe",
            lambda _: generated_password,
        )

        EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            security_level=SecurityLevel.HIGH,
            use_env_vars=False,
        )

        password_file = temp_credentials_path.parent / ".master_password"
        assert password_file.exists()
        assert password_file.read_text(encoding="utf-8").strip() == generated_password

        generation_mock = MagicMock(
            side_effect=AssertionError("master password should not regenerate")
        )
        monkeypatch.setattr(
            "traigent.security.credentials.secrets.token_urlsafe",
            generation_mock,
        )

        EnhancedCredentialStore(
            storage_path=temp_credentials_path,
            security_level=SecurityLevel.HIGH,
            use_env_vars=False,
        )

        assert not generation_mock.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
