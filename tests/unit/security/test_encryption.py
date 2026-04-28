"""Tests for encryption and data protection systems"""

import json
import os
import tempfile
from datetime import UTC, datetime, timedelta

import pytest

from traigent.security.encryption import (
    DataClassification,
    DataProtectionManager,
    EncryptionManager,
    KeyManager,
    PIIDetector,
    PIIType,
    SecureStorage,
)


class TestKeyManager:
    """Test KeyManager class"""

    def test_key_generation(self):
        """Test encryption key generation"""
        key_manager = KeyManager()

        # Generate AES-256 key
        key_id = key_manager.generate_key("AES-256")
        assert key_id is not None

        # Retrieve key
        key = key_manager.get_key(key_id)
        assert key is not None
        assert len(key) == 32  # 256 bits / 8 = 32 bytes

        # Check metadata
        metadata = key_manager.key_metadata[key_id]
        assert metadata.algorithm == "AES-256"
        assert metadata.key_length == 256
        assert metadata.is_valid()

    def test_key_expiration(self):
        """Test key expiration"""
        key_manager = KeyManager()

        # Generate key with expiration
        key_id = key_manager.generate_key("AES-256", expires_in_days=1)
        metadata = key_manager.key_metadata[key_id]

        assert metadata.expires_at is not None
        assert not metadata.is_expired()

        # Mock expiration
        metadata.expires_at = datetime.now(UTC) - timedelta(days=1)
        assert metadata.is_expired()
        assert not metadata.is_valid()

        # Should not be able to retrieve expired key
        assert key_manager.get_key(key_id) is None

    def test_key_rotation(self):
        """Test key rotation"""
        key_manager = KeyManager()

        # Generate original key
        old_key_id = key_manager.generate_key("AES-256")
        old_key = key_manager.get_key(old_key_id)

        # Rotate key
        new_key_id = key_manager.rotate_key(old_key_id)
        new_key = key_manager.get_key(new_key_id)

        assert new_key_id != old_key_id
        assert new_key != old_key
        assert not key_manager.key_metadata[old_key_id].is_active
        assert key_manager.key_metadata[new_key_id].is_active

    def test_key_deletion(self):
        """Test secure key deletion"""
        key_manager = KeyManager()

        key_id = key_manager.generate_key("AES-256")
        assert key_manager.get_key(key_id) is not None

        # Delete key
        assert key_manager.delete_key(key_id)
        assert key_manager.get_key(key_id) is None
        assert key_id not in key_manager.keys
        assert key_id not in key_manager.key_metadata


class TestEncryptionManager:
    """Test EncryptionManager class"""

    def test_encrypt_decrypt_string(self):
        """Test string encryption and decryption"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)

        original_data = "This is sensitive data"

        # Encrypt data
        encrypted_result = encryption_manager.encrypt(original_data)

        assert "ciphertext" in encrypted_result
        assert "iv" in encrypted_result
        assert "tag" in encrypted_result
        assert "key_id" in encrypted_result
        assert encrypted_result["algorithm"] == "AES-256-GCM"
        assert encrypted_result["classification"] == DataClassification.INTERNAL.value

        # Decrypt data
        decrypted_data = encryption_manager.decrypt(encrypted_result)
        assert decrypted_data.decode("utf-8") == original_data

    def test_encrypt_decrypt_bytes(self):
        """Test bytes encryption and decryption"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)

        original_data = b"Binary data content"

        # Encrypt data
        encrypted_result = encryption_manager.encrypt(original_data)

        # Decrypt data
        decrypted_data = encryption_manager.decrypt(encrypted_result)
        assert decrypted_data == original_data

    def test_encryption_with_classification(self):
        """Test encryption with data classification"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)

        data = "Confidential information"

        # Encrypt with specific classification
        encrypted_result = encryption_manager.encrypt(
            data, classification=DataClassification.CONFIDENTIAL
        )

        assert (
            encrypted_result["classification"] == DataClassification.CONFIDENTIAL.value
        )

    def test_zeroize_key_buffer_mutates_buffer_in_place(self):
        """Key buffer zeroization should clear all bytes."""
        key_buffer = bytearray(b"secret")

        EncryptionManager._zeroize_key_buffer(key_buffer)

        assert key_buffer == bytearray(b"\x00" * 6)

    def test_encrypt_invokes_key_buffer_zeroization(self, monkeypatch):
        """Encrypt path should invoke best-effort key buffer zeroization."""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        observed_buffers: list[bytes] = []

        def capture_and_zeroize(key_buffer: bytearray | None) -> None:
            if key_buffer is not None:
                observed_buffers.append(bytes(key_buffer))
                for idx in range(len(key_buffer)):
                    key_buffer[idx] = 0

        monkeypatch.setattr(
            EncryptionManager,
            "_zeroize_key_buffer",
            staticmethod(capture_and_zeroize),
        )

        encryption_manager.encrypt("buffer wipe check")

        assert observed_buffers
        assert any(any(byte != 0 for byte in before) for before in observed_buffers)

    def test_decrypt_invokes_key_buffer_zeroization(self, monkeypatch):
        """Decrypt path should invoke best-effort key buffer zeroization."""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        encrypted_result = encryption_manager.encrypt("decrypt wipe check")
        observed_buffers: list[bytes] = []

        def capture_and_zeroize(key_buffer: bytearray | None) -> None:
            if key_buffer is not None:
                observed_buffers.append(bytes(key_buffer))
                for idx in range(len(key_buffer)):
                    key_buffer[idx] = 0

        monkeypatch.setattr(
            EncryptionManager,
            "_zeroize_key_buffer",
            staticmethod(capture_and_zeroize),
        )

        decrypted = encryption_manager.decrypt(encrypted_result)

        assert decrypted == b"decrypt wipe check"
        assert observed_buffers
        assert any(any(byte != 0 for byte in before) for before in observed_buffers)

    def test_file_encryption(self):
        """Test file encryption and decryption"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("File content to encrypt")
            temp_file = f.name

        try:
            # Encrypt file
            encrypted_file = encryption_manager.encrypt_file(temp_file)
            assert os.path.exists(encrypted_file)
            assert encrypted_file.endswith(".encrypted")

            # Verify encrypted file contains JSON
            with open(encrypted_file) as f:
                encrypted_data = json.load(f)
                assert "ciphertext" in encrypted_data

            # Decrypt file
            decrypted_file = encryption_manager.decrypt_file(encrypted_file)
            assert os.path.exists(decrypted_file)

            # Verify content
            with open(decrypted_file) as f:
                content = f.read()
                assert content == "File content to encrypt"

        finally:
            # Cleanup
            for file_path in [temp_file, encrypted_file, decrypted_file]:
                if os.path.exists(file_path):
                    os.unlink(file_path)

    def test_encrypt_input_validation(self):
        """Encrypt should validate payloads and classifications."""
        encryption_manager = EncryptionManager(KeyManager())

        with pytest.raises(TypeError):
            encryption_manager.encrypt(123)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            encryption_manager.encrypt("")

        with pytest.raises(ValueError):
            encryption_manager.encrypt("data", classification="public")  # type: ignore[arg-type]

    def test_decrypt_structure_validation(self):
        """Decrypt should reject malformed payloads."""
        encryption_manager = EncryptionManager(KeyManager())

        with pytest.raises(TypeError):
            encryption_manager.decrypt("not a dict")  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            encryption_manager.decrypt({})

        malformed = {
            "ciphertext": b"abc",
            "iv": b"short",
            "tag": b"tag",
            "key_id": "",
        }
        with pytest.raises(ValueError):
            encryption_manager.decrypt(malformed)


class TestPIIDetector:
    """Test PIIDetector class"""

    def test_email_detection(self):
        """Test email PII detection"""
        detector = PIIDetector()

        text = "Contact us at support@example.com or admin@test.org"
        detections = detector.detect_pii(text)

        email_detections = [d for d in detections if d.pii_type == PIIType.EMAIL]
        assert len(email_detections) == 2
        assert "support@example.com" in [d.value for d in email_detections]
        assert "admin@test.org" in [d.value for d in email_detections]

    def test_phone_detection(self):
        """Test phone number PII detection"""
        detector = PIIDetector()

        text = "Call me at 555-123-4567 or (555) 987-6543"
        detections = detector.detect_pii(text)

        phone_detections = [d for d in detections if d.pii_type == PIIType.PHONE]
        assert len(phone_detections) == 2
        assert "555-123-4567" in [d.value for d in phone_detections]
        assert "(555) 987-6543" in [d.value for d in phone_detections]

    def test_ssn_detection(self):
        """Test SSN PII detection"""
        detector = PIIDetector()

        text = "SSN: 123-45-6789"
        detections = detector.detect_pii(text)

        ssn_detections = [d for d in detections if d.pii_type == PIIType.SSN]
        assert len(ssn_detections) == 1
        assert ssn_detections[0].value == "123-45-6789"

    def test_credit_card_detection(self):
        """Test credit card PII detection"""
        detector = PIIDetector()

        text = "Card number: 1234 5678 9012 3456 or 1234-5678-9012-3456"
        detections = detector.detect_pii(text)

        cc_detections = [d for d in detections if d.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_detections) == 2

    def test_ip_address_detection(self):
        """Test IP address PII detection"""
        detector = PIIDetector()

        text = "Server IP: 192.168.1.1"
        detections = detector.detect_pii(text)

        ip_detections = [d for d in detections if d.pii_type == PIIType.IP_ADDRESS]
        assert len(ip_detections) == 1
        assert ip_detections[0].value == "192.168.1.1"

    def test_anonymize_text(self):
        """Test text anonymization"""
        detector = PIIDetector()

        text = "Contact support@example.com or call 555-123-4567"
        anonymized = detector.anonymize_text(text)

        assert "support@example.com" not in anonymized
        assert "555-123-4567" not in anonymized
        assert "[EMAIL_REDACTED]" in anonymized
        assert "[PHONE_REDACTED]" in anonymized

    def test_pseudonymize_text(self):
        """Test text pseudonymization"""
        detector = PIIDetector()

        text = "Email: test@example.com"
        pseudonymized = detector.pseudonymize_text(text)

        assert "test@example.com" not in pseudonymized
        assert "[EMAIL_" in pseudonymized

        # Same email should produce same pseudonym
        pseudonymized2 = detector.pseudonymize_text(text)
        assert pseudonymized == pseudonymized2

    def test_selective_anonymization(self):
        """Test selective PII anonymization"""
        detector = PIIDetector()

        text = "Contact test@example.com or call 555-123-4567"

        # Only anonymize emails
        anonymized = detector.anonymize_text(text, pii_types=[PIIType.EMAIL])

        assert "test@example.com" not in anonymized
        assert "555-123-4567" in anonymized  # Phone should remain
        assert "[EMAIL_REDACTED]" in anonymized

    def test_custom_pattern_validation(self):
        """Invalid custom patterns should raise ValueError."""
        detector = PIIDetector()

        with pytest.raises(ValueError):
            detector.add_custom_pattern("bad", "(abc")


class TestDataProtectionManager:
    """Test DataProtectionManager class"""

    def test_protect_data_with_encryption(self):
        """Test data protection with encryption"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        pii_detector = PIIDetector()
        protection_manager = DataProtectionManager(encryption_manager, pii_detector)

        data = "User email: test@example.com, phone: 555-123-4567"

        # Protect data
        protected_result = protection_manager.protect_data(
            data=data,
            classification=DataClassification.CONFIDENTIAL,
            anonymize_pii=True,
            encrypt=True,
        )

        assert protected_result["is_encrypted"]
        assert "encryption" in protected_result["metadata"]
        assert protected_result["metadata"]["pii_detected"] > 0
        assert "pii_anonymization" in protected_result["metadata"]["protection_applied"]
        assert "encryption" in protected_result["metadata"]["protection_applied"]

        # Unprotect data
        unprotected_data = protection_manager.unprotect_data(protected_result)

        # Should be anonymized but decrypted
        assert "test@example.com" not in unprotected_data
        assert "555-123-4567" not in unprotected_data
        assert "[EMAIL_REDACTED]" in unprotected_data
        assert "[PHONE_REDACTED]" in unprotected_data

    def test_protect_data_without_encryption(self):
        """Test data protection without encryption"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        pii_detector = PIIDetector()
        protection_manager = DataProtectionManager(encryption_manager, pii_detector)

        data = "User email: test@example.com"

        # Protect data without encryption
        protected_result = protection_manager.protect_data(
            data=data,
            classification=DataClassification.PUBLIC,
            anonymize_pii=True,
            encrypt=False,
        )

        assert not protected_result["is_encrypted"]
        assert "encryption" not in protected_result["metadata"]
        assert protected_result["metadata"]["pii_detected"] > 0

        # Unprotect data
        unprotected_data = protection_manager.unprotect_data(protected_result)
        assert "test@example.com" not in unprotected_data
        assert "[EMAIL_REDACTED]" in unprotected_data

    def test_retention_policies(self):
        """Test data retention policies"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        pii_detector = PIIDetector()
        protection_manager = DataProtectionManager(encryption_manager, pii_detector)

        # Check retention periods
        assert (
            protection_manager.get_retention_period(DataClassification.PUBLIC) == 3650
        )
        assert (
            protection_manager.get_retention_period(DataClassification.CONFIDENTIAL)
            == 1825
        )
        assert (
            protection_manager.get_retention_period(DataClassification.TOP_SECRET)
            == 365
        )

        # Test expiration checking
        old_date = datetime.now(UTC) - timedelta(days=400)

        # Should be expired for TOP_SECRET (365 days retention)
        assert protection_manager.is_retention_expired(
            old_date, DataClassification.TOP_SECRET
        )

        # Should not be expired for CONFIDENTIAL (1825 days retention)
        assert not protection_manager.is_retention_expired(
            old_date, DataClassification.CONFIDENTIAL
        )


class TestSecureStorage:
    """Test SecureStorage class"""

    def test_store_and_retrieve_data(self):
        """Test storing and retrieving data"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        pii_detector = PIIDetector()
        protection_manager = DataProtectionManager(encryption_manager, pii_detector)
        storage = SecureStorage(protection_manager)

        data = {"username": "testuser", "email": "test@example.com"}

        # Store data
        storage.store("user123", data, DataClassification.INTERNAL)

        # Retrieve data
        retrieved_data = storage.retrieve("user123")

        assert retrieved_data is not None
        assert retrieved_data["username"] == "testuser"
        # Email should be anonymized
        assert "test@example.com" not in str(retrieved_data)

    def test_data_not_found(self):
        """Test retrieving non-existent data"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        pii_detector = PIIDetector()
        protection_manager = DataProtectionManager(encryption_manager, pii_detector)
        storage = SecureStorage(protection_manager)

        assert storage.retrieve("nonexistent") is None

    def test_delete_data(self):
        """Test secure data deletion"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        pii_detector = PIIDetector()
        protection_manager = DataProtectionManager(encryption_manager, pii_detector)
        storage = SecureStorage(protection_manager)

        data = {"test": "data"}
        storage.store("test_key", data)

        # Verify data exists
        assert storage.retrieve("test_key") is not None

        # Delete data
        assert storage.delete("test_key")
        assert storage.retrieve("test_key") is None

        # Delete non-existent key
        assert not storage.delete("nonexistent")

    def test_list_keys(self):
        """Test listing stored keys"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        pii_detector = PIIDetector()
        protection_manager = DataProtectionManager(encryption_manager, pii_detector)
        storage = SecureStorage(protection_manager)

        # Store data with different classifications
        storage.store("public1", {"data": "public"}, DataClassification.PUBLIC)
        storage.store("internal1", {"data": "internal"}, DataClassification.INTERNAL)
        storage.store(
            "confidential1", {"data": "confidential"}, DataClassification.CONFIDENTIAL
        )

        # List all keys
        all_keys = storage.list_keys()
        assert len(all_keys) == 3
        assert "public1" in all_keys
        assert "internal1" in all_keys
        assert "confidential1" in all_keys

        # List by classification
        public_keys = storage.list_keys(DataClassification.PUBLIC)
        assert len(public_keys) == 1
        assert "public1" in public_keys

        confidential_keys = storage.list_keys(DataClassification.CONFIDENTIAL)
        assert len(confidential_keys) == 1
        assert "confidential1" in confidential_keys

    def test_retention_cleanup(self):
        """Test automatic cleanup of expired data"""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        pii_detector = PIIDetector()
        protection_manager = DataProtectionManager(encryption_manager, pii_detector)
        storage = SecureStorage(protection_manager)

        # Store data
        storage.store("test_key", {"data": "test"}, DataClassification.TOP_SECRET)

        # Mock old creation time to simulate expiration
        record = storage.storage["test_key"]
        old_time = datetime.now(UTC) - timedelta(
            days=400
        )  # Older than TOP_SECRET retention (365 days)
        record["created_at"] = old_time.isoformat()

        # Try to retrieve - should auto-delete expired data
        retrieved_data = storage.retrieve("test_key")
        assert retrieved_data is None
        assert "test_key" not in storage.storage

        # Test cleanup_expired method
        storage.store("test_key2", {"data": "test2"}, DataClassification.TOP_SECRET)
        record2 = storage.storage["test_key2"]
        record2["created_at"] = old_time.isoformat()

        cleaned_count = storage.cleanup_expired()
        assert cleaned_count == 1
        assert "test_key2" not in storage.storage

    def test_store_data_encrypts_payload(self):
        """Ensure store_data uses encryption manager and records metadata."""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        storage = SecureStorage(encryption_manager)

        record = storage.store_data(
            data="Highly Confidential Payload",
            record_id="record-1",
            classification=DataClassification.CONFIDENTIAL,
            metadata={"source": "unit-test"},
        )

        assert record.record_id == "record-1"
        assert record.encryption_key_id is not None
        assert record.data != b"Highly Confidential Payload"
        assert record.metadata["source"] == "unit-test"
        encryption_metadata = record.metadata["encryption_metadata"]
        assert isinstance(encryption_metadata["iv"], bytes)
        assert isinstance(encryption_metadata["tag"], bytes)
        assert encryption_metadata["algorithm"] == "AES-256-GCM"

    def test_retrieve_data_decrypts_payload(self):
        """Ensure retrieve_data rebuilds encryption payload for decryption."""
        key_manager = KeyManager()
        encryption_manager = EncryptionManager(key_manager)
        storage = SecureStorage(encryption_manager)

        storage.store_data(
            data="Secret One-Time Token",
            record_id="record-2",
            classification=DataClassification.RESTRICTED,
        )

        decrypted = storage.retrieve_data("record-2")
        assert decrypted == "Secret One-Time Token"


class TestEncryptionFailsClosedWithoutCrypto:
    """Encryption must fail closed when ``cryptography`` is unavailable.

    Previously the encrypt()/decrypt() methods had a TRAIGENT_MOCK_LLM-gated
    fallback that returned ``b"mock_" + plaintext`` (encrypt) and stripped
    that prefix on decrypt — which is not encryption at all and silently
    leaked plaintext-as-ciphertext if the env var was set in production.
    The fallback is now removed entirely; both methods raise RuntimeError
    when crypto is not available, regardless of any env var.
    """

    def test_encrypt_raises_without_crypto(self):
        """encrypt() must raise when cryptography is unavailable, no env override."""
        key_manager = KeyManager()
        em = EncryptionManager(key_manager)
        em.crypto_available = False
        with pytest.raises(RuntimeError, match="requires the 'cryptography' package"):
            em.encrypt("sensitive data")

    def test_decrypt_raises_without_crypto(self):
        """decrypt() must raise when cryptography is unavailable, no env override."""
        key_manager = KeyManager()
        em = EncryptionManager(key_manager)
        em.crypto_available = False
        key_id = key_manager.generate_key("AES-256")
        encrypted = {
            "ciphertext": b"mock_test",
            "iv": os.urandom(12),
            "tag": os.urandom(16),
            "key_id": key_id,
        }
        with pytest.raises(RuntimeError, match="requires the 'cryptography' package"):
            em.decrypt(encrypted)

    def test_encrypt_ignores_traigent_mock_llm(self):
        """Setting TRAIGENT_MOCK_LLM=true must NOT enable mock encryption."""
        import unittest.mock

        key_manager = KeyManager()
        with unittest.mock.patch.dict(
            os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=False
        ):
            em = EncryptionManager(key_manager)
            em.crypto_available = False
            with pytest.raises(
                RuntimeError, match="requires the 'cryptography' package"
            ):
                em.encrypt("sensitive data")

    def test_decrypt_ignores_traigent_mock_llm(self):
        """Setting TRAIGENT_MOCK_LLM=true must NOT enable mock decryption."""
        import unittest.mock

        key_manager = KeyManager()
        with unittest.mock.patch.dict(
            os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=False
        ):
            em = EncryptionManager(key_manager)
            em.crypto_available = False
            key_id = key_manager.generate_key("AES-256")
            encrypted = {
                "ciphertext": b"mock_test",
                "iv": os.urandom(12),
                "tag": os.urandom(16),
                "key_id": key_id,
            }
            with pytest.raises(
                RuntimeError, match="requires the 'cryptography' package"
            ):
                em.decrypt(encrypted)
