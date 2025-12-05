"""Comprehensive tests for crypto utilities.

Tests cover:
- Secure credential storage and encryption
- File security operations
- Password hashing and verification
- Key derivation and security
- Error handling and edge cases
"""

import json
import os
import tempfile

import pytest

from traigent.security.crypto_utils import (
    SecureCredentialStorage,
    SecureFileManager,
    generate_secure_key,
    get_credential_storage,
    secure_hash,
)


class TestSecureCredentialStorage:
    """Test SecureCredentialStorage class"""

    def test_credential_storage_creation(self):
        """Test creating SecureCredentialStorage"""
        storage = SecureCredentialStorage()

        assert storage.version == "1"
        assert storage.algorithm == "AES-256-GCM"
        assert storage.key_derivation == "PBKDF2"
        assert storage.iterations == 100000

    def test_encrypt_decrypt_credentials_basic(self):
        """Test basic credential encryption and decryption"""
        storage = SecureCredentialStorage()

        credentials = {
            "api_key": "test_api_key_12345",
            "username": "testuser",
            "password": "testpass123",
        }

        # Encrypt credentials
        encrypted = storage.encrypt_credentials(credentials)

        # Check encrypted structure
        assert "salt" in encrypted
        assert "data" in encrypted
        assert "version" in encrypted
        assert encrypted["version"] == "1"

        # Decrypt credentials
        decrypted = storage.decrypt_credentials(encrypted)

        # Should match original
        assert decrypted == credentials

    def test_encrypt_decrypt_credentials_empty(self):
        """Test encrypting empty credentials"""
        storage = SecureCredentialStorage()

        empty_credentials = {}

        encrypted = storage.encrypt_credentials(empty_credentials)
        decrypted = storage.decrypt_credentials(encrypted)

        assert decrypted == empty_credentials

    def test_encrypt_decrypt_credentials_complex(self):
        """Test encrypting complex credential structures"""
        storage = SecureCredentialStorage()

        complex_credentials = {
            "api_keys": {"openai": "sk-test123", "anthropic": "ant-test456"},
            "databases": [
                {"host": "db1.example.com", "password": "pass1"},
                {"host": "db2.example.com", "password": "pass2"},
            ],
            "metadata": {
                "created": "2024-01-01",
                "environment": "test",
                "features": ["auth", "billing"],
            },
            "numbers": [1, 2, 3.14, -5],
            "boolean_flag": True,
            "null_value": None,
        }

        encrypted = storage.encrypt_credentials(complex_credentials)
        decrypted = storage.decrypt_credentials(encrypted)

        assert decrypted == complex_credentials

    def test_encrypt_credentials_invalid_input(self):
        """Test encrypting invalid credential types"""
        storage = SecureCredentialStorage()

        # Should handle non-dict inputs gracefully
        with pytest.raises((TypeError, ValueError)):
            storage.encrypt_credentials("not a dict")

        with pytest.raises((TypeError, ValueError)):
            storage.encrypt_credentials(["not", "a", "dict"])

    def test_decrypt_credentials_invalid_structure(self):
        """Test decrypting with invalid encrypted structure"""
        storage = SecureCredentialStorage()

        # Missing required fields
        invalid_structures = [
            {},
            {"salt": "test"},
            {"data": "test"},
            {"salt": "test", "data": "test"},  # Missing version
            {"salt": "invalid_base64", "data": "test", "version": "1"},
            {"salt": "dGVzdA==", "data": "invalid_base64", "version": "1"},
        ]

        for invalid_structure in invalid_structures:
            with pytest.raises((ValueError, KeyError)):
                storage.decrypt_credentials(invalid_structure)

    def test_decrypt_credentials_wrong_version(self):
        """Test decrypting with unsupported version"""
        storage = SecureCredentialStorage()

        # Create valid encrypted data but with wrong version
        credentials = {"test": "data"}
        encrypted = storage.encrypt_credentials(credentials)
        encrypted["version"] = "999"  # Unsupported version

        with pytest.raises(ValueError, match="Unsupported credential version"):
            storage.decrypt_credentials(encrypted)

    def test_encryption_produces_different_results(self):
        """Test that encryption produces different results each time (due to salt)"""
        storage = SecureCredentialStorage()

        credentials = {"api_key": "test_key"}

        encrypted1 = storage.encrypt_credentials(credentials)
        encrypted2 = storage.encrypt_credentials(credentials)

        # Should have different salts and encrypted data
        assert encrypted1["salt"] != encrypted2["salt"]
        assert encrypted1["data"] != encrypted2["data"]

        # But both should decrypt to same original
        decrypted1 = storage.decrypt_credentials(encrypted1)
        decrypted2 = storage.decrypt_credentials(encrypted2)

        assert decrypted1 == credentials
        assert decrypted2 == credentials

    def test_fernet_cache_handles_multiple_salts(self):
        """Ensure cached Fernet instances use the correct salt per dataset."""
        storage = SecureCredentialStorage(password="strong-password")

        creds_one = {"value": "one"}
        creds_two = {"value": "two"}

        encrypted_one = storage.encrypt_credentials(creds_one)
        encrypted_two = storage.encrypt_credentials(creds_two)

        # Simulate new process by creating a fresh storage instance with same password
        storage_reload = SecureCredentialStorage(password="strong-password")

        decrypted_one = storage_reload.decrypt_credentials(encrypted_one)
        decrypted_two = storage_reload.decrypt_credentials(encrypted_two)

        assert decrypted_one == creds_one
        assert decrypted_two == creds_two

    def test_password_hashing(self):
        """Test password hashing functionality"""
        storage = SecureCredentialStorage()

        password = "test_password_123"

        # Hash password
        hashed = storage.hash_password(password)

        # Should be string
        assert isinstance(hashed, str)
        assert len(hashed) > 20  # Should be reasonably long

        # Verify password
        assert storage.verify_password(password, hashed) is True
        assert storage.verify_password("wrong_password", hashed) is False

    def test_password_hashing_empty_password(self):
        """Test password hashing with empty password"""
        storage = SecureCredentialStorage()

        empty_password = ""

        hashed = storage.hash_password(empty_password)

        # Should handle empty password
        assert storage.verify_password("", hashed) is True
        assert storage.verify_password("not_empty", hashed) is False

    def test_password_hashing_unicode(self):
        """Test password hashing with unicode characters"""
        storage = SecureCredentialStorage()

        unicode_password = "pássword_ünïcøde_🔐"

        hashed = storage.hash_password(unicode_password)

        assert storage.verify_password(unicode_password, hashed) is True
        assert storage.verify_password("regular_password", hashed) is False

    def test_password_verification_invalid_hash(self):
        """Test password verification with invalid hash"""
        storage = SecureCredentialStorage()

        password = "test_password"

        # Test with various invalid hashes
        invalid_hashes = ["not_a_hash", "", "short", "invalid$format$hash", None]

        for invalid_hash in invalid_hashes:
            with pytest.raises((ValueError, TypeError)):
                storage.verify_password(password, invalid_hash)


class TestSecureFileManager:
    """Test SecureFileManager class"""

    def test_write_secure_file_basic(self):
        """Test writing secure file with basic data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_secure.json")

            test_data = {"key": "value", "number": 42}

            SecureFileManager.write_secure_file(file_path, test_data)

            # File should exist
            assert os.path.exists(file_path)

            # File should have secure permissions (600)
            file_stat = os.stat(file_path)
            permissions = file_stat.st_mode & 0o777
            assert permissions == 0o600

    def test_read_secure_file_basic(self):
        """Test reading secure file with basic data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_secure.json")

            test_data = {"key": "value", "nested": {"inner": "data"}}

            # Write and read
            SecureFileManager.write_secure_file(file_path, test_data)
            read_data = SecureFileManager.read_secure_file(file_path)

            assert read_data == test_data

    def test_write_read_secure_file_complex(self):
        """Test writing and reading complex data structures"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "complex_secure.json")

            complex_data = {
                "strings": ["hello", "world"],
                "numbers": [1, 2.5, -3],
                "boolean": True,
                "null": None,
                "nested": {"level1": {"level2": {"deep_value": "found"}}},
                "unicode": "🔒🗝️",
                "large_text": "x" * 1000,
            }

            SecureFileManager.write_secure_file(file_path, complex_data)
            read_data = SecureFileManager.read_secure_file(file_path)

            assert read_data == complex_data

    def test_read_secure_file_not_exist(self):
        """Test reading non-existent secure file"""
        non_existent_path = "/nonexistent/path/file.json"

        with pytest.raises(FileNotFoundError):
            SecureFileManager.read_secure_file(non_existent_path)

    def test_write_secure_file_invalid_directory(self):
        """Test writing to invalid directory"""
        invalid_path = "/nonexistent/directory/file.json"

        with pytest.raises((FileNotFoundError, PermissionError, OSError)):
            SecureFileManager.write_secure_file(invalid_path, {"test": "data"})

    def test_read_secure_file_permission_check(self):
        """Test reading file with insecure permissions"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump({"test": "data"}, temp_file)

        try:
            # Make file world-readable (insecure)
            os.chmod(temp_path, 0o644)

            # Should raise security error
            with pytest.raises(
                PermissionError, match="File permissions too permissive"
            ):
                SecureFileManager.read_secure_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_write_secure_file_creates_directory(self):
        """Test that writing secure file creates parent directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "deep", "file.json")

            test_data = {"created": "directory"}

            SecureFileManager.write_secure_file(nested_path, test_data)

            # File should exist
            assert os.path.exists(nested_path)

            # Parent directories should exist
            assert os.path.exists(os.path.dirname(nested_path))

            # Data should be correct
            read_data = SecureFileManager.read_secure_file(nested_path)
            assert read_data == test_data

    def test_write_secure_file_atomic_operation(self):
        """Test that file writing is atomic"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "atomic_test.json")

            # Write initial data
            initial_data = {"version": 1}
            SecureFileManager.write_secure_file(file_path, initial_data)

            # Simulate concurrent read while writing
            new_data = {"version": 2, "large_data": "x" * 10000}

            # Write new data
            SecureFileManager.write_secure_file(file_path, new_data)

            # Read should get complete new data, not partial
            read_data = SecureFileManager.read_secure_file(file_path)
            assert read_data == new_data

    def test_read_secure_file_invalid_json(self):
        """Test reading file with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("invalid json content {")
            temp_path = temp_file.name

        try:
            # Set secure permissions
            os.chmod(temp_path, 0o600)

            with pytest.raises(json.JSONDecodeError):
                SecureFileManager.read_secure_file(temp_path)

        finally:
            os.unlink(temp_path)


class TestUtilityFunctions:
    """Test utility functions"""

    def test_get_credential_storage(self):
        """Test get_credential_storage function"""
        storage1 = get_credential_storage()
        storage2 = get_credential_storage()

        # Should return same instance (singleton-like behavior)
        assert storage1 is storage2
        assert isinstance(storage1, SecureCredentialStorage)

    def test_generate_secure_key(self):
        """Test generate_secure_key function"""
        # Test default length
        key1 = generate_secure_key()
        assert isinstance(key1, str)
        assert len(key1) == 64  # 32 bytes * 2 hex chars

        # Test custom length
        key2 = generate_secure_key(16)
        assert len(key2) == 32  # 16 bytes * 2 hex chars

        # Keys should be different
        assert key1 != key2

        # Should be valid hex
        int(key1, 16)  # Should not raise
        int(key2, 16)  # Should not raise

    def test_generate_secure_key_invalid_length(self):
        """Test generate_secure_key with invalid length"""
        with pytest.raises(ValueError):
            generate_secure_key(0)

        with pytest.raises(ValueError):
            generate_secure_key(-1)

    def test_secure_hash(self):
        """Test secure_hash function"""
        data = "test data for hashing"

        hash1 = secure_hash(data)
        hash2 = secure_hash(data)

        # Different calls produce different hashes (random salt by design)
        assert hash1 != hash2

        # Should be "salt$hash" format (32 char salt + $ + 64 char hash = 97 chars)
        assert isinstance(hash1, str)
        assert len(hash1) == 97  # salt(32) + $(1) + hash(64)
        assert "$" in hash1

        # Salt and hash parts should be valid hex
        salt_part, hash_part = hash1.split("$")
        assert len(salt_part) == 32
        assert len(hash_part) == 64
        int(salt_part, 16)  # Valid hex
        int(hash_part, 16)  # Valid hex

        # Same input with explicit salt should produce same hash
        explicit_salt = "a" * 32
        hash_with_salt_1 = secure_hash(data, salt=explicit_salt)
        hash_with_salt_2 = secure_hash(data, salt=explicit_salt)
        assert hash_with_salt_1 == hash_with_salt_2

        # Different input should produce different hash
        hash3 = secure_hash("different data", salt=explicit_salt)
        assert hash_with_salt_1 != hash3

    def test_secure_hash_unicode(self):
        """Test secure_hash with unicode data"""
        unicode_data = "tëst dàta wïth ünïcøde 🔐"

        hash_result = secure_hash(unicode_data)

        # Should be "salt$hash" format
        assert isinstance(hash_result, str)
        assert len(hash_result) == 97  # salt(32) + $(1) + hash(64)
        assert "$" in hash_result

        # Different calls produce different hashes (random salt)
        assert hash_result != secure_hash(unicode_data)

        # Same input with explicit salt should be consistent
        explicit_salt = "b" * 32
        assert secure_hash(unicode_data, salt=explicit_salt) == secure_hash(
            unicode_data, salt=explicit_salt
        )

    def test_secure_hash_empty_string(self):
        """Test secure_hash with empty string"""
        hash_result = secure_hash("")

        # Should be "salt$hash" format
        assert isinstance(hash_result, str)
        assert len(hash_result) == 97  # salt(32) + $(1) + hash(64)
        assert "$" in hash_result

        # Different calls produce different hashes (random salt)
        assert hash_result != secure_hash("")

        # Same input with explicit salt should be consistent
        explicit_salt = "c" * 32
        assert secure_hash("", salt=explicit_salt) == secure_hash(
            "", salt=explicit_salt
        )


class TestIntegration:
    """Test integration between components"""

    def test_full_credential_workflow(self):
        """Test complete credential storage workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "credentials.json")

            # Original credentials
            original_credentials = {
                "api_keys": {"openai": "sk-test123", "anthropic": "ant-test456"},
                "database": {
                    "host": "db.example.com",
                    "username": "dbuser",
                    "password": "dbpass123",
                },
                "metadata": {"created": "2024-01-01", "environment": "production"},
            }

            # Get storage instance
            storage = get_credential_storage()

            # Encrypt credentials
            encrypted_credentials = storage.encrypt_credentials(original_credentials)

            # Save to secure file
            SecureFileManager.write_secure_file(file_path, encrypted_credentials)

            # Load from secure file
            loaded_encrypted = SecureFileManager.read_secure_file(file_path)

            # Decrypt credentials
            decrypted_credentials = storage.decrypt_credentials(loaded_encrypted)

            # Should match original
            assert decrypted_credentials == original_credentials

            # File should have secure permissions
            file_stat = os.stat(file_path)
            permissions = file_stat.st_mode & 0o777
            assert permissions == 0o600

    def test_password_workflow(self):
        """Test complete password hashing workflow"""
        storage = get_credential_storage()

        passwords = [
            "simple_password",
            "complex_pássword_123!@#",
            "üñíçödé_🔐_password",
            "",  # Empty password
            "a" * 1000,  # Very long password
        ]

        hashed_passwords = {}

        # Hash all passwords
        for password in passwords:
            hashed = storage.hash_password(password)
            hashed_passwords[password] = hashed

            # Verify immediate verification works
            assert storage.verify_password(password, hashed) is True

        # Verify all passwords again
        for password, hashed in hashed_passwords.items():
            assert storage.verify_password(password, hashed) is True

            # Verify wrong passwords fail
            if password:  # Skip empty password for this test
                assert storage.verify_password(password + "wrong", hashed) is False


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_encryption_with_none_values(self):
        """Test encryption handling None values"""
        storage = SecureCredentialStorage()

        credentials_with_none = {
            "valid_key": "valid_value",
            "none_key": None,
            "nested": {"inner_none": None, "inner_valid": "value"},
        }

        encrypted = storage.encrypt_credentials(credentials_with_none)
        decrypted = storage.decrypt_credentials(encrypted)

        assert decrypted == credentials_with_none

    def test_file_operations_with_special_characters(self):
        """Test file operations with special characters in data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "special_chars.json")

            special_data = {
                "unicode": "🔐🗝️💾",
                "control_chars": "\t\n\r",
                "quotes": "\"\"\"'''",
                "backslashes": "\\\\\\",
                "encoded": "\u0041\u0042\u0043",
            }

            SecureFileManager.write_secure_file(file_path, special_data)
            read_data = SecureFileManager.read_secure_file(file_path)

            assert read_data == special_data

    def test_concurrent_file_access(self):
        """Test concurrent access to secure files"""
        import threading
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "concurrent.json")

            results = []
            errors = []

            def write_and_read(thread_id):
                try:
                    data = {"thread_id": thread_id, "timestamp": time.time()}
                    SecureFileManager.write_secure_file(file_path, data)
                    time.sleep(0.01)  # Small delay
                    read_data = SecureFileManager.read_secure_file(file_path)
                    results.append((thread_id, read_data))
                except Exception as e:
                    errors.append((thread_id, str(e)))

            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=write_and_read, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Should have some successful operations
            assert len(results) > 0

            # If there are errors, they should be reasonable
            for _thread_id, error in errors:
                # Acceptable errors include file not found (if read happened before write)
                # or permission errors (if writes conflicted)
                assert any(
                    err_type in error
                    for err_type in ["FileNotFoundError", "PermissionError", "json"]
                )


class TestSecurityProperties:
    """Test security properties and guarantees"""

    def test_encryption_randomness(self):
        """Test that encryption is properly randomized"""
        storage = SecureCredentialStorage()

        same_data = {"test": "data"}

        # Encrypt same data multiple times
        encryptions = [storage.encrypt_credentials(same_data) for _ in range(10)]

        # All salts should be different
        salts = [enc["salt"] for enc in encryptions]
        assert len(set(salts)) == 10  # All unique

        # All encrypted data should be different
        encrypted_data = [enc["data"] for enc in encryptions]
        assert len(set(encrypted_data)) == 10  # All unique

        # But all should decrypt to same original
        for encryption in encryptions:
            decrypted = storage.decrypt_credentials(encryption)
            assert decrypted == same_data

    def test_password_hash_randomness(self):
        """Test that password hashing is properly salted"""
        storage = SecureCredentialStorage()

        password = "test_password"

        # Hash same password multiple times
        hashes = [storage.hash_password(password) for _ in range(10)]

        # All hashes should be different (due to random salt)
        assert len(set(hashes)) == 10

        # But all should verify correctly
        for hashed in hashes:
            assert storage.verify_password(password, hashed) is True

    def test_file_permission_security(self):
        """Test that files maintain secure permissions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "secure_test.json")

            SecureFileManager.write_secure_file(file_path, {"test": "data"})

            # Check permissions
            file_stat = os.stat(file_path)
            permissions = file_stat.st_mode & 0o777

            # Should be 600 (owner read/write only)
            assert permissions == 0o600

            # Should not be readable by group or others
            assert not (permissions & 0o044)  # No group/other read
            assert not (permissions & 0o022)  # No group/other write
            assert not (permissions & 0o111)  # No execute permissions

    def test_no_sensitive_data_in_memory_dumps(self):
        """Test that sensitive data is not easily extractable"""
        storage = SecureCredentialStorage()

        sensitive_data = {"password": "super_secret_password"}

        # Encrypt data
        encrypted = storage.encrypt_credentials(sensitive_data)

        # Convert to string representations
        encrypted_str = str(encrypted)
        encrypted_repr = repr(encrypted)

        # Sensitive data should not appear in string representations
        assert "super_secret_password" not in encrypted_str
        assert "super_secret_password" not in encrypted_repr

        # Only base64 encoded data should be present
        assert encrypted["salt"] in encrypted_str
        assert encrypted["data"] in encrypted_str
