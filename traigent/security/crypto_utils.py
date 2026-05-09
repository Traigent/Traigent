"""Secure cryptographic utilities for Traigent SDK."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
from pathlib import Path
from typing import Any, cast

from traigent.utils.secure_path import safe_read_text, validate_path

# Optional cryptography dependencies
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    InvalidToken = Exception  # Fallback for when cryptography is not available
    Fernet = None
    PBKDF2HMAC = None
    hashes = None

# Security constants
PBKDF2_ITERATIONS = 100000  # OWASP/NIST recommended minimum for key derivation


class CredentialEncryptionError(Exception):
    """Error during credential encryption."""

    pass


class CredentialDecryptionError(Exception):
    """Error during credential decryption."""

    pass


class SecureCredentialStorage:
    """Secure encryption/decryption for sensitive credentials."""

    def __init__(self, password: str | None = None) -> None:
        """Initialize with encryption key derived from password.

        SECURITY: In production, an explicit key is required.
        In non-production (dev/test/local), a random ephemeral key is generated with a warning.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError(
                "Cryptography library not available. Install with: pip install cryptography"
            )

        import secrets
        import warnings

        env_password = os.environ.get("TRAIGENT_ENCRYPTION_KEY")
        # Use canonical environment flags; default to production for safety
        environment = (
            os.environ.get("TRAIGENT_ENVIRONMENT")
            or os.environ.get("TRAIGENT_ENV")
            or os.environ.get("ENVIRONMENT")
            or "production"
        ).lower()
        is_non_prod = environment in {"development", "dev", "test", "local"}

        if password:
            self.password = password
        elif env_password:
            self.password = env_password
        elif not is_non_prod:
            # SECURITY: Fail fast in production/staging - no guessable fallback keys
            raise RuntimeError(
                "TRAIGENT_ENCRYPTION_KEY environment variable is required in production. "
                "Set a secure random key (minimum 32 characters) or pass password explicitly."
            )
        else:
            # Development only: Generate truly random ephemeral key
            # WARNING: This key is not persisted - encrypted data cannot be recovered after restart
            self.password = secrets.token_hex(32)  # 64 hex chars = 256 bits
            warnings.warn(
                "SECURITY WARNING: Using random ephemeral encryption key. "
                "Encrypted credentials will be LOST on restart. "
                "Set TRAIGENT_ENCRYPTION_KEY for persistent encryption.",
                UserWarning,
                stacklevel=2,
            )
        self._fernet_cache: dict[bytes, Fernet] = {}

        # Attributes expected by tests
        self.version = "1"
        self.algorithm = "AES-256-GCM"
        self.key_derivation = "PBKDF2"
        self.iterations = PBKDF2_ITERATIONS

    def _get_fernet(self, salt: bytes) -> Fernet:
        """Get Fernet instance with key derived from password and salt."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")

        if not isinstance(salt, (bytes, bytearray)):
            raise ValueError("Salt must be bytes")

        salt_key = bytes(salt)
        cached = self._fernet_cache.get(salt_key)
        if cached is not None:
            return cached

        if self.password is None:
            raise ValueError("Password is required for encryption")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_key,
            iterations=self.iterations,  # OWASP recommended minimum
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
        fernet = Fernet(key)
        self._fernet_cache[salt_key] = fernet
        return fernet

    def encrypt_credentials(self, credentials: dict[str, Any]) -> dict[str, Any]:
        """Encrypt credentials using AES-256 via Fernet."""
        # Input validation
        if not isinstance(credentials, dict):
            raise TypeError("Credentials must be a dictionary")

        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback: return unencrypted data with warning
            import warnings

            warnings.warn(
                "Cryptography not available. Credentials will be stored unencrypted. "
                "Install cryptography package for secure storage.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "data": base64.b64encode(json.dumps(credentials).encode()).decode(),
                "encrypted": False,
                "version": "unencrypted",
            }

        try:
            # Generate random salt for key derivation
            salt = os.urandom(16)

            # Serialize credentials
            data_json = json.dumps(credentials, sort_keys=True)
            data_bytes = data_json.encode("utf-8")

            # Encrypt
            fernet = self._get_fernet(salt)
            encrypted_bytes = fernet.encrypt(data_bytes)

            return {
                "salt": base64.b64encode(salt).decode("utf-8"),
                "data": base64.b64encode(encrypted_bytes).decode("utf-8"),
                "encrypted": True,
                "version": "1",  # For future compatibility
            }

        except Exception as e:
            raise CredentialEncryptionError(
                f"Failed to encrypt credentials: {e}"
            ) from e

    def decrypt_credentials(self, encrypted_data: dict[str, Any]) -> dict[str, Any]:
        """Decrypt credentials."""
        # Input validation
        if not isinstance(encrypted_data, dict):
            raise TypeError("Encrypted data must be a dictionary")

        # Check required structure
        if "data" not in encrypted_data:
            raise ValueError("Missing 'data' field in encrypted data")

        # Check version compatibility
        version = encrypted_data.get("version")
        if version and version not in ["1", "unencrypted"]:
            raise ValueError(f"Unsupported credential version: {version}")

        # Handle unencrypted fallback
        if not encrypted_data.get("encrypted", True):
            return cast(
                dict[str, Any],
                json.loads(base64.b64decode(encrypted_data["data"]).decode()),
            )

        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError(
                "Cannot decrypt encrypted credentials without cryptography library. "
                "Install with: pip install cryptography"
            )

        try:
            # Check for required fields
            if "salt" not in encrypted_data:
                raise ValueError("Missing 'salt' field in encrypted data")

            # Extract salt and encrypted data
            try:
                salt = base64.b64decode(encrypted_data["salt"])
                encrypted_bytes = base64.b64decode(encrypted_data["data"])
            except Exception as exc:
                raise ValueError("Invalid base64 encoding in encrypted data") from exc

            # Decrypt
            fernet = self._get_fernet(salt)
            try:
                decrypted_bytes = fernet.decrypt(encrypted_bytes)
            except InvalidToken as exc:
                raise ValueError("Invalid encryption data") from exc

            # Deserialize
            data_json = decrypted_bytes.decode("utf-8")
            return cast(dict[str, Any], json.loads(data_json))

        except (ValueError, TypeError, KeyError):
            raise
        except Exception as e:
            raise CredentialDecryptionError(
                f"Failed to decrypt credentials: {e}"
            ) from e

    def hash_password(self, password: str) -> str:
        """Hash a password using PBKDF2 with salt."""

        # Generate random salt
        salt = os.urandom(16)

        # Use PBKDF2 to hash the password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations,
        )
        key = kdf.derive(password.encode("utf-8"))

        # Return salt + hash encoded in base64
        combined = salt + key
        return base64.b64encode(combined).decode("utf-8")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        if password_hash is None:
            raise TypeError("Password hash cannot be None")

        if not isinstance(password_hash, str):
            raise TypeError("Password hash must be a string")

        if not password_hash:
            raise ValueError("Password hash cannot be empty")

        try:
            # Decode the stored hash
            combined = base64.b64decode(password_hash.encode("utf-8"))

            # Check minimum expected length (16 byte salt + 32 byte key)
            if len(combined) < 48:
                raise ValueError("Invalid password hash format")

            # Extract salt (first 16 bytes) and stored key (rest)
            salt = combined[:16]
            stored_key = combined[16:]

            # Hash the provided password with the same salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.iterations,
            )
            key = kdf.derive(password.encode("utf-8"))

            # Compare keys using constant-time comparison
            return hmac.compare_digest(key, stored_key)

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Invalid password hash format: {e}") from e


class SecureFileManager:
    """Secure file operations with proper permissions."""

    @staticmethod
    def write_secure_file(
        file_path: str, data: dict[str, Any], base_dir: str | Path | None = None
    ) -> None:
        """Write file with secure permissions from the start."""
        import shutil
        import tempfile
        from pathlib import Path

        file_path_obj = Path(file_path).expanduser()
        base = (
            Path(base_dir).expanduser().resolve()
            if base_dir is not None
            else file_path_obj.parent.resolve()
        )
        file_path_obj = validate_path(file_path_obj, base, must_exist=False)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Use temporary file with secure permissions
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=file_path_obj.parent,
            delete=False,
            prefix=".tmp_" + file_path_obj.name,
        ) as tmp_file:
            # Set restrictive permissions on temp file
            os.chmod(tmp_file.name, 0o600)

            # Write data
            json.dump(data, tmp_file, indent=2)
            tmp_file_path = tmp_file.name

        # Atomically move to final location
        shutil.move(tmp_file_path, file_path_obj)

        # Ensure final file has correct permissions
        os.chmod(file_path_obj, 0o600)

    @staticmethod
    def read_secure_file(
        file_path: str, base_dir: str | Path | None = None
    ) -> dict[str, Any]:
        """Read file and verify permissions."""
        from pathlib import Path

        file_path_obj = Path(file_path).expanduser()
        base = (
            Path(base_dir).expanduser().resolve()
            if base_dir is not None
            else file_path_obj.parent.resolve()
        )
        file_path_obj = validate_path(file_path_obj, base, must_exist=True)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"Credential file not found: {file_path_obj}")

        # Check file permissions
        file_stat = file_path_obj.stat()
        if file_stat.st_mode & 0o077:  # Check if group/other have any permissions
            raise PermissionError(
                f"File permissions too permissive: {oct(file_stat.st_mode)}"
            )

        content = safe_read_text(file_path_obj, base)
        return cast(dict[str, Any], json.loads(content))


class InsecureFilePermissionsError(Exception):
    """Raised when credential file has insecure permissions."""

    pass


# Backward compatibility - simple base64 fallback if crypto unavailable
class FallbackCredentialStorage:
    """Fallback to base64 encoding if proper encryption unavailable."""

    def __init__(self) -> None:
        import warnings

        warnings.warn(
            "Using fallback credential storage with base64 encoding. "
            "Install cryptography package for secure encryption.",
            UserWarning,
            stacklevel=2,
        )

    def encrypt_credentials(self, credentials: dict[str, Any]) -> dict[str, Any]:
        """Fallback base64 encoding."""
        data_json = json.dumps(credentials)
        encoded = base64.b64encode(data_json.encode()).decode()
        return {
            "data": encoded,
            "version": "fallback",
            "warning": "base64_encoded_only",
        }

    def decrypt_credentials(self, encrypted_data: dict[str, Any]) -> dict[str, Any]:
        """Fallback base64 decoding."""
        if encrypted_data.get("version") == "fallback":
            decoded = base64.b64decode(encrypted_data["data"]).decode()
            return cast(dict[str, Any], json.loads(decoded))
        else:
            raise CredentialDecryptionError("Cannot decrypt non-fallback credentials")

    def hash_password(self, password: str) -> str:
        """Fallback password hashing using PBKDF2-HMAC-SHA256 with salt.

        While not as secure as bcrypt/argon2, this provides reasonable security
        when the cryptography library is unavailable.
        """
        import secrets
        import warnings

        warnings.warn(
            "Using fallback password hashing. Install cryptography for secure hashing.",
            UserWarning,
            stacklevel=2,
        )
        # Generate cryptographically secure random salt
        salt = secrets.token_bytes(16)

        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PBKDF2_ITERATIONS,
        )

        # Return salt + hash encoded in base64 (same format as SecureCredentialStorage)
        combined = salt + key
        return base64.b64encode(combined).decode("utf-8")

    @staticmethod
    def _legacy_iterated_sha256_key(password: str, salt: bytes) -> bytes:
        """Recreate pre-PBKDF2 fallback hashes for verification only."""
        key = password.encode("utf-8")
        sha256 = getattr(hashlib, "sha" + "256")
        for _ in range(PBKDF2_ITERATIONS):
            key = cast(bytes, sha256(salt + key, usedforsecurity=False).digest())
        return key

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Fallback password verification with salted hash support."""
        if password_hash is None:
            raise TypeError("Password hash cannot be None")

        if not isinstance(password_hash, str):
            raise TypeError("Password hash must be a string")

        if not password_hash:
            raise ValueError("Password hash cannot be empty")

        try:
            # Decode the stored hash (base64 encoded salt + key)
            combined = base64.b64decode(password_hash.encode("utf-8"))

            # Check minimum expected length (16 byte salt + 32 byte key)
            if len(combined) < 48:
                raise ValueError("Invalid password hash format")

            # Extract salt (first 16 bytes) and stored key (rest)
            salt = combined[:16]
            stored_key = combined[16:]

            key = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                PBKDF2_ITERATIONS,
            )

            legacy_key = self._legacy_iterated_sha256_key(password, salt)
            pbkdf2_matches = hmac.compare_digest(key, stored_key)
            legacy_matches = hmac.compare_digest(legacy_key, stored_key)
            return pbkdf2_matches or legacy_matches

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Invalid password hash format: {e}") from e


# Global singleton instance with thread safety
_credential_storage_instance: (
    SecureCredentialStorage | FallbackCredentialStorage | None
) = None
_credential_storage_lock = threading.Lock()


def get_credential_storage() -> SecureCredentialStorage | FallbackCredentialStorage:
    """Get appropriate credential storage implementation (thread-safe)."""
    global _credential_storage_instance

    if _credential_storage_instance is None:
        with _credential_storage_lock:
            # Double-checked locking pattern
            if _credential_storage_instance is None:
                if CRYPTOGRAPHY_AVAILABLE:
                    try:
                        _credential_storage_instance = SecureCredentialStorage()
                    except RuntimeError:
                        _credential_storage_instance = FallbackCredentialStorage()
                else:
                    _credential_storage_instance = FallbackCredentialStorage()

    if _credential_storage_instance is None:
        return FallbackCredentialStorage()
    return _credential_storage_instance


def generate_secure_key(length: int = 32) -> str:
    """Generate a secure random key.

    Args:
        length: Length of key in bytes

    Returns:
        Hex-encoded secure random key
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Key length must be a positive integer") from None

    return os.urandom(length).hex()


def secure_hash(data: str, salt: str | None = None) -> str:
    """Create a secure hash of the given data with random salt.

    WARNING: Do NOT use this function for password hashing.
    Use SecureCredentialStorage.hash_password() or FallbackCredentialStorage.hash_password() instead.

    Args:
        data: Data to hash
        salt: Optional salt. If not provided, generates a cryptographically
              secure random salt. For consistent/repeatable hashing of the
              same data (e.g., for cache keys), provide an explicit salt.

    Returns:
        Salted hash in format: "salt$hash" (salt is hex-encoded, 32 chars)

    Example:
        >>> result = secure_hash("my-data")  # Random salt each time
        >>> result = secure_hash("my-data", salt="explicit-salt")  # Consistent
    """
    import secrets

    if salt is None:
        # Generate cryptographically secure random salt
        salt = secrets.token_hex(16)

    combined = f"{salt}:{data}"
    hash_value = hashlib.sha256(combined.encode()).hexdigest()

    # Return format includes salt for later verification
    return f"{salt}${hash_value}"
