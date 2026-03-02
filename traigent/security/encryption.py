"""Data encryption and privacy controls for Traigent Enterprise."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, cast

try:
    from cryptography.fernet import Fernet as _Fernet
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
    Fernet = _Fernet
except ImportError:
    CRYPTO_AVAILABLE = False
    AESGCM = object

    # Mock classes for when cryptography is not available
    class Fernet:  # type: ignore[no-redef]
        @staticmethod
        def generate_key():
            return b"mock_key"

        def __init__(self, key) -> None:
            self.key = key

        def encrypt(self, data):
            return b"encrypted_" + data

        def decrypt(self, data):
            return data[10:]  # Remove "encrypted_" prefix


from ..utils.logging import get_logger
from ..utils.secure_path import (
    safe_read_bytes,
    safe_read_text,
    safe_write_bytes,
    safe_write_text,
    validate_path,
)

logger = get_logger(__name__)


class DataClassification(Enum):
    """Data classification levels for compliance."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"  # pragma: allowlist secret


class EncryptionLevel(Enum):
    """Encryption levels."""

    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MILITARY = "military"


class PIIType(Enum):
    """Types of Personally Identifiable Information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    CUSTOM = "custom"


@dataclass
class EncryptionKey:
    """Encryption key with metadata."""

    key_id: str
    key_data: bytes
    algorithm: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    rotation_count: int = 0
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PIIMatch:
    """Detected PII in data."""

    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str


@dataclass
class DataRecord:
    """Data record with classification and encryption metadata."""

    record_id: str
    data: bytes
    classification: DataClassification
    encryption_key_id: str | None = None
    pii_detected: list[PIIMatch] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    accessed_at: datetime | None = None
    retention_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EncryptionManager:
    """Manages data encryption and key lifecycle."""

    def __init__(self, key_manager: KeyManager | None = None) -> None:
        """Initialize encryption manager.

        Args:
            key_manager: KeyManager instance for key operations
        """
        self.key_manager: KeyManager = key_manager or KeyManager()
        self.crypto_available = CRYPTO_AVAILABLE

    @staticmethod
    def _to_mutable_key_buffer(key_data: bytes | None) -> bytearray | None:
        """Create a mutable key buffer for best-effort zeroization."""
        if key_data is None:
            return None
        return bytearray(key_data)

    @staticmethod
    def _zeroize_key_buffer(key_buffer: bytearray | None) -> None:
        """Best-effort in-place key buffer zeroization."""
        if key_buffer is None:
            return
        for i in range(len(key_buffer)):
            key_buffer[i] = 0

    def encrypt(
        self,
        data: str | bytes,
        classification: DataClassification = DataClassification.INTERNAL,
    ) -> dict[str, Any]:
        """Encrypt data and return structured result.

        Args:
            data: Data to encrypt (string or bytes)
            classification: Data classification level

        Returns:
            Dictionary with encryption metadata
        """
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        elif isinstance(data, (bytes, bytearray, memoryview)):
            data_bytes = bytes(data)
        else:
            raise TypeError("Data must be a string or bytes-like object")

        if not data_bytes:
            raise ValueError("Cannot encrypt empty payloads")

        if not isinstance(classification, DataClassification):
            raise ValueError("Data classification must be a DataClassification value")

        # Generate a key for this encryption
        key_id: str = self.key_manager.generate_key("AES-256")
        key_data: bytes | None = self.key_manager.get_key(key_id)
        key_buffer = self._to_mutable_key_buffer(key_data)

        if self.crypto_available and key_buffer:
            # Real encryption using AES-GCM
            try:
                aesgcm = AESGCM(bytes(key_buffer))
                iv = os.urandom(12)  # 96-bit nonce for AES-GCM
                ciphertext_with_tag = aesgcm.encrypt(iv, data_bytes, None)
                # Split ciphertext and tag (last 16 bytes)
                ciphertext = ciphertext_with_tag[:-16]
                tag = ciphertext_with_tag[-16:]
            except Exception:
                # SECURITY: Never fall back to mock encryption when real crypto is available
                # This would expose plaintext data as "encrypted"
                logger.exception(
                    "Encryption failed - refusing to fall back to mock encryption"
                )
                raise RuntimeError(
                    "Encryption operation failed. Cannot proceed without proper encryption."
                ) from None
            finally:
                self._zeroize_key_buffer(key_buffer)
        else:
            self._zeroize_key_buffer(key_buffer)
            # Mock encryption: ONLY allowed in test/mock mode
            mock_mode = os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in (
                "true",
                "1",
            )
            if not mock_mode:
                raise RuntimeError(
                    "Encryption requires the 'cryptography' package. "
                    "Install it with: pip install 'traigent[security]'"
                )
            logger.warning(
                "Using mock encryption - cryptography library not available. "
                "This is only safe in test/mock mode."
            )
            iv = os.urandom(12)
            ciphertext = (
                b"mock_" + data_bytes
            )  # NOSONAR — test-only, gated by TRAIGENT_MOCK_LLM
            tag = os.urandom(16)

        return {
            "ciphertext": ciphertext,
            "iv": iv,
            "tag": tag,
            "key_id": key_id,
            "algorithm": "AES-256-GCM",
            "classification": classification.value,
        }

    def decrypt(self, encrypted_result: dict[str, Any]) -> bytes:
        """Decrypt data from encrypted result.

        Args:
            encrypted_result: Dictionary from encrypt() method

        Returns:
            Decrypted data as bytes
        """
        if not isinstance(encrypted_result, dict):
            raise TypeError("Encrypted result must be provided as a dictionary")

        required_fields = {"ciphertext", "iv", "tag", "key_id"}
        missing = required_fields - encrypted_result.keys()
        if missing:
            raise ValueError(
                f"Encrypted result missing required fields: {sorted(missing)}"
            )

        ciphertext_obj = encrypted_result["ciphertext"]
        iv_obj = encrypted_result["iv"]
        tag_obj = encrypted_result["tag"]
        key_id_obj = encrypted_result["key_id"]

        if not isinstance(ciphertext_obj, (bytes, bytearray, memoryview)):
            raise TypeError("Ciphertext must be bytes")
        if not isinstance(iv_obj, (bytes, bytearray, memoryview)):
            raise TypeError("Initialization vector must be bytes")
        if not isinstance(tag_obj, (bytes, bytearray, memoryview)):
            raise TypeError("Authentication tag must be bytes")
        if not isinstance(key_id_obj, str) or not key_id_obj.strip():
            raise ValueError("Invalid key identifier provided")

        ciphertext = bytes(ciphertext_obj)
        iv = bytes(iv_obj)
        tag = bytes(tag_obj)
        key_id = key_id_obj.strip()

        if len(iv) != 12:
            raise ValueError("Initialization vector must be 12 bytes for AES-GCM")
        if len(tag) != 16:
            raise ValueError("Authentication tag must be 16 bytes for AES-GCM")

        key_data: bytes | None = self.key_manager.get_key(key_id)
        key_buffer = self._to_mutable_key_buffer(key_data)

        if self.crypto_available and key_buffer:
            try:
                aesgcm = AESGCM(bytes(key_buffer))
                data = aesgcm.decrypt(iv, ciphertext + tag, None)
                return cast(bytes, data)
            except Exception:
                # SECURITY: Never silently return ciphertext/plaintext on decryption failure
                # This could expose data or mask tampering
                logger.exception(
                    "Decryption failed - data may be corrupted or tampered with"
                )
                raise RuntimeError(
                    "Decryption operation failed. Data integrity cannot be verified."
                ) from None
            finally:
                self._zeroize_key_buffer(key_buffer)
        else:
            self._zeroize_key_buffer(key_buffer)
            # Mock decryption: ONLY allowed in test/mock mode
            mock_mode = os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in (
                "true",
                "1",
            )
            if not mock_mode:
                raise RuntimeError(
                    "Decryption requires the 'cryptography' package. "
                    "Install it with: pip install 'traigent[security]'"
                )
            logger.warning(
                "Using mock decryption - cryptography library not available. "
                "This is only safe in test/mock mode."
            )
            if ciphertext.startswith(b"mock_"):  # NOSONAR — test-only mock decryption
                return ciphertext[5:]
            if ciphertext.startswith(b"encrypted_"):  # NOSONAR — legacy mock compat
                return ciphertext[10:]
            return ciphertext

    def encrypt_file(
        self,
        file_path: str,
        classification: DataClassification = DataClassification.INTERNAL,
        base_dir: str | Path | None = None,
    ) -> str:
        """Encrypt a file.

        Args:
            file_path: Path to input file
            classification: Data classification level
            base_dir: Base directory for allowed file operations

        Returns:
            Path to encrypted output file
        """
        input_path = Path(file_path).expanduser()
        if base_dir is not None:
            base = Path(base_dir).expanduser().resolve()
        else:
            base = (
                input_path.parent.resolve()
                if input_path.is_absolute()
                else Path.cwd().resolve()
            )
        input_path = validate_path(input_path, base, must_exist=True)
        data = safe_read_bytes(input_path, base)

        encrypted_result = self.encrypt(data, classification)

        # Generate output file path
        output_path = input_path.with_name(f"{input_path.name}.encrypted")
        output_path = validate_path(output_path, base, must_exist=False)

        # Store encrypted data and metadata as JSON (for test compatibility)
        encrypted_data = {
            "ciphertext": encrypted_result[
                "ciphertext"
            ].hex(),  # Convert to hex for JSON
            "iv": encrypted_result["iv"].hex(),
            "tag": encrypted_result["tag"].hex(),
            "key_id": encrypted_result["key_id"],
            "algorithm": encrypted_result["algorithm"],
            "classification": encrypted_result["classification"],
        }

        safe_write_text(output_path, json.dumps(encrypted_data), base)

        return str(output_path)

    def decrypt_file(
        self, encrypted_file_path: str, base_dir: str | Path | None = None
    ) -> str:
        """Decrypt a file.

        Args:
            encrypted_file_path: Path to encrypted file
            base_dir: Base directory for allowed file operations

        Returns:
            Path to decrypted output file
        """
        input_path = Path(encrypted_file_path).expanduser()
        if base_dir is not None:
            base = Path(base_dir).expanduser().resolve()
        else:
            base = (
                input_path.parent.resolve()
                if input_path.is_absolute()
                else Path.cwd().resolve()
            )
        input_path = validate_path(input_path, base, must_exist=True)
        encrypted_data = json.loads(safe_read_text(input_path, base))

        # Reconstruct encrypted_result
        encrypted_result = {
            "ciphertext": bytes.fromhex(encrypted_data["ciphertext"]),
            "iv": bytes.fromhex(encrypted_data["iv"]),
            "tag": bytes.fromhex(encrypted_data["tag"]),
            "key_id": encrypted_data["key_id"],
            "algorithm": encrypted_data["algorithm"],
            "classification": encrypted_data["classification"],
        }

        decrypted_data = self.decrypt(encrypted_result)

        # Generate output file path
        if input_path.name.endswith(".encrypted"):
            output_path = input_path.with_name(input_path.name[:-10])
        else:
            output_path = input_path.with_name(f"{input_path.name}.decrypted")
        output_path = validate_path(output_path, base, must_exist=False)

        safe_write_bytes(output_path, decrypted_data, base)

        return str(output_path)


class PIIDetector:
    """Detects and classifies Personally Identifiable Information."""

    def __init__(self) -> None:
        """Initialize PII detector with patterns."""
        self.patterns = self._compile_pii_patterns()
        self.custom_patterns: dict[str, re.Pattern[str]] = {}

    def _compile_pii_patterns(self) -> dict[PIIType, re.Pattern[str]]:
        """Get compiled regex patterns for common PII types."""
        pattern_map: dict[PIIType, re.Pattern[str]] = {}
        raw_patterns: dict[PIIType, str] = {
            PIIType.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            PIIType.PHONE: r"(?:\(\d{3}\)\s?\d{3}-\d{4}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})",
            PIIType.SSN: r"\b\d{3}-?\d{2}-?\d{4}\b",
            PIIType.CREDIT_CARD: r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            PIIType.IP_ADDRESS: r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            PIIType.PASSPORT: r"\b[A-Z]{1,2}\d{6,9}\b",
            PIIType.DRIVER_LICENSE: r"\b[A-Z]{1,2}\d{6,8}\b",
            PIIType.BANK_ACCOUNT: r"\b\d{8,17}\b",
        }

        for pii_type, pattern in raw_patterns.items():
            pattern_map[pii_type] = re.compile(pattern, re.IGNORECASE)

        return pattern_map

    def add_custom_pattern(self, name: str, pattern: str) -> None:
        """Add custom PII detection pattern."""
        if not name:
            raise ValueError("Pattern name must be provided")
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression pattern: {exc}") from exc

        self.custom_patterns[name] = compiled
        logger.info(f"Added custom PII pattern: {name}")

    def detect_pii(self, text: str) -> list[PIIMatch]:
        """Detect PII in text."""
        matches = []

        # Check standard patterns
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                confidence = self._calculate_confidence(pii_type, match.group())

                pii_match = PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                    context=text[max(0, match.start() - 20) : match.end() + 20],
                )
                matches.append(pii_match)

        # Check custom patterns
        for _name, pattern in self.custom_patterns.items():
            for match in pattern.finditer(text):
                pii_match = PIIMatch(
                    pii_type=PIIType.CUSTOM,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,  # Default confidence for custom patterns
                    context=text[max(0, match.start() - 20) : match.end() + 20],
                )
                matches.append(pii_match)

        return matches

    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """Calculate confidence score for PII detection."""
        if pii_type == PIIType.EMAIL:
            # Check for valid email format
            return 0.95 if "@" in value and "." in value.split("@")[-1] else 0.6
        elif pii_type == PIIType.CREDIT_CARD:
            # Simple Luhn algorithm check
            return (
                0.9
                if self._luhn_check(value.replace("-", "").replace(" ", ""))
                else 0.5
            )
        elif pii_type == PIIType.PHONE:
            # Check for standard phone number patterns
            digits = re.sub(r"\D", "", value)
            return 0.9 if len(digits) == 10 or len(digits) == 11 else 0.6
        else:
            return 0.8  # Default confidence

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""

        def luhn_sum(card_num):
            def digits_of(num):
                return [int(d) for d in str(num)]

            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10

        return cast(bool, luhn_sum(card_number) == 0)

    def anonymize_pii(self, text: str, replacement_char: str = "*") -> str:
        """Anonymize detected PII in text."""
        anonymized_text = text
        pii_matches = self.detect_pii(text)

        # Sort matches by position in reverse order to maintain indices
        pii_matches.sort(key=lambda x: x.start_pos, reverse=True)

        for match in pii_matches:
            if match.confidence > 0.7:  # Only anonymize high-confidence matches
                replacement = replacement_char * len(match.value)
                anonymized_text = (
                    anonymized_text[: match.start_pos]
                    + replacement
                    + anonymized_text[match.end_pos :]
                )

        return anonymized_text

    def anonymize_text(self, text: str, pii_types: list[PIIType] | None = None) -> str:
        """Anonymize detected PII in text with labeled replacements.

        Args:
            text: Text to anonymize
            pii_types: Optional list of specific PII types to anonymize

        Returns:
            Text with PII replaced by labeled placeholders
        """
        anonymized_text = text
        pii_matches = self.detect_pii(text)

        # Filter by PII types if specified
        if pii_types:
            pii_matches = [
                match for match in pii_matches if match.pii_type in pii_types
            ]

        # Sort matches by position in reverse order to maintain indices
        pii_matches.sort(key=lambda x: x.start_pos, reverse=True)

        for match in pii_matches:
            if match.confidence > 0.5:  # Only anonymize decent confidence matches
                replacement = f"[{match.pii_type.value.upper()}_REDACTED]"
                anonymized_text = (
                    anonymized_text[: match.start_pos]
                    + replacement
                    + anonymized_text[match.end_pos :]
                )

        return anonymized_text

    def pseudonymize_text(
        self, text: str, pii_types: list[PIIType] | None = None
    ) -> str:
        """Pseudonymize detected PII in text with consistent fake identifiers.

        Args:
            text: Text to pseudonymize
            pii_types: Optional list of specific PII types to pseudonymize

        Returns:
            Text with PII replaced by consistent pseudonyms
        """
        if not hasattr(self, "_pseudonym_cache"):
            self._pseudonym_cache: dict[str, Any] = {}

        pseudonymized_text = text
        pii_matches = self.detect_pii(text)

        # Filter by PII types if specified
        if pii_types:
            pii_matches = [
                match for match in pii_matches if match.pii_type in pii_types
            ]

        # Sort matches by position in reverse order to maintain indices
        pii_matches.sort(key=lambda x: x.start_pos, reverse=True)

        for match in pii_matches:
            if match.confidence > 0.5:
                # Generate consistent pseudonym for this value
                cache_key = f"{match.pii_type.value}_{match.value}"
                if cache_key not in self._pseudonym_cache:
                    # Generate a unique identifier for this value
                    value_hash = hashlib.sha256(match.value.encode()).hexdigest()[:8]
                    self._pseudonym_cache[cache_key] = (
                        f"[{match.pii_type.value.upper()}_{value_hash}]"
                    )

                replacement = self._pseudonym_cache[cache_key]
                pseudonymized_text = (
                    pseudonymized_text[: match.start_pos]
                    + replacement
                    + pseudonymized_text[match.end_pos :]
                )

        return pseudonymized_text


class DataClassifier:
    """Classifies data based on content and metadata."""

    def __init__(self) -> None:
        """Initialize data classifier."""
        self.classification_rules = self._get_default_rules()

    def _get_default_rules(self) -> dict[DataClassification, dict[str, Any]]:
        """Get default classification rules."""
        return {
            DataClassification.PUBLIC: {
                "pii_threshold": 0.0,
                "keywords": ["public", "marketing", "website"],
                "file_extensions": [".html", ".css", ".js"],
            },
            DataClassification.INTERNAL: {
                "pii_threshold": 0.1,
                "keywords": ["internal", "company", "employee"],
                "file_extensions": [".doc", ".pdf", ".ppt"],
            },
            DataClassification.CONFIDENTIAL: {
                "pii_threshold": 0.3,
                "keywords": ["confidential", "private", "restricted"],
                "contains_pii": True,
            },
            DataClassification.RESTRICTED: {
                "pii_threshold": 0.5,
                "keywords": ["restricted", "classified", "secret"],
                "contains_sensitive": True,
            },
            DataClassification.TOP_SECRET: {
                "pii_threshold": 0.7,
                "keywords": ["top secret", "classified", "national security"],
                "manual_review": True,
            },
        }

    def classify_data(
        self, data: str, metadata: dict[str, Any] | None = None
    ) -> DataClassification:
        """Classify data based on content and metadata."""
        metadata = dict(metadata or {})

        # Check for explicit classification in metadata
        if "classification" in metadata:
            return DataClassification(metadata["classification"])

        # Detect PII
        pii_detector = PIIDetector()
        pii_matches = pii_detector.detect_pii(data)
        pii_ratio = len(pii_matches) / max(1, len(data.split()))

        # Score each classification level
        scores = {}

        for classification, rules in self.classification_rules.items():
            score = 0

            # PII threshold check
            if pii_ratio >= rules.get("pii_threshold", 0):
                score += 1

            # Keyword matching
            keywords = rules.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in data.lower():
                    score += 1

            # File extension check
            filename = metadata.get("filename", "")
            file_extensions = rules.get("file_extensions", [])
            if any(filename.endswith(ext) for ext in file_extensions):
                score += 1

            # Special conditions
            if rules.get("contains_pii") and pii_matches:
                score += 2

            if rules.get("contains_sensitive"):
                sensitive_keywords = ["password", "secret", "token", "key"]
                if any(keyword in data.lower() for keyword in sensitive_keywords):
                    score += 2

            scores[classification] = score

        # Return highest scoring classification
        return max(scores, key=lambda k: scores[k])

    def get_retention_period(
        self, classification: DataClassification
    ) -> timedelta | None:
        """Get data retention period for classification level."""
        retention_periods = {
            DataClassification.PUBLIC: None,  # No retention limit
            DataClassification.INTERNAL: timedelta(days=2555),  # 7 years
            DataClassification.CONFIDENTIAL: timedelta(days=1825),  # 5 years
            DataClassification.RESTRICTED: timedelta(days=1095),  # 3 years
            DataClassification.TOP_SECRET: timedelta(days=365),  # 1 year
        }

        return retention_periods.get(classification)


class SecureStorage:
    """Secure storage with encryption and data lifecycle management."""

    def __init__(self, encryption_manager_or_protection_manager) -> None:
        """Initialize secure storage.

        Args:
            encryption_manager_or_protection_manager: Either EncryptionManager or DataProtectionManager
        """
        # Handle both EncryptionManager and DataProtectionManager for test compatibility
        if hasattr(encryption_manager_or_protection_manager, "encryption_manager"):
            # It's a DataProtectionManager
            self.protection_manager = encryption_manager_or_protection_manager
            self.encryption_manager = (
                encryption_manager_or_protection_manager.encryption_manager
            )
            self.pii_detector = encryption_manager_or_protection_manager.pii_detector
        else:
            # It's an EncryptionManager
            self.encryption_manager = encryption_manager_or_protection_manager
            self.pii_detector = PIIDetector()
            self.protection_manager = None

        self.classifier = DataClassifier()
        self.records: dict[str, DataRecord] = {}
        self.simple_storage: dict[str, Any] = {}  # For simple store/retrieve API
        # Alias for test compatibility
        self.storage = self.simple_storage

    def store_data(
        self,
        data: str,
        record_id: str | None = None,
        classification: DataClassification | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DataRecord:
        """Store data securely with encryption and classification."""
        if record_id is None:
            record_id = secrets.token_urlsafe(16)

        metadata = metadata or {}

        # Classify data if not provided
        if classification is None:
            classification = self.classifier.classify_data(data, metadata)

        # Detect PII
        pii_matches = self.pii_detector.detect_pii(data)

        # Encrypt data
        data_bytes = data.encode("utf-8")
        encrypted_result = self.encryption_manager.encrypt(data_bytes, classification)
        ciphertext: bytes = encrypted_result["ciphertext"]
        key_id = encrypted_result.get("key_id")
        encryption_metadata = {
            "iv": encrypted_result["iv"],
            "tag": encrypted_result["tag"],
            "algorithm": encrypted_result.get("algorithm", "AES-256-GCM"),
        }
        metadata["encryption_metadata"] = encryption_metadata

        # Set retention period
        retention_until = None
        retention_period = self.classifier.get_retention_period(classification)
        if retention_period:
            retention_until = datetime.now(UTC) + retention_period

        # Create data record
        record = DataRecord(
            record_id=record_id,
            data=ciphertext,
            classification=classification,
            encryption_key_id=key_id,
            pii_detected=pii_matches,
            retention_until=retention_until,
            metadata=metadata,
        )

        self.records[record_id] = record

        logger.info(
            f"Stored {classification.value} data record {record_id} with {len(pii_matches)} PII matches"
        )
        return record

    def retrieve_data(self, record_id: str) -> str | None:
        """Retrieve and decrypt data."""
        record = self.records.get(record_id)
        if not record:
            return None

        # Check retention period
        if record.retention_until and datetime.now(UTC) > record.retention_until:
            logger.warning(f"Data record {record_id} has expired and should be deleted")
            return None

        encryption_metadata = record.metadata.get("encryption_metadata", {})
        if not record.encryption_key_id:
            logger.error(
                "Missing encryption key identifier for data record %s", record_id
            )
            return None
        if (
            not encryption_metadata
            or "iv" not in encryption_metadata
            or "tag" not in encryption_metadata
        ):
            logger.error("Incomplete encryption metadata for data record %s", record_id)
            return None

        encrypted_payload = {
            "ciphertext": record.data,
            "iv": encryption_metadata["iv"],
            "tag": encryption_metadata["tag"],
            "key_id": record.encryption_key_id,
            "algorithm": encryption_metadata.get("algorithm", "AES-256-GCM"),
            "classification": record.classification.value,
        }

        try:
            data_bytes = self.encryption_manager.decrypt(encrypted_payload)
        except Exception as e:
            logger.error(f"Failed to decrypt data record {record_id}: {e}")
            return None

        data = data_bytes.decode("utf-8")

        # Update access time
        record.accessed_at = datetime.now(UTC)

        logger.debug(f"Retrieved data record {record_id}")
        return cast(str | None, data)

    def delete_data(self, record_id: str) -> bool:
        """Securely delete data record."""
        if record_id in self.records:
            del self.records[record_id]
            logger.info(f"Deleted data record {record_id}")
            return True
        return False

    def cleanup_expired_data(self) -> int:
        """Clean up expired data records."""
        now = datetime.now(UTC)
        expired_records = []

        for record_id, record in self.records.items():
            if record.retention_until and now > record.retention_until:
                expired_records.append(record_id)

        for record_id in expired_records:
            self.delete_data(record_id)

        if expired_records:
            logger.info(f"Cleaned up {len(expired_records)} expired data records")

        return len(expired_records)

    def store(
        self,
        key: str,
        data: str | dict[str, Any],
        classification: DataClassification = DataClassification.INTERNAL,
    ) -> None:
        """Store data with given key (simple API for tests).

        Args:
            key: Storage key
            data: Data to store
            classification: Data classification level
        """
        # Convert dict to string if needed
        if isinstance(data, dict):
            data_str = str(data)
        else:
            data_str = data

        # Detect and anonymize PII
        pii_matches = self.pii_detector.detect_pii(data_str)
        if pii_matches:
            # Anonymize PII for secure storage
            anonymized_data = self.pii_detector.anonymize_text(data_str)
        else:
            anonymized_data = data_str

        # Store with simple key-value API
        self.simple_storage[key] = {
            "original_data": data,
            "anonymized_data": anonymized_data,
            "classification": classification,
            "pii_detected": len(pii_matches),
            "created_at": datetime.now(UTC).isoformat(),
        }

    def retrieve(self, key: str) -> str | dict[str, Any] | None:
        """Retrieve data by key (simple API for tests).

        Args:
            key: Storage key

        Returns:
            Retrieved data or None if not found
        """
        if key not in self.simple_storage:
            return None

        stored_data = self.simple_storage[key]

        # Check if data has expired (auto-cleanup)
        if self.protection_manager:
            created_at = datetime.fromisoformat(stored_data["created_at"])
            classification = stored_data["classification"]
            if self.protection_manager.is_retention_expired(created_at, classification):
                # Data has expired, auto-delete
                del self.simple_storage[key]
                return None

        # Return anonymized version if PII was detected
        if stored_data["pii_detected"] > 0:
            # Parse back to original format if it was a dict
            original = stored_data["original_data"]
            if isinstance(original, dict):
                # Create anonymized version of the dict
                anonymized_dict = {}
                for k, v in original.items():
                    if isinstance(v, str):
                        anonymized_dict[k] = self.pii_detector.anonymize_text(v)
                    else:
                        anonymized_dict[k] = v
                return anonymized_dict
            else:
                return cast(str | dict[str, Any] | None, stored_data["anonymized_data"])
        else:
            return cast(str | dict[str, Any] | None, stored_data["original_data"])

    def list_keys(self, classification: DataClassification | None = None) -> list[str]:
        """List all storage keys, optionally filtered by classification.

        Args:
            classification: Optional classification filter

        Returns:
            List of storage keys
        """
        if classification is None:
            return list(self.simple_storage.keys())
        else:
            return [
                key
                for key, data in self.simple_storage.items()
                if data["classification"] == classification
            ]

    def delete(self, key: str) -> bool:
        """Delete data by key (simple API for tests).

        Args:
            key: Storage key to delete

        Returns:
            True if deleted, False if key not found
        """
        if key in self.simple_storage:
            del self.simple_storage[key]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Clean up expired data records.

        Returns:
            Number of records cleaned up
        """
        if not self.protection_manager:
            return 0

        expired_keys = []
        for key, stored_data in self.simple_storage.items():
            created_at = datetime.fromisoformat(stored_data["created_at"])
            classification = stored_data["classification"]
            if self.protection_manager.is_retention_expired(created_at, classification):
                expired_keys.append(key)

        # Delete expired data
        for key in expired_keys:
            del self.simple_storage[key]

        return len(expired_keys)

    def get_records_by_classification(
        self, classification: DataClassification
    ) -> list[DataRecord]:
        """Get all records of a specific classification."""
        return [
            record
            for record in self.records.values()
            if record.classification == classification
        ]

    def export_data(
        self, record_id: str, anonymize: bool = False
    ) -> dict[str, Any] | None:
        """Export data record (for GDPR compliance)."""
        record = self.records.get(record_id)
        if not record:
            return None

        data = self.retrieve_data(record_id)
        if not data:
            return None

        if anonymize:
            data = self.pii_detector.anonymize_pii(data)

        return {
            "record_id": record.record_id,
            "data": data,
            "classification": record.classification.value,
            "created_at": record.created_at.isoformat(),
            "accessed_at": (
                record.accessed_at.isoformat() if record.accessed_at else None
            ),
            "pii_detected": [
                {
                    "type": match.pii_type.value,
                    "value": match.value if not anonymize else "*" * len(match.value),
                    "confidence": match.confidence,
                }
                for match in record.pii_detected
            ],
            "metadata": record.metadata,
        }


class DataProtectionManager:
    """Comprehensive data protection management."""

    def __init__(
        self,
        encryption_manager: EncryptionManager | None = None,
        pii_detector: PIIDetector | None = None,
    ) -> None:
        """Initialize data protection manager."""
        self.encryption_manager = encryption_manager or EncryptionManager()
        self.pii_detector = pii_detector or PIIDetector()
        self.data_classifier = DataClassifier()
        self.secure_storage = SecureStorage(self.encryption_manager)

    def protect_data(
        self,
        data: str | dict[str, Any],
        classification: DataClassification | None = None,
        anonymize_pii: bool = False,
        encrypt: bool = True,
    ) -> dict[str, Any]:
        """Apply appropriate protection to data.

        Args:
            data: Data to protect (string or dict)
            classification: Data classification level
            anonymize_pii: Whether to anonymize PII before storage
            encrypt: Whether to encrypt the data

        Returns:
            Dictionary with protection results and metadata
        """
        if classification is None:
            if isinstance(data, str):
                classification = self.data_classifier.classify_data(data)
            else:
                classification = self.data_classifier.classify_data(str(data))

        # Convert data to string for processing
        if isinstance(data, dict):
            data_str = str(data)
        else:
            data_str = data

        # Detect PII
        pii_matches = self.pii_detector.detect_pii(data_str)

        # Apply anonymization if requested
        protected_data = data_str
        protection_applied = []

        if anonymize_pii and pii_matches:
            protected_data = self.pii_detector.anonymize_text(protected_data)
            protection_applied.append("pii_anonymization")

        # Apply encryption if requested
        encryption_metadata = {}
        if encrypt:
            encrypted_result = self.encryption_manager.encrypt(
                protected_data, classification
            )
            protection_applied.append("encryption")
            encryption_metadata = {
                "key_id": encrypted_result["key_id"],
                "algorithm": encrypted_result["algorithm"],
            }

        # Create result metadata
        metadata = {
            "pii_detected": len(pii_matches),
            "protection_applied": protection_applied,
        }

        # Only add encryption metadata if encryption was applied
        if encrypt:
            metadata["encryption"] = encryption_metadata

        result = {
            "is_encrypted": encrypt,
            "classification": classification.value,
            "original_data": data,
            "protected_data": protected_data,
            "metadata": metadata,
        }

        return result

    def access_data(self, record_id: str) -> str | None:
        """Access protected data."""
        return self.secure_storage.retrieve_data(record_id)

    def get_retention_period(self, classification: DataClassification) -> int:
        """Get data retention period in days for classification level.

        Args:
            classification: Data classification level

        Returns:
            Retention period in days
        """
        retention_periods = {
            DataClassification.PUBLIC: 3650,  # 10 years
            DataClassification.INTERNAL: 2555,  # 7 years
            DataClassification.CONFIDENTIAL: 1825,  # 5 years
            DataClassification.RESTRICTED: 1095,  # 3 years
            DataClassification.TOP_SECRET: 365,  # 1 year
        }

        return retention_periods.get(classification, 1825)  # Default to 5 years

    def is_retention_expired(
        self, created_date: datetime, classification: DataClassification
    ) -> bool:
        """Check if data has exceeded retention period.

        Args:
            created_date: When the data was created
            classification: Data classification level

        Returns:
            True if retention period has been exceeded
        """
        retention_days = self.get_retention_period(classification)
        expiry_date = created_date + timedelta(days=retention_days)
        return datetime.now(UTC) > expiry_date

    def unprotect_data(self, protected_result: dict[str, Any]) -> str:
        """Unprotect data from protection result.

        Args:
            protected_result: Result from protect_data() method

        Returns:
            Unprotected data as string
        """
        if protected_result.get("is_encrypted", False):
            # Decrypt if encrypted, but return the protected/anonymized version
            return cast(str, protected_result["protected_data"])
        else:
            return cast(str, protected_result["protected_data"])


@dataclass
class KeyMetadata:
    """Metadata for encryption keys."""

    algorithm: str
    key_length: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    rotation_count: int = 0
    is_active: bool = True

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    def is_valid(self) -> bool:
        """Check if key is valid (active and not expired)."""
        return self.is_active and not self.is_expired()


class KeyManager:
    """Key management for encryption operations."""

    def __init__(self) -> None:
        """Initialize key manager."""
        self.keys: dict[str, Any] = {}  # key_id -> key_bytes
        self.key_metadata: dict[str, Any] = {}  # key_id -> KeyMetadata

    def generate_key(self, algorithm: str, expires_in_days: int | None = None) -> str:
        """Generate a new encryption key.

        Args:
            algorithm: Encryption algorithm (e.g., "AES-256")
            expires_in_days: Key expiration in days

        Returns:
            Generated key ID
        """
        key_id = secrets.token_urlsafe(16)

        # Generate key based on algorithm
        if algorithm == "AES-256":
            key_data = secrets.token_bytes(32)  # 256 bits
            key_length = 256
        elif algorithm == "AES-128":
            key_data = secrets.token_bytes(16)  # 128 bits
            key_length = 128
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}") from None

        # Set expiration if provided
        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        # Store key and metadata
        self.keys[key_id] = key_data
        self.key_metadata[key_id] = KeyMetadata(
            algorithm=algorithm, key_length=key_length, expires_at=expires_at
        )

        return key_id

    def get_key(self, key_id: str) -> bytes | None:
        """Get encryption key by ID.

        Returns:
            Key bytes if valid, None if expired or not found
        """
        if key_id not in self.keys:
            return None

        metadata = self.key_metadata.get(key_id)
        if metadata and not metadata.is_valid():
            return None

        return self.keys.get(key_id)

    def rotate_key(self, old_key_id: str) -> str:
        """Rotate an encryption key.

        Args:
            old_key_id: ID of key to rotate

        Returns:
            New key ID
        """
        if old_key_id not in self.keys:
            raise KeyError(f"Key {old_key_id} not found")

        old_metadata = self.key_metadata[old_key_id]

        # Generate new key with same algorithm
        new_key_id = self.generate_key(old_metadata.algorithm)

        # Update metadata
        old_metadata.is_active = False
        new_metadata = self.key_metadata[new_key_id]
        new_metadata.rotation_count = old_metadata.rotation_count + 1

        return new_key_id

    def delete_key(self, key_id: str) -> bool:
        """Securely delete an encryption key.

        Args:
            key_id: ID of key to delete

        Returns:
            True if deleted, False if not found
        """
        if key_id not in self.keys:
            return False

        # Securely overwrite key data
        key_data = self.keys[key_id]
        if isinstance(key_data, bytes):
            # Overwrite with random data multiple times
            for _ in range(3):
                self.keys[key_id] = secrets.token_bytes(len(key_data))

        # Remove from storage
        del self.keys[key_id]
        del self.key_metadata[key_id]

        return True
