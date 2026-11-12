"""Data encryption and privacy controls for Traigent Enterprise."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, cast

try:  # POSIX advisory file locks; absent on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

try:  # Windows byte-range file locks
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore[assignment]

try:
    from cryptography.fernet import Fernet as _Fernet
    from cryptography.fernet import InvalidToken
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
    Fernet = _Fernet
except ImportError as exc:  # pragma: no cover - fail-closed import guard
    # SECURITY: Do not provide a silent mock fallback. Fake "encrypted_" prefix
    # encryption masquerading as real crypto is a serious security issue. Fail
    # closed and force callers to install the real cryptography library.
    raise ImportError(
        "Encryption requires `cryptography>=46.0`. "
        "Install with `pip install cryptography`. "
        "The fake fallback was removed in favor of fail-closed behavior."
    ) from exc


from ..utils.logging import get_logger
from ..utils.secure_path import (
    safe_read_bytes,
    safe_read_text,
    safe_write_bytes,
    safe_write_text,
    validate_path,
)
from .credentials import SecurityError

logger = get_logger(__name__)


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


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


class EncryptionKeyNotFoundError(SecurityError):
    """Raised when the key needed to decrypt is absent from the key store."""

    def __init__(self, key_id: str) -> None:
        self.key_id = key_id
        super().__init__(
            f"Encryption key '{key_id}' is not available in this key store "
            "(missing or deleted); data can only be decrypted with "
            "the key store that encrypted it."
        )


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

        # Reuse the active key; a key is created only when none is usable.
        key_id: str = self.key_manager.get_or_create_active_key("AES-256")
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
            # SECURITY: No mock-encryption fallback. The previous fallback was
            # gated by TRAIGENT_MOCK_LLM and emitted ``b"mock_" + plaintext``,
            # which is not encryption at all and silently exposed plaintext as
            # ciphertext if the env var was ever set in production. Always
            # require real crypto.
            raise RuntimeError(
                "Encryption requires the 'cryptography' package. "
                "Install it with: pip install 'traigent[security]'"
            )

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

        # Decrypt-only keys (rotated or expired) remain usable for decryption.
        key_data: bytes | None = self.key_manager.get_key_for_decryption(key_id)
        if key_data is None:
            raise EncryptionKeyNotFoundError(key_id)
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
            # SECURITY: No mock-decryption fallback. The previous fallback was
            # gated by TRAIGENT_MOCK_LLM and stripped ``b"mock_"`` /
            # ``b"encrypted_"`` prefixes (or simply returned the input) — i.e.
            # it never decrypted anything, but masquerading as decrypt() it
            # would silently surface attacker-supplied data on a system with
            # the env var set. Always require real crypto.
            raise RuntimeError(
                "Decryption requires the 'cryptography' package. "
                "Install it with: pip install 'traigent[security]'"
            )

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
            PIIType.NAME: r"\b(?:full_?name|customer_?name|name)['\"]?\s*[:=]\s*['\"]?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b",
            PIIType.ADDRESS: r"\b(?:street_?address|mailing_?address|address)['\"]?\s*[:=]\s*['\"]?\d{1,6}\s+[A-Za-z0-9 .'-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
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
            if match.confidence >= 0.7:
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

        # A card-shaped number is sensitive even when its checksum fails.
        eligible = [
            match
            for match in pii_matches
            if match.confidence > 0.5 or match.pii_type == PIIType.CREDIT_CARD
        ]

        for start, end, winner in reversed(self._merge_overlapping(eligible)):
            replacement = f"[{winner.pii_type.value.upper()}_REDACTED]"
            anonymized_text = (
                anonymized_text[:start] + replacement + anonymized_text[end:]
            )

        return anonymized_text

    @staticmethod
    def _merge_overlapping(
        matches: list[PIIMatch],
    ) -> list[tuple[int, int, PIIMatch]]:
        """Merge overlapping matches into (start, end, winning match) spans.

        The winner is the highest-confidence match (credit cards win ties),
        and the span extends to the union of all overlapping matches.
        """
        merged: list[list[Any]] = []
        for match in sorted(matches, key=lambda m: (m.start_pos, -m.end_pos)):
            if merged and match.start_pos < merged[-1][1]:
                span = merged[-1]
                span[1] = max(span[1], match.end_pos)
                span[2].append(match)
            else:
                merged.append([match.start_pos, match.end_pos, [match]])

        def rank(m: PIIMatch) -> tuple[float, bool, int]:
            return (
                m.confidence,
                m.pii_type == PIIType.CREDIT_CARD,
                m.end_pos - m.start_pos,
            )

        return [(s, e, max(group, key=rank)) for s, e, group in merged]

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


def _classification_score(
    rules: dict[str, Any],
    data: str,
    metadata: dict[str, Any],
    pii_matches: Any,
    pii_ratio: float,
) -> int:
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

    return score


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
            score = _classification_score(rules, data, metadata, pii_matches, pii_ratio)

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

    def _detect_pii_in_value(self, value: Any) -> list[PIIMatch]:
        """Recursively detect PII while preserving structured field context."""
        matches: list[PIIMatch] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str):
                    matches.extend(self.pii_detector.detect_pii(f"{key}: {item}"))
                else:
                    matches.extend(self._detect_pii_in_value(item))
            return matches
        if isinstance(value, list):
            for item in value:
                matches.extend(self._detect_pii_in_value(item))
            return matches
        if isinstance(value, str):
            return list(self.pii_detector.detect_pii(value))
        return []

    def _anonymize_value(self, value: Any, field_name: str | None = None) -> Any:
        """Recursively anonymize PII without flattening dict/list payloads."""
        if isinstance(value, dict):
            return {
                key: self._anonymize_value(item, str(key))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._anonymize_value(item, field_name) for item in value]
        if isinstance(value, str):
            scan_text = f"{field_name}: {value}" if field_name else value
            matches = self.pii_detector.detect_pii(scan_text)
            if not matches:
                return value
            if field_name:
                for pii_type in {match.pii_type for match in matches}:
                    value = self.pii_detector.anonymize_text(
                        f"{field_name}: {value}", pii_types=[pii_type]
                    ).split(": ", 1)[-1]
                return value
            return self.pii_detector.anonymize_text(value)
        return value

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
            raise SecurityError(f"Decryption failed for {record_id}: {e}") from e

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
        pii_matches = self._detect_pii_in_value(data)
        anonymized_data = self._anonymize_value(data) if pii_matches else data

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
            Dictionary with protection results and metadata. When encrypted,
            ``protected_data`` is a JSON envelope (hex ciphertext, iv, tag and
            ``key_id``); the key itself is held only by this manager's key
            store. The default ``KeyManager`` is in-memory, so the envelope
            can be decrypted only while that key store is alive, or with a
            manager sharing the same (persisted) key store.
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
            # Return the ciphertext envelope, never the plaintext, when the
            # result is labelled as encrypted.
            protected_data = json.dumps(
                {
                    "ciphertext": encrypted_result["ciphertext"].hex(),
                    "iv": encrypted_result["iv"].hex(),
                    "tag": encrypted_result["tag"].hex(),
                    "key_id": encrypted_result["key_id"],
                    "algorithm": encrypted_result["algorithm"],
                    "classification": encrypted_result["classification"],
                }
            )

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
            "protected_data": protected_data,
            "metadata": metadata,
        }
        if not encrypt:
            result["original_data"] = data

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

    def unprotect_data(
        self,
        protected_result: dict[str, Any],
        *,
        require_encrypted: bool = False,
    ) -> str:
        """Unprotect data from protection result.

        Behaviour is decided by the ``is_encrypted`` flag only. If it is true,
        ``protected_data`` must be a complete, valid encryption envelope, which
        is decrypted and authenticated. If it is false or absent,
        ``protected_data`` is returned unchanged.

        The returned record is not self-authenticating: the flag travels with
        the data and can be altered by whoever holds it. Callers that stored
        encrypted data should pass ``require_encrypted=True``.

        Decryption needs the same key store that encrypted the data.

        Args:
            protected_result: Result from protect_data() method
            require_encrypted: Reject records whose ``is_encrypted`` is not true

        Returns:
            Unprotected data as string

        Raises:
            SecurityError: ``require_encrypted`` is set and the record is not
                marked encrypted.
            ValueError: the record is marked encrypted but the envelope is
                missing or malformed.
            EncryptionKeyNotFoundError: the key id is not in the key store.
            RuntimeError: authentication failed (tampered data).
        """
        encrypted = protected_result.get("is_encrypted") is True
        if require_encrypted and not encrypted:
            raise SecurityError("Expected encrypted data but record is not encrypted")
        protected = protected_result["protected_data"]
        if not encrypted:
            return cast(str, protected)

        envelope = self._parse_envelope(protected)
        if envelope is None:
            raise ValueError(
                "Result is marked encrypted but protected_data is not a "
                "valid encryption envelope"
            )
        try:
            parts = {
                "ciphertext": bytes.fromhex(envelope["ciphertext"]),
                "iv": bytes.fromhex(envelope["iv"]),
                "tag": bytes.fromhex(envelope["tag"]),
                "key_id": envelope["key_id"],
                "algorithm": envelope.get("algorithm", "AES-256-GCM"),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed encryption envelope") from exc
        return self.encryption_manager.decrypt(parts).decode("utf-8")

    @staticmethod
    def _parse_envelope(protected: Any) -> dict[str, Any] | None:
        """Return the envelope dict if complete, else None."""
        if not isinstance(protected, str):
            return None
        try:
            parsed = json.loads(protected)
        except ValueError:
            return None
        required = {"ciphertext", "iv", "tag", "key_id"}
        if isinstance(parsed, dict) and required <= parsed.keys():
            return parsed
        return None


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

    KEY_STORE_VERSION = 2
    KEY_STORE_KDF_ITERATIONS = 390_000

    def __init__(
        self,
        storage_path: str | Path | None = None,
        master_password: str | None = None,
    ) -> None:
        """Initialize key manager."""
        self._lock = threading.RLock()
        self._flock_depth = 0
        self._flock_handle: Any = None
        self._fernet_cache: dict[bytes, Any] = {}
        self._write_salt: bytes | None = None
        self._last_stat: tuple[int, int, int] | None = None
        self.keys: dict[str, Any] = {}  # key_id -> key_bytes
        self.key_metadata: dict[str, Any] = {}  # key_id -> KeyMetadata
        configured_path = storage_path or os.getenv("TRAIGENT_KEY_MANAGER_STORE")
        self.storage_path = (
            Path(configured_path).expanduser() if configured_path else None
        )
        self._store_master_password = self._resolve_store_master_password(
            master_password
        )
        if self.storage_path and self._lock_backend() is None:
            raise RuntimeError(
                "A persistent KeyManager store needs a cross-process file lock "
                "(fcntl or msvcrt), but neither is available on this platform."
            )
        if self.storage_path:
            if self._store_master_password is None:
                raise ValueError(
                    "Persistent KeyManager storage requires a wrapping secret. "
                    "Set TRAIGENT_KEY_MANAGER_MASTER_PASSWORD, set "
                    "TRAIGENT_MASTER_PASSWORD, or pass master_password explicitly."
                )
            self._load_keys()

    def _resolve_store_master_password(self, master_password: str | None) -> str | None:
        return (
            master_password
            or os.getenv("TRAIGENT_KEY_MANAGER_MASTER_PASSWORD")
            or os.getenv("TRAIGENT_MASTER_PASSWORD")
        )

    def _store_fernet(self, salt: bytes) -> Any:
        if self._store_master_password is None:
            raise ValueError("Key store wrapping secret is not configured")
        cached = self._fernet_cache.get(salt)
        if cached is not None:
            return cached
        key_material = hashlib.pbkdf2_hmac(
            "sha256",
            self._store_master_password.encode("utf-8"),
            salt,
            self.KEY_STORE_KDF_ITERATIONS,
            dklen=32,
        )
        fernet = Fernet(base64.urlsafe_b64encode(key_material))
        self._fernet_cache[salt] = fernet
        return fernet

    @staticmethod
    def _lock_backend() -> str | None:
        """Name of the cross-process lock primitive available on this platform."""
        if fcntl is not None:
            return "fcntl"
        if msvcrt is not None:
            return "msvcrt"
        return None

    def _os_lock(self, fd: int, backend: str) -> None:
        if backend == "fcntl":
            fcntl.flock(fd, fcntl.LOCK_EX)
            return
        os.lseek(fd, 0, os.SEEK_SET)
        while True:  # LK_LOCK gives up after ~10 s; keep waiting
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
                return
            except OSError:
                continue

    def _os_unlock(self, fd: int, backend: str) -> None:
        if backend == "fcntl":
            fcntl.flock(fd, fcntl.LOCK_UN)
            return
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]

    def _stat_signature(self) -> tuple[int, int, int] | None:
        if self.storage_path is None:
            return None
        try:
            st = self.storage_path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    @contextmanager
    def _store_lock(self) -> Iterator[None]:
        """Hold the in-process lock and, if persistent, an OS lock on the store."""
        with self._lock:
            backend = self._lock_backend() if self.storage_path is not None else None
            if self.storage_path is not None and backend is None:
                raise RuntimeError(
                    "A persistent KeyManager store needs a cross-process file "
                    "lock (fcntl or msvcrt), but neither is available on this "
                    "platform; refusing to share the store without one."
                )
            use_flock = backend is not None
            if use_flock and self._flock_depth == 0:
                assert self.storage_path is not None and backend is not None
                self.storage_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                lock_path = self.storage_path.with_name(
                    f".{self.storage_path.name}.lock"
                )
                fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    self._os_lock(fd, backend)
                except BaseException:
                    os.close(fd)
                    raise
                self._flock_handle = (fd, backend)
            if use_flock:
                self._flock_depth += 1
            try:
                yield
            finally:
                if use_flock:
                    self._flock_depth -= 1
                    if self._flock_depth == 0:
                        fd, used_backend = self._flock_handle
                        self._flock_handle = None
                        try:
                            self._os_unlock(fd, used_backend)
                        finally:
                            os.close(fd)

    def _refresh_from_store(self) -> None:
        """Re-read the store (caller holds the store lock) so no write is stale."""
        if self.storage_path is None:
            return
        signature = self._stat_signature()
        if signature is None or signature == self._last_stat:
            return  # nothing changed on disk since the last load or write
        self._load_keys()

    def _load_keys(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        signature = self._stat_signature()
        stored_payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        if (
            stored_payload.get("version") == self.KEY_STORE_VERSION
            and "payload" in stored_payload
        ):
            salt = base64.b64decode(stored_payload["salt"])
            self._write_salt = salt
            try:
                decrypted = self._store_fernet(salt).decrypt(
                    stored_payload["payload"].encode("ascii")
                )
            except InvalidToken as exc:
                raise ValueError(
                    "Unable to decrypt KeyManager store with the configured "
                    "wrapping secret"
                ) from exc
            payload = json.loads(decrypted.decode("utf-8"))
        elif "keys" in stored_payload and _truthy(
            os.getenv("TRAIGENT_ALLOW_PLAINTEXT_KEY_STORE")
        ):
            logger.warning(
                "Loading legacy plaintext KeyManager store at %s for migration. "
                "The next write will persist an encrypted key store.",
                self.storage_path,
            )
            payload = stored_payload
        elif "keys" in stored_payload:
            raise ValueError(
                "Refusing to load plaintext KeyManager key material. Set "
                "TRAIGENT_ALLOW_PLAINTEXT_KEY_STORE=true only long enough to "
                "migrate this store with a configured wrapping secret."
            )
        else:
            raise ValueError("Unsupported KeyManager store format")
        loaded_keys: dict[str, Any] = {}
        loaded_meta: dict[str, Any] = {}
        for key_id, key_entry in payload.get("keys", {}).items():
            loaded_keys[key_id] = base64.b64decode(key_entry["key"])
            metadata = key_entry["metadata"]
            loaded_meta[key_id] = KeyMetadata(
                algorithm=metadata["algorithm"],
                key_length=int(metadata["key_length"]),
                created_at=datetime.fromisoformat(metadata["created_at"]),
                expires_at=(
                    datetime.fromisoformat(metadata["expires_at"])
                    if metadata.get("expires_at")
                    else None
                ),
                rotation_count=int(metadata.get("rotation_count", 0)),
                is_active=bool(metadata.get("is_active", True)),
            )
        # The store is authoritative for which keys exist. A key that was ever
        # deactivated stays deactivated; keys only held in memory are kept.
        for key_id, meta in loaded_meta.items():
            local = self.key_metadata.get(key_id)
            if local is not None and not local.is_active:
                meta.is_active = False
        for key_id in list(self.keys):
            if key_id not in loaded_keys:
                del self.keys[key_id]
                self.key_metadata.pop(key_id, None)
        self.keys.update(loaded_keys)
        self.key_metadata.update(loaded_meta)
        self._last_stat = signature

    def _persist_keys(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload: dict[str, Any] = {"keys": {}}
        for key_id, key_data in self.keys.items():
            metadata = self.key_metadata[key_id]
            payload["keys"][key_id] = {
                "key": base64.b64encode(key_data).decode("ascii"),
                "metadata": {
                    "algorithm": metadata.algorithm,
                    "key_length": metadata.key_length,
                    "created_at": metadata.created_at.isoformat(),
                    "expires_at": (
                        metadata.expires_at.isoformat() if metadata.expires_at else None
                    ),
                    "rotation_count": metadata.rotation_count,
                    "is_active": metadata.is_active,
                },
            }
        if self._write_salt is None:
            self._write_salt = secrets.token_bytes(16)
        salt = self._write_salt
        encrypted_payload = self._store_fernet(salt).encrypt(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        )
        stored_payload = {
            "version": self.KEY_STORE_VERSION,
            "kdf": "pbkdf2-sha256",
            "iterations": self.KEY_STORE_KDF_ITERATIONS,
            "salt": base64.b64encode(salt).decode("ascii"),
            "payload": encrypted_payload.decode("ascii"),
        }
        # Atomic replace: write a private temp file, fsync, then rename so a
        # crash cannot truncate the existing key store.
        tmp_path = self.storage_path.with_name(
            f".{self.storage_path.name}.{secrets.token_hex(4)}.tmp"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(tmp_path, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(stored_payload, sort_keys=True))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.storage_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        self._last_stat = self._stat_signature()

    def generate_key(
        self,
        algorithm: str,
        expires_in_days: int | None = None,
        *,
        persist: bool = True,
    ) -> str:
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
        with self._store_lock():
            self._refresh_from_store()
            self.keys[key_id] = key_data
            self.key_metadata[key_id] = KeyMetadata(
                algorithm=algorithm, key_length=key_length, expires_at=expires_at
            )
            if persist:
                self._persist_keys()

        return key_id

    def get_active_key_id(self, algorithm: str = "AES-256") -> str | None:
        """Return the newest active, unexpired key id for ``algorithm`` (or None)."""
        with self._lock:
            candidates = [
                (meta.created_at, key_id)
                for key_id, meta in self.key_metadata.items()
                if key_id in self.keys
                and meta.algorithm == algorithm
                and meta.is_valid()
            ]
        if not candidates:
            return None
        return max(candidates)[1]

    def get_or_create_active_key(self, algorithm: str = "AES-256") -> str:
        """Return the active key id, generating a key only if none exists."""
        with self._store_lock():
            self._refresh_from_store()
            key_id = self.get_active_key_id(algorithm)
            if key_id is None:
                key_id = self.generate_key(algorithm)
            return key_id

    def get_key_for_decryption(self, key_id: str) -> bytes | None:
        """Get key bytes for decryption only.

        Rotated (inactive) and expired keys are decrypt-only: they are never
        returned by :meth:`get_key` for new encryption but stay available
        here until explicitly deleted.
        """
        if self.storage_path is None:
            with self._lock:
                return cast("bytes | None", self.keys.get(key_id))
        # Another manager sharing this store may have created or deleted the
        # key: a cheap stat check under the lock reloads only if it changed,
        # and a key deleted elsewhere is never served from the cache.
        with self._store_lock():
            self._refresh_from_store()
            return cast("bytes | None", self.keys.get(key_id))

    def get_key(self, key_id: str) -> bytes | None:
        """Get an encryption key by ID for new encryption.

        Returns:
            Key bytes if active and unexpired, None if decrypt-only or not found
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
        with self._store_lock():
            self._refresh_from_store()
            if old_key_id not in self.keys:
                raise KeyError(f"Key {old_key_id} not found")

            algorithm = self.key_metadata[old_key_id].algorithm

            # Create the new key and retire the old one in a single write.
            new_key_id = self.generate_key(algorithm, persist=False)
            old_metadata = self.key_metadata[old_key_id]
            old_metadata.is_active = False
            new_metadata = self.key_metadata[new_key_id]
            new_metadata.rotation_count = old_metadata.rotation_count + 1
            self._persist_keys()

        return new_key_id

    def delete_key(self, key_id: str) -> bool:
        """Securely delete an encryption key.

        Args:
            key_id: ID of key to delete

        Returns:
            True if deleted, False if not found
        """
        with self._store_lock():
            self._refresh_from_store()
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
            self._persist_keys()

        return True
