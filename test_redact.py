import re
from typing import Any

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|secret|token|api[_-]?key|authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)

_SENSITIVE_VALUE_PATTERNS = (
    re.compile(
        r"(?i)(?:password|secret|token|api[_-]?key|authorization|credential|private[_-]?key)"
        r"\s*[:=]\s*\S+"
    ),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/\-=]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
)

def _is_sensitive_key(key: str) -> bool:
    return bool(_SENSITIVE_KEY_PATTERN.search(key))

def _contains_sensitive_value(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SENSITIVE_VALUE_PATTERNS)

def _redact_observability_object(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _redact_observability_object(item, str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_observability_object(item, parent_key) for item in value]
    if parent_key and _is_sensitive_key(parent_key):
        return "[REDACTED]"
    if isinstance(value, str) and _contains_sensitive_value(value):
        return "[REDACTED]"
    return value

print(_redact_observability_object({"api_key": {"nested": "secret_value"}}))
