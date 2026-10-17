"""Workload identity: agent key, registration provenance and environment.

One definition of "which agent" and "which environment" for traces, shared by
the SDKs and the backend through the TraigentSchema identity vectors
(``identity_vectors_v1.json``).

* Canonical agent key: the backend's normalize + derive (``AgentService.
  _normalize_function_identity`` then ``derive_agent_key``), re-implemented here
  only to PRE-compute the same key. The vectors prove the two agree.
* Provenance: whether the registration identity may label a trace.
  ``function_derived`` is the only eligible class; ``user_named`` is a customer
  chosen name (content, never written to traces); ``ineligible`` is
  an anonymous registration (a ``__name__`` containing ``<``, or no ``__name__``).
* Environment: explicit > ``TRAIGENT_ENVIRONMENT`` > OTel
  ``deployment.environment.name`` > ``ENVIRONMENT`` > legacy ``TRAIGENT_ENV`` >
  ``default``. An explicit argument is a free label (at most 64 characters);
  every env-var rung is accepted only when it is a known label (shell state may
  carry arbitrary text, so unknown values are never sent). Production is never
  inferred.

Warnings emitted from here name an environment variable, never its value, and
never an agent name.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal
from urllib.parse import unquote

from traigent.utils.env_config import _KNOWN_ENVIRONMENT_NAMES
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

Provenance = Literal["function_derived", "user_named", "ineligible"]

AGENT_KEY_MAX_LENGTH = 128
_AGENT_KEY_DIGEST_HEX = 10
_AGENT_KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")
_AGENT_KEY_INVALID_RUN = re.compile(r"[^A-Za-z0-9_-]+")
_FALLBACK_IDENTITY = "traigent_function"
_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_function_identity(value: object) -> str:
    """Backend ``_normalize_function_identity``: strip controls, collapse spaces."""
    raw = str(value or "")
    without_controls = "".join(
        char for char in raw if char.isprintable() or char.isspace()
    )
    normalized = _WHITESPACE_RUN.sub(" ", without_controls.strip())
    return normalized or _FALLBACK_IDENTITY


def derive_agent_key(name: object) -> str:
    """Backend ``derive_agent_key``: a key-shaped name is its own key."""
    text = name.strip() if isinstance(name, str) else ""
    if _AGENT_KEY_PATTERN.fullmatch(text):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:_AGENT_KEY_DIGEST_HEX]
    slug = _AGENT_KEY_INVALID_RUN.sub("-", text).strip("-")
    slug = slug[: AGENT_KEY_MAX_LENGTH - _AGENT_KEY_DIGEST_HEX - 1] or "agent"
    return f"{slug}-{digest}"


def canonical_agent_key(registration_identity: object) -> str:
    """The key the backend stores for a registration identity."""
    return derive_agent_key(normalize_function_identity(registration_identity))


@dataclass(frozen=True)
class AgentBinding:
    """What an agent scope carries: the registration identity and its class."""

    registration_identity: str
    provenance: Provenance
    key: str | None
    local_warning: str | None = None


def classify_python_registration(
    *,
    registration_identity: str,
    function_name: str | None,
    declared_name: str | None,
) -> AgentBinding:
    """Classify a Python ``@optimize`` registration.

    ``declared_name`` is the explicit ``agent_name`` / ``experiment_name`` or the
    ``TRAIGENT_EXPERIMENT_NAME`` value (``OptimizedFunction.declared_agent_name``):
    any of them makes the scope ``user_named`` and wins over an anonymous
    ``__name__``. Otherwise only a ``__name__`` containing ``<`` (``<lambda>``),
    or a callable with no ``__name__``, is ineligible; a ``<locals>`` qualname
    alone is eligible.
    """
    if declared_name:
        return AgentBinding(registration_identity, "user_named", None)
    if not function_name or "<" in function_name:
        return AgentBinding(
            registration_identity, "ineligible", None, "anonymous_registration"
        )
    return AgentBinding(
        registration_identity,
        "function_derived",
        canonical_agent_key(registration_identity),
    )


# --- environment -----------------------------------------------------------

ENVIRONMENT_MAX_LENGTH = 64
_OTEL_ENVIRONMENT_ATTRIBUTE = "deployment.environment.name"

EnvironmentSource = Literal[
    "explicit",
    "traigent_environment",
    "otel_resource",
    "environment",
    "traigent_env_legacy",
    "default",
]
EnvironmentWarning = Literal[
    "legacy_env_var_deprecated", "undeclared_default", "invalid_env_value_ignored"
]


class EnvironmentTooLongError(ValueError):
    """An explicit environment longer than 64 characters (programmer error)."""


@dataclass(frozen=True)
class EnvironmentResolution:
    environment: str
    source: EnvironmentSource
    warning: EnvironmentWarning | None


_warned_lock = threading.Lock()
_warned: set[str] = set()


def _warn_once(code: str, emit: Callable[[], None]) -> None:
    with _warned_lock:
        if code in _warned:
            return
        _warned.add(code)
    emit()


def _warn_invalid_env_value(var_name: str) -> None:
    def emit() -> None:
        logger.warning(
            "Ignoring unknown environment label in %s (not a known deployment "
            "label); falling back to the next environment source.",
            var_name,
        )

    _warn_once(f"invalid:{var_name}", emit)


def reset_environment_warnings() -> None:
    """Forget which one-time warnings fired (tests only)."""
    with _warned_lock:
        _warned.clear()


def _otel_environment(raw: str | None) -> str | None:
    if not raw:
        return None
    for pair in raw.split(","):
        key, sep, value = pair.partition("=")
        if sep and key.strip() == _OTEL_ENVIRONMENT_ATTRIBUTE:
            return unquote(value.strip())
    return None


def normalize_environment(value: str) -> str:
    """Trim and lower-case an environment label."""
    return value.strip().lower()


def resolve_environment(
    explicit: str | None = None,
    env: Mapping[str, str] | None = None,
) -> EnvironmentResolution:
    """Resolve the deployment environment label.

    Raises :class:`EnvironmentTooLongError` for an explicit value over 64
    characters after trimming; an empty explicit value counts as unset. An
    env-var value that is not a known label (``_KNOWN_ENVIRONMENT_NAMES`` in
    ``traigent.utils.env_config``; this covers empty, over-long and free-text
    values) is ignored with a value-free warning and resolution falls through.
    An explicit argument is a free label. Production is never inferred from
    ``NODE_ENV``, ``FLASK_ENV`` or similar.
    """
    environ = os.environ if env is None else env
    if explicit is not None:
        label = normalize_environment(explicit)
        if len(label) > ENVIRONMENT_MAX_LENGTH:
            raise EnvironmentTooLongError(
                f"environment must be at most {ENVIRONMENT_MAX_LENGTH} characters"
            )
        if label:
            return EnvironmentResolution(label, "explicit", None)

    ignored = False
    rungs: tuple[tuple[str, EnvironmentSource, Callable[[], str | None]], ...] = (
        (
            "TRAIGENT_ENVIRONMENT",
            "traigent_environment",
            lambda: environ.get("TRAIGENT_ENVIRONMENT"),
        ),
        (
            "OTEL_RESOURCE_ATTRIBUTES",
            "otel_resource",
            lambda: _otel_environment(environ.get("OTEL_RESOURCE_ATTRIBUTES")),
        ),
        ("ENVIRONMENT", "environment", lambda: environ.get("ENVIRONMENT")),
        ("TRAIGENT_ENV", "traigent_env_legacy", lambda: environ.get("TRAIGENT_ENV")),
    )
    for var_name, source, read in rungs:
        raw = read()
        if raw is None:
            continue
        label = normalize_environment(raw)
        if label not in _KNOWN_ENVIRONMENT_NAMES:
            ignored = True
            _warn_invalid_env_value(var_name)
            continue
        warning: EnvironmentWarning | None = (
            "invalid_env_value_ignored" if ignored else None
        )
        if source == "traigent_env_legacy":
            _warn_once(
                "legacy:TRAIGENT_ENV",
                lambda: logger.warning(
                    "TRAIGENT_ENV is deprecated; set TRAIGENT_ENVIRONMENT instead."
                ),
            )
            warning = warning or "legacy_env_var_deprecated"
        return EnvironmentResolution(label, source, warning)

    _warn_once(
        "undeclared",
        lambda: logger.warning(
            "No deployment environment declared; using 'default'. Set "
            "TRAIGENT_ENVIRONMENT (or OTEL_RESOURCE_ATTRIBUTES "
            "deployment.environment.name) to label traces."
        ),
    )
    return EnvironmentResolution(
        "default",
        "default",
        "invalid_env_value_ignored" if ignored else "undeclared_default",
    )
