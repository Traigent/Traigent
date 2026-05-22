"""Security primitives for the Traigent UI plugin.

This module hardens the plugin's user/LLM-facing surfaces against the
high-severity findings tracked in issue ``Traigent/Traigent#846``:

- Prompt injection: ``sanitize_inline_text`` strips control characters and
  bounds length; ``wrap_untrusted`` isolates untrusted content inside
  delimiter sentinels so a malicious payload cannot close the surrounding
  prompt structure or smuggle directives.
- XSS in Streamlit ``unsafe_allow_html`` templates: ``escape_html`` and
  ``escape_html_attr`` produce HTML-safe strings; ``escape_html_dict`` is a
  convenience for templates that interpolate many fields.
- Path traversal in plugin file I/O: ``validate_problem_name`` enforces an
  allow-list pattern, ``safe_problem_module_path`` resolves the resulting
  filename strictly under the langchain_problems package directory.
- LLM-into-source injection: ``safe_python_value_literal`` rejects anything
  that is not a JSON-style primitive before it is embedded in generated
  Python source.
- Claude Code SDK sandbox: ``safe_claude_code_options`` builds an options
  object that never enables ``bypassPermissions`` unless the operator sets
  an explicit, narrow, local-only opt-in environment variable.

These helpers are intentionally dependency-free apart from the standard
library so they can run in the lightweight environments the plugin uses.
"""

# Traceability: Traigent/Traigent#846 high-severity plugin-ui findings

from __future__ import annotations

import html
import math
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

__all__ = [
    "BYPASS_OPT_IN_ENV",
    "DEFAULT_INLINE_CHAR_BUDGET",
    "DEFAULT_UNTRUSTED_CHAR_BUDGET",
    "PROBLEM_NAME_PATTERN",
    "UnsafeProblemNameError",
    "UnsafeValueError",
    "escape_html",
    "escape_html_attr",
    "escape_html_dict",
    "format_currency",
    "format_duration_minutes",
    "format_int",
    "format_percent",
    "safe_claude_code_options",
    "safe_problem_module_path",
    "safe_python_value_literal",
    "sanitize_inline_text",
    "validate_problem_name",
    "wrap_untrusted",
]


# ---------------------------------------------------------------------------
# Prompt-injection primitives
# ---------------------------------------------------------------------------


DEFAULT_INLINE_CHAR_BUDGET: Final[int] = 2000
DEFAULT_UNTRUSTED_CHAR_BUDGET: Final[int] = 8000

# Match all ASCII control characters EXCEPT \t (0x09) and \n (0x0A); also drop
# the C1 range used by some prompt-injection payloads.
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# Stricter pattern used for short inline labels: also drops \t (0x09) and
# \n (0x0A) so a payload cannot start a new instruction line inside a
# structured prompt header.
_LABEL_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_LABEL_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_inline_text(
    text: Any,
    *,
    max_chars: int = DEFAULT_INLINE_CHAR_BUDGET,
    collapse_newlines: bool = True,
) -> str:
    """Return ``text`` stripped of control characters and bounded in length.

    Use this for short, untrusted strings that need to appear inline in a
    prompt without their own delimiter block (e.g. a single-word domain or
    a one-line description). By default newlines and tabs are stripped so
    a short label cannot start a new instruction line; set
    ``collapse_newlines=False`` to preserve them (for legitimately
    multi-line short text like a sanitized first line of a docstring).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if text is None:
        return ""
    raw = text if isinstance(text, str) else str(text)
    pattern = _LABEL_CONTROL_PATTERN if collapse_newlines else _CONTROL_CHAR_PATTERN
    cleaned = pattern.sub("", raw)
    cleaned = cleaned.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "...[truncated]"
    return cleaned


def wrap_untrusted(
    label: str,
    content: Any,
    *,
    max_chars: int = DEFAULT_UNTRUSTED_CHAR_BUDGET,
) -> str:
    """Wrap ``content`` as an untrusted-data block usable inside an LLM prompt.

    The returned string starts and ends with explicit sentinel markers so
    downstream prompt templates can instruct the model not to treat the
    enclosed text as instructions. Embedded copies of the sentinel are
    rewritten so the inner payload can never close the wrapper early.
    Content longer than ``max_chars`` is truncated with a visible marker.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    safe_label = _LABEL_SAFE_PATTERN.sub("_", str(label or "data"))[:64] or "data"

    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        text = str(content)

    cleaned = _CONTROL_CHAR_PATTERN.sub("", text)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "...[truncated]"

    sentinel = f"untrusted_{safe_label}"
    cleaned = cleaned.replace(f"<{sentinel}>", f"<{sentinel}_literal>").replace(
        f"</{sentinel}>", f"</{sentinel}_literal>"
    )
    return f"<{sentinel}>\n{cleaned}\n</{sentinel}>"


# ---------------------------------------------------------------------------
# HTML escaping for Streamlit ``unsafe_allow_html=True`` templates
# ---------------------------------------------------------------------------


def escape_html(value: Any) -> str:
    """Return a string that is safe to interpolate into HTML body context."""
    if value is None:
        return ""
    return html.escape(str(value), quote=False)


def escape_html_attr(value: Any) -> str:
    """Return a string that is safe to interpolate into an HTML attribute.

    Quotes are escaped in addition to ``< > &``. Callers must still wrap the
    interpolated value in matching ``"`` quotes inside the template.
    """
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def escape_html_dict(
    data: Mapping[str, Any],
    *,
    keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return a dict of ``{key: html_escaped_value}`` for templated rendering.

    When ``keys`` is provided only those keys are returned; missing keys map
    to the empty string. When ``keys`` is omitted all keys in ``data`` are
    escaped.
    """
    if keys is None:
        return {key: escape_html(value) for key, value in data.items()}
    return {key: escape_html(data.get(key, "")) for key in keys}


def format_percent(value: Any, *, fallback: str = "N/A") -> str:
    """Format ``value`` as a percentage string (e.g. ``96.4%``) or ``fallback``.

    The value is parsed as a float; non-numeric input falls back to the
    placeholder. The result never contains HTML-active characters so it is
    safe to interpolate into ``unsafe_allow_html`` templates.
    """
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return fallback


def format_currency(value: Any, *, fallback: str = "N/A", precision: int = 4) -> str:
    """Format ``value`` as a currency string (e.g. ``$0.0123``) or ``fallback``."""
    try:
        return f"${float(value):.{precision}f}"
    except (TypeError, ValueError):
        return fallback


def format_duration_minutes(value: Any, *, fallback: str = "N/A") -> str:
    """Format minutes as e.g. ``2.5m``; falls back to ``fallback`` on bad input."""
    try:
        return f"{float(value):.1f}m"
    except (TypeError, ValueError):
        return fallback


def format_int(value: Any, *, fallback: str = "N/A") -> str:
    """Format ``value`` as a base-10 integer string or ``fallback``."""
    try:
        return f"{int(value)}"
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Problem-name and module-path validation
# ---------------------------------------------------------------------------


# Problem names ultimately become Python module filenames. We require the
# typical lowercase identifier shape: ASCII letters, digits, underscores, and
# hyphens; no leading digit, no separators, no relative-path segments. Hard
# cap the length to defend against pathologically long inputs.
PROBLEM_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"
)


class UnsafeProblemNameError(ValueError):
    """Raised when a problem name fails the allow-list shape check."""


class UnsafeValueError(ValueError):
    """Raised when an LLM-supplied value cannot be safely embedded in source."""


def validate_problem_name(name: Any) -> str:
    """Validate ``name`` is a safe problem identifier.

    Returns the validated name unchanged. Raises ``UnsafeProblemNameError``
    when the input is not a string, contains path separators, NUL bytes, or
    otherwise fails the ``PROBLEM_NAME_PATTERN`` allow-list.
    """
    if not isinstance(name, str):
        raise UnsafeProblemNameError(
            f"Problem name must be a string, got {type(name).__name__}"
        )
    if "\x00" in name:
        raise UnsafeProblemNameError("Problem name contains NUL byte")
    if not PROBLEM_NAME_PATTERN.fullmatch(name):
        raise UnsafeProblemNameError(
            "Problem name must match [A-Za-z][A-Za-z0-9_-]{0,63} "
            "and contain no path separators"
        )
    return name


def safe_problem_module_path(name: str, base_dir: str | os.PathLike[str]) -> Path:
    """Return the validated absolute path to ``{base_dir}/{name}.py``.

    ``base_dir`` is resolved up front; the resulting path is then resolved
    and checked to live strictly inside ``base_dir``. ``name`` is also
    validated with :func:`validate_problem_name`. The file is *not* required
    to exist; callers handle the missing-file case explicitly.
    """
    validated_name = validate_problem_name(name)
    root = Path(os.fspath(base_dir)).resolve(strict=False)
    candidate = (root / f"{validated_name}.py").resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise UnsafeProblemNameError(
            f"Resolved path {candidate} escapes base directory {root}"
        ) from exc
    return candidate


# ---------------------------------------------------------------------------
# LLM-generated value validation for code emission
# ---------------------------------------------------------------------------


_ALLOWED_PRIMITIVE_TYPES: Final[tuple[type, ...]] = (str, int, float, bool, type(None))


def safe_python_value_literal(value: Any, *, max_str_chars: int = 4000) -> str:
    """Return a safe ``repr()`` of ``value`` for embedding in generated source.

    Only JSON-style primitives plus ``list``/``dict`` of primitives are
    accepted. ``str`` values are length-bounded and stripped of control
    characters so an LLM cannot smuggle a long-line payload, embedded NUL,
    or escape sequences that break the surrounding code structure.

    Raises ``UnsafeValueError`` for any other type, including arbitrary
    classes with custom ``__repr__`` methods (the original vulnerability).
    """
    return repr(_normalize_value(value, max_str_chars=max_str_chars))


def _normalize_value(value: Any, *, max_str_chars: int) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsafeValueError("Float values must be finite")
        return value
    if isinstance(value, str):
        cleaned = _CONTROL_CHAR_PATTERN.sub("", value)
        if len(cleaned) > max_str_chars:
            cleaned = cleaned[:max_str_chars]
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item, max_str_chars=max_str_chars) for item in value]
    if isinstance(value, dict):
        return {
            _coerce_dict_key(key): _normalize_value(item, max_str_chars=max_str_chars)
            for key, item in value.items()
        }
    raise UnsafeValueError(
        f"Value of type {type(value).__name__} is not allowed in generated source"
    )


def _coerce_dict_key(key: Any) -> str:
    if not isinstance(key, str):
        raise UnsafeValueError(f"Dict key must be str, got {type(key).__name__}")
    cleaned = _CONTROL_CHAR_PATTERN.sub("", key)
    if len(cleaned) > 200:
        raise UnsafeValueError("Dict key exceeds 200 characters")
    return cleaned


# ---------------------------------------------------------------------------
# Claude Code SDK sandbox helper
# ---------------------------------------------------------------------------


BYPASS_OPT_IN_ENV: Final[str] = "TRAIGENT_UI_ALLOW_CLAUDE_BYPASS"


def safe_claude_code_options(
    options_cls: Any,
    *,
    system_prompt: str | None = None,
    max_turns: int = 1,
    **extra: Any,
) -> Any:
    """Build a ``ClaudeCodeOptions`` instance with sandbox guardrails enforced.

    Default behavior: do NOT set ``permission_mode``. The Claude Code SDK
    keeps its default sandbox in place, which prompts the operator before
    any tool invocation.

    ``permission_mode`` may only be promoted to ``bypassPermissions`` when
    the environment variable named by :data:`BYPASS_OPT_IN_ENV` is set to a
    truthy value AND the caller explicitly passes ``permission_mode=
    "bypassPermissions"`` (or the legacy default flag) in ``extra``. Any
    other value of ``permission_mode`` (e.g. ``"acceptEdits"``) is allowed
    unconditionally; the helper exists to prevent silent fallthrough into
    full bypass, not to police lighter modes.

    A ``ValueError`` is raised if a bypass is requested but the operator
    opt-in is missing.
    """
    desired_mode = extra.pop("permission_mode", None)

    if desired_mode == "bypassPermissions":
        if not _bypass_opt_in_enabled():
            raise ValueError(
                "ClaudeCodeOptions permission_mode='bypassPermissions' requires "
                f"the operator to set {BYPASS_OPT_IN_ENV}=1. Run only in a "
                "local development environment; never enable in production."
            )
        return options_cls(
            system_prompt=system_prompt,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            **extra,
        )

    if desired_mode is None:
        return options_cls(
            system_prompt=system_prompt,
            max_turns=max_turns,
            **extra,
        )

    return options_cls(
        system_prompt=system_prompt,
        max_turns=max_turns,
        permission_mode=desired_mode,
        **extra,
    )


def _bypass_opt_in_enabled() -> bool:
    """Return True only if the operator explicitly opted in via env var."""
    raw = os.environ.get(BYPASS_OPT_IN_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}
