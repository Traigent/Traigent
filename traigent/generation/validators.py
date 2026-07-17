"""Validation/dedupe/injection-sanity helpers for client-side generation.

Seed content and LLM-produced candidates are UNTRUSTED I/O (the llm-security
guidance): a rewritten prompt or synthesized example could carry an injection
payload. These helpers reject empty/oversized candidates, dedupe against what
already exists, and flag obvious prompt-injection markers so a poisoned
candidate is dropped rather than folded into the config space or dataset.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

MAX_PROMPT_CHARS = 8000
MAX_EXAMPLE_CHARS = 20000

# Default-ignorable/format code points that render invisibly (or as zero
# width) but survive lowercasing + whitespace collapse. An attacker can
# interleave these mid-word inside a marker phrase — e.g.
# "ig" + ZERO WIDTH SPACE + "nore previous instructions" — to defeat a plain
# substring scan while the text still *looks* clean to a human. Most of
# these are the Unicode "Format" (Cf) general category, which
# `_strip_scan_ignorables` removes wholesale by category (so newly assigned
# Cf code points are covered without a code change); COMBINING GRAPHEME
# JOINER (U+034F) is a Default_Ignorable_Code_Point that is *not* Cf (its
# general category is Mn), so it is listed explicitly below alongside the
# rest of the deliberately-enumerated set (kept as \u escapes, never as
# literal invisible characters, so the source stays human-auditable).
_EXPLICIT_SCAN_IGNORABLES = frozenset(
    "\u200b\u200c\u200d\u200e\u200f"  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "\u202a\u202b\u202c\u202d\u202e"  # LRE, RLE, PDF, LRO, RLO (bidi overrides)
    "\u2060\u2061\u2062\u2063\u2064"  # word joiner, invisible operators
    "\u2066\u2067\u2068\u2069"  # LRI, RLI, FSI, PDI (bidi isolates)
    "\ufeff"  # BOM / zero width no-break space
    "\u00ad"  # soft hyphen
    "\u034f"  # combining grapheme joiner (Mn, not Cf)
    "\u061c"  # Arabic letter mark
)

# Prompt-injection marker phrases, matched after whitespace normalization via
# plain substring search (no backtracking regex -> no ReDoS surface). Each entry
# is the normalized (lowercased, single-spaced) form.
_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore prior instructions",
    "ignore all prior instructions",
    "ignore above instructions",
    "ignore all above instructions",
    "disregard the system prompt",
    "disregard system prompt",
    "disregard the previous prompt",
    "disregard previous prompt",
    "you are now a developer",
    "you are now developer",
    "you are now a dan",
    "you are now dan",
    "you are now a jailbroken",
    "you are now jailbroken",
    "reveal your system prompt",
    "reveal system prompt",
    "reveal your instructions",
    "reveal instructions",
    "reveal your hidden",
    "reveal hidden",
    # role-tag spoofing (whitespace around the tag is collapsed before matching)
    "</system>",
    "< /system>",
    "<system>",
    "</assistant>",
    "< /assistant>",
    "<assistant>",
)


def _strip_scan_ignorables(text: str) -> str:
    """Return a normalized COPY of ``text`` for injection-marker scanning only.

    NFKC-folds compatibility/fullwidth forms, then drops every Unicode
    "Format" (Cf) code point plus the explicit non-Cf default-ignorables in
    ``_EXPLICIT_SCAN_IGNORABLES`` (see module-level comment). This closes the
    gap where a marker phrase is split by an invisible character — e.g.
    "ig" + ZERO WIDTH SPACE + "nore previous instructions" — which survives
    plain lowercasing + whitespace collapse untouched.

    This function never mutates or returns anything the caller stores: it
    only feeds the marker scan below. Callers keep persisting their own
    original (non-normalized) text when it is clean.
    """
    folded = unicodedata.normalize("NFKC", text)
    return "".join(
        ch
        for ch in folded
        if unicodedata.category(ch) != "Cf" and ch not in _EXPLICIT_SCAN_IGNORABLES
    )


def looks_like_injection(text: str) -> bool:
    """True if the text contains an obvious prompt-injection marker.

    Scans a normalized COPY of ``text`` (NFKC + default-ignorable strip, see
    ``_strip_scan_ignorables``) so zero-width/bidi/format characters cannot
    be used to split a marker past the substring match. The caller's own
    ``text`` is untouched by this check -- only a bool is returned, so clean
    text is stored exactly as received.
    """
    scan_copy = _strip_scan_ignorables(text)
    normalized = re.sub(r"\s+", " ", scan_copy.lower())
    return any(marker in normalized for marker in _INJECTION_MARKERS)


def is_valid_prompt_candidate(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(text) > MAX_PROMPT_CHARS:
        return False
    return not looks_like_injection(stripped)


def clean_prompt_candidates(candidates: list[Any], existing: list[str]) -> list[str]:
    """Filter to valid, non-injection, de-duplicated prompt candidates.

    Preserves order; drops duplicates of each other and of ``existing``.
    """
    seen = {e.strip() for e in existing if isinstance(e, str)}
    out: list[str] = []
    for cand in candidates:
        if not is_valid_prompt_candidate(cand):
            continue
        key = cand.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _example_key(input_data: Any, expected_output: Any) -> str:
    try:
        return json.dumps(
            {"i": input_data, "e": expected_output}, sort_keys=True, default=str
        )
    except (TypeError, ValueError):
        return f"{input_data!r}|{expected_output!r}"


def is_valid_synth_example(input_data: Any, expected_output: Any) -> bool:
    """A synthesized example must have non-empty input + expected output, bounded size, no injection."""
    if not input_data:
        return False
    if expected_output is None or (
        isinstance(expected_output, str) and not expected_output.strip()
    ):
        return False
    blob = f"{input_data!r}{expected_output!r}"
    if len(blob) > MAX_EXAMPLE_CHARS:
        return False
    return not looks_like_injection(blob)


def dedupe_example_keys(
    candidates: list[tuple[Any, Any]],
    existing_keys: set[str],
) -> list[tuple[Any, Any]]:
    """Filter (input_data, expected_output) pairs to valid, novel, non-injection ones."""
    out: list[tuple[Any, Any]] = []
    for input_data, expected_output in candidates:
        if not is_valid_synth_example(input_data, expected_output):
            continue
        key = _example_key(input_data, expected_output)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        out.append((input_data, expected_output))
    return out


def extract_json_block(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response (handles ``` fences).

    Uses bounded string scans (find/rfind), never backtracking regexes, so it is
    not vulnerable to catastrophic-backtracking ReDoS on adversarial LLM output.
    """
    candidate = text.strip()
    # Strip a leading ``` / ```json fence (and its closing fence) via string ops.
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        closing_fence = candidate.rfind("```")
        if first_newline != -1 and closing_fence > first_newline:
            candidate = candidate[first_newline + 1 : closing_fence].strip()
    try:
        return json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        # Fall back to the first {...} or [...] span located by index, not regex.
        obj_start = candidate.find("{")
        arr_start = candidate.find("[")
        if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
            start, closer = arr_start, "]"
        elif obj_start != -1:
            start, closer = obj_start, "}"
        else:
            return None
        end = candidate.rfind(closer)
        if end <= start:
            return None
        try:
            return json.loads(candidate[start : end + 1])
        except (ValueError, json.JSONDecodeError):
            return None


__all__ = [
    "MAX_PROMPT_CHARS",
    "MAX_EXAMPLE_CHARS",
    "looks_like_injection",
    "is_valid_prompt_candidate",
    "clean_prompt_candidates",
    "is_valid_synth_example",
    "dedupe_example_keys",
    "extract_json_block",
]
