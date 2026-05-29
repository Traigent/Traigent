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
from typing import Any

MAX_PROMPT_CHARS = 8000
MAX_EXAMPLE_CHARS = 20000

# Conservative prompt-injection markers. A candidate matching any is dropped.
_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE
    ),
    re.compile(r"disregard\s+(the\s+)?(system|previous)\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(developer|dan|jailbroken)", re.IGNORECASE),
    re.compile(
        r"reveal\s+(your\s+)?(system\s+prompt|instructions|hidden)", re.IGNORECASE
    ),
    re.compile(r"<\s*/?\s*(system|assistant)\s*>", re.IGNORECASE),
]


def looks_like_injection(text: str) -> bool:
    """True if the text contains an obvious prompt-injection marker."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


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
