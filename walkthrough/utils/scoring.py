"""Shared scoring utilities for walkthrough examples.

This module provides token-based matching for evaluating LLM outputs against
expected answers. Used by real examples that need fuzzy matching to handle
paraphrased responses.
"""

from __future__ import annotations

import re

_COMMA_IN_NUMBER_RE = re.compile(r"(?<=\d),(?=\d)")
_WORD_RE = re.compile(r"[a-z0-9]+")
_DIGIT_RE = re.compile(r"^\d+$")

# Maximum character difference for prefix matching (e.g., "optimize" matches "optimization")
_PREFIX_MATCH_MAX = 4

# Minimum fraction of expected tokens that must match
_REQUIRED_FRACTION = 0.8

# Common words to ignore when comparing tokens
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "both",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "was",
    "were",
    "that",
    "this",
    "about",
    "into",
}


def tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens, normalizing comma-separated numbers."""
    normalized = _COMMA_IN_NUMBER_RE.sub("", text.lower())
    return _WORD_RE.findall(normalized)


def token_matches(expected_token: str, output_tokens: set[str]) -> bool:
    """Check if expected_token matches any output token (exact or prefix match).

    For numeric tokens, requires exact match.
    For text tokens, allows prefix matching within _PREFIX_MATCH_MAX characters.
    """
    if expected_token in output_tokens:
        return True
    if _DIGIT_RE.match(expected_token):
        return False
    for token in output_tokens:
        if (
            token.startswith(expected_token)
            and len(token) - len(expected_token) <= _PREFIX_MATCH_MAX
        ):
            return True
        if (
            expected_token.startswith(token)
            and len(expected_token) - len(token) <= _PREFIX_MATCH_MAX
        ):
            return True
    return False


def token_match_score(output: str, expected: str, **_) -> float:
    """Return 1.0 when >=80% of expected tokens appear in output (case-insensitive).

    This is a fuzzy matching scorer that:
    - Tokenizes both strings into lowercase words
    - Removes common stopwords
    - Allows prefix matching for text tokens
    - Requires exact match for numeric tokens
    - Returns 1.0 if at least 80% of expected tokens match, else 0.0
    """
    if output is None or expected is None:
        return 0.0
    output_text = str(output)
    expected_text = str(expected).strip()
    if not expected_text:
        return 0.0
    output_tokens = {t for t in tokenize(output_text) if t not in STOPWORDS}
    expected_tokens = [t for t in tokenize(expected_text) if t not in STOPWORDS]
    if not expected_tokens:
        return 0.0
    matched = sum(token_matches(token, output_tokens) for token in expected_tokens)
    return 1.0 if (matched / len(expected_tokens)) >= _REQUIRED_FRACTION else 0.0


def semantic_overlap_score(output: str, expected: str, **_) -> float:
    """Score based on key-term overlap between output and expected.

    Returns a continuous score (0.0 to 1.0) based on what fraction of
    expected tokens appear in the output. Good for RAG and open-ended answers.
    """
    if output is None or expected is None:
        return 0.0
    output_tokens = set(tokenize(str(output))) - STOPWORDS
    expected_tokens = set(tokenize(str(expected))) - STOPWORDS
    if not expected_tokens:
        return 0.0
    overlap = len(output_tokens & expected_tokens) / len(expected_tokens)
    return float(max(0.0, min(1.0, overlap)))
