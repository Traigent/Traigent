"""Classify metric names by whole underscore/dash-delimited tokens.

Substring tests misclassify names that carry two markers (``token_f1`` holds
``token`` and ``f1``; ``programs_passed`` holds ``ms``). Matching whole tokens
and resolving the few ambiguous shapes in one place keeps the callers agreeing.
Where a run has a declared objective orientation, prefer that; this is the
fallback for bare names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Connectors that turn "quality" + "unit" into a ratio of the two, e.g.
# ``cost_per_accuracy_point`` or ``time_to_accuracy_s``: the unit side wins.
_RATIO_CONNECTORS = frozenset({"per", "to", "vs", "over", "ratio"})
# ``error_free_rate``, ``bug_free_rate``: the absence of the unit word is the
# quality being measured.
_NEGATING_TOKENS = frozenset({"free"})
# Unit words that are also ordinary nouns in quality names: ``token_f1`` is an
# F1 score computed over tokens, so a quality marker beats these. A strong unit
# word (``time``, ``cost``...) beside a quality marker keeps the name a unit
# metric (``score_time``).
_WEAK_UNIT_TOKENS = frozenset({"token"})
# Tokens saying a cost-named metric is not an amount of money.
_NON_AMOUNT_TOKENS = frozenset(
    {"ratio", "efficiency", "count", "counts", "pct", "percent", "rank", "rate"}
)


_CAMEL_DIGIT_RE = re.compile(r"([A-Z][a-z]+)(?=[0-9])")
_CAMEL_LOWER_UPPER_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_ACRONYM_RE = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def metric_name_tokens(name: str) -> tuple[str, ...]:
    """Lower-cased tokens of ``name``.

    Splits on every non-alphanumeric run and on camelCase / PascalCase
    boundaries (``totalCost`` -> total, cost; ``accuracyPer1kCost`` ->
    accuracy, per, 1k, cost). An all-lowercase letter+digit token such as
    ``f1`` or ``p95`` stays whole.
    """
    spaced = _CAMEL_DIGIT_RE.sub(r"\1_", name)
    spaced = _CAMEL_LOWER_UPPER_RE.sub("_", spaced)
    spaced = _CAMEL_ACRONYM_RE.sub("_", spaced)
    return tuple(tok for tok in _SPLIT_RE.split(spaced.lower()) if tok)


def _token_matches(token: str, markers: frozenset[str]) -> bool:
    if token in markers:
        return True
    if token.endswith("es") and token[:-2] in markers:
        return True
    return token.endswith("s") and token[:-1] in markers


def has_marker_token(name: str, markers: Iterable[str]) -> bool:
    """True when some whole token of ``name`` is (a plural of) a marker."""
    marker_set = frozenset(markers)
    return any(_token_matches(tok, marker_set) for tok in metric_name_tokens(name))


def classify_metric_kind(
    name: str,
    quality_markers: Iterable[str],
    unit_markers: Iterable[str],
) -> str | None:
    """Return ``"quality"``, ``"continuous"`` or ``None`` for a metric name.

    A name with only one kind of marker gets that kind. A name with both is a
    ratio when a connector token (``per``, ``to``...) is present, and then the
    unit side wins. A name with only unit words is quality when it has a
    ``free`` token (``error_free_rate``). A weak unit word such as ``token`` loses
    to a quality marker (``token_f1`` is an F1 score, not a token count) while a
    strong one such as ``time`` wins (``score_time``).
    """
    tokens = metric_name_tokens(name)
    quality = frozenset(quality_markers)
    unit = frozenset(unit_markers)
    is_quality = any(_token_matches(tok, quality) for tok in tokens)
    unit_hits = [tok for tok in tokens if _token_matches(tok, unit)]
    is_unit = bool(unit_hits)
    if is_quality and is_unit:
        if any(tok in _RATIO_CONNECTORS for tok in tokens):
            return "continuous"
        if all(_token_matches(tok, _WEAK_UNIT_TOKENS) for tok in unit_hits):
            return "quality"
        return "continuous"
    if is_unit and any(tok in _NEGATING_TOKENS for tok in tokens):
        return "quality"
    if is_quality:
        return "quality"
    if is_unit:
        return "continuous"
    return None


def depends_on_cost(name: str) -> bool:
    """True when ``name`` is derived from cost on either side (``cost_efficiency``,
    ``accuracy_per_1k_cost``, ``total_cost``): unknowable while cost is unmeasured."""
    return has_marker_token(name, ("cost",))


def is_usd_cost_metric(name: str) -> bool:
    """True when ``name`` denotes an amount of money (``total_cost``).

    Counts, percentages, ranks and quality-per-cost ratios
    (``accuracy_per_1k_cost``) carry the word cost but are not dollars.
    ``cost_per_call`` is still money.
    """
    tokens = metric_name_tokens(name)
    cost_positions = [i for i, tok in enumerate(tokens) if tok in {"cost", "costs"}]
    if not cost_positions:
        return False
    if any(tok in _NON_AMOUNT_TOKENS for tok in tokens):
        return False
    if any(
        a == "per" and b == "dollar" for a, b in zip(tokens, tokens[1:], strict=False)
    ):
        return False
    first_cost = cost_positions[0]
    if "per" in tokens[:first_cost]:
        return False
    return True
