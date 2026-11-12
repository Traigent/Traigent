"""Traigent#3218: builtin pricing must not prefix-match unknown model names."""

from __future__ import annotations

import pytest

from traigent.utils.cost_calculator import (
    UnknownModelError,
    _find_fallback_pricing,
    cost_from_tokens,
    get_model_token_pricing,
)

UNKNOWN_SIBLING_NAMES = [
    "gpt-4oo",
    "gpt-4o2",
    "gpt-4o-mini-high",
    "claude-haiku-4-5-typo",
    "gemini-1.5-flash-8b",
]


@pytest.mark.parametrize("name", UNKNOWN_SIBLING_NAMES)
def test_unknown_stem_sharing_names_raise_under_strict(name: str) -> None:
    with pytest.raises(UnknownModelError):
        cost_from_tokens(1000, 500, name, strict=True)


@pytest.mark.parametrize("name", UNKNOWN_SIBLING_NAMES)
def test_unknown_stem_sharing_names_are_not_builtin_priced(name: str) -> None:
    with pytest.raises(UnknownModelError):
        get_model_token_pricing(name)
    assert _find_fallback_pricing(name) == (None, None)


@pytest.mark.parametrize(
    ("snapshot", "base"),
    [
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
    ],
)
def test_dated_snapshots_still_price_as_their_base(snapshot: str, base: str) -> None:
    assert cost_from_tokens(1000, 500, snapshot, strict=True) == cost_from_tokens(
        1000, 500, base, strict=True
    )


def test_fallback_matches_only_exact_key_plus_snapshot_suffix() -> None:
    _, exact_key = _find_fallback_pricing("gpt-4o")
    assert exact_key == "gpt-4o"
    _, dated_key = _find_fallback_pricing("gpt-4o-2024-08-06")
    assert dated_key == "gpt-4o"
    # The longer key wins: not mispriced as gpt-4o.
    _, mini_key = _find_fallback_pricing("gpt-4o-mini-2024-07-18")
    assert mini_key == "gpt-4o-mini"
    assert _find_fallback_pricing("gpt-4o-2024-08-06-extra") == (None, None)
