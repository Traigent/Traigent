"""Traigent#3218: MCP cost estimate must not substring-match unknown models."""

from __future__ import annotations

import pytest

from traigent.mcp.tools import _estimate_model_cost_per_call


@pytest.mark.parametrize(
    "name",
    ["gpt-4o-mini-high", "gemini-1.5-flash-8b", "gpt-4oo", "claude-haiku-4-5-typo"],
)
def test_unknown_stem_sharing_names_are_unpriced(name: str) -> None:
    assert _estimate_model_cost_per_call(name) == (None, None)


@pytest.mark.parametrize(
    "name", ["gpt-4o", "openai/gpt-4o", "OpenAI/GPT-4o", "gpt-4o-2024-08-06"]
)
def test_exact_prefixed_and_dated_names_get_the_base_estimate(name: str) -> None:
    base = _estimate_model_cost_per_call("gpt-4o")
    assert base[0] is not None and base[1] == "sdk_estimation_table"
    assert _estimate_model_cost_per_call(name) == base


def test_no_model_is_unpriced() -> None:
    assert _estimate_model_cost_per_call(None) == (None, None)
