"""Partial provider usage (one of input/output missing) is unknown, not cheap."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import _checked_tokens, extract_llm_metrics


@pytest.mark.parametrize(
    ("inp", "out", "total"),
    [(11, None, None), (None, 7, None), (11, None, 20), (None, 7, 30)],
)
def test_partial_counts_are_unknown(inp, out, total):
    assert _checked_tokens(inp, out, total).invalid is True


@pytest.mark.parametrize(
    ("inp", "out", "total", "expected"),
    [
        (11, 7, None, (11, 7, 18)),
        (None, None, None, (0, 0, 0)),
        (11, None, 11, (11, 0, 11)),  # input-only endpoint (e.g. embeddings)
        (None, 7, 7, (0, 7, 7)),
    ],
)
def test_complete_or_consistent_counts_are_kept(inp, out, total, expected):
    t = _checked_tokens(inp, out, total)
    assert t.invalid is False
    assert (t.input_tokens, t.output_tokens, t.total_tokens) == expected


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 11, "completion_tokens": None},
        {"prompt_tokens": 11},
        {"input_tokens": 11, "output_tokens": None},
    ],
)
def test_partial_usage_response_is_unmeasured_not_priced(usage):
    response = {"usage": usage, "model": "gpt-4o-mini"}
    m = extract_llm_metrics(
        response=response, model_name="gpt-4o-mini", estimate_from_text=False
    )
    assert m.cost.unmeasured is True
    assert m.cost.total_cost == 0


def test_input_only_usage_with_matching_total_is_priced():
    response = {"usage": {"prompt_tokens": 11, "total_tokens": 11}}
    m = extract_llm_metrics(
        response=response, model_name="gpt-4o-mini", estimate_from_text=False
    )
    assert m.cost.unmeasured is False
    assert m.tokens.input_tokens == 11


def test_price_one_call_disables_text_estimation():
    with patch("traigent.evaluators.local.extract_llm_metrics") as mocked:
        mocked.return_value = SimpleNamespace(
            tokens=SimpleNamespace(total_tokens=0),
            cost=SimpleNamespace(cost_explicit=False, unmeasured=True, total_cost=0),
        )
        try:
            LocalEvaluator._price_one_call(
                object(), agent_model="gpt-4o-mini", is_agent_call=True
            )
        except Exception:
            pass
    assert mocked.call_args.kwargs["estimate_from_text"] is False
