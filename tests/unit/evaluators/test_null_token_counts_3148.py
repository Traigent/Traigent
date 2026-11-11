"""#3148: null/non-numeric provider token counts must not abort extraction."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-EVALUATORS
# Traceability: REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.evaluators.metrics_tracker import extract_llm_metrics

NULLS = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def _tokens(response: object):
    return extract_llm_metrics(response, model_name="gpt-4o-mini").tokens


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(response_metadata={"token_usage": dict(NULLS)}, content="x"),
        SimpleNamespace(
            response_metadata={"token_usage": {"prompt_tokens": None}}, content="x"
        ),
        SimpleNamespace(
            usage_metadata={"input_tokens": None, "output_tokens": None}, content="x"
        ),
        SimpleNamespace(llm_output={"token_usage": dict(NULLS)}, content="x"),
        SimpleNamespace(llm_output={"token_usage": None}, content="x"),
        SimpleNamespace(
            response_metadata={
                "usage": {"prompt_tokens": None, "completion_tokens": None}
            },
            content="x",
        ),
        {"input_tokens": None, "output_tokens": None},
        {"usage": {"prompt_tokens": None, "completion_tokens": None}},
        {"input_tokens": "10", "output_tokens": "5"},
        {"input_tokens": True, "output_tokens": [1]},
    ],
)
def test_null_or_non_numeric_counts_are_unmeasured_not_an_error(response) -> None:
    tokens = _tokens(response)

    assert (tokens.input_tokens, tokens.output_tokens, tokens.total_tokens) == (
        0,
        0,
        0,
    )


def test_partial_null_keeps_the_valid_field() -> None:
    response = SimpleNamespace(
        usage_metadata={"input_tokens": None, "output_tokens": 5}, content="x"
    )

    tokens = _tokens(response)

    assert (tokens.input_tokens, tokens.output_tokens, tokens.total_tokens) == (0, 5, 5)


def test_valid_counts_unchanged() -> None:
    response = SimpleNamespace(
        response_metadata={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        },
        content="x",
    )

    tokens = _tokens(response)

    assert (tokens.input_tokens, tokens.output_tokens, tokens.total_tokens) == (
        10,
        5,
        15,
    )


def test_dict_response_valid_counts_unchanged() -> None:
    tokens = _tokens({"input_tokens": 3, "output_tokens": 4})

    assert (tokens.input_tokens, tokens.output_tokens, tokens.total_tokens) == (3, 4, 7)


def test_numpy_token_counts_are_counted() -> None:
    """numpy integer counts are real counts, not unmeasured (base counted them)."""
    np = pytest.importorskip("numpy")
    from traigent.evaluators.metrics_tracker import _token_count
    from traigent.integrations.bedrock_client import _int_or_none

    assert _token_count(np.int64(7)) == 7
    assert _token_count(np.float32(3.0)) == 3
    assert _token_count(np.float64("nan")) == 0
    assert _token_count(True) == 0
    assert _int_or_none(np.int64(7)) == 7
    assert _int_or_none(True) is None
