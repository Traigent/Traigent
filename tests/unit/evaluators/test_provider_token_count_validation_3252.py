"""Provider token counts are validated; invalid ones are unknown usage (#3252).

A negative, boolean, float, string or absurdly large count must never be stored
as tokens or priced as a measured $0. It makes the whole usage unknown, so the
example is ``cost_unmeasured`` (``cost_measured`` is False) like any response
that carried no usage.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.evaluators.metrics_tracker import extract_llm_metrics
from traigent.utils.numeric_validation import MAX_TOKEN_COUNT, validate_token_count

BAD_COUNTS = [
    -5,
    True,
    False,
    10**400,
    "12",
    12.0,
    math.nan,
    math.inf,
    MAX_TOKEN_COUNT + 1,
]
BAD_IDS = ["neg", "true", "false", "huge", "str", "float", "nan", "inf", "over-bound"]


class TestValidateTokenCount:
    @pytest.mark.parametrize("value", [0, 1, 7, MAX_TOKEN_COUNT])
    def test_accepts_real_nonnegative_ints(self, value: int) -> None:
        assert validate_token_count(value) == value

    @pytest.mark.parametrize("value", [*BAD_COUNTS, None, [], {}])
    def test_rejects_everything_else(self, value: Any) -> None:
        assert validate_token_count(value) is None


def _openai(prompt: Any, completion: Any, total: Any = 18) -> Any:
    return SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
        ),
    )


def _anthropic(inp: Any, out: Any) -> Any:
    return SimpleNamespace(
        model="claude-3-5-sonnet-20241022",
        usage={"input_tokens": inp, "output_tokens": out},
    )


def _langchain_usage_metadata(inp: Any, out: Any) -> Any:
    return SimpleNamespace(
        usage_metadata={"input_tokens": inp, "output_tokens": out},
        response_metadata={},
    )


def _langchain_token_usage(inp: Any, out: Any) -> Any:
    return SimpleNamespace(
        response_metadata={
            "token_usage": {"prompt_tokens": inp, "completion_tokens": out}
        }
    )


def _dict_usage(inp: Any, out: Any) -> Any:
    return {"usage": {"prompt_tokens": inp, "completion_tokens": out}}


def _dict_direct(inp: Any, out: Any) -> Any:
    return {"input_tokens": inp, "output_tokens": out}


def _generic(inp: Any, out: Any) -> Any:
    return SimpleNamespace(
        metadata={"usage": {"input_tokens": inp, "output_tokens": out}}
    )


BUILDERS = [
    pytest.param(lambda bad: _openai(bad, 7), id="openai-prompt"),
    pytest.param(lambda bad: _openai(11, bad), id="openai-completion"),
    pytest.param(lambda bad: _openai(11, 7, bad), id="openai-total"),
    pytest.param(lambda bad: _anthropic(bad, 7), id="anthropic"),
    pytest.param(lambda bad: _langchain_usage_metadata(bad, 7), id="lc-usage-metadata"),
    pytest.param(lambda bad: _langchain_token_usage(bad, 7), id="lc-token-usage"),
    pytest.param(lambda bad: _dict_usage(bad, 7), id="dict-usage"),
    pytest.param(lambda bad: _dict_direct(bad, 7), id="dict-direct"),
    pytest.param(lambda bad: _generic(bad, 7), id="generic-metadata"),
]


@pytest.mark.parametrize("build", BUILDERS)
@pytest.mark.parametrize("bad", BAD_COUNTS, ids=BAD_IDS)
@pytest.mark.parametrize("strict", ["true", "false"])
def test_invalid_count_is_unknown_usage_never_measured_zero(
    build: Any, bad: Any, strict: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", strict)
    monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "false")
    metrics = extract_llm_metrics(build(bad), model_name="gpt-4o-mini")

    assert metrics.tokens.input_tokens == 0
    assert metrics.tokens.output_tokens == 0
    assert metrics.tokens.total_tokens == 0
    assert metrics.cost.unmeasured is True
    assert metrics.cost_measured is False


@pytest.mark.parametrize(
    "build",
    [
        lambda: _openai(11, 7, 18),
        lambda: _anthropic(11, 7),
        lambda: _langchain_usage_metadata(11, 7),
        lambda: _langchain_token_usage(11, 7),
        lambda: _dict_usage(11, 7),
        lambda: _dict_direct(11, 7),
        lambda: _generic(11, 7),
    ],
    ids=[
        "openai",
        "anthropic",
        "lc-usage",
        "lc-token-usage",
        "dict-usage",
        "dict",
        "generic",
    ],
)
def test_valid_counts_still_priced_exactly(
    build: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.utils.cost_calculator import cost_from_tokens

    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "false")
    model = "claude-3-5-sonnet-20241022" if "claude" in str(build()) else "gpt-4o-mini"
    metrics = extract_llm_metrics(build(), model_name=model)

    assert (metrics.tokens.input_tokens, metrics.tokens.output_tokens) == (11, 7)
    assert metrics.tokens.total_tokens == 18
    expected_in, expected_out = cost_from_tokens(11, 7, model, strict=True)
    assert metrics.cost.total_cost == pytest.approx(expected_in + expected_out)
    assert metrics.cost.total_cost > 0
    assert metrics.cost_measured is True
