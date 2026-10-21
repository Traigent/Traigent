"""Provider-billed cost is exempt from the plausibility floor (#2505)."""

from __future__ import annotations

import pytest

from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    COST_SOURCE_MIXED,
    COST_SOURCE_PROVIDER_BILLED,
    COST_SOURCE_SELF_REPORTED,
    COST_SOURCE_TABLE,
    ExampleMetrics,
    MetricsTracker,
    extract_llm_metrics,
)
from traigent.utils.cost_calculator import cost_from_tokens

MODEL = "gpt-4o-mini"
IN_TOK, OUT_TOK = 1000, 500


def _table_total() -> float:
    return sum(cost_from_tokens(IN_TOK, OUT_TOK, MODEL, strict=False))


class _Usage:
    prompt_tokens = IN_TOK
    completion_tokens = OUT_TOK
    total_tokens = IN_TOK + OUT_TOK

    def __init__(self, cost: float) -> None:
        self.cost = cost


def _billed_response(cost: float):
    class _Resp:
        model = MODEL
        usage = _Usage(cost)

    return _Resp()


@pytest.fixture(autouse=True)
def _no_mocks(monkeypatch):
    monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")


def test_provider_billed_cost_below_floor_is_kept(caplog):
    billed = _table_total() * 0.1  # far below the 0.5 floor
    with caplog.at_level("WARNING"):
        m = extract_llm_metrics(_billed_response(billed), model_name=MODEL)
    assert m.cost.total_cost == pytest.approx(billed)
    assert m.cost.cost_source == COST_SOURCE_PROVIDER_BILLED
    assert "implausibly below" not in caplog.text
    # Breakdown is backfilled and consistent with the billed total.
    assert m.cost.input_cost + m.cost.output_cost == pytest.approx(billed)


def test_self_reported_cost_below_floor_is_still_floored(caplog):
    m = ExampleMetrics()
    m.tokens.input_tokens, m.tokens.output_tokens = IN_TOK, OUT_TOK
    m.tokens.total_tokens = IN_TOK + OUT_TOK
    m.cost.total_cost = _table_total() * 0.1
    m.cost.cost_source = COST_SOURCE_SELF_REPORTED
    from traigent.evaluators.metrics_tracker import (
        _reconcile_reported_cost_with_tokens,
    )

    with caplog.at_level("WARNING"):
        _reconcile_reported_cost_with_tokens(m, MODEL)
    assert m.cost.total_cost == pytest.approx(_table_total())
    assert m.cost.cost_source == COST_SOURCE_TABLE
    assert "implausibly below token-derived estimate" in caplog.text


def test_unknown_source_fails_closed_and_keeps_floor():
    m = ExampleMetrics()
    m.tokens.input_tokens, m.tokens.output_tokens = IN_TOK, OUT_TOK
    m.cost.total_cost = _table_total() * 0.1
    assert m.cost.cost_source is None
    from traigent.evaluators.metrics_tracker import (
        _reconcile_reported_cost_with_tokens,
    )

    _reconcile_reported_cost_with_tokens(m, MODEL)
    assert m.cost.total_cost == pytest.approx(_table_total())


def test_injected_meta_cost_overrides_provider_tag_and_keeps_floor():
    ev = LocalEvaluator.__new__(LocalEvaluator)
    m = extract_llm_metrics(_billed_response(_table_total() * 0.1), model_name=MODEL)
    assert m.cost.cost_source == COST_SOURCE_PROVIDER_BILLED

    class _Out(dict):
        pass

    out = {"__traigent_meta__": {"total_cost": _table_total() * 0.1}}
    ev._extract_and_inject_traigent_meta(out, m, model_name=MODEL)
    assert m.cost.cost_source in (COST_SOURCE_SELF_REPORTED, COST_SOURCE_TABLE)
    assert m.cost.total_cost == pytest.approx(_table_total())


def test_price_table_cost_is_tagged_table():
    class _Resp:
        model = MODEL

        class usage:  # noqa: N801
            prompt_tokens = IN_TOK
            completion_tokens = OUT_TOK
            total_tokens = IN_TOK + OUT_TOK

    m = extract_llm_metrics(_Resp(), model_name=MODEL)
    assert m.cost.cost_source == COST_SOURCE_TABLE


def _row(source: str | None, cost: float = 0.001) -> ExampleMetrics:
    r = ExampleMetrics()
    r.cost.total_cost = cost
    r.cost.cost_source = source
    r.measured = True
    return r


def test_per_trial_cost_source_recorded_and_mixed():
    t = MetricsTracker()
    t.example_metrics = [
        _row(COST_SOURCE_PROVIDER_BILLED),
        _row(COST_SOURCE_PROVIDER_BILLED),
    ]
    assert t.cost_source_label() == COST_SOURCE_PROVIDER_BILLED
    t.example_metrics.append(_row(COST_SOURCE_SELF_REPORTED))
    assert t.cost_source_label() == COST_SOURCE_MIXED
    assert MetricsTracker().cost_source_label() is None


def test_combine_calls_mixed_and_uniform_source():
    a, b = _row(COST_SOURCE_PROVIDER_BILLED), _row(COST_SOURCE_PROVIDER_BILLED)
    assert (
        LocalEvaluator._combine_call_metrics([a, b]).cost.cost_source
        == COST_SOURCE_PROVIDER_BILLED
    )
    c = _row(COST_SOURCE_TABLE)
    assert (
        LocalEvaluator._combine_call_metrics([a, c]).cost.cost_source
        == COST_SOURCE_MIXED
    )
