"""Golden characterization tests for the legacy ``_compile_constraints``."""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.tvl.spec_loader import TVLValidationError, _compile_constraints

P = Path("spec.tvl.yml")


def test_empty_list() -> None:
    assert _compile_constraints([], P) == []


def test_non_mapping_entry_raises_before_later_entries() -> None:
    with pytest.raises(TVLValidationError, match="constraint entries must be mappings"):
        _compile_constraints([{"rule": "params.x > 1"}, "nope"], P)


def test_expression_default_type_and_ids() -> None:
    out = _compile_constraints(
        [
            {"rule": "params.x > 1"},
            {"id": "named", "type": "EXPRESSION", "rule": "metrics.acc > 0.5"},
        ],
        P,
    )
    assert [c.identifier for c in out] == ["constraint_0", "named"]
    assert [c.constraint_type for c in out] == ["expression", "expression"]
    assert [c.requires_metrics for c in out] == [False, True]
    assert out[0].description == "Constraint constraint_0 violated"
    assert out[0].evaluator({"x": 2}, None) is True
    assert out[0].evaluator({"x": 0}, None) is False
    assert out[1].evaluator({}, {"acc": 0.9}) is True


def test_type_none_or_empty_defaults_to_expression_and_error_message() -> None:
    out = _compile_constraints(
        [{"type": None, "id": "", "rule": "params.x > 1", "error_message": "bad x"}], P
    )
    assert out[0].identifier == "constraint_0"
    assert out[0].constraint_type == "expression"
    assert out[0].description == "bad x"


def test_forbidden_negates_rule() -> None:
    (c,) = _compile_constraints(
        [{"type": "Forbidden", "id": "f", "rule": "params.x > 1"}], P
    )
    assert c.constraint_type == "forbidden"
    assert c.evaluator({"x": 2}, None) is False
    assert c.evaluator({"x": 0}, None) is True


def test_unknown_type_behaves_like_expression() -> None:
    (c,) = _compile_constraints([{"type": "weird", "rule": "params.x > 1"}], P)
    assert c.constraint_type == "weird"
    assert c.evaluator({"x": 2}, None) is True


def test_missing_rule_raises() -> None:
    with pytest.raises(
        TVLValidationError, match="Constraint 'c' requires a string 'rule'"
    ):
        _compile_constraints([{"id": "c", "type": "forbidden", "rule": 5}], P)
    with pytest.raises(TVLValidationError, match="requires a string 'rule'"):
        _compile_constraints([{"id": "c"}], P)


def test_conditional_semantics_and_metrics_flag() -> None:
    (c,) = _compile_constraints(
        [
            {
                "id": "cond",
                "type": "conditional",
                "when": "params.x > 1",
                "then": "metrics.acc > 0.5",
            }
        ],
        P,
    )
    assert c.requires_metrics is True
    assert c.evaluator({"x": 0}, {"acc": 0.0}) is True
    assert c.evaluator({"x": 2}, {"acc": 0.9}) is True
    assert c.evaluator({"x": 2}, {"acc": 0.1}) is False

    (d,) = _compile_constraints(
        [
            {
                "type": "conditional",
                "when": "metrics.a > 1",
                "then": "params.x > 1",
            }
        ],
        P,
    )
    assert d.requires_metrics is True
    (e,) = _compile_constraints(
        [{"type": "conditional", "when": "params.y > 1", "then": "params.x > 1"}], P
    )
    assert e.requires_metrics is False


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "conditional", "id": "c", "when": "params.x > 1"},
        {"type": "conditional", "id": "c", "then": "params.x > 1"},
        {"type": "conditional", "id": "c", "when": 1, "then": "params.x > 1"},
    ],
)
def test_conditional_requires_string_when_and_then(entry: dict) -> None:
    with pytest.raises(
        TVLValidationError, match="Conditional constraint 'c' requires 'when' and"
    ):
        _compile_constraints([entry], P)
