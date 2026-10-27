"""Characterization of AgentCostBreakdown validation (messages, order, edges)."""

from __future__ import annotations

import pytest

from traigent.cloud.agent_dtos import AgentCostBreakdown

HEADER = "AgentCostBreakdown validation failed:\n"
INF = float("inf")
NAN = float("nan")

BASE = {
    "agent_id": "a1",
    "agent_name": "Agent One",
    "input_tokens": 10,
    "output_tokens": 5,
    "total_tokens": 15,
    "input_cost": 0.25,
    "output_cost": 0.5,
    "total_cost": 0.75,
    "model_used": "m1",
}


def _build(**over):
    return AgentCostBreakdown(**{**BASE, **over})


def _errors(**over):
    with pytest.raises(ValueError) as err:
        _build(**over)
    text = str(err.value)
    assert text.startswith(HEADER)
    lines = text[len(HEADER) :].split("\n")
    assert all(line.startswith("  - ") for line in lines)
    return [line[4:] for line in lines]


class TestAcceptedValues:
    def test_valid_instance_keeps_fields(self):
        item = _build()
        assert (item.agent_id, item.total_tokens, item.total_cost) == ("a1", 15, 0.75)

    def test_zero_values_are_valid(self):
        item = _build(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
        )
        assert item.total_tokens == 0

    def test_cost_tolerance_boundary_is_inclusive(self):
        item = _build(input_cost=0.0, output_cost=0.0, total_cost=0.0001)
        assert item.total_cost == 0.0001

    def test_cost_just_past_tolerance_is_rejected(self):
        assert _errors(input_cost=0.0, output_cost=0.0, total_cost=0.00011) == [
            "total_cost (0.00011000) must equal input_cost (0.00000000) + "
            "output_cost (0.00000000) = 0.00000000"
        ]

    def test_negative_difference_uses_absolute_value(self):
        assert len(_errors(total_cost=0.74)) == 1
        assert len(_errors(total_cost=0.76)) == 1


class TestIdentifiers:
    @pytest.mark.parametrize(
        ("field", "message"),
        [
            ("agent_id", "agent_id cannot be empty"),
            ("agent_name", "agent_name cannot be empty"),
            ("model_used", "model_used cannot be empty"),
        ],
    )
    @pytest.mark.parametrize("empty", ["", None])
    def test_each_empty_identifier_is_rejected(self, field, message, empty):
        assert _errors(**{field: empty}) == [message]

    def test_all_empty_identifiers_report_in_field_order(self):
        assert _errors(agent_id="", agent_name="", model_used="") == [
            "agent_id cannot be empty",
            "agent_name cannot be empty",
            "model_used cannot be empty",
        ]

    def test_whitespace_identifier_is_accepted(self):
        assert _build(agent_id=" ").agent_id == " "


class TestTokens:
    def test_negative_input_tokens(self):
        assert _errors(input_tokens=-1, total_tokens=4) == [
            "input_tokens cannot be negative: -1"
        ]

    def test_negative_output_tokens(self):
        assert _errors(output_tokens=-1, total_tokens=9) == [
            "output_tokens cannot be negative: -1"
        ]

    def test_negative_total_tokens_also_breaks_arithmetic(self):
        assert _errors(total_tokens=-1) == [
            "total_tokens cannot be negative: -1",
            "total_tokens (-1) must equal input_tokens (10) + output_tokens (5) = 15",
        ]

    def test_arithmetic_mismatch(self):
        assert _errors(total_tokens=16) == [
            "total_tokens (16) must equal input_tokens (10) + output_tokens (5) = 15"
        ]

    def test_all_token_problems_in_order(self):
        assert _errors(input_tokens=-2, output_tokens=-3, total_tokens=-4) == [
            "input_tokens cannot be negative: -2",
            "output_tokens cannot be negative: -3",
            "total_tokens cannot be negative: -4",
            "total_tokens (-4) must equal input_tokens (-2) + output_tokens (-3) = -5",
        ]


class TestCostFields:
    @pytest.mark.parametrize("field", ["input_cost", "output_cost", "total_cost"])
    def test_negative_cost_is_reported_for_each_field(self, field):
        errors = _errors(**{field: -1.0})
        assert errors[0] == f"{field} cannot be negative: -1.0"

    def test_negative_input_cost_full_result(self):
        assert _errors(input_cost=-0.25, total_cost=0.25) == [
            "input_cost cannot be negative: -0.25"
        ]

    def test_nan_costs(self):
        assert _errors(input_cost=NAN, total_cost=0.5) == ["input_cost cannot be NaN"]

    def test_nan_total_cost_skips_arithmetic_message(self):
        assert _errors(total_cost=NAN) == ["total_cost cannot be NaN"]

    def test_positive_infinity_in_parts_and_total(self):
        assert _errors(output_cost=INF, total_cost=INF) == [
            "output_cost cannot be Inf",
            "total_cost cannot be Inf",
        ]

    def test_infinite_total_with_finite_parts(self):
        assert _errors(total_cost=INF) == [
            "total_cost cannot be Inf",
            "total_cost (inf) must equal input_cost (0.25000000) + "
            "output_cost (0.50000000) = 0.75000000",
        ]

    def test_negative_infinity_reports_negative_then_inf(self):
        errors = _errors(input_cost=-INF, total_cost=0.5)
        assert errors[:2] == [
            "input_cost cannot be negative: -inf",
            "input_cost cannot be Inf",
        ]

    def test_fields_are_checked_in_input_output_total_order(self):
        errors = _errors(input_cost=-1.0, output_cost=-2.0, total_cost=-3.0)
        assert errors[:3] == [
            "input_cost cannot be negative: -1.0",
            "output_cost cannot be negative: -2.0",
            "total_cost cannot be negative: -3.0",
        ]

    def test_cost_arithmetic_message_formatting(self):
        assert _errors(total_cost=1.0) == [
            "total_cost (1.00000000) must equal input_cost (0.25000000) + "
            "output_cost (0.50000000) = 0.75000000"
        ]


class TestCombinedOrder:
    def test_sections_are_reported_in_identifier_token_cost_order(self):
        errors = _errors(
            agent_id="",
            total_tokens=99,
            total_cost=9.0,
        )
        assert errors == [
            "agent_id cannot be empty",
            "total_tokens (99) must equal input_tokens (10) + output_tokens (5) = 15",
            "total_cost (9.00000000) must equal input_cost (0.25000000) + "
            "output_cost (0.50000000) = 0.75000000",
        ]


class TestMalformedInputs:
    @pytest.mark.parametrize(
        "over",
        [
            {"input_tokens": "10"},
            {"output_tokens": None},
            {"total_tokens": "15"},
            {"input_cost": "0.25"},
            {"output_cost": None},
            {"total_cost": "0.75"},
        ],
    )
    def test_non_numeric_values_raise_type_error_not_validation_error(self, over):
        with pytest.raises(TypeError):
            _build(**over)

    def test_type_error_wins_over_earlier_identifier_errors(self):
        with pytest.raises(TypeError):
            _build(agent_id="", input_cost="x")

    def test_integer_costs_are_accepted(self):
        item = _build(input_cost=1, output_cost=2, total_cost=3)
        assert item.total_cost == 3
