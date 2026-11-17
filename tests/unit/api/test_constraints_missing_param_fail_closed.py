"""Regression tests for #3230: missing/unresolved params stay failures under NOT."""

from __future__ import annotations

import pytest

from traigent.api.constraints import implies, require
from traigent.api.parameter_ranges import Choices, Range


@pytest.fixture
def t() -> Range:
    return Range(0.0, 1.0, step=0.25, name="t")


@pytest.fixture
def m() -> Choices:
    return Choices(["a", "b"], name="m")


def test_negated_atom_on_missing_param_fails_closed(t: Range) -> None:
    names = {id(t): "t"}
    assert t.gt(0.5).evaluate_config({}, names) is False
    assert (~t.gt(0.5)).evaluate_config({}, names) is False
    assert (~t.lte(0.5)).evaluate_config({}, names) is False


def test_negated_atom_on_unresolved_param_fails_closed(t: Range) -> None:
    assert (~t.gt(0.5)).evaluate_config({"t": 0.25}, {}) is False


def test_not_equals_and_negated_equals_agree_when_missing(m: Choices) -> None:
    names = {id(m): "m"}
    assert m.not_equals("a").evaluate_config({}, names) is False
    assert (~m.equals("a")).evaluate_config({}, names) is False


def test_negation_still_works_when_present(t: Range) -> None:
    names = {id(t): "t"}
    assert (~t.gt(0.5)).evaluate_config({"t": 0.25}, names) is True
    assert (~t.gt(0.5)).evaluate_config({"t": 0.75}, names) is False
    assert (~~t.gt(0.5)).evaluate_config({"t": 0.75}, names) is True
    assert (~~t.gt(0.5)).evaluate_config({}, names) is False


def test_unknown_propagates_through_and_or(t: Range, m: Choices) -> None:
    names = {id(t): "t", id(m): "m"}
    # unknown AND true -> unknown -> failure; negated stays failure
    expr = m.equals("a") & t.lte(0.5)
    assert expr.evaluate_config({"t": 0.25}, names) is False
    assert (~expr).evaluate_config({"t": 0.25}, names) is False
    # false AND unknown -> definitely false -> negation is true
    assert (~(m.equals("a") & t.gt(0.9))).evaluate_config({"t": 0.25}, names) is True
    # true OR unknown -> true
    assert (m.equals("a") | t.lte(0.5)).evaluate_config({"t": 0.25}, names) is True
    assert (~(m.equals("a") | t.lte(0.5))).evaluate_config({"t": 0.25}, names) is False


def test_implies_with_missing_guard_param_fails_closed(t: Range, m: Choices) -> None:
    names = {id(t): "t", id(m): "m"}
    c = implies(m.equals("a"), t.lte(0.5))
    assert c.evaluate({"t": 0.7}, names) is False
    assert c.evaluate({"t": 0.25}, names) is False
    # present guard keeps normal semantics
    assert c.evaluate({"m": "b", "t": 0.7}, names) is True
    assert c.evaluate({"m": "a", "t": 0.7}, names) is False
    assert c.evaluate({"m": "a", "t": 0.25}, names) is True


def test_negated_standalone_constraint_on_missing_param(t: Range) -> None:
    c = require(~t.gt(0.5))
    assert c.evaluate({}, {id(t): "t"}) is False


def test_and_or_short_circuit_on_definite_result(t: Range, m: Choices) -> None:
    names = {id(t): "t", id(m): "m"}
    # False first operand must not evaluate a later comparison against None.
    assert (m.equals("b") & t.gt(0.5)).evaluate_config(
        {"m": "a", "t": None}, names
    ) is False
    assert (m.equals("a") | t.gt(0.5)).evaluate_config(
        {"m": "a", "t": None}, names
    ) is True
