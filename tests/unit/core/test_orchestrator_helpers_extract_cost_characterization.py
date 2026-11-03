"""Characterization tests for ``extract_cost_from_results``.

Pin how examples-attempted and total cost are resolved from the progress
state, per-example metrics, aggregated metrics and the evaluator-reported
example total, including the unmeasured-cost rules.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace as NS

import pytest

from traigent.core.orchestrator_helpers import extract_cost_from_results


def ex(metrics=None, success=True, **kw):
    return NS(metrics=metrics, success=success, **kw)


def res(**kw):
    return NS(**kw)


CASES = {
    "no_attrs": (res(), None),
    "empty_progress_dict": (res(), {}),
    "progress_only": (res(), {"evaluated": 4, "total_cost": 0.5}),
    "progress_partial": (res(), {"evaluated": 4}),
    "examples_sum_cost_key": (
        res(example_results=[ex({"cost": 0.1}), ex({"cost": 0.2})]),
        None,
    ),
    "key_priority": (
        res(
            example_results=[ex({"total_cost": 1.0, "cost": 9.0, "example_cost": 8.0})]
        ),
        None,
    ),
    "key_none_falls_to_next": (
        res(
            example_results=[
                ex({"total_cost": None, "cost": None, "example_cost": 0.3})
            ]
        ),
        None,
    ),
    "zero_cost_counts": (res(example_results=[ex({"cost": 0})]), None),
    "string_cost_parsed": (res(example_results=[ex({"cost": "0.25"})]), None),
    "unparsable_keeps_progress": (
        res(example_results=[ex({"cost": "abc"})]),
        {"evaluated": 1, "total_cost": 0.9},
    ),
    "no_samples_keeps_progress": (
        res(example_results=[ex({"x": 1})]),
        {"evaluated": 7, "total_cost": 0.9},
    ),
    "progress_examples_kept": (
        res(example_results=[ex({"cost": 0.1}), ex({"cost": 0.1})]),
        {"evaluated": 5, "total_cost": None},
    ),
    "examples_override_progress_cost": (
        res(example_results=[ex({"cost": 0.1})]),
        {"evaluated": 5, "total_cost": 9.9},
    ),
    "none_metrics": (res(example_results=[ex(None)]), None),
    "success_row_missing_cost_unknown": (
        res(example_results=[ex({"cost": 0.1}), ex({})]),
        {"total_cost": 3.0},
    ),
    "unknown_then_agg_fallback": (
        res(
            example_results=[ex({"cost": 0.1}), ex({})],
            aggregated_metrics={"total_cost": 0.77},
        ),
        None,
    ),
    "failed_row_no_telemetry_unknown": (
        res(example_results=[ex({"cost": 0.1}), ex({}, success=False)]),
        None,
    ),
    "failed_row_zero_attempts": (
        res(
            example_results=[
                ex({"cost": 0.1}),
                ex({"provider_attempts": 0}, success=False),
            ]
        ),
        None,
    ),
    "failed_row_explicit_cost": (
        res(example_results=[ex({"cost": 0.1}), ex({"cost": 0}, success=False)]),
        None,
    ),
    "row_flag_attr": (
        res(example_results=[ex({"cost": 0.1}, cost_unmeasured=True)]),
        None,
    ),
    "row_flag_metric": (
        res(example_results=[ex({"cost": 0.1, "cost_unmeasured": 1.0})]),
        None,
    ),
    "agg_flag_overrides_all": (
        res(
            example_results=[ex({"cost": 0.1})],
            aggregated_metrics={"cost_unmeasured": 1.0, "total_cost": 5},
        ),
        {"evaluated": 1, "total_cost": 2},
    ),
    "agg_total_cost": (res(aggregated_metrics={"total_cost": 0.4, "cost": 0.9}), None),
    "agg_cost_fallback": (res(aggregated_metrics={"cost": 0.9}), None),
    "agg_none": (res(aggregated_metrics=None), None),
    "agg_zero_total_cost_not_replaced": (
        res(aggregated_metrics={"total_cost": 0, "cost": 3}),
        None,
    ),
    "total_examples_sets": (res(total_examples=10), None),
    "total_examples_caps": (res(total_examples=3), {"evaluated": 8}),
    "total_examples_noop_when_smaller_progress": (
        res(total_examples=10),
        {"evaluated": 8},
    ),
    "total_examples_zero_ignored": (res(total_examples=0), {"evaluated": 8}),
    "total_examples_float_ignored": (res(total_examples=3.0), {"evaluated": 8}),
    "empty_example_list_ignored": (
        res(example_results=[], aggregated_metrics={"cost": 1.5}),
        {"evaluated": 2},
    ),
    "len_used_when_evaluated_zero": (
        res(example_results=[ex({"cost": 1}), ex({"cost": 1})]),
        {"evaluated": 0},
    ),
}
EXPECTED = {
    "no_attrs": (None, None),
    "empty_progress_dict": (None, None),
    "progress_only": (4, 0.5),
    "progress_partial": (4, None),
    "examples_sum_cost_key": (2, 0.3),
    "key_priority": (1, 1.0),
    "key_none_falls_to_next": (1, None),
    "zero_cost_counts": (1, 0.0),
    "string_cost_parsed": (1, None),
    "unparsable_keeps_progress": (1, 0.9),
    "no_samples_keeps_progress": (7, 0.9),
    "progress_examples_kept": (5, 0.2),
    "examples_override_progress_cost": (5, 0.1),
    "none_metrics": (1, None),
    "success_row_missing_cost_unknown": (2, None),
    "unknown_then_agg_fallback": (2, 0.77),
    "failed_row_no_telemetry_unknown": (2, None),
    "failed_row_zero_attempts": (2, 0.1),
    "failed_row_explicit_cost": (2, 0.1),
    "row_flag_attr": (1, None),
    "row_flag_metric": (1, None),
    "agg_flag_overrides_all": (1, None),
    "agg_total_cost": (None, 0.4),
    "agg_cost_fallback": (None, 0.9),
    "agg_none": (None, None),
    "agg_zero_total_cost_not_replaced": (None, 0),
    "total_examples_sets": (10, None),
    "total_examples_caps": (3, None),
    "total_examples_noop_when_smaller_progress": (8, None),
    "total_examples_zero_ignored": (8, None),
    "total_examples_float_ignored": (8, None),
    "empty_example_list_ignored": (2, 1.5),
    "len_used_when_evaluated_zero": (0, 2.0),
}


assert CASES.keys() == EXPECTED.keys()


@pytest.mark.parametrize("name", sorted(CASES))
def test_extract_cost_from_results(name):
    eval_result, progress = CASES[name]
    expected_examples, expected_cost = EXPECTED[name]

    examples, cost = extract_cost_from_results(eval_result, progress, "t1")

    assert examples == expected_examples
    if expected_cost is None:
        assert cost is None
    else:
        assert cost == pytest.approx(expected_cost)


def test_example_results_are_logged_and_unparsable_costs_noted(caplog):
    result = res(example_results=[ex({"cost": "abc"}), ex({"cost": 0.5})])
    with caplog.at_level(logging.DEBUG):
        extract_cost_from_results(result, None, "trial-9")
    messages = [r.getMessage() for r in caplog.records]
    assert "Trial trial-9: preserved 2 example results" in messages
    assert "Unable to parse example cost abc for trial trial-9" in messages
