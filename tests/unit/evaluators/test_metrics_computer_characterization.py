"""Characterization tests pinning ``MetricsComputer.compute_metrics``."""

from __future__ import annotations

import pytest

from traigent.evaluators.metrics import MetricsComputer
from traigent.invokers.base import InvocationResult

ALL = [
    "accuracy",
    "success_rate",
    "error_rate",
    "avg_execution_time",
    "avg_output_length",
]


class _BadLen:
    def __len__(self):
        raise TypeError("no len")


class _Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def ok(output, t=0.0):
    return InvocationResult(result=output, execution_time=t)


def bad(t=0.0):
    return InvocationResult(error="boom", execution_time=t)


# name -> (metrics, results factory, expected outputs)
CASES = {
    "default_metrics": (None, lambda: [ok("a"), ok("b"), bad()], ["a", "x", "c"]),
    "empty_list_uses_default": ([], lambda: [ok("a")], ["a"]),
    "empty_inputs_all": (ALL, lambda: [], []),
    "all_metrics": (
        ALL,
        lambda: [ok("abc", 1.0), ok("zz", 3.0), bad(2.0), ok(None, 0.0)],
        ["abc", "no", "q", "r"],
    ),
    "accuracy_no_pairs": (["accuracy"], lambda: [ok("a")], [None]),
    "accuracy_all_failed": (["accuracy"], lambda: [bad(), bad()], ["a", "b"]),
    "accuracy_correct_half": (
        ["accuracy"],
        lambda: [ok(1), ok(2), ok(3), ok(4)],
        [1, 0, 3, 0],
    ),
    "error_rate_alone": (["error_rate"], lambda: [ok(1), bad()], [1, 1]),
    "error_rate_with_success": (
        ["success_rate", "error_rate"],
        lambda: [ok(1), bad(), bad()],
        [1, 1, 1],
    ),
    "success_rate_empty": (["success_rate"], lambda: [], []),
    "exec_time_none_positive": (
        ["avg_execution_time"],
        lambda: [ok(1, 0.0), ok(2, 0.0)],
        [1, 2],
    ),
    "exec_time_mixed": (
        ["avg_execution_time"],
        lambda: [ok(1, 2.0), ok(2, 4.0), bad(10.0), ok(3, -1.0)],
        [1, 2, 3, 4],
    ),
    "output_length_no_valid": (
        ["avg_output_length"],
        lambda: [bad(), ok(None)],
        [1, 2],
    ),
    "output_length_mixed_types": (
        ["avg_output_length"],
        lambda: [ok("ab"), ok([1, 2, 3]), ok({"a": 1}), ok(b"xyzw"), ok(_Sized(0))],
        [1] * 5,
    ),
    "output_length_unsized_only": (
        ["avg_output_length"],
        lambda: [ok(5), ok(2.5), ok(True)],
        [1] * 3,
    ),
    "output_length_bad_len_raises": (
        ["avg_output_length"],
        lambda: [ok(_BadLen())],
        [1],
    ),
    "unknown_metric_ignored": (["nope"], lambda: [ok(1)], [1]),
    "expected_none_excluded_from_pairs": (
        ["accuracy", "success_rate"],
        lambda: [ok(1), ok(2)],
        [None, 2],
    ),
}


def _snapshot(name):
    metrics, factory, expected = CASES[name]
    computer = MetricsComputer(list(metrics) if metrics is not None else None)
    try:
        res = computer.compute_metrics(factory(), list(expected))
    except Exception as exc:  # noqa: BLE001 - the exception type is the pin
        return {"raised": f"{type(exc).__name__}: {exc}"}
    meta = dict(res.metadata)
    for key in ("evaluation_start_time", "evaluation_end_time"):
        meta.pop(key)
    return {
        "metrics": repr(res.metrics),
        "total": res.total_invocations,
        "ok": res.successful_invocations,
        "meta": repr(meta),
    }


GOLDEN: dict = {
    "default_metrics": {
        "metrics": "{'success_rate': 0.6666666666666666, 'accuracy': 0.5}",
        "total": 3,
        "ok": 2,
        "meta": "{'metrics_requested': ['accuracy', 'success_rate'], "
        "'successful_pairs': 2, 'failed_invocations': 1, 'metric_errors': "
        "[]}",
    },
    "empty_list_uses_default": {
        "metrics": "{'success_rate': 1.0, 'accuracy': 1.0}",
        "total": 1,
        "ok": 1,
        "meta": "{'metrics_requested': ['accuracy', 'success_rate'], "
        "'successful_pairs': 1, 'failed_invocations': 0, "
        "'metric_errors': []}",
    },
    "empty_inputs_all": {
        "metrics": "{'success_rate': 0.0, 'error_rate': 1.0, 'accuracy': 0.0, "
        "'avg_execution_time': 0.0, 'avg_output_length': 0.0}",
        "total": 0,
        "ok": 0,
        "meta": "{'metrics_requested': ['accuracy', 'success_rate', 'error_rate', "
        "'avg_execution_time', 'avg_output_length'], 'successful_pairs': 0, "
        "'failed_invocations': 0, 'metric_errors': []}",
    },
    "all_metrics": {
        "metrics": "{'success_rate': 0.75, 'error_rate': 0.25, 'accuracy': "
        "0.3333333333333333, 'avg_execution_time': 2.0, 'avg_output_length': "
        "2.5}",
        "total": 4,
        "ok": 3,
        "meta": "{'metrics_requested': ['accuracy', 'success_rate', 'error_rate', "
        "'avg_execution_time', 'avg_output_length'], 'successful_pairs': 3, "
        "'failed_invocations': 1, 'metric_errors': []}",
    },
    "accuracy_no_pairs": {
        "metrics": "{'accuracy': 0.0}",
        "total": 1,
        "ok": 1,
        "meta": "{'metrics_requested': ['accuracy'], 'successful_pairs': 0, "
        "'failed_invocations': 0, 'metric_errors': []}",
    },
    "accuracy_all_failed": {
        "metrics": "{'accuracy': 0.0}",
        "total": 2,
        "ok": 0,
        "meta": "{'metrics_requested': ['accuracy'], 'successful_pairs': 0, "
        "'failed_invocations': 2, 'metric_errors': []}",
    },
    "accuracy_correct_half": {
        "metrics": "{'accuracy': 0.5}",
        "total": 4,
        "ok": 4,
        "meta": "{'metrics_requested': ['accuracy'], 'successful_pairs': 4, "
        "'failed_invocations': 0, 'metric_errors': []}",
    },
    "error_rate_alone": {
        "metrics": "{'error_rate': 1.0}",
        "total": 2,
        "ok": 1,
        "meta": "{'metrics_requested': ['error_rate'], 'successful_pairs': 1, "
        "'failed_invocations': 1, 'metric_errors': []}",
    },
    "error_rate_with_success": {
        "metrics": "{'success_rate': 0.3333333333333333, 'error_rate': "
        "0.6666666666666667}",
        "total": 3,
        "ok": 1,
        "meta": "{'metrics_requested': ['success_rate', 'error_rate'], "
        "'successful_pairs': 1, 'failed_invocations': 2, "
        "'metric_errors': []}",
    },
    "success_rate_empty": {
        "metrics": "{'success_rate': 0.0}",
        "total": 0,
        "ok": 0,
        "meta": "{'metrics_requested': ['success_rate'], 'successful_pairs': 0, "
        "'failed_invocations': 0, 'metric_errors': []}",
    },
    "exec_time_none_positive": {
        "metrics": "{'avg_execution_time': 0.0}",
        "total": 2,
        "ok": 2,
        "meta": "{'metrics_requested': ['avg_execution_time'], "
        "'successful_pairs': 2, 'failed_invocations': 0, "
        "'metric_errors': []}",
    },
    "exec_time_mixed": {
        "metrics": "{'avg_execution_time': 5.333333333333333}",
        "total": 4,
        "ok": 3,
        "meta": "{'metrics_requested': ['avg_execution_time'], 'successful_pairs': 3, "
        "'failed_invocations': 1, 'metric_errors': []}",
    },
    "output_length_no_valid": {
        "metrics": "{'avg_output_length': 0.0}",
        "total": 2,
        "ok": 1,
        "meta": "{'metrics_requested': ['avg_output_length'], "
        "'successful_pairs': 1, 'failed_invocations': 1, "
        "'metric_errors': []}",
    },
    "output_length_mixed_types": {
        "metrics": "{'avg_output_length': 2.0}",
        "total": 5,
        "ok": 5,
        "meta": "{'metrics_requested': ['avg_output_length'], "
        "'successful_pairs': 5, 'failed_invocations': 0, "
        "'metric_errors': []}",
    },
    "output_length_unsized_only": {
        "metrics": "{'avg_output_length': 0.0}",
        "total": 3,
        "ok": 3,
        "meta": "{'metrics_requested': ['avg_output_length'], "
        "'successful_pairs': 3, 'failed_invocations': 0, "
        "'metric_errors': []}",
    },
    "output_length_bad_len_raises": {"raised": "TypeError: no len"},
    "unknown_metric_ignored": {
        "metrics": "{}",
        "total": 1,
        "ok": 1,
        "meta": "{'metrics_requested': ['nope'], 'successful_pairs': 1, "
        "'failed_invocations': 0, 'metric_errors': []}",
    },
    "expected_none_excluded_from_pairs": {
        "metrics": "{'success_rate': 1.0, 'accuracy': 1.0}",
        "total": 2,
        "ok": 2,
        "meta": "{'metrics_requested': ['accuracy', "
        "'success_rate'], 'successful_pairs': 1, "
        "'failed_invocations': 0, 'metric_errors': []}",
    },
}


@pytest.mark.parametrize("name", list(CASES))
def test_compute_metrics(name):
    assert _snapshot(name) == GOLDEN[name]


def test_mismatched_lengths_raise_and_reset_errors():
    computer = MetricsComputer(["accuracy"])
    computer.metric_errors = [{"stale": True}]
    with pytest.raises(ValueError, match=r"Number of invocation results \(1\)"):
        computer.compute_metrics([ok(1)], [1, 2])
    assert computer.metric_errors == []


def test_custom_metrics_merged_after_builtins():
    computer = MetricsComputer(["success_rate"])
    computer._custom_metrics = {
        "success_rate": lambda o, e: 0.25,
        "mine": lambda o, e: 2,
    }
    res = computer.compute_metrics([ok(1)], [1])
    assert res.metrics == {"success_rate": 0.25, "mine": 2.0}
