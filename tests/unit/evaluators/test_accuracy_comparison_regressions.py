"""Regression tests for evaluator exact-match comparison semantics."""

from __future__ import annotations

import pytest

from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    _accuracy_values_match,
)
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics import MetricsComputer
from traigent.invokers.base import InvocationResult


class _DummyBaseEvaluator(BaseEvaluator):
    async def evaluate(self, func, config, dataset, **kwargs):  # noqa: D401, ANN001
        raise NotImplementedError


def test_exact_match_coerces_string_output_for_typed_expected(caplog) -> None:
    """String outputs matching typed scalar expected values should score correctly."""
    caplog.set_level("WARNING", logger="traigent.evaluators.base")
    base = _DummyBaseEvaluator()
    local = LocalEvaluator(metrics=["accuracy"])
    dataset = Dataset(
        [
            EvaluationExample({"q": "int"}, 42),
            EvaluationExample({"q": "bool"}, True),
            EvaluationExample({"q": "float"}, 3.5),
        ]
    )
    outputs = ["42", "true", "3.5"]
    expected = [example.expected_output for example in dataset.examples]
    errors = [None, None, None]

    assert base._compute_accuracy(outputs, expected, errors) == pytest.approx(1.0)
    assert local._calculate_example_accuracy("42", 42) == pytest.approx(1.0)
    assert local._compute_accuracy_aggregated(outputs, dataset)[0] == pytest.approx(1.0)
    assert local._compute_real_accuracy("true", True) == pytest.approx(1.0)

    metrics_result = MetricsComputer(metrics=["accuracy"]).compute_metrics(
        [InvocationResult(result=value, is_successful=True) for value in outputs],
        expected,
    )
    assert metrics_result.metrics["accuracy"] == pytest.approx(1.0)
    assert "Coercing string output" in caplog.text


def test_numeric_accuracy_uses_float_tolerance_across_paths() -> None:
    """Float representation noise should not make numeric accuracy miss."""
    actual = 0.1 + 0.2
    expected = 0.3
    base = _DummyBaseEvaluator()
    local = LocalEvaluator(metrics=["accuracy"])
    dataset = Dataset([EvaluationExample({"q": "float"}, expected)])

    assert base._compute_accuracy([actual], [expected], [None]) == pytest.approx(1.0)
    assert local._calculate_example_accuracy(actual, expected) == pytest.approx(1.0)
    assert local._compute_accuracy_aggregated([actual], dataset)[0] == pytest.approx(
        1.0
    )
    assert local._compute_real_accuracy(actual, expected) == pytest.approx(1.0)

    metrics_result = MetricsComputer(metrics=["accuracy"]).compute_metrics(
        [InvocationResult(result=actual, is_successful=True)],
        [expected],
    )
    assert metrics_result.metrics["accuracy"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_issue_1464_numeric_accuracy_tolerance_keeps_real_mismatches() -> None:
    """Numeric exact-match accuracy tolerates float noise but not real mismatches."""
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    dataset = Dataset(
        [
            EvaluationExample({"case": "float_noise"}, 0.3),
            EvaluationExample({"case": "different"}, 2.0),
        ],
        name="numeric_tolerance_regression",
    )

    def numeric_outputs(input_data: dict[str, str]) -> float:
        if input_data["case"] == "float_noise":
            return 0.1 + 0.2
        return 1.0

    result = await evaluator.evaluate(numeric_outputs, {}, dataset)

    assert result.metrics["accuracy"] == pytest.approx(0.5)
    assert result.example_results[0].metrics["accuracy"] == pytest.approx(1.0)
    assert result.example_results[1].metrics["accuracy"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_issue_1463_string_outputs_match_typed_expected_values(caplog) -> None:
    """Exact-match accuracy compares string model outputs to typed expected values."""
    caplog.set_level("WARNING", logger="traigent.evaluators.base")
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    dataset = Dataset(
        [
            EvaluationExample({"case": "int"}, 42),
            EvaluationExample({"case": "float"}, 3.0),
            EvaluationExample({"case": "mismatch"}, 42),
        ],
        name="typed_expected_regression",
    )

    def string_outputs(input_data: dict[str, str]) -> str:
        outputs = {
            "int": "42",
            "float": "3.0",
            "mismatch": "43",
        }
        return outputs[input_data["case"]]

    result = await evaluator.evaluate(string_outputs, {}, dataset)

    assert result.metrics["accuracy"] == pytest.approx(2 / 3)
    assert result.example_results[0].metrics["accuracy"] == pytest.approx(1.0)
    assert result.example_results[1].metrics["accuracy"] == pytest.approx(1.0)
    assert result.example_results[2].metrics["accuracy"] == pytest.approx(0.0)
    assert "Coercing string output" in caplog.text


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(["Paris"], ["paris"], id="list_case_insensitive"),
        pytest.param([" a ", "b"], ["a", "B"], id="list_whitespace_and_case"),
        pytest.param({"k": "X"}, {"k": "x"}, id="dict_case_insensitive"),
        pytest.param({"a": ["X"], "b": 1}, {"a": ["x"], "b": 1}, id="nested_dict_list"),
    ],
)
def test_issue_1772_container_elements_get_scalar_normalization(
    actual: object, expected: object
) -> None:
    """List/dict elements must get the same normalization scalars already get."""
    assert _accuracy_values_match(actual, expected) is True


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(["Paris"], ["Rome"], id="list_real_mismatch"),
        pytest.param({"k": "X"}, {"k": "y"}, id="dict_real_mismatch"),
        pytest.param({"k": "X"}, {"other": "X"}, id="dict_key_mismatch"),
        pytest.param(["a"], ["a", "b"], id="list_length_mismatch"),
    ],
)
def test_issue_1772_container_elements_keep_real_mismatches(
    actual: object, expected: object
) -> None:
    """Widening container matching must not paper over a genuine mismatch."""
    assert _accuracy_values_match(actual, expected) is False


def test_issue_1772_json_string_output_matches_structured_expected(caplog) -> None:
    """A JSON-string output must match a dict/list expected value, not always
    score 0.0 (Traigent#1772)."""
    caplog.set_level("WARNING", logger="traigent.evaluators.base")

    assert _accuracy_values_match('{"a": 1, "b": "X"}', {"a": 1, "b": "x"}) is True
    assert _accuracy_values_match('["Paris", "Rome"]', ["paris", "rome"]) is True
    assert _accuracy_values_match("not json", {"a": 1}) is False
    assert "Coercing string output" in caplog.text


def test_issue_1772_numeric_string_string_pairs_are_coerced(caplog) -> None:
    """String-string numeric pairs (the common JSONL habit) must be coerced
    the same way a typed expected value already is (Traigent#1772)."""
    caplog.set_level("WARNING", logger="traigent.evaluators.base")

    assert _accuracy_values_match("1.0", "1") is True
    assert _accuracy_values_match(".5", "0.5") is True
    assert _accuracy_values_match("1", "2") is False
    assert _accuracy_values_match("abc", "abc") is True
    assert "Coercing string output" in caplog.text


@pytest.mark.asyncio
async def test_issue_1772_end_to_end_structured_and_numeric_string_outputs() -> None:
    """Full LocalEvaluator run: structured JSON-string and numeric-string
    outputs should no longer carry a config-independent accuracy ceiling."""
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    dataset = Dataset(
        [
            EvaluationExample({"case": "json_dict"}, {"city": "Paris"}),
            EvaluationExample({"case": "numeric_string"}, "1"),
            EvaluationExample({"case": "list_case"}, ["Rome"]),
        ],
        name="issue_1772_regression",
    )

    def outputs(input_data: dict[str, str]) -> object:
        return {
            "json_dict": '{"city": "paris"}',
            "numeric_string": "1.0",
            "list_case": ["rome"],
        }[input_data["case"]]

    result = await evaluator.evaluate(outputs, {}, dataset)

    assert result.metrics["accuracy"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "actual,expected",
    [
        ("007", "7"),
        ("7", "007"),
        ("0012", "12"),
        ("-007", "-7"),
        ("00", "0"),
    ],
)
def test_issue_1772_zero_padded_identifiers_are_not_coerced(actual, expected) -> None:
    """Numeric coercion must not turn an identifier into a correct answer.

    The string-string coercion above exists for the JSONL habit of storing a
    numeric gold label as a string. It cannot tell that habit apart from a
    fixed-width identifier -- a zip code, an order number, a SKU, a phone
    extension -- and without this guard ``"007"`` scored as a correct answer to
    ``"7"``.

    That is a false positive in ACCURACY, which is the value the optimizer
    argmaxes: a config that returns the wrong identifier would be ranked first,
    confidently. An exact-match evaluator loosened this far is no longer doing
    the job its name promises.
    """
    assert _accuracy_values_match(actual, expected) is False


@pytest.mark.parametrize(
    "actual,expected",
    [
        ("1.0", "1"),
        (".5", "0.5"),
        ("0.5", ".5"),
        (" 42 ", "42"),
        ("1e5", "100000"),
        ("0.10", "0.1"),
    ],
)
def test_issue_1772_ordinary_numeric_formatting_still_coerces(actual, expected) -> None:
    """Control: the guard must not switch the fix off.

    A leading zero before a DECIMAL POINT (``0.5``, ``0.10``) is ordinary
    numeric formatting, not an identifier; only a leading zero before another
    DIGIT is. A guard that rejected ``"0.5"`` would pass the test above and
    undo the PR.
    """
    assert _accuracy_values_match(actual, expected) is True
