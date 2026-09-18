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
@pytest.mark.asyncio
async def test_issue_1771_non_detailed_aggregate_unwraps_strict_tuple_outputs() -> None:
    """Non-detailed lane must unwrap a strict (output, metrics) tuple before
    comparing accuracy, the same way the detailed per-example path already
    does -- otherwise a mix of plain and tuple-shaped CORRECT outputs
    understates the aggregate (Traigent#1771)."""
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=False)
    dataset = Dataset(
        [
            EvaluationExample({"q": "a"}, "Paris"),
            EvaluationExample({"q": "b"}, "Rome"),
        ],
        name="issue_1771_mixed_shapes",
    )

    def mixed_shape_outputs(input_data: dict[str, str]) -> object:
        if input_data["q"] == "a":
            return "Paris"
        return ("Rome", {"m": 1.0})

    result = await evaluator.evaluate(mixed_shape_outputs, {}, dataset)

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
def test_issue_1771_compute_accuracy_aggregated_unwraps_tuple_and_dict_directly() -> (
    None
):
    """Direct unit coverage for the aggregate-accuracy comparator itself."""
    local = LocalEvaluator(metrics=["accuracy"])
    dataset = Dataset(
        [
            EvaluationExample({"q": "a"}, "Paris"),
            EvaluationExample({"q": "b"}, "Rome"),
            EvaluationExample({"q": "c"}, "Berlin"),
        ]
    )
    outputs = ["Paris", ("Rome", {"m": 1.0}), {"text": "Berlin"}]

    accuracy, total = local._compute_accuracy_aggregated(outputs, dataset)

    assert accuracy == pytest.approx(1.0)
    assert total == 3


def test_issue_1771_registry_compute_accuracy_unwraps_tuple_outputs() -> None:
    """The registry ``_compute_accuracy`` (used by any non-Local caller of
    ``compute_metrics``) must unwrap the strict tuple the same way."""
    base = _DummyBaseEvaluator()
    outputs = ["Paris", ("Rome", {"m": 1.0})]
    expected = ["Paris", "Rome"]
    errors = [None, None]

    assert base._compute_accuracy(outputs, expected, errors) == pytest.approx(1.0)


def test_issue_1771_progress_accuracy_metrics_unwraps_tuple_and_dict_outputs() -> None:
    """Live progress accuracy must not stream a wrong 0.0 for a correct
    tuple- or dict-shaped output (Traigent#1771)."""
    base = _DummyBaseEvaluator()

    tuple_metrics = base._build_progress_accuracy_metrics(
        output=("Rome", {"m": 1.0}),
        expected_output="Rome",
        error=None,
        example_id="ex-0",
    )
    dict_metrics = base._build_progress_accuracy_metrics(
        output={"text": "Berlin"},
        expected_output="Berlin",
        error=None,
        example_id="ex-1",
    )

    assert tuple_metrics["accuracy"] == pytest.approx(1.0)
    assert dict_metrics["accuracy"] == pytest.approx(1.0)


class TestUnwrappingIsTriedSecond:
    """The unwrap must never turn a MATCH into a mismatch (Traigent#1771).

    The first version of this fix applied ``_unpack_user_metrics``
    unconditionally, on a docstring claim that re-unpacking an already-unpacked
    value is a no-op. It is not. A user output that is ITSELF a ``(value,
    dict)`` 2-tuple matches the unpack contract, so a second unpack splits it
    again -- and the detailed path unpacks once already, at
    ``_evaluate_single_detailed``. Measured end to end, with the agent
    returning ``(("Rome", {"x": 1.0}), {"m": 1.0})``:

        detailed=False    0.0 on develop  ->  1.0   (the fix)
        detailed=True     1.0 on develop  ->  0.0   (a NEW failure)

    ``local.py`` records exactly this hazard at its own carrier check; the new
    call sites walked into it. The comparison now runs DIRECT first and unwraps
    only on failure, so the transformation can add a match and never remove
    one.
    """

    def test_a_nested_tuple_output_survives_on_both_lanes(self) -> None:
        import asyncio

        nested = ("Rome", {"x": 1.0})

        def agent(**_config):
            return (nested, {"m": 1.0})

        dataset = Dataset([EvaluationExample({"q": "capital"}, nested)])

        for detailed in (False, True):
            evaluator = LocalEvaluator(metrics=["accuracy"], detailed=detailed)
            result = asyncio.run(evaluator.evaluate(agent, {}, dataset))
            assert result.metrics.get("accuracy") == pytest.approx(1.0), (
                f"detailed={detailed}: a correct nested-tuple answer scored "
                f"{result.metrics.get('accuracy')}"
            )

    def test_a_plain_dict_output_is_not_flattened_to_none(self) -> None:
        """``{"a": 1}.get("text")`` is ``None``.

        The dict branch called ``.get("text")`` on every dict, so a structured
        output that was not a ``{"text": ...}`` wrapper became ``None`` before
        the comparison -- a correct structured answer scored wrong, the same
        failure the PR set out to fix, introduced by the fix.
        """
        base = _DummyBaseEvaluator()
        metrics = base._build_progress_accuracy_metrics(
            output={"a": 1},
            expected_output={"a": 1},
            error=None,
            example_id="ex-dict",
        )
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_a_real_mismatch_is_still_a_mismatch(self) -> None:
        """Control: retrying after an unwrap must not invent matches."""
        base = _DummyBaseEvaluator()
        assert base._compute_accuracy(
            [("Paris", {"m": 1.0})], ["Rome"], [None]
        ) == pytest.approx(0.0)


class TestTheSweepReachesEveryComparator:
    """The title claims "every accuracy comparison site". Three were missed.

    A previous review dismissed ``metrics.py`` as dead code -- on a grep for
    ``MetricsCalculator``, which is a DIFFERENT class in ``metrics_tracker.py``.
    The class here is ``MetricsComputer``, and this very file constructs it.
    """

    def test_metrics_computer_scores_a_tuple_output_correctly(self) -> None:
        result = MetricsComputer(metrics=["accuracy"]).compute_metrics(
            [
                InvocationResult(result=("Rome", {"m": 1.0}), is_successful=True),
                InvocationResult(result="Rome", is_successful=True),
            ],
            ["Rome", "Rome"],
        )
        assert result.metrics["accuracy"] == pytest.approx(1.0)

    @pytest.mark.parametrize("actual", [{"text": "Rome"}, ("Rome", {"m": 1.0}), "Rome"])
    def test_outcome_signals_verified_match_unwraps(self, actual) -> None:
        """Consumed by ``core/trial_result_factory.py`` -- a live path."""
        from traigent.utils.outcome_signals import verified_match

        assert verified_match(actual, "Rome") == pytest.approx(1.0)

    def test_outcome_signals_still_reports_a_real_miss(self) -> None:
        from traigent.utils.outcome_signals import verified_match

        assert verified_match({"text": "Paris"}, "Rome") == pytest.approx(0.0)

    def test_execution_adapter_exact_match_unwraps(self) -> None:
        """Constructed at ``traigent_client.py:311,551``."""
        from traigent.evaluators.base import _accuracy_matches_after_unwrap

        assert _accuracy_matches_after_unwrap(("Rome", {"m": 1.0}), "Rome") is True
        assert _accuracy_matches_after_unwrap({"text": "Rome"}, "Rome") is True
        assert _accuracy_matches_after_unwrap("Paris", "Rome") is False


class TestAMappingExpectedValueIsComparedAsAMapping:
    """Found by composing this PR with #1772, not by reviewing either alone.

    #1772 makes ``_accuracy_values_match`` coerce a JSON STRING to the
    structured value it encodes. This PR makes a failed comparison retry
    against an unwrapped output. Composed on a tree carrying both, a structured
    answer that happens to carry a ``text`` field had its real data discarded
    and its ``text`` re-parsed:

        {"text": '{"id":"7"}', "id": "007"}   vs   {"id": "7"}   ->  True

    The actual ``id`` is ``"007"``. It was thrown away, the ``text`` string was
    parsed into ``{"id": "7"}``, and a wrong answer scored as correct. Measured
    on a tree with both branches merged; NEITHER produces it alone, so no
    per-PR test run could have seen it.

    The rule: a mapping expected value is compared AS a mapping. Pulling one
    field out of the actual and comparing that against a whole structure is
    never the right question, whatever the field is called.

    The obvious alternative -- "a wrapper is a mapping whose only key is
    ``text``" -- was tried first and is wrong: the real SDK response wrapper is
    ``{"text": ..., "raw_response": ...}`` and
    ``tests/unit/evaluators/test_litellm_integration.py`` and
    ``test_tuple_metrics_channel.py`` both pin it. Keying on the EXPECTED shape
    keeps those working and still closes the hole.
    """

    def test_a_mapping_expected_value_does_not_unwrap_a_text_field(self) -> None:
        from traigent.evaluators.base import _accuracy_matches_after_unwrap

        # The `text` field holds the structure directly, so this fails on THIS
        # branch alone -- no JSON coercion needed. The composed case from #1772
        # is the same defect reached through a parsed string.
        assert (
            _accuracy_matches_after_unwrap(
                {"text": {"id": "7"}, "id": "007"}, {"id": "7"}
            )
            is False
        ), (
            "the actual id is 007, not 7; pulling the `text` field out and "
            "comparing THAT against the whole expected mapping awards a match "
            "for an answer that is wrong in the field the caller asked about"
        )

    def test_a_string_expected_value_still_unwraps_a_multi_key_wrapper(self) -> None:
        """Control: the SDK's own response shape must keep working.

        ``{"text": ..., "raw_response": ...}`` against a string expected value
        is the real wrapper this feature exists for. A fix that required
        ``text`` to be the only key would pass the test above and break it.
        """
        from traigent.evaluators.base import _accuracy_matches_after_unwrap

        assert (
            _accuracy_matches_after_unwrap({"text": "YES", "other": "ignored"}, "YES")
            is True
        )
        assert _accuracy_matches_after_unwrap({"text": "Rome"}, "Rome") is True

    def test_two_mappings_that_genuinely_agree_still_match(self) -> None:
        """Control: the rule must not make dict-vs-dict comparison impossible."""
        from traigent.evaluators.base import _accuracy_matches_after_unwrap

        assert _accuracy_matches_after_unwrap({"id": "7"}, {"id": "7"}) is True
