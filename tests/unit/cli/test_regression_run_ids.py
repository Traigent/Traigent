"""Run results are tied to rows by ids this command assigns, never by ids an
evaluator may derive from the customer's input (``traigent regression``)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.cli.regression_run_helpers import (
    MODULE_SOURCE,
    invoke,
    make_row,
    read_rows,
    statuses,
    write_rows,
)
from traigent.api.types import ExampleResult
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.hybrid_api import HybridAPIEvaluator
from traigent.utils import regression_confirm


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text(MODULE_SOURCE, encoding="utf-8")
    return tmp_path


# Row A fails, row B passes; each row's input names the OTHER row's id.
ROW_A = make_row("A", {"q": "fail", "example_id": "B"})
ROW_B = make_row("B", {"q": "pass", "example_id": "A"})


def _trial(ids_and_success: list[tuple[Any, bool]]) -> Any:
    return SimpleNamespace(
        metadata={
            "example_results": [
                ExampleResult(
                    example_id=example_id,
                    input_data={"q": "x"},
                    expected_output="ok",
                    actual_output="x",
                    metrics={"accuracy": 1.0 if ok else 0.0},
                    execution_time=0.1,
                    success=True,
                    error_message=None,
                ).to_dict()
                for example_id, ok in ids_and_success
            ]
        }
    )


def _fake_compare(
    id_rule: str, calls: list[Any] | None = None, drop: set[int] = frozenset()
):
    """A ``compare`` whose result ids follow ``id_rule`` for the dataset it gets."""

    def compare(self, configs, *, eval_dataset, repeats=1, budget=None):
        examples = eval_dataset.examples
        if calls is not None:
            calls.append(eval_dataset)
        if id_rule == "hybrid":
            evaluator = object.__new__(HybridAPIEvaluator)
            ids: list[Any] = evaluator._resolve_example_ids(examples)
        elif id_rule == "duplicate":
            ids = [examples[0].metadata["example_id"]] * len(examples)
        else:
            ids = [e.metadata["example_id"] for e in examples]
        trials = [
            _trial(
                [
                    (ids[i], e.input_data["q"].startswith("pass"))
                    for i, e in enumerate(examples)
                    if i not in drop
                ]
            )
            for _ in range(repeats)
        ]
        compared = SimpleNamespace(repeat_trials=trials, total_cost=0.0)
        return SimpleNamespace(configs=[compared], stop_reason=None)

    return compare


def test_local_evaluator_keeps_permuted_input_ids_on_their_own_rows(ws: Path) -> None:
    # Guard: through the real local evaluator the dataset's metadata id decides
    # the result id, so inputs naming each other's ids cannot cross rows.
    write_rows(ws, [ROW_A, ROW_B])
    result = invoke(ws, "confirm", "--repeats", "2")
    assert result.exit_code == 0, result.output
    assert statuses(ws) == {"A": "known_failure", "B": "armed"}


def test_hybrid_id_rule_cannot_arm_the_wrong_row(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The hybrid evaluator names results by input.example_id, so A's result
    # comes back as "B" and B's as "A". Matching by those ids armed A.
    monkeypatch.setattr(OptimizedFunction, "compare", _fake_compare("hybrid"))
    write_rows(ws, [ROW_A, ROW_B])
    before = (ws / "reg" / "regression.jsonl").read_bytes()
    result = invoke(ws, "confirm", "--repeats", "2")
    assert result.exit_code == 2, result.output
    assert "assigns its own example ids" in result.output
    assert (ws / "reg" / "regression.jsonl").read_bytes() == before
    assert statuses(ws) == {"A": "unconfirmed", "B": "unconfirmed"}
    # Content never appears in the closed message.
    assert "fail" not in result.output.replace("failure", "")


def test_check_refuses_with_exit_2_when_ids_are_not_assignable(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_rows(ws, [ROW_A, ROW_B])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    monkeypatch.setattr(OptimizedFunction, "compare", _fake_compare("hybrid"))
    result = invoke(ws, "check", "--json")
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "error"
    assert payload["error"] == "ids_not_assignable"


def test_every_dataset_example_gets_an_opaque_run_scoped_id(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(OptimizedFunction, "compare", _fake_compare("echo", calls))
    write_rows(ws, [ROW_A, ROW_B])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    assigned = [e.metadata["example_id"] for e in calls[0].examples]
    assert len(set(assigned)) == 2
    assert not {"A", "B"} & set(assigned)
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    again = [e.metadata["example_id"] for e in calls[1].examples]
    assert not set(assigned) & set(again)  # run-scoped, fresh per run


def test_a_duplicated_result_id_leaves_those_rows_unrecorded(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OptimizedFunction, "compare", _fake_compare("duplicate"))
    write_rows(ws, [ROW_A, ROW_B])
    result = invoke(ws, "confirm", "--repeats", "2")
    # The first row's id came back twice and the second never: neither is
    # recorded, and the closed reasons are printed.
    assert "id_duplicated" in result.output and "id_missing" in result.output
    assert all("observations" not in v["metadata"]["regression"]
               for v in read_rows(ws).values())


def test_a_missing_result_id_leaves_only_that_row_unrecorded(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OptimizedFunction, "compare", _fake_compare("echo", drop={0}))
    write_rows(ws, [ROW_A, ROW_B])
    result = invoke(ws, "confirm", "--repeats", "2")
    assert result.exit_code == 0, result.output
    assert "A (id_missing)" in result.output
    assert statuses(ws) == {"A": "unconfirmed", "B": "armed"}


def test_check_is_inconclusive_for_a_row_whose_id_is_missing_or_duplicated(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_rows(ws, [ROW_B, make_row("C", {"q": "pass2"})])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    for rule, drop, reason in (("echo", {0}, "id_missing"),):
        monkeypatch.setattr(OptimizedFunction, "compare", _fake_compare(rule, drop=drop))
        result = invoke(ws, "check", "--json")
        assert result.exit_code == 2, result.output
        payload = json.loads(result.stdout)
        assert payload["verdict"] == "inconclusive"
        assert [i["reason"] for i in payload["inconclusive"]] == [reason]
    monkeypatch.setattr(OptimizedFunction, "compare", _fake_compare("duplicate"))
    result = invoke(ws, "check", "--json")
    assert result.exit_code == 2
    reasons = {i["reason"] for i in json.loads(result.stdout)["inconclusive"]}
    assert reasons <= {"id_duplicated", "id_missing"} and reasons


def test_resolve_attempt_requires_the_returned_set_to_equal_the_assigned_set() -> None:
    row_ids = {"rc_x_0": "A", "rc_x_1": "B"}
    ok, bad, foreign, seen = regression_confirm.resolve_attempt(
        _trial([("rc_x_0", True), ("rc_x_1", False)]), row_ids
    )
    assert ok == {"A": "pass", "B": "fail"} and not bad and not foreign and seen
    extra, bad, foreign, _ = regression_confirm.resolve_attempt(
        _trial([("rc_x_0", True), ("rc_x_1", True), ("B", True)]), row_ids
    )
    assert foreign and set(bad) == {"A", "B"} and extra == {}
