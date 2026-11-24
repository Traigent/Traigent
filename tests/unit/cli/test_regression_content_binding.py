"""Arming is bound to the content it was measured on (``traigent regression``)."""

from __future__ import annotations

import json
from pathlib import Path
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
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.regression_set import (
    RunEvidence,
    content_digest,
    merge_failures,
    read_regression_rows,
)

PASSING = make_row("P", {"q": "pass"})


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text(MODULE_SOURCE, encoding="utf-8")
    return tmp_path


def _edit_content(ws: Path, row_id: str, new_input: dict[str, Any]) -> None:
    """Edit a row's input and recompute its digest, as the diagnostic advises."""
    path = ws / "reg" / "regression.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        row = json.loads(line)
        if row["metadata"]["example_id"] == row_id:
            row["input"] = new_input
            row["metadata"]["content_digest"] = content_digest(
                row["input"], row["output"]
            )
            lines[index] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_observations_record_the_content_digest_they_measured(ws: Path) -> None:
    write_rows(ws, [PASSING])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    row = read_rows(ws)["P"]
    observation = row["metadata"]["regression"]["observations"][0]
    assert observation["content_digest"] == row["metadata"]["content_digest"]


def test_editing_an_armed_row_unarms_it_and_check_says_so(ws: Path) -> None:
    write_rows(ws, [PASSING])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    assert statuses(ws) == {"P": "armed"}
    # The new content would FAIL; it has never passed, so it must not gate and
    # the user must be told that a previously gated row no longer gates.
    _edit_content(ws, "P", {"q": "fail"})
    result = invoke(ws, "check")
    assert result.exit_code == 0, result.output
    assert "not gated (content changed since it was confirmed): 1 row(s): P" in (
        result.output
    )
    assert "0 armed rows" in result.output
    as_json = json.loads(invoke(ws, "check", "--json").stdout)
    assert as_json["armed"] == 0
    assert as_json["not_gated"]["content_changed"] == ["P"]
    assert "P" not in as_json["not_gated"]["unconfirmed"]


def test_reconfirming_changed_content_keeps_history_and_arms_only_if_it_passes(
    ws: Path,
) -> None:
    write_rows(ws, [PASSING])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    _edit_content(ws, "P", {"q": "fail"})
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    regression = read_rows(ws)["P"]["metadata"]["regression"]
    assert regression["status"] == "known_failure"
    digests = [o["content_digest"] for o in regression["observations"]]
    assert len(digests) == 2 and digests[0] != digests[1]
    assert regression["armed_at"] is None
    # Back to the earlier content: observations are matched by digest alone,
    # so the all_pass measured on that very content counts again.
    _edit_content(ws, "P", {"q": "pass"})
    payload = json.loads(invoke(ws, "check", "--json").stdout)
    assert payload["armed"] == 1 and payload["verdict"] == "pass"


def test_a_hand_edit_that_recomputes_the_digest_cannot_keep_the_arming(
    ws: Path,
) -> None:
    write_rows(ws, [PASSING, make_row("Q", {"q": "pass2"})])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    _edit_content(ws, "Q", {"q": "fail-now"})
    result = invoke(ws, "check", "--json")
    payload = json.loads(result.stdout)
    assert payload["armed"] == 1  # only P still gates
    assert payload["not_gated"]["content_changed"] == ["Q"]
    assert result.exit_code == 0


def test_replace_in_add_resets_the_row_to_unconfirmed(ws: Path) -> None:
    write_rows(ws, [PASSING])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    assert statuses(ws) == {"P": "armed"}
    incoming = Dataset(
        examples=[
            EvaluationExample(
                input_data={"q": "fail"},
                expected_output="ok",
                metadata={"example_id": "P"},
            )
        ]
    )
    evidence = RunEvidence("run-2", "run-2", "2026-02-02", "best_config", 0.5, None, {})
    merge_failures(ws / "reg", incoming, evidence=evidence, on_id_conflict="replace")
    rows = read_regression_rows(ws / "reg" / "regression.jsonl")
    regression = rows[0]["metadata"]["regression"]
    assert "observations" not in regression and regression["label"] == "seen_once"
    result = invoke(ws, "check")
    assert result.exit_code == 0 and "0 armed rows" in result.output


def test_a_version_two_observation_without_a_content_digest_is_refused(
    ws: Path,
) -> None:
    write_rows(ws, [PASSING])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    path = ws / "reg" / "regression.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    del row["metadata"]["regression"]["observations"][0]["content_digest"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    before = path.read_bytes()
    for verb in ("check", "confirm"):
        result = invoke(ws, verb)
        assert result.exit_code == 3, result.output
        assert "content_digest" in result.output
    assert path.read_bytes() == before


def test_disarm_records_the_content_it_applies_to(ws: Path) -> None:
    write_rows(ws, [PASSING])
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0
    from click.testing import CliRunner

    from traigent.cli.main import cli

    result = CliRunner().invoke(
        cli,
        [
            "regression",
            "confirm",
            "--dir",
            str(ws / "reg"),
            "--disarm",
            "P",
            "--reason",
            "other",
        ],
    )
    assert result.exit_code == 0, result.output
    regression = read_rows(ws)["P"]["metadata"]["regression"]
    assert (
        regression["observations"][-1]["content_digest"]
        == (read_rows(ws)["P"]["metadata"]["content_digest"])
    )
    assert regression["status"] == "known_failure"
