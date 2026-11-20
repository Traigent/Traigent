"""Arming fields are a pure function of the observations and the current content."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from tests.unit.cli.regression_run_helpers import (
    MODULE_SOURCE,
    invoke,
    make_row,
    read_rows,
    write_rows,
)
from traigent.utils import regression_confirm
from traigent.utils.regression_set import content_digest, read_regression_rows
from traigent.utils.regression_status import (
    append_observation,
    derive_arming,
    derive_status,
    status_when_last_measured,
)


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text(MODULE_SOURCE, encoding="utf-8")
    return tmp_path


def _edit_content(ws: Path, new_input: dict[str, Any]) -> None:
    path = ws / "reg" / "regression.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["input"] = new_input
    row["metadata"]["content_digest"] = content_digest(row["input"], row["output"])
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_content_restored_after_a_failing_edit_is_confirmed_without_corruption(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_rows(ws, [make_row("P", {"q": "pass"})])
    monkeypatch.setattr(regression_confirm, "today", lambda: "2026-01-01")
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0  # A passes
    first = read_rows(ws)["P"]["metadata"]["regression"]
    assert first["status"] == "armed" and first["armed_at"] == "2026-01-01"

    _edit_content(ws, {"q": "fail"})
    monkeypatch.setattr(regression_confirm, "today", lambda: "2026-01-02")
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0  # B fails
    second = read_rows(ws)["P"]["metadata"]["regression"]
    assert second["status"] == "known_failure" and second["armed_at"] is None

    _edit_content(ws, {"q": "pass"})
    monkeypatch.setattr(regression_confirm, "today", lambda: "2026-01-03")
    again = invoke(ws, "confirm", "--repeats", "2")  # A again
    assert again.exit_code == 0, again.output
    third = read_rows(ws)["P"]["metadata"]["regression"]
    assert third["status"] == "armed"
    assert third["armed_at"] == "2026-01-01"  # A's first all_pass
    assert third["armed_config_digest"] == first["armed_config_digest"]
    read_regression_rows(ws / "reg" / "regression.jsonl")  # the reader accepts it
    check = invoke(ws, "check", "--json")
    assert check.exit_code == 0, check.output
    assert json.loads(check.stdout)["verdict"] == "pass"


def _obs(outcome: str, content: str, day: int, cfg: str = "c") -> dict[str, Any]:
    return {
        "config_digest": None if outcome == "disarmed" else cfg,
        "content_digest": content,
        "date": f"2026-01-{day:02d}",
        "reps": 0 if outcome == "disarmed" else 2,
        "passes": 2 if outcome == "all_pass" else 0,
        "fails": 2 if outcome == "all_fail" else 0,
        "errored": 2 if outcome == "errored" else 0,
        "pass_rate_low": None,
        "pass_rate_high": None,
        "outcome": outcome,
        **({"reason": "other"} if outcome == "disarmed" else {}),
    }


def test_derive_arming_is_first_all_pass_after_the_last_disarm_on_the_content() -> None:
    obs = [
        _obs("all_pass", "A", 1, "c1"),
        _obs("all_fail", "B", 2),
        _obs("all_pass", "A", 3, "c2"),
        _obs("disarmed", "A", 4),
        _obs("all_fail", "A", 5),
        _obs("all_pass", "A", 6, "c3"),
        _obs("all_pass", "A", 7, "c4"),
    ]
    assert derive_arming(obs[:3], "A") == ("2026-01-01", "c1")
    assert derive_arming(obs[:3], "B") == (None, None)
    assert derive_arming(obs[:4], "A") == (None, None)
    assert derive_arming(obs, "A") == ("2026-01-06", "c3")
    assert derive_arming([], "A") == (None, None)


STEPS = (
    "pass",
    "fail",
    "mixed",
    "errored",
    "content",
    "disarm",
)
OUTCOME_OF = {
    "pass": "all_pass",
    "fail": "all_fail",
    "mixed": "mixed",
    "errored": "errored",
}


def test_random_sequences_always_write_rows_the_reader_accepts(tmp_path: Path) -> None:
    rng = random.Random(20261120)
    contents = ["A", "B", "C"]
    path = tmp_path / "regression.jsonl"
    for _ in range(300):
        row = make_row("R", {"q": "x"})
        regression = row["metadata"]["regression"]
        current = "A"
        for day in range(1, rng.randint(2, 14)):
            step = rng.choice(STEPS)
            if step == "content":
                current = rng.choice(contents)
                continue
            if step == "disarm":
                if derive_status(regression.get("observations"), current) != "armed":
                    continue
                observation = _obs("disarmed", current, day)
            else:
                observation = _obs(OUTCOME_OF[step], current, day, f"cfg{day}")
            append_observation(regression, observation, 2)
            observations = regression["observations"]
            assert regression["status"] == derive_status(observations, current)
            assert (
                regression["armed_at"],
                regression["armed_config_digest"],
            ) == derive_arming(observations, current)
            assert status_when_last_measured(regression) == regression["status"]
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            read_regression_rows(path)  # raises when the reader rejects the row
