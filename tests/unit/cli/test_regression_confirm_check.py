"""Tests for ``traigent regression confirm`` and ``check`` (regression set, slice 2).

No paid calls and no network: the decorated function is a stub whose answer per
example and per call is scripted by the configuration's ``mode`` (mock mode is
on for the whole suite; the egress harness would flag any socket use).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from tests.egress_canary.harness import capture_egress
from traigent.api.types import (
    ExampleResult,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.cli.main import cli
from traigent.utils import regression_confirm
from traigent.utils.persistence import PersistenceManager
from traigent.utils.regression_set import (
    load_regression_dataset,
    read_regression_rows,
)

ROWS: list[dict[str, Any]] = [
    {"input": {"q": f"synthetic question {i}"}, "output": f"a{i}"} for i in range(4)
]

# Rows 0..3 below are, under mode "old": 0 fails every call, 1 passes only on
# its first two calls (2 of 5), 2 always passes, 3 raises every call.
AGENT_SOURCE = """
import threading
import time
import traigent

ANSWERS = {f"synthetic question {i}": f"a{i}" for i in range(4)}
_CALLS = {}
_LOCK = threading.Lock()


def _call_index(question):
    with _LOCK:
        index = _CALLS.get(question, 0)
        _CALLS[question] = index + 1
    return index


@traigent.optimize(
    eval_dataset=[{"input": {"q": "synthetic question 2"}, "output": "a2"}],
    objectives=["accuracy"],
    configuration_space={"mode": ["old", "fixed", "allerr", "costed", "slow"]},
    execution_mode="local",
    max_trials=1,
)
def answer(q: str) -> str:
    mode = traigent.get_config().get("mode")
    index = _call_index(q)
    right = ANSWERS[q]
    if mode == "allerr":
        raise RuntimeError("scripted failure")
    if mode == "slow":
        time.sleep(0.4)
        with open("calls.log", "a") as log:
            log.write(f"{time.time()}\\n")
        return right
    if mode == "costed":
        return traigent.with_usage(right, total_cost=0.01)
    if q.endswith("0"):
        return right if mode == "fixed" else "wrong"
    if q.endswith("1"):
        return right if index < 2 else "wrong"
    if q.endswith("3"):
        raise RuntimeError("scripted failure")
    return right
"""


def _example(index: int) -> dict:
    return ExampleResult(
        example_id=f"example_{index}",
        input_data=ROWS[index]["input"],
        expected_output=ROWS[index]["output"],
        actual_output="x",
        metrics={"accuracy": 0.0},
        execution_time=0.1,
        success=True,
        error_message=None,
    ).to_dict()


def _failing_result(indices: list[int], oid: str) -> OptimizationResult:
    trial = TrialResult(
        trial_id="t0",
        config={"model": "a"},
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={
            "example_results": [
                ExampleResult(
                    example_id=f"example_{i}",
                    input_data=ROWS[i]["input"],
                    expected_output=ROWS[i]["output"],
                    actual_output="x",
                    metrics={"accuracy": 0.0 if i in indices else 1.0},
                    execution_time=0.1,
                    success=True,
                    error_message=None,
                ).to_dict()
                for i in range(len(ROWS))
            ]
        },
    )
    return OptimizationResult(
        trials=[trial],
        best_config={"model": "a"},
        best_score=0.5,
        optimization_id=oid,
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="GridSearchOptimizer",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={"function_name": "f"},
    )


def _write_config(ws: Path, mode: str) -> str:
    path = ws / f"config_{mode}.json"
    path.write_text(json.dumps({"mode": mode}), encoding="utf-8")
    return str(path)


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace whose regression file was produced by the real ``add``."""
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent_mod.py").write_text(AGENT_SOURCE, encoding="utf-8")
    (tmp_path / "data.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8"
    )
    PersistenceManager(str(tmp_path / "results")).save_result(
        _failing_result([0, 1, 2, 3], "opt-a"), name="run_a"
    )
    result = CliRunner().invoke(
        cli,
        [
            "regression",
            "add",
            "run_a",
            "--dataset",
            str(tmp_path / "data.jsonl"),
            "--dir",
            str(tmp_path / "reg"),
            "--storage-dir",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 0, result.output
    return tmp_path


def _confirm(ws: Path, mode: str, *extra: str):
    return CliRunner().invoke(
        cli,
        [
            "regression",
            "confirm",
            "--module",
            "agent_mod.py:answer",
            "--config",
            _write_config(ws, mode),
            "--dir",
            str(ws / "reg"),
            *extra,
        ],
    )


def _check(ws: Path, mode: str, *extra: str, config: str | None = None):
    return CliRunner().invoke(
        cli,
        [
            "regression",
            "check",
            "--module",
            "agent_mod.py:answer",
            "--config",
            config or _write_config(ws, mode),
            "--dir",
            str(ws / "reg"),
            *extra,
        ],
    )


def _file(ws: Path) -> Path:
    return ws / "reg" / "regression.jsonl"


def _reg(ws: Path) -> dict[str, dict[str, Any]]:
    return {
        r["metadata"]["example_id"]: r["metadata"]["regression"]
        for r in read_regression_rows(_file(ws))
    }


def _snapshot(ws: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(ws)): p.read_bytes()
        for p in sorted((ws / "reg").rglob("*"))
        if p.is_file()
    }


def _statuses(ws: Path) -> dict[str, str]:
    return {k: v["status"] for k, v in _reg(ws).items()}


# --------------------------------------------------------------------- confirm


def test_confirm_marks_rows_by_what_they_do_and_deletes_nothing(ws: Path) -> None:
    before_ids = [r["metadata"]["example_id"] for r in read_regression_rows(_file(ws))]
    result = _confirm(ws, "old")
    assert result.exit_code == 0, result.output

    assert _statuses(ws) == {
        "example_0": "known_failure",
        "example_1": "intermittent",
        "example_2": "armed",
        "example_3": "unconfirmed",
    }
    assert [
        r["metadata"]["example_id"] for r in read_regression_rows(_file(ws))
    ] == before_ids
    reg = _reg(ws)
    from traigent.utils.hashing import generate_config_hash

    digest = generate_config_hash({"mode": "old"})
    counts = {
        eid: (o["passes"], o["fails"], o["errored"], o["outcome"])
        for eid, r in reg.items()
        for o in r["observations"]
    }
    assert counts == {
        "example_0": (0, 5, 0, "all_fail"),
        "example_1": (2, 3, 0, "mixed"),
        "example_2": (5, 0, 0, "all_pass"),
        "example_3": (0, 0, 5, "errored"),
    }
    for r in reg.values():
        assert r["schema_version"] == 2
        assert [o["config_digest"] for o in r["observations"]] == [digest]
        assert r["observations"][0]["reps"] == 5
    assert reg["example_2"]["armed_config_digest"] == digest
    assert reg["example_2"]["armed_at"] == reg["example_2"]["observations"][0]["date"]
    assert reg["example_0"]["armed_at"] is None
    assert "example question" not in result.output
    assert "synthetic question" not in result.output
    assert "example_2: unconfirmed -> armed" in result.output


def test_a_version_one_file_written_by_add_is_upgraded_in_memory_then_on_write(
    ws: Path,
) -> None:
    rows = read_regression_rows(_file(ws))
    assert {r["metadata"]["regression"]["schema_version"] for r in rows} == {1}
    assert all("observations" not in r["metadata"]["regression"] for r in rows)
    # in memory, a version 1 row reads as unconfirmed
    assert len(load_regression_dataset(ws / "reg", labels=("unconfirmed",))) == 4
    assert len(load_regression_dataset(ws / "reg")) == 4
    assert _check(ws, "old").exit_code == 0  # v1 rows are unconfirmed: nothing armed

    assert _confirm(ws, "old").exit_code == 0
    assert {r["schema_version"] for r in _reg(ws).values()} == {2}


def test_load_regression_dataset_labels_stay_meaningful_after_confirm(ws: Path) -> None:
    assert _confirm(ws, "old").exit_code == 0
    reliable = load_regression_dataset(ws / "reg", labels=("reliable",))
    assert [e.metadata["example_id"] for e in reliable.examples] == ["example_2"]
    armed = load_regression_dataset(ws / "reg", labels=("armed",))
    assert len(armed) == 1
    inter = load_regression_dataset(ws / "reg", labels=("intermittent",))
    assert [e.metadata["example_id"] for e in inter.examples] == ["example_1"]
    assert len(load_regression_dataset(ws / "reg")) == 4  # default keeps all


def test_check_right_after_confirm_passes_and_lists_known_failure_not_gated(
    ws: Path,
) -> None:
    assert _confirm(ws, "old").exit_code == 0
    result = _check(ws, "old")
    assert result.exit_code == 0, result.output
    assert "1 armed row(s): 1 passed, 0 regression(s)" in result.output
    assert "not gated (known_failure): example_0" in result.output
    assert "not gated (intermittent): example_1" in result.output


def test_known_failure_becomes_armed_once_a_confirm_sees_it_pass(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(regression_confirm, "today", lambda: "2026-01-01")
    assert _confirm(ws, "old").exit_code == 0
    assert _reg(ws)["example_0"]["armed_at"] is None

    monkeypatch.setattr(regression_confirm, "today", lambda: "2026-02-02")
    result = _confirm(ws, "fixed")
    assert result.exit_code == 0, result.output
    row = _reg(ws)["example_0"]
    assert row["status"] == "armed"
    assert row["armed_at"] == "2026-02-02"
    assert [o["outcome"] for o in row["observations"]] == ["all_fail", "all_pass"]
    assert "example_0: known_failure -> armed" in result.output

    monkeypatch.setattr(regression_confirm, "today", lambda: "2026-03-03")
    assert _confirm(ws, "fixed").exit_code == 0
    assert _reg(ws)["example_0"]["armed_at"] == "2026-02-02"  # set once
    assert len(_reg(ws)["example_0"]["observations"]) == 3


def test_check_flags_the_regression_under_the_old_config_not_the_fixed_one(
    ws: Path,
) -> None:
    assert _confirm(ws, "old").exit_code == 0
    assert _confirm(ws, "fixed").exit_code == 0
    fixed = _check(ws, "fixed")
    assert fixed.exit_code == 0, fixed.output
    old = _check(ws, "old")
    assert old.exit_code == 1, old.output
    assert "REGRESSION example_0: attempts=1 passes=0 fails=1 errored=0" in old.output
    assert "REGRESSION example_2" not in old.output


def test_an_armed_row_stays_armed_after_a_later_failure_and_is_never_demoted(
    ws: Path,
) -> None:
    assert _confirm(ws, "old").exit_code == 0
    assert _confirm(ws, "fixed").exit_code == 0
    assert _confirm(ws, "old").exit_code == 0  # example_0 fails again
    row = _reg(ws)["example_0"]
    assert row["status"] == "armed"
    assert row["observations"][-1]["outcome"] == "all_fail"
    assert row["armed_at"] is not None


def test_check_exits_2_when_an_armed_row_errors_on_every_attempt(ws: Path) -> None:
    assert _confirm(ws, "old").exit_code == 0
    result = _check(ws, "allerr")
    assert result.exit_code == 2, result.output
    assert "INCONCLUSIVE example_2" in result.output


def test_fail_on_intermittent_turns_majority_failure_into_exit_1(ws: Path) -> None:
    assert _confirm(ws, "old").exit_code == 0
    default = _check(ws, "old", "--repeats", "5")
    assert default.exit_code == 0, default.output
    gated = _check(ws, "old", "--repeats", "5", "--fail-on-intermittent")
    assert gated.exit_code == 1, gated.output
    assert "REGRESSION example_1: attempts=5 passes=2 fails=3" in gated.output
    assert "example_1" not in default.output.split("not gated")[0]


def test_check_json_is_one_closed_shape_object(ws: Path) -> None:
    assert _confirm(ws, "old").exit_code == 0
    assert _confirm(ws, "fixed").exit_code == 0
    result = _check(ws, "old", "--json")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert set(payload) == {
        "schema",
        "verdict",
        "exit_code",
        "repeats",
        "armed",
        "passed",
        "run_complete",
        "regressions",
        "inconclusive",
        "not_gated",
    }
    assert payload["verdict"] == "regression" and payload["exit_code"] == 1
    assert [r["id"] for r in payload["regressions"]] == ["example_0"]
    assert set(payload["regressions"][0]) == {
        "id",
        "attempts",
        "passes",
        "fails",
        "errored",
        "pass_rate_low",
        "pass_rate_high",
    }
    assert "synthetic question" not in result.output
    assert result.stdout.count("\n") == 1  # one object, logs stay on stderr


def test_zero_armed_rows_exit_0_and_say_so(ws: Path) -> None:
    result = _check(ws, "old")
    assert result.exit_code == 0
    assert "0 armed rows; 0 known failures not gated" in result.output
    assert _confirm(ws, "old", "--only-id", "example_0").exit_code == 0
    again = _check(ws, "old")
    assert again.exit_code == 0
    assert "0 armed rows; 1 known failures not gated" in again.output
    assert "not gated (known_failure): example_0" in again.output


def test_check_never_changes_the_file_on_any_path(ws: Path) -> None:
    assert _confirm(ws, "old").exit_code == 0
    assert _confirm(ws, "fixed").exit_code == 0
    before = _snapshot(ws)
    names_before = sorted(before)
    for argv in (
        ("old",),
        ("fixed",),
        ("allerr",),
        ("old", "--fail-on-intermittent", "--repeats", "3"),
        ("old", "--json"),
    ):
        _check(ws, *argv)
    # failure paths: bad config, bad module
    assert _check(ws, "old", config=str(ws / "missing.json")).exit_code == 2
    bad = CliRunner().invoke(
        cli,
        [
            "regression",
            "check",
            "--module",
            "nope.py:f",
            "--config",
            _write_config(ws, "old"),
            "--dir",
            str(ws / "reg"),
        ],
    )
    assert bad.exit_code == 2
    assert _snapshot(ws) == before
    assert sorted(_snapshot(ws)) == names_before  # no new lock or temp files


def test_check_does_not_create_a_lock_file_in_a_fresh_confirmed_dir(ws: Path) -> None:
    assert _confirm(ws, "old").exit_code == 0
    for p in (ws / "reg").glob(".*.lock"):
        p.unlink()
    assert _check(ws, "old").exit_code == 0
    assert list((ws / "reg").glob(".*.lock")) == []


# ---------------------------------------------------------------------- disarm


def test_disarm_returns_a_row_to_known_failure_and_records_why(ws: Path) -> None:
    assert _confirm(ws, "old").exit_code == 0
    result = CliRunner().invoke(
        cli,
        [
            "regression",
            "confirm",
            "--dir",
            str(ws / "reg"),
            "--disarm",
            "example_2",
            "--reason",
            "known_flaky",
        ],
    )
    assert result.exit_code == 0, result.output
    row = _reg(ws)["example_2"]
    assert row["status"] == "known_failure"
    assert row["armed_at"] is None
    last = row["observations"][-1]
    assert last["outcome"] == "disarmed" and last["reason"] == "known_flaky"
    assert len(row["observations"]) == 2  # history kept
    zero = _check(ws, "old")
    assert zero.exit_code == 0 and "0 armed rows" in zero.output


def test_disarm_needs_a_closed_reason_and_an_armed_row_and_writes_nothing_else(
    ws: Path,
) -> None:
    assert _confirm(ws, "old").exit_code == 0
    before = _snapshot(ws)
    runner = CliRunner()
    base = ["regression", "confirm", "--dir", str(ws / "reg"), "--disarm"]
    assert runner.invoke(cli, [*base, "example_2"]).exit_code == 2
    assert (
        runner.invoke(cli, [*base, "example_2", "--reason", "because"]).exit_code == 2
    )
    not_armed = runner.invoke(cli, [*base, "example_0", "--reason", "other"])
    assert not_armed.exit_code == 3
    assert _snapshot(ws) == before


# -------------------------------------------------------------------- max cost


def test_max_cost_stops_early_writes_partial_observations_and_exits_4(
    ws: Path,
) -> None:
    result = _confirm(ws, "costed", "--max-cost", "0.09", "--repeats", "5")
    assert result.exit_code == 4, result.output
    rows = _reg(ws)
    reps = {o["reps"] for r in rows.values() for o in r["observations"]}
    assert reps == {2}  # each repeat costs $0.04; a third would pass the cap
    assert all(r["observations"] for r in rows.values())
    assert "Stopped early" in result.output


def test_max_cost_with_fewer_than_two_repeats_writes_nothing(ws: Path) -> None:
    before = _snapshot(ws)
    result = _confirm(ws, "costed", "--max-cost", "0.05", "--repeats", "5")
    assert result.exit_code == 4, result.output
    assert _snapshot(ws) == before


# ------------------------------------------------------------ bad file / input


def test_missing_file_check_exits_3_and_confirm_refuses(ws: Path) -> None:
    _file(ws).unlink()
    assert _check(ws, "old").exit_code == 3
    result = _confirm(ws, "old")
    assert result.exit_code == 3
    assert not _file(ws).exists()


def test_corrupt_file_check_exits_3_confirm_writes_nothing(ws: Path) -> None:
    _file(ws).write_text("{not json\n", encoding="utf-8")
    before = _snapshot(ws)
    assert _check(ws, "old").exit_code == 3
    assert _confirm(ws, "old").exit_code == 3
    assert _snapshot(ws) == before


def test_hand_edited_row_with_a_stale_digest_exits_3_everywhere(ws: Path) -> None:
    lines = _file(ws).read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["output"] = "tampered"
    lines[0] = json.dumps(row)
    _file(ws).write_text("\n".join(lines) + "\n", encoding="utf-8")
    before = _snapshot(ws)
    check = _check(ws, "old")
    assert check.exit_code == 3 and "content_digest" in check.output
    assert _confirm(ws, "old").exit_code == 3
    assert _snapshot(ws) == before


def test_a_status_edited_without_observations_is_corrupt_not_armed(ws: Path) -> None:
    assert _confirm(ws, "old").exit_code == 0
    lines = _file(ws).read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["metadata"]["regression"]["status"] = "armed"
    lines[0] = json.dumps(row)
    _file(ws).write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _check(ws, "old").exit_code == 3


def test_check_exit_2_for_an_unloadable_function_or_invalid_config(ws: Path) -> None:
    assert _confirm(ws, "old").exit_code == 0
    wrong_name = CliRunner().invoke(
        cli,
        [
            "regression",
            "check",
            "--module",
            "agent_mod.py:absent",
            "--config",
            _write_config(ws, "old"),
            "--dir",
            str(ws / "reg"),
        ],
    )
    assert wrong_name.exit_code == 2
    (ws / "bad.json").write_text("[1]", encoding="utf-8")
    assert _check(ws, "old", config=str(ws / "bad.json")).exit_code == 2
    out_of_space = ws / "oos.json"
    out_of_space.write_text(json.dumps({"mode": "nonsense"}), encoding="utf-8")
    refused = _check(ws, "old", config=str(out_of_space))
    assert refused.exit_code == 2, refused.output


def test_confirm_unknown_only_id_is_a_usage_error(ws: Path) -> None:
    before = _snapshot(ws)
    result = _confirm(ws, "old", "--only-id", "nope")
    assert result.exit_code == 2
    assert _snapshot(ws) == before


# ----------------------------------------------------------------- concurrency


def test_two_confirms_on_different_ids_both_land(ws: Path) -> None:
    results: dict[str, Any] = {}

    def run(name: str, eid: str) -> None:
        results[name] = _confirm(ws, "slow", "--only-id", eid, "--repeats", "2")

    threads = [
        threading.Thread(target=run, args=("a", "example_0")),
        threading.Thread(target=run, args=("b", "example_1")),
    ]
    # Distinct config file names are written per call; the module loads twice.
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results["a"].exit_code == 0, results["a"].output
    assert results["b"].exit_code == 0, results["b"].output
    reg = _reg(ws)
    assert len(reg["example_0"]["observations"]) == 1
    assert len(reg["example_1"]["observations"]) == 1
    assert reg["example_2"].get("observations", []) == []


def test_an_add_during_a_slow_confirm_succeeds_and_its_row_survives(ws: Path) -> None:
    done: dict[str, Any] = {}

    def slow_confirm() -> None:
        done["confirm"] = _confirm(ws, "slow", "--repeats", "2")

    thread = threading.Thread(target=slow_confirm)
    thread.start()
    time.sleep(1.0)  # the confirm is inside its model calls, outside the lock
    assert thread.is_alive()

    extra = {"input": {"q": "synthetic question added"}, "output": "ax"}
    rows = [*ROWS, extra]
    (ws / "data2.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    result_obj = _failing_result([0, 1, 2, 3], "opt-b")
    result_obj.trials[0].metadata["example_results"].append(
        ExampleResult(
            example_id="example_4",
            input_data=extra["input"],
            expected_output=extra["output"],
            actual_output="x",
            metrics={"accuracy": 0.0},
            execution_time=0.1,
            success=True,
            error_message=None,
        ).to_dict()
    )
    PersistenceManager(str(ws / "results")).save_result(result_obj, name="run_b")
    added = CliRunner().invoke(
        cli,
        [
            "regression",
            "add",
            "run_b",
            "--dataset",
            str(ws / "data2.jsonl"),
            "--dir",
            str(ws / "reg"),
            "--storage-dir",
            str(ws / "results"),
        ],
    )
    assert added.exit_code == 0, added.output
    added_at = time.time()
    # The add returned while the confirm was still inside its model calls: it
    # did not wait for them, so the lock is not held across them.
    assert "confirm" not in done
    thread.join()
    last_call = max(float(t) for t in (ws / "calls.log").read_text().split())
    assert added_at < last_call
    assert done["confirm"].exit_code == 0, done["confirm"].output
    reg = _reg(ws)
    assert "example_4" in reg  # the confirm's write did not discard the added row
    assert all(len(reg[f"example_{i}"]["observations"]) == 1 for i in range(4))
    assert reg["example_4"].get("observations", []) == []


# --------------------------------------------------------------- egress & help


def test_neither_command_touches_the_network(ws: Path) -> None:
    with capture_egress() as cap:
        assert _confirm(ws, "old").exit_code == 0
        assert _check(ws, "old").exit_code == 0
    assert cap.requests == []
    assert cap.guard_violations == []
    assert cap.dns_lookups == []


def test_help_lists_three_commands_and_states_what_each_writes() -> None:
    runner = CliRunner()
    group = runner.invoke(cli, ["regression", "--help"])
    section = group.output.split("Commands:")[1]
    assert [line.split()[0] for line in section.splitlines() if line.strip()] == [
        "add",
        "check",
        "confirm",
    ]
    text = lambda argv: " ".join(runner.invoke(cli, argv).output.split())  # noqa: E731
    assert "WRITES <dir>/regression.jsonl" in text(["regression", "confirm", "--help"])
    assert "never deleted" in text(["regression", "confirm", "--help"])
    assert "WRITES NOTHING" in text(["regression", "check", "--help"])
    assert "writes the file" in " ".join(group.output.split())
    assert "writes nothing" in " ".join(group.output.split())
