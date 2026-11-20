"""``check`` passes an armed row only when passes beat fails among scored attempts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.unit.cli.regression_run_helpers import make_row, read_rows, write_rows
from traigent.cli.main import cli

# The call-by-call result of the one row is scripted by the config's pattern:
# P passes, F fails, E raises.
SOURCE = """
import traigent

_CALLS = []

@traigent.optimize(
    eval_dataset=[{"input": {"q": "pass"}, "output": "ok"}],
    objectives=["accuracy"],
    configuration_space={"pattern": ["PP", "PF", "PFPF", "PPF", "PFF", "PFE", "FF", "EE"]},
    execution_mode="local",
    max_trials=1,
)
def answer(q: str, example_id: str = "") -> str:
    pattern = traigent.get_config().get("pattern")
    index = len(_CALLS)
    _CALLS.append(index)
    step = pattern[index] if index < len(pattern) else "P"
    if step == "E":
        raise RuntimeError("scripted failure")
    return "ok" if step == "P" else "wrong"
"""


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text(SOURCE, encoding="utf-8")
    write_rows(tmp_path, [make_row("P", {"q": "pass"}, "ok")])
    return tmp_path


def _run(ws: Path, verb: str, pattern: str, *extra: str):
    config = ws / f"config_{pattern}.json"
    config.write_text(json.dumps({"pattern": pattern}), encoding="utf-8")
    return CliRunner().invoke(
        cli,
        [
            "regression",
            verb,
            "--module",
            "agent.py:answer",
            "--config",
            str(config),
            "--dir",
            str(ws / "reg"),
            *extra,
        ],
    )


def _arm(ws: Path) -> None:
    done = _run(ws, "confirm", "PP", "--repeats", "2")
    assert done.exit_code == 0, done.output
    assert read_rows(ws)["P"]["metadata"]["regression"]["status"] == "armed"


@pytest.mark.parametrize(
    ("pattern", "repeats", "code", "verdict"),
    [
        ("PF", 2, 2, "inconclusive"),  # 1 pass, 1 fail: a tie
        ("PFPF", 4, 2, "inconclusive"),  # 2 pass, 2 fail: a tie
        ("PPF", 3, 0, "pass"),  # 2 pass, 1 fail
        ("PFF", 3, 1, "regression"),  # 1 pass, 2 fail
        ("PFE", 3, 2, "inconclusive"),  # 1 pass, 1 fail, 1 errored: a tie
        ("FF", 2, 1, "regression"),
        ("PP", 2, 0, "pass"),
    ],
)
def test_the_exit_code_follows_passes_against_fails(
    ws: Path, pattern: str, repeats: int, code: int, verdict: str
) -> None:
    _arm(ws)
    result = _run(ws, "check", pattern, "--repeats", str(repeats), "--json")
    assert result.exit_code == code, result.output
    payload = json.loads(result.stdout)
    assert payload["verdict"] == verdict
    assert payload["exit_code"] == code
    if verdict == "inconclusive":
        assert [(r["id"], r["reason"]) for r in payload["inconclusive"]] == [
            ("P", "tie")
        ]
        assert payload["passed"] == 0


def test_a_tie_is_named_in_the_text_output(ws: Path) -> None:
    _arm(ws)
    result = _run(ws, "check", "PF", "--repeats", "2")
    assert result.exit_code == 2
    assert "INCONCLUSIVE P: tie" in result.output
    assert "passes=1 fails=1" in result.output


def test_a_regression_still_wins_over_a_tie_for_the_exit_code(ws: Path) -> None:
    rows = [
        make_row("P", {"q": "pass"}, "ok"),
        make_row("Q", {"q": "pass2"}, "ok"),
    ]
    write_rows(ws, rows)
    (ws / "agent.py").write_text(
        SOURCE.replace(
            "    index = len(_CALLS)\n    _CALLS.append(index)\n",
            "    index = _CALLS.count(q)\n    _CALLS.append(q)\n"
            '    pattern = "FF" if q == "pass2" and pattern == "PF" else pattern\n',
        ),
        encoding="utf-8",
    )
    assert _run(ws, "confirm", "PP", "--repeats", "2").exit_code == 0
    result = _run(ws, "check", "PF", "--repeats", "2", "--json")
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert [r["id"] for r in payload["regressions"]] == ["Q"]
    assert [r["id"] for r in payload["inconclusive"]] == ["P"]


@pytest.mark.parametrize(
    ("pattern", "repeats", "code"),
    [("PF", 2, 2), ("PFPF", 4, 2), ("PPF", 3, 0), ("PFF", 3, 1)],
)
def test_fail_on_intermittent_rows_follow_the_same_strict_rule(
    ws: Path, pattern: str, repeats: int, code: int
) -> None:
    done = _run(ws, "confirm", "PF", "--repeats", "2")
    assert done.exit_code == 0, done.output
    assert read_rows(ws)["P"]["metadata"]["regression"]["status"] == "intermittent"
    result = _run(
        ws,
        "check",
        pattern,
        "--repeats",
        str(repeats),
        "--fail-on-intermittent",
        "--json",
    )
    assert result.exit_code == code, result.output


def test_a_row_with_no_scored_attempt_stays_inconclusive(ws: Path) -> None:
    _arm(ws)
    result = _run(ws, "check", "EE", "--repeats", "2", "--json")
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["inconclusive"][0]["errored"] == 2
    assert "reason" not in payload["inconclusive"][0]


def test_the_help_text_states_the_strict_rule() -> None:
    text = " ".join(
        CliRunner().invoke(cli, ["regression", "check", "--help"]).output.split()
    )
    assert "more scored attempts pass than fail" in text
    assert 'reason "tie"' in text


def test_check_never_returns_0_unless_every_armed_row_passed_strictly(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tally that is neither pass, regression nor tie must not exit 0."""
    from traigent.utils import regression_confirm

    _arm(ws)
    monkeypatch.setattr(regression_confirm.Tally, "passed_strictly", False)
    monkeypatch.setattr(regression_confirm.Tally, "regressed", False)
    monkeypatch.setattr(regression_confirm.Tally, "tied", False)
    result = _run(ws, "check", "PP", "--repeats", "2", "--json")
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "inconclusive" and payload["passed"] == 0
