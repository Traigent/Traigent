"""Tests for ``traigent results export-failures`` (Traigent#2565).

The command selects a run's failing example ids from already-collected
outcomes, joins each to the ORIGINAL local dataset row whose input and
expected_output equal what the run recorded for that id, and writes a JSONL
regression dataset loadable by ``Dataset.from_jsonl``. Covered here:

- the ``<run>`` source is a saved result name (``PersistenceManager``, same
  default storage dir as ``results show``); a directory (an
  optimization-logger run, which records no example inputs) is refused;
- ``--threshold`` / ``--mode`` pass through to the shared selector;
- fail loudly (non-zero exit, ``--out`` untouched) on unresolvable ids, runs
  that did not log example content, missing runs, ``local:<session_id>`` runs,
  and ``--out`` resolving to the dataset itself;
- zero failures exit 0 with a message and never touch ``--out``.

No LLM calls and no network: results are built in memory and saved locally.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from traigent.api.types import (
    ExampleResult,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.cli.main import cli
from traigent.evaluators.base import Dataset
from traigent.utils.persistence import PersistenceManager

#: The original dataset rows. Positional ids example_0..example_2 are what the
#: local evaluator emits for rows without a metadata example_id.
DATASET_ROWS: list[dict[str, Any]] = [
    {"input": {"question": "2+2?"}, "output": "4"},
    {"input": {"question": "Capital of France? — ünïcode"}, "output": "Paris"},
    {"input": {"question": "Color of the sky?"}, "output": "blue", "tag": "sky"},
]

#: Per-trial accuracies. example_1 is wrong in BOTH configs; example_2 only in
#: the winner (trial t0).
ACCURACIES = {
    "t0": {"example_0": 1.0, "example_1": 0.0, "example_2": 0.0},
    "t1": {"example_0": 1.0, "example_1": 0.0, "example_2": 1.0},
}
CONFIGS = {"t0": {"model": "a"}, "t1": {"model": "b"}}
BEST_CONFIG = {"model": "a"}

NO_FAILURES_MESSAGE = "No failing examples; no dataset written"


#: What the run recorded for each id: the content of the row it evaluated.
RECORDED = {
    f"example_{index}": (row["input"], row["output"])
    for index, row in enumerate(DATASET_ROWS)
}


def _example(example_id: str, accuracy: float) -> dict[str, Any]:
    input_data, expected = RECORDED.get(example_id, ({"q": example_id}, "?"))
    return ExampleResult(
        example_id=example_id,
        input_data=input_data,
        expected_output=expected,
        actual_output="x",
        metrics={"accuracy": accuracy},
        execution_time=0.1,
        success=True,
    ).to_dict()


def _result(accuracies: dict[str, dict[str, float]] = ACCURACIES) -> OptimizationResult:
    trials = [
        TrialResult(
            trial_id=trial_id,
            config=CONFIGS[trial_id],
            metrics={"accuracy": sum(per_ex.values()) / len(per_ex)},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metadata={
                "example_results": [
                    _example(ex_id, acc) for ex_id, acc in per_ex.items()
                ]
            },
        )
        for trial_id, per_ex in accuracies.items()
    ]
    return OptimizationResult(
        trials=trials,
        best_config=BEST_CONFIG,
        best_score=0.5,
        optimization_id="opt-2565",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="GridSearchOptimizer",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={"function_name": "f"},
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _contents(dataset: Dataset) -> list[tuple[Any, Any, dict[str, Any]]]:
    return [(ex.input_data, ex.expected_output, ex.metadata) for ex in dataset]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp workspace that is also the dataset root, holding the dataset."""
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in DATASET_ROWS),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def storage_dir(workspace: Path) -> Path:
    storage = workspace / "results"
    PersistenceManager(str(storage)).save_result(_result(), name="run1")
    return storage


def _invoke(
    runner: CliRunner,
    run: str,
    workspace: Path,
    *extra: str,
    out_name: str = "failures.jsonl",
):
    return runner.invoke(
        cli,
        [
            "results",
            "export-failures",
            run,
            "--dataset",
            str(workspace / "data.jsonl"),
            "--out",
            str(workspace / out_name),
            *extra,
        ],
    )


class TestFixtureSanity:
    """Guards the fixtures themselves, independently of the new command."""

    def test_saved_result_keeps_the_per_example_ids(self, storage_dir: Path) -> None:
        loaded = PersistenceManager(str(storage_dir)).load_result("run1")
        assert [row["example_id"] for row in loaded.example_matrix["examples"]] == [
            "example_0",
            "example_1",
            "example_2",
        ]
        assert loaded.best_config == BEST_CONFIG


class TestRegistration:
    def test_command_is_registered_with_its_options(
        self, runner: CliRunner, plain: Callable[[str], str]
    ) -> None:
        result = runner.invoke(cli, ["results", "export-failures", "--help"])
        assert result.exit_code == 0, result.output
        text = plain(result.output)
        for option in ("--dataset", "--out", "--threshold", "--mode", "--storage-dir"):
            assert option in text
        assert "all_configs" in text
        assert "best_config" in text

    def test_invalid_mode_is_a_usage_error(
        self, runner: CliRunner, workspace: Path, storage_dir: Path
    ) -> None:
        result = _invoke(
            runner,
            "run1",
            workspace,
            "--storage-dir",
            str(storage_dir),
            "--mode",
            "any_config",
        )
        assert result.exit_code != 0
        assert not (workspace / "failures.jsonl").exists()


class TestSavedResultSource:
    def test_exports_examples_no_config_got_right(
        self,
        runner: CliRunner,
        workspace: Path,
        storage_dir: Path,
        plain: Callable[[str], str],
    ) -> None:
        result = _invoke(runner, "run1", workspace, "--storage-dir", str(storage_dir))

        assert result.exit_code == 0, result.output
        out = workspace / "failures.jsonl"
        rows = _read_rows(out)
        assert len(rows) == 1
        assert rows[0]["input"] == DATASET_ROWS[1]["input"]
        assert rows[0]["expected_output"] == DATASET_ROWS[1]["output"]
        # A row without a metadata example_id gains its source id.
        assert rows[0]["metadata"] == {"example_id": "example_1"}
        assert "failures.jsonl" in plain(result.output)

    def test_output_round_trips_through_from_jsonl(
        self, runner: CliRunner, workspace: Path, storage_dir: Path
    ) -> None:
        result = _invoke(runner, "run1", workspace, "--storage-dir", str(storage_dir))
        assert result.exit_code == 0, result.output

        exported = Dataset.from_jsonl(str(workspace / "failures.jsonl"))
        original = Dataset.from_jsonl(str(workspace / "data.jsonl"))
        assert _contents(exported) == [
            (
                {"question": "Capital of France? — ünïcode"},
                "Paris",
                {"example_id": "example_1"},
            )
        ]
        assert exported.examples[0].input_data == original.examples[1].input_data

    def test_best_config_mode_and_metadata_are_preserved(
        self, runner: CliRunner, workspace: Path, storage_dir: Path
    ) -> None:
        result = _invoke(
            runner,
            "run1",
            workspace,
            "--storage-dir",
            str(storage_dir),
            "--mode",
            "best_config",
        )
        assert result.exit_code == 0, result.output

        exported = Dataset.from_jsonl(str(workspace / "failures.jsonl"))
        original = Dataset.from_jsonl(str(workspace / "data.jsonl"))
        # Winner (t0) got example_1 and example_2 wrong; original order kept,
        # and example_2's top-level "tag" survives as metadata.
        assert [(e.input_data, e.expected_output) for e in exported] == [
            (e.input_data, e.expected_output) for e in original.examples[1:]
        ]
        assert exported.examples[1].metadata == {
            "tag": "sky",
            "example_id": "example_2",
        }

    def test_default_storage_dir_matches_results_show(
        self,
        runner: CliRunner,
        workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(workspace)
        PersistenceManager(".traigent").save_result(_result(), name="run1")

        result = _invoke(runner, "run1", workspace)

        assert result.exit_code == 0, result.output
        assert len(_read_rows(workspace / "failures.jsonl")) == 1

    def test_threshold_option_changes_selection(
        self,
        runner: CliRunner,
        workspace: Path,
        storage_dir: Path,
        plain: Callable[[str], str],
    ) -> None:
        # accuracy 0.0 is not < 0.0, so nothing fails at threshold 0.
        result = _invoke(
            runner,
            "run1",
            workspace,
            "--storage-dir",
            str(storage_dir),
            "--threshold",
            "0",
        )
        assert result.exit_code == 0, result.output
        assert NO_FAILURES_MESSAGE in plain(result.output)
        assert not (workspace / "failures.jsonl").exists()


class TestDirectorySourceIsRefused:
    def test_a_run_directory_is_refused_with_a_reason(
        self,
        runner: CliRunner,
        workspace: Path,
        plain: Callable[[str], str],
    ) -> None:
        run_dir = workspace / "logs" / "runs" / "r1"
        (run_dir / "artifacts").mkdir(parents=True)

        result = _invoke(runner, str(run_dir), workspace)

        assert result.exit_code != 0
        text = plain(result.output)
        assert "is a directory, not a saved result" in text
        assert "select_examples_by_id" in text
        assert not (workspace / "failures.jsonl").exists()


class TestNameResolution:
    def test_a_same_named_directory_in_cwd_does_not_shadow_the_saved_result(
        self,
        runner: CliRunner,
        workspace: Path,
        storage_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(workspace)
        (workspace / "run1").mkdir()

        result = _invoke(runner, "run1", workspace, "--storage-dir", str(storage_dir))

        assert result.exit_code == 0, result.output
        assert len(_read_rows(workspace / "failures.jsonl")) == 1


class TestCorruptSavedResult:
    def test_a_corrupt_saved_result_keeps_its_own_error(
        self,
        runner: CliRunner,
        workspace: Path,
        storage_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        plain: Callable[[str], str],
    ) -> None:
        # A same-named directory in cwd must not mask the real load failure.
        monkeypatch.chdir(workspace)
        (workspace / "run1").mkdir()
        for path in (storage_dir / "run1").iterdir():
            if path.is_file():
                path.write_bytes(b"not json")

        result = _invoke(runner, "run1", workspace, "--storage-dir", str(storage_dir))

        assert result.exit_code != 0
        assert "is a directory, not a saved result" not in plain(result.output)
        assert not (workspace / "failures.jsonl").exists()


class TestFailLoudly:
    def test_unresolvable_id_exits_nonzero_and_writes_nothing(
        self,
        runner: CliRunner,
        workspace: Path,
        plain: Callable[[str], str],
    ) -> None:
        storage = workspace / "results"
        ghost = {
            "t0": {"example_0": 1.0, "ghost-row": 0.0},
            "t1": {"example_0": 1.0, "ghost-row": 0.0},
        }
        PersistenceManager(str(storage)).save_result(_result(ghost), name="ghost")

        result = _invoke(runner, "ghost", workspace, "--storage-dir", str(storage))

        assert result.exit_code != 0
        assert "ghost-row" in plain(result.output)
        assert not (workspace / "failures.jsonl").exists()

    def test_unresolvable_id_does_not_replace_an_existing_output(
        self, runner: CliRunner, workspace: Path
    ) -> None:
        storage = workspace / "results"
        ghost = {"t0": {"ghost-row": 0.0}, "t1": {"ghost-row": 0.0}}
        PersistenceManager(str(storage)).save_result(_result(ghost), name="ghost")
        out = workspace / "failures.jsonl"
        out.write_text("previous export\n", encoding="utf-8")

        result = _invoke(runner, "ghost", workspace, "--storage-dir", str(storage))

        assert result.exit_code != 0
        assert out.read_text(encoding="utf-8") == "previous export\n"

    def test_run_without_logged_example_content_is_refused(
        self,
        runner: CliRunner,
        workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
        plain: Callable[[str], str],
    ) -> None:
        # Saved with example content opted out: ids and metrics survive, the
        # inputs do not, so no id can be tied to a row.
        monkeypatch.setenv("TRAIGENT_LOG_EXAMPLE_CONTENT", "false")
        storage = workspace / "results"
        PersistenceManager(str(storage)).save_result(_result(), name="nocontent")
        monkeypatch.delenv("TRAIGENT_LOG_EXAMPLE_CONTENT")

        result = _invoke(runner, "nocontent", workspace, "--storage-dir", str(storage))

        assert result.exit_code != 0
        assert "example_1" in plain(result.output)
        assert not (workspace / "failures.jsonl").exists()

    def test_unknown_run_exits_nonzero(
        self, runner: CliRunner, workspace: Path, storage_dir: Path
    ) -> None:
        result = _invoke(
            runner, "no-such-run", workspace, "--storage-dir", str(storage_dir)
        )
        assert result.exit_code != 0
        assert not (workspace / "failures.jsonl").exists()

    def test_local_session_runs_are_rejected(
        self,
        runner: CliRunner,
        workspace: Path,
        storage_dir: Path,
        plain: Callable[[str], str],
    ) -> None:
        result = _invoke(
            runner, "local:abc123", workspace, "--storage-dir", str(storage_dir)
        )
        assert result.exit_code != 0
        assert "local:" in plain(result.output)
        assert not (workspace / "failures.jsonl").exists()

    @pytest.mark.parametrize("spelling", ["same", "dotdot"])
    def test_out_resolving_to_the_dataset_is_refused(
        self,
        runner: CliRunner,
        workspace: Path,
        storage_dir: Path,
        spelling: str,
    ) -> None:
        dataset = workspace / "data.jsonl"
        before = dataset.read_bytes()
        if spelling == "same":
            out = str(dataset)
        else:
            (workspace / "sub").mkdir()
            out = str(workspace / "sub" / ".." / "data.jsonl")

        result = runner.invoke(
            cli,
            [
                "results",
                "export-failures",
                "run1",
                "--dataset",
                str(dataset),
                "--out",
                out,
                "--storage-dir",
                str(storage_dir),
            ],
        )

        assert result.exit_code != 0
        assert dataset.read_bytes() == before


class TestZeroFailures:
    def test_zero_failures_exits_zero_and_leaves_out_untouched(
        self,
        runner: CliRunner,
        workspace: Path,
        plain: Callable[[str], str],
    ) -> None:
        storage = workspace / "results"
        all_right = {
            "t0": {"example_0": 1.0, "example_1": 1.0},
            "t1": {"example_0": 1.0, "example_1": 0.0},
        }
        PersistenceManager(str(storage)).save_result(_result(all_right), name="ok")
        out = workspace / "failures.jsonl"
        out.write_text("previous export\n", encoding="utf-8")

        result = _invoke(runner, "ok", workspace, "--storage-dir", str(storage))

        assert result.exit_code == 0, result.output
        assert NO_FAILURES_MESSAGE in plain(result.output)
        assert out.read_text(encoding="utf-8") == "previous export\n"
