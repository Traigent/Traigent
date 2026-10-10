"""End-to-end CLI tests: every name ``results list`` prints must be usable.

``traigent results list`` prints local sessions as ``local:<session_id>``.
``results show``, ``export``, ``plot`` (and ``compare`` / ``rerank``) must
resolve those names through one shared resolver, fail with a non-zero exit code
when a name cannot be resolved, and never read outside the results directory.
All data is synthetic and lives in a temporary results directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.storage.local_storage import LocalStorageManager


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_session(store: Path, function_name: str, scores: list[float]) -> str:
    storage = LocalStorageManager(str(store))
    session_id = storage.create_session(
        function_name, optimization_config={"algorithm": "random"}
    )
    for index, score in enumerate(scores):
        storage.add_trial_result(session_id, {"temperature": 0.1 * (index + 1)}, score)
    storage.finalize_session(session_id)
    return session_id


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "results"
    _make_session(path, "triage_agent", [0.5, 0.9, 0.7])
    return path


def _printed_names(runner: CliRunner, store: Path) -> list[str]:
    output = runner.invoke(cli, ["results", "list", "-d", str(store)]).output
    return re.findall(r"local:[A-Za-z0-9_\-]+", output)


class TestListedNamesRoundTrip:
    def test_listed_name_is_accepted_by_show_export_and_plot(
        self, runner: CliRunner, store: Path, tmp_path: Path
    ) -> None:
        names = _printed_names(runner, store)
        assert len(names) == 1
        name = names[0]

        shown = runner.invoke(cli, ["results", "show", name, "-d", str(store)])
        assert shown.exit_code == 0, shown.output
        assert "not found" not in shown.output.lower()
        assert "triage_agent" in shown.output

        out = tmp_path / "cand.json"
        exported = runner.invoke(
            cli, ["export", name, "-d", str(store), "-o", str(out)]
        )
        assert exported.exit_code == 0, exported.output
        data = json.loads(out.read_text())
        assert data["config"] == {"temperature": pytest.approx(0.2)}
        assert data["function_name"] == "triage_agent"
        assert data["metrics"]["score"] == pytest.approx(0.9)

        plotted = runner.invoke(cli, ["plot", name, "-d", str(store)])
        assert plotted.exit_code == 0, plotted.output
        assert "not found" not in plotted.output.lower()

    def test_bare_session_id_and_unique_prefix_are_accepted(
        self, runner: CliRunner, store: Path
    ) -> None:
        name = _printed_names(runner, store)[0]
        session_id = name.removeprefix("local:")
        for ident in (session_id, name[: len("local:") + 12], session_id[:12]):
            result = runner.invoke(cli, ["results", "show", ident, "-d", str(store)])
            assert result.exit_code == 0, (ident, result.output)
            assert "not found" not in result.output.lower(), ident
            assert "triage_agent" in result.output, ident

    def test_export_default_output_name_has_no_colon(
        self,
        runner: CliRunner,
        store: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        name = _printed_names(runner, store)[0]
        result = runner.invoke(cli, ["export", name, "-d", str(store)])
        assert result.exit_code == 0, result.output
        written = list((tmp_path / "configs").glob("*.json"))
        assert len(written) == 1
        assert ":" not in written[0].name


class TestNotFoundExitsNonZero:
    @pytest.mark.parametrize(
        "argv",
        [
            ["results", "show"],
            ["export"],
            ["plot"],
        ],
    )
    def test_unknown_name_exits_nonzero_and_names_it(
        self, runner: CliRunner, store: Path, argv: list[str]
    ) -> None:
        result = runner.invoke(cli, [*argv, "local:does-not-exist", "-d", str(store)])
        assert result.exit_code != 0
        assert "local:does-not-exist" in result.output
        assert "traigent results list" in result.output

    def test_export_unknown_name_writes_no_file(
        self, runner: CliRunner, store: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "cand.json"
        result = runner.invoke(
            cli, ["export", "nope", "-d", str(store), "-o", str(out)]
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_export_without_best_config_exits_nonzero(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        empty = tmp_path / "results"
        session_id = LocalStorageManager(str(empty)).create_session("f")
        out = tmp_path / "cand.json"
        result = runner.invoke(
            cli,
            ["export", f"local:{session_id}", "-d", str(empty), "-o", str(out)],
        )
        assert result.exit_code != 0
        assert not out.exists()


class TestAmbiguousPrefix:
    def test_ambiguous_prefix_is_rejected_with_candidates(
        self, runner: CliRunner, store: Path
    ) -> None:
        second = _make_session(store, "triage_agent", [0.4])
        first = _printed_names(runner, store)
        assert len(first) == 2
        ids = [n.removeprefix("local:") for n in first]
        common = ""
        for a, b in zip(*ids, strict=False):
            if a != b:
                break
            common += a
        assert common, "synthetic sessions should share a timestamp prefix"

        result = runner.invoke(
            cli, ["results", "show", f"local:{common}", "-d", str(store)]
        )
        assert result.exit_code != 0
        assert "ambiguous" in result.output.lower()
        for session_id in ids:
            assert session_id in result.output
        assert second in result.output


class TestPathHandling:
    def test_traversal_in_name_cannot_read_outside_results_dir(
        self, runner: CliRunner, store: Path, tmp_path: Path
    ) -> None:
        # A syntactically valid session file *outside* the results dir.
        outside = tmp_path / "outside"
        outside_id = _make_session(outside, "secret_agent", [1.0])
        target = outside / "sessions" / outside_id

        for ident in (
            f"local:../../outside/sessions/{outside_id}",
            f"local:{target}",
            f"../outside/sessions/{outside_id}",
        ):
            result = runner.invoke(cli, ["results", "show", ident, "-d", str(store)])
            assert result.exit_code != 0, (ident, result.output)
            assert "Best Configuration" not in result.output
            assert "Summary" not in result.output


class TestUnsafeNamesAndDefaultDestination:
    @pytest.mark.parametrize(
        "ident",
        [
            "../victim",
            "local:../victim",
            "a/b",
            "local:a/b",
            "/etc/passwd",
            "local:",
            "..",
            "a\\b",
            "C:victim",
            "C:",
            "c:\\x",
            "\\\\server\\share\\x",
            "con",
            "nul.json",
        ],
    )
    def test_unsafe_names_exit_nonzero_and_write_nothing(
        self,
        runner: CliRunner,
        store: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ident: str,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(cli, ["export", ident, "-d", str(store)])
        assert result.exit_code != 0, result.output
        assert not (tmp_path / "victim.json").exists()
        assert not (tmp_path / "configs").exists()

    def test_nul_in_name_exits_nonzero(
        self, runner: CliRunner, store: Path, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(cli, ["export", "local:a\x00b", "-d", str(store)])
        assert result.exit_code != 0
        assert not (tmp_path / "configs").exists()

    def test_stored_session_id_with_dotdot_is_not_resolved(
        self,
        runner: CliRunner,
        store: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sessions = store / "sessions"
        src = next(sessions.glob("*.json"))
        data = json.loads(src.read_text())
        data["session_id"] = "../victim"
        (sessions / "evil.json").write_text(json.dumps(data))
        monkeypatch.chdir(tmp_path)
        for ident in ("local:../victim", "local:../vic"):
            result = runner.invoke(cli, ["export", ident, "-d", str(store)])
            assert result.exit_code != 0, (ident, result.output)
        assert not (tmp_path / "victim.json").exists()
        assert not (tmp_path / "configs").exists()
        assert not list(tmp_path.glob("victim*"))

    def test_default_destination_stays_in_configs(
        self,
        runner: CliRunner,
        store: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        name = _printed_names(runner, store)[0]
        result = runner.invoke(cli, ["export", name, "-d", str(store)])
        assert result.exit_code == 0, result.output
        written = list(tmp_path.rglob("*.json"))
        written = [p for p in written if store not in p.parents]
        assert [p.parent for p in written] == [(tmp_path / "configs").resolve()]

    def test_explicit_output_path_is_unchanged(
        self, runner: CliRunner, store: Path, tmp_path: Path
    ) -> None:
        name = _printed_names(runner, store)[0]
        out = tmp_path / "nested" / "custom.json"
        result = runner.invoke(cli, ["export", name, "-d", str(store), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()


WINDOWS_UNSAFE_NAMES = [
    "C:victim",
    "C:",
    "c:\\x",
    "\\\\server\\share\\x",
    "con",
    "nul.json",
    "COM1",
    "name.",
    "name ",
]


class TestWindowsUnsafeComponents:
    @pytest.mark.parametrize("value", WINDOWS_UNSAFE_NAMES)
    def test_rejected_as_component(self, value: str) -> None:
        from traigent.cli.main import _is_safe_component

        assert _is_safe_component(value) is False

    def test_ordinary_ids_still_accepted(self) -> None:
        from traigent.cli.main import _is_safe_component

        assert _is_safe_component("session_1234-abc") is True

    @pytest.mark.parametrize("value", WINDOWS_UNSAFE_NAMES)
    def test_stored_session_id_is_not_resolved_or_written(
        self,
        runner: CliRunner,
        store: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
    ) -> None:
        sessions = store / "sessions"
        src = next(sessions.glob("*.json"))
        data = json.loads(src.read_text())
        data["session_id"] = value
        (sessions / "planted.json").write_text(json.dumps(data))
        monkeypatch.chdir(tmp_path)
        for ident in (value, f"local:{value}"):
            result = runner.invoke(cli, ["export", ident, "-d", str(store)])
            assert result.exit_code != 0, (ident, result.output)
        assert not (tmp_path / "configs").exists()


class TestLegacyLocalPrefixedName:
    def test_saved_result_literally_named_local_prefix_still_loads(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from traigent.api.types import OptimizationResult, OptimizationStatus
        from traigent.utils.persistence import PersistenceManager

        legacy_dir = tmp_path / "legacy"
        PersistenceManager(str(legacy_dir)).save_result(
            OptimizationResult(
                trials=[],
                best_config={"temperature": 0.3},
                best_score=0.8,
                optimization_id="legacy-1",
                duration=0.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=["accuracy"],
                algorithm="random",
                timestamp=__import__("datetime").datetime.now(),
                metadata={"function_name": "legacy_fn"},
            ),
            "local:manual",
        )
        out = tmp_path / "legacy_out.json"
        result = runner.invoke(
            cli,
            ["export", "local:manual", "-d", str(legacy_dir), "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(out.read_text())["config"] == {"temperature": 0.3}

    def test_exact_legacy_name_wins_over_session_id_prefix(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from traigent.api.types import OptimizationResult, OptimizationStatus
        from traigent.utils.persistence import PersistenceManager

        store = tmp_path / "mixed"
        session_id = _make_session(store, "session_fn", [0.5])
        prefix = session_id[:6]
        PersistenceManager(str(store)).save_result(
            OptimizationResult(
                trials=[],
                best_config={"temperature": 0.3},
                best_score=0.8,
                optimization_id="legacy-2",
                duration=0.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=["accuracy"],
                algorithm="random",
                timestamp=__import__("datetime").datetime.now(),
                metadata={"function_name": "legacy_fn"},
            ),
            f"local:{prefix}",
        )
        out = tmp_path / "mixed_out.json"
        result = runner.invoke(
            cli, ["export", f"local:{prefix}", "-d", str(store), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(out.read_text())["function_name"] == "legacy_fn"
