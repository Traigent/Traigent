"""Behavioral tests for the local pre-push gate."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_GATE = REPO_ROOT / "scripts" / "local_gate.sh"


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_gate_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts" / "ci").mkdir(parents=True)
    (repo / "tests" / "unit").mkdir(parents=True)
    gate = repo / "scripts" / "local_gate.sh"
    shutil.copy2(LOCAL_GATE, gate)
    (repo / "scripts" / "ci" / "spine_preflight.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    changed_test = repo / "tests" / "unit" / "test_changed.py"
    changed_test.write_text("def test_before():\n    pass\n", encoding="utf-8")

    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "local-gate@example.invalid")
    _run_git(repo, "config", "user.name", "Local Gate Test")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-qm", "fixture")
    _run_git(repo, "update-ref", "refs/remotes/origin/develop", "HEAD")
    _run_git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    changed_test.write_text("def test_after():\n    pass\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "ruff").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (fake_bin / "pytest").write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s|%s|%s\\n' \"${TRAIGENT_MOCK_LLM:-}\" "
        '"${TRAIGENT_OFFLINE_MODE:-}" "${PYTHONPATH:-}" "$*" '
        '>> "$FAKE_PYTEST_LOG"\n'
        'if [[ " $* " == *" --collect-only "* ]]; then\n'
        '  exit "${FAKE_COLLECTION_RC:-0}"\n'
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    (fake_bin / "ruff").chmod(0o755)
    (fake_bin / "pytest").chmod(0o755)
    return repo, fake_bin


def _run_gate(
    repo: Path, fake_bin: Path, log_path: Path, collection_rc: int
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "LOCAL_GATE_SKIP": "schema-types,sonar",
            "FAKE_COLLECTION_RC": str(collection_rc),
            "FAKE_PYTEST_LOG": str(log_path),
        }
    )
    return subprocess.run(
        ["bash", "scripts/local_gate.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_local_gate_stops_before_smoke_when_full_collection_fails(
    tmp_path: Path,
) -> None:
    repo, fake_bin = _make_gate_repo(tmp_path)
    pytest_log = tmp_path / "pytest.log"

    result = _run_gate(repo, fake_bin, pytest_log, collection_rc=23)

    assert result.returncode == 1
    assert (
        "whole-suite collection failed; smoke/full-unit tier not run" in result.stdout
    )
    assert pytest_log.read_text(encoding="utf-8").splitlines() == [
        "true|true|.|--collect-only -q tests/ -o addopts= -p no:cacheprovider -rs"
    ]


def test_local_gate_runs_smoke_only_after_full_collection_passes(
    tmp_path: Path,
) -> None:
    repo, fake_bin = _make_gate_repo(tmp_path)
    pytest_log = tmp_path / "pytest.log"

    result = _run_gate(repo, fake_bin, pytest_log, collection_rc=0)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = pytest_log.read_text(encoding="utf-8").splitlines()
    assert calls[0] == (
        "true|true|.|--collect-only -q tests/ -o addopts= -p no:cacheprovider -rs"
    )
    assert calls[1].endswith(
        "tests/unit/test_init_imports.py tests/unit/api/test_types.py "
        "tests/unit/wrapper/test_wrapper_service.py -n 0"
    )
