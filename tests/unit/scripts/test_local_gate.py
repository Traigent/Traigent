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


def _install_fake_venv(repo: Path, log_path: Path, xdist_ok: bool) -> None:
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "pytest").write_text(
        f'#!/usr/bin/env bash\necho venv-pytest "$*" >> {log_path}\nexit 0\n',
        encoding="utf-8",
    )
    (venv_bin / "ruff").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (venv_bin / "python").write_text(
        f"#!/usr/bin/env bash\nexit {0 if xdist_ok else 1}\n", encoding="utf-8"
    )
    for name in ("pytest", "ruff", "python"):
        (venv_bin / name).chmod(0o755)


def test_local_gate_prefers_repo_venv_pytest_over_path(tmp_path: Path) -> None:
    repo, fake_bin = _make_gate_repo(tmp_path)
    path_log = tmp_path / "pytest.log"
    venv_log = tmp_path / "venv.log"
    _install_fake_venv(repo, venv_log, xdist_ok=True)

    result = _run_gate(repo, fake_bin, path_log, collection_rc=0)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pytest=.venv/bin/pytest" in result.stdout
    assert "ruff=.venv/bin/ruff" in result.stdout
    assert venv_log.read_text(encoding="utf-8").count("venv-pytest") == 2
    assert not path_log.exists()


def test_local_gate_fails_clearly_when_venv_lacks_xdist(tmp_path: Path) -> None:
    repo, fake_bin = _make_gate_repo(tmp_path)
    _install_fake_venv(repo, tmp_path / "venv.log", xdist_ok=False)

    result = _run_gate(repo, fake_bin, tmp_path / "pytest.log", collection_rc=0)

    assert result.returncode == 1
    assert "cannot import pytest-xdist" in result.stdout
    assert ".venv/bin/python" in result.stdout


def _schema_gate_script(tmp_path: Path, pin: str | None) -> Path:
    """Copy the gate into a sandbox repo with a stubbed generator + pin file."""
    repo = tmp_path / "gate"
    (repo / "scripts" / "ci").mkdir(parents=True)
    shutil.copy2(LOCAL_GATE, repo / "scripts" / "local_gate.sh")
    (repo / "scripts" / "ci" / "spine_preflight.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    (repo / "scripts" / "generate_schema_types.py").write_text(
        "import os\nprint('SCHEMA_REPO=' + os.environ['TRAIGENT_SCHEMA_REPO'])\n",
        encoding="utf-8",
    )
    if pin is not None:
        (repo / "scripts" / "ci" / "schema-pin.txt").write_text(
            f"traigent-schema @ git+https://example.invalid/s.git@{pin}\n",
            encoding="utf-8",
        )
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "t@example.invalid")
    _run_git(repo, "config", "user.name", "t")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-qm", "fixture")
    _run_git(repo, "update-ref", "refs/remotes/origin/develop", "HEAD")
    _run_git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _schema_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    schema = tmp_path / "schema"
    schema.mkdir()
    _run_git(schema, "init", "-q")
    _run_git(schema, "config", "user.email", "t@example.invalid")
    _run_git(schema, "config", "user.name", "t")
    (schema / "f").write_text("one", encoding="utf-8")
    _run_git(schema, "add", ".")
    _run_git(schema, "commit", "-qm", "one")
    first = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=schema, capture_output=True, text=True
    ).stdout.strip()
    (schema / "f").write_text("two", encoding="utf-8")
    _run_git(schema, "commit", "-qam", "two")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=schema, capture_output=True, text=True
    ).stdout.strip()
    return schema, first, head


def _run_schema_gate(repo: Path, schema: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "TRAIGENT_SCHEMA_REPO": str(schema),
            "LOCAL_GATE_SKIP": "pytest,sonar",
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


def test_local_gate_uses_schema_pin_when_checkout_is_elsewhere(
    tmp_path: Path,
) -> None:
    schema, pinned, _head = _schema_checkout(tmp_path)
    repo = _schema_gate_script(tmp_path, pinned)
    result = _run_schema_gate(repo, schema)
    out = result.stdout + result.stderr
    assert "exported to a temp dir" in out
    assert f"\nSCHEMA_REPO={schema}\n" not in out
    assert "\nSCHEMA_REPO=" in out


def test_local_gate_exports_pin_even_when_checkout_is_at_pin_and_dirty(
    tmp_path: Path,
) -> None:
    schema, _first, head = _schema_checkout(tmp_path)
    (schema / "f").write_text("dirty", encoding="utf-8")
    (schema / "untracked").write_text("x", encoding="utf-8")
    repo = _schema_gate_script(tmp_path, head)
    result = _run_schema_gate(repo, schema)
    out = result.stdout + result.stderr
    assert "exported to a temp dir" in out
    assert f"\nSCHEMA_REPO={schema}\n" not in out


def test_local_gate_fails_closed_without_a_readable_pin(tmp_path: Path) -> None:
    schema, _first, _head = _schema_checkout(tmp_path)
    repo = _schema_gate_script(tmp_path, None)
    result = _run_schema_gate(repo, schema)
    assert result.returncode != 0
    assert "cannot read the TraigentSchema pin" in result.stdout + result.stderr
