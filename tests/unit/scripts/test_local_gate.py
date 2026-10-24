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


def _advance_remote(repo: Path, remote_branch: str) -> None:
    """Move refs/remotes/origin/<branch> one commit ahead of HEAD."""
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    new = subprocess.run(
        ["git", "commit-tree", tree, "-p", head, "-m", f"advance {remote_branch}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run_git(repo, "update-ref", f"refs/remotes/origin/{remote_branch}", new)


def _gate_on_branch(
    tmp_path: Path, branch: str, advanced: str
) -> subprocess.CompletedProcess[str]:
    repo, fake_bin = _make_gate_repo(tmp_path)
    _run_git(repo, "checkout", "-q", "-b", branch)
    _advance_remote(repo, advanced)
    return _run_gate(repo, fake_bin, tmp_path / "pytest.log", collection_rc=0)


def test_local_gate_hotfix_behind_develop_but_containing_main_passes(
    tmp_path: Path,
) -> None:
    result = _gate_on_branch(tmp_path, "hotfix/x", advanced="develop")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "expected for a hotfix cut from main" in result.stdout
    assert "❌ HEAD is behind" not in result.stdout


def test_local_gate_hotfix_behind_main_fails(tmp_path: Path) -> None:
    result = _gate_on_branch(tmp_path, "hotfix/x", advanced="main")

    assert result.returncode == 1
    assert "HEAD is behind origin/main" in result.stdout


def test_local_gate_release_behind_develop_still_fails(tmp_path: Path) -> None:
    result = _gate_on_branch(tmp_path, "release/x", advanced="develop")

    assert result.returncode == 1
    assert "❌ HEAD is behind origin/develop" in result.stdout


def test_local_gate_hotfix_strict_freshness_still_fails_behind_develop(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_GATE_STRICT_FRESHNESS", "1")
    result = _gate_on_branch(tmp_path, "hotfix/x", advanced="develop")

    assert result.returncode == 1
    assert "❌ HEAD is behind origin/develop" in result.stdout


def _run_sonar_gate(
    tmp_path: Path, *, with_docker: bool, version_line: str = 'version = "9.8.7"'
) -> tuple[subprocess.CompletedProcess[str], list[str], Path]:
    repo, fake_bin = _make_gate_repo(tmp_path)
    (repo / "pyproject.toml").write_text(
        f'[build-system]\nrequires = []\n\n[project]\nname = "x"\n{version_line}\n'
        '\n[tool.other]\nversion = "0.0.1"\n',
        encoding="utf-8",
    )
    scanner_log = tmp_path / "scanner.log"
    stub = '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$FAKE_SCANNER_LOG"\nexit 0\n'
    for name in ("docker", "sonar-scanner"):
        if name == "docker" and not with_docker:
            continue
        (fake_bin / name).write_text(stub, encoding="utf-8")
        (fake_bin / name).chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "LOCAL_GATE_SKIP": "schema-types",
            "LOCAL_GATE_SONAR": "1",
            "SONAR_TOKEN": "dummy",
            "SONAR_HOST_URL": "https://sonar.example.invalid",
            "FAKE_PYTEST_LOG": str(tmp_path / "pytest.log"),
            "FAKE_SCANNER_LOG": str(scanner_log),
        }
    )
    if not with_docker:
        # Shadow the system tools without docker so the gate falls back to
        # the stubbed sonar-scanner even on machines that have docker.
        nodocker = tmp_path / "nodocker-bin"
        nodocker.mkdir()
        for sysdir in ("/usr/bin", "/bin"):
            for tool in Path(sysdir).iterdir():
                link = nodocker / tool.name
                if tool.name.startswith("docker") or link.exists():
                    continue
                link.symlink_to(tool)
        env["PATH"] = f"{fake_bin}:{nodocker}"
    result = subprocess.run(
        ["bash", "scripts/local_gate.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        scanner_log.read_text(encoding="utf-8").splitlines()
        if scanner_log.exists()
        else []
    )
    return result, calls, repo


def test_local_gate_sonar_docker_passes_version_and_same_path_mount(
    tmp_path: Path,
) -> None:
    result, calls, repo = _run_sonar_gate(tmp_path, with_docker=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 1
    args = calls[0].split()
    # [project].version, not the [tool.other] one.
    assert "-Dsonar.projectVersion=9.8.7" in args
    mount = args[args.index("-v") + 1]
    host_path, container_path = mount.rsplit(":", 1)
    resolved = str(repo.resolve())
    assert host_path in (str(repo), resolved)
    assert container_path == host_path
    assert args[args.index("-w") + 1] == host_path
    assert f"-Dsonar.projectBaseDir={host_path}" in args
    assert "-Dsonar.projectKey=Traigent_Traigent" in args
    assert "-Dsonar.qualitygate.wait=true" in args


def test_local_gate_sonar_local_scanner_passes_version(tmp_path: Path) -> None:
    result, calls, _ = _run_sonar_gate(tmp_path, with_docker=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "-Dsonar.projectVersion=9.8.7" in calls[0].split()
    assert "-Dsonar.projectKey=Traigent_Traigent" in calls[0].split()


def test_local_gate_sonar_fails_without_a_readable_project_version(
    tmp_path: Path,
) -> None:
    result, calls, _ = _run_sonar_gate(
        tmp_path, with_docker=True, version_line='dynamic = ["version"]'
    )

    assert result.returncode == 1
    assert calls == []
    assert "could not read [project].version" in result.stdout
