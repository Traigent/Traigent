"""Tests for scripts/ci/get_ruff_pin.py.

Guards the guard: the SDK Required PR Gate's ``preflight`` job used to run a
bare, unpinned ``pip install ruff``, so a new upstream ruff release could
retroactively fail unrelated PRs on pre-existing test-file lint debt (this
blocked PR #1540). ``pyproject.toml``'s ``dev`` extra is now the single
source of truth for the exact ruff version; this script reads it, and these
tests guard against that pin silently reverting to a loose spec.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "ci" / "get_ruff_pin.py"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def ruff_pin_module(monkeypatch: pytest.MonkeyPatch):
    """Load the script with its module-level pyproject path redirected."""
    spec = importlib.util.spec_from_file_location("_get_ruff_pin", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pyproject(root: Path, dev_deps: list[str]) -> None:
    deps_block = ", ".join(repr(s) for s in dev_deps)
    (root / "pyproject.toml").write_text(
        textwrap.dedent(f"""
            [project]
            name = "traigent"
            version = "0.0.0"

            [project.optional-dependencies]
            dev = [{deps_block}]
            """).strip()
    )


class TestGetRuffPin:
    def test_returns_exact_pinned_version(
        self, ruff_pin_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_pyproject(tmp_path, ["pytest>=9.0.3", "ruff==0.15.12", "bandit>=1.7.0"])
        monkeypatch.setattr(
            ruff_pin_module, "PYPROJECT_PATH", tmp_path / "pyproject.toml"
        )

        assert ruff_pin_module.get_ruff_pin() == "0.15.12"

    def test_loose_spec_raises(
        self, ruff_pin_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A loose ``ruff>=...`` spec is exactly the regression this guards
        against (PRs #1540 blocked by unpinned latest) — must fail loudly,
        not silently fall back to installing whatever's newest."""
        _write_pyproject(tmp_path, ["pytest>=9.0.3", "ruff>=0.1.0"])
        monkeypatch.setattr(
            ruff_pin_module, "PYPROJECT_PATH", tmp_path / "pyproject.toml"
        )

        with pytest.raises(ValueError, match="loose spec"):
            ruff_pin_module.get_ruff_pin()

    def test_missing_ruff_entry_raises(
        self, ruff_pin_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_pyproject(tmp_path, ["pytest>=9.0.3"])
        monkeypatch.setattr(
            ruff_pin_module, "PYPROJECT_PATH", tmp_path / "pyproject.toml"
        )

        with pytest.raises(ValueError, match="no 'ruff' entry"):
            ruff_pin_module.get_ruff_pin()

    def test_main_prints_version_and_returns_zero(
        self,
        ruff_pin_module,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_pyproject(tmp_path, ["ruff==0.15.12"])
        monkeypatch.setattr(
            ruff_pin_module, "PYPROJECT_PATH", tmp_path / "pyproject.toml"
        )

        rc = ruff_pin_module.main()
        captured = capsys.readouterr()

        assert rc == 0
        assert captured.out.strip() == "0.15.12"

    def test_main_returns_nonzero_on_missing_pin(
        self,
        ruff_pin_module,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_pyproject(tmp_path, ["pytest>=9.0.3"])
        monkeypatch.setattr(
            ruff_pin_module, "PYPROJECT_PATH", tmp_path / "pyproject.toml"
        )

        rc = ruff_pin_module.main()
        captured = capsys.readouterr()

        assert rc == 1
        assert "ruff" in captured.err


class TestRuffPinConsistencyAcrossRepo:
    """Real-repo checks (not synthetic fixtures) that the pin is actually
    wired up end-to-end, matching the guard-script convention used by
    ``test_check_dep_floor_drift.py``."""

    def test_real_pyproject_has_exact_ruff_pin(self, ruff_pin_module) -> None:
        # Exercises the real, checked-in pyproject.toml (module-level
        # PYPROJECT_PATH default, not monkeypatched) — this is the actual
        # value the CI preflight job will install.
        assert ruff_pin_module.get_ruff_pin()  # raises if not an exact pin

    def test_precommit_ruff_rev_matches_pyproject_pin(self, ruff_pin_module) -> None:
        """.pre-commit-config.yaml's ruff-pre-commit rev must track the same
        exact version as pyproject.toml's dev extra, or local pre-commit and
        CI preflight silently disagree on which ruff lints your diff."""
        pinned_version = ruff_pin_module.get_ruff_pin()

        precommit_config = yaml.safe_load(
            (_REPO_ROOT / ".pre-commit-config.yaml").read_text()
        )
        ruff_repo = next(
            repo
            for repo in precommit_config["repos"]
            if repo["repo"] == "https://github.com/astral-sh/ruff-pre-commit"
        )
        rev = ruff_repo["rev"].lstrip("v")

        assert rev == pinned_version

    def test_preflight_job_does_not_install_unpinned_ruff(self) -> None:
        """Regression guard for the exact bug in issue #1550: preflight must
        never fall back to a bare, unpinned `pip install ruff`."""
        workflow = yaml.safe_load(
            (_REPO_ROOT / ".github" / "workflows" / "pr-gate.yml").read_text()
        )
        preflight_steps = workflow["jobs"]["preflight"]["steps"]
        install_step = next(
            step for step in preflight_steps if step.get("name") == "Install ruff"
        )
        run_lines = [line.strip() for line in install_step["run"].splitlines()]

        assert "pip install ruff" not in run_lines
        assert any("get_ruff_pin.py" in line for line in run_lines)
