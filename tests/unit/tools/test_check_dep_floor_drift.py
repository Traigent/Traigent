"""Tests for scripts/ci/check_dep_floor_drift.py.

Guards the guard: the CI hook that prevents pyproject.toml ↔ requirements/*.txt
dependency floor drift. Regressions here would silently re-open the
three-time Greptile-P1 drift class (CVEs pinned in pyproject but left
vulnerable in a requirements file that Docker layers resolve from).
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_dep_floor_drift.py"
)


@pytest.fixture
def drift_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Load the script with its module-level paths redirected at a tmp repo."""
    spec = importlib.util.spec_from_file_location("_drift_check", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    (tmp_path / "requirements").mkdir()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "PYPROJECT_PATH", tmp_path / "pyproject.toml")
    monkeypatch.setattr(module, "REQUIREMENTS_DIR", tmp_path / "requirements")

    return module, tmp_path


def _write_pyproject(root: Path, *extras_by_name: tuple[str, list[str]]) -> None:
    extras_block = "\n".join(
        f'{name} = [{", ".join(repr(s) for s in specs)}]' for name, specs in extras_by_name
    )
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "traigent"
            version = "0.0.0"
            dependencies = [
                "cryptography>=46.0.7",
                "aiohttp>=3.13.4,<4.0",
                "langchain-core>=1.2.11",
                "rank-bm25",
            ]

            [project.optional-dependencies]
            {extras_block}
            """
        ).strip()
    )


def test_clean_tree_returns_zero(drift_module) -> None:
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\n"
    )

    assert module.main() == 0


def test_requirements_lagging_behind_pyproject_is_detected(
    drift_module, capsys: pytest.CaptureFixture[str]
) -> None:
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=44.0.1\n"  # the exact drift from PR #730
        "aiohttp>=3.13.4\n"
    )

    rc = module.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "cryptography" in captured.err
    assert "44.0.1" in captured.err
    assert "46.0.7" in captured.err


def test_requirements_equal_to_pyproject_is_clean(drift_module) -> None:
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\n"
    )

    assert module.main() == 0


def test_requirements_ahead_of_pyproject_is_clean(drift_module) -> None:
    """A requirements file tighter than pyproject is safer than required, not drift."""
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.8\n"
    )

    assert module.main() == 0


def test_package_only_in_pyproject_is_ignored(drift_module) -> None:
    """We only compare packages that appear in both files. A pyproject-only
    dep doesn't require mention in requirements/*.txt."""
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\n"  # matches pyproject
        # aiohttp intentionally omitted — that's ok, means the requirements
        # file just doesn't pin it, not a drift.
    )

    assert module.main() == 0


def test_extra_dependencies_are_also_checked(drift_module) -> None:
    """Drifts against an optional-dependency extra (e.g. [integrations]) must
    also be caught — that's where the mlflow / langchain-core drift lived."""
    module, root = drift_module
    _write_pyproject(
        root,
        ("integrations", ["mlflow>=3.11.1", "openai>=2.0.0"]),
    )
    (root / "requirements" / "requirements-integrations.txt").write_text(
        "mlflow>=3.8.1\n"  # lag
        "openai>=2.0.0\n"  # aligned
    )

    rc = module.main()
    assert rc == 1


def test_ignores_loose_specs_without_floor(drift_module) -> None:
    """A bare ``package`` line carries no floor, so it contributes nothing to
    drift — don't false-positive on it."""
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\n"
        "rank-bm25\n"  # no pin
    )

    assert module.main() == 0


def test_ignores_option_lines(drift_module) -> None:
    """-r chained requirements and -e editable installs are options, not pins."""
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements-security.txt").write_text(
        "-r requirements.txt\n"
        "-e .\n"
        "cryptography>=46.0.7\n"
    )

    assert module.main() == 0


def test_pep503_normalization_matches_across_styles(drift_module) -> None:
    """lodash_es vs lodash-es etc. — normalized name should match so drift
    between pyproject ``langchain_core`` and requirements ``langchain-core``
    is still detected."""
    module, _ = drift_module
    assert module._normalize_name("Langchain_Core") == "langchain-core"
    assert module._normalize_name("langchain.core") == "langchain-core"
    assert module._normalize_name("langchain-core") == "langchain-core"
