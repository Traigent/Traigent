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
        f"{name} = [{', '.join(repr(s) for s in specs)}]"
        for name, specs in extras_by_name
    )
    (root / "pyproject.toml").write_text(
        textwrap.dedent(f"""
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
            """).strip()
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
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\n"
    )

    assert module.main() == 0


def test_requirements_ahead_of_pyproject_is_clean(drift_module) -> None:
    """A requirements file tighter than pyproject is safer than required, not drift."""
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.8\naiohttp>=3.13.4\nlangchain-core>=1.2.11\nrank-bm25\n"
    )

    assert module.main() == 0


def test_core_dependency_absent_from_requirements_is_a_finding(drift_module) -> None:
    """A core floor missing from the mirror is the worst case, not an exemption.

    This test previously asserted the opposite -- that a pyproject-only
    dependency "doesn't require mention in requirements/*.txt" -- and that
    assumption was the blind spot. The floor-comparison loop iterates the
    requirements file and looks each name up in pyproject, so a package present
    in pyproject and **absent** from the mirror was never examined at all.

    Absence is strictly worse than drift: a lagging floor still bounds the
    resolver somewhere, while an absent one bounds it nowhere. Found on PR
    #2210, where newly-declared ``yarl``/``filelock`` floors left this script
    green, and ``anyio``/``requests``/``PyJWT``/``pydantic`` proved to have been
    missing from the mirror for far longer.
    """
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\n"  # matches pyproject; aiohttp + langchain-core absent
    )

    assert module.main() == 1


def test_core_dependency_listed_without_a_floor_is_a_finding(drift_module) -> None:
    """A bare name in the mirror is presence, not protection.

    The first version of this check treated any package-shaped line as
    satisfying the requirement, so ``langchain-core`` with no specifier passed
    while ``pip install -r requirements/requirements.txt`` applied no floor for
    it -- reproducing, inside the guard, the exact defect the guard exists to
    catch. Caught in review round 2 of PR #2210.
    """
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core\n"
    )

    assert module.main() == 1


def test_core_dependency_unfloored_in_pyproject_needs_no_mirror_floor(
    drift_module,
) -> None:
    """``rank-bm25`` carries no floor in pyproject, so there is none to enforce.

    This is the case the previous test's docstring described but its body did
    not exercise -- it wrote a *floored* package instead.
    """
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\nrank-bm25\n"
    )

    assert module.main() == 0


def test_environment_marker_does_not_disguise_a_missing_floor(drift_module) -> None:
    """``langchain-core; python_version >= "3.11"`` is still an unfloored line."""
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\n"
        'langchain-core; python_version >= "3.11"\n'
    )

    assert module.main() == 1


def test_floor_behind_an_environment_marker_is_a_finding(drift_module) -> None:
    """A conditional floor is not an unconditional one.

    ``langchain-core>=1.2.11; python_version < "3.11"`` looks pinned and reads
    as pinned, but pip applies it only where the marker holds. When pyproject
    declares the dependency unconditionally -- and especially when the project's
    own ``requires-python`` excludes the marker's range -- such a line protects
    nothing while defeating a presence-and-floor check. Found in review round 3
    of PR #2210, after rounds 1 and 2 had each found a different hole in this
    same guard.
    """
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\n"
        'langchain-core>=1.2.11; python_version < "3.11"\n'
    )

    assert module.main() == 1


def test_unconditional_floor_alongside_a_conditional_one_is_clean(
    drift_module,
) -> None:
    """One applicable floor is enough; a conditional duplicate does not taint it."""
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\n"
        'langchain-core>=1.2.11; python_version < "3.11"\n'
        "langchain-core>=1.2.11\n"
    )

    assert module.main() == 0


def test_missing_core_requirements_file_is_a_finding(drift_module) -> None:
    """An absent mirror must fail, not silently pass.

    ``_find_unprotected_core_floors`` used to return ``[]`` when
    ``requirements/requirements.txt`` did not exist, so deleting the file made
    the check green -- an absent-input false green. Caught in review round 2 of
    PR #2210.
    """
    module, root = drift_module
    _write_pyproject(root)
    # requirements/ exists (fixture) but the core mirror does not.

    assert module.main() == 1


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


def test_extra_floor_absent_from_its_mirror_file_is_a_finding(drift_module) -> None:
    """An extra floor missing from its own requirements-<extra>.txt is a finding.

    This is issue #2211: check 3 only ever looked at ``[project.dependencies]``,
    so a package declared only under an extra (e.g. ``pyotp`` under
    ``security``) could drift out of its mirror file with nothing catching it.
    """
    module, root = drift_module
    _write_pyproject(
        root,
        ("security", ["pyotp>=2.9.0", "redis>=4.0.0"]),
    )
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\n"
    )
    (root / "requirements" / "requirements-security.txt").write_text(
        "-r requirements.txt\nredis>=4.0.0\n"  # pyotp missing
    )

    rc = module.main()
    assert rc == 1


def test_extra_with_no_mirror_file_is_not_checked(drift_module) -> None:
    """An extra that has no requirements-<extra>.txt is out of check 4's scope.

    There is nothing to compare against, and inventing the file is a product
    decision (issue #2211), not this guard's job.
    """
    module, root = drift_module
    _write_pyproject(
        root,
        ("docs", ["mkdocs>=1.5.0"]),
    )
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\n"
    )

    assert module.main() == 0


def test_extra_floor_already_in_core_mirror_is_not_double_flagged(drift_module) -> None:
    """A dependency declared under an extra but already floored in the core
    mirror (over-inclusion, not drift) is not this check's problem.

    Mirrors the real ``hybrid`` extra's ``claude-code-sdk``/``mcp``: both are
    declared under ``optional-dependencies.hybrid`` in pyproject.toml but
    already floored directly in ``requirements/requirements.txt``, which every
    extra file pulls in via ``-r requirements.txt``.
    """
    module, root = drift_module
    _write_pyproject(
        root,
        ("hybrid", ["httpx>=0.24.0", "mcp>=1.28.1"]),
    )
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\nmcp>=1.28.1\n"
    )
    (root / "requirements" / "requirements-hybrid.txt").write_text(
        "-r requirements.txt\nhttpx>=0.24.0\n"  # mcp intentionally not repeated
    )

    assert module.main() == 0


def test_extra_floor_marker_matching_pyprojects_own_marker_is_clean(
    drift_module,
) -> None:
    """A mirror line carrying the exact marker pyproject itself used protects
    exactly what pyproject promised (e.g. real-world ``faiss-cpu; sys_platform
    != 'win32'`` under the ``integrations`` extra) and must not be flagged
    ``conditional``.
    """
    module, root = drift_module
    _write_pyproject(
        root,
        ("integrations", ["faiss-cpu>=1.7.0; sys_platform != 'win32'"]),
    )
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\n"
    )
    (root / "requirements" / "requirements-integrations.txt").write_text(
        "-r requirements.txt\nfaiss-cpu>=1.7.0; sys_platform != 'win32'\n"
    )

    assert module.main() == 0


def test_extra_floor_marker_narrower_than_pyprojects_is_a_finding(
    drift_module,
) -> None:
    """A mirror marker that is *not* the one pyproject declared still protects
    only some environments and must be flagged, even though pyproject's own
    declaration is itself conditional."""
    module, root = drift_module
    _write_pyproject(
        root,
        ("integrations", ["faiss-cpu>=1.7.0; sys_platform != 'win32'"]),
    )
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\n"
    )
    (root / "requirements" / "requirements-integrations.txt").write_text(
        '-r requirements.txt\nfaiss-cpu>=1.7.0; python_version < "3.12"\n'
    )

    assert module.main() == 1


def test_ignores_loose_specs_without_floor(drift_module) -> None:
    """A bare ``package`` line carries no floor, so it contributes nothing to
    drift — don't false-positive on it."""
    module, root = drift_module
    _write_pyproject(root)
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\n"
        "rank-bm25\n"  # no pin
    )

    assert module.main() == 0


def test_ignores_option_lines(drift_module) -> None:
    """-r chained requirements and -e editable installs are options, not pins."""
    module, root = drift_module
    _write_pyproject(root)
    # The core mirror must exist and be complete, or its own check fires and
    # this test would pass/fail for a reason unrelated to option-line parsing.
    (root / "requirements" / "requirements.txt").write_text(
        "cryptography>=46.0.7\naiohttp>=3.13.4\nlangchain-core>=1.2.11\n"
    )
    (root / "requirements" / "requirements-security.txt").write_text(
        "-r requirements.txt\n-e .\ncryptography>=46.0.7\n"
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
