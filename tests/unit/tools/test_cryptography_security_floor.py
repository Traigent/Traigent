"""Regression guards for the cryptography floor and MLflow packaging split."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CRYPTOGRAPHY_FLOOR = Version("50.0.0")
_BARE_MLFLOW_INSTALL_PATTERNS = (
    re.compile(
        rb"\b(?:python(?:3)?\s+-m\s+pip|pip3?|uv\s+pip)\s+install\b"
        rb"[^\r\n#;]*?(?<![\w-])mlflow(?!-skinny|[\w-])",
        re.IGNORECASE,
    ),
    re.compile(
        rb"\binstall(?:\s+the)?\s+optional\s+`?mlflow`?\s+"
        rb"(?:dependency|package)\b",
        re.IGNORECASE,
    ),
)


def _requirements(path: Path) -> dict[str, Requirement]:
    parsed = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-r ")):
            continue
        requirement = Requirement(line)
        parsed[canonicalize_name(requirement.name)] = requirement
    return parsed


def _assert_cryptography_floor(requirement: Requirement, location: str) -> None:
    admission_message = f"{location} must admit cryptography {_CRYPTOGRAPHY_FLOOR}"
    assert _CRYPTOGRAPHY_FLOOR in requirement.specifier, admission_message

    rejection_message = f"{location} must reject cryptography 49.0.0"
    assert Version("49.0.0") not in requirement.specifier, rejection_message


def test_published_cryptography_specs_enforce_security_floor() -> None:
    """Core metadata and legacy requirement files reject pre-fix releases."""
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"]
    core = {
        canonicalize_name(requirement.name): requirement
        for dependency in project["dependencies"]
        if canonicalize_name((requirement := Requirement(dependency)).name)
        == "cryptography"
    }
    _assert_cryptography_floor(core["cryptography"], "project.dependencies")

    for relative_path in (
        "requirements/requirements.txt",
        "requirements/requirements-security.txt",
    ):
        requirement = _requirements(_REPO_ROOT / relative_path)["cryptography"]
        _assert_cryptography_floor(requirement, relative_path)


def test_lock_resolves_cryptography_at_or_above_security_floor() -> None:
    """The generated lock selects a cryptography artifact containing the fix."""
    lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())
    matches = [
        package for package in lock["package"] if package["name"] == "cryptography"
    ]
    assert len(matches) == 1
    assert Version(matches[0]["version"]) >= _CRYPTOGRAPHY_FLOOR


def test_integrations_publish_skinny_mlflow_and_explicit_pandas() -> None:
    """The integrations surface keeps tracking imports and DataFrame support."""
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"]
    integrations = {
        canonicalize_name(requirement.name): requirement
        for dependency in project["optional-dependencies"]["integrations"]
        if (requirement := Requirement(dependency))
    }
    legacy = _requirements(_REPO_ROOT / "requirements/requirements-integrations.txt")

    for location, requirements in (
        ("project.optional-dependencies.integrations", integrations),
        ("requirements/requirements-integrations.txt", legacy),
    ):
        assert "mlflow" not in requirements, f"full mlflow remains in {location}"
        assert "mlflow-skinny" in requirements, f"mlflow-skinny missing from {location}"
        assert "pandas" in requirements, f"explicit pandas missing from {location}"


def test_lock_contains_skinny_mlflow_but_not_full_mlflow() -> None:
    """The resolved graph no longer carries full MLflow's cryptography ceiling."""
    lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())
    names = {package["name"] for package in lock["package"]}
    assert "mlflow-skinny" in names
    assert "mlflow" not in names


def test_shipped_guidance_has_no_bare_full_mlflow_install_instruction() -> None:
    """Shipped guidance must not reopen full MLflow's cryptography conflict."""
    shipped_root = _REPO_ROOT / "traigent"
    offenders = []
    for path in shipped_root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        contents = path.read_bytes()
        if any(pattern.search(contents) for pattern in _BARE_MLFLOW_INSTALL_PATTERNS):
            offenders.append(path.relative_to(_REPO_ROOT).as_posix())

    assert not offenders, (
        "bare full-MLflow install instructions remain in shipped surfaces: "
        + ", ".join(offenders)
    )


def test_shipped_mlflow_example_explicitly_opts_into_file_store() -> None:
    """The skinny-only example must not rely on MLflow's local default."""
    skill = (
        _REPO_ROOT / "traigent" / "skills" / "traigent-integrations" / "SKILL.md"
    ).read_text()
    tracker_source = (
        _REPO_ROOT / "traigent" / "integrations" / "observability" / "mlflow.py"
    ).read_text()

    assert 'os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"' in skill
    assert "mlflow.set_tracking_uri(tracking_dir.as_uri())" in skill
    assert 'mlflow.set_experiment("traigent_optimization")' in skill
    assert "defaults to local" not in tracker_source


_FIRST_VERSIONED_HEADING = re.compile(r"^## \[\d+\.\d+\.\d+\]", re.MULTILINE)


def test_current_release_records_cryptography_advisory_closure() -> None:
    """The dependency remediation belongs to a released section, not Unreleased.

    Split at the first *versioned* heading rather than a literal version string:
    cutting a release renames that heading (``## [0.27.0] - unreleased`` becomes
    ``## [0.27.0] - 2026-08-22``), and a guard that hard-codes the pre-release
    spelling stops guarding the moment it would matter most. Everything above the
    first versioned heading is the ``## [Unreleased]`` staging area; everything
    from it down has shipped or is shipping now.
    """
    changelog = (_REPO_ROOT / "CHANGELOG.md").read_text()
    boundary = _FIRST_VERSIONED_HEADING.search(changelog)
    assert boundary is not None, "CHANGELOG.md has no versioned release section"
    unreleased = changelog[: boundary.start()]
    current_release = changelog[boundary.start() :]
    current_release_lower = current_release.lower()

    assert "mlflow-skinny" not in unreleased
    assert "cryptography 49.0.0" in current_release_lower
    assert "cryptography 50.0.1" in current_release_lower
    assert "CVE-2026-69247" in current_release
    assert "GHSA-g6cj-pr64-35w5" in current_release
