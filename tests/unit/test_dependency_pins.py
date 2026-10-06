"""Regression test for dependency specifier ranges.

Guards that the litellm specifier in pyproject.toml admits the demo-required
version 1.88.0 (and beyond), not just the baseline 1.87.1. Before the fix
for GitHub issue #1418, the spec was ``litellm==1.87.1``, which caused:
  - customer installs to conflict with the demo (which needs 1.88.0)
  - the pyproject to be unduly rigid for any minor litellm release

This test FAILS on the pre-fix ``==1.87.1`` pin and PASSES on the loosened
``>=1.87.1,<2`` range.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet

# File is at tests/unit/test_dependency_pins.py
# parents[0] = tests/unit/, parents[1] = tests/, parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
QUICKSTART_MAIN_PATH = (
    _REPO_ROOT / "traigent" / "examples" / "quickstart" / "__main__.py"
)
REQUIREMENTS_CORE_PATH = _REPO_ROOT / "requirements" / "requirements.txt"


def _get_litellm_specifier_from_pyproject() -> str:
    """Extract the raw litellm specifier string from pyproject.toml."""
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    deps: list[str] = data["project"]["dependencies"]
    for dep in deps:
        normalized = (
            dep.strip()
            .lower()
            .split("[")[0]
            .split(">=")[0]
            .split("==")[0]
            .split(",")[0]
            .strip()
        )
        if normalized.rstrip() == "litellm":
            # Extract the specifier part: everything after "litellm"
            specifier = dep.strip()[len("litellm") :]
            # Strip trailing comments
            specifier = specifier.split("#")[0].strip()
            return specifier
    raise AssertionError("litellm not found in pyproject.toml [project.dependencies]")


def test_litellm_specifier_admits_later_1x_minor():
    """litellm specifier must stay a range admitting later 1.x minors (#1418).

    The security floor now sits above the old 1.88.0 demo requirement; the
    intent (a range, not an exact pin) is checked against a later minor.
    """
    raw_spec = _get_litellm_specifier_from_pyproject()
    spec_set = SpecifierSet(raw_spec)
    assert spec_set.contains("1.101.0"), (
        f"litellm specifier '{raw_spec}' does NOT admit 1.101.0. "
        "Keep a range ('>=<floor>,<2') rather than an exact pin (see GitHub issue #1418)."
    )


def test_litellm_specifier_admits_security_floor_1_100_4():
    """litellm specifier must admit the patched security floor 1.100.4."""
    raw_spec = _get_litellm_specifier_from_pyproject()
    spec_set = SpecifierSet(raw_spec)
    assert spec_set.contains("1.100.4"), (
        f"litellm specifier '{raw_spec}' does NOT admit 1.100.4, the patched security floor."
    )


def test_litellm_specifier_rejects_major_2():
    """litellm specifier should exclude major version 2 (unverified breaking changes)."""
    raw_spec = _get_litellm_specifier_from_pyproject()
    spec_set = SpecifierSet(raw_spec)
    assert not spec_set.contains("2.0.0"), (
        f"litellm specifier '{raw_spec}' admits 2.0.0 (a future major version). "
        "Add an upper bound '<2' to guard against unverified breaking changes."
    )


def test_litellm_specifier_is_not_exact_pin():
    """litellm specifier must not be an exact == pin, which breaks co-installation."""
    raw_spec = _get_litellm_specifier_from_pyproject()
    assert raw_spec.strip().startswith(">="), (
        f"litellm specifier '{raw_spec}' appears to be an exact pin (==). "
        "Library packages should use version ranges, not exact pins, "
        "so customers can co-install the SDK with other packages that need "
        "a different litellm minor (e.g. the demo needs 1.88.0). "
        "See GitHub issue #1418."
    )


def test_quickstart_install_hint_uses_range():
    """The quickstart error hint should NOT hardcode a specific exact version."""
    source = QUICKSTART_MAIN_PATH.read_text()
    assert "litellm==1.87.1" not in source, (
        "traigent/examples/quickstart/__main__.py still hardcodes 'litellm==1.87.1' "
        "in the install hint. Update it to match the pyproject.toml range "
        "(e.g. 'litellm>=1.100.4,<2'). See GitHub issue #1418."
    )


def test_requirements_txt_not_exact_litellm_pin():
    """requirements/requirements.txt must not exact-pin litellm."""
    requirements_text = REQUIREMENTS_CORE_PATH.read_text()
    assert "litellm==1.87.1" not in requirements_text, (
        "requirements/requirements.txt still has 'litellm==1.87.1'. "
        "Update it to match the pyproject.toml range (e.g. 'litellm>=1.100.4,<2'). "
        "See GitHub issue #1418."
    )
