"""Developer setup scripts must point at files that exist (#2996).

install-dev.sh and the Playground launchers named paths that are not in the
repository and a Python floor below the one the package declares.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_INSTALL_DEV = _REPO / "scripts" / "setup" / "install-dev.sh"
_LAUNCH_PY = _REPO / "scripts" / "examples" / "launch_control_center.py"
_LAUNCH_SH = _REPO / "scripts" / "examples" / "launch_control_center.sh"

# Repo-relative file paths mentioned in a script (docs/x.md, scripts/a/b.py, ...).
_PATH_RE = re.compile(
    r"(?<![\w/.-])((?:docs|examples|scripts|plugins)/[\w./-]+\.\w+|[A-Z]+\.md|hello_world\.py)"
)


def test_install_dev_mentions_only_existing_files() -> None:
    text = _INSTALL_DEV.read_text(encoding="utf-8")
    missing = sorted({p for p in _PATH_RE.findall(text) if not (_REPO / p).exists()})
    assert missing == []


def test_install_dev_python_floor_matches_pyproject() -> None:
    declared = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["requires-python"]
    floor = re.search(r"(\d+\.\d+)", declared).group(1)
    script = re.search(
        r'required_version="([^"]+)"', _INSTALL_DEV.read_text(encoding="utf-8")
    ).group(1)
    assert script == floor


def test_python_launcher_app_path_exists() -> None:
    spec = importlib.util.spec_from_file_location("_launch_cc", _LAUNCH_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.APP_PATH.is_file()


def test_shell_launcher_app_path_exists() -> None:
    text = _LAUNCH_SH.read_text(encoding="utf-8")
    target = re.search(r'streamlit run "\$REPO_ROOT/([^"]+)"', text)
    assert target is not None, "launcher must run the app relative to the repo root"
    assert (_REPO / target.group(1)).is_file()


_README = _REPO / "scripts" / "README.md"
# A scripts/ path, or a bold backticked file name such as **`tool.py`**.
_README_PATH_RE = re.compile(r"(?<![\w/.-])(scripts/[\w./-]*[\w/])")
_README_NAME_RE = re.compile(r"\*\*`([\w.-]+\.\w+)`\*\*|^#### `([\w.-]+\.\w+)`", re.M)
# Written by the maintenance tools at run time, not shipped in the repo.
_GENERATED = {"scripts/maintenance/logs"}


def test_scripts_readme_paths_exist() -> None:
    text = _README.read_text(encoding="utf-8")
    missing = sorted(
        {
            p.rstrip("/.")
            for p in _README_PATH_RE.findall(text)
            if p.rstrip("/.") not in _GENERATED
            and not (_REPO / p.rstrip("/.")).exists()
        }
    )
    assert missing == []


def test_scripts_readme_named_files_exist() -> None:
    text = _README.read_text(encoding="utf-8")
    names = {a or b for a, b in _README_NAME_RE.findall(text)}
    scripts = _REPO / "scripts"
    missing = sorted(n for n in names if next(scripts.rglob(n), None) is None)
    assert missing == []
