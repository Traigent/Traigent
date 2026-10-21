"""Offline guard against dead links on the SDK's published surfaces (#2598).

Scans package metadata, the docs-site config and pages, example/plugin READMEs
and string literals in the runtime package. Fails on:

* hosts that do not serve Traigent documentation (NXDOMAIN, parked, or a
  marketing-page fallback), and
* ``github.com/Traigent/<repo>`` links to any repository that is not in
  ``PUBLIC_GITHUB_REPOS`` (verified public with an anonymous request).

To add a GitHub repository link, first confirm the repository opens without
credentials, then add its name to ``PUBLIC_GITHUB_REPOS``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Repositories under github.com/Traigent that return 200 to an anonymous request.
# Lowercase; GitHub owner/repo names are case-insensitive.
PUBLIC_GITHUB_REPOS = frozenset(
    {"traigent", "traigent-skills", "traigentschema", "tvl"}
)

DEAD_HOST_PATTERNS = (
    re.compile(r"docs\.traigent\.(?:ai|com|io)", re.IGNORECASE),
    re.compile(r"community\.traigent\.ai", re.IGNORECASE),
    re.compile(r"(?<![\w.-])(?:[\w-]+\.)*traigent\.com\b", re.IGNORECASE),
    re.compile(r"(?<![\w.-])(?:www\.)?traigent\.ai/docs\b", re.IGNORECASE),
)
GITHUB_REPO_LINK = re.compile(r"github\.com/Traigent/([A-Za-z0-9._-]+)", re.IGNORECASE)

TEXT_GLOBS = (
    "pyproject.toml",
    "mkdocs.yml",
    "README.md",
    "CHANGELOG.md",
    "docs/**/*.md",
    "walkthrough/**/*.md",
    "examples/**/README.md",
    "plugins/**/pyproject.toml",
    "plugins/**/README.md",
)


def _text_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in TEXT_GLOBS:
        files.update(p for p in REPO_ROOT.glob(pattern) if p.is_file())
    return sorted(files)


def _python_string_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _problems(text: str) -> list[str]:
    found = [m.group(0) for p in DEAD_HOST_PATTERNS for m in p.finditer(text)]
    for match in GITHUB_REPO_LINK.finditer(text):
        repo = match.group(1).lower().removesuffix(".git")
        if repo not in PUBLIC_GITHUB_REPOS:
            found.append(match.group(0))
    return found


def _scan() -> list[str]:
    hits: list[str] = []
    for path in _text_files():
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            hits.extend(f"{rel}:{lineno}: {bad}" for bad in _problems(line))
    for path in sorted((REPO_ROOT / "traigent").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        for lineno, value in _python_string_literals(path):
            hits.extend(f"{rel}:{lineno}: {bad}" for bad in _problems(value))
    return hits


def test_published_surfaces_have_no_dead_or_private_links() -> None:
    hits = _scan()
    assert not hits, "Dead or non-public links on published surfaces:\n" + "\n".join(
        hits
    )


@pytest.mark.parametrize(
    "text",
    [
        "https://docs.traigent.ai",
        "https://docs.traigent.com/security/api-keys",
        "https://docs.traigent.io/plugins/tracing",
        "https://community.traigent.ai",
        "https://traigent.com/lander",
        "https://traigent.ai/docs",
        "https://github.com/Traigent/not-an-allowlisted-repo",
    ],
)
def test_detector_flags_known_bad_shapes(text: str) -> None:
    assert _problems(text)


@pytest.mark.parametrize(
    "text",
    [
        "https://traigent.github.io/Traigent/guides/secrets_management/",
        "https://github.com/Traigent/Traigent/issues",
        "https://github.com/traigent/traigent",
        "git+https://github.com/Traigent/TraigentSchema.git@abc",
        "https://portal.traigent.ai",
        "https://traigent.ai/pricing",
    ],
)
def test_detector_allows_live_targets(text: str) -> None:
    assert not _problems(text)
