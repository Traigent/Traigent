"""Tests for scripts/ci/promotion_closes_block.py pure helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "ci" / "promotion_closes_block.py"
)
_spec = importlib.util.spec_from_file_location("promotion_closes_block", _PATH)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


@pytest.mark.parametrize(
    "text",
    [
        "Closes #5",
        "closes #5",
        "CLOSED #5",
        "Close #5",
        "Fixes #5",
        "fix #5",
        "Fixed #5",
        "Resolves #5",
        "resolve #5",
        "Resolved #5",
        "Fixes: #5",
        "Closes https://github.com/Traigent/Traigent/issues/5",
    ],
)
def test_accepted_keyword_forms(text: str) -> None:
    refs = mod.extract_closing_refs(text)
    assert [n for _, n in refs] == [5]


def test_cross_repo_ref_and_rejects() -> None:
    assert mod.extract_closing_refs("Fixes Org/Repo#7") == [("org/repo", 7)]
    for bad in ["Prefixes #5", "see #5", "Closes 5", "Closes #5abc", "unfixes #5"]:
        assert mod.extract_closing_refs(bad) == [], bad


def test_one_keyword_closes_one_issue() -> None:
    assert mod.extract_closing_refs("Closes #1, #2") == [("", 1)]


def test_ignores_code_fences_and_comments() -> None:
    text = "Closes #1\n```\nCloses #2\n```\n<!-- Closes #3 -->\nFixes #4"
    assert [n for _, n in mod.extract_closing_refs(text)] == [1, 4]


def test_block_dedupes_and_sorts() -> None:
    texts = ["Closes #30\nFixes #4", "resolves #4", "Closes Other/Repo#2", "Closes #12"]
    assert mod.build_closes_block(texts) == [
        "Closes #4",
        "Closes #12",
        "Closes #30",
        "Closes other/repo#2",
    ]


def test_block_exclude_and_repo_shortening() -> None:
    texts = ["Closes Traigent/Traigent#9\nCloses #9\nCloses #10"]
    assert mod.build_closes_block(texts, repo="Traigent/Traigent") == [
        "Closes #9",
        "Closes #10",
    ]
    assert mod.build_closes_block(texts, repo="Traigent/Traigent", exclude=[9]) == [
        "Closes #10"
    ]


def test_empty_input() -> None:
    assert mod.build_closes_block([]) == []
    assert mod.extract_closing_refs("") == []


def test_pr_numbers_from_subjects() -> None:
    msgs = ["feat: x (#12)\n\nbody (#99)", "Merge pull request #7 from a/b", "plain"]
    assert mod.pr_numbers(msgs) == [7, 12]


def test_offline_git_range(tmp_path: Path) -> None:
    def git(*a: str) -> None:
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                *a,
            ],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("commit", "--allow-empty", "-q", "-m", "base Fixes #1")
    git("tag", "base")
    git("commit", "--allow-empty", "-q", "-m", "feat: a (#2)\n\nCloses #8\nFixes #3")
    git("commit", "--allow-empty", "-q", "-m", "fix: b\n\nresolves #8")
    out = subprocess.run(
        [sys.executable, str(_PATH), "base..HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.stdout.splitlines() == ["Closes #3", "Closes #8"]
