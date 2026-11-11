"""No tracked file may match a .gitignore rule (Traigent#3179).

A tracked-but-ignored file hides its siblings from a plain ``git add`` and is
lost by clean-and-regenerate cycles. Either negate the rule or untrack the file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests._git_isolation import scrubbed_git_env

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_tracked_file_matches_gitignore() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-ci", "--exclude-standard"],
        cwd=_REPO_ROOT,
        env=scrubbed_git_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    offenders = result.stdout.split()
    assert not offenders, (
        "tracked files match .gitignore rules (add a '!path' negation next to "
        "the rule, or `git rm --cached` them):\n" + "\n".join(offenders)
    )
