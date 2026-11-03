"""Environment for test helpers that run ``git`` inside a throwaway repository.

Under a git hook (pre-commit, pre-push) git exports ``GIT_DIR`` and friends pointing at the REAL
repository. A helper that runs ``git config user.email ...`` or ``git commit`` inside a tmp repo
then acts on the real repo instead, rewriting its shared config. Drop every variable that
redirects git before running such a helper.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

REPO_REDIRECT_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    # ``GIT_CONFIG`` makes ``git config`` write to that file; the others inject configuration.
    "GIT_CONFIG",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
)
_REDIRECT_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")


def scrubbed_git_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """``base`` (default: the process environment) without the repo-redirecting variables."""
    source = os.environ if base is None else base
    return {
        k: v
        for k, v in source.items()
        if k not in REPO_REDIRECT_VARS and not k.startswith(_REDIRECT_PREFIXES)
    }
