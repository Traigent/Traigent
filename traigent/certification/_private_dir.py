"""Create a private (0700) directory without touching pre-existing ones."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_private_dir(directory: Path) -> None:
    """Create ``directory`` (and parents) and chmod it 0700 only if *we* created it.

    ``mkdir`` without ``exist_ok`` is the creation witness, so a directory
    that appears concurrently, or already exists (cwd, $HOME, a shared
    folder), keeps the caller's permissions. Files inside stay 0600.
    """
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        if not directory.is_dir():
            raise
        return
    except FileNotFoundError:
        directory.parent.mkdir(parents=True, exist_ok=True)
        ensure_private_dir(directory)
        return
    os.chmod(directory, 0o700)
