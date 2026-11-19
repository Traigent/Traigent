"""``traigent local show`` prints the best score and no baseline-dependent line.

The SDK never records a baseline score for a local session, so ``show`` must
not render an "Improvement" line, in either view.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.local_commands import local_commands
from traigent.storage.local_storage import LocalStorageManager


def _env(root: Path) -> dict[str, str]:
    return {
        "TRAIGENT_RESULTS_FOLDER": str(root),
        "TRAIGENT_MINIMAL_LOGGING": "true",
        "TRAIGENT_OFFLINE": "true",
        "TRAIGENT_OFFLINE_MODE": "true",
        "TRAIGENT_ENABLE_USAGE_ANALYTICS": "false",
    }


@pytest.mark.parametrize("fmt", ["summary", "detailed"])
def test_show_prints_best_score_without_improvement(tmp_path: Path, fmt: str) -> None:
    storage = LocalStorageManager(str(tmp_path))
    session_id = storage.create_session("show_fn")
    storage.add_trial_result(session_id, {"x": 1}, 0.6)
    storage.add_trial_result(session_id, {"x": 2}, 0.9)
    storage.finalize_session(session_id, "completed")

    result = CliRunner().invoke(
        local_commands,
        ["show", session_id, "--format", fmt],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "Best Score: 0.9000" in result.output
    assert "Improvement" not in result.output
