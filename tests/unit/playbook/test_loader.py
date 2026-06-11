from pathlib import Path

import pytest

from traigent.playbook import StageStatus, load_playbook, scaffold_playbook


def test_load_playbook_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "traigent.playbook.yaml"
    path.write_text(
        scaffold_playbook(
            name="support-agent",
            agent_type="rag",
            entrypoint="agent:run",
        ),
        encoding="utf-8",
    )

    playbook = load_playbook(path)

    assert playbook.playbook_version == "1.0.0"
    assert playbook.agent == {
        "name": "support-agent",
        "entrypoint": "agent:run",
        "agent_type": "rag",
    }
    assert set(playbook.stages) == {
        "dataset",
        "metric",
        "evaluator",
        "optimize",
        "gate",
    }
    assert playbook.stages["dataset"].status is StageStatus.PENDING
    assert playbook.raw["agent"]["name"] == "support-agent"


def test_load_playbook_missing_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="agent build playbook not found"):
        load_playbook(tmp_path / "missing.yaml")
