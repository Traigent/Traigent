from datetime import UTC, datetime

import yaml

from traigent.playbook import (
    STAGE_ORDER,
    Playbook,
    StageStatus,
    compute_staleness,
    scaffold_playbook,
    validate_playbook,
)
from traigent.playbook.model import Stage


def test_scaffold_validate_round_trip_has_zero_issues() -> None:
    payload = yaml.safe_load(
        scaffold_playbook(
            name="support-agent",
            agent_type="rag",
            entrypoint="agent:run",
        )
    )

    assert validate_playbook(payload) == []


def test_validator_reports_schema_bad_enum_issue() -> None:
    payload = yaml.safe_load(scaffold_playbook(name="support-agent"))
    payload["stages"]["dataset"]["status"] = "done"

    issues = validate_playbook(payload)

    assert len(issues) == 1
    assert issues[0].location == "$.stages.dataset.status"
    assert "'done' is not one of" in issues[0].message


def test_validator_reports_pinned_stage_without_pin() -> None:
    payload = yaml.safe_load(scaffold_playbook(name="support-agent"))
    payload["stages"]["dataset"]["status"] = "pinned"

    issues = validate_playbook(payload)

    assert [(issue.location, issue.message) for issue in issues] == [
        ("$.stages.dataset", "pinned stages must include pin")
    ]


def test_validator_reports_deprecated_stage_without_reason() -> None:
    payload = yaml.safe_load(scaffold_playbook(name="support-agent"))
    payload["stages"]["metric"]["status"] = "deprecated"

    issues = validate_playbook(payload)

    assert [(issue.location, issue.message) for issue in issues] == [
        ("$.stages.metric", "deprecated stages must include deprecation_reason")
    ]


def test_validator_reports_unparseable_pinned_at() -> None:
    payload = yaml.safe_load(scaffold_playbook(name="support-agent"))
    payload["stages"]["dataset"]["pinned_at"] = "2026-01-01"

    issues = validate_playbook(payload)

    assert [(issue.location, issue.message) for issue in issues] == [
        ("$.stages.dataset.pinned_at", "pinned_at must be a parseable ISO datetime")
    ]


def test_staleness_marks_later_stage_stale_when_earlier_repin_is_later() -> None:
    playbook = Playbook(
        playbook_version="1.0.0",
        agent={"name": "support-agent"},
        stages={
            "dataset": Stage(
                status=StageStatus.PINNED,
                pinned_at=datetime(2026, 2, 2, tzinfo=UTC),
                pin={"dataset_ref": "eval.jsonl"},
            ),
            "metric": Stage(
                status=StageStatus.PINNED,
                pinned_at=datetime(2026, 2, 1, tzinfo=UTC),
                pin={"metric_name": "accuracy"},
            ),
            "evaluator": Stage(status=StageStatus.PENDING),
            "optimize": Stage(
                status=StageStatus.PINNED,
                pinned_at=None,
                pin={"objectives": ["accuracy"]},
            ),
        },
        provenance=None,
        raw={},
    )

    stale = compute_staleness(playbook)

    assert stale["dataset"] is False
    assert stale["metric"] is True
    assert stale["optimize"] is False
    assert set(stale) == set(STAGE_ORDER)
