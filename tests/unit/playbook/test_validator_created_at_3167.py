"""provenance.created_at is checked as a date-time like pinned_at (Traigent#3167)."""

from __future__ import annotations

import yaml

from traigent.playbook import scaffold_playbook, validate_playbook


def _payload() -> dict:
    payload = yaml.safe_load(scaffold_playbook(name="support-agent"))
    payload["provenance"] = {"created_by": "x", "created_at": "2026-01-01T00:00:00Z"}
    return payload


def test_valid_created_at_is_accepted() -> None:
    assert validate_playbook(_payload()) == []


def test_junk_created_at_is_reported() -> None:
    payload = _payload()
    payload["provenance"]["created_at"] = "not-a-date"

    issues = validate_playbook(payload)

    assert [(i.location, i.message) for i in issues] == [
        ("$.provenance.created_at", "created_at must be a parseable ISO datetime")
    ]
