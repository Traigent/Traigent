"""Validation for agent build playbooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from traigent.playbook.model import Playbook

_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schemas" / "agent_playbook_schema.json"
)


@dataclass(frozen=True)
class ValidationIssue:
    """A validation issue with a JSON-path-ish location."""

    location: str
    message: str


def validate_playbook(
    playbook_or_dict: Playbook | dict[str, Any],
) -> list[ValidationIssue]:
    """Return all schema and semantic validation issues for a playbook."""
    payload = _as_payload(playbook_or_dict)
    issues = _schema_issues(payload)
    if isinstance(payload, dict):
        issues.extend(_semantic_issues(payload))
    return issues


def _as_payload(playbook_or_dict: Playbook | dict[str, Any]) -> Any:
    if isinstance(playbook_or_dict, Playbook):
        return playbook_or_dict.raw
    return playbook_or_dict


def _schema_issues(payload: Any) -> list[ValidationIssue]:
    validator = Draft7Validator(_load_schema())
    return [
        ValidationIssue(location=_json_path(error.path), message=error.message)
        for error in sorted(
            validator.iter_errors(payload), key=lambda item: list(item.path)
        )
    ]


def _semantic_issues(payload: dict[str, Any]) -> list[ValidationIssue]:
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        return []

    issues: list[ValidationIssue] = []
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            continue

        location = f"$.stages.{stage_name}"
        status = stage.get("status")
        if status == "pinned" and "pin" not in stage:
            issues.append(
                ValidationIssue(
                    location=location,
                    message="pinned stages must include pin",
                )
            )
        if (
            status == "deprecated"
            and not str(stage.get("deprecation_reason", "")).strip()
        ):
            issues.append(
                ValidationIssue(
                    location=location,
                    message="deprecated stages must include deprecation_reason",
                )
            )
        if "pinned_at" in stage and not _is_parseable_datetime(stage.get("pinned_at")):
            issues.append(
                ValidationIssue(
                    location=f"{location}.pinned_at",
                    message="pinned_at must be a parseable ISO datetime",
                )
            )
    return issues


def _load_schema() -> dict[str, Any]:
    with _SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    if not isinstance(schema, dict):
        raise ValueError("agent build playbook schema must be a JSON object")
    return schema


def _json_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _is_parseable_datetime(value: Any) -> bool:
    if isinstance(value, datetime):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or not _has_datetime_separator(text):
        return False
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def _has_datetime_separator(value: str) -> bool:
    return "T" in value or " " in value
