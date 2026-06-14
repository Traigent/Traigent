"""Load agent build playbooks from YAML."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from traigent.playbook.model import Playbook, Stage, StageStatus

DEFAULT_PLAYBOOK_FILENAME = "traigent.playbook.yaml"


def load_playbook(path: str | Path) -> Playbook:
    """Load a playbook YAML file into a parsed ``Playbook``."""
    payload = _load_yaml_mapping(path)
    return _playbook_from_mapping(payload)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    playbook_path = Path(path)
    if not playbook_path.exists():
        raise FileNotFoundError(f"agent build playbook not found: {playbook_path}")

    with playbook_path.open(encoding="utf-8") as playbook_file:
        payload = yaml.safe_load(playbook_file)

    if not isinstance(payload, dict):
        raise ValueError("agent build playbook YAML must be a mapping")
    return payload


def _playbook_from_mapping(payload: dict[str, Any]) -> Playbook:
    stages_payload = payload.get("stages", {})
    if not isinstance(stages_payload, dict):
        stages_payload = {}

    stages = {
        str(stage_name): _stage_from_mapping(stage_payload)
        for stage_name, stage_payload in stages_payload.items()
        if isinstance(stage_payload, dict)
    }
    agent = payload.get("agent", {})
    provenance = payload.get("provenance")
    return Playbook(
        playbook_version=str(payload.get("playbook_version", "")),
        agent=dict(agent) if isinstance(agent, dict) else {},
        stages=stages,
        provenance=dict(provenance) if isinstance(provenance, dict) else None,
        raw=payload,
    )


def _stage_from_mapping(payload: dict[str, Any]) -> Stage:
    return Stage(
        status=StageStatus(payload.get("status", StageStatus.PENDING.value)),
        pinned_at=_parse_datetime_value(payload.get("pinned_at")),
        deprecation_reason=_optional_string(payload.get("deprecation_reason")),
        pin=dict(payload["pin"]) if isinstance(payload.get("pin"), dict) else None,
    )


def _parse_datetime_value(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not _has_datetime_separator(text):
            raise ValueError("pinned_at must be an ISO datetime string")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text)
    raise ValueError(
        f"pinned_at must be an ISO datetime string, got {type(value).__name__}"
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _has_datetime_separator(value: str) -> bool:
    return "T" in value or " " in value
