"""The SDK declares its own identity in session metadata (read by the certification minter)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from traigent.core.backend_session_manager import BackendSessionManager
from traigent.utils import optimization_only_egress as oo

_SHA = "a" * 40


def _metadata(monkeypatch: pytest.MonkeyPatch, sdk_ref: str | None) -> dict[str, Any]:
    if sdk_ref is None:
        monkeypatch.delenv("TRAIGENT_SDK_REF", raising=False)
    else:
        monkeypatch.setenv("TRAIGENT_SDK_REF", sdk_ref)
    manager: Any = object.__new__(BackendSessionManager)
    manager._optimization_id = "opt-1"
    manager._local_sequencing_active = lambda: False
    arguments: dict[str, Any] = {
        "dataset": [1, 2],
        "function_descriptor": SimpleNamespace(
            identifier="fn", module="m", relative_path="m.py"
        ),
        "function_display_name": None,
        "function_slug": "fn",
        "evaluation_set_name": "es",
        "max_trials_value": 3,
        "max_samples_value": None,
        "agent_configuration": None,
        "warm_start_from": None,
    }
    result: dict[str, Any] = manager._build_session_create_metadata(**arguments)
    return result


def test_session_metadata_carries_sdk_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = _metadata(monkeypatch, _SHA)
    identity = metadata["sdk_identity"]
    assert identity["sdk_ref"] == _SHA
    assert identity["sdk_version"].count(".") == 2


def test_unresolvable_identity_is_omitted_not_invented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "importlib.metadata.distribution",
        lambda _name: SimpleNamespace(read_text=lambda _f: None),
    )
    assert "sdk_identity" not in _metadata(monkeypatch, None)


def test_egress_projection_keeps_well_formed_identity() -> None:
    body = {
        "function_name": "fn",
        "privacy_mode": True,
        "sdk_identity": {"sdk_ref": _SHA, "sdk_version": "0.33.0"},
    }
    projected = oo._SESSION_METADATA(body, oo._Ctx(), 0)
    assert projected["sdk_identity"] == body["sdk_identity"]


def test_egress_projection_drops_partial_identity() -> None:
    body = {
        "function_name": "fn",
        "privacy_mode": True,
        "sdk_identity": {"sdk_ref": _SHA},
    }
    assert "sdk_identity" not in oo._SESSION_METADATA(body, oo._Ctx(), 0)
