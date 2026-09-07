# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""The coarse ``task_type`` hint must reach the wire on EVERY session-create path.

Why it exists: the backend's evaluator-quality anchor policy designates a verifiable
anchor (``mcq_exact`` today) only from (a) a curated dataset registry, (b) a task-family
map, or (c) a client-declared coarse ``task_type``. Both registries are empty on the
service, so (c) is the only live route -- and until this field the SDK never sent one,
so the first outcome cells ever written (2026-09-03) audited as ``no_anchor_designation``.

Same shape as ``test_session_narrative_wire``: one serializer helper, pinned on both
paths so they cannot drift apart.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import SessionCreationRequest, session_task_type_to_wire

pytestmark = pytest.mark.backend_online


def _request(**kwargs: Any) -> SessionCreationRequest:
    defaults: dict[str, Any] = {
        "function_name": "mcq_agent",
        "configuration_space": {"temperature": [0.0, 0.7]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 8},
    }
    defaults.update(kwargs)
    return SessionCreationRequest(**defaults)


def _direct_payload(**kwargs: Any) -> dict[str, Any]:
    client = object.__new__(TraigentCloudClient)
    client._ensure_owner_metadata = lambda metadata: metadata or {}
    return TraigentCloudClient._serialize_session_request(client, _request(**kwargs))


def _orchestrator_payload(**kwargs: Any) -> dict[str, Any]:
    return ApiOperations(MagicMock())._build_typed_session_payload(
        _request(**kwargs), max_trials=10
    )


class TestHelper:
    def test_strips_whitespace_and_keeps_the_token(self):
        assert session_task_type_to_wire(
            _request(task_type="  multiple_choice \n")
        ) == {"task_type": "multiple_choice"}

    @pytest.mark.parametrize("value", [None, "", "   ", 42, {"kind": "mcq"}])
    def test_absent_blank_or_non_string_is_omitted(self, value):
        req = _request()
        req.task_type = value  # bypass the dataclass default deliberately
        assert session_task_type_to_wire(req) == {}

    def test_does_not_normalize_vocabulary(self):
        """The server owns the vocabulary; the client must not rewrite the hint."""
        assert session_task_type_to_wire(_request(task_type="Text2SQL")) == {
            "task_type": "Text2SQL"
        }


class TestDirectSerializer:
    def test_task_type_reaches_the_wire(self):
        assert (
            _direct_payload(task_type="multiple_choice")["task_type"]
            == "multiple_choice"
        )

    def test_absent_hint_leaves_the_body_unchanged(self):
        assert "task_type" not in _direct_payload()

    def test_blank_hint_is_omitted_not_sent_empty(self):
        """The schema forbids the empty string; omit rather than fail validation."""
        assert "task_type" not in _direct_payload(task_type="   ")


class TestOrchestratorSerializer:
    def test_task_type_reaches_the_wire(self):
        assert (
            _orchestrator_payload(task_type="multiple_choice")["task_type"]
            == "multiple_choice"
        )

    def test_absent_hint_leaves_the_body_unchanged(self):
        assert "task_type" not in _orchestrator_payload()


class TestBothPathsAgree:
    def test_direct_and_orchestrator_serialize_identically(self):
        direct = _direct_payload(task_type="text2sql")
        orchestrated = _orchestrator_payload(task_type="text2sql")
        assert direct["task_type"] == orchestrated["task_type"] == "text2sql"


def test_request_declares_the_field_with_a_none_default():
    """Additive: existing constructors keep working, and the field is real, not ad hoc."""
    import dataclasses

    names = {f.name: f for f in dataclasses.fields(SessionCreationRequest)}
    assert "task_type" in names
    assert names["task_type"].default is None
    assert _request().task_type is None
