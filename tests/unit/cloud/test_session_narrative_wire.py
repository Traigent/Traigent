# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""Agent identity and per-run narrative must reach the wire on EVERY path.

A pre-merge review found `TraigentCloudClient._serialize_session_request` posting only
`function_name` while `SessionCreationRequest` declared `agent_key`, `run_title`,
and `run_description`. A caller who pinned a stable `agent_key` alongside a
descriptive `function_name` therefore had identity resolved from the description
server-side: a new Agent per run, and the (agent, dataset) optimization history
this feature exists to build shattered into one-run cohorts. Nothing failed —
the field was simply dropped.

Both session-create serializers now share `session_narrative_to_wire`, and these
tests pin both so the two paths cannot drift apart again.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import SessionCreationRequest, session_narrative_to_wire

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online

AGENT_KEY = "txt2sql_agent"
RUN_TITLE = "Check best router model"
RUN_DESCRIPTION = "Compare 4 routers at fixed temperature."


def _request(**kwargs: Any) -> SessionCreationRequest:
    defaults: dict[str, Any] = {
        "function_name": "test_func",
        "configuration_space": {"param": [1, 2, 3]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 1},
    }
    defaults.update(kwargs)
    return SessionCreationRequest(**defaults)


def _direct_payload(**kwargs: Any) -> dict[str, Any]:
    """Serialize through the direct client path (the one that dropped them)."""
    client = object.__new__(TraigentCloudClient)
    client._ensure_owner_metadata = lambda metadata: metadata or {}
    return TraigentCloudClient._serialize_session_request(client, _request(**kwargs))


def _orchestrator_payload(**kwargs: Any) -> dict[str, Any]:
    return ApiOperations(MagicMock())._build_typed_session_payload(
        _request(**kwargs), max_trials=10
    )


class TestDirectSerializer:
    def test_narrative_reaches_the_wire(self):
        payload = _direct_payload(
            agent_key=AGENT_KEY, run_title=RUN_TITLE, run_description=RUN_DESCRIPTION
        )
        assert payload["agent_key"] == AGENT_KEY
        assert payload["run_title"] == RUN_TITLE
        assert payload["run_description"] == RUN_DESCRIPTION

    def test_absent_narrative_leaves_the_body_unchanged(self):
        """Additive only: an existing caller's request body must not grow keys."""
        payload = _direct_payload()
        assert "agent_key" not in payload
        assert "run_title" not in payload
        assert "run_description" not in payload

    def test_blank_values_are_omitted_not_sent_empty(self):
        payload = _direct_payload(agent_key="   ", run_title="", run_description="  ")
        assert "agent_key" not in payload
        assert "run_title" not in payload
        assert "run_description" not in payload


class TestBothPathsAgree:
    def test_direct_and_orchestrator_serialize_identically(self):
        """The drift that caused the defect: two serializers, one behavior."""
        kwargs = {
            "agent_key": AGENT_KEY,
            "run_title": RUN_TITLE,
            "run_description": RUN_DESCRIPTION,
        }
        direct = _direct_payload(**kwargs)
        orchestrated = _orchestrator_payload(**kwargs)
        keys = ("agent_key", "run_title", "run_description")
        assert {k: direct[k] for k in keys} == {k: orchestrated[k] for k in keys}


class TestNarrativeSerializer:
    def test_titles_are_truncated_but_identity_is_not(self):
        """Truncating a label loses cosmetics; truncating an identity loses history.

        Two agent keys sharing a 255-char prefix would collapse into ONE agent if
        the client shortened them, silently MERGING two optimization histories —
        the same mis-grouping this feature prevents, inverted. Over-length keys go
        to the server untouched, for it to reject.
        """
        long_key = "k" * 300
        wire = session_narrative_to_wire(
            _request(
                agent_key=long_key,
                run_title="t" * 600,
                run_description="d" * 5000,
            )
        )
        assert wire["agent_key"] == long_key
        assert len(wire["run_title"]) == 512
        assert len(wire["run_description"]) == 4000

    def test_non_string_values_are_ignored(self):
        wire = session_narrative_to_wire(
            _request(agent_key=42, run_title=[], run_description={})
        )
        assert wire == {}

    def test_values_are_trimmed(self):
        wire = session_narrative_to_wire(
            _request(agent_key=f"  {AGENT_KEY}  ", run_title=f"\t{RUN_TITLE}\n")
        )
        assert wire["agent_key"] == AGENT_KEY
        assert wire["run_title"] == RUN_TITLE
