# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""Crew declarations from code: ``@optimize(crew=..., role=...)``.

The keys are customer content: they must reach the wire only as session metadata
``crew_declaration`` and appear in no log, error text, repr or local record.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

import traigent
from traigent.api.types import OptimizationStatus
from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import SessionCreationRequest
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.objectives import create_default_objectives
from traigent.core.session_types import SessionCreationResult
from traigent.evaluators.base import Dataset
from traigent.utils.crew_declaration import (
    CONFLICT_WARNING,
    UNKNOWN_BINDING,
    CrewDeclaration,
    crew_declaration_to_wire,
    parse_crew_binding,
    validate_crew_declaration,
)
from traigent.utils.function_identity import resolve_function_descriptor

pytestmark = pytest.mark.backend_online

CANARY_CREW = "canary-crew-7f3a91"
CANARY_ROLE = "canary role 7f3a91 (triage)"
CREW_ID = "crew_" + "ab" * 16
BOUND_IDS = {
    "crew_id": CREW_ID,
    "role_id": "crol_" + "cd" * 16,
    "role_token": "rtk_abcdefghijklmnopqrstuvwxyz",  # pragma: allowlist secret
    "crew_revision_id": "crev_" + "ef" * 16,
}


def _request(**kwargs: Any) -> SessionCreationRequest:
    defaults: dict[str, Any] = {
        "function_name": "agent",
        "configuration_space": {"temperature": [0.0, 0.7]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 8},
    }
    defaults.update(kwargs)
    return SessionCreationRequest(**defaults)


class TestValidation:
    def test_neither_is_none(self):
        assert validate_crew_declaration(None, None) is None

    @pytest.mark.parametrize("crew,role", [("a", None), (None, "a")])
    def test_half_pair_raises(self, crew, role):
        with pytest.raises(ValueError, match="both or neither"):
            validate_crew_declaration(crew, role)

    @pytest.mark.parametrize(
        "crew",
        ["", "has space", "a/b", "dot.dot", "x" * 129, "café", "ab\n", 7],
    )
    def test_invalid_crew_matches_backend_pattern(self, crew):
        with pytest.raises(ValueError, match="crew must be"):
            validate_crew_declaration(crew, "triage")

    @pytest.mark.parametrize("role", ["", "r" * 129, 3, None.__class__])
    def test_invalid_role(self, role):
        with pytest.raises(ValueError, match="role must be"):
            validate_crew_declaration("support-desk", role)

    def test_boundaries_accepted(self):
        decl = validate_crew_declaration(
            "A" * 128, "any text, even spaces " + "r" * 106
        )
        assert decl is not None
        assert validate_crew_declaration("a_b-9", "r") is not None

    def test_errors_and_repr_never_echo_keys(self):
        with pytest.raises(ValueError) as info:
            validate_crew_declaration(CANARY_CREW + " bad", CANARY_ROLE)
        assert CANARY_CREW not in str(info.value)
        assert CANARY_ROLE not in str(info.value)
        decl = validate_crew_declaration(CANARY_CREW, CANARY_ROLE)
        assert CANARY_CREW not in repr(decl) and CANARY_ROLE not in repr(decl)


class TestDecorator:
    def _decorate(self, **kwargs: Any):
        @traigent.optimize(
            configuration_space={"t": [0.1, 0.2]},
            objectives=["accuracy"],
            eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
            **kwargs,
        )
        def agent(q: str) -> str:
            return q

        return agent

    def test_declaration_is_stored(self):
        fn = self._decorate(crew="support-desk", role="triage")
        assert fn.crew_declaration == CrewDeclaration("support-desk", "triage")
        assert fn.crew_binding is None  # nothing has run yet

    def test_absent_leaves_no_declaration(self):
        assert self._decorate().crew_declaration is None

    def test_half_pair_and_bad_keys_fail_locally(self):
        with pytest.raises(ValueError, match="both or neither"):
            self._decorate(crew="support-desk")
        with pytest.raises(ValueError, match="crew must be"):
            self._decorate(crew="bad key", role="triage")

    def test_agents_combination_rejected_without_echo(self):
        with pytest.raises(ValueError, match="cannot be combined with agents") as info:
            self._decorate(
                crew=CANARY_CREW, role=CANARY_ROLE, agents={"a": MagicMock()}
            )
        assert CANARY_CREW not in str(info.value)


class TestWire:
    def test_helper_shape(self):
        decl = CrewDeclaration("c", "r")
        assert crew_declaration_to_wire(_request(crew_declaration=decl)) == {
            "crew_declaration": {"crew_key": "c", "role_key": "r"}
        }
        assert crew_declaration_to_wire(_request()) == {}

    def test_orchestrator_payload_carries_it_only_in_metadata(self):
        decl = CrewDeclaration(CANARY_CREW, CANARY_ROLE)
        payload = ApiOperations(MagicMock())._build_typed_session_payload(
            _request(crew_declaration=decl), max_trials=5
        )
        assert payload["metadata"]["crew_declaration"] == {
            "crew_key": CANARY_CREW,
            "role_key": CANARY_ROLE,
        }
        rest = {k: v for k, v in payload.items() if k != "metadata"}
        assert CANARY_CREW not in repr(rest)
        meta_rest = {
            k: v for k, v in payload["metadata"].items() if k != "crew_declaration"
        }
        assert CANARY_CREW not in repr(meta_rest)

    def test_direct_payload_carries_it(self):
        client = object.__new__(TraigentCloudClient)
        client._ensure_owner_metadata = lambda metadata: metadata or {}
        payload = TraigentCloudClient._serialize_session_request(
            client, _request(crew_declaration=CrewDeclaration("c", "r"))
        )
        assert payload["metadata"]["crew_declaration"] == {
            "crew_key": "c",
            "role_key": "r",
        }

    def test_absent_declaration_changes_nothing(self):
        payload = ApiOperations(MagicMock())._build_typed_session_payload(
            _request(), max_trials=5
        )
        assert "crew_declaration" not in payload["metadata"]


class TestResponse:
    @pytest.mark.parametrize(
        "raw",
        [None, "bound", {}, {"status": "nonsense"}, {"status": 3}, []],
    )
    def test_missing_or_unrecognised_is_unknown(self, raw):
        assert parse_crew_binding(raw) == UNKNOWN_BINDING
        assert UNKNOWN_BINDING.status == "unknown"

    def test_bound_keeps_ids_only(self):
        binding = parse_crew_binding(
            {
                "status": "bound",
                **BOUND_IDS,
                "revision_no": 2,
            }
        )
        assert binding.status == "bound" and binding.revision_no == 2
        assert binding.crew_id == BOUND_IDS["crew_id"]
        assert CANARY_CREW not in repr(binding)

    def test_extra_field_such_as_a_name_is_unknown(self):
        raw = {
            "status": "bound",
            **BOUND_IDS,
            "revision_no": 2,
            "crew_key": CANARY_CREW,
        }
        assert parse_crew_binding(raw) == UNKNOWN_BINDING

    @pytest.mark.parametrize(
        "raw",
        [
            {"status": "bound", **BOUND_IDS},  # revision_no missing
            {"status": "bound", "crew_id": CREW_ID, "revision_no": 1},  # ids missing
            {"status": "bound", **BOUND_IDS, "revision_no": 0},
            {"status": "bound", **BOUND_IDS, "revision_no": True},
            {"status": "bound", **{**BOUND_IDS, "crew_id": "crew_1"}, "revision_no": 1},
            {"status": "bound", **{**BOUND_IDS, "role_token": "tok"}, "revision_no": 1},
            {
                "status": "bound",
                **{**BOUND_IDS, "crew_id": CREW_ID + "\n"},
                "revision_no": 1,
            },
            {
                "status": "bound",
                **{**BOUND_IDS, "role_id": CANARY_ROLE},
                "revision_no": 1,
            },
            {"status": "pending_portal", **BOUND_IDS, "revision_no": 1},
            {"status": "conflict", "role_id": BOUND_IDS["role_id"]},
            {"status": "disabled", "crew_id": CREW_ID},
            {"status": "unknown", "crew_id": CREW_ID},
        ],
    )
    def test_malformed_binding_is_unknown(self, raw):
        assert parse_crew_binding(raw) == UNKNOWN_BINDING

    @pytest.mark.parametrize(
        "raw",
        [
            {"status": "pending_portal"},
            {"status": "pending_portal", "crew_id": CREW_ID},
            {"status": "conflict", "crew_id": CREW_ID},
            {"status": "disabled"},
            {"status": "unknown"},
        ],
    )
    def test_valid_non_bound_statuses_are_kept(self, raw):
        assert parse_crew_binding(raw).status == raw["status"]

    def test_repr_has_status_and_ids_only(self):
        text = repr(parse_crew_binding({"status": "conflict", "crew_id": CREW_ID}))
        assert "conflict" in text and CREW_ID in text
        assert CANARY_CREW not in text

    def test_direct_client_response_exposes_binding(self):
        client = object.__new__(TraigentCloudClient)
        data = {
            "session_id": "s",
            "status": "active",
            "crew_binding": {"status": "pending_portal", "crew_id": CREW_ID},
        }
        assert (
            client._deserialize_session_response(data).crew_binding.crew_id == CREW_ID
        )
        data.pop("crew_binding")
        assert (
            client._deserialize_session_response(data).crew_binding.status == "unknown"
        )

    @pytest.mark.asyncio
    async def test_session_response_exposes_binding(self):
        response = AsyncMock()
        response.json = AsyncMock(
            return_value={
                "session_id": "s",
                "metadata": {"experiment_id": "e", "experiment_run_id": "r"},
                "crew_binding": {"status": "pending_portal", "crew_id": CREW_ID},
            }
        )
        parsed = await ApiOperations(Mock())._parse_session_response(response)
        assert parsed.crew_binding.status == "pending_portal"
        assert parsed.crew_binding.crew_id == CREW_ID

    @pytest.mark.asyncio
    async def test_older_backend_reads_unknown(self):
        response = AsyncMock()
        response.json = AsyncMock(
            return_value={
                "session_id": "s",
                "metadata": {"experiment_id": "e", "experiment_run_id": "r"},
            }
        )
        parsed = await ApiOperations(Mock())._parse_session_response(response)
        assert parsed.crew_binding.status == "unknown"


def _manager(client: Mock, monkeypatch) -> BackendSessionManager:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    config = TraigentConfig()
    config.execution_mode = "local"
    optimizer = Mock()
    optimizer.objectives = ["accuracy"]
    optimizer.config_space = {"p": [1, 2]}
    return BackendSessionManager(
        backend_client=client,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=create_default_objectives(
            objective_names=["accuracy"],
            orientations={"accuracy": "maximize"},
            weights={"accuracy": 1.0},
        ),
        optimizer=optimizer,
        optimization_id="opt",
        optimization_status=OptimizationStatus.RUNNING,
    )


def _create(manager: BackendSessionManager, **kwargs: Any):
    def func(x):
        return x

    dataset = Mock(spec=Dataset)
    dataset.name = "ds"
    dataset.examples = []
    dataset.__len__ = Mock(return_value=0)
    return manager.create_session(
        func=func,
        dataset=dataset,
        function_descriptor=resolve_function_descriptor(func),
        max_trials=3,
        start_time=1.0,
        **kwargs,
    )


class TestSessionManager:
    def _client(self, binding: Any) -> Mock:
        client = Mock()
        client.create_session = Mock(
            return_value=SessionCreationResult.connected(
                session_id="sid", crew_binding=binding
            )
        )
        client.get_session_mapping = Mock(return_value=MagicMock())
        client.upload_example_features = Mock(return_value=True)
        client.auth_manager = Mock(has_api_key=Mock(return_value=True))
        return client

    def test_declaration_forwarded_and_keys_not_in_metadata_or_logs(
        self, monkeypatch, caplog
    ):
        client = self._client(parse_crew_binding({"status": "pending_portal"}))
        decl = CrewDeclaration(CANARY_CREW, CANARY_ROLE)
        with caplog.at_level(logging.DEBUG):
            ctx = _create(_manager(client, monkeypatch), crew_declaration=decl)
        kwargs = client.create_session.call_args[1]
        assert kwargs["crew_declaration"] is decl
        assert CANARY_CREW not in repr(kwargs["metadata"])
        assert ctx.crew_binding.status == "pending_portal"
        assert CANARY_CREW not in caplog.text and CANARY_ROLE not in caplog.text

    def test_undeclared_does_not_pass_the_kwarg(self, monkeypatch):
        client = self._client(None)
        _create(_manager(client, monkeypatch))
        assert "crew_declaration" not in client.create_session.call_args[1]

    def test_conflict_warns_once_with_exact_text(self, monkeypatch, caplog):
        client = self._client(parse_crew_binding({"status": "conflict"}))
        decl = CrewDeclaration(CANARY_CREW, CANARY_ROLE)
        with caplog.at_level(logging.WARNING):
            _create(_manager(client, monkeypatch), crew_declaration=decl)
        messages = [r.getMessage() for r in caplog.records]
        assert messages.count(CONFLICT_WARNING) == 1
        assert CONFLICT_WARNING == (
            "Crew declaration conflicts with portal membership; "
            "portal membership was retained."
        )
        assert CANARY_CREW not in caplog.text

    def test_no_warning_when_not_conflict(self, monkeypatch, caplog):
        client = self._client(parse_crew_binding({"status": "disabled"}))
        with caplog.at_level(logging.WARNING):
            _create(
                _manager(client, monkeypatch),
                crew_declaration=CrewDeclaration("c", "r"),
            )
        assert CONFLICT_WARNING not in caplog.text


class TestOptimizedFunctionBinding:
    def test_offline_or_missing_binding_is_unknown(self):
        @traigent.optimize(
            configuration_space={"t": [0.1]},
            objectives=["accuracy"],
            eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
            crew="c",
            role="r",
        )
        def agent(q: str) -> str:
            return q

        agent._adopt_crew_binding(Mock(crew_binding=None))
        assert agent.crew_binding.status == "unknown"
        agent._adopt_crew_binding(
            Mock(
                crew_binding=parse_crew_binding(
                    {"status": "bound", **BOUND_IDS, "revision_no": 1}
                )
            )
        )
        assert agent.crew_binding.status == "bound"
