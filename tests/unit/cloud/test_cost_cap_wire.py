# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""SDK#1613: the backend's server-side per-run cost cap (budget.max_cost_usd
on typed session create, read by interactive_session_service._normalize_budget)
was DORMANT for every SDK run — the SDK built the client-side CostEnforcer from
the user's cost_limit but never put budget.max_cost_usd on the wire, so only
the client-side enforcer stopped spend (fails open on REST-direct bypass / a
modified SDK). These tests pin the fix: cost_limit threads from
SessionOperations.create_session into SessionCreationRequest.budget, and the
typed payload builder forwards it onto the wire.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.models import OptimizationSession, SessionCreationRequest
from traigent.cloud.session_operations import SessionOperations


def _make_api_ops() -> ApiOperations:
    return ApiOperations(MagicMock())


def _typed_request(**kwargs: Any) -> SessionCreationRequest:
    defaults: dict[str, Any] = {
        "function_name": "test_func",
        "configuration_space": {"param": [1, 2, 3]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 1},
        "promotion_policy": None,
        "tvl_governance": None,
    }
    defaults.update(kwargs)
    return SessionCreationRequest(**defaults)


class TestBuildTypedSessionPayloadBudget:
    """_build_typed_session_payload forwards session_request.budget onto the wire."""

    def test_budget_included_when_set(self):
        ops = _make_api_ops()
        request = _typed_request(budget={"max_cost_usd": 12.5})
        payload = ops._build_typed_session_payload(request, max_trials=5)
        assert payload["budget"] == {"max_cost_usd": 12.5}

    def test_budget_omitted_when_none(self):
        ops = _make_api_ops()
        request = _typed_request(budget=None)
        payload = ops._build_typed_session_payload(request, max_trials=5)
        assert "budget" not in payload

    def test_no_regression_baseline_payload_omits_budget_key(self):
        ops = _make_api_ops()
        request = _typed_request()
        payload = ops._build_typed_session_payload(request, max_trials=10)
        assert "budget" not in payload
        assert "function_name" in payload
        assert "configuration_space" in payload
        assert "objectives" in payload
        assert "max_trials" in payload


class TestBuildTypedSessionPayloadOptimizationStrategy:
    """_build_typed_session_payload forwards backend smart strategy."""

    def test_optimization_strategy_included_when_set(self):
        ops = _make_api_ops()
        request = _typed_request(
            optimization_strategy={"algorithm": "optuna", "sampler": "tpe"}
        )
        payload = ops._build_typed_session_payload(request, max_trials=5)
        assert payload["optimization_strategy"] == {
            "algorithm": "optuna",
            "sampler": "tpe",
        }

    def test_optimization_strategy_omitted_when_none(self):
        ops = _make_api_ops()
        request = _typed_request(optimization_strategy=None)
        payload = ops._build_typed_session_payload(request, max_trials=5)
        assert "optimization_strategy" not in payload


# ---------------------------------------------------------------------------
# Tests: SessionOperations.create_session threads cost_limit -> budget.max_cost_usd
# ---------------------------------------------------------------------------


class TrackingLock:
    def __init__(self) -> None:
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeSessionBridge:
    def create_session_mapping(self, **kwargs) -> None:
        return None

    def get_session_mapping(self, session_id: str):
        return SimpleNamespace(experiment_run_id="run-1")

    _session_mappings: dict[str, Any] = {}


class FakeAuth:
    async def get_headers(self) -> dict[str, str]:
        return {}


class FakeAuthManager:
    def __init__(self) -> None:
        self.auth = FakeAuth()

    def has_api_key(self) -> bool:
        return True


class CapturingFakeClient:
    """FakeClient that records the SessionCreationRequest passed to the API."""

    def __init__(self) -> None:
        self._active_sessions_lock = TrackingLock()
        self._active_sessions: dict[str, OptimizationSession] = {}
        self._max_active_sessions = 5
        self.session_bridge = FakeSessionBridge()
        self.backend_config = SimpleNamespace(api_base_url=None, backend_base_url=None)
        self.auth_manager = FakeAuthManager()
        self._register_security_session = MagicMock()
        self.local_storage = None
        self.captured_session_request: SessionCreationRequest | None = None

    async def _ensure_session(self):
        return SimpleNamespace(post=AsyncMock(), get=AsyncMock())

    async def _create_traigent_session_via_api(
        self, session_request: SessionCreationRequest
    ):
        self.captured_session_request = session_request
        return ("session-001", "exp-001", "run-001")

    def _revoke_security_session(self, *args, **kwargs) -> None:
        return None


@pytest.fixture(autouse=True)
def _offline_disabled(monkeypatch):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")


class TestSessionOperationsCostLimitThreading:
    """create_session(cost_limit=...) arms budget.max_cost_usd on the request."""

    def _make_ops(self) -> tuple[SessionOperations, CapturingFakeClient]:
        client = CapturingFakeClient()
        return SessionOperations(cast(Any, client)), client

    def test_cost_limit_threads_to_session_request_budget(self):
        ops, client = self._make_ops()
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
            cost_limit=7.5,
        )
        assert client.captured_session_request is not None
        assert client.captured_session_request.budget == {"max_cost_usd": 7.5}

    def test_no_cost_limit_leaves_budget_none(self):
        ops, client = self._make_ops()
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
        )
        assert client.captured_session_request is not None
        assert client.captured_session_request.budget is None

    @pytest.mark.parametrize("bad_limit", [0, -3.0])
    def test_non_positive_cost_limit_leaves_budget_none(self, bad_limit):
        """The backend rejects budget.max_cost_usd <= 0 — never send a
        non-positive value onto the wire; stay uncapped instead."""
        ops, client = self._make_ops()
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
            cost_limit=bad_limit,
        )
        assert client.captured_session_request is not None
        assert client.captured_session_request.budget is None

    def test_cost_limit_end_to_end_reaches_typed_wire_payload(self):
        """Full path: cost_limit -> SessionCreationRequest.budget -> typed payload."""
        ops, client = self._make_ops()
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
            cost_limit=2.5,
        )
        request = client.captured_session_request
        assert request is not None
        api_ops = _make_api_ops()
        payload = api_ops._build_typed_session_payload(request, max_trials=5)
        assert payload["budget"] == {"max_cost_usd": 2.5}

    def test_optimization_strategy_end_to_end_reaches_typed_wire_payload(self):
        """Full path: SessionOperations.create_session -> typed payload."""
        ops, client = self._make_ops()
        strategy = {"algorithm": "optuna", "sampler": "tpe"}
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
            optimization_strategy=strategy,
        )
        request = client.captured_session_request
        assert request is not None
        assert request.optimization_strategy == strategy
        api_ops = _make_api_ops()
        payload = api_ops._build_typed_session_payload(request, max_trials=5)
        assert payload["optimization_strategy"] == strategy
