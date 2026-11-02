"""#2853: the finalize request body carries the run's terminal outcome."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.api.types import OptimizationStatus
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.utils import optimization_only_egress as oo_egress

pytestmark = pytest.mark.backend_online

SID = "sess-2853"


@pytest.fixture
def client():
    with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
        c = BackendIntegratedClient(api_key="k", base_url="http://localhost:5000")
        c.auth_manager = Mock()
        c.auth_manager.augment_headers = AsyncMock(
            return_value={"Authorization": "Bearer test-key"}
        )
        c.session_bridge.create_session_mapping(
            session_id=SID,
            experiment_id="exp-1",
            experiment_run_id="run-1",
            function_name="f",
            configuration_space={},
            objectives=["accuracy"],
        )
        c.local_storage = Mock()
        return c


def _http():
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"best_config": {}})
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    session = AsyncMock()
    session.post = Mock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


async def _finalize_body(client, *args, **kwargs) -> dict:
    session = _http()
    with patch("aiohttp.ClientSession", return_value=session):
        await client._session_ops.finalize_session(SID, *args, **kwargs)
    return session.post.call_args.kwargs["json"]


class TestFinalizeWireBody:
    @pytest.mark.asyncio
    async def test_failed_run_body_carries_failed_outcome(self, client):
        body = await _finalize_body(client, False, optimization_outcome="failed")
        assert body["optimization_outcome"] == "failed"
        assert body["experiment_run_id"] == "run-1"
        client.local_storage.finalize_session.assert_called_once_with(SID, "failed")

    @pytest.mark.asyncio
    async def test_cancelled_run_body_and_local_status(self, client):
        body = await _finalize_body(client, False, optimization_outcome="cancelled")
        assert body["optimization_outcome"] == "cancelled"
        client.local_storage.finalize_session.assert_called_once_with(SID, "cancelled")

    @pytest.mark.asyncio
    async def test_completed_run_body(self, client):
        body = await _finalize_body(client, True, optimization_outcome="completed")
        assert body["optimization_outcome"] == "completed"
        client.local_storage.finalize_session.assert_called_once_with(SID, "completed")

    @pytest.mark.asyncio
    async def test_legacy_caller_without_outcome_sends_no_key(self, client):
        body = await _finalize_body(client, True)
        assert "optimization_outcome" not in body
        client.local_storage.finalize_session.assert_called_once_with(SID, "completed")

    @pytest.mark.asyncio
    async def test_unknown_outcome_is_refused_before_any_send(self, client):
        session = _http()
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(ValueError):
                await client._session_ops.finalize_session(
                    SID, False, optimization_outcome="running"
                )
        session.post.assert_not_called()


class TestEgressProjection:
    @pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
    def test_outcome_survives_the_finalize_projection(self, outcome):
        out = oo_egress.project_finalize_payload(
            {"reason": "sdk_explicit_finalization", "optimization_outcome": outcome}
        )
        assert out["optimization_outcome"] == outcome

    @pytest.mark.parametrize("bad", ["COMPLETED", "running", "", 1, ["failed"], {}])
    def test_invalid_outcome_is_refused_not_sent(self, bad):
        with pytest.raises(oo_egress.MetadataContractViolation):
            oo_egress.project_finalize_payload(
                {"reason": "sdk_explicit_finalization", "optimization_outcome": bad}
            )


class TestManagerMapping:
    @pytest.mark.parametrize(
        ("status", "outcome"),
        [
            (OptimizationStatus.COMPLETED, "completed"),
            (OptimizationStatus.FAILED, "failed"),
            (OptimizationStatus.CANCELLED, "cancelled"),
            (OptimizationStatus.UNKNOWN, "failed"),
            (OptimizationStatus.RUNNING, "failed"),
        ],
    )
    def test_status_maps_to_wire_outcome(self, status, outcome):
        assert BackendSessionManager.finalize_outcome_for_status(status) == outcome
