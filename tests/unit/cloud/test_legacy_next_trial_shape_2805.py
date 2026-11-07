"""#2805: every next-trial reader understands the legacy (has_next) reply shape,
and a 413 renders with its closed code."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from traigent.cloud.trial_operations import TrialOperations

LEGACY_NEXT = {
    "has_next": True,
    "trial": {"trial_id": "trial-legacy-1", "configuration": {"model": "m1"}},
    "configuration_run_id": "run-legacy-1",
    "dataset_subset_indices": None,
}
LEGACY_DONE = {"has_next": False, "trial": None}


def test_slot_reader_acquires_a_legacy_trial() -> None:
    slot = TrialOperations._slot_from_next_trial_data(LEGACY_NEXT, "s1")
    assert slot.trial_id == "trial-legacy-1"
    assert slot.optimization_complete is False


def test_slot_reader_reports_legacy_completion_not_unavailable() -> None:
    slot = TrialOperations._slot_from_next_trial_data(LEGACY_DONE, "s1")
    assert slot.optimization_complete is True
    assert slot.trial_id is None


def test_typed_payload_is_untouched() -> None:
    typed = {"should_continue": True, "suggestion": {"trial_id": "t-typed"}}
    slot = TrialOperations._slot_from_next_trial_data(typed, "s1")
    assert slot.trial_id == "t-typed"


def _sync_manager_with(payload: dict, status: int = 200):
    from traigent.cloud.sync_manager import SyncManager

    manager = SyncManager.__new__(SyncManager)
    manager.base_url = "https://api.example.test/api/v1"
    manager._request_timeout = 5
    manager._raise_if_backend_egress_disabled = lambda *_a, **_k: None
    manager._optimization_only_policy = lambda: None
    response = SimpleNamespace(status_code=status, text="", json=lambda: payload)
    manager._session = SimpleNamespace(post=Mock(return_value=response))
    manager._response_json = lambda r: r.json()
    return manager


def test_sync_next_trial_reads_legacy_slot_and_completion() -> None:
    assert _sync_manager_with(LEGACY_NEXT)._sync_next_trial("s1") == {
        "success": True,
        "trial_id": "trial-legacy-1",
    }
    assert _sync_manager_with(LEGACY_DONE)._sync_next_trial("s1") == {
        "success": True,
        "complete": True,
    }


def test_client_deserializer_accepts_the_legacy_reply() -> None:
    from traigent.cloud.client import TraigentCloudClient

    client = TraigentCloudClient.__new__(TraigentCloudClient)
    response = client._deserialize_next_trial_response(LEGACY_NEXT)
    assert response.should_continue is True
    assert response.suggestion is not None
    assert response.suggestion.trial_id == "trial-legacy-1"
    assert response.suggestion.config == {"model": "m1"}
    done = client._deserialize_next_trial_response(LEGACY_DONE)
    assert done.should_continue is False
    assert done.suggestion is None


def test_413_renders_with_its_closed_code() -> None:
    from traigent.core.session_types import safe_http_error_summary

    summary = safe_http_error_summary(413, '{"error_code": "PAYLOAD_TOO_LARGE"}')
    assert summary == "HTTP 413 | payload_too_large"


@pytest.mark.parametrize("status", [413])
def test_payload_too_large_is_in_the_closed_vocabulary(status: int) -> None:
    from traigent.core.session_types import SAFE_HTTP_ERROR_CODES

    assert "payload_too_large" in SAFE_HTTP_ERROR_CODES


# --- #2805 option A: fail fast when the legacy contract cannot carry the run ---


async def _create_with_typed_400(space: dict, max_trials: int = 10):
    from unittest.mock import AsyncMock, patch

    from traigent.cloud.api_operations import ApiOperations
    from traigent.cloud.client import CloudServiceError
    from traigent.cloud.models import SessionCreationRequest
    from traigent.cloud.session_types import SessionCreationFailureDetail

    client = Mock()
    client.auth_manager.augment_headers = AsyncMock(return_value={})
    client.no_egress = False
    client._url_invalid = False
    ops = ApiOperations(client)
    request = SessionCreationRequest(
        function_name="test_func",
        configuration_space=space,
        objectives=["maximize"],
        max_trials=max_trials,
    )
    exc = CloudServiceError("Session creation failed: HTTP 400")
    exc.session_creation_failure = SessionCreationFailureDetail.from_http_response(
        400, '{"error_code": "validation_error"}'
    )
    post = AsyncMock(side_effect=[exc, ("sid", "exp", "run")])
    with (
        patch("traigent.cloud.api_operations.is_backend_offline", return_value=False),
        patch("traigent.cloud.api_operations.AIOHTTP_AVAILABLE", True),
        patch.object(ops, "_post_session_creation", new=post),
        patch.object(ops, "_is_governed_request", return_value=False),
        patch.object(ops, "_session_contract", return_value="auto"),
    ):
        try:
            return await ops.create_traigent_session_via_api(request), post
        except Exception as raised:  # noqa: BLE001 - returned for assertions
            return raised, post


@pytest.mark.asyncio
@pytest.mark.backend_online
async def test_unsupported_knob_fails_fast_without_legacy_post() -> None:
    from traigent.cloud.client import SessionContractError

    err, post = await _create_with_typed_400(
        {"model": ["a", "b"], "my_secret_knob": [1, 2]}
    )
    assert isinstance(err, SessionContractError)
    assert post.await_count == 1
    assert "my_secret_knob" in str(err)
    assert "TRAIGENT_SESSION_CONTRACT=typed" in str(err)


@pytest.mark.asyncio
@pytest.mark.backend_online
async def test_forced_grid_over_max_trials_fails_fast_without_values() -> None:
    from traigent.cloud.client import SessionContractError

    err, post = await _create_with_typed_400(
        {"model": ["m1", "m2", "m3", "m4"], "temperature": [0.1, 0.2, 0.3, 0.4]},
        max_trials=4,
    )
    assert isinstance(err, SessionContractError)
    assert post.await_count == 1
    assert "16 configurations" in str(err)
    assert "m1" not in str(err)


@pytest.mark.asyncio
@pytest.mark.backend_online
async def test_oversize_legacy_payload_fails_fast() -> None:
    from traigent.cloud.client import SessionContractError

    err, post = await _create_with_typed_400({"prompt_template": ["x" * 1_100_000]})
    assert isinstance(err, SessionContractError)
    assert post.await_count == 1
    assert "413" in str(err)
    assert "xxxx" not in str(err)


@pytest.mark.asyncio
@pytest.mark.backend_online
async def test_carryable_legacy_run_still_retries_once() -> None:
    result, post = await _create_with_typed_400(
        {"model": ["a", "b"], "temperature": [0.1, 0.2]}, max_trials=10
    )
    assert result == ("sid", "exp", "run")
    assert post.await_count == 2
