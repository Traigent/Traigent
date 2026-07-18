"""Transport, auth/project scoping, idempotency, retries, and honest results (WI-B)."""

from __future__ import annotations

import importlib.util
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None
pytestmark = pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")

from traigent.economics import (  # noqa: E402
    EconomicsBatchTooLarge,
    EconomicsIdempotencyConflict,
    EconomicsResponseError,
    EconomicsSchemaUnavailable,
    EconomicsTelemetryAuthError,
    EconomicsTelemetryClient,
    EconomicsTelemetryContractError,
    EconomicsTelemetryTransportError,
    EgressPolicyError,
    funnel_eligible_event,
)
from traigent.economics import client as client_mod  # noqa: E402
from traigent.economics.contract import (  # noqa: E402
    IDEMPOTENCY_KEY_HEADER,
    PROJECT_ID_HEADER,
)

_FULL_RUN_EVENT = {
    "event_type": "run_economics",
    "event_id": "evt-run-1",
    "occurred_at": "2026-07-18T10:00:00.000Z",
    "project_ref": "proj-1",
    "run_id": "run-1",
    "archetype": "solo_coding_builder",
    "characterization": {
        "overrides": {"loss_per_bad_output_usd": 4000},
        "field_reports": [
            {
                "field": "loss_per_bad_output_usd",
                "provenance": "inferred",
                "confidence": 0.7,
                "sharing_outcome": "shared",
                "evidence_status": "provided",
                "evidence_pointer": "SENSITIVE-incident-ledger-4k-escalation",
            }
        ],
    },
    "budget": {"authored_by": "backend", "recommended_daily_usd": 5, "cap_usd": 50},
    "actual_spend_usd": 0.0,
    "usage": {"input_tokens": 0, "output_tokens": 0, "model_calls": 0},
    "model_prices": [
        {
            "model_id": "gpt-4o-mini",
            "input_usd_per_mtok": 0.15,
            "output_usd_per_mtok": 0.6,
            "price_source": "provider_published",
            "as_of": "2026-07-18T10:00:00.000Z",
        }
    ],
    "evidence_identity": {
        "baseline_run_id": "run-0",
        "candidate_run_id": "run-1",
        "dataset_hash": "a" * 64,
        "evaluator_version": "exec-match-v2",
        "objective_weights": [{"objective": "accuracy", "weight": 1.0}],
        "effect_estimate": {
            "estimate": 0.1,
            "lower": 0.05,
            "upper": 0.15,
            "level": 0.95,
            "unit": "proportion",
        },
        "support": {"n_examples": 100, "n_paired": 100},
        "exclusions": [],
    },
    "advisory": {
        "advice_id": "adv-1",
        "recommended_action": "run_optimization",
        "client_action": "followed",
    },
    "labor_proxies": {},
}


@pytest.fixture(autouse=True)
def _online(monkeypatch: pytest.MonkeyPatch) -> None:
    # The suite sets TRAIGENT_OFFLINE_MODE=true; mocked-transport tests need the
    # boundary open. The explicit offline test overrides this to true itself.
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod.asyncio, "sleep", AsyncMock())


def _client() -> EconomicsTelemetryClient:
    return EconomicsTelemetryClient(
        backend_url="https://api.traigent.ai", api_key="test-key", max_retries=3
    )


def _event(project_ref: str = "proj-1") -> dict[str, Any]:
    return funnel_eligible_event(
        project_ref, event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )


def _mock_transport(client: EconomicsTelemetryClient, post_result: Any) -> MagicMock:
    mock_http = MagicMock()
    if isinstance(post_result, list):
        mock_http.post = AsyncMock(side_effect=post_result)
    else:
        mock_http.post = AsyncMock(return_value=post_result)
    client._client = mock_http
    return mock_http


def _resp(
    status: int,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body if body is not None else {}
    response.headers = headers or {}
    return response


def _ingest_body(
    prepared: Any,
    *,
    status: int = 201,
    replayed: bool = False,
    accepted: int | None = None,
    duplicate: int = 0,
    rejected: int = 0,
    rejections: list | None = None,
) -> dict[str, Any]:
    submitted = prepared.submitted
    if accepted is None:
        accepted = submitted - duplicate - rejected
    return {
        "contract": "economics_telemetry",
        "contract_version": "1.0.0",
        "batch_id": prepared.batch_id,
        "idempotency_key": prepared.idempotency_key,
        "received_at": "2026-07-18T10:00:02.000Z",
        "replayed": replayed,
        "counts": {
            "submitted": submitted,
            "accepted": accepted,
            "duplicate": duplicate,
            "rejected": rejected,
        },
        "rejections": rejections or [],
    }


# --- prepare / immutability -----------------------------------------------------


def test_prepare_produces_matching_key_bytes_and_headers() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    assert prepared.headers[PROJECT_ID_HEADER] == "proj-1"
    assert prepared.headers[IDEMPOTENCY_KEY_HEADER] == prepared.idempotency_key
    # content is the exact serialized body carrying the same key.
    parsed = json.loads(prepared.content)
    assert parsed["idempotency_key"] == prepared.idempotency_key
    assert parsed["batch_id"] == prepared.batch_id


def test_prepare_is_deterministic_for_identical_stable_tuple() -> None:
    client = _client()
    a = client.prepare(
        [_event()],
        project_id="proj-1",
        batch_id="b1",
        sent_at="2026-07-18T10:00:01.000Z",
    )
    b = client.prepare(
        [_event()],
        project_id="proj-1",
        batch_id="b1",
        sent_at="2026-07-18T10:00:01.000Z",
    )
    # Cross-call recovery: the SAME complete stable tuple yields identical bytes/key.
    assert a.content == b.content
    assert a.idempotency_key == b.idempotency_key


# --- accepted / replay / rejected results --------------------------------------


async def test_accepted_201_is_fully_accepted() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    _mock_transport(client, _resp(201, _ingest_body(prepared)))
    result = await client.submit_prepared(prepared)
    assert result.http_status == 201
    assert result.fully_accepted
    assert result.no_rejections
    assert not result.replayed


async def test_submit_end_to_end_sends_scope_and_matching_idempotency_header() -> None:
    client = _client()
    prepared = client.prepare(
        [_event()],
        project_id="proj-1",
        batch_id="b1",
        sent_at="2026-07-18T10:00:01.000Z",
    )
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    await client.submit(
        [_event()],
        project_id="proj-1",
        batch_id="b1",
        sent_at="2026-07-18T10:00:01.000Z",
    )
    kwargs = mock_http.post.call_args.kwargs
    headers = kwargs["headers"]
    sent = json.loads(kwargs["content"])
    assert headers[PROJECT_ID_HEADER] == "proj-1"
    assert headers[IDEMPOTENCY_KEY_HEADER] == sent["idempotency_key"]
    assert sent["idempotency_key"] == prepared.idempotency_key


async def test_replay_200_marks_replayed() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    _mock_transport(
        client, _resp(200, _ingest_body(prepared, status=200, replayed=True))
    )
    result = await client.submit_prepared(prepared)
    assert result.http_status == 200
    # replayed is the authoritative 'no new state written' signal; the counts
    # still echo the original disposition.
    assert result.replayed


async def test_all_duplicate_is_not_fully_accepted() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = _ingest_body(prepared, accepted=0, duplicate=1)
    _mock_transport(client, _resp(201, body))
    result = await client.submit_prepared(prepared)
    assert result.all_duplicate
    assert result.no_rejections
    assert not result.fully_accepted


async def test_all_rejected_422_is_not_disguised_as_success() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = _ingest_body(
        prepared,
        status=422,
        accepted=0,
        rejected=1,
        rejections=[{"event_index": 0, "reason": "tenant_scope_violation"}],
    )
    _mock_transport(client, _resp(422, body))
    result = await client.submit_prepared(prepared)
    assert result.http_status == 422
    assert result.all_rejected
    assert not result.fully_accepted
    assert result.rejection_reasons == ("tenant_scope_violation",)


# --- fail-closed response parsing ----------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda b: b.update({"contract": "something_else"}), id="wrong-contract"
        ),
        pytest.param(
            lambda b: b.update({"contract_version": "9.9.9"}), id="wrong-version"
        ),
        pytest.param(lambda b: b.update({"replayed": True}), id="status-flag-disagree"),
        pytest.param(
            lambda b: b.update({"idempotency_key": "econ-tel-wrongkey"}),
            id="key-mismatch",
        ),
        pytest.param(
            lambda b: b.update({"batch_id": "batch-wrong"}), id="batch-mismatch"
        ),
        pytest.param(
            lambda b: b["counts"].update({"accepted": 5}), id="counts-unreconciled"
        ),
        pytest.param(
            lambda b: b.update({"rejections": [{"event_index": 0, "reason": "x"}]}),
            id="rejections-mismatch-count",
        ),
    ],
)
async def test_malformed_response_fails_closed(mutate) -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = _ingest_body(prepared)
    mutate(body)
    _mock_transport(client, _resp(201, body))
    with pytest.raises(EconomicsResponseError):
        await client.submit_prepared(prepared)


async def test_response_with_unknown_rejection_reason_fails_closed() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = _ingest_body(
        prepared,
        status=422,
        accepted=0,
        rejected=1,
        rejections=[{"event_index": 0, "reason": "made_up_reason"}],
    )
    _mock_transport(client, _resp(422, body))
    with pytest.raises(EconomicsResponseError):
        await client.submit_prepared(prepared)


async def test_non_json_result_body_fails_closed() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    bad = MagicMock()
    bad.status_code = 201
    bad.json.side_effect = ValueError("not json")
    bad.headers = {}
    _mock_transport(client, bad)
    with pytest.raises(EconomicsResponseError):
        await client.submit_prepared(prepared)


# --- error envelopes ------------------------------------------------------------


async def test_conflict_409_raises() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    _mock_transport(client, _resp(409))
    with pytest.raises(EconomicsIdempotencyConflict):
        await client.submit_prepared(prepared)


async def test_batch_too_large_413_raises() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    _mock_transport(client, _resp(413))
    with pytest.raises(EconomicsBatchTooLarge):
        await client.submit_prepared(prepared)


async def test_bad_request_400_raises_contract_error() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    _mock_transport(client, _resp(400))
    with pytest.raises(EconomicsTelemetryContractError):
        await client.submit_prepared(prepared)


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failures_raise(status: int) -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    _mock_transport(client, _resp(status))
    with pytest.raises(EconomicsTelemetryAuthError):
        await client.submit_prepared(prepared)


# --- retries --------------------------------------------------------------------


async def test_transient_status_is_retried_then_succeeds() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    mock_http = _mock_transport(
        client, [_resp(503), _resp(201, _ingest_body(prepared))]
    )
    result = await client.submit_prepared(prepared)
    assert result.fully_accepted
    assert mock_http.post.await_count == 2


async def test_transient_exhaustion_raises_transport_error() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    mock_http = _mock_transport(client, [_resp(500), _resp(500), _resp(500)])
    with pytest.raises(EconomicsTelemetryTransportError):
        await client.submit_prepared(prepared)
    assert mock_http.post.await_count == 3


async def test_no_sleep_after_final_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(client_mod.asyncio, "sleep", sleep)
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    _mock_transport(client, [_resp(500), _resp(500), _resp(500)])
    with pytest.raises(EconomicsTelemetryTransportError):
        await client.submit_prepared(prepared)
    # 3 attempts -> at most 2 backoffs; never a sleep after the final failure.
    assert sleep.await_count == 2


async def test_timeout_exception_is_retried() -> None:
    import httpx

    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    mock_http = _mock_transport(
        client, [httpx.TimeoutException("t"), _resp(201, _ingest_body(prepared))]
    )
    result = await client.submit_prepared(prepared)
    assert result.fully_accepted
    assert mock_http.post.await_count == 2


async def test_retries_send_identical_bytes_and_key() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    mock_http = _mock_transport(
        client, [_resp(503), _resp(201, _ingest_body(prepared))]
    )
    await client.submit_prepared(prepared)
    calls = mock_http.post.call_args_list
    assert calls[0].kwargs["content"] == calls[1].kwargs["content"]
    assert (
        calls[0].kwargs["headers"][IDEMPOTENCY_KEY_HEADER]
        == calls[1].kwargs["headers"][IDEMPOTENCY_KEY_HEADER]
    )


async def test_bounded_retry_after_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []

    async def _capture(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(client_mod.asyncio, "sleep", _capture)
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    hostile = _resp(503, headers={"Retry-After": "100000"})
    _mock_transport(client, [hostile, _resp(201, _ingest_body(prepared))])
    await client.submit_prepared(prepared)
    assert delays and all(d <= client_mod._MAX_RETRY_AFTER_SECONDS for d in delays)


# --- guards: egress + schema fail closed before transport ----------------------


async def test_project_scope_mismatch_is_guarded_before_transport() -> None:
    client = _client()
    mock_http = _mock_transport(client, _resp(201))
    with pytest.raises(EconomicsTelemetryContractError, match="project_ref"):
        await client.submit([_event("other-project")], project_id="proj-1")
    mock_http.post.assert_not_called()


async def test_egress_violation_blocks_before_transport() -> None:
    run_event = {
        "event_type": "run_economics",
        "event_id": "evt-2",
        "occurred_at": "2026-07-18T10:00:00.000Z",
        "project_ref": "proj-1",
        "run_id": "run-1",
        "characterization": {
            "bands": {"error_cost_band": "not_measured"},
            "field_reports": [
                {
                    "field": "error_cost_band",
                    "provenance": "asked",
                    "confidence": 1.0,
                    "sharing_outcome": "withheld_by_policy",
                }
            ],
        },
    }
    client = _client()
    mock_http = _mock_transport(client, _resp(201))
    with pytest.raises(EgressPolicyError):
        await client.submit([run_event], project_id="proj-1")
    mock_http.post.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [EconomicsSchemaUnavailable("absent"), EconomicsTelemetryContractError("invalid")],
)
async def test_schema_failure_blocks_before_transport(monkeypatch, failure) -> None:
    def _raise(_body: dict) -> None:
        raise failure

    monkeypatch.setattr(client_mod, "validate_request_or_fail", _raise)
    client = _client()
    mock_http = _mock_transport(client, _resp(201))
    with pytest.raises(type(failure)):
        await client.submit([_event()], project_id="proj-1")
    mock_http.post.assert_not_called()


# --- auth / offline plumbing ----------------------------------------------------


def test_auth_headers_use_api_key() -> None:
    client = EconomicsTelemetryClient(
        backend_url="https://api.traigent.ai", api_key="secret-key"
    )
    headers = client._auth_headers()
    assert headers.get("X-API-Key") == "secret-key"
    assert "Authorization" not in headers


def test_auth_headers_use_jwt_when_no_api_key() -> None:
    client = EconomicsTelemetryClient(
        backend_url="https://api.traigent.ai", jwt_token="jwt-abc"
    )
    assert client._auth_headers() == {"Authorization": "Bearer jwt-abc"}


def test_offline_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from traigent.utils.error_handler import OfflineModeError

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    client = EconomicsTelemetryClient(
        backend_url="https://api.traigent.ai", api_key="k"
    )
    with pytest.raises(OfflineModeError):
        client._get_client()


# --- privacy: no sensitive values in logs --------------------------------------


async def test_no_sensitive_values_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    client = _client()
    prepared = client.prepare([_FULL_RUN_EVENT], project_id="proj-1")
    _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with caplog.at_level(logging.DEBUG, logger="traigent.economics.client"):
        await client.submit_prepared(prepared)
    assert "SENSITIVE-incident-ledger-4k-escalation" not in caplog.text
    assert "4000" not in caplog.text
