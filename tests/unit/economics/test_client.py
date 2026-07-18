"""Transport, auth/project scoping, idempotency, retries, and honest results (WI-B)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
from datetime import UTC, datetime
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
    PreparedTelemetryBatch,
    funnel_eligible_event,
)
from traigent.economics import client as client_mod  # noqa: E402
from traigent.economics import schema as schema_mod  # noqa: E402
from traigent.economics.contract import (  # noqa: E402
    IDEMPOTENCY_KEY_HEADER,
    PROJECT_ID_HEADER,
)
from traigent.economics.payload import (  # noqa: E402
    build_telemetry_request,
    canonical_json,
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


# --- HIGH 1: canonical key vs wire bytes ---------------------------------------


def test_reordered_events_yield_identical_batch_key_and_wire_bytes() -> None:
    client = _client()
    ordered = {
        "event_type": "funnel_event",
        "event_id": "evt-1",
        "occurred_at": "2026-07-18T10:00:00.000Z",
        "project_ref": "proj-1",
        "stage": "eligible",
        "outcome": "entered",
    }
    reordered = {
        "outcome": "entered",
        "stage": "eligible",
        "project_ref": "proj-1",
        "occurred_at": "2026-07-18T10:00:00.000Z",
        "event_id": "evt-1",
        "event_type": "funnel_event",
    }
    a = client.prepare(
        [ordered], project_id="proj-1", sent_at="2026-07-18T10:00:01.000Z"
    )
    b = client.prepare(
        [reordered], project_id="proj-1", sent_at="2026-07-18T10:00:01.000Z"
    )
    # Same content, any key order -> same id, same key, and IDENTICAL wire bytes.
    assert a.batch_id == b.batch_id
    assert a.idempotency_key == b.idempotency_key
    assert a.content == b.content


async def test_reordered_batch_replays_with_identical_wire_bytes() -> None:
    client = _client()
    e1 = {
        "event_type": "funnel_event",
        "event_id": "evt-1",
        "occurred_at": "2026-07-18T10:00:00.000Z",
        "project_ref": "proj-1",
        "stage": "eligible",
        "outcome": "entered",
    }
    e2 = {k: e1[k] for k in reversed(list(e1))}
    a = client.prepare([e1], project_id="proj-1", sent_at="2026-07-18T10:00:01.000Z")
    b = client.prepare([e2], project_id="proj-1", sent_at="2026-07-18T10:00:01.000Z")
    mock_http = _mock_transport(client, _resp(201, _ingest_body(a)))
    await client.submit_prepared(a)
    await client.submit_prepared(b)
    calls = mock_http.post.call_args_list
    # Cross-call replay: reordered mapping sends byte-identical content and key.
    assert calls[0].kwargs["content"] == calls[1].kwargs["content"]
    assert (
        calls[0].kwargs["headers"][IDEMPOTENCY_KEY_HEADER]
        == calls[1].kwargs["headers"][IDEMPOTENCY_KEY_HEADER]
    )


# --- HIGH 2: response schema + rejection reconciliation -------------------------


async def test_response_unknown_top_level_key_fails_closed() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = _ingest_body(prepared)
    body["surprise_key"] = "x"
    _mock_transport(client, _resp(201, body))
    with pytest.raises(EconomicsResponseError):
        await client.submit_prepared(prepared)


async def test_response_malformed_timestamp_fails_closed() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = _ingest_body(prepared)
    body["received_at"] = "2026-07-18 10:00:00"  # missing T/Z
    _mock_transport(client, _resp(201, body))
    with pytest.raises(EconomicsResponseError):
        await client.submit_prepared(prepared)


async def test_duplicate_rejection_index_fails_closed() -> None:
    client = _client()
    e_a = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    e_b = funnel_eligible_event(
        "proj-1", event_id="evt-2", occurred_at="2026-07-18T10:00:00.000Z"
    )
    prepared = client.prepare([e_a, e_b], project_id="proj-1")
    body = _ingest_body(
        prepared,
        status=422,
        accepted=0,
        rejected=2,
        rejections=[
            {"event_index": 0, "reason": "tenant_scope_violation"},
            {"event_index": 0, "reason": "tenant_scope_violation"},
        ],
    )
    _mock_transport(client, _resp(422, body))
    with pytest.raises(EconomicsResponseError):
        await client.submit_prepared(prepared)


async def test_rejection_event_id_mismatch_fails_closed() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")  # event_id evt-1
    body = _ingest_body(
        prepared,
        status=422,
        accepted=0,
        rejected=1,
        rejections=[
            {
                "event_index": 0,
                "reason": "tenant_scope_violation",
                "event_id": "not-the-submitted-id",
            }
        ],
    )
    _mock_transport(client, _resp(422, body))
    with pytest.raises(EconomicsResponseError):
        await client.submit_prepared(prepared)


async def test_rejection_with_matching_event_id_is_accepted() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = _ingest_body(
        prepared,
        status=422,
        accepted=0,
        rejected=1,
        rejections=[
            {
                "event_index": 0,
                "reason": "tenant_scope_violation",
                "event_id": prepared.event_ids[0],
            }
        ],
    )
    _mock_transport(client, _resp(422, body))
    result = await client.submit_prepared(prepared)
    assert result.all_rejected
    assert result.rejections[0].event_id == prepared.event_ids[0]


# --- HIGH 3: exact-Schema fingerprint fails closed before transport ------------


async def test_fingerprint_mismatch_blocks_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        schema_mod, "compute_economics_schema_fingerprint", lambda *a, **k: "0" * 64
    )
    schema_mod.reset_request_validator_cache()
    try:
        client = _client()
        mock_http = _mock_transport(client, _resp(201))
        with pytest.raises(EconomicsSchemaUnavailable):
            await client.submit([_event()], project_id="proj-1")
        mock_http.post.assert_not_called()
    finally:
        schema_mod.reset_request_validator_cache()


# --- MEDIUM 4: Retry-After (delta-seconds + HTTP-date, injectable clock) --------


def test_retry_after_parsing_variants() -> None:
    now = datetime(2026, 7, 18, 10, 0, 0, tzinfo=UTC)
    parse = client_mod._parse_retry_after
    assert parse("5", now) == 5.0
    assert parse("100000", now) == 100000.0  # clamping happens in _backoff
    assert parse("inf", now) is None
    assert parse("nan", now) is None
    assert parse("garbage", now) is None
    assert parse(None, now) is None
    # HTTP-date: future -> delta seconds; past -> zero.
    assert parse("Sat, 18 Jul 2026 10:00:30 GMT", now) == 30.0
    assert parse("Sat, 18 Jul 2026 09:59:00 GMT", now) == 0.0


async def test_future_http_date_retry_after_is_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        client_mod, "_utcnow", lambda: datetime(2026, 7, 18, 10, 0, 0, tzinfo=UTC)
    )
    delays: list[float] = []

    async def _capture(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(client_mod.asyncio, "sleep", _capture)
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    far_future = _resp(503, headers={"Retry-After": "Sat, 18 Jul 2026 11:00:00 GMT"})
    _mock_transport(client, [far_future, _resp(201, _ingest_body(prepared))])
    await client.submit_prepared(prepared)
    assert delays and max(delays) == client_mod._MAX_RETRY_AFTER_SECONDS


# --- CRITICAL 1: only prepare()-issued batches are submittable -----------------
#
# The PRIMARY gate is provenance: a public/constructed/replaced batch is not
# submittable even when every field is internally consistent. The re-derivation
# checks (content/key/ids/project/schema/egress) are defense-in-depth for a batch
# that carries provenance yet is internally inconsistent (e.g. tampered via
# object.__setattr__); the tests below mint provenance explicitly to reach them.


def _mint_provenance(batch: PreparedTelemetryBatch) -> PreparedTelemetryBatch:
    """Grant the module provenance capability (white-box, for defense-in-depth tests)."""
    object.__setattr__(batch, "_provenance", client_mod._PREPARE_PROVENANCE)
    return batch


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return the exception and its __cause__/__context__ chain (bounded)."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and not any(c is current for c in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _prepared_from_body(
    body: dict[str, Any],
    *,
    project_id: str = "proj-1",
    provenanced: bool = False,
    **overrides: Any,
) -> PreparedTelemetryBatch:
    """Construct a PreparedTelemetryBatch directly from a (possibly forged) body."""
    events = body["events"]
    fields: dict[str, Any] = {
        "project_id": project_id,
        "idempotency_key": body["idempotency_key"],
        "batch_id": body["batch_id"],
        "submitted": len(events),
        "content": canonical_json(body).encode("utf-8"),
        "body": body,
        "event_ids": tuple(str(e["event_id"]) for e in events),
    }
    fields.update(overrides)
    batch = PreparedTelemetryBatch(**fields)
    if provenanced:
        _mint_provenance(batch)
    return batch


# -- primary gate: provenance --------------------------------------------------


async def test_direct_construction_fully_valid_is_non_submittable() -> None:
    # Fully schema-valid, canonical, self-consistent — but not issued by prepare().
    client = _client()
    mock_http = _mock_transport(client, _resp(201))
    body = build_telemetry_request([_event()])
    forged = _prepared_from_body(body)  # no provenance
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(forged)
    mock_http.post.assert_not_called()


async def test_replace_with_no_changes_loses_provenance() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    tampered = dataclasses.replace(prepared)  # every public field identical
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(tampered)
    mock_http.post.assert_not_called()


async def test_replace_all_fields_recomputed_loses_provenance() -> None:
    # Rebuild every public/body/content/key/id/count/scope field self-consistently.
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = dict(prepared.body)
    events = body["events"]
    tampered = dataclasses.replace(
        prepared,
        project_id="proj-1",
        idempotency_key=body["idempotency_key"],
        batch_id=body["batch_id"],
        submitted=len(events),
        content=canonical_json(body).encode("utf-8"),
        body=body,
        event_ids=tuple(str(e["event_id"]) for e in events),
    )
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(tampered)
    mock_http.post.assert_not_called()


async def test_provenance_refusal_is_payload_free() -> None:
    # A hand-built batch carrying a sensitive-looking value is refused without
    # the value (or its evidence pointer) appearing in the error or its chain.
    client = _client()
    _mock_transport(client, _resp(201))
    body = build_telemetry_request([_event()])
    body["events"] = [_FULL_RUN_EVENT]  # carries the SENSITIVE evidence pointer
    forged = _prepared_from_body(body, event_ids=("evt-run-1",))
    with pytest.raises(EconomicsTelemetryContractError) as excinfo:
        await client.submit_prepared(forged)
    chain = " ".join(str(e) for e in _exception_chain(excinfo.value))
    assert "SENSITIVE-incident-ledger-4k-escalation" not in chain
    assert "4000" not in chain


async def test_exact_prepared_object_submits_and_retries_identical() -> None:
    # The object returned by prepare() carries provenance and submits; a retry
    # replays byte-identical content and key.
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    mock_http = _mock_transport(
        client, [_resp(503), _resp(201, _ingest_body(prepared))]
    )
    result = await client.submit_prepared(prepared)
    assert result.fully_accepted
    calls = mock_http.post.call_args_list
    assert calls[0].kwargs["content"] == calls[1].kwargs["content"]
    assert (
        calls[0].kwargs["headers"][IDEMPOTENCY_KEY_HEADER]
        == calls[1].kwargs["headers"][IDEMPOTENCY_KEY_HEADER]
    )


# -- defense-in-depth: provenanced but internally inconsistent -----------------


async def test_provenanced_content_tamper_is_refused() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    tampered = _mint_provenance(
        dataclasses.replace(prepared, content=b'{"forged":true}')
    )
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="content does not match"):
        await client.submit_prepared(tampered)
    mock_http.post.assert_not_called()


async def test_provenanced_arbitrary_body_is_refused() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    forged_body = dict(prepared.body)
    forged_body["events"] = [dict(forged_body["events"][0], surprise="x")]
    tampered = _mint_provenance(
        dataclasses.replace(
            prepared,
            body=forged_body,
            content=canonical_json(forged_body).encode("utf-8"),
        )
    )
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError):
        await client.submit_prepared(tampered)
    mock_http.post.assert_not_called()


async def test_provenanced_withheld_value_body_is_refused() -> None:
    client = _client()
    mock_http = _mock_transport(client, _resp(201))
    forged_run = dict(_FULL_RUN_EVENT)
    forged_run["characterization"] = {
        "overrides": {"loss_per_bad_output_usd": 4000},
        "field_reports": [
            {
                "field": "loss_per_bad_output_usd",
                "provenance": "asked",
                "confidence": 1.0,
                "sharing_outcome": "withheld_by_policy",  # withheld BUT present
            }
        ],
    }
    body = build_telemetry_request([_event()])
    body["events"] = [forged_run]
    forged = _prepared_from_body(body, provenanced=True, event_ids=("evt-run-1",))
    with pytest.raises((EgressPolicyError, EconomicsTelemetryContractError)):
        await client.submit_prepared(forged)
    mock_http.post.assert_not_called()


async def test_provenanced_idempotency_key_tamper_is_refused() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    tampered = _mint_provenance(
        dataclasses.replace(prepared, idempotency_key="econ-tel-differentkey")
    )
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="idempotency key"):
        await client.submit_prepared(tampered)
    mock_http.post.assert_not_called()


async def test_provenanced_event_ids_tamper_is_refused() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    tampered = _mint_provenance(dataclasses.replace(prepared, event_ids=("evt-other",)))
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="event ids"):
        await client.submit_prepared(tampered)
    mock_http.post.assert_not_called()


async def test_provenanced_project_tamper_is_refused() -> None:
    client = _client()
    prepared = client.prepare([_event("proj-1")], project_id="proj-1")
    tampered = _mint_provenance(
        dataclasses.replace(prepared, project_id="other-project")
    )
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="project_ref"):
        await client.submit_prepared(tampered)
    mock_http.post.assert_not_called()


async def test_prepared_submit_fails_closed_when_schema_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    prepared = client.prepare(
        [_event()], project_id="proj-1"
    )  # built while schema present, carries provenance
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    monkeypatch.setattr(
        schema_mod, "compute_economics_schema_fingerprint", lambda *a, **k: "0" * 64
    )
    schema_mod.reset_request_validator_cache()
    try:
        with pytest.raises(EconomicsSchemaUnavailable):
            await client.submit_prepared(prepared)
        mock_http.post.assert_not_called()
    finally:
        schema_mod.reset_request_validator_cache()


# --- HIGH 2: all-rejected must be 422, never a fresh 201 -----------------------


async def test_all_rejected_fresh_201_is_refused() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = _ingest_body(
        prepared,
        status=201,
        accepted=0,
        rejected=1,
        rejections=[{"event_index": 0, "reason": "tenant_scope_violation"}],
    )
    _mock_transport(client, _resp(201, body))
    with pytest.raises(EconomicsResponseError):
        await client.submit_prepared(prepared)


async def test_partial_rejection_201_is_accepted() -> None:
    client = _client()
    e_a = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    e_b = funnel_eligible_event(
        "proj-1", event_id="evt-2", occurred_at="2026-07-18T10:00:00.000Z"
    )
    prepared = client.prepare([e_a, e_b], project_id="proj-1")
    body = _ingest_body(
        prepared,
        status=201,
        accepted=1,
        rejected=1,
        rejections=[{"event_index": 1, "reason": "unknown_reference"}],
    )
    _mock_transport(client, _resp(201, body))
    result = await client.submit_prepared(prepared)
    assert result.any_rejected
    assert not result.all_rejected
    assert result.accepted == 1


async def test_duplicate_only_replay_200_is_accepted() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    body = _ingest_body(prepared, status=200, replayed=True, accepted=0, duplicate=1)
    _mock_transport(client, _resp(200, body))
    result = await client.submit_prepared(prepared)
    assert result.replayed
    assert result.all_duplicate
    assert result.no_rejections


# --- redirect/credential safety -------------------------------------------------


async def test_redirect_is_not_treated_as_success() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    _mock_transport(client, _resp(302, headers={"Location": "https://evil.example"}))
    with pytest.raises(EconomicsTelemetryTransportError):
        await client.submit_prepared(prepared)


def test_http_client_does_not_follow_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _FakeAsyncClient)
    client = _client()
    client._get_client()
    assert captured.get("follow_redirects") is False
