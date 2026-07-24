"""Transport, auth/project scoping, idempotency, retries, and honest results (WI-B)."""

from __future__ import annotations

import copy
import dataclasses
import gc
import hashlib
import hmac
import importlib.util
import json
import logging
import pickle
import weakref
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


def test_http_client_carries_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """When an API key is configured, it rides on every request via the client's
    default headers (never re-derived per call, never dropped)."""
    captured: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _FakeAsyncClient)
    client = _client()
    client._get_client()
    assert (
        captured.get("headers", {}).get("X-API-Key") == "test-key"
    )  # pragma: allowlist secret


def test_http_client_carries_jwt_header_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _FakeAsyncClient)
    client = EconomicsTelemetryClient(
        backend_url="https://api.traigent.ai", jwt_token="jwt-abc"
    )
    client._get_client()
    assert captured.get("headers", {}).get("Authorization") == "Bearer jwt-abc"


async def test_submit_without_credential_never_sends_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No API key/JWT resolved anywhere -> fail-safe no-op.

    The prior behavior was to build an httpx client with EMPTY auth headers and
    POST anyway (unauthenticated telemetry egress). The client must instead
    refuse locally, before any bytes leave the machine -- exactly like
    EgressPolicyError / EconomicsSchemaUnavailable already do for other
    pre-transport failures.
    """
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CredentialManager.get_api_key",
        lambda *a, **k: None,
    )
    client = EconomicsTelemetryClient(backend_url="https://api.traigent.ai")
    assert client.api_key is None
    assert client.jwt_token is None

    mock_http = _mock_transport(client, _resp(201))
    with pytest.raises(EconomicsTelemetryAuthError, match="no API key or JWT"):
        await client.submit([_event()], project_id="proj-1")

    mock_http.post.assert_not_called()


async def test_submit_prepared_without_credential_never_sends_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same fail-safe gate on the submit_prepared() path (prepare() has no I/O,
    so a caller could hold a validly-prepared batch and only lose credentials —
    e.g. a revoked/rotated key — before submitting it)."""
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    # Simulate the credential disappearing between prepare() and submit.
    client.api_key = None
    client.jwt_token = None

    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryAuthError, match="no API key or JWT"):
        await client.submit_prepared(prepared)
    mock_http.post.assert_not_called()


async def test_submit_without_credential_logs_debug_not_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CredentialManager.get_api_key",
        lambda *a, **k: None,
    )
    client = EconomicsTelemetryClient(backend_url="https://api.traigent.ai")
    mock_http = _mock_transport(client, _resp(201))
    with caplog.at_level(logging.DEBUG, logger="traigent.economics.client"):
        with pytest.raises(EconomicsTelemetryAuthError):
            await client.submit([_event()], project_id="proj-1")
    assert "economics telemetry request not sent" in caplog.text
    mock_http.post.assert_not_called()


# --- privacy: no sensitive values in logs --------------------------------------


async def test_no_sensitive_values_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    client = _client()
    prepared = client.prepare([_FULL_RUN_EVENT], project_id="proj-1")
    _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with caplog.at_level(logging.DEBUG, logger="traigent.economics.client"):
        await client.submit_prepared(prepared)
    assert "SENSITIVE-incident-ledger-4k-escalation" not in caplog.text
    assert "4000" not in caplog.text


# --- P1: PreparedTelemetryBatch repr must not leak payload / evidence -----------
#
# The default dataclass repr exposed `content`/`body` — the raw payload including
# shared evidence pointers — through repr(), str(), f"{batch!r}", and `%r` logging.
# `content`/`body` are now `field(repr=False)`; only identifiers appear. The wire
# bytes are unchanged (redaction is repr-only), so byte-identity is preserved.


async def test_prepared_batch_repr_does_not_leak_payload_or_evidence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client()
    # _FULL_RUN_EVENT carries a unique sensitive evidence pointer inside its
    # payload body (NOT in any identifier field). An alphabetic sentinel cannot
    # collide with the hex-only idempotency_key/batch_id, so the absence checks
    # are robust regardless of the run-varying hashes.
    prepared = client.prepare([_FULL_RUN_EVENT], project_id="proj-1")
    sentinel = "SENSITIVE-incident-ledger-4k-escalation"

    # The sentinel is genuinely on the wire — redaction is repr-only, not a payload
    # change — so the existing byte-identity guarantee is untouched.
    assert sentinel.encode("utf-8") in prepared.content

    renderings = [
        repr(prepared),
        str(prepared),
        f"{prepared!r}",
    ]
    for rendering in renderings:
        assert sentinel not in rendering, rendering

    # A logging call formatted with %r must not leak the payload either. (logging
    # renders the record via %-formatting, so this exercises the %r path directly.)
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("traigent.economics.client").info(
            "prepared batch %r", prepared
        )
    assert sentinel not in caplog.text

    # Identifiers stay visible so the repr remains useful for debugging.
    shown = repr(prepared)
    assert prepared.batch_id in shown
    assert prepared.idempotency_key in shown
    assert "evt-run-1" in shown  # the event id is an identifier, not payload


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


# --- P1: backend rejection `detail` is validated then discarded, never surfaced -


async def test_rejection_detail_is_discarded_and_never_surfaced() -> None:
    # A schema-valid 422 whose rejection `detail` carries a unique sensitive
    # sentinel (a would-be echoed payload value). index/reason/event_id must still
    # surface; the sentinel must appear in NO public field, repr, str, or the
    # rejection reasons.
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    sentinel = "SENSITIVE-REJECT-DETAIL-loss_per_bad_output_usd=88888"
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
                "detail": sentinel,
            }
        ],
    )
    _mock_transport(client, _resp(422, body))
    result = await client.submit_prepared(prepared)

    rejection = result.rejections[0]
    # Identifiers are surfaced correctly.
    assert rejection.event_index == 0
    assert rejection.reason == "tenant_scope_violation"
    assert rejection.event_id == prepared.event_ids[0]

    # `detail` is not even a field of the public type, and the sentinel appears in
    # no public field, repr, or str of the rejection or the whole result.
    assert not hasattr(rejection, "detail")
    haystack = " ".join([repr(rejection), str(rejection), repr(result), str(result)])
    assert sentinel not in haystack
    assert "88888" not in haystack
    assert sentinel not in result.rejection_reasons


async def test_malformed_rejection_detail_still_fails_closed() -> None:
    # The `detail` shape check is preserved: a non-string detail (which could smuggle
    # structured payload) still raises before any result is returned.
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
                "detail": {"nested": "SENSITIVE-structured-leak"},
            }
        ],
    )
    _mock_transport(client, _resp(422, body))
    with pytest.raises(EconomicsResponseError):
        await client.submit_prepared(prepared)


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


def _recover_issuance_secret() -> bytes:
    """Recover the closure-held issuance secret via deliberate introspection.

    This is the honest statement of the trust boundary. Minting is NOT reachable
    through any module global: there is no ``_ISSUED_BATCHES`` registry, no
    ``_verify_issuance`` module global, and no module-level mint/stamp callable.
    The verifier and the secret live only inside closures installed ON THE CLASS
    (``prepare`` and ``_require_issued``). To forge a token a white-box attacker
    (or this test) must walk those class-attribute closures to reach the secret,
    exactly as the client docstring states — an ordinary import cannot. Used only
    to exercise defense-in-depth (a forged-provenance but internally inconsistent
    batch must still fail the re-derivation checks).
    """
    seen: list[Any] = []
    # Start from the installed class-attribute closures, never a module global.
    stack: list[Any] = [
        client_mod.EconomicsTelemetryClient.prepare,
        client_mod.EconomicsTelemetryClient._require_issued,
    ]
    while stack:
        fn = stack.pop()
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                val = cell.cell_contents
            except ValueError:  # pragma: no cover - empty cell
                continue
            if isinstance(val, (bytes, bytearray)) and len(val) == 32:
                return bytes(val)
            if callable(val) and all(val is not s for s in seen):
                seen.append(val)
                stack.append(val)
    raise AssertionError(
        "issuance secret is not reachable even via closure introspection"
    )


def _is_issued(batch: PreparedTelemetryBatch) -> bool:
    """True iff the installed class verifier accepts this batch's issuance token."""
    try:
        client_mod.EconomicsTelemetryClient._require_issued(batch)
        return True
    except EconomicsTelemetryContractError:
        return False


def _register_issued(batch: PreparedTelemetryBatch) -> PreparedTelemetryBatch:
    """Forge a valid identity-bound issuance token on an arbitrary batch.

    White-box helper: mints the HMAC directly from the recovered closure secret
    (see :func:`_recover_issuance_secret`) and stamps it, so defense-in-depth
    tests can reach the re-derivation checks with a batch that has valid IDENTITY
    provenance but an internally inconsistent field. Reaching the secret requires
    closure introspection; no module-global mint exists.
    """
    token = hmac.new(
        _recover_issuance_secret(), str(id(batch)).encode("ascii"), hashlib.sha256
    ).digest()
    object.__setattr__(batch, client_mod._ISSUANCE_ATTR, token)
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
        _register_issued(batch)
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
    tampered = _register_issued(
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
    tampered = _register_issued(
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
    tampered = _register_issued(
        dataclasses.replace(prepared, idempotency_key="econ-tel-differentkey")
    )
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="idempotency key"):
        await client.submit_prepared(tampered)
    mock_http.post.assert_not_called()


async def test_provenanced_event_ids_tamper_is_refused() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    tampered = _register_issued(dataclasses.replace(prepared, event_ids=("evt-other",)))
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="event ids"):
        await client.submit_prepared(tampered)
    mock_http.post.assert_not_called()


async def test_provenanced_project_tamper_is_refused() -> None:
    client = _client()
    prepared = client.prepare([_event("proj-1")], project_id="proj-1")
    tampered = _register_issued(
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


# --- CRITICAL: identity issuance — copies/deepcopies/unpickled are refused ------


async def test_copy_copy_of_prepared_is_non_submittable() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    shallow = copy.copy(prepared)
    assert shallow is not prepared
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(shallow)
    mock_http.post.assert_not_called()


async def test_deepcopy_of_prepared_is_non_submittable() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    try:
        deep = copy.deepcopy(prepared)
    except Exception:
        # The read-only MappingProxyType body cannot be deep-copied at all —
        # a prepared batch cannot even be reconstituted this way. Nothing to submit.
        mock_http.post.assert_not_called()
        return
    assert deep is not prepared
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(deep)
    mock_http.post.assert_not_called()


async def test_pickle_roundtrip_of_prepared_is_non_submittable() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    try:
        data = pickle.dumps(prepared)
    except Exception:
        # Policy: a batch that cannot even be pickled cannot be reconstituted and
        # submitted across a round-trip. Nothing reaches transport.
        mock_http.post.assert_not_called()
        return
    restored = pickle.loads(data)
    assert restored is not prepared
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(restored)
    mock_http.post.assert_not_called()


async def test_cross_client_issued_batch_is_submittable() -> None:
    # Documented in-process rule: a batch prepared by one client instance is
    # submittable by another client instance in the same process.
    issuer = _client()
    submitter = _client()
    prepared = issuer.prepare([_event()], project_id="proj-1")
    _mock_transport(submitter, _resp(201, _ingest_body(prepared)))
    result = await submitter.submit_prepared(prepared)
    assert result.fully_accepted


def test_no_module_global_issuance_registry_or_mint() -> None:
    # The removed vulnerable primitives are gone: no writable registry, no
    # sentinel, no minting authority reachable from module globals.
    assert not hasattr(client_mod, "_ISSUED_BATCHES")
    assert not hasattr(client_mod, "_PREPARE_PROVENANCE")
    # No module global is a mutable container that could serve as a forge-able
    # issuance registry. Immutable constants (frozensets, ints, strings) are fine;
    # a dict/list/set/weak-map that maps identity -> batch is exactly the removed
    # attack surface.
    mutable_containers = (
        dict,
        list,
        set,
        weakref.WeakValueDictionary,
        weakref.WeakKeyDictionary,
    )
    for name, val in vars(client_mod).items():
        if name.startswith("__"):  # skip dunders (e.g. __all__)
            continue
        assert not isinstance(val, mutable_containers), (
            f"module global {name!r} is a mutable container reachable for forging "
            "issuance"
        )
    # No module-global verifier/mint remains (the old `_verify_issuance` surface
    # is gone); the verifier lives only in a class-installed closure.
    assert not hasattr(client_mod, "_verify_issuance")
    # The installed class verifier refuses an unissued batch (fail-closed), and it
    # is the ONLY issuance capability — there is no module-level mint.
    unissued = _prepared_from_body(build_telemetry_request([_event()]))
    assert _is_issued(unissued) is False


def test_no_module_global_verifier_or_mint_to_reassign() -> None:
    # Module-global reassignment/bypass audit: there is nothing at module scope to
    # monkeypatch into an always-accept verifier or a mint. The verifier + secret
    # live inside class-attribute closures only.
    assert not hasattr(client_mod, "_verify_issuance")
    assert not hasattr(client_mod, "_ISSUED_BATCHES")
    assert not hasattr(client_mod, "_PREPARE_PROVENANCE")
    # The one-shot installer is deleted after its single import-time call, so it is
    # not a module-global that could be re-invoked to recapture a hostile builder.
    assert not hasattr(client_mod, "_install_issuance")
    # No module global is a callable that verifies/mints, and none is the 32-byte
    # secret. (Any callable named like the removed surface would be a bypass hook.)
    for name, val in vars(client_mod).items():
        if name.startswith("__"):
            continue
        assert not (isinstance(val, (bytes, bytearray)) and len(val) == 32), (
            f"module global {name!r} looks like the issuance secret"
        )
        assert "verify_issuance" not in name
        assert "install_issuance" not in name
        assert "mint" not in name.lower()
    # The verifier IS reachable only by deliberately walking class closures.
    assert isinstance(_recover_issuance_secret(), bytes)


async def test_installer_is_not_reinvokable_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exact re-install bypass: monkeypatch _build_prepared to return a forged
    # batch, then try to re-run the installer to recapture that hostile builder
    # into the mint. The one-shot installer is gone, so the recapture cannot happen.
    client = _client()
    mock_http = _mock_transport(client, _resp(201))
    forged = _prepared_from_body(build_telemetry_request([_event()]))
    assert not _is_issued(forged)

    # There is no installer name left in module globals to re-invoke.
    assert not hasattr(client_mod, "_install_issuance")
    monkeypatch.setattr(
        client_mod.EconomicsTelemetryClient,
        "_build_prepared",
        lambda self, *args, **kwargs: forged,
    )
    with pytest.raises(AttributeError):
        client_mod._install_issuance()  # type: ignore[attr-defined]

    # The mint was never re-pointed at the hostile builder: forged stays unissued
    # and cannot be submitted.
    assert not _is_issued(forged)
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(forged)
    mock_http.post.assert_not_called()


def test_issuance_secret_is_not_a_module_global() -> None:
    # Honest boundary: the process-random secret is never a plain module attribute
    # (an ordinary import cannot read it), yet it IS recoverable by deliberately
    # walking the verifier's closure — an API trust boundary, not a crypto seal.
    module_secrets = [
        name
        for name, val in vars(client_mod).items()
        if isinstance(val, (bytes, bytearray)) and len(val) == 32
    ]
    assert module_secrets == []
    assert isinstance(_recover_issuance_secret(), bytes)


async def test_exact_import_and_register_reproducer_cannot_post() -> None:
    # The precise prior exploit: `import ...client as m; m._ISSUED_BATCHES[
    # id(forged)] = forged; submit`. A fully schema-valid, self-consistent forged
    # batch is built, then the registry write is attempted.
    client = _client()
    mock_http = _mock_transport(client, _resp(201))
    body = build_telemetry_request([_event()])
    forged = _prepared_from_body(body)  # self-consistent, unissued
    # The registry the exploit wrote to no longer exists.
    with pytest.raises(AttributeError):
        client_mod._ISSUED_BATCHES[id(forged)] = forged  # type: ignore[attr-defined]
    # With no mint/registry reachable, the forged batch is refused before transport.
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(forged)
    mock_http.post.assert_not_called()


async def test_evil_issuer_normal_call_cannot_mint_forged_object() -> None:
    # Normal-call mint bypass: an object whose _build_prepared returns a pre-forged
    # batch must NOT get that forged object returned or HMAC-stamped. The issuing
    # prepare calls the captured genuine builder, never self._build_prepared.
    client = _client()
    mock_http = _mock_transport(client, _resp(201))
    forged = _prepared_from_body(build_telemetry_request([_event()]))
    assert not _is_issued(forged)

    class EvilIssuer:
        def _build_prepared(self, *args: Any, **kwargs: Any) -> PreparedTelemetryBatch:
            return forged

    # The exact reproducer: call the unbound issuing prepare with an evil self.
    try:
        maybe = EconomicsTelemetryClient.prepare(
            EvilIssuer(), [_event()], project_id="proj-1"
        )
    except (AttributeError, TypeError, EconomicsTelemetryContractError):
        # The genuine, dispatch-free builder rejects a self lacking the real
        # client machinery — the forged object is never produced.
        maybe = None
    # Whatever happened, prepare NEVER returned the pre-forged object...
    assert maybe is not forged
    # ...and the forged object was never stamped, so it stays non-submittable.
    assert not _is_issued(forged)
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(forged)
    mock_http.post.assert_not_called()


async def test_evil_subclass_prepare_uses_genuine_builder_not_override() -> None:
    # A subclass that overrides _build_prepared to return a forged batch: prepare
    # uses the CAPTURED genuine builder, so the override is bypassed — the forged
    # object is neither returned nor stamped.
    forged = _prepared_from_body(build_telemetry_request([_event()]))

    class EvilSubclass(EconomicsTelemetryClient):
        def _build_prepared(self, *args: Any, **kwargs: Any) -> PreparedTelemetryBatch:
            return forged

    evil = EvilSubclass(backend_url="https://api.traigent.ai", api_key="k")
    result = evil.prepare([_event()], project_id="proj-1")
    assert result is not forged  # subclass override bypassed
    assert not _is_issued(forged)  # the injected object was never stamped
    assert _is_issued(result)  # the genuinely-built batch is issued

    submitter = _client()
    mock_http = _mock_transport(submitter, _resp(201))
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await submitter.submit_prepared(forged)
    mock_http.post.assert_not_called()


async def test_equal_but_distinct_batch_is_non_submittable() -> None:
    # Equality trick: a twin that compares == to the issued batch (dataclass eq)
    # but is a different object. Verification is identity-bound, not equality-based,
    # so the twin is refused.
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    twin = _prepared_from_body(dict(prepared.body), project_id=prepared.project_id)
    assert twin == prepared  # equal by value (dict == MappingProxyType)...
    assert twin is not prepared  # ...but a distinct identity
    mock_http = _mock_transport(client, _resp(201, _ingest_body(prepared)))
    with pytest.raises(EconomicsTelemetryContractError, match="was not issued by"):
        await client.submit_prepared(twin)
    mock_http.post.assert_not_called()


def test_prepared_batch_is_not_retained_by_module() -> None:
    # Lifetime / leak-free: no module-level registry retains the batch, so it is
    # collected once the caller drops it (and its identity-bound token dies with
    # it). This is the direct proof that issuance holds no memory.
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    ref = weakref.ref(prepared)
    del prepared
    gc.collect()
    assert ref() is None


def test_repeated_prepare_grows_no_module_state() -> None:
    # No per-batch module bookkeeping accumulates across many prepares.
    client = _client()
    baseline = len(vars(client_mod))
    kept = [client.prepare([_event()], project_id="proj-1") for _ in range(50)]
    assert len(vars(client_mod)) == baseline
    # And the batches are collectable — nothing pins them at module scope.
    refs = [weakref.ref(batch) for batch in kept]
    del kept
    gc.collect()
    assert all(ref() is None for ref in refs)


# --- HIGH: boundary errors are payload-free (no cause/context leak) -------------


async def test_invalid_json_response_error_is_payload_free() -> None:
    client = _client()
    prepared = client.prepare([_event()], project_id="proj-1")
    marker = "SENSITIVE-RAW-RESPONSE-BODY-4242"
    bad = MagicMock()
    bad.status_code = 201
    # JSONDecodeError.doc carries the raw response body — it must not leak.
    bad.json.side_effect = json.JSONDecodeError("Expecting value", marker, 0)
    bad.headers = {}
    _mock_transport(client, bad)
    with pytest.raises(EconomicsResponseError) as excinfo:
        await client.submit_prepared(prepared)
    chain = _exception_chain(excinfo.value)
    assert all(marker not in str(link) for link in chain)
    # No JSONDecodeError (with its .doc) rides along the cause/context chain.
    for link in chain:
        assert getattr(link, "doc", None) != marker
    assert not any(isinstance(link, json.JSONDecodeError) for link in chain)


# --- HIGH: canonical UTF-8 chokepoint — lone surrogates fail closed, no leak ----
#
# `ensure_ascii=False` lets a lone surrogate through json.dumps as a valid str;
# the later `.encode("utf-8")` raises UnicodeEncodeError whose `.object` carries
# the ENTIRE canonical payload. The wire path routes through
# `canonical_json_bytes`, which confines that error to its handler and raises a
# fresh payload-free contract error OUTSIDE it (cause/context both None).

_LONE_SURROGATE = "\ud800"  # valid Python str, invalid UTF-8


def test_wire_serialize_lone_surrogate_is_payload_free() -> None:
    # The wire serialization path (_serialize -> canonical_json_bytes) fails closed
    # on a lone surrogate without leaking the payload via message/cause/context.
    marker = f"SENSITIVE-WIRE-{_LONE_SURROGATE}-9999"
    body = {"contract": "economics_telemetry", "events": [{"note": marker}]}
    with pytest.raises(EconomicsTelemetryContractError) as excinfo:
        client_mod._serialize(body)
    chain = _exception_chain(excinfo.value)
    assert all("SENSITIVE-WIRE" not in str(link) for link in chain)
    assert not any(isinstance(link, UnicodeError) for link in chain)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


async def test_submit_lone_surrogate_event_fails_closed_zero_post(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # End-to-end: a lone surrogate anywhere in an event fails closed during build
    # (batch-id derivation), so nothing is serialized or POSTed, and neither the
    # error chain nor the logs carry the payload.
    client = _client()
    mock_http = _mock_transport(client, _resp(201))
    marker = f"SENSITIVE-SUBMIT-{_LONE_SURROGATE}-1212"
    event = _event()
    event["note"] = marker
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EconomicsTelemetryContractError) as excinfo:
            await client.submit([event], project_id="proj-1")
    mock_http.post.assert_not_called()
    chain = _exception_chain(excinfo.value)
    assert all("SENSITIVE-SUBMIT" not in str(link) for link in chain)
    assert not any(isinstance(link, UnicodeError) for link in chain)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert "SENSITIVE-SUBMIT" not in caplog.text
