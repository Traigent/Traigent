"""Fail-closed request validation against the exact economics Schema (WI-B)."""

from __future__ import annotations

import pytest

import traigent_schema.validator as tsv

from traigent.economics import build_telemetry_request, funnel_eligible_event
from traigent.economics import schema as schema_mod
from traigent.economics.errors import (
    EconomicsSchemaUnavailable,
    EconomicsTelemetryContractError,
)

_REQUEST_SCHEMA = schema_mod.REQUEST_SCHEMA_NAME


@pytest.fixture(autouse=True)
def _reset_validator_cache() -> None:
    schema_mod.reset_request_validator_cache()
    yield
    schema_mod.reset_request_validator_cache()


def _valid_body() -> dict:
    event = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    return build_telemetry_request([event])


def test_valid_request_passes_against_installed_schema() -> None:
    schema_mod.validate_request_or_fail(_valid_body())


def test_extra_event_key_is_rejected() -> None:
    body = _valid_body()
    body["events"][0]["surprise_extra_key"] = "leaked"
    with pytest.raises(EconomicsTelemetryContractError):
        schema_mod.validate_request_or_fail(body)


def test_old_schema_without_economics_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OldValidator:
        available_schemas: list[str] = ["some_other_schema"]

    monkeypatch.setattr(tsv, "SchemaValidator", _OldValidator)
    schema_mod.reset_request_validator_cache()
    with pytest.raises(EconomicsSchemaUnavailable):
        schema_mod.validate_request_or_fail(_valid_body())


def test_validator_that_raises_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Throwing:
        available_schemas = [
            _REQUEST_SCHEMA,
            *schema_mod.RESPONSE_SCHEMA_BY_STATUS.values(),
        ]

        def validate_json(self, body: dict, name: str) -> list[str]:
            raise RuntimeError("boom")

    monkeypatch.setattr(tsv, "SchemaValidator", _Throwing)
    schema_mod.reset_request_validator_cache()
    with pytest.raises(EconomicsSchemaUnavailable):
        schema_mod.validate_request_or_fail(_valid_body())


def test_validator_construction_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Broken:
        def __init__(self) -> None:
            raise ImportError("missing transitive dep")

    monkeypatch.setattr(tsv, "SchemaValidator", _Broken)
    schema_mod.reset_request_validator_cache()
    with pytest.raises(EconomicsSchemaUnavailable):
        schema_mod.get_request_validator()


def test_successful_validator_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real = tsv.SchemaValidator

    class _Counting(real):  # type: ignore[valid-type, misc]
        def __init__(self) -> None:
            calls["n"] += 1
            super().__init__()

    monkeypatch.setattr(tsv, "SchemaValidator", _Counting)
    schema_mod.reset_request_validator_cache()
    schema_mod.get_request_validator()
    schema_mod.get_request_validator()
    assert calls["n"] == 1


# --- exact-Schema content fingerprint ------------------------------------------


def test_installed_fingerprint_matches_pinned_expected() -> None:
    # The runtime-computed fingerprint of the installed exact-commit contract
    # equals the pinned constant — the version string is never trusted alone.
    assert (
        schema_mod.compute_economics_schema_fingerprint()
        == schema_mod.EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT
    )


def test_fingerprint_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A name-only impostor (right schema names, different content) is rejected.
    monkeypatch.setattr(
        schema_mod,
        "compute_economics_schema_fingerprint",
        lambda *a, **k: "0" * 64,
    )
    schema_mod.reset_request_validator_cache()
    with pytest.raises(EconomicsSchemaUnavailable):
        schema_mod.validate_request_or_fail(_valid_body())


def test_fingerprint_is_content_sensitive(tmp_path) -> None:
    import json
    import os

    import traigent_schema

    src = os.path.join(
        os.path.dirname(traigent_schema.__file__), "schemas", "economics"
    )
    for name in os.listdir(src):
        if name.endswith(".json"):
            with open(os.path.join(src, name), encoding="utf-8") as fh:
                doc = json.load(fh)
            # Mutate one file's content.
            if name == "economics_common_schema.json":
                doc["definitions"]["ContractId"]["const"] = "tampered"
            with open(tmp_path / name, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
    mutated = schema_mod.compute_economics_schema_fingerprint(str(tmp_path))
    assert mutated != schema_mod.EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT


# --- response validation --------------------------------------------------------


def _valid_response(*, status: int = 201) -> dict:
    return {
        "contract": "economics_telemetry",
        "contract_version": "1.0.0",
        "batch_id": "batch-1",
        "idempotency_key": "econ-tel-abcdefgh",
        "received_at": "2026-07-18T10:00:00.000Z",
        "replayed": status == 200,
        "counts": {"submitted": 1, "accepted": 1, "duplicate": 0, "rejected": 0},
        "rejections": [],
    }


def test_valid_response_passes_schema() -> None:
    schema_mod.validate_response_or_fail(_valid_response(status=201), http_status=201)
    schema_mod.validate_response_or_fail(_valid_response(status=200), http_status=200)


def test_response_unknown_top_level_key_rejected() -> None:
    from traigent.economics.errors import EconomicsResponseError

    body = _valid_response()
    body["surprise"] = "extra"
    with pytest.raises(EconomicsResponseError):
        schema_mod.validate_response_or_fail(body, http_status=201)


def test_response_malformed_timestamp_rejected() -> None:
    from traigent.economics.errors import EconomicsResponseError

    body = _valid_response()
    body["received_at"] = "not-a-timestamp"
    with pytest.raises(EconomicsResponseError):
        schema_mod.validate_response_or_fail(body, http_status=201)


def test_response_status_flag_binding_enforced_by_schema() -> None:
    from traigent.economics.errors import EconomicsResponseError

    # A 200 body with replayed=false violates the replay schema's const.
    body = _valid_response(status=200)
    body["replayed"] = False
    with pytest.raises(EconomicsResponseError):
        schema_mod.validate_response_or_fail(body, http_status=200)


# --- boundary errors are payload-free (no cause/context leak) -------------------


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and not any(c is current for c in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def test_request_validator_exception_is_payload_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "SENSITIVE-REQUEST-VALUE-9931"

    class _Throwing:
        available_schemas = [
            _REQUEST_SCHEMA,
            *schema_mod.RESPONSE_SCHEMA_BY_STATUS.values(),
        ]

        def validate_json(self, body: dict, name: str) -> list[str]:
            raise RuntimeError(f"validator blew up on {marker}")

    monkeypatch.setattr(tsv, "SchemaValidator", _Throwing)
    schema_mod.reset_request_validator_cache()
    with pytest.raises(EconomicsSchemaUnavailable) as excinfo:
        schema_mod.validate_request_or_fail(_valid_body())
    assert all(marker not in str(link) for link in _exception_chain(excinfo.value))


def test_response_validator_exception_is_payload_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.economics.errors import EconomicsResponseError

    marker = "SENSITIVE-RESPONSE-VALUE-7777"

    class _Throwing:
        def validate_json(self, body: dict, name: str) -> list[str]:
            raise RuntimeError(f"validator blew up on {marker}")

    monkeypatch.setattr(schema_mod, "_get_bundle", lambda: _Throwing())
    with pytest.raises(EconomicsResponseError) as excinfo:
        schema_mod.validate_response_or_fail({"any": "body"}, http_status=201)
    assert all(marker not in str(link) for link in _exception_chain(excinfo.value))
