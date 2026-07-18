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
        available_schemas = [_REQUEST_SCHEMA]

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
