"""The final Backend's deploy rules (6ee39655b), enforced by the strict fake and driven here."""

from __future__ import annotations

import json
from typing import Any

import pytest

from traigent.runtime.apply import _receipt
from traigent.runtime.hosted import ReactiveError, deploy_key
from traigent.runtime.model import RuntimeRequirementError

from .conftest import Reactive
from .reactive_fake import make_ref
from .test_deploy_controller import FP, GATE, SPEC, approved_deploy

PUBLISH = "publishTvlDeployResult"


def _reserved(reactive: Reactive, *, started: bool) -> tuple[str, str]:
    ref = approved_deploy(reactive)
    b = reactive.backend
    b.add_best_config("answer", 1, SPEC["config"])
    handle = reactive.client.reserve(ref, b.attempts[ref]["grant_ref"])
    if started:
        reactive.client.start(handle)
    return ref, deploy_key(ref, FP, GATE)


def _path(reactive: Reactive, key: str) -> str:
    return reactive.client._head_path(f"/publish-results/{key}")


def test_publish_requires_a_recorded_start(reactive: Reactive) -> None:
    ref, key = _reserved(reactive, started=False)
    with pytest.raises(ReactiveError) as err:
        reactive.client.publish_deploy(key=key, best_config_id="answer", version=1)
    assert err.value.code == "attempt_not_open"
    assert (
        reactive.backend.head.generation == 3 and not reactive.backend.publish_results
    )


def test_only_the_started_holder_may_publish_finalize_or_report(
    reactive: Reactive,
) -> None:
    b = reactive.backend
    ref, key = _reserved(reactive, started=True)
    b.principal = "another-ingest-key"
    for call in (
        lambda: reactive.client.publish_deploy(
            key=key, best_config_id="answer", version=1
        ),
        lambda: reactive.client.finalize_publish(key),
    ):
        with pytest.raises(ReactiveError) as err:
            call()
        assert err.value.code == "reference_not_found"
    res = b.reservations[b.attempts[ref]["reservation_ref"]]
    from traigent.runtime.hosted import ReservationHandle

    stolen = ReservationHandle(
        res["reservation_ref"], res["generation"], ref, "deploy", res["expires_at"]
    )
    for call in (
        lambda: reactive.client.heartbeat(stolen),
        lambda: reactive.client.report_authority(stolen, state="reachable"),
    ):
        with pytest.raises(ReactiveError) as err:
            call()
        assert err.value.code == "reference_not_found"
    assert b.head.generation == 3 and not b.publish_results
    b.principal = "ingest-key-1"
    assert (
        reactive.client.publish_deploy(
            key=key, best_config_id="answer", version=1
        ).state
        == "succeeded"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: {**d, "spec": {"config": {}}},  # the configuration never travels
        lambda d: {**d, "schema_version": "1"},  # an integer, not a string
        lambda d: {**d, "schema_version": True},
        lambda d: {k: v for k, v in d.items() if k != "version"},
        lambda d: {**d, "version": 0},
        lambda d: {**d, "best_config_id": "not valid!"},
    ],
)
def test_publish_body_is_closed(reactive: Reactive, mutate: Any) -> None:
    ref, key = _reserved(reactive, started=True)
    good = {"schema_version": 1, "best_config_id": "answer", "version": 1}
    status, _, _ = reactive.backend.transport(
        "POST", _path(reactive, key), None, mutate(good)
    )
    assert status == 400 and not reactive.backend.publish_results
    reactive.backend.violations.clear()  # the refusal was the point


def test_duplicate_json_keys_are_refused(reactive: Reactive) -> None:
    ref, key = _reserved(reactive, started=True)
    raw = (
        '{"schema_version": 1, "best_config_id": "answer", "version": 1, "version": 2}'
    )
    status, body, _ = reactive.backend.transport(
        "POST", _path(reactive, key), None, raw
    )
    assert status == 400 and body["error"] == "invalid_request"
    assert not reactive.backend.publish_results


def test_sdk_publish_body_is_exactly_the_closed_request(reactive: Reactive) -> None:
    ref, key = _reserved(reactive, started=True)
    reactive.client.publish_deploy(key=key, best_config_id="answer", version=1)
    (sent,) = reactive.backend.requests_of(PUBLISH)
    assert sent["body"] == {
        "schema_version": 1,
        "best_config_id": "answer",
        "version": 1,
    }
    assert json.dumps(sent["body"]).count("schema_version") == 1


def test_attempt_read_is_served_not_a_documented_gap(reactive: Reactive) -> None:
    ref = approved_deploy(reactive)
    assert reactive.client.get_attempt(ref)["attempt_ref"] == ref
    assert reactive.backend.violations == []


def test_zero_native_import_returns_a_reactive_receipt() -> None:
    doc = {
        "receipt_kind": "reactive_import",
        "schema_version": 1,
        "contract_digest": "a" * 64,
        "policy_revision_ref": make_ref("policyrevision", 1),
        "environment_ref": "env_" + "0" * 8,
        "created": True,
        "monitors": [],
    }
    receipt = _receipt(doc, True)
    assert receipt.created is True and receipt.status == "reactive_import"
    assert receipt.as_dict()["contract_digest"] == "a" * 64
    assert _receipt({**doc, "created": False}, False).created is False
    with pytest.raises(RuntimeRequirementError):
        _receipt({**doc, "contract_digest": "short"}, True)
    with pytest.raises(RuntimeRequirementError):
        _receipt(doc, False)  # status 200 must say created: false
