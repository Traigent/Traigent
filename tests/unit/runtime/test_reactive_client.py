"""The typed reactive client: closed requests, strict responses, fail-closed tokens."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from traigent.runtime.head_loader import (
    E_HEAD_MOVED,
    config_values,
    load_pinned_config,
)
from traigent.runtime.hosted import (
    ReactiveClient,
    ReactiveError,
    build_candidate,
    deploy_key,
    parse_refusal,
)
from traigent.runtime.model import E_BACKEND, E_STATE, RuntimeRequirementError

from .conftest import CANARY, Reactive
from .reactive_fake import SCHEMA_ROOT, SCHEMAS, make_ref, refusal


def test_vendored_contracts_are_byte_exact_and_pinned() -> None:
    pins: dict[str, str] = {}
    commit = ""
    for line in (SCHEMAS / "SOURCE.md").read_text().splitlines():
        if line.startswith("commit sha1: "):
            commit = line.split(": ", 1)[1]
        elif " sha256: " in line:
            name, sha = line.split(" sha256: ")
            pins[name] = sha
    assert len(commit) == 40
    for name, sha in pins.items():
        assert hashlib.sha256((SCHEMA_ROOT / name).read_bytes()).hexdigest() == sha, (
            name
        )
    on_disk = {str(p.relative_to(SCHEMA_ROOT)) for p in SCHEMA_ROOT.rglob("*.json")}
    assert on_disk == set(pins)


def test_every_call_is_accepted_by_the_strict_contract_fake(reactive: Reactive) -> None:
    c, b = reactive.client, reactive.backend
    rev = "b" * 64
    c.stage(
        3,
        policy_revision_ref=make_ref("policyrevision", 1),
        environment="production",
        producer_kind="customer_monitor",
        revisions=[
            {
                "kind": "monitor",
                "target": reactive.requirement.monitor_name,
                "value": rev,
            }
        ],
        windows=[
            {
                "monitor": reactive.requirement.monitor_name,
                "start": 0,
                "end": 6,
                "verdict": "UNKNOWN",
                "unknown_reason": "insufficient",
                "usable": False,
            }
        ],
        episodes=[
            {
                "monitor": reactive.requirement.monitor_name,
                "scope": "episode-1",
                "event": "opened",
            },
            {
                "monitor": reactive.requirement.monitor_name,
                "scope": "episode-1",
                "event": "closed",
                "reason": "resolved",
            },
        ],
        liveness=[
            {
                "monitor": reactive.requirement.monitor_name,
                "event": "stale",
                "scope": "episode-1",
            }
        ],
        optimizations=[
            {
                "response": reactive.requirement.ladder_name,
                "revision": rev,
                "rule": "fix",
                "scope": "episode-1",
                "occurrence": 0,
                "fact": "fired",
            }
        ],
    )
    c.complete(
        3,
        policy_revision_ref=make_ref("policyrevision", 1),
        environment="production",
        producer_kind="customer_monitor",
        monitor=reactive.requirement.monitor_name,
        completed_through_end=6,
    )
    c.complete(
        3,
        policy_revision_ref=make_ref("policyrevision", 1),
        environment="production",
        producer_kind="optimization_reporter",
        disposition="defer",
    )
    ref = reactive.approved_optimize()
    assert c.list_attempts()["items"][0]["attempt_ref"] == ref
    assert c.get_attempt(ref)["state"] == "approved"
    grant = c.get_grant(b.attempts[ref]["grant_ref"])
    assert grant["ceiling"]["max_trials"] == 20
    res = c.reserve(ref, grant["grant_ref"])
    c.start(res)
    c.heartbeat(res)
    c.report_authority(res, state="reachable")
    c.report_finished(
        res,
        completion="normal",
        candidates=[
            {
                "evidence_valid": True,
                "legacy_decision": "Promote",
                "monitor_verdicts": ["MET", "UNKNOWN(insufficient)"],
                "fingerprint": "c" * 64,
                "gate_result_digest": "d" * 64,
            }
        ],
    )
    assert c.list_notifications(decision="delivered")["items"] == []
    with pytest.raises(ReactiveError) as nf:
        c.get_notification(make_ref("notification", 1))
    assert nf.value.code == "reference_not_found" and not nf.value.ambiguous
    with pytest.raises(ReactiveError) as forbidden:
        c.cancel(ref)  # an SDK key is never a person
    assert forbidden.value.code == "forbidden"
    pinned = c.get_pinned_config()
    assert pinned.generation == 3 and pinned.version == 7
    key = deploy_key(ref, "c" * 64, "d" * 64)
    with pytest.raises(ReactiveError) as stranger:  # not the holder: no oracle
        c.finalize_publish(key)
    assert stranger.value.code == "reference_not_found"


def test_names_are_sent_as_hosted_tokens_only(reactive: Reactive) -> None:
    reactive.client.stage(
        1,
        policy_revision_ref=make_ref("policyrevision", 1),
        environment="production",
        producer_kind="customer_monitor",
        windows=[
            {
                "monitor": reactive.requirement.monitor_name,
                "start": 0,
                "end": 6,
                "verdict": "MET",
                "usable": True,
            }
        ],
    )
    (sent,) = reactive.backend.requests_of("stageTvlReactiveEvaluation")
    text = json.dumps(sent["body"])
    for local in (
        reactive.requirement.monitor_name,
        "production",
        reactive.requirement.name,
    ):
        assert local not in text
    assert sent["body"]["windows"][0]["monitor"] == reactive.tokens.hosted(
        "monitor", reactive.requirement.monitor_name
    )


@pytest.mark.parametrize("damage", ["missing", "corrupt", "wrong_format"])
def test_token_map_unavailable_fails_closed_and_sends_nothing(
    reactive: Reactive, damage: str
) -> None:
    path = reactive.state / "tvl-token-map.json"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("{not json")
    else:
        path.write_text(json.dumps({"format": "other", "tokens": []}))
    with pytest.raises(RuntimeRequirementError) as err:
        reactive.client.stage(
            1,
            policy_revision_ref=make_ref("policyrevision", 1),
            environment="production",
            producer_kind="customer_monitor",
        )
    assert err.value.code == E_STATE
    assert (
        reactive.backend.log == []
    )  # nothing was sent: no fallback to the name or a hash


def test_a_name_without_a_token_is_never_sent(reactive: Reactive) -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        reactive.client.stage(
            1,
            policy_revision_ref=make_ref("policyrevision", 1),
            environment="production",
            producer_kind="customer_monitor",
            windows=[
                {
                    "monitor": CANARY,  # content in a name field
                    "start": 0,
                    "end": 6,
                    "verdict": "MET",
                    "usable": True,
                }
            ],
        )
    assert err.value.code == E_STATE and reactive.backend.log == []


def test_content_is_refused_before_it_is_sent(reactive: Reactive) -> None:
    c = reactive.client
    with pytest.raises(RuntimeRequirementError) as extra:
        c.report_finished(
            _holder(reactive),
            completion="normal",
            candidates=[{"evidence_valid": False, "output_text": CANARY}],
        )
    assert extra.value.code == "invalid_request"
    with pytest.raises(RuntimeRequirementError):
        c.report_finished(_holder(reactive), completion=CANARY)
    with pytest.raises(RuntimeRequirementError):
        build_candidate({"evidence_valid": True, "monitor_verdicts": [CANARY]})
    with pytest.raises(
        RuntimeRequirementError
    ):  # valid evidence without the join inputs
        build_candidate({"evidence_valid": True})
    with pytest.raises(RuntimeRequirementError):  # invalid evidence carries none
        build_candidate({"evidence_valid": False, "legacy_decision": "Promote"})
    assert build_candidate({"evidence_valid": False}) == {"evidence_valid": False}
    with pytest.raises(RuntimeRequirementError):
        c.stage(
            1,
            policy_revision_ref=CANARY,
            environment="production",
            producer_kind="customer_monitor",
        )
    assert reactive.backend.log == []


def _holder(reactive: Reactive) -> Any:
    from traigent.runtime.hosted import ReservationHandle

    return ReservationHandle(
        make_ref("reservation", 9),
        1,
        make_ref("attempt", 9),
        "optimize",
        "2026-10-23T12:05:00Z",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: {**d, "unexpected": 1},
        lambda d: {k: v for k, v in d.items() if k != "state"},
        lambda d: {**d, "state": "pending_review"},
        lambda d: {**d, "occurrence": True},
        lambda d: {**d, "attempt_ref": "attemptref_XYZ"},
        lambda d: {**d, "opened_at": "yesterday"},
    ],
)
def test_responses_are_parsed_strictly(reactive: Reactive, mutate: Any) -> None:
    ref = reactive.approved_optimize()
    good = reactive.backend.transport(
        "GET",
        f"/api/v1/production-reactive/agents/{reactive.agent_ref}/attempts/{ref}",
        None,
        None,
    )[1]

    def transport(*_a: Any) -> tuple[int, Any, dict[str, str]]:
        return 200, mutate(good), {}

    client = ReactiveClient(reactive.agent_ref, reactive.tokens, transport=transport)
    with pytest.raises(ReactiveError) as err:
        client.get_attempt(ref)
    assert err.value.code == E_BACKEND and err.value.ambiguous


def test_a_response_for_another_attempt_is_refused(reactive: Reactive) -> None:
    one, two = reactive.approved_optimize(), reactive.approved_optimize()
    doc = reactive.client.get_attempt(one)

    client = ReactiveClient(
        reactive.agent_ref, reactive.tokens, transport=lambda *_a: (200, doc, {})
    )
    with pytest.raises(ReactiveError):
        client.get_attempt(two)


def test_refusal_envelope_is_closed(reactive: Reactive) -> None:
    assert parse_refusal(refusal("grant_not_valid")) == "grant_not_valid"
    for bad in (
        {**refusal("grant_not_valid"), "detail": "x"},
        {**refusal("grant_not_valid"), "message": "Grant is bad."},
        {**refusal("grant_not_valid"), "error": "forbidden"},
        {"error_code": "grant_not_valid"},
        {**refusal("grant_not_valid"), "error_code": "made_up", "error": "made_up"},
    ):
        assert parse_refusal(bad) is None
    client = ReactiveClient(
        reactive.agent_ref,
        reactive.tokens,
        transport=lambda *_a: (
            409,
            {**refusal("grant_not_valid"), "detail": CANARY},
            {},
        ),
    )
    with pytest.raises(ReactiveError) as err:
        client.get_pinned_config()
    assert err.value.code == E_BACKEND and CANARY not in str(err.value)


def test_rate_limit_carries_retry_after_and_5xx_is_ambiguous(
    reactive: Reactive,
) -> None:
    reactive.backend.fail_next("getAgentHeadPinnedConfig", "429", "503")
    with pytest.raises(ReactiveError) as limited:
        reactive.client.get_pinned_config()
    assert limited.value.code == "rate_limited" and limited.value.retry_after == 2
    assert not limited.value.ambiguous
    with pytest.raises(ReactiveError) as down:
        reactive.client.get_pinned_config()
    assert down.value.code == "service_unavailable" and down.value.ambiguous


def test_head_loader_reads_the_exact_pinned_version(reactive: Reactive) -> None:
    pinned = load_pinned_config(reactive.client)
    assert (pinned.generation, pinned.version) == (3, 7)
    assert config_values(pinned) == {"temperature": 0.1}
    (call,) = reactive.backend.requests_of("getAgentHeadPinnedConfig")
    assert call["path"].endswith("/agent-heads/" + reactive.agent_ref + "/config")
    with pytest.raises(RuntimeRequirementError) as moved:
        load_pinned_config(reactive.client, expected_generation=2)
    assert moved.value.code == E_HEAD_MOVED
    reactive.backend.head.spec = {
        "schema_version": "traigent.best_config.v1",
        "config_id": "a",
        "config": {"k": 1},
    }
    assert config_values(load_pinned_config(reactive.client)) == {"k": 1}


def test_canary_is_in_no_request_of_any_call(reactive: Reactive) -> None:
    """Customer content (prompts/outputs) has no field to travel in."""
    ref = reactive.approved_optimize()
    res = reactive.client.reserve(ref, reactive.backend.attempts[ref]["grant_ref"])
    reactive.client.start(res)
    reactive.client.report_finished(res, completion="normal", candidates=[])
    for sent in reactive.backend.log:
        assert CANARY not in json.dumps(sent)
