"""Deploy controller: approval-time generation, idempotent publish, finalize on ambiguity."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any

import pytest

from traigent.runtime.deploy import (
    DEPLOYED,
    FAILED,
    REFUSED,
    DeployController,
    DeployHold,
)
from traigent.runtime.hosted import ReactiveError, deploy_key
from traigent.runtime.model import RuntimeRequirementError

from .conftest import CANARY, Reactive
from .reactive_fake import canonical_digest

GATE = "d" * 64
SPEC = {
    "schema_version": "traigent.best_config.v1",
    "config_id": "answer",
    "config": {"temperature": 0.2, "model": "m-small"},
}
FP = canonical_digest(SPEC["config"])  # the Backend recomputes this from the spec
PUBLISH = "publishTvlDeployResult"
LOOKUP = "getTvlDeployPublishResult"
FINALIZE = "finalizeTvlDeployPublishResult"


def approved_deploy(
    reactive: Reactive, *, approval_generation: int | None = None
) -> str:
    ref = reactive.attempt("deploy")
    reactive.backend.grant(
        ref,
        deploy={
            "candidate_fingerprint": FP,
            "gate_result_digest": GATE,
            "expected_head_generation": (
                reactive.backend.head.generation
                if approval_generation is None
                else approval_generation
            ),
        },
    )
    return ref


def controller(reactive: Reactive, **kw: Any) -> DeployController:
    async def sleep(seconds: float) -> None:
        reactive.clock.advance(seconds=seconds)

    kw.setdefault("sleep", sleep)
    kw.setdefault("retry_delay", 1.0)
    return DeployController(
        reactive.client, state_dir=reactive.state, clock=reactive.clock, **kw
    )


def deploy(
    reactive: Reactive, ref: str, ctl: DeployController | None = None, **kw: Any
) -> Any:
    reactive.backend.add_best_config("answer", 1, SPEC["config"])
    return asyncio.run(
        (ctl or controller(reactive, **kw)).run(
            ref,
            grant_ref=reactive.backend.attempts[ref]["grant_ref"],
            best_config_id="answer",
            version=1,
            candidate_fingerprint=FP,
            gate_result_digest=GATE,
        )
    )


def test_happy_path_publishes_with_the_approval_generation_and_key(
    reactive: Reactive,
) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    outcome = deploy(reactive, ref)
    assert outcome.state == DEPLOYED and outcome.result_recorded
    (publication,) = b.requests_of(PUBLISH)
    body = publication["body"]
    # closed DeployPublishRequest: ids and an integer, never the configuration
    assert body == {"schema_version": 1, "best_config_id": "answer", "version": 1}
    assert publication["path"].endswith(f"/publish-results/{deploy_key(ref, FP, GATE)}")
    assert (b.head.generation, b.head.version) == (4, 8) and b.head.spec == SPEC
    (result,) = b.results
    assert (result["checks"], result["write"], result["retries_exhausted"]) == (
        "passed",
        "succeeded",
        False,
    )
    assert result["publish_result_key"] == deploy_key(ref, FP, GATE)
    # deploy_started was accepted before the write
    order = [r["operation"] for r in b.log]
    assert order.index("startTvlReactiveAttempt") < order.index(PUBLISH)
    # The head is read back through the exact-version endpoint, not a listing.
    assert reactive.client.get_pinned_config().spec == SPEC
    assert b.attempts[ref]["deploy_outcome"] == "deployed"


def test_head_moved_after_approval_refuses_and_changes_nothing(
    reactive: Reactive,
) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive, approval_generation=3)
    b.head.generation = 4  # someone promoted after the person approved
    outcome = deploy(reactive, ref)
    assert outcome.state == REFUSED
    assert len(b.requests_of(PUBLISH)) == 1  # a refusal is never retried
    assert (b.head.generation, b.head.spec["config"]) == (4, {"temperature": 0.1})
    (result,) = b.results
    assert (result["checks"], result["write"]) == ("refused", "not_attempted")
    assert "publish_result_key" not in result
    assert b.attempts[ref]["deploy_outcome"] == "deploy_refused"


def test_lost_response_after_commit_is_found_by_lookup_not_failed(
    reactive: Reactive,
) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    b.fail_next(PUBLISH, "drop_after_commit")
    outcome = deploy(reactive, ref)
    assert outcome.state == DEPLOYED and outcome.head_generation == 4
    assert (
        len(b.requests_of(PUBLISH)) == 2
    )  # the same key is replayed; the head advances once
    assert b.head.generation == 4
    (result,) = b.results
    assert result["write"] == "succeeded"


def test_lost_response_with_lookup_down_replays_the_same_key_past_the_stale_generation(
    reactive: Reactive,
) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    b.fail_next(PUBLISH, "drop_after_commit")
    b.fail_next(LOOKUP, "transport")
    outcome = deploy(reactive, ref)
    assert outcome.state == DEPLOYED
    first, second = b.requests_of(PUBLISH)
    assert first["body"] == second["body"]  # same key, same bindings
    # The expected generation (3) is now stale (head is 4): the replay still answers first.
    assert second["response"]["state"] == "succeeded"
    assert b.head.generation == 4  # exactly one head advance


def test_negative_lookup_is_not_a_decision(reactive: Reactive) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    b.fail_next(PUBLISH, "transport", "503")
    outcome = deploy(reactive, ref)
    assert outcome.state == DEPLOYED
    assert b.requests_of(FINALIZE) == []  # never needed
    assert len(b.requests_of(PUBLISH)) == 3


def test_exhaustion_finalizes_to_a_tombstone_then_records_deploy_failed_once(
    reactive: Reactive,
) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    b.fail_next(PUBLISH, "transport", "503", "transport")
    outcome = deploy(reactive, ref)
    assert outcome.state == FAILED
    assert len(b.requests_of(PUBLISH)) == 3  # the bound: three, not four
    (fin,) = b.requests_of(FINALIZE)
    assert fin["response"]["state"] == "cancelled"
    (result,) = b.results
    assert (result["write"], result["retries_exhausted"]) == ("failed", True)
    assert result["publish_result_key"] == deploy_key(ref, FP, GATE)
    assert b.head.generation == 3 and b.head.spec["config"] == {"temperature": 0.1}
    # The tombstone is binding: a late publication of the same key cannot commit.
    with pytest.raises(ReactiveError) as late:
        reactive.client.publish_deploy(
            key=deploy_key(ref, FP, GATE), best_config_id="answer", version=1
        )
    assert late.value.code == "publication_cancelled"
    assert b.head.generation == 3


def test_exhaustion_where_the_write_actually_committed_finalizes_to_success(
    reactive: Reactive,
) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    # Commit then lose the answer; then the lookup and the retries all fail.
    b.fail_next(PUBLISH, "drop_after_commit", "503", "503")
    b.fail_next(LOOKUP, "transport", "transport", "transport")
    outcome = deploy(reactive, ref)
    assert outcome.state == DEPLOYED
    assert (
        b.requests_of(FINALIZE)
        and b.requests_of(FINALIZE)[0]["response"]["state"] == "succeeded"
    )
    (result,) = b.results
    assert result["write"] == "succeeded"  # a lost response never becomes deploy_failed
    assert b.head.generation == 4


def test_finalization_unavailable_is_a_hold_with_no_outcome_and_no_new_publication(
    reactive: Reactive,
) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    b.fail_next(PUBLISH, "transport", "transport", "transport")
    b.fail_next(FINALIZE, "transport", "503", "transport")
    with pytest.raises(DeployHold) as hold:
        deploy(reactive, ref)
    assert hold.value.path == "finalize_unavailable" and hold.value.outcome is None
    assert b.results == []  # no terminal outcome
    assert len(b.requests_of(PUBLISH)) == 3  # and no replacement publication
    assert b.head.generation == 3
    assert b.attempts[ref]["deploy_outcome"] == "none"


def test_retry_window_bounds_the_attempts(reactive: Reactive) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    b.fail_next(PUBLISH, *["transport"] * 5)
    outcome = deploy(reactive, ref, window=timedelta(seconds=1), retry_delay=2.0)
    assert outcome.state == FAILED
    assert (
        len(b.requests_of(PUBLISH)) == 1
    )  # the 15-minute window (here 1 s) cuts retries


def test_rate_limit_waits_for_retry_after(reactive: Reactive) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    b.fail_next(PUBLISH, "429")
    t0 = reactive.clock()
    outcome = deploy(reactive, ref)
    assert outcome.state == DEPLOYED
    assert reactive.clock() - t0 >= timedelta(seconds=2)  # Retry-After honoured


def test_the_candidate_in_hand_must_be_the_approved_one(reactive: Reactive) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    b.add_best_config("answer", 1, SPEC["config"])
    try:  # the Backend, not the SDK, binds the key to the grant: a foreign fingerprint is refused
        asyncio.run(
            controller(reactive).run(
                ref,
                grant_ref=b.attempts[ref]["grant_ref"],
                best_config_id="answer",
                version=1,
                candidate_fingerprint="9" * 64,
                gate_result_digest=GATE,
            )
        )
    except (ReactiveError, DeployHold, RuntimeRequirementError):
        pass
    assert b.head.generation == 3 and not b.publish_results
    b.violations.clear()


def test_only_the_publication_carries_content_and_the_canary_never_travels(
    reactive: Reactive,
) -> None:
    b = reactive.backend
    ref = approved_deploy(reactive)
    deploy(reactive, ref)
    for sent in b.log:
        assert CANARY not in json.dumps(sent)
        # no route carries the configuration: the publish names a server-issued version
        assert not (isinstance(sent["body"], dict) and "spec" in sent["body"])
