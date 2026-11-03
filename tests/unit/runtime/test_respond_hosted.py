"""Hosted respond: reserve -> accepted start -> run -> report, heartbeat, lease loss."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.runtime.model import (
    E_APPROVAL_EXPIRED,
    E_FIX_CEILING,
    E_HOLD,
    E_OCCURRENCE_USED,
    E_STATE,
    RuntimeRequirementError,
)
from traigent.runtime.respond import (
    E_AMBIGUOUS_ATTEMPT,
    E_NO_APPROVED_ATTEMPT,
    respond,
    respond_hosted,
)

from .conftest import CANARY, Reactive

FAST = timedelta(milliseconds=10)


class Fn:
    """Stands in for an optimized function; records what the backend knew at launch."""

    def __init__(self, reactive: Reactive, body: Any = None) -> None:
        self.reactive = reactive
        self.calls: list[dict[str, Any]] = []
        self.starts_at_launch: list[int] = []
        self.body = body

    async def optimize(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.starts_at_launch.append(
            sum(self.reactive.backend.starts_accepted.values())
        )
        if self.body is not None:
            return await self.body()
        return SimpleNamespace(
            optimization_id="run-1", stop_reason="max_trials_reached"
        )


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def go(reactive: Reactive, fn: Any, **kw: Any) -> Any:
    kw.setdefault("heartbeat_interval", FAST)
    kw.setdefault("retry_delay", 0.0)
    return run(
        respond_hosted(
            fn,
            reactive.requirement,
            agent_ref=reactive.agent_ref,
            client=reactive.client,
            state_dir=reactive.state,
            clock=reactive.clock,
            **kw,
        )
    )


def ops(reactive: Reactive) -> list[str]:
    return [r["operation"] for r in reactive.backend.log]


def test_start_is_accepted_before_the_run_and_the_result_carries_the_generation(
    reactive: Reactive,
) -> None:
    ref = reactive.approved_optimize()
    fn = Fn(reactive)
    result = go(reactive, fn)
    assert fn.starts_at_launch == [1]  # the backend had accepted the start at launch
    assert fn.calls == [{"cost_limit": 5.0, "max_trials": 20}]
    order = ops(reactive)
    assert order.index("reserveTvlReactiveAttempt") < order.index(
        "startTvlReactiveAttempt"
    )
    assert order.index("startTvlReactiveAttempt") < order.index(
        "reportTvlReactiveAttemptResult"
    )
    (report,) = reactive.backend.results
    assert report["reservation_generation"] == result.reservation_generation == 1
    assert report["fact"] == "finished" and report["completion"] == "normal"
    assert (
        result.attempt_ref == ref
        and result.result_recorded
        and result.run_id == "run-1"
    )


def test_no_second_start_for_an_occurrence(reactive: Reactive, tmp_path: Any) -> None:
    ref = reactive.approved_optimize()
    go(reactive, Fn(reactive))
    assert reactive.backend.starts_accepted[ref] == 1
    # Same state dir: the local lease cache already holds it, nothing is asked again.
    before = len(reactive.backend.log)
    with pytest.raises(RuntimeRequirementError) as again:
        go(reactive, Fn(reactive), attempt_ref=ref)
    assert again.value.code in (E_OCCURRENCE_USED, E_NO_APPROVED_ATTEMPT)
    # Even an explicit retry with the attempt finished is refused by state, never re-started.
    assert reactive.backend.starts_accepted[ref] == 1
    assert [r["operation"] for r in reactive.backend.log[before:]].count(
        "startTvlReactiveAttempt"
    ) == 0


def test_a_fresh_controller_is_refused_by_the_backend_not_by_the_cache(
    reactive: Reactive,
) -> None:
    from traigent.runtime import manifest

    ref = reactive.approved_optimize()
    go(reactive, Fn(reactive))
    other = reactive.state.parent / "other-state"
    manifest.ensure_tokens(other, [])
    # A second controller sees the attempt as finished: there is nothing left to approve.
    with pytest.raises(RuntimeRequirementError) as err:
        run(
            respond_hosted(
                Fn(reactive),
                reactive.requirement,
                agent_ref=reactive.agent_ref,
                client=reactive.client,
                attempt_ref=ref,
                state_dir=other,
                clock=reactive.clock,
            )
        )
    assert err.value.code == E_NO_APPROVED_ATTEMPT
    assert reactive.backend.starts_accepted[ref] == 1


def test_reservation_conflict_for_a_second_holder(reactive: Reactive) -> None:
    ref = reactive.approved_optimize()
    reactive.client.reserve(ref, reactive.backend.attempts[ref]["grant_ref"])
    fn = Fn(reactive)
    with pytest.raises(RuntimeRequirementError):
        go(reactive, fn, attempt_ref=ref)
    assert fn.calls == [] and reactive.backend.starts_accepted[ref] == 0


def test_start_refused_means_nothing_runs(reactive: Reactive) -> None:
    ref = reactive.approved_optimize()
    grant = reactive.backend.attempts[ref]["grant_ref"]
    original = reactive.backend._op_startTvlReactiveAttempt

    def revoke_then_start(v: Any, q: Any, body: Any) -> Any:
        reactive.backend.revoke(grant)  # approval revoked between reserve and start
        return original(v, q, body)

    reactive.backend._op_startTvlReactiveAttempt = revoke_then_start  # type: ignore[method-assign]
    fn = Fn(reactive)
    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive, fn)
    assert err.value.code == "start_refused"
    assert fn.calls == [] and reactive.backend.starts_accepted[ref] == 0
    assert reactive.backend.results == []  # no outcome without a start


def test_unconfirmed_start_holds_without_running_or_asking_twice(
    reactive: Reactive,
) -> None:
    ref = reactive.approved_optimize()
    reactive.backend.fail_next(
        "startTvlReactiveAttempt", "transport", "503", "transport"
    )
    fn = Fn(reactive)
    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive, fn)
    assert err.value.code == E_HOLD and err.value.path == "start_unconfirmed"
    keys = {
        r["body"]["start_key"]
        for r in reactive.backend.requests_of("startTvlReactiveAttempt")
    }
    assert len(keys) == 1  # three tries are ONE request, replayed under one key
    assert fn.calls == [] and reactive.backend.starts_accepted[ref] == 0
    # The cache remembers: a later call does not even try again.
    with pytest.raises(RuntimeRequirementError):
        go(reactive, fn, attempt_ref=ref)
    assert len(reactive.backend.requests_of("startTvlReactiveAttempt")) == 3


def test_lost_start_response_is_replayed_not_restarted(reactive: Reactive) -> None:
    ref = reactive.approved_optimize()
    reactive.backend.fail_next("startTvlReactiveAttempt", "drop_after_commit")
    fn = Fn(reactive)
    result = go(reactive, fn)
    starts = reactive.backend.requests_of("startTvlReactiveAttempt")
    assert (
        len(starts) == 2
        and starts[0]["body"]["start_key"] == starts[1]["body"]["start_key"]
    )
    assert starts[1]["response"]["replayed"] is True
    assert reactive.backend.starts_accepted[ref] == 1 and len(fn.calls) == 1
    assert result.result_recorded


def test_heartbeat_renews_the_reservation_while_the_run_is_in_flight(
    reactive: Reactive,
) -> None:
    reactive.approved_optimize()

    async def slow() -> Any:
        await asyncio.sleep(0.2)
        return SimpleNamespace(optimization_id="run-slow")

    go(reactive, Fn(reactive, slow))
    beats = reactive.backend.requests_of("renewTvlReactiveAttemptLease")
    assert len(beats) >= 3
    first = reactive.backend.requests_of("reserveTvlReactiveAttempt")[0]["response"]
    assert all(
        b["body"]["reservation_generation"] == first["generation"]
        and b["body"]["reservation_ref"] == first["reservation_ref"]
        for b in beats
    )  # bound to the EXISTING reservation: never a new one, never a second start


def test_lease_loss_cancels_the_run_and_reports_nothing(reactive: Reactive) -> None:
    ref = reactive.approved_optimize()
    cancelled = {"flag": False}

    async def hang() -> Any:
        # The holder is superseded mid-run: its generation no longer matches.
        reactive.backend.reservations[
            reactive.backend.attempts[ref]["reservation_ref"]
        ]["generation"] += 1
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled["flag"] = True
            raise

    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive, Fn(reactive, hang))
    assert err.value.code == E_HOLD and err.value.path == "lease"
    assert cancelled["flag"] is True  # the SDK cancels a run whose lease it lost
    assert reactive.backend.results == []  # and never reports under a lost reservation


def test_heartbeat_ambiguity_is_tolerated_only_until_the_vouched_deadline(
    reactive: Reactive,
) -> None:
    reactive.approved_optimize()

    async def tolerated() -> Any:
        await asyncio.sleep(0.12)
        return SimpleNamespace(optimization_id="run-ok")

    reactive.backend.fail_next(
        "renewTvlReactiveAttemptLease", "transport", "503", "transport", "429"
    )
    assert go(reactive, Fn(reactive, tolerated)).result_recorded

    reactive2 = Reactive(reactive.state.parent / "second")
    reactive2.approved_optimize()
    reactive2.backend.fail_next("renewTvlReactiveAttemptLease", *["transport"] * 50)
    cancelled = {"flag": False}

    async def until_deadline() -> Any:
        await asyncio.sleep(0.05)
        reactive2.clock.advance(minutes=30)  # the server's last word has expired
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled["flag"] = True
            raise

    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive2, Fn(reactive2, until_deadline))
    assert err.value.path == "lease" and cancelled["flag"]
    assert reactive2.backend.violations == []


def test_failed_run_reports_error_and_never_retries(reactive: Reactive) -> None:
    ref = reactive.approved_optimize()

    async def boom() -> Any:
        raise ValueError(CANARY)

    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive, Fn(reactive, boom))
    assert err.value.code == "optimize_failed" and CANARY not in str(err.value)
    (report,) = reactive.backend.results
    assert report["completion"] == "error" and report["candidates"] == []
    assert reactive.backend.attempts[ref]["optimize_outcome"] == "optimize_failed"
    assert reactive.backend.starts_accepted[ref] == 1


def test_candidates_are_closed_facts_and_a_gate_is_never_guessed(
    reactive: Reactive,
) -> None:
    ref = reactive.approved_optimize()
    fn = Fn(reactive)
    go(
        reactive,
        fn,
        candidates=lambda _r, _g: [
            {
                "evidence_valid": True,
                "legacy_decision": "Promote",
                "monitor_verdicts": ["MET"],
                "fingerprint": "e" * 64,
                "gate_result_digest": "f" * 64,
            }
        ],
    )
    assert reactive.backend.attempts[ref]["optimize_outcome"] == "promoted"
    # With no gate evidence on the result the report carries no candidate.
    ref2 = reactive.approved_optimize()
    go(reactive, Fn(reactive), attempt_ref=ref2)
    assert (
        reactive.backend.attempts[ref2]["optimize_outcome"] == "optimize_no_candidate"
    )


def test_ceilings_come_from_the_grant_and_can_only_be_tightened(
    reactive: Reactive,
) -> None:
    reactive.approved_optimize(max_cost_micro_usd=2_000_000, max_trials=7)
    fn = Fn(reactive)
    go(reactive, fn)
    assert fn.calls == [{"cost_limit": 2.0, "max_trials": 7}]
    reactive.approved_optimize(max_cost_micro_usd=50_000_000)
    fn2 = Fn(reactive)
    go(reactive, fn2)  # declared 5.0 is lower than the grant's 50.0
    assert fn2.calls[0]["cost_limit"] == 5.0
    ref = reactive.approved_optimize()
    reactive.backend.grants[reactive.backend.attempts[ref]["grant_ref"]].pop("ceiling")
    fn3 = Fn(reactive)
    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive, fn3, attempt_ref=ref)
    assert err.value.code == E_FIX_CEILING and fn3.calls == []


def test_only_an_approved_attempt_of_this_rule_runs(reactive: Reactive) -> None:
    fn = Fn(reactive)
    with pytest.raises(RuntimeRequirementError) as none:
        go(reactive, fn)
    assert none.value.code == E_NO_APPROVED_ATTEMPT
    reactive.attempt()  # open but not approved
    with pytest.raises(RuntimeRequirementError):
        go(reactive, fn)
    reactive.approved_optimize()
    reactive.approved_optimize()
    with pytest.raises(RuntimeRequirementError) as two:
        go(reactive, fn)
    assert two.value.code == E_AMBIGUOUS_ATTEMPT and fn.calls == []


def test_expired_or_revoked_grant_runs_nothing(reactive: Reactive) -> None:
    ref = reactive.approved_optimize(hours=1)
    reactive.clock.advance(hours=2)
    fn = Fn(reactive)
    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive, fn, attempt_ref=ref)
    assert err.value.code in (E_APPROVAL_EXPIRED, "approval_required")
    assert (
        fn.calls == []
        and reactive.backend.requests_of("reserveTvlReactiveAttempt") == []
    )


def test_unavailable_token_map_asks_and_reserves_nothing(reactive: Reactive) -> None:
    reactive.approved_optimize()
    (reactive.state / "tvl-token-map.json").unlink()
    fn = Fn(reactive)
    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive, fn)
    assert err.value.code == E_STATE
    assert fn.calls == [] and reactive.backend.log == []


def test_content_canary_never_reaches_any_request(reactive: Reactive) -> None:
    reactive.approved_optimize()

    async def with_content() -> Any:
        return SimpleNamespace(
            optimization_id="run-1",
            prompts=[CANARY],
            outputs=[CANARY],
            best_config={"prompt": CANARY},
            stop_reason="max_trials_reached",
        )

    go(reactive, Fn(reactive, with_content))
    assert reactive.backend.log
    for sent in reactive.backend.log:
        assert CANARY not in json.dumps(sent)


def test_respond_dispatches_to_the_hosted_path_with_an_agent_ref(
    reactive: Reactive,
) -> None:
    reactive.approved_optimize()
    result = run(
        respond(
            Fn(reactive),
            reactive.requirement,
            project_id="p",
            agent_id="a",
            agent_ref=reactive.agent_ref,
            reactive=reactive.client,
            state_dir=reactive.state,
            clock=reactive.clock,
        )
    )
    assert result.attempt_ref and result.result_recorded


# --- review follow-ups: post-start lease, reclaim of an unstarted reservation, crashes ------


def test_lease_loss_is_judged_by_the_start_receipt_before_any_heartbeat_succeeds(
    reactive: Reactive,
) -> None:
    ref = reactive.approved_optimize()
    reactive.backend.fail_next("renewTvlReactiveAttemptLease", *["transport"] * 200)
    cancelled = {"flag": False}

    async def work() -> Any:
        await asyncio.sleep(0.03)
        # 3 minutes: past the 2-minute post-start lease, inside the 5-minute reservation.
        reactive.clock.advance(minutes=3)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled["flag"] = True
            raise

    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive, Fn(reactive, work))
    assert err.value.path == "lease" and cancelled["flag"]
    assert reactive.backend.results == []
    start = reactive.backend.requests_of("startTvlReactiveAttempt")[0]["response"]
    assert "lease_expires_at" in start  # the deadline comes from the receipt
    assert reactive.backend.starts_accepted[ref] == 1


def test_crash_after_reserve_frees_the_attempt_when_the_reservation_expires(
    reactive: Reactive,
) -> None:
    ref = reactive.approved_optimize()
    grant = reactive.backend.attempts[ref]["grant_ref"]
    reactive.client.reserve(ref, grant)  # a controller that then died
    fn = Fn(reactive)
    with pytest.raises(RuntimeRequirementError):  # still held: nobody else may take it
        go(reactive, fn, attempt_ref=ref)
    reactive.clock.advance(
        minutes=6
    )  # the reservation expired unstarted: it simply frees
    # The first refusal happened at reserve, before any local cache entry existed.
    result = go(reactive, fn, attempt_ref=ref)
    assert result.reservation_generation == 2 and result.result_recorded
    assert reactive.backend.starts_accepted[ref] == 1


def test_crash_after_start_accepted_never_allows_a_second_start_but_keeps_the_result_path(
    reactive: Reactive,
) -> None:
    ref = reactive.approved_optimize()
    grant = reactive.backend.attempts[ref]["grant_ref"]
    dead = reactive.client.reserve(ref, grant)
    reactive.client.start(dead)  # started, then the process died before optimize
    fn = Fn(reactive)
    for _ in range(2):  # before and after the lease lapses
        with pytest.raises(RuntimeRequirementError) as err:
            go(reactive, fn, attempt_ref=ref)
        assert err.value.code == E_NO_APPROVED_ATTEMPT
        reactive.clock.advance(minutes=10)
    assert fn.calls == [] and reactive.backend.starts_accepted[ref] == 1
    assert reactive.backend.attempts[ref]["state"] == "lapsed"
    # The original holder (once restarted with its handle) may still report the true result.
    reactive.client.report_finished(dead, completion="normal", candidates=[])
    assert reactive.backend.attempts[ref]["state"] == "finished"


def test_crash_after_optimize_before_results_leaves_the_result_reportable(
    reactive: Reactive,
) -> None:
    ref = reactive.approved_optimize()
    reactive.backend.fail_next(
        "reportTvlReactiveAttemptResult", "transport", "transport", "transport"
    )
    with pytest.raises(RuntimeRequirementError) as err:
        go(reactive, Fn(reactive))
    assert err.value.path == "result_unreported"
    assert reactive.backend.attempts[ref]["state"] == "started"
    reactive.clock.advance(minutes=10)  # lapsed meanwhile
    res = reactive.backend.reservations[
        reactive.backend.attempts[ref]["reservation_ref"]
    ]
    from traigent.runtime.hosted import ReservationHandle

    handle = ReservationHandle(
        res["reservation_ref"], res["generation"], ref, "optimize", res["expires_at"]
    )
    assert reactive.client.report_finished(handle, completion="normal", candidates=[])[
        "recorded"
    ]
    assert reactive.backend.starts_accepted[ref] == 1


def test_attempt_ref_without_agent_ref_is_a_usage_error(user_module: Any) -> None:
    from click.testing import CliRunner

    from traigent.cli.main import cli

    out = CliRunner().invoke(
        cli,
        [
            "requirements",
            "respond",
            __file__,
            "--function",
            "answer",
            "--project-id",
            "p",
            "--agent-id",
            "a",
            "--attempt-ref",
            "attemptref_" + "0" * 32,
        ],
    )
    assert out.exit_code == 2 and "--attempt-ref requires --agent-ref" in out.output
