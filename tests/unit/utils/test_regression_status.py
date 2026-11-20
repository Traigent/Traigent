"""Replaying observations into a regression row's status."""

from __future__ import annotations

import pytest

from traigent.utils.regression_status import (
    append_observation,
    confirmed_fields_problem,
    derive_status,
    effective_status,
)


def _obs(outcome: str, date: str = "2026-01-01") -> dict:
    return {
        "config_digest": "d",
        "date": date,
        "reps": 5,
        "passes": 5 if outcome == "all_pass" else 0,
        "fails": 0,
        "errored": 0,
        "pass_rate_low": None,
        "pass_rate_high": None,
        "outcome": outcome,
    }


@pytest.mark.parametrize(
    ("outcomes", "status"),
    [
        ([], "unconfirmed"),
        (["errored"], "unconfirmed"),
        (["all_fail"], "known_failure"),
        (["mixed"], "intermittent"),
        (["all_pass"], "armed"),
        (["all_fail", "errored"], "known_failure"),
        (["all_fail", "mixed"], "intermittent"),
        (["all_fail", "all_pass"], "armed"),
        (["all_pass", "all_fail"], "armed"),
        (["all_pass", "mixed", "errored"], "armed"),
    ],
)
def test_status_replays_from_observations(outcomes, status) -> None:
    assert derive_status([_obs(o) for o in outcomes]) == status


def test_a_version_one_block_reads_as_unconfirmed() -> None:
    assert effective_status({"schema_version": 1, "label": "seen_once"}) == (
        "unconfirmed"
    )


def test_arming_is_recorded_once_and_cleared_by_a_disarm() -> None:
    block: dict = {"schema_version": 1, "label": "seen_once", "provenance": []}
    assert append_observation(block, _obs("all_fail", "d1"), 2) == (
        "unconfirmed",
        "known_failure",
    )
    assert block["armed_at"] is None
    append_observation(block, _obs("all_pass", "d2"), 2)
    append_observation(block, _obs("all_pass", "d3"), 2)
    assert (block["armed_at"], block["armed_config_digest"]) == ("d2", "d")
    assert block["label"] == "reliable" and block["schema_version"] == 2
    disarm = {**_obs("disarmed", "d4"), "config_digest": None, "reason": "other"}
    append_observation(block, disarm, 2)
    assert block["status"] == "known_failure" and block["armed_at"] is None
    assert confirmed_fields_problem(block) is None


def test_corrupt_version_two_fields_are_named() -> None:
    block: dict = {"observations": [_obs("all_pass")], "status": "known_failure"}
    assert "does not match" in (confirmed_fields_problem(block) or "")
    bad = {"observations": [{**_obs("all_pass"), "reps": -1}], "status": "armed"}
    assert "reps" in (confirmed_fields_problem(bad) or "")
    no_reason = {
        "observations": [{**_obs("disarmed"), "config_digest": None}],
        "status": "known_failure",
    }
    assert "reason" in (confirmed_fields_problem(no_reason) or "")
