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
        "content_digest": "c",
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


def test_only_observations_on_the_current_content_count() -> None:
    old = {**_obs("all_pass"), "content_digest": "old"}
    new_fail = {**_obs("all_fail"), "content_digest": "new"}
    assert derive_status([old], "old") == "armed"
    assert derive_status([old], "new") == "unconfirmed"
    assert derive_status([old, new_fail], "new") == "known_failure"
    assert derive_status([old, new_fail], "old") == "armed"
    block = {
        "observations": [old],
        "status": "armed",
        "armed_at": old["date"],
        "armed_config_digest": old["config_digest"],
    }
    assert effective_status(block, "old") == "armed"
    assert effective_status(block, "new") == "unconfirmed"
    assert confirmed_fields_problem(block) is None  # edited content is not corrupt


def test_arming_does_not_carry_over_to_new_content() -> None:
    block: dict = {"schema_version": 1, "label": "seen_once", "provenance": []}
    append_observation(block, {**_obs("all_pass"), "content_digest": "a"}, 2)
    assert block["status"] == "armed"
    before, after = append_observation(
        block, {**_obs("all_fail"), "content_digest": "b"}, 2
    )
    assert (before, after) == ("unconfirmed", "known_failure")
    assert block["armed_at"] is None and block["label"] == "seen_once"


def test_an_observation_without_a_content_digest_is_refused() -> None:
    legacy = {k: v for k, v in _obs("all_pass").items() if k != "content_digest"}
    problem = confirmed_fields_problem({"observations": [legacy], "status": "armed"})
    assert "content_digest" in (problem or "")


def test_arming_fields_are_validated_against_the_observations() -> None:
    block: dict = {"schema_version": 1, "label": "seen_once", "provenance": []}
    append_observation(block, _obs("all_pass", "2026-01-05"), 2)
    assert confirmed_fields_problem(block) is None
    for key, wrong in (
        ("armed_at", None),
        ("armed_at", "2020-01-01"),
        ("armed_config_digest", None),
        ("armed_config_digest", "other"),
    ):
        problem = confirmed_fields_problem({**block, key: wrong})
        assert problem is not None and key in problem
    unarmed: dict = {"schema_version": 1, "label": "seen_once", "provenance": []}
    append_observation(unarmed, _obs("all_fail"), 2)
    assert confirmed_fields_problem({**unarmed, "armed_at": "2026-01-01"})
