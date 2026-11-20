"""Confirmation status of a regression row, derived from its observations.

# Traceability: CONC-Layer-Data FUNC-STORAGE FUNC-ANALYTICS

A row saved by ``traigent regression add`` fails when it is saved. It becomes a
CI gate only after a confirm run has seen it pass:

* ``unconfirmed``: no usable observation yet (never confirmed, or every
  confirm run errored).
* ``known_failure``: never observed passing every repeat; its latest usable
  observation failed every repeat (or it was disarmed). Reported, never gated.
* ``intermittent``: its latest usable observation mixed passes and failures
  and the row was never armed. Reported, gated only on request.
* ``armed``: some confirm run saw it pass every repeat. A later failure is a
  regression. An armed row stays armed whatever later observations say; only
  an explicit ``disarm`` observation returns it to ``known_failure``.

The status is a pure function of the append-only ``observations`` list
(newest last), so it is replayed, never trusted from the stored field. Rows are
schema version 2 once confirmed; a version 1 row (written by ``add``) has no
observations and reads as ``unconfirmed``.

An observation holds counts, an interval, a date and a config digest. Never
example content.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

STATUS_UNCONFIRMED = "unconfirmed"
STATUS_KNOWN_FAILURE = "known_failure"
STATUS_INTERMITTENT = "intermittent"
STATUS_ARMED = "armed"
STATUSES: tuple[str, ...] = (
    STATUS_UNCONFIRMED,
    STATUS_KNOWN_FAILURE,
    STATUS_INTERMITTENT,
    STATUS_ARMED,
)

OUTCOME_ALL_PASS = "all_pass"
OUTCOME_ALL_FAIL = "all_fail"
OUTCOME_MIXED = "mixed"
OUTCOME_ERRORED = "errored"
OUTCOME_DISARMED = "disarmed"
OBSERVATION_OUTCOMES: tuple[str, ...] = (
    OUTCOME_ALL_PASS,
    OUTCOME_ALL_FAIL,
    OUTCOME_MIXED,
    OUTCOME_ERRORED,
    OUTCOME_DISARMED,
)

#: Closed set of reasons recorded with a disarm observation.
DISARM_REASONS: tuple[str, ...] = (
    "expected_behavior_change",
    "example_is_wrong",
    "known_flaky",
    "other",
)

#: ``label`` (the version 1 vocabulary) mirrors the status so a label filter
#: keeps working on a confirmed file.
LABEL_FOR_STATUS = {
    STATUS_UNCONFIRMED: "seen_once",
    STATUS_KNOWN_FAILURE: "seen_once",
    STATUS_INTERMITTENT: "intermittent",
    STATUS_ARMED: "reliable",
}


def derive_status(observations: Sequence[Mapping[str, Any]] | None) -> str:
    """Replay ``observations`` (oldest first) into the row's status."""
    status = STATUS_UNCONFIRMED
    for observation in observations or ():
        outcome = observation.get("outcome")
        if outcome == OUTCOME_DISARMED:
            status = STATUS_KNOWN_FAILURE
        elif outcome == OUTCOME_ALL_PASS:
            status = STATUS_ARMED
        elif status == STATUS_ARMED:
            continue  # armed is sticky: a later failure is a regression
        elif outcome == OUTCOME_ALL_FAIL:
            status = STATUS_KNOWN_FAILURE
        elif outcome == OUTCOME_MIXED:
            status = STATUS_INTERMITTENT
        # "errored" says nothing: the status stays as it was
    return status


def effective_status(regression: Mapping[str, Any]) -> str:
    """The status of a ``metadata.regression`` block of either schema version."""
    observations = regression.get("observations")
    return derive_status(observations if isinstance(observations, list) else None)


def latest_outcome(regression: Mapping[str, Any]) -> str | None:
    """Outcome of the newest observation, or ``None`` when there is none."""
    observations = regression.get("observations")
    if isinstance(observations, list) and observations:
        last = observations[-1]
        return last.get("outcome") if isinstance(last, dict) else None
    return None


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _observation_problem(index: int, observation: Any) -> str | None:
    name = f"'metadata.regression.observations[{index}]'"
    if not isinstance(observation, dict):
        return f"{name} is not an object"
    outcome = observation.get("outcome")
    if outcome not in OBSERVATION_OUTCOMES:
        return f"{name}.outcome is not one of {list(OBSERVATION_OUTCOMES)}"
    if not isinstance(observation.get("date"), str):
        return f"{name}.date is not a string"
    for key in ("reps", "passes", "fails", "errored"):
        if not _is_count(observation.get(key)):
            return f"{name}.{key} is not a count"
    if outcome == OUTCOME_DISARMED:
        if observation.get("reason") not in DISARM_REASONS:
            return f"{name}.reason is not one of {list(DISARM_REASONS)}"
    elif not isinstance(observation.get("config_digest"), str):
        return f"{name}.config_digest is not a string"
    return None


def confirmed_fields_problem(regression: Mapping[str, Any]) -> str | None:
    """What is wrong with the version 2 fields of a row, or ``None``.

    The stored ``status`` must equal the status the observations replay to; a
    hand edit of one without the other is corruption, not a quiet arming.
    """
    observations = regression.get("observations")
    if not isinstance(observations, list):
        return "'metadata.regression.observations' is not a list"
    for index, observation in enumerate(observations):
        problem = _observation_problem(index, observation)
        if problem is not None:
            return problem
    derived = derive_status(observations)
    if regression.get("status") != derived:
        return (
            "'metadata.regression.status' does not match its observations "
            f"(they replay to {derived!r}); the row was edited by hand"
        )
    if derived == STATUS_ARMED:
        if not isinstance(regression.get("armed_at"), str):
            return "'metadata.regression.armed_at' is not a string on an armed row"
    return None


def _latest_usable(observations: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for observation in reversed(observations):
        if observation.get("outcome") in (
            OUTCOME_ALL_PASS,
            OUTCOME_ALL_FAIL,
            OUTCOME_MIXED,
        ):
            return observation
    return {}


def append_observation(
    regression: dict[str, Any], observation: dict[str, Any], schema_version: int
) -> tuple[str, str]:
    """Append ``observation`` to a ``metadata.regression`` block, in place.

    Upgrades a version 1 block to ``schema_version``, replays the status,
    records ``armed_at`` and ``armed_config_digest`` the first time the row is
    armed (cleared again by a disarm), and keeps ``label``, ``fails`` and
    ``reps`` in step. Returns ``(status_before, status_after)``.
    """
    observations = regression.setdefault("observations", [])
    before = derive_status(observations)
    observations.append(observation)
    after = derive_status(observations)
    regression["schema_version"] = schema_version
    regression["status"] = after
    regression["label"] = LABEL_FOR_STATUS[after]
    if after == STATUS_ARMED:
        if before != STATUS_ARMED:
            regression["armed_at"] = observation["date"]
            regression["armed_config_digest"] = observation.get("config_digest")
    else:
        regression["armed_at"] = None
        regression["armed_config_digest"] = None
    usable = _latest_usable(observations)
    regression["fails"] = usable.get("fails")
    regression["reps"] = usable.get("reps")
    return before, after
