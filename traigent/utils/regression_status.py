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
(newest last) and the row's current ``content_digest``, so it is replayed,
never trusted from the stored field. Rows are schema version 2 once confirmed;
a version 1 row (written by ``add``) has no observations and reads as
``unconfirmed``.

Every observation records the ``content_digest`` of the example it was measured
on, and only observations whose digest equals the row's current content digest
count. Editing a row's content (and recomputing its digest) therefore returns
it to ``unconfirmed`` until a confirm run sees the new content pass; the old
observations stay in the history, belonging to the earlier content, and do not
arm. (Matching is by digest alone: restoring content that once passed makes
its own observations count again.) ``add --on-id-conflict replace`` writes a fresh row, with the same effect.
A version 2 observation written without a ``content_digest`` (only an unmerged
preview of this format did) is refused as an invalid file: it cannot be tied
to any content, so it is not silently counted and not silently dropped.

An observation holds counts, an interval, a date, a config digest and a content
digest. Never example content.

Trust: the regression file is edited by people and is NOT authenticated
evidence. A hand edit that keeps the file internally consistent (observations,
stored status, digests) can arm or disarm a row, and replay validation cannot
tell it from a real confirm run. Changes to the file belong in your
repository's normal review; this module does not sign or verify who wrote it.
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


def derive_status(
    observations: Sequence[Mapping[str, Any]] | None,
    content_digest: str | None = None,
) -> str:
    """Replay ``observations`` (oldest first) into the row's status.

    With ``content_digest`` only observations measured on that content count;
    ``None`` replays every observation (the unbound form, for callers that have
    no digest to bind to).
    """
    status = STATUS_UNCONFIRMED
    for observation in observations or ():
        if (
            content_digest is not None
            and observation.get("content_digest") != content_digest
        ):
            continue  # measured on other content: history, not evidence
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


def effective_status(
    regression: Mapping[str, Any], content_digest: str | None = None
) -> str:
    """The status of a ``metadata.regression`` block of either schema version.

    Pass the row's current ``metadata.content_digest`` to count only the
    observations measured on its current content.
    """
    observations = regression.get("observations")
    return derive_status(
        observations if isinstance(observations, list) else None, content_digest
    )


def _observation_list(regression: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    observations = regression.get("observations")
    return observations if isinstance(observations, list) else []


def status_when_last_measured(regression: Mapping[str, Any]) -> str:
    """The status the row had on the content of its newest observation."""
    observations = _observation_list(regression)
    if not observations:
        return STATUS_UNCONFIRMED
    return derive_status(observations, observations[-1].get("content_digest"))


def content_changed_since_measured(
    regression: Mapping[str, Any], content_digest: str
) -> bool:
    """The row has observations, none of them on its current content."""
    observations = _observation_list(regression)
    return bool(observations) and all(
        o.get("content_digest") != content_digest for o in observations
    )


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
    if not isinstance(observation.get("content_digest"), str):
        return (
            f"{name}.content_digest is not a string (an observation written "
            "before observations were bound to the content they measured; "
            "re-create the file with 'traigent regression add', then confirm)"
        )
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

    The stored ``status`` must equal the status the observations replay to on
    the content of the newest one (the status as last written); a hand edit of
    one without the other is corruption, not a quiet arming. A row whose
    content was edited since is still valid here: its effective status, bound
    to its current content digest, is what gates (see ``effective_status``).
    """
    observations = regression.get("observations")
    if not isinstance(observations, list):
        return "'metadata.regression.observations' is not a list"
    for index, observation in enumerate(observations):
        problem = _observation_problem(index, observation)
        if problem is not None:
            return problem
    derived = status_when_last_measured(regression)
    if regression.get("status") != derived:
        return (
            "'metadata.regression.status' does not match its observations "
            f"(they replay to {derived!r}); the row was edited by hand"
        )
    if derived == STATUS_ARMED:
        if not isinstance(regression.get("armed_at"), str):
            return "'metadata.regression.armed_at' is not a string on an armed row"
    return None


def _latest_usable(
    observations: Sequence[Mapping[str, Any]], content_digest: Any
) -> Mapping[str, Any]:
    for observation in reversed(observations):
        if observation.get("content_digest") != content_digest:
            continue
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

    The observation's ``content_digest`` names the content it was measured on;
    the status is replayed over the observations of that content only (older
    ones stay in the list but count for nothing). Upgrades a version 1 block to
    ``schema_version``, replays the status,
    records ``armed_at`` and ``armed_config_digest`` the first time the row is
    armed (cleared again by a disarm), and keeps ``label``, ``fails`` and
    ``reps`` in step. Returns ``(status_before, status_after)``.
    """
    observations = regression.setdefault("observations", [])
    bound_to = observation["content_digest"]
    before = derive_status(observations, bound_to)
    observations.append(observation)
    after = derive_status(observations, bound_to)
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
    usable = _latest_usable(observations, bound_to)
    regression["fails"] = usable.get("fails")
    regression["reps"] = usable.get("reps")
    return before, after
