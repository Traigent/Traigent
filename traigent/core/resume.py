"""Resume a local optimization run from a prior result (Traigent#1697).

``optimize(resume_from=...)`` reuses trials a previous run already finished, so
an interrupted or extended study does not pay for them twice. Reuse is
fail-closed:

* A prior trial is reused only when the prior run's dataset, evaluator (scorer),
  agent and configuration-space fingerprints all equal the current run's. The
  fingerprints are the existing ``fp1`` artifact fingerprints
  (:mod:`traigent.utils.artifact_fingerprints`); no new hashing scheme is used.
  A missing fingerprint on either side counts as a mismatch.
* Only trials with status ``completed`` and a measured or properly-unknown cost
  are reused. Running, pending, failed, cancelled and unrecognised trials are
  neither reused nor retried automatically; they are simply not counted.
* A result loaded by name goes through ``PersistenceManager.load_result``, which
  verifies the #1961 sha256 integrity manifest. A failed check raises
  ``ValueError``; nothing is reused.

Cost rule: a reused trial's cost is reported in the result totals (it is part of
the study's spend) but is not charged again against the new run's cost limit,
because the new run did not spend it. Unknown cost stays unknown: the
``cost_unmeasured`` flag travels with the trial.

All warnings and errors here are value-free: they name the identity component
or the reason, never a path, a name, a score or a config value.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.utils.logging import get_logger
from traigent.utils.trial_costs import coerce_finite_cost, is_cost_unmeasured

logger = get_logger(__name__)

#: Identity components that must all match for any trial to be reused.
IDENTITY_COMPONENTS: tuple[str, ...] = ("dataset", "evaluator", "agent", "config_space")

#: Result metadata key under which a run records its fingerprints.
FINGERPRINTS_METADATA_KEY = "artifact_fingerprints"
#: Trial metadata key marking a trial reused from a prior run.
REUSED_TRIAL_KEY = "resumed_from_prior_run"
#: Result metadata key summarising the resume.
RESUME_METADATA_KEY = "resume"

LOCAL_ONLY_MESSAGE = "resume_from supports local runs only for now"


@dataclass(frozen=True)
class ResumePlan:
    """What a resume will reuse, and why anything was left out."""

    reusable: tuple[TrialResult, ...] = ()
    not_reusable_unfinished: int = 0
    not_reusable_other: int = 0
    mismatched_components: tuple[str, ...] = ()
    identity_unavailable: bool = False
    dropped_over_max_trials: int = 0


def load_resume_source(resume_from: Any) -> OptimizationResult:
    """Return the prior result named by ``resume_from``.

    Accepts an ``OptimizationResult`` or a saved-result handle: a name (under
    ``.traigent``) or a directory path returned by
    ``PersistenceManager.save_result``. Raises ``TypeError`` for any other type
    and ``ValueError`` (value-free) for a missing, corrupted or tampered result.
    """
    if isinstance(resume_from, OptimizationResult):
        return resume_from
    if isinstance(resume_from, (str, os.PathLike)):
        from traigent.utils.persistence import PersistenceManager

        raw = Path(os.fspath(resume_from))
        if raw.name in ("", ".", "..") or "\x00" in str(raw):
            raise ValueError("resume_from does not name a saved result")
        if raw.is_absolute() or len(raw.parts) > 1:
            manager = PersistenceManager(base_dir=raw.parent)
        else:
            manager = PersistenceManager()
        try:
            return manager.load_result(raw.name)
        except FileNotFoundError:
            raise ValueError("resume_from: saved result not found") from None
        except ValueError:
            raise ValueError(
                "resume_from: saved result is corrupted or failed integrity "
                "verification; nothing was reused"
            ) from None
        except Exception:
            raise ValueError(
                "resume_from: saved result could not be loaded; nothing was reused"
            ) from None
    raise TypeError(
        "resume_from must be an OptimizationResult or the name/path of a result "
        "saved with PersistenceManager.save_result"
    )


def _trial_cost_ok(trial: TrialResult) -> bool:
    """True when the trial's cost is measured or properly flagged unknown."""
    metrics = trial.metrics or {}
    if is_cost_unmeasured(metrics):
        return True
    for key in ("total_cost", "cost"):
        if key in metrics and coerce_finite_cost(metrics[key]) is not None:
            return True
    metadata = trial.metadata or {}
    for key in ("total_example_cost", "total_cost", "cost"):
        if key in metadata and coerce_finite_cost(metadata[key]) is not None:
            return True
    return False


def differing_components(
    prior: Mapping[str, Any] | None, current: Mapping[str, Any] | None
) -> tuple[tuple[str, ...], bool]:
    """Return (components that differ, identity_unavailable).

    A component missing on either side differs; ``identity_unavailable`` is set
    when either side has no fingerprints at all.
    """
    if not isinstance(prior, Mapping) or not isinstance(current, Mapping):
        return IDENTITY_COMPONENTS, True
    differing: list[str] = []
    for key in IDENTITY_COMPONENTS:
        a, b = prior.get(key), current.get(key)
        if not isinstance(a, str) or not isinstance(b, str) or a != b:
            differing.append(key)
    return tuple(differing), False


def plan_resume(
    prior: OptimizationResult,
    current_fingerprints: Mapping[str, Any] | None,
    *,
    max_trials: int | None = None,
) -> ResumePlan:
    """Decide which prior trials to reuse. Never raises on mismatch."""
    prior_fps = (prior.metadata or {}).get(FINGERPRINTS_METADATA_KEY)
    mismatched, unavailable = differing_components(prior_fps, current_fingerprints)

    unfinished = 0
    other = 0
    eligible: list[TrialResult] = []
    for trial in prior.trials:
        if trial.status == TrialStatus.COMPLETED:
            if _trial_cost_ok(trial):
                eligible.append(trial)
            else:
                other += 1
        elif trial.status in (
            TrialStatus.RUNNING,
            TrialStatus.PENDING,
            TrialStatus.NOT_STARTED,
        ):
            unfinished += 1
        else:
            other += 1

    if mismatched:
        return ResumePlan(
            reusable=(),
            not_reusable_unfinished=unfinished,
            not_reusable_other=other,
            mismatched_components=mismatched,
            identity_unavailable=unavailable,
        )

    dropped = 0
    if max_trials is not None and len(eligible) > max_trials:
        dropped = len(eligible) - max_trials
        eligible = eligible[:max_trials]

    reused: list[TrialResult] = []
    for trial in eligible:
        clone = copy.deepcopy(trial)
        clone.metadata = {
            **(clone.metadata or {}),
            REUSED_TRIAL_KEY: True,
            "resumed_original_trial_id": str(trial.trial_id),
        }
        clone.trial_id = f"resumed-{trial.trial_id}"
        reused.append(clone)

    return ResumePlan(
        reusable=tuple(reused),
        not_reusable_unfinished=unfinished,
        not_reusable_other=other,
        dropped_over_max_trials=dropped,
    )


def warn_if_not_reusable(plan: ResumePlan) -> None:
    """Emit one value-free warning when identity prevented any reuse."""
    if not plan.mismatched_components:
        return
    if plan.identity_unavailable:
        logger.warning(
            "resume_from: reusing no prior trials; run identity "
            "(fingerprints) is unavailable on the prior result or this run."
        )
        return
    logger.warning(
        "resume_from: reusing no prior trials; these identity components "
        "differ from the prior run: %s.",
        ", ".join(plan.mismatched_components),
    )
