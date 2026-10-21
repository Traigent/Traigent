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
  are reused. Running, pending and unrecognised-status trials are neither
  reused nor retried: they are not counted, and their configurations are
  excluded from this run's suggestions. Failed and cancelled trials are not
  reused and may be proposed again.
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
import json
import os
from collections.abc import Mapping, Sequence
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
    blocked_configs: tuple[dict[str, Any], ...] = ()
    partial_trials: int = 0
    prefill: ResumePrefill | None = None
    not_reusable_unfinished: int = 0
    not_reusable_other: int = 0
    mismatched_components: tuple[str, ...] = ()
    identity_unavailable: bool = False
    dropped_over_max_trials: int = 0


class _UnverifiedResult(Exception):
    """Internal marker: the saved result has no integrity manifest."""


def load_resume_source(
    resume_from: Any, *, allow_unverified_resume: bool = False
) -> OptimizationResult:
    """Return the prior result named by ``resume_from``.

    Accepts an ``OptimizationResult`` or a saved-result handle: a name (under
    ``.traigent``) or a directory path returned by
    ``PersistenceManager.save_result``. Raises ``TypeError`` for any other type
    and ``ValueError`` (value-free) for a missing, corrupted or tampered result,
    or one without a verifiable integrity manifest (unless
    ``allow_unverified_resume`` is set).
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
            if not allow_unverified_resume:
                from traigent.utils.file_versioning import verify_integrity_manifest
                from traigent.utils.persistence import _INTEGRITY_FILES

                result_dir = manager._resolve_path(manager.base_dir / raw.name)
                if not result_dir.exists():
                    raise FileNotFoundError
                if not verify_integrity_manifest(result_dir, list(_INTEGRITY_FILES)):
                    raise _UnverifiedResult
            return manager.load_result(raw.name)
        except _UnverifiedResult:
            raise ValueError(
                "resume_from: saved result has no integrity manifest, so it "
                "cannot be verified; nothing was reused. Re-save it with "
                "PersistenceManager.save_result, or pass "
                "allow_unverified_resume=True (unsafe)."
            ) from None
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


RESUMED_EXAMPLE_KEY = "resumed_example_from_prior_run"


def config_key(config: Mapping[str, Any], keys: Sequence[str] | None = None) -> str:
    """Canonical, order-independent key for a configuration."""
    items = {k: v for k, v in config.items() if not keys or k in keys}
    return json.dumps(items, sort_keys=True, default=str)


def _row_dict(row: Any) -> dict[str, Any] | None:
    if isinstance(row, Mapping):
        return dict(row)
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            return None
    return None


def stored_example_rows(trial: TrialResult) -> list[dict[str, Any]] | None:
    """Per-example results persisted on a trial, or None when it carries none."""
    metadata = trial.metadata or {}
    raw: Any = None
    evaluation = metadata.get("evaluation_result")
    if isinstance(evaluation, Mapping):
        raw = evaluation.get("example_results")
    elif evaluation is not None:
        raw = getattr(evaluation, "example_results", None)
    if not raw:
        raw = metadata.get("example_results")
    if not isinstance(raw, list) or not raw:
        return None
    rows = [_row_dict(r) for r in raw]
    return [r for r in rows if r is not None] or None


def _row_completed(row: Mapping[str, Any]) -> bool:
    return (
        row.get("success") is True
        and not row.get("error_message")
        and row.get("actual_output") is not None
        and isinstance(row.get("metrics"), Mapping)
        and bool(row.get("metrics"))
        and isinstance(row.get("example_id"), str)
    )


class ResumePrefill:
    """Completed example results of partially-run prior trials, by config.

    Consulted by the evaluator row by row: a row is reused only when its example
    id, input and expected output all equal the current row's. Never raises on a
    miss; a miss simply means the row is evaluated.
    """

    def __init__(
        self,
        pool: dict[str, dict[str, dict[str, Any]]],
        keys: Sequence[str] | None = None,
    ) -> None:
        self._pool = pool
        self._keys = tuple(keys or ())
        self.active = True
        self._stats: dict[str, list[Any]] = {}

    def lookup(
        self,
        config: Mapping[str, Any],
        row_id: str,
        input_data: Any,
        expected_output: Any,
    ) -> dict[str, Any] | None:
        if not self.active:
            return None
        key = config_key(config, self._keys)
        row = self._pool.get(key, {}).get(row_id)
        if row is None:
            return None
        if row.get("input_data") != input_data or (
            row.get("expected_output") != expected_output
        ):
            return None
        metrics = row["metrics"]
        cost: float | None = None
        if not is_cost_unmeasured(metrics):
            cost = coerce_finite_cost(metrics.get("total_cost"))
        stats = self._stats.setdefault(key, [0, 0.0])
        stats[0] += 1
        if cost is not None:
            stats[1] += cost
        return row

    def pop_stats(self, config: Mapping[str, Any]) -> tuple[int, float]:
        """(rows reused, their known cost) for this config since the last pop."""
        stats = self._stats.pop(config_key(config, self._keys), None)
        return (int(stats[0]), float(stats[1])) if stats else (0, 0.0)

    def configs(self) -> int:
        return len(self._pool)


def _known_cost(trial: TrialResult) -> float | None:
    """The measured cost of a trial from any alias, or None."""
    metrics = trial.metrics or {}
    for key in ("total_cost", "cost"):
        if key in metrics:
            cost = coerce_finite_cost(metrics[key])
            if cost is not None:
                return cost
    metadata = trial.metadata or {}
    for key in ("total_example_cost", "total_cost", "cost"):
        if key in metadata:
            cost = coerce_finite_cost(metadata[key])
            if cost is not None:
                return cost
    return None


def _trial_cost_ok(trial: TrialResult) -> bool:
    """True when the trial's cost is measured or properly flagged unknown."""
    return is_cost_unmeasured(trial.metrics or {}) or _known_cost(trial) is not None


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
    dataset_example_ids: Sequence[str] | None = None,
    config_keys: Sequence[str] | None = None,
) -> ResumePlan:
    """Decide which prior trials to reuse. Never raises on mismatch."""
    prior_fps = (prior.metadata or {}).get(FINGERPRINTS_METADATA_KEY)
    mismatched, unavailable = differing_components(prior_fps, current_fingerprints)

    unfinished = 0
    other = 0
    partial = 0
    blocked: list[dict[str, Any]] = []
    eligible: list[TrialResult] = []
    pool: dict[str, dict[str, dict[str, Any]]] = {}
    current_ids = set(dataset_example_ids) if dataset_example_ids else None
    for trial in prior.trials:
        in_flight = trial.status not in (
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.CANCELLED,
            TrialStatus.PRUNED,
        )
        if in_flight:
            # running / pending / not started / unrecognised: outcome unknown.
            unfinished += 1
            blocked.append(copy.deepcopy(trial.config))
            continue
        rows = stored_example_rows(trial) if current_ids is not None else None
        if rows is not None and current_ids is not None:
            done = {
                r["example_id"]: r
                for r in rows
                if _row_completed(r) and r["example_id"] in current_ids
            }
            if len(done) < len(current_ids):
                # Partial: its finished examples are reused when this config
                # runs again; the trial itself is not counted.
                partial += 1
                slot = pool.setdefault(config_key(trial.config, config_keys), {})
                for example_id, row in done.items():
                    slot.setdefault(example_id, row)
                continue
        if trial.status == TrialStatus.COMPLETED and _trial_cost_ok(trial):
            eligible.append(trial)
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
        if not is_cost_unmeasured(clone.metrics):
            # Normalize to the canonical key the totals aggregator reads.
            cost = _known_cost(trial)
            if cost is not None:
                clone.metrics = {**clone.metrics, "total_cost": cost}
        clone.trial_id = f"resumed-{trial.trial_id}"
        reused.append(clone)

    return ResumePlan(
        reusable=tuple(reused),
        blocked_configs=tuple(blocked),
        partial_trials=partial,
        prefill=ResumePrefill(pool, config_keys) if pool else None,
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
