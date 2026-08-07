"""Issue #2047: a failed trial must reload as a failed trial, in BOTH formats.

The nested twin of #2031. `OptimizationResult` got an explicit persistence
manifest there; `TrialResult` was deliberately left out of scope, so both
persisted formats rebuilt trials from a hand-written 8-of-10-field constructor
call and dropped `error` and `score` on the way back.

The two losses had different shapes, which is why both formats are tested here:

* ``config_state`` dumps via ``asdict``, so both fields were always ON DISK;
  only the decoder discarded them.
* ``persistence`` wrote neither key, so the loss happened at write time.

Either way a crashed trial reloaded indistinguishable from one that merely
scored badly, and failure-rate or error-clustering analysis over reloaded
results silently under-reported.

Note that `error_message` is NOT part of this defect — both formats already
round-tripped it (see the comment at `utils/persistence.py`, which records that
earlier fix). Only `error` and `score` were lost.
"""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from traigent.api.types import (
    OptimizationResult,
    TrialError,
    TrialResult,
    TrialStatus,
)
from traigent.core.config_state_manager import ConfigStateManager
from traigent.utils.optimization_result_persistence import (
    TRIAL_RESET,
    TRIAL_RESTORE,
    _SENTINELS,
    _TRIAL_SENTINELS,
)
from traigent.utils.persistence import PersistenceManager

_TRIALS_JSON = "trials.json.gz"


def _failed_trial() -> TrialResult:
    """A trial that CRASHED, as distinct from one that scored badly.

    ``score=None`` is the honest value for a crashed trial: there is no
    objective value, and coercing it to 0.0 would enter selection as a
    legitimate losing score.
    """
    return TrialResult(
        trial_id="trial-2047-failed",
        config={"model": "smart", "temperature": 0.7},
        metrics={},
        status=TrialStatus.FAILED,
        duration=2.5,
        timestamp=datetime(2026, 4, 1, 9, 15, 0, tzinfo=UTC),
        error_message="provider refused the request",
        metadata={"replicate": 2},
        error=TrialError(
            message="provider refused the request",
            error_type="RuntimeError",
            traceback="Traceback (most recent call last):\n  RuntimeError: provider refused",
            timestamp=datetime(2026, 4, 1, 9, 15, 0, tzinfo=UTC),
            config={"model": "smart"},
        ),
        score=None,
    )


def _scored_trial() -> TrialResult:
    """A trial that succeeded and carries a non-default optimization signal."""
    return TrialResult(
        trial_id="trial-2047-scored",
        config={"model": "cheap"},
        metrics={"accuracy": 0.82, "score": 0.82},
        status=TrialStatus.COMPLETED,
        duration=1.25,
        timestamp=datetime(2026, 4, 1, 9, 16, 0, tzinfo=UTC),
        metadata={},
        error=None,
        score=0.82,
    )


def _result_with(trials: list[TrialResult]) -> OptimizationResult:
    data = copy.deepcopy(_SENTINELS)
    data["trials"] = trials
    return OptimizationResult(**data)


def _manager(tmp_path) -> ConfigStateManager:
    def _fn(x):
        return x

    return ConfigStateManager(
        func=_fn,
        default_config={"model": "cheap"},
        local_storage_path=str(tmp_path / "store"),
        configuration_space={"model": ["cheap", "smart"]},
        auto_load_best=False,
        load_from=None,
        setup_wrapper_callback=lambda: None,
    )


def _config_state_round_trip(tmp_path, trials: list[TrialResult]) -> OptimizationResult:
    path = tmp_path / "results.json"
    saver = _manager(tmp_path)
    saver._optimization_results = _result_with(trials)
    saver.save_optimization_results(str(path))

    loader = _manager(tmp_path)
    loader.load_optimization_results(str(path))
    loaded = loader._optimization_results
    assert loaded is not None
    return loaded


def _persistence_round_trip(tmp_path, trials: list[TrialResult]) -> OptimizationResult:
    store = PersistenceManager(base_dir=tmp_path)
    store.save_result(_result_with(trials), "round-trip")
    return store.load_result("round-trip")


# ---------------------------------------------------------------------------
# The manifest itself
# ---------------------------------------------------------------------------


def test_trial_manifest_covers_every_field() -> None:
    """Adding a TrialResult field without classifying it must fail here.

    This is the guard the issue asks for: the manifest is derived from the
    dataclass, so a new field is a test failure rather than a silent tenth
    thing that quietly stops round-tripping.
    """
    declared = {f.name for f in dataclasses.fields(TrialResult)}
    classified = TRIAL_RESTORE | TRIAL_RESET

    unclassified = declared - classified
    assert not unclassified, (
        f"TrialResult field(s) {sorted(unclassified)} are in neither "
        f"TRIAL_RESTORE nor TRIAL_RESET. Classify each one: restored because it "
        f"is a durable fact about the trial, or reset with the reason why."
    )

    stale = classified - declared
    assert not stale, (
        f"The trial manifest names {sorted(stale)}, which TrialResult no longer "
        f"declares. Remove them so the manifest cannot drift."
    )


def test_trial_manifest_partition_is_disjoint() -> None:
    assert not (TRIAL_RESTORE & TRIAL_RESET)


def test_error_and_score_are_classified_as_restored() -> None:
    """The two fields this issue is about, pinned by name.

    Without this, a future edit could satisfy the completeness guard above by
    moving them into TRIAL_RESET and reintroducing the exact defect.
    """
    assert "error" in TRIAL_RESTORE
    assert "score" in TRIAL_RESTORE


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — a failed trial reloads carrying its error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("round_trip", ["config_state", "persistence"])
def test_failed_trial_reloads_carrying_its_error(tmp_path, round_trip) -> None:
    """Both formats. Before #2047 `loaded.error` was None in each."""
    trip = (
        _config_state_round_trip
        if round_trip == "config_state"
        else _persistence_round_trip
    )
    loaded = trip(tmp_path, [_failed_trial()])

    (trial,) = loaded.trials
    assert trial.error is not None, (
        f"{round_trip}: the crashed trial reloaded with error=None, so it is "
        f"indistinguishable from a trial that merely scored badly."
    )
    assert trial.error.error_type == "RuntimeError"
    assert trial.error.message == "provider refused the request"
    assert "RuntimeError: provider refused" in trial.error.traceback
    assert trial.error.timestamp == datetime(2026, 4, 1, 9, 15, 0, tzinfo=UTC)
    assert trial.error.config == {"model": "smart"}
    assert trial.status is TrialStatus.FAILED


@pytest.mark.parametrize("round_trip", ["config_state", "persistence"])
def test_score_survives_the_round_trip(tmp_path, round_trip) -> None:
    """The optimization signal best_config argmaxes (#1845) must come back."""
    trip = (
        _config_state_round_trip
        if round_trip == "config_state"
        else _persistence_round_trip
    )
    loaded = trip(tmp_path, [_scored_trial(), _failed_trial()])

    by_id = {t.trial_id: t for t in loaded.trials}
    assert by_id["trial-2047-scored"].score == pytest.approx(0.82)
    assert by_id["trial-2047-failed"].score is None, (
        "a crashed trial has no objective value; None must not become 0.0, "
        "which would enter selection as a legitimate losing score"
    )


@pytest.mark.parametrize("round_trip", ["config_state", "persistence"])
def test_every_restored_trial_field_survives(tmp_path, round_trip) -> None:
    """Driven off TRIAL_RESTORE so it cannot drift from the manifest."""
    trip = (
        _config_state_round_trip
        if round_trip == "config_state"
        else _persistence_round_trip
    )
    original = _failed_trial()
    (loaded,) = trip(tmp_path, [original]).trials

    for name in sorted(TRIAL_RESTORE):
        assert getattr(loaded, name) == getattr(original, name), (
            f"{round_trip}: TrialResult.{name} is in TRIAL_RESTORE but did not "
            f"survive save -> load"
        )


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — legacy payloads still load
# ---------------------------------------------------------------------------


def test_legacy_persistence_payload_without_new_keys_still_loads(tmp_path) -> None:
    """An artifact written before this format emitted error/score.

    Those trials genuinely had no persisted error, so None is the correct
    restored value -- not a load failure.
    """
    store = PersistenceManager(base_dir=tmp_path)
    result_dir = Path(store.save_result(_result_with([_failed_trial()]), "legacy"))

    trials_file = result_dir / _TRIALS_JSON
    with gzip.open(trials_file, "rt") as handle:
        trials_data = json.load(handle)
    for trial in trials_data:
        trial.pop("error", None)
        trial.pop("score", None)
    with gzip.open(trials_file, "wt") as handle:
        json.dump(trials_data, handle)

    (loaded,) = store.load_result("legacy").trials
    assert loaded.error is None
    assert loaded.score is None
    assert loaded.trial_id == "trial-2047-failed"


def test_legacy_config_state_payload_without_new_keys_still_loads(tmp_path) -> None:
    path = tmp_path / "results.json"
    saver = _manager(tmp_path)
    saver._optimization_results = _result_with([_failed_trial()])
    saver.save_optimization_results(str(path))

    payload = json.loads(path.read_text())
    for trial in payload["trials"]:
        trial.pop("error", None)
        trial.pop("score", None)
    path.write_text(json.dumps(payload))

    loader = _manager(tmp_path)
    loader.load_optimization_results(str(path))
    assert loader._optimization_results is not None
    (loaded,) = loader._optimization_results.trials
    assert loaded.error is None
    assert loaded.score is None


# ---------------------------------------------------------------------------
# Malformed payloads must not destroy the load (red-team findings 1-4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "corrupt",
    ["a-legacy-stringified-error", 42, ["not", "a", "mapping"]],
    ids=["str", "int", "list"],
)
def test_non_mapping_error_payload_does_not_abort_the_whole_load(
    tmp_path, corrupt, caplog
) -> None:
    """`asdict` does not recurse into a duck-typed `trial.error`.

    The legacy `json.dump(default=str)` writer therefore stringified it, and
    such artifacts are on disk today. An earlier revision of this fix raised on
    them, which turned a recoverable missing-field into an unloadable run. The
    trial still loads; the loss is logged rather than silent.
    """
    store = PersistenceManager(base_dir=tmp_path)
    result_dir = Path(
        store.save_result(_result_with([_failed_trial()]), "legacy-shape")
    )

    trials_file = result_dir / _TRIALS_JSON
    with gzip.open(trials_file, "rt") as handle:
        trials_data = json.load(handle)
    trials_data[0]["error"] = corrupt
    with gzip.open(trials_file, "wt") as handle:
        json.dump(trials_data, handle)

    with caplog.at_level(logging.WARNING):
        (loaded,) = store.load_result("legacy-shape").trials

    assert loaded.trial_id == "trial-2047-failed"
    assert loaded.status is TrialStatus.FAILED
    assert loaded.error_message == "provider refused the request"
    assert loaded.error is None
    assert "not a mapping" in caplog.text


def test_non_numeric_score_does_not_abort_the_load(tmp_path, caplog) -> None:
    store = PersistenceManager(base_dir=tmp_path)
    result_dir = Path(store.save_result(_result_with([_scored_trial()]), "bad-score"))

    trials_file = result_dir / _TRIALS_JSON
    with gzip.open(trials_file, "rt") as handle:
        trials_data = json.load(handle)
    trials_data[0]["score"] = "not-a-number"
    with gzip.open(trials_file, "wt") as handle:
        json.dump(trials_data, handle)

    with caplog.at_level(logging.WARNING):
        (loaded,) = store.load_result("bad-score").trials

    assert loaded.score is None, "a non-numeric score must not be guessed at"
    assert "not a number" in caplog.text


# ---------------------------------------------------------------------------
# Regression: persisting error/score must not be able to destroy a run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_config", "label"),
    [
        ({"tags": {"a", "b"}}, "set"),
        ({"cutoff": datetime(2026, 4, 1, tzinfo=UTC)}, "datetime"),
        ({"client": object()}, "opaque-object"),
    ],
)
def test_a_non_json_primitive_error_config_still_saves(
    tmp_path, bad_config, label
) -> None:
    """`_atomic_write_gzip_json` calls `json.dump` with no `default=`.

    Writing `error.config` unhardened therefore raised mid-write and left the
    run with only `metadata.json`, so `load_result` then failed outright --
    destroying a completed optimization to avoid losing one field. Config
    values of these shapes pass `validate_configuration_space`, so this is
    reachable from the public API.
    """
    trial = _failed_trial()
    trial.error.config = bad_config
    trial.config = bad_config

    store = PersistenceManager(base_dir=tmp_path)
    result_dir = Path(store.save_result(_result_with([trial]), f"hardened-{label}"))

    assert (result_dir / _TRIALS_JSON).exists(), (
        f"{label}: the trials file was never written, so the whole run is gone"
    )
    (loaded,) = store.load_result(f"hardened-{label}").trials
    assert loaded.status is TrialStatus.FAILED
    assert loaded.error is not None


def test_a_decimal_score_still_saves(tmp_path) -> None:
    trial = _scored_trial()
    trial.score = Decimal("0.5")

    store = PersistenceManager(base_dir=tmp_path)
    result_dir = Path(store.save_result(_result_with([trial]), "decimal-score"))
    assert (result_dir / _TRIALS_JSON).exists()
    (loaded,) = store.load_result("decimal-score").trials
    assert loaded.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The completeness guard must be unsatisfiable by classification alone
# ---------------------------------------------------------------------------


def test_trial_sentinel_table_covers_every_restored_field() -> None:
    """Classifying a field is not enough -- it needs a sentinel too.

    Without this, a field could be added to TRIAL_RESTORE, pass the
    completeness guard, and still be silently dropped by both formats, because
    a hand-written fixture compares a new field's default against itself.
    """
    missing = TRIAL_RESTORE - set(_TRIAL_SENTINELS)
    assert not missing, (
        f"{sorted(missing)} are in TRIAL_RESTORE but have no entry in "
        f"_TRIAL_SENTINELS, so no test proves they survive a round trip."
    )
    stale = set(_TRIAL_SENTINELS) - TRIAL_RESTORE
    assert not stale, f"_TRIAL_SENTINELS names non-restored field(s) {sorted(stale)}"


@pytest.mark.parametrize("round_trip", ["config_state", "persistence"])
def test_every_sentinel_field_survives_the_round_trip(tmp_path, round_trip) -> None:
    """Driven off the sentinel table, so a new field is actually exercised.

    `error` is compared field-by-field: `to_dict()` redacts `message` and
    `traceback` on the way out by that object's own contract, so the restored
    text is deliberately not byte-equal to the original.
    """
    trip = (
        _config_state_round_trip
        if round_trip == "config_state"
        else _persistence_round_trip
    )
    original = TrialResult(**copy.deepcopy(_TRIAL_SENTINELS))
    (loaded,) = trip(tmp_path, [original]).trials

    for name in sorted(TRIAL_RESTORE - {"error"}):
        assert getattr(loaded, name) == getattr(original, name), (
            f"{round_trip}: TrialResult.{name} has a sentinel but did not "
            f"survive save -> load"
        )

    assert loaded.error is not None
    assert loaded.error.error_type == original.error.error_type
    assert loaded.error.timestamp == original.error.timestamp
