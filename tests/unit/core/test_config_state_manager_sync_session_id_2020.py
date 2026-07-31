"""Issue #2020: `ConfigStateManager` must SAVE `sync_session_id` but not RESTORE it.

``OptimizationResult.sync_session_id``'s docstring promises the id "is not
restored by ``ConfigStateManager.load_optimization_results``". Unlike the
``PersistenceManager`` side, nothing enforces that: the loader rebuilds the
result with an explicit constructor call that simply *omits* the field, so
adding one innocuous ``sync_session_id=result_dict.get("sync_session_id")`` line
would break the promise with every test still green.

That promise is a real guarantee, not an implementation detail. The save side
uses ``asdict(...)``, so the JSON on disk genuinely carries the id — and a
result reloaded from that JSON can come from another machine, or from a store
that has since been cleaned (``traigent sync --clean``). Restoring the id would
hand the caller a session id ``traigent sync`` rejects: the exact #2020 failure
this field exists to fix.

So: assert the id is in the serialized JSON, and assert it is ``None`` after a
round-trip.
"""
# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.config_state_manager import ConfigStateManager

_SYNC_ID = "20260727_005247_744918_answer_b1960eba"


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


def _result_with_sync_id() -> OptimizationResult:
    return OptimizationResult(
        trials=[
            TrialResult(
                trial_id="t0",
                config={"model": "cheap"},
                metrics={"accuracy": 0.9},
                status=TrialStatus.COMPLETED,
                duration=0.01,
                timestamp=datetime.now(UTC),
                metadata={"local_session_id": _SYNC_ID},
            )
        ],
        best_config={"model": "cheap"},
        best_score=0.9,
        optimization_id="opt-2020-round-trip",
        duration=0.1,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
        metadata={"local_session_id": _SYNC_ID},
        sync_session_id=_SYNC_ID,
    )


@pytest.fixture
def round_trip(tmp_path):
    """Save a result carrying a sync id, then load it back through the manager."""
    path = tmp_path / "results.json"

    saver = _manager(tmp_path)
    saver._optimization_results = _result_with_sync_id()
    saver.save_optimization_results(str(path))

    loader = _manager(tmp_path)
    loader.load_optimization_results(str(path))
    return path, loader


def test_saved_json_really_carries_the_sync_session_id(round_trip) -> None:
    """The save side is `asdict(...)`, so the field IS on disk.

    Asserted against the real serialized shape rather than the dataclass, so the
    non-restoration below is provably the loader's doing and not an artifact of
    the id never having been written.
    """
    path, _ = round_trip
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sync_session_id"] == _SYNC_ID


def test_loaded_result_does_not_restore_the_sync_session_id(round_trip) -> None:
    """A restored id may be stale or from another machine — the exact #2020 bug.

    ``traigent sync`` would reject it, so the loader deliberately leaves the
    field at its ``None`` default even though the JSON supplies a value.
    """
    _, loader = round_trip
    loaded = loader._optimization_results
    assert loaded is not None
    # The round-trip really happened …
    assert loaded.optimization_id == "opt-2020-round-trip"
    assert len(loaded.trials) == 1
    # … and the id did not come back with it.
    assert loaded.sync_session_id is None


def test_metadata_mirror_does_round_trip_and_is_not_a_sync_target(
    round_trip,
) -> None:
    """The documented asymmetry, pinned.

    ``metadata`` is restored verbatim, so ``metadata["local_session_id"]`` rides
    along and is exactly as stale as a restored field would be. That is why the
    docstring says to read ``sync_session_id`` and never the mirror — this test
    exists so the asymmetry is a recorded contract rather than a surprise.
    """
    _, loader = round_trip
    loaded = loader._optimization_results
    assert loaded.metadata["local_session_id"] == _SYNC_ID
    assert loaded.sync_session_id is None
