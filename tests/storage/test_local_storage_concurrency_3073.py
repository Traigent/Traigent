"""Concurrent mutators and non-finite scores in LocalStorageManager (#3073)."""

from __future__ import annotations

import threading

from traigent.storage.local_storage import LocalStorageManager


def _manager(tmp_path) -> LocalStorageManager:
    return LocalStorageManager(str(tmp_path))


def test_concurrent_add_trial_result_keeps_every_trial(tmp_path) -> None:
    """Two writers are forced to overlap their read step.

    ``load_session`` is wrapped so that each writer waits (briefly) for the
    other after loading. Without the session lock both load the same snapshot
    and one trial is lost; with it the second writer cannot load until the first
    has saved, so the wait times out harmlessly.
    """
    manager = _manager(tmp_path)
    session_id = manager.create_session("fn", {"objectives": ["accuracy"]})
    overlap = threading.Barrier(2, timeout=0.5)
    real_load = manager.load_session

    def load_then_wait(sid):
        session = real_load(sid)
        try:
            overlap.wait()
        except threading.BrokenBarrierError:
            pass
        return session

    manager.load_session = load_then_wait  # type: ignore[method-assign]
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            manager.add_trial_result(session_id, {"i": index}, 0.1 * index)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    session = real_load(session_id)
    assert session is not None
    assert len(session.trials) == 2
    assert sorted(t.trial_id for t in session.trials) == [1, 2]


def test_nan_first_score_does_not_pin_best_score(tmp_path) -> None:
    manager = _manager(tmp_path)
    session_id = manager.create_session("fn", {"objectives": ["accuracy"]})
    manager.add_trial_result(session_id, {"i": 0}, float("nan"))
    manager.add_trial_result(session_id, {"i": 1}, 0.9)
    session = manager.load_session(session_id)
    assert session is not None
    assert session.best_score == 0.9
    assert session.best_config == {"i": 1}


def test_inf_score_is_never_the_best(tmp_path) -> None:
    manager = _manager(tmp_path)
    session_id = manager.create_session("fn", {"objectives": ["accuracy"]})
    manager.add_trial_result(session_id, {"i": 0}, float("inf"))
    session = manager.load_session(session_id)
    assert session is not None
    assert session.best_score is None
