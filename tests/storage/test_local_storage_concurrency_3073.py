"""Concurrent mutators and non-finite scores in LocalStorageManager (#3073)."""

from __future__ import annotations

import threading

from traigent.storage.local_storage import LocalStorageManager


def _manager(tmp_path) -> LocalStorageManager:
    return LocalStorageManager(str(tmp_path))


def test_concurrent_add_trial_result_keeps_every_trial(tmp_path) -> None:
    manager = _manager(tmp_path)
    session_id = manager.create_session("fn", {"objectives": ["accuracy"]})
    workers = 8
    barrier = threading.Barrier(workers)
    errors: list[BaseException] = []

    def work(index: int) -> None:
        barrier.wait()
        try:
            manager.add_trial_result(session_id, {"i": index}, 0.1 * index)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    session = manager.load_session(session_id)
    assert session is not None
    assert len(session.trials) == workers
    assert session.completed_trials == workers
    assert sorted(t.trial_id for t in session.trials) == list(range(1, workers + 1))


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
