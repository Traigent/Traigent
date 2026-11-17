"""Concurrent loggers must not lose entries in the shared index (#3096)."""

from __future__ import annotations

import json
import threading

from traigent.utils import optimization_logger as optimization_logger_module
from traigent.utils.optimization_logger import OptimizationLogger


def test_concurrent_loggers_keep_every_run_in_index(tmp_path, monkeypatch) -> None:
    """Writers are forced to overlap inside the index read.

    Each index read waits (briefly) for the other writer. Without the lock both
    read the same snapshot and one run is lost; with it the second writer
    cannot read until the first has written, so the wait times out harmlessly.
    """
    workers = 2
    loggers = [
        OptimizationLogger(
            experiment_name="exp",
            session_id=f"session_{i}",
            execution_mode="local",
            base_path=tmp_path,
        )
        for i in range(workers)
    ]
    # An existing index forces every writer through the read step.
    (tmp_path / "index.json").write_text(
        json.dumps({"experiments": {}, "version": "2.0.0"})
    )
    barrier = threading.Barrier(workers)
    overlap = threading.Barrier(workers, timeout=0.5)
    real_load = json.load

    def load_then_wait(handle, *args, **kwargs):
        data = real_load(handle, *args, **kwargs)
        if str(getattr(handle, "name", "")).endswith("index.json"):
            try:
                overlap.wait()
            except threading.BrokenBarrierError:
                pass
        return data

    monkeypatch.setattr(optimization_logger_module.json, "load", load_then_wait)
    errors: list[BaseException] = []

    def work(log: OptimizationLogger) -> None:
        barrier.wait()
        try:
            log._update_experiment_index()
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(log,)) for log in loggers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    index = json.loads((tmp_path / "index.json").read_text())
    run_ids = {run["run_id"] for run in index["experiments"]["exp"]["runs"]}
    assert run_ids == {log.run_id for log in loggers}
