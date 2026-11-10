"""Concurrent loggers must not lose entries in the shared index (#3096)."""

from __future__ import annotations

import json
import threading

from traigent.utils.optimization_logger import OptimizationLogger


def test_concurrent_loggers_keep_every_run_in_index(tmp_path) -> None:
    workers = 8
    loggers = [
        OptimizationLogger(
            experiment_name="exp",
            session_id=f"session_{i}",
            execution_mode="local",
            base_path=tmp_path,
        )
        for i in range(workers)
    ]
    barrier = threading.Barrier(workers)
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
