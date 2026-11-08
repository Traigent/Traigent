"""Logger readers must not depend on the platform default encoding (#3260)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_SCRIPT = textwrap.dedent("""
    import locale, sys
    from datetime import UTC, datetime
    from pathlib import Path

    from traigent.api.types import OptimizationResult, OptimizationStatus, TrialResult
    from traigent.api.types import TrialStatus
    from traigent.utils.optimization_logger import OptimizationLogger

    assert locale.getpreferredencoding(False).lower() in ("ascii", "ansi_x3.4-1968"), (
        locale.getpreferredencoding(False)
    )
    base = Path(sys.argv[1])
    log = OptimizationLogger(
        experiment_name="cafe_exp",
        session_id="sess12345678",
        execution_mode="local",
        base_path=base,
    )
    trial = TrialResult(
        trial_id="t1", config={"model": "m"}, metrics={"accuracy": 0.9},
        status=TrialStatus("completed"), duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC), metadata={},
    )
    log.save_checkpoint(
        optimizer_state={"note": "na\\u00efve \\u2713"},
        trials_history=[trial],
        trial_count=1,
    )
    loaded = OptimizationLogger.load_checkpoint(
        "cafe_exp", log.run_id, base_path=base
    )
    assert loaded["optimizer_state"]["note"] == "na\\u00efve \\u2713", loaded
    result = OptimizationResult(
        trials=[trial], best_config={"model": "m"}, best_score=0.9,
        optimization_id="opt", duration=1.0, convergence_info={},
        status=OptimizationStatus.COMPLETED, objectives=["accuracy"],
        algorithm="grid", timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    log.log_session_end(result, error="bo\\u00efm")
    log.log_session_end(result)  # second call re-reads session.json and index.json
    print("OK")
    """)


def test_logger_roundtrips_non_ascii_under_ascii_locale(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.getcwd(),
        "TRAIGENT_MOCK_LLM": "true",
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path / "logs")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-1500:]
