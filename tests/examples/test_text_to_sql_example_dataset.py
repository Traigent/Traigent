import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from traigent.utils.validation import validate_dataset_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "examples/datasets/text-to-sql/evaluation_set.jsonl"
RUN_PATH = PROJECT_ROOT / "examples/core/text-to-sql/run.py"


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_text_to_sql_example_dataset_validates_and_run_module_imports() -> None:
    validate_dataset_path(str(DATASET_PATH))

    rows = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "Expected the shipped text-to-sql dataset to contain rows"
    assert all("input" in row for row in rows)
    assert all("expected" in row for row in rows)
    assert all("question" not in row for row in rows)

    env = os.environ.copy()
    env.pop("PYTEST_CURRENT_TEST", None)
    env["TRAIGENT_MOCK_LLM"] = "true"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, pathlib; "
                f"module_path = pathlib.Path({str(RUN_PATH)!r}); "
                "spec = importlib.util.spec_from_file_location("
                "'text_to_sql_example_run', module_path"
                "); "
                "module = importlib.util.module_from_spec(spec); "
                "assert spec and spec.loader; "
                "spec.loader.exec_module(module)"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, (
        "Importing examples/core/text-to-sql/run.py in mock mode failed.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
