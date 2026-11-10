"""#3129: tunable_params order (and so compute_hash) must not depend on PYTHONHASHSEED.

``_extract_tunable_params`` returned ``list(set(params))``; string hashing is salted
per process, so the same function produced a different payload and dedup hash on
every interpreter start. Runs the extraction in subprocesses with different seeds.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

_SCRIPT = """
import json
from traigent.integrations.observability.workflow_traces import (
    WorkflowGraphPayload, WorkflowNode, _extract_tunable_params,
)

def f(temperature=0, max_tokens=1, top_p=1, model=2):
    pass

params = _extract_tunable_params(f)
node = WorkflowNode(id="n", type="llm", display_name="n", tunable_params=params)
graph = WorkflowGraphPayload(experiment_id="e", nodes=[node], edges=[])
print(json.dumps({"params": params, "hash": graph.compute_hash()}))
"""


def _run(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed, "TRAIGENT_MOCK_LLM": "true"}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_tunable_params_and_hash_stable_across_hash_seeds():
    results = {_run(seed) for seed in ("1", "2", "3")}
    assert len(results) == 1, results


def test_tunable_params_are_sorted_and_unique():
    from traigent.integrations.observability.workflow_traces import (
        _extract_tunable_params,
    )

    class Node:
        config = {}

        def __call__(self, max_tokens=1, temperature=0, model="m"):
            pass

    params = _extract_tunable_params(Node())
    assert params == sorted(set(params))
    assert params == ["max_tokens", "model", "temperature"]
