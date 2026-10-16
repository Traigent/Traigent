"""Run a small optimization with canaries in every customer-controlled input."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canaries import CANARIES, Leak, find_leaks
from .harness import CapturedRequest, EgressCapture

FAKE_API_KEY = "uk_" + ("a" * 43)
BACKEND_URL = "https://backend.example.test"

_ENV = {
    "TRAIGENT_API_KEY": FAKE_API_KEY,
    "TRAIGENT_BACKEND_URL": BACKEND_URL,
    "TRAIGENT_MOCK_LLM": "true",
    "MOCK_MODE": "true",
    "TRAIGENT_ENVIRONMENT": "development",
    "TRAIGENT_RUN_APPROVED": "1",
    "TRAIGENT_APPROVED_BY": "egress-canary",
    "TRAIGENT_COST_APPROVED": "true",
    "TRAIGENT_OFFLINE_MODE": "false",
    "TRAIGENT_OFFLINE": "false",
}


@contextmanager
def scenario_env(results_dir: Path, dataset_root: Path) -> Iterator[None]:
    """Connected-mode environment; transports are mocked by EgressCapture."""

    env = dict(_ENV)
    env["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)
    env["TRAIGENT_DATASET_ROOT"] = str(dataset_root)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@dataclass
class ScenarioResult:
    requests: list[CapturedRequest]
    guard_violations: list[str]
    unhandled_routes: list[str]
    optimization_error: str | None
    result_source: str | None
    exerciser_outcomes: dict[str, str | None] = field(default_factory=dict)
    leaks: list[Leak] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)


def write_dataset(dataset_root: Path) -> Path:
    """JSONL dataset whose file stem, ids, text and metadata are canaries."""

    c = CANARIES
    path = dataset_root / f"{c['dataset_file_name']}.jsonl"
    rows = []
    for i in range(2):
        rows.append(
            {
                "input": {
                    "question": f"{c['input_text']} question {i}",
                    "prompt": f"{c['prompt_template']} for row {i}",
                },
                "output": f"{c['output_text']} answer {i}",
                "metadata": {
                    "example_id": f"{c['example_id']}-{i}",
                    c["metadata_key"]: c["metadata_value"],
                },
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def run_scenario(
    workdir: Path, trials: int = 3, direct_paths: bool = True
) -> ScenarioResult:
    """Run the optimization under capture; never raises for SDK-side failures."""

    from traigent import get_config, optimize
    from traigent.evaluators.base import Dataset
    from traigent.api.decorators import EvaluationOptions
    from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

    c = CANARIES
    dataset_root = workdir / "datasets"
    results_dir = workdir / "results"
    dataset_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = write_dataset(dataset_root)

    objectives = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition(name=c["metric_name"], orientation="maximize", weight=1.0)]
    )

    def canary_metric(output: Any, expected: Any = None, **_: Any) -> float:
        return 1.0 if output else 0.0

    calls = {"n": 0}

    capture = EgressCapture()
    error: str | None = None
    source: str | None = None
    with scenario_env(results_dir, dataset_root), capture:
        dataset = Dataset.from_jsonl(str(dataset_path))
        dataset.name = c["evaluation_set"]
        dataset.description = f"canary dataset {c['evaluation_set']}"
        dataset.metadata = {
            **(dataset.metadata or {}),
            c["metadata_key"]: c["metadata_value"],
        }

        @optimize(
            eval_dataset=dataset,
            objectives=objectives,
            configuration_space={
                c["knob_name"]: [f"{c['choice_value']}-a", f"{c['choice_value']}-b"],
                f"{c['knob_name']}-tpl": [
                    c["prompt_template"],
                    f"{c['prompt_template']}-v2",
                ],
            },
            agent_name=c["agent_name"],
            run_title=c["run_title"],
            run_description=c["run_description"],
            evaluation=EvaluationOptions(
                metric_functions={c["metric_name"]: canary_metric}
            ),
            algorithm="auto",
            max_trials=trials,
            offline=False,
            local_storage_path=str(results_dir),
        )
        def canary_app(example: dict[str, Any]) -> str:
            get_config()
            calls["n"] += 1
            if calls["n"] <= 2:  # every example of the first trial fails
                raise ValueError(f"boom {c['exception_message']}")
            return f"{c['model_output']} reply"

        try:
            res = canary_app.optimize_sync(progress_bar=False)
            source = getattr(res, "source", None)
        except Exception as exc:  # noqa: BLE001 - recorded, ratchet still evaluates traffic
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        if direct_paths:
            from .direct_paths import run_exercisers

            outcomes = run_exercisers()
        else:
            outcomes = {}

    reqs = list(capture.requests)
    out = ScenarioResult(
        requests=reqs,
        guard_violations=list(capture.guard_violations),
        unhandled_routes=sorted(set(capture.backend.unhandled)),
        optimization_error=error,
        result_source=source,
        exerciser_outcomes=outcomes,
    )
    out.leaks = find_leaks(reqs)
    out.routes = sorted({r.route for r in reqs})
    return out
