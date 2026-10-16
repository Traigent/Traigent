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
from .harness import CapturedRequest, EgressCapture, Fault

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
    faults: list[Fault] = field(default_factory=list)
    #: Requests the offline optimize() itself made (must stay empty).
    offline_optimize_requests: int = 0
    offline_error: str | None = None
    offline_synced_sessions: int = 0


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


def _run_offline_optimize_and_sync(
    workdir: Path, capture: EgressCapture, dataset_path: Path
) -> tuple[int, str | None, int]:
    """Offline optimize() (constraints, default_config) then SyncManager replay.

    Returns ``(requests made by optimize, optimize error, sessions replayed)``.
    """

    from traigent import Choices, get_config, optimize
    from traigent.api.constraints import Constraint
    from traigent.api.decorators import EvaluationOptions
    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig
    from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
    from traigent.evaluators.base import Dataset

    c = CANARIES
    off_dir = workdir / "offline"
    results = off_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_jsonl(str(dataset_path))
    dataset.name = c["evaluation_set"]
    knob = Choices([f"{c['choice_value']}-a", f"{c['choice_value']}-b"])
    constraints = [
        Constraint(
            when=knob.equals(f"{c['choice_value']}-a"),
            then=knob.equals(f"{c['choice_value']}-a"),
            id=c["constraint_name"],
            description=c["constraint_name"],
        )
    ]
    objectives = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition(name=c["metric_name"], orientation="maximize", weight=1.0)]
    )

    def canary_metric(output: Any, expected: Any = None, **_: Any) -> float:
        return 1.0 if output else 0.0

    before = len(capture.requests)
    error: str | None = None
    env = {"TRAIGENT_OFFLINE_MODE": "true", "TRAIGENT_OFFLINE": "true"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:

        @optimize(
            eval_dataset=dataset,
            objectives=objectives,
            configuration_space={c["constraint_knob"]: knob},
            default_config={c["constraint_knob"]: f"{c['choice_value']}-a"},
            constraints=constraints,
            agent_name=c["agent_name"],
            run_title=c["run_title"],
            run_description=c["run_description"],
            evaluation=EvaluationOptions(
                metric_functions={c["metric_name"]: canary_metric},
                evaluator_id=c["evaluator_id"],
                dataset_id=c["dataset_identity"],
            ),
            algorithm="grid",
            max_trials=2,
            offline=True,
            local_storage_path=str(results),
        )
        def offline_app(example: dict[str, Any]) -> str:
            get_config()
            return f"{c['model_output']} offline"

        try:
            offline_app.optimize_sync(progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    made = len(capture.requests) - before

    # Connected replay of what the offline run (and a hand-built session) stored.
    from .direct_paths import build_offline_session

    build_offline_session(str(results))
    manager = SyncManager(
        TraigentConfig(local_storage_path=str(results)), api_key=FAKE_API_KEY
    )
    synced = 0
    for session in manager.storage.list_sessions():
        manager.sync_session_to_cloud(session.session_id, force=True)
        synced += 1
    return made, error, synced


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
            default_config={
                c["knob_name"]: c["default_config_value"],
            },
            evaluation=EvaluationOptions(
                metric_functions={c["metric_name"]: canary_metric},
                evaluator_id=c["evaluator_id"],
                dataset_id=c["dataset_identity"],
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

            outcomes = run_exercisers(capture)
        else:
            outcomes = {}
        off_made, off_error, off_synced = 0, None, 0
        if direct_paths:
            try:
                off_made, off_error, off_synced = _run_offline_optimize_and_sync(
                    workdir, capture, dataset_path
                )
            except Exception as exc:  # noqa: BLE001 - recorded, ratchet still evaluates traffic
                off_error = f"{type(exc).__name__}: {str(exc)[:200]}"

    reqs = list(capture.requests)
    out = ScenarioResult(
        requests=reqs,
        guard_violations=list(capture.guard_violations),
        unhandled_routes=sorted(set(capture.backend.unhandled)),
        optimization_error=error,
        result_source=source,
        exerciser_outcomes=outcomes,
        faults=list(capture.backend.fault_log),
        offline_optimize_requests=off_made,
        offline_error=off_error,
        offline_synced_sessions=off_synced,
    )
    out.leaks = find_leaks(reqs)
    out.routes = sorted({r.route for r in reqs})
    return out
