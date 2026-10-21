"""Regression: example-map / content-feature ids must equal the wire ids.

FE issue Traigent/TraigentFrontend#2550 -- the portal joins the imported
example map to results by exact example id. The exact path keys the map from
the run's own recorded measures (``trials[*].metadata.measures``); expected
hashed ids are fixed literals so a resolver change cannot hide a regression.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.core.metadata_helpers import _build_measures_full
from traigent.evaluators.base import load_dataset_from_file
from traigent.evaluators.local import LocalEvaluator
from traigent.identity.examples import identify_dataset
from traigent.identity.keys import ContentIdentityKeys, set_content_identity_keys
from traigent.reporting.example_map import build_example_content_map

# sha256-derived, hard-coded (dataset name "ds"), indices 0..2.
HASHED_DS = ["ex_319aca3f_0", "ex_accffaef_1", "ex_c121956a_2"]


@pytest.fixture(autouse=True)
def _dataset_root(tmp_path, monkeypatch):
    # Only the run-side loader (used to produce results here) confines datasets.
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))


def _write(tmp_path, rows, name="ds.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _plain_rows():
    return [{"input": {"q": f"q{i}"}, "output": f"a{i}"} for i in range(3)]


def _meta_id_rows():
    rows = _plain_rows()
    rows[1]["metadata"] = {"example_id": "customer-authored-id"}
    return rows


def _session_from(example_results, dataset_name="ds", *, connected=True):
    """A local session as storage would persist it after a run.

    ``connected`` mirrors ``session_operations._persist_connected_session_locally``
    (``metadata.backend_session``); offline sessions reach the portal only via sync.
    """
    measures = _build_measures_full(example_results, "accuracy", dataset_name)
    return {
        "metadata": {"backend_session": True} if connected else {"offline": True},
        "trials": [
            {
                "trial_id": 0,
                "metadata": {
                    "measures": measures,
                    "_traigent_local": {"rows": "full"},
                },
            }
        ],
    }


def _run(path):
    dataset = load_dataset_from_file(str(path))
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    return dataset, asyncio.run(evaluator.evaluate(lambda **kw: "x", {}, dataset))


def _sent(session):
    return [m["example_id"] for m in session["trials"][0]["metadata"]["measures"]]


def _result(example_id, accuracy=1.0):
    return SimpleNamespace(
        example_id=example_id, metrics={"accuracy": accuracy}, success=True
    )


@pytest.fixture
def content_keys():
    keys = ContentIdentityKeys(
        tenant_id="t1",
        kid="k" + "0" * 16,
        example_id_key=bytes([1]) * 32,
        example_version_key=bytes([2]) * 32,
    )
    set_content_identity_keys(keys)
    yield keys
    set_content_identity_keys(None)


def test_fallback_without_artifact_is_sdk_minted_ids(tmp_path):
    path = _write(tmp_path, _plain_rows())
    _, result = _run(path)
    assert _sent(_session_from(result.example_results)) == [
        "example_0",
        "example_1",
        "example_2",
    ]
    assert list(build_example_content_map(path)["example_map"]) == _sent(
        _session_from(result.example_results)
    )


def test_artifact_keyed_run(tmp_path, content_keys):
    path = _write(tmp_path, _plain_rows())
    dataset = load_dataset_from_file(str(path))
    identify_dataset(dataset, content_keys)
    result = asyncio.run(
        LocalEvaluator(metrics=["accuracy"], detailed=True).evaluate(
            lambda **kw: "x", {}, dataset
        )
    )
    assert result.example_results[0].example_id.startswith("ex1")
    session = _session_from(result.example_results)
    assert _sent(session) == HASHED_DS
    # A fresh process has no keys: the artifact, not ambient keys, decides.
    set_content_identity_keys(None)
    payload = build_example_content_map(path, run_session=session)
    assert list(payload["example_map"]) == HASHED_DS


def test_artifact_positional_id_evaluator_with_metadata_ids(tmp_path):
    path = _write(tmp_path, _meta_id_rows())
    # evaluator_wrapper records positional ids even for rows with metadata ids
    session = _session_from([_result(f"example_{i}") for i in range(3)])
    payload = build_example_content_map(path, run_session=session)
    assert list(payload["example_map"]) == ["example_0", "example_1", "example_2"]
    assert "customer-authored-id" not in payload["example_map"]


def test_artifact_custom_evaluator_ids_are_hashed_never_customer_ids(tmp_path):
    path = _write(tmp_path, _plain_rows())
    session = _session_from([_result("example_0"), _result("my-own-id"), _result(2)])
    payload = build_example_content_map(path, run_session=session)
    assert list(payload["example_map"]) == ["example_0", HASHED_DS[1], HASHED_DS[2]]


def test_whitespace_candidate_is_normalized_like_the_wire(tmp_path):
    path = _write(tmp_path, _plain_rows())
    session = _session_from([_result(" example_0 "), _result("example_1")])
    assert _sent(session) == ["example_0", "example_1"]
    payload = build_example_content_map(path, run_session=session)
    assert list(payload["example_map"]) == ["example_0", "example_1"]


def test_artifact_rejects_unrecognised_id_shapes_without_echo(tmp_path):
    path = _write(tmp_path, _plain_rows())
    bad = {
        "trials": [{"metadata": {"measures": [{"example_id": "Alice_has_diabetes"}]}}]
    }
    with pytest.raises(Exception) as exc:
        build_example_content_map(path, run_session=bad)
    assert "Alice_has_diabetes" not in str(exc.value)


def test_map_does_not_touch_dataset_root_env(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAIGENT_DATASET_ROOT", raising=False)
    path = _write(tmp_path, _plain_rows())
    monkeypatch.chdir(tmp_path.parent)  # file is outside the CWD
    before = dict(os.environ)
    payload = build_example_content_map(path)
    assert len(payload["example_map"]) == 3
    assert dict(os.environ) == before
    assert "TRAIGENT_DATASET_ROOT" not in os.environ


def test_cli_with_run_artifact(tmp_path, monkeypatch):
    path = _write(tmp_path, _meta_id_rows())
    session = _session_from([_result("example_0"), _result("x"), _result("example_2")])
    artifact = tmp_path / "session.json"
    artifact.write_text(json.dumps(session), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "report-example-map",
            "--dataset",
            str(path),
            "--output",
            "map.json",
            "--run-artifact",
            str(artifact),
        ],
    )
    assert result.exit_code == 0, result.output
    keys = list(json.loads((tmp_path / "map.json").read_text())["example_map"])
    assert keys == ["example_0", HASHED_DS[1], "example_2"]


def test_cli_without_artifact_notes_the_limit_and_leaks_no_customer_ids(
    tmp_path, monkeypatch
):
    rows = _plain_rows()
    rows[0]["metadata"] = {"example_id": "secret-patient-17"}
    rows[1]["metadata"] = {"example_id": "secret-patient-17"}  # duplicate
    path = _write(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli, ["report-example-map", "--dataset", str(path), "--output", "map.json"]
    )
    assert "secret-patient-17" not in result.output
    assert result.exit_code == 0, result.output
    assert "will not match" in result.output.replace("\n", " ")


@pytest.mark.parametrize(
    ("offline", "parallel"),
    [("true", False), ("false", False), ("true", True), ("false", True)],
)
def test_end_to_end_run_then_report_example_map_by_session_id(
    tmp_path, monkeypatch, offline, parallel
):
    """A real optimize() run persists its sent ids; --session-id keys the map."""
    import traigent
    from traigent.evaluators.base import Dataset, EvaluationExample
    from traigent.storage.local_storage import LocalStorageManager

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", offline)
    monkeypatch.setenv("TRAIGENT_OFFLINE", offline)

    rows = [{"input": {"text": f"case-{i}"}, "output": "ok"} for i in range(3)]
    path = _write(tmp_path, rows, name="e2e_run.jsonl")
    dataset = Dataset(
        [EvaluationExample({"text": f"case-{i}"}, "ok") for i in range(3)],
        name="e2e_run",
    )

    from traigent.core.orchestrator import OptimizationOrchestrator

    parallel_calls = []
    original = OptimizationOrchestrator._run_trial_recording_subset

    async def spy(*args, **kwargs):
        parallel_calls.append(1)
        return await original(*args, **kwargs)

    monkeypatch.setattr(
        OptimizationOrchestrator, "_run_trial_recording_subset", staticmethod(spy)
    )
    extra = {"parallel_config": {"trial_concurrency": 2}} if parallel else {}

    @traigent.optimize(
        eval_dataset=dataset,
        objectives=["accuracy"],
        configuration_space={"x": ["a", "b"]},
        injection_mode="parameter",
        **extra,
    )
    async def agent(text: str, config=None) -> str:
        return "ok"

    asyncio.run(agent.optimize(algorithm="grid"))
    # sanity: the parallel scheduler really ran (or really did not)
    assert bool(parallel_calls) == parallel

    storage = LocalStorageManager(str(tmp_path / "results"))
    sessions = storage.list_sessions()
    assert sessions, "the run persisted no local session"
    session = sessions[-1]
    # every persisted trial -- sequential or parallel -- records explicit rows
    assert session.trials
    assert all(
        (t.metadata or {}).get("_traigent_local", {}).get("rows") == "full"
        for t in session.trials
    )
    persisted = [
        m["example_id"]
        for trial in session.trials
        for m in (trial.metadata or {}).get("measures", [])
    ]
    assert persisted, "no per-example measures were persisted locally"

    out = tmp_path / "map.json"
    result = CliRunner().invoke(
        cli,
        [
            "report-example-map",
            "--dataset",
            str(path),
            "--output",
            str(out),
            "--session-id",
            session.session_id,
        ],
    )
    assert result.exit_code == 0, result.output
    keys = list(json.loads(out.read_text())["example_map"])
    assert keys == sorted(set(persisted), key=persisted.index)
    assert set(keys) <= {"example_0", "example_1", "example_2"}


def test_session_id_not_found_is_a_clean_error(tmp_path, monkeypatch):
    path = _write(tmp_path, _plain_rows())
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    result = CliRunner().invoke(
        cli,
        [
            "report-example-map",
            "--dataset",
            str(path),
            "--output",
            str(tmp_path / "m.json"),
            "--session-id",
            "nope",
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


# --- P2-A / P2-B / P2-C / P3 (round 5) -------------------------------------


def _trial(measures, subset=None):
    rows = subset if subset is not None else "full"
    return {"metadata": {"measures": measures, "_traigent_local": {"rows": rows}}}


def _m(example_id):
    return {"example_id": example_id, "metrics": {"accuracy": 1.0}}


def test_offline_run_then_sync_map_equals_synced_payload_ids(tmp_path, monkeypatch):
    """Offline sessions reach the portal through sync, which rewrites ids."""
    from dataclasses import asdict

    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig
    from traigent.storage.local_storage import LocalStorageManager

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    storage = LocalStorageManager(str(tmp_path / "results"))
    session_id = storage.create_session(
        function_name="f",
        optimization_config={"search_space": {"x": ["a"]}},
        metadata={"execution_mode": "offline", "offline": True},
    )
    results = [_result("example_0"), _result("custom-id"), _result(2)]
    measures = _build_measures_full(results, "accuracy", "ds")
    assert measures[1]["example_id"] == HASHED_DS[1]  # persisted, pre-sync
    storage.add_trial_result(
        session_id=session_id,
        config={"x": "a"},
        score=1.0,
        metadata={"measures": measures, "_traigent_local": {"rows": "full"}},
    )
    session = storage.load_session(session_id)

    sync = SyncManager(TraigentConfig())
    posted = []

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"id": "r1"}

    class _Http:
        def post(self, url, json=None, **kw):  # noqa: A002
            posted.append(json)
            return _Resp()

    monkeypatch.setattr(sync, "_raise_if_backend_egress_disabled", lambda op: None)
    monkeypatch.setattr(
        sync, "_sync_next_trial", lambda sid: {"success": True, "trial_id": "t1"}
    )
    monkeypatch.setattr(sync, "_session", _Http())
    runs = sync._convert_trials_to_configuration_runs(session.trials)
    outcome = sync._sync_session_results("s1", runs)
    assert outcome["success"], outcome
    synced_ids = [m["example_id"] for m in posted[0]["measures"]]
    assert synced_ids == ["example_0", "example_1", "example_2"]

    path = _write(tmp_path, _plain_rows())
    payload = build_example_content_map(path, run_session=asdict(session))
    assert list(payload["example_map"]) == synced_ids


def test_connected_run_map_equals_submitted_request_ids(tmp_path, monkeypatch):
    """Capture the real submitted payload of a connected trial."""
    from dataclasses import asdict
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, Mock

    from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
    from traigent.config.types import TraigentConfig
    from traigent.core.backend_session_manager import BackendSessionManager
    from traigent.optimizers.grid import GridSearchOptimizer
    from traigent.storage.local_storage import LocalStorageManager

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    storage = LocalStorageManager(str(tmp_path / "results"))
    session_id = storage.create_session(
        function_name="f",
        optimization_config={"search_space": {"x": ["a"]}},
        metadata={"execution_mode": "connected", "backend_session": True},
        session_id="be-session-1",
    )

    client = Mock()
    client.get_session_mapping = Mock(return_value=Mock(experiment_run_id="run_1"))
    client.request_trial_slot = AsyncMock(return_value="be_trial_1")
    client._submit_trial_result_via_session = AsyncMock(return_value=True)
    client.submit_result = Mock(
        side_effect=lambda session_id, config, score, metadata=None: (
            storage.add_trial_result(
                session_id=session_id,
                config=config,
                score=score,
                metadata=metadata or {},
            )
        )
    )
    client.auth_manager = Mock(has_api_key=Mock(return_value=True))
    manager = BackendSessionManager(
        backend_client=client,
        traigent_config=TraigentConfig(),
        objectives=["accuracy"],
        objective_schema=None,
        optimizer=GridSearchOptimizer({"x": ["a"]}, ["accuracy"]),
        optimization_id="opt-1",
        optimization_status=OptimizationStatus.RUNNING,
    )
    trial = TrialResult(
        trial_id="t0",
        config={"x": "a"},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
        metadata={
            "example_results": [_result("example_0"), _result("mine"), _result(2)],
            "_traigent_local": {"subset_indices": [0, 1, 2]},
        },
    )
    trial._traigent_row_indices = [0, 1, 2]
    asyncio.run(manager.submit_trial(trial, session_id, dataset_name="ds"))

    kwargs = client._submit_trial_result_via_session.call_args.kwargs
    sent = [m["example_id"] for m in kwargs["metrics"]["measures"]]
    assert sent == ["example_0", HASHED_DS[1], HASHED_DS[2]]
    assert "_traigent_local" not in kwargs["metadata"]  # local record only
    stored_local = (
        storage.load_session(session_id).trials[0].metadata["_traigent_local"]
    )
    assert stored_local["rows"] == [0, 1, 2]  # a subset trial records its list

    path = _write(tmp_path, _plain_rows())
    stored = asdict(storage.load_session(session_id))
    payload = build_example_content_map(path, run_session=stored)
    assert list(payload["example_map"]) == sent


def test_reordered_subset_maps_positions_to_dataset_rows(tmp_path):
    path = _write(tmp_path, _plain_rows())
    session = {
        "metadata": {"backend_session": True},
        "trials": [_trial([_m("example_0"), _m("example_1")], subset=[2, 0])],
    }
    payload = build_example_content_map(path, run_session=session)["example_map"]
    assert payload["example_0"]["expected_output"] == "a2"
    assert payload["example_0"]["example_num"] == 3
    assert payload["example_1"]["expected_output"] == "a0"


def test_subset_conflict_between_trials_raises(tmp_path):
    path = _write(tmp_path, _plain_rows())
    session = {
        "metadata": {"backend_session": True},
        "trials": [
            _trial([_m("example_0")], subset=[2]),
            _trial([_m("example_0")], subset=[1]),
        ],
    }
    with pytest.raises(Exception, match="different dataset rows"):
        build_example_content_map(path, run_session=session)


def test_subset_position_outside_subset_raises(tmp_path):
    path = _write(tmp_path, _plain_rows())
    session = {
        "metadata": {"backend_session": True},
        "trials": [_trial([_m("example_1")], subset=[2])],
    }
    with pytest.raises(Exception, match="outside its trial subset"):
        build_example_content_map(path, run_session=session)


def test_out_of_range_row_raises(tmp_path):
    path = _write(tmp_path, _plain_rows())
    session = {
        "metadata": {"backend_session": True},
        "trials": [_trial([_m("example_9")])],
    }
    with pytest.raises(Exception, match="beyond the end"):
        build_example_content_map(path, run_session=session)


def test_rows_the_run_never_reported_are_omitted(tmp_path):
    path = _write(tmp_path, _plain_rows())
    session = {
        "metadata": {"backend_session": True},
        "trials": [_trial([_m("example_0"), _m("example_2")])],
    }
    keys = list(build_example_content_map(path, run_session=session)["example_map"])
    assert keys == ["example_0", "example_2"]


@pytest.mark.parametrize(
    "session",
    [
        {"trials": "nope"},
        {"trials": ["not-a-mapping"]},
        {"trials": [{"metadata": "x"}]},
        {"trials": [{"metadata": {"measures": "x"}}]},
        {"trials": [{"metadata": {"measures": [{"example_id": 5}]}}]},
        {
            "trials": [
                {
                    "metadata": {
                        "measures": [{"example_id": "example_0"}],
                        "_traigent_local": {"rows": "x"},
                    }
                }
            ]
        },
        {"metadata": "x", "trials": []},
        {"trials": [_trial([_m("example_0")]), "SECRET-VALUE"]},
    ],
)
def test_malformed_artifacts_are_rejected_without_echo(tmp_path, session):
    path = _write(tmp_path, _plain_rows())
    with pytest.raises(Exception) as exc:
        build_example_content_map(path, run_session=session)
    assert "SECRET-VALUE" not in str(exc.value)
    assert "Malformed" in str(exc.value)


@pytest.mark.parametrize(
    ("requested", "rows"),
    [
        # origin/develop traigent/core/orchestrator.py:2661-2666 -- the filter is
        # `0 <= j < len(examples)`, then `dataset.examples[j]`; any exception
        # falls back to the whole dataset (:2676-2680). Reasoned per value:
        ([2, 9, -1, 0], [2, 0]),  # 9 and -1 fail the range filter; order kept
        ([True], [1]),  # 0 <= True < 3 and examples[True] is row 1
        ([0.5], None),  # passes the filter; examples[0.5] raises TypeError
        ([1.0], None),  # passes the filter; examples[1.0] raises TypeError
        (["1"], None),  # `0 <= "1"` raises TypeError
        ([-1], []),  # filtered out; an empty subset dataset, no exception
        ([], []),  # `[] or []`: an empty subset dataset, no exception
    ],
)
def test_effective_subset_reproduces_develop_semantics(requested, rows):
    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.core.orchestrator_helpers import subset_indices_of_dataset
    from traigent.evaluators.base import Dataset, EvaluationExample

    dataset = Dataset(
        [EvaluationExample({"t": str(i)}, "ok") for i in range(3)], name="d"
    )
    cfg = {"x": "a", "__subset_indices__": requested}
    (descriptor,) = OptimizationOrchestrator._build_trial_descriptors(
        None, [cfg], dataset
    )
    ds_for_trial = descriptor[2]
    if rows is None:
        assert ds_for_trial is dataset  # develop falls back to the full dataset
        assert subset_indices_of_dataset(ds_for_trial) is None  # no provenance
    else:
        assert [e.input_data for e in ds_for_trial.examples] == [
            {"t": str(r)} for r in rows
        ]
        assert subset_indices_of_dataset(ds_for_trial) == rows
        assert all(type(r) is int for r in rows)


def test_subset_recording_never_touches_the_trial_config_or_its_hash():
    """The config run_trial/callbacks/tell see, and the client trial id hashed
    from it, are exactly what they were before this change."""
    import copy

    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.evaluators.base import Dataset, EvaluationExample
    from traigent.utils.hashing import generate_trial_hash

    dataset = Dataset(
        [EvaluationExample({"t": str(i)}, "ok") for i in range(3)], name="d"
    )
    cfg = {"x": "a", "__subset_indices__": [2, 0], "_optuna_trial_id": 4}
    pristine = copy.deepcopy(cfg)
    (descriptor,) = OptimizationOrchestrator._build_trial_descriptors(
        None, [cfg], dataset
    )
    assert descriptor[0] is cfg and cfg == pristine  # same object, no new keys
    assert generate_trial_hash(
        session_id="s", config=cfg, dataset_name="d"
    ) == generate_trial_hash(session_id="s", config=pristine, dataset_name="d")


def _trial_result_factory():
    from datetime import UTC, datetime

    from traigent.api.types import TrialResult, TrialStatus

    async def trial(*args, **kwargs):
        return TrialResult(
            trial_id="t",
            config={"x": "a"},
            metrics={},
            status=TrialStatus.COMPLETED,
            duration=0.0,
            timestamp=datetime.now(UTC),
        )

    return trial


def test_run_wrapper_stamps_subset_on_the_result_only():
    from traigent.core.orchestrator import OptimizationOrchestrator

    trial = _trial_result_factory()
    run = OptimizationOrchestrator._run_trial_recording_subset
    result = asyncio.run(run(trial, [2, 0], "func", 1, key="v"))
    assert result.metadata["_traigent_local"] == {"subset_indices": [2, 0]}
    assert "subset_indices" not in result.metadata  # no bare, collidable key
    assert result.config == {"x": "a"}
    result = asyncio.run(run(trial, None))
    assert "_traigent_local" not in result.metadata


def test_run_wrapper_closed_before_start_leaves_no_unawaited_coroutine():
    import gc
    import warnings

    from traigent.core.orchestrator import OptimizationOrchestrator

    calls = []

    async def trial(*args, **kwargs):  # pragma: no cover - must never be created
        calls.append(1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        wrapper = OptimizationOrchestrator._run_trial_recording_subset(trial, [0], "a")
        wrapper.close()  # permit denied / cancelled before it ran
        del wrapper
        gc.collect()
    assert calls == []


def _sync_posted_ids(tmp_path, monkeypatch, session):
    """Run the real sync conversion + upload over ``session``; return posted ids."""
    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig

    sync = SyncManager(TraigentConfig())
    posted = []

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"id": "r1"}

    class _Http:
        def post(self, url, json=None, **kw):  # noqa: A002
            posted.append(json)
            return _Resp()

    monkeypatch.setattr(sync, "_raise_if_backend_egress_disabled", lambda op: None)
    monkeypatch.setattr(
        sync, "_sync_next_trial", lambda sid: {"success": True, "trial_id": "t1"}
    )
    monkeypatch.setattr(sync, "_session", _Http())
    runs = sync._convert_trials_to_configuration_runs(session.trials)
    assert sync._sync_session_results("s1", runs)["success"]
    return [[m["example_id"] for m in p.get("measures", [])] for p in posted]


def _connected_session(tmp_path, monkeypatch):
    """A connected session with two trials that carry DIFFERENT measures."""
    from traigent.storage.local_storage import LocalStorageManager

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    storage = LocalStorageManager(str(tmp_path / "results"))
    session_id = storage.create_session(
        function_name="f",
        optimization_config={"search_space": {"x": ["a"]}},
        metadata={"execution_mode": "connected", "backend_session": True},
        session_id="be-session-2",
    )
    trial_results = [
        [_result("example_0"), _result("custom"), _result(2)],
        [_result("custom"), _result("example_1"), _result("example_2")],
    ]
    persisted = []
    for results in trial_results:
        measures = _build_measures_full(results, "accuracy", "ds")
        persisted.append([m["example_id"] for m in measures])
        storage.add_trial_result(
            session_id=session_id,
            config={"x": "a"},
            score=1.0,
            metadata={"measures": measures, "_traigent_local": {"rows": "full"}},
        )
    assert persisted == [
        ["example_0", HASHED_DS[1], HASHED_DS[2]],
        [HASHED_DS[0], "example_1", "example_2"],
    ]
    return storage.load_session(session_id), persisted


def _run_cli(tmp_path, monkeypatch, session, *extra):
    from dataclasses import asdict

    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(asdict(session), default=str), encoding="utf-8")
    path = _write(tmp_path, _plain_rows())
    result = CliRunner().invoke(
        cli,
        [
            "report-example-map",
            "--dataset",
            str(path),
            "--output",
            "map.json",
            "--run-artifact",
            str(artifact),
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    keys = set(json.loads((tmp_path / "map.json").read_text())["example_map"])
    return keys, result.output


def test_target_synced_equals_posted_sync_ids_for_a_connected_session(
    tmp_path, monkeypatch
):
    session, _ = _connected_session(tmp_path, monkeypatch)
    posted = _sync_posted_ids(tmp_path, monkeypatch, session)
    assert posted == [["example_0", "example_1", "example_2"]] * 2
    keys, out = _run_cli(tmp_path, monkeypatch, session, "--target", "synced")
    assert keys == set(posted[0])
    assert "Target: synced" in out and "--target connected" in out


def test_target_connected_equals_the_persisted_submitted_ids(tmp_path, monkeypatch):
    session, persisted = _connected_session(tmp_path, monkeypatch)
    keys, out = _run_cli(tmp_path, monkeypatch, session, "--target", "connected")
    assert keys == {i for ids in persisted for i in ids}
    assert "Target: connected" in out and "--target synced" in out


def test_default_target_connected_for_connected_session_synced_for_offline(
    tmp_path, monkeypatch
):
    session, persisted = _connected_session(tmp_path, monkeypatch)
    keys, out = _run_cli(tmp_path, monkeypatch, session)  # backend_session true
    assert "Target: connected" in out
    assert keys == {i for ids in persisted for i in ids}

    session.metadata = {"execution_mode": "offline", "offline": True}
    keys, out = _run_cli(tmp_path, monkeypatch, session)
    assert "Target: synced" in out
    assert keys == {"example_0", "example_1", "example_2"}


def test_legacy_artifact_without_row_provenance_is_refused_unless_assumed(
    tmp_path, monkeypatch
):
    from dataclasses import asdict

    session, persisted = _connected_session(tmp_path, monkeypatch)
    legacy = asdict(session)
    for trial in legacy["trials"]:
        trial["metadata"].pop("_traigent_local")  # recorded before row provenance
    path = _write(tmp_path, _plain_rows())
    with pytest.raises(Exception, match="--assume-full-dataset") as exc:
        build_example_content_map(path, run_session=legacy)
    assert "example_" not in str(exc.value)

    payload = build_example_content_map(
        path, run_session=legacy, assume_full_dataset=True
    )
    assert set(payload["example_map"]) == {i for ids in persisted for i in ids}

    # and through the CLI
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "legacy.json"
    artifact.write_text(json.dumps(legacy, default=str), encoding="utf-8")
    base = [
        "report-example-map",
        "--dataset",
        str(path),
        "--output",
        "legacy_map.json",
        "--run-artifact",
        str(artifact),
    ]
    refused = CliRunner().invoke(cli, base)
    assert refused.exit_code != 0 and "--assume-full-dataset" in refused.output
    ok = CliRunner().invoke(cli, [*base, "--assume-full-dataset"])
    assert ok.exit_code == 0, ok.output


def test_invalid_target_is_rejected(tmp_path):
    path = _write(tmp_path, _plain_rows())
    with pytest.raises(Exception, match="target must be one of"):
        build_example_content_map(
            path, run_session=_session_from([_result("example_0")]), target="both"
        )


def test_privacy_mode_connected_run_refuses_connected_map(tmp_path, monkeypatch):
    """Through the connected transport: privacy runs post no per-example ids."""
    from dataclasses import asdict
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, Mock

    from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
    from traigent.config.types import TraigentConfig
    from traigent.core.backend_session_manager import BackendSessionManager
    from traigent.optimizers.grid import GridSearchOptimizer
    from traigent.storage.local_storage import LocalStorageManager

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    storage = LocalStorageManager(str(tmp_path / "results"))
    session_id = storage.create_session(
        function_name="f",
        optimization_config={"search_space": {"x": ["a"]}},
        metadata={"execution_mode": "connected", "backend_session": True},
        session_id="be-session-p",
    )
    client = Mock()
    client.get_session_mapping = Mock(return_value=Mock(experiment_run_id="run_1"))
    client.request_trial_slot = AsyncMock(return_value="be_trial_1")
    client._submit_trial_result_via_session = AsyncMock(return_value=True)
    client.submit_result = Mock(
        side_effect=lambda session_id, config, score, metadata=None: (
            storage.add_trial_result(
                session_id=session_id,
                config=config,
                score=score,
                metadata=metadata or {},
            )
        )
    )
    client.auth_manager = Mock(has_api_key=Mock(return_value=True))
    config = TraigentConfig()
    config.privacy_enabled = True
    manager = BackendSessionManager(
        backend_client=client,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=None,
        optimizer=GridSearchOptimizer({"x": ["a"]}, ["accuracy"]),
        optimization_id="opt-1",
        optimization_status=OptimizationStatus.RUNNING,
    )
    trial = TrialResult(
        trial_id="t0",
        config={"x": "a"},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
        metadata={"example_results": [_result("example_0"), _result("mine")]},
    )
    trial._traigent_row_indices = [0, 1]
    asyncio.run(manager.submit_trial(trial, session_id, dataset_name="ds"))
    # privacy runs never post the per-example measures over the live path
    for call in client._submit_trial_result_via_session.call_args_list:
        assert "measures" not in call.kwargs["metrics"]
        assert "_traigent_local" not in call.kwargs["metadata"]

    stored = asdict(storage.load_session(session_id))
    assert stored["trials"][0]["metadata"]["_traigent_local"]["privacy_mode"] is True
    path = _write(tmp_path, _plain_rows())
    with pytest.raises(Exception, match="privacy mode") as exc:
        build_example_content_map(path, run_session=stored, target="connected")
    assert "mine" not in str(exc.value)
    # the synced map still works (only if it was later synced)
    synced = build_example_content_map(path, run_session=stored, target="synced")
    assert list(synced["example_map"]) == ["example_0", "example_1"]


def test_synced_map_omits_trials_sync_skips_for_empty_config(tmp_path, monkeypatch):
    from dataclasses import asdict

    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig
    from traigent.storage.local_storage import LocalStorageManager

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    storage = LocalStorageManager(str(tmp_path / "results"))
    session_id = storage.create_session(
        function_name="f",
        optimization_config={"search_space": {"x": ["a"]}},
        metadata={"execution_mode": "offline", "offline": True},
    )
    kept = _build_measures_full([_result("example_0")], "accuracy", "ds")
    skipped = _build_measures_full(
        [_result("example_0"), _result("example_1"), _result("example_2")],
        "accuracy",
        "ds",
    )
    storage.add_trial_result(
        session_id,
        {"x": "a"},
        1.0,
        metadata={"measures": kept, "_traigent_local": {"rows": "full"}},
    )
    storage.add_trial_result(
        session_id,
        {},
        1.0,
        metadata={"measures": skipped, "_traigent_local": {"rows": "full"}},
    )
    session = storage.load_session(session_id)

    sync = SyncManager(TraigentConfig())
    posted = []

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"id": "r"}

    class _Http:
        def post(self, url, json=None, **kw):  # noqa: A002
            posted.append(json)
            return _Resp()

    monkeypatch.setattr(sync, "_raise_if_backend_egress_disabled", lambda op: None)
    monkeypatch.setattr(
        sync, "_sync_next_trial", lambda sid: {"success": True, "trial_id": "t"}
    )
    monkeypatch.setattr(sync, "_session", _Http())
    runs = sync._convert_trials_to_configuration_runs(session.trials)
    outcome = sync._sync_session_results("s1", runs)
    assert not outcome["success"]  # the empty-config trial was skipped
    synced_ids = [m["example_id"] for p in posted for m in p.get("measures", [])]
    assert synced_ids == ["example_0"]
    payload = build_example_content_map(
        _write(tmp_path, _plain_rows()), run_session=asdict(session)
    )
    assert list(payload["example_map"]) == synced_ids


def test_marker_never_collides_with_a_flattened_custom_metric(tmp_path, monkeypatch):
    """minimal_logging flattens metrics into trial metadata: a metric named
    ``subset_indices`` must survive, and sync must see exactly what it sees on
    develop (no marker at all)."""
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, Mock

    from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig
    from traigent.core.backend_session_manager import BackendSessionManager
    from traigent.optimizers.grid import GridSearchOptimizer
    from traigent.storage.local_storage import LocalStorageManager

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")

    def run(tag, local_marker):
        root = tmp_path / tag
        monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(root))
        storage = LocalStorageManager(str(root))
        sid = storage.create_session(
            function_name="f",
            optimization_config={"search_space": {"x": ["a"]}},
            metadata={"execution_mode": "connected", "backend_session": True},
            session_id="be-" + tag,
        )
        client = Mock()
        client.get_session_mapping = Mock(return_value=Mock(experiment_run_id="r"))
        client.request_trial_slot = AsyncMock(return_value="be_trial")
        client._submit_trial_result_via_session = AsyncMock(return_value=True)
        client.submit_result = Mock(
            side_effect=lambda session_id, config, score, metadata=None: (
                storage.add_trial_result(
                    session_id=session_id,
                    config=config,
                    score=score,
                    metadata=metadata or {},
                )
            )
        )
        client.auth_manager = Mock(has_api_key=Mock(return_value=True))
        config = TraigentConfig(minimal_logging=True)
        manager = BackendSessionManager(
            backend_client=client,
            traigent_config=config,
            objectives=["accuracy"],
            objective_schema=None,
            optimizer=GridSearchOptimizer({"x": ["a"]}, ["accuracy"]),
            optimization_id="opt",
            optimization_status=OptimizationStatus.RUNNING,
        )
        metadata = {"example_results": [_result("example_0"), _result("example_1")]}
        if local_marker:
            metadata["_traigent_local"] = {"subset_indices": [1, 0]}
        trial = TrialResult(
            trial_id="t0",
            config={"x": "a"},
            metrics={"accuracy": 1.0, "subset_indices": 0.7, "privacy_mode": 0.2},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
            metadata=metadata,
        )
        trial._traigent_row_indices = [0, 1]
        asyncio.run(manager.submit_trial(trial, sid, dataset_name="ds"))
        return storage.load_session(sid)

    with_marker = run("with", True)
    without_marker = run("without", False)
    for trial in without_marker.trials:  # what develop persists: no marker at all
        trial.metadata.pop("_traigent_local", None)
    stored = with_marker.trials[0].metadata
    assert stored["subset_indices"] == 0.7  # the customer metric is intact
    assert stored["privacy_mode"] == 0.2
    assert stored["_traigent_local"] == {"rows": [1, 0]}

    sync = SyncManager(TraigentConfig())
    a = sync.convert_session_to_traigent_format(with_marker)
    b = sync.convert_session_to_traigent_format(without_marker)
    runs_a = a["configuration_runs"][0]
    runs_b = b["configuration_runs"][0]
    assert runs_a["measures"]["subset_indices"] == 0.7
    assert runs_a["measures"] == runs_b["measures"]
    assert runs_a.get("example_measures") == runs_b.get("example_measures")
    assert "_traigent_local" not in json.dumps(a)
    assert SyncManager._compute_payload_hash(a) == SyncManager._compute_payload_hash(b)


@pytest.mark.parametrize("target", ["connected", "synced"])
def test_metric_named_like_the_reserved_key_makes_the_map_refuse(tmp_path, target):
    path = _write(tmp_path, _plain_rows())
    session = {
        "metadata": {"backend_session": True},
        "trials": [
            {
                "config": {"x": "a"},
                "metadata": {
                    "measures": [_m("example_0")],
                    "_traigent_local": 0.7,  # a customer metric hid the marker
                },
            }
        ],
    }
    with pytest.raises(Exception, match="metric named _traigent_local") as exc:
        build_example_content_map(path, run_session=session, target=target)
    assert "0.7" not in str(exc.value)


def _submit_trial_through_manager(
    tmp_path, monkeypatch, status, results, row_indices=None
):
    """Persist one trial through the real manager; return the stored session dict."""
    from dataclasses import asdict
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, Mock

    from traigent.api.types import OptimizationStatus, TrialResult
    from traigent.config.types import TraigentConfig
    from traigent.core.backend_session_manager import BackendSessionManager
    from traigent.optimizers.grid import GridSearchOptimizer
    from traigent.storage.local_storage import LocalStorageManager

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    storage = LocalStorageManager(str(tmp_path / "results"))
    sid = storage.create_session(
        function_name="f",
        optimization_config={"search_space": {"x": ["a"]}},
        metadata={"execution_mode": "connected", "backend_session": True},
        session_id="be-pruned",
    )
    client = Mock()
    client.get_session_mapping = Mock(return_value=Mock(experiment_run_id="r"))
    client.request_trial_slot = AsyncMock(return_value="be_trial")
    client._submit_trial_result_via_session = AsyncMock(return_value=True)
    client.submit_result = Mock(
        side_effect=lambda session_id, config, score, metadata=None: (
            storage.add_trial_result(
                session_id=session_id,
                config=config,
                score=score,
                metadata=metadata or {},
            )
        )
    )
    client.auth_manager = Mock(has_api_key=Mock(return_value=True))
    manager = BackendSessionManager(
        backend_client=client,
        traigent_config=TraigentConfig(),
        objectives=["accuracy"],
        objective_schema=None,
        optimizer=GridSearchOptimizer({"x": ["a"]}, ["accuracy"]),
        optimization_id="opt",
        optimization_status=OptimizationStatus.RUNNING,
    )
    trial_metadata = {"example_results": results}
    trial = TrialResult(
        trial_id="t0",
        config={"x": "a"},
        metrics={"accuracy": 1.0},
        status=status,
        duration=0.1,
        timestamp=datetime.now(UTC),
        metadata=trial_metadata,
    )
    if row_indices is not None:
        trial._traigent_row_indices = row_indices
    asyncio.run(manager.submit_trial(trial, sid, dataset_name="ds"))
    return asdict(storage.load_session(sid))


def _sent_wire(stored):
    return [m["example_id"] for m in stored["trials"][0]["metadata"]["measures"]]


def _rows_after_persist(tmp_path, monkeypatch, status, results, row_indices=None):
    stored = _submit_trial_through_manager(
        tmp_path, monkeypatch, status, results, row_indices
    )
    return stored["trials"][0]["metadata"]["_traigent_local"]["rows"], stored


def test_pruned_concurrent_trial_with_real_row_indices(tmp_path, monkeypatch):
    """Rows [A, B] with 2 workers; B finishes first and pruning cancels A. The
    partial list is [B] at position 0; the evaluator stamped B's true row, 1."""
    from traigent.api.types import TrialStatus

    rows, stored = _rows_after_persist(
        tmp_path, monkeypatch, TrialStatus.PRUNED, [_result("example_1")], [1]
    )
    assert rows == [1]
    assert _sent_wire(stored) == [HASHED_DS[0]]
    payload = build_example_content_map(
        _write(tmp_path, _plain_rows()), run_session=stored
    )["example_map"]
    assert list(payload) == [HASHED_DS[0]]
    assert payload[HASHED_DS[0]]["expected_output"] == "a1"
    assert payload[HASHED_DS[0]]["example_num"] == 2


def test_customer_ids_spelled_like_rows_are_not_row_provenance(tmp_path, monkeypatch):
    """sol's scenario: customer ids ["example_1", "example_0"] on a completed run.
    The evaluator stamped rows [0, 1]; the spelling of the ids proves nothing."""
    from traigent.api.types import TrialStatus

    rows, stored = _rows_after_persist(
        tmp_path,
        monkeypatch,
        TrialStatus.COMPLETED,
        [_result("example_1"), _result("example_0")],
        [0, 1],
    )
    assert rows == "full"
    payload = build_example_content_map(
        _write(tmp_path, _plain_rows()), run_session=stored
    )["example_map"]
    # positions 0 and 1 carry hashed ids (the candidates are not their own
    # positions); each is keyed to the row at that position: a0 then a1.
    assert [(k, v["expected_output"]) for k, v in payload.items()] == [
        (HASHED_DS[0], "a0"),
        (HASHED_DS[1], "a1"),
    ]


def test_custom_evaluator_results_without_provenance_are_unordered_and_refused(
    tmp_path, monkeypatch
):
    """A custom BaseEvaluator subclass building its own results, returned [B, A]:
    nothing stamped a row, so the trial is "unordered" and the map refuses it."""
    from traigent.api.types import TrialStatus
    from traigent.core.trial_result_factory import _stamp_trial_rows

    results = [_result("whatever-b"), _result("whatever-a")]
    probe = SimpleNamespace()
    _stamp_trial_rows(probe, results)
    assert not hasattr(probe, "_traigent_row_indices")  # no evaluator provenance
    rows, stored = _rows_after_persist(
        tmp_path, monkeypatch, TrialStatus.COMPLETED, results
    )
    assert rows == "unordered"
    path = _write(tmp_path, _plain_rows())
    for target in ("connected", "synced"):
        with pytest.raises(Exception, match="assume-full-dataset") as exc:
            build_example_content_map(path, run_session=stored, target=target)
        assert "whatever" not in str(exc.value)
    payload = build_example_content_map(
        path, run_session=stored, target="connected", assume_full_dataset=True
    )
    assert len(payload["example_map"]) == 2  # positional, on the user's say-so


def test_failed_trial_with_out_of_order_partial_results(tmp_path, monkeypatch):
    from traigent.api.types import TrialStatus

    rows, _ = _rows_after_persist(
        tmp_path,
        monkeypatch,
        TrialStatus.FAILED,
        [_result("example_2"), _result("example_0")],
        [2, 0],
    )
    assert rows == [2, 0]


def test_budget_cut_prefix_records_full_without_the_dataset_length(
    tmp_path, monkeypatch
):
    from traigent.api.types import TrialStatus

    rows, stored = _rows_after_persist(
        tmp_path,
        monkeypatch,
        TrialStatus.COMPLETED,
        [_result("example_0"), _result("example_1")],  # 2 of 3 rows
        [0, 1],
    )
    assert rows == "full"
    payload = build_example_content_map(
        _write(tmp_path, _plain_rows()), run_session=stored
    )["example_map"]
    assert list(payload) == ["example_0", "example_1"]


def test_provenance_length_mismatch_is_unordered(tmp_path, monkeypatch):
    from traigent.api.types import TrialStatus

    rows, _ = _rows_after_persist(
        tmp_path,
        monkeypatch,
        TrialStatus.COMPLETED,
        [_result("example_0"), _result("example_1")],
        [0],
    )
    assert rows == "unordered"


def test_row_provenance_helpers_literals():
    from traigent.utils.example_id import (
        row_indices_of,
        rows_from_indices,
        stamp_row_index,
    )

    a, b = SimpleNamespace(), SimpleNamespace()
    stamp_row_index(a, 2)
    stamp_row_index(b, 0)
    assert row_indices_of([a, b]) == [2, 0]
    assert row_indices_of([a, SimpleNamespace()]) is None  # one unstamped
    assert row_indices_of([a, a]) is None  # duplicate rows
    assert row_indices_of([]) == []
    assert row_indices_of(None) is None
    assert rows_from_indices([0, 1, 2], 3, None) == "full"
    assert rows_from_indices([0, 1], 2, None) == "full"
    assert rows_from_indices([2, 0], 2, None) == [2, 0]
    assert rows_from_indices([0, 1], 2, [5, 7]) == [5, 7]
    assert rows_from_indices([1, 0], 2, [5, 7]) == [7, 5]
    assert rows_from_indices([2], 1, [5, 7]) == "unordered"  # outside the subset
    assert rows_from_indices(None, 1, None) == "unordered"
    assert rows_from_indices([0], 2, None) == "unordered"  # length mismatch


def test_builtin_evaluator_stamps_true_rows_and_the_stamp_never_serializes():
    """Real LocalEvaluator, 2 workers, row 0 slowest: the results carry their
    true rows, and neither to_dict, asdict, the factory's wire payload nor the
    backend metadata contains the stamp."""
    import dataclasses

    from traigent.core.metadata_helpers import build_backend_metadata
    from traigent.core.trial_result_factory import _stamp_trial_rows
    from traigent.evaluators.base import Dataset, EvaluationExample
    from traigent.utils.example_id import ROW_INDEX_ATTR

    dataset = Dataset([EvaluationExample({"i": i}, "ok") for i in range(3)], name="c")

    async def slow_first(i: int) -> str:
        await asyncio.sleep(0.2 if i == 0 else 0.0)
        return "ok"

    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True, max_workers=2)
    result = asyncio.run(evaluator.evaluate(slow_first, {}, dataset))
    assert [getattr(r, ROW_INDEX_ATTR) for r in result.example_results] == [0, 1, 2]

    first = result.example_results[0]
    assert ROW_INDEX_ATTR not in first.to_dict()
    assert ROW_INDEX_ATTR not in dataclasses.asdict(first)
    assert ROW_INDEX_ATTR not in json.dumps(first.to_dict(), default=str)

    from datetime import UTC, datetime

    from traigent.api.types import TrialResult, TrialStatus
    from traigent.config.types import TraigentConfig
    from traigent.core.trial_result_factory import _redact_example_results_with_signals

    metadata: dict = {
        "example_results": _redact_example_results_with_signals(result.example_results)
    }
    trial = TrialResult(
        trial_id="t",
        config={},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
        metadata=metadata,
    )
    _stamp_trial_rows(trial, result.example_results)
    assert trial._traigent_row_indices == [0, 1, 2]
    assert "_traigent_local" not in trial.metadata  # nothing new in metadata
    wire = build_backend_metadata(trial, "accuracy", TraigentConfig(), "ds")
    blob = json.dumps(wire, default=str)
    assert ROW_INDEX_ATTR not in blob
    assert "row_indices" not in blob and "_traigent_local" not in blob
