"""Legacy best-config files never bypass the requested identity (sol F1d review, round F1e).

A runtime must never serve a best config approved for another agent, function or
environment, whatever file or source it comes from. The legacy (non-canonical) file
forms - a bare ``config`` / ``best_config`` dict, a slim/full export, the SDK's own
development-log artifact - used to be read with their identity fields discarded.

Ruling (captain, round F1e):

a. a file carrying identity fields that contradict the request is refused, always
   (strict raises; non-strict skips - it is never loaded);
b. a file declaring a ``schema_version`` the SDK does not support is refused, never
   read through the legacy fallback;
c. an identity-free legacy file loads ONLY when the user named it explicitly
   (``load_from`` / ``TRAIGENT_CONFIG_PATH``), with a warning; ``best_config_strict``
   refuses it;
d. dev-log discovery loads only artifacts whose identity matches; the SDK's logger
   writes that identity (qualified ``function_ref``, ``config_id``, ``environment``);
   identity-free dev logs are skipped; two functions sharing a bare name are never
   confused.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from traigent.core.best_config_runtime import BestConfigSource, function_ref_for
from traigent.core.config_state_manager import ConfigStateManager
from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ConfigurationError


def _agent(text: str) -> str:
    return text


_REF = function_ref_for(_agent)


def _other_agent(text: str) -> str:
    return text


# Same bare name, another module: the development-log directory is keyed on the bare
# name, so both functions share it.
_other_agent.__name__ = "_agent"
_other_agent.__qualname__ = "_agent"
_other_agent.__module__ = "other_pkg.agents"
_OTHER_REF = function_ref_for(_other_agent)
assert _OTHER_REF != _REF


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAIGENT_CONFIG_PATH", raising=False)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.chdir(tmp_path)


def _manager(
    *,
    func: Any = _agent,
    environment: str = "prod",
    strict: bool = False,
    load_from: str | None = None,
    dev_logs: bool = False,
) -> ConfigStateManager:
    return ConfigStateManager(
        func=func,
        default_config={"t": 1},
        local_storage_path=None,
        configuration_space={"t": [1, 2]},
        auto_load_best=dev_logs,
        load_from=load_from,
        setup_wrapper_callback=lambda: None,
        config_id="agent",
        best_config_source="off",
        best_config_strict=strict,
        best_config_environment=environment,
        enable_auto_load_dev_logs=dev_logs,
    )


def _legacy(**override: Any) -> dict[str, Any]:
    """A legacy envelope carrying the requested identity (agent / _REF / prod)."""
    data: dict[str, Any] = {
        "config_id": "agent",
        "function_ref": _REF,
        "environment": "prod",
        "config": {"t": 2},
    }
    data.update(override)
    return {key: value for key, value in data.items() if value != "__omit__"}


_IDENTITY_FREE = {"config": {"t": 2}}


def _write(path: Path, data: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _dev_log(tmp_path: Path, data: dict[str, Any], run: str = "r1", name="_agent"):
    return _write(
        tmp_path
        / ".traigent"
        / "optimization_logs"
        / "experiments"
        / name
        / "runs"
        / run
        / "artifacts"
        / "best_config.json",
        data,
    )


def _entry(kind: str, tmp_path: Path, monkeypatch, data: dict[str, Any], **kwargs):
    """Build a manager whose ONLY source is ``data`` reached through ``kind``."""
    if kind == "load_from":
        path = _write(tmp_path / "explicit.json", data)
        return _manager(load_from=path, **kwargs)
    if kind == "env":
        path = _write(tmp_path / "explicit.json", data)
        monkeypatch.setenv("TRAIGENT_CONFIG_PATH", path)
        return _manager(**kwargs)
    assert kind == "dev_log"
    _dev_log(tmp_path, data)
    return _manager(dev_logs=True, **kwargs)


ENTRY_POINTS = ["load_from", "env", "dev_log"]
EXPLICIT_ENTRY_POINTS = ["load_from", "env"]


def _is_default(manager: ConfigStateManager) -> bool:
    snap = manager._best_config_snapshot
    return snap.source == BestConfigSource.DEFAULT.value and snap.config["t"] == 1


# --- ruling a: contradicting identity is refused, always ------------------------------

CONTRADICTIONS = {
    "config_id": {"config_id": "other"},
    "function_ref": {"function_ref": "other:agent"},
    "environment": {"environment": "staging"},
    "all_three": {
        "config_id": "other",
        "function_ref": "other:agent",
        "environment": "staging",
    },
}


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("field", sorted(CONTRADICTIONS))
def test_a_contradicting_identity_is_refused_strict(
    entry, field, tmp_path, monkeypatch
) -> None:
    manager = _entry(
        entry, tmp_path, monkeypatch, _legacy(**CONTRADICTIONS[field]), strict=True
    )
    if entry == "dev_log" and "function_ref" in CONTRADICTIONS[field]:
        # Discovery only ever considers artifacts naming THIS function (ruling d):
        # another function's log - two modules share the bare-name directory - is
        # never a candidate, so it is skipped rather than raised on, and never loaded.
        manager.maybe_auto_load_config()
        assert _is_default(manager)
        return
    with pytest.raises(ConfigurationError, match="identity|mismatch"):
        manager.maybe_auto_load_config()


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("field", sorted(CONTRADICTIONS))
def test_a_contradicting_identity_is_skipped_not_loaded_non_strict(
    entry, field, tmp_path, monkeypatch
) -> None:
    manager = _entry(
        entry, tmp_path, monkeypatch, _legacy(**CONTRADICTIONS[field]), strict=False
    )
    manager.maybe_auto_load_config()
    assert _is_default(manager)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_contradicting_identity_is_refused_even_when_identity_is_partial(
    entry, tmp_path, monkeypatch
) -> None:
    """A file naming only a foreign environment is still foreign."""
    data = {"environment": "staging", "config": {"t": 2}}
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=True)
    if entry == "dev_log":  # no function_ref: not a discovery candidate (ruling d)
        manager.maybe_auto_load_config()
        assert _is_default(manager)
        return
    with pytest.raises(ConfigurationError):
        manager.maybe_auto_load_config()
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=False)
    manager.maybe_auto_load_config()
    assert _is_default(manager)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_non_string_identity_field_is_a_contradiction(
    entry, tmp_path, monkeypatch
) -> None:
    manager = _entry(
        entry, tmp_path, monkeypatch, _legacy(function_ref=["x"]), strict=False
    )
    manager.maybe_auto_load_config()
    assert _is_default(manager)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_matching_identity_legacy_file_loads_even_when_strict(
    entry, tmp_path, monkeypatch
) -> None:
    manager = _entry(entry, tmp_path, monkeypatch, _legacy(), strict=True)
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_matching_identity_is_normalised_like_the_canonical_path(
    entry, tmp_path, monkeypatch
) -> None:
    data = _legacy(config_id=" agent ", function_ref=f" {_REF} ", environment=" PROD ")
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=True)
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_bound_legacy_file_without_environment_is_the_default_environment(
    entry, tmp_path, monkeypatch
) -> None:
    data = _legacy(environment="__omit__")
    prod = _entry(entry, tmp_path, monkeypatch, data, strict=False)
    prod.maybe_auto_load_config()
    assert _is_default(prod)
    default = _entry(
        entry, tmp_path, monkeypatch, data, strict=False, environment="default"
    )
    default.maybe_auto_load_config()
    assert default._best_config_snapshot.config["t"] == 2


# --- ruling b: an unsupported schema_version never reaches the legacy fallback --------


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("version", ["traigent.best_config.v2", "v0", "", 7])
def test_b_an_unsupported_schema_version_is_refused_strict(
    entry, version, tmp_path, monkeypatch
) -> None:
    manager = _entry(
        entry,
        tmp_path,
        monkeypatch,
        _legacy(schema_version=version),
        strict=True,
    )
    with pytest.raises(ConfigurationError, match="schema_version"):
        manager.maybe_auto_load_config()


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_b_an_unsupported_schema_version_is_skipped_non_strict(
    entry, tmp_path, monkeypatch
) -> None:
    manager = _entry(
        entry,
        tmp_path,
        monkeypatch,
        {"schema_version": "traigent.best_config.v2", "config": {"t": 2}},
        strict=False,
    )
    manager.maybe_auto_load_config()
    assert _is_default(manager)


# --- ruling c: identity-free legacy loads only when the user named the file ------------


@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
def test_c_identity_free_legacy_file_loads_when_named_with_a_warning(
    entry, tmp_path, monkeypatch, caplog
) -> None:
    manager = _entry(entry, tmp_path, monkeypatch, _IDENTITY_FREE, strict=False)
    with caplog.at_level(logging.WARNING):
        manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2
    assert any(
        "identity" in r.getMessage() and r.levelno >= logging.WARNING
        for r in caplog.records
    )


@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
@pytest.mark.parametrize(
    "data",
    [
        {"config": {"t": 2}},
        {"best_config": {"t": 2}},
        {"t": 2},
        {"config": {"t": 2}, "function_name": "_agent", "traigent_version": "0"},
    ],
    ids=["config", "best_config", "bare_dict", "slim_export"],
)
def test_c_strict_refuses_an_identity_free_legacy_file(
    entry, data, tmp_path, monkeypatch
) -> None:
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=True)
    with pytest.raises(ConfigurationError, match="identity"):
        manager.maybe_auto_load_config()


@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
def test_c_a_file_naming_only_environment_or_id_cannot_establish_the_function(
    entry, tmp_path, monkeypatch
) -> None:
    """Without a ``function_ref`` the function is unverified: same as identity-free."""
    data = {"config_id": "agent", "environment": "prod", "config": {"t": 2}}
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=True)
    with pytest.raises(ConfigurationError, match="identity"):
        manager.maybe_auto_load_config()
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=False)
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2


def test_c_the_explicit_path_is_the_users_not_a_discovered_one(
    tmp_path, monkeypatch
) -> None:
    """An identity-free file reached by discovery is NOT 'named by the user'."""
    manager = _entry("dev_log", tmp_path, monkeypatch, _IDENTITY_FREE, strict=False)
    manager.maybe_auto_load_config()
    assert _is_default(manager)


# --- ruling d: dev-log discovery -------------------------------------------------------


def test_d_identity_free_dev_logs_are_skipped_when_bound_even_strict(
    tmp_path, monkeypatch
) -> None:
    for strict in (False, True):
        manager = _entry(
            "dev_log", tmp_path, monkeypatch, _IDENTITY_FREE, strict=strict
        )
        manager.maybe_auto_load_config()
        assert _is_default(manager)
        assert manager._find_latest_config_path() is None


def test_d_discovery_never_confuses_two_functions_sharing_a_bare_name(
    tmp_path, monkeypatch
) -> None:
    # An older run of THIS function and a newer run of the other module's function
    # share ``experiments/_agent``.
    older = _dev_log(tmp_path, _legacy(config={"t": 2}), run="r1")
    newer = _dev_log(
        tmp_path,
        _legacy(function_ref=_OTHER_REF, config={"t": 1}),
        run="r2",
    )
    import os

    os.utime(Path(older).parents[1], (1_000, 1_000))
    os.utime(Path(newer).parents[1], (2_000, 2_000))

    mine = _manager(dev_logs=True, strict=True)
    assert mine._find_latest_config_path() == older
    mine.maybe_auto_load_config()
    assert mine._best_config_snapshot.config["t"] == 2

    theirs = _manager(func=_other_agent, dev_logs=True, strict=True)
    assert theirs._find_latest_config_path() == newer


def test_d_a_dev_log_for_only_the_other_function_is_never_served(
    tmp_path, monkeypatch
) -> None:
    _dev_log(tmp_path, _legacy(function_ref=_OTHER_REF))
    for strict in (False, True):
        manager = _manager(dev_logs=True, strict=strict)
        assert manager._find_latest_config_path() is None
        manager.maybe_auto_load_config()
        assert _is_default(manager)


def test_d_discovery_prefers_the_artifact_for_the_requested_environment(
    tmp_path, monkeypatch
) -> None:
    import os

    prod = _dev_log(tmp_path, _legacy(config={"t": 2}), run="r1")
    staging = _dev_log(
        tmp_path, _legacy(environment="staging", config={"t": 1}), run="r2"
    )
    os.utime(Path(prod).parents[1], (1_000, 1_000))
    os.utime(Path(staging).parents[1], (2_000, 2_000))
    manager = _manager(dev_logs=True, strict=True)
    assert manager._find_latest_config_path() == prod
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2


def test_d_a_function_matching_artifact_for_another_environment_is_still_refused(
    tmp_path, monkeypatch
) -> None:
    _dev_log(tmp_path, _legacy(environment="staging"))
    strict = _manager(dev_logs=True, strict=True)
    with pytest.raises(ConfigurationError, match="environment"):
        strict.maybe_auto_load_config()
    loose = _manager(dev_logs=True, strict=False)
    loose.maybe_auto_load_config()
    assert _is_default(loose)


# --- ruling d: the SDK's own logger writes the identity --------------------------------


def _logger(tmp_path: Path, **kwargs: Any):
    from traigent.utils.optimization_logger import OptimizationLogger

    return OptimizationLogger(
        experiment_name="_agent",
        session_id="sess12345678",
        execution_mode="local",
        base_path=tmp_path / "logs",
        **kwargs,
    )


def _result():
    from datetime import UTC, datetime

    from traigent.api.types import (
        OptimizationResult,
        OptimizationStatus,
        TrialResult,
        TrialStatus,
    )

    return OptimizationResult(
        trials=[
            TrialResult(
                trial_id="t1",
                config={"t": 2},
                metrics={"accuracy": 0.9},
                status=TrialStatus("completed"),
                duration=1.0,
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                metadata={},
            )
        ],
        best_config={"t": 2},
        best_score=0.9,
        optimization_id="opt-1",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )


IDENTITY = {"config_id": "agent", "function_ref": _REF, "environment": "prod"}


def _artifact(log) -> dict[str, Any]:
    files = list((log.run_path / "artifacts").glob("best_config_v*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("writer", ["session_end", "log_results"])
def test_d_the_logger_writes_the_best_config_identity(tmp_path, writer) -> None:
    log = _logger(tmp_path, best_config_identity=IDENTITY)
    if writer == "session_end":
        log.log_session_end(_result())
    else:
        log.log_results(_result())
    data = _artifact(log)
    for key, value in IDENTITY.items():
        assert data[key] == value
    assert data["config"] == {"t": 2}


def test_d_the_logger_omits_identity_it_was_not_given(tmp_path) -> None:
    log = _logger(tmp_path)
    log.log_session_end(_result())
    data = _artifact(log)
    assert not {"config_id", "function_ref", "environment"} & set(data)


def test_d_a_logger_artifact_round_trips_through_a_bound_manager(tmp_path) -> None:
    log = _logger(
        tmp_path / "x",
        best_config_identity=IDENTITY,
    )
    log.log_session_end(_result())
    artifact = next((log.run_path / "artifacts").glob("best_config_v*.json"))
    dest = Path(".traigent/optimization_logs/experiments/_agent/runs/r1/artifacts")
    dest.mkdir(parents=True)
    (dest / "best_config.json").write_text(artifact.read_text(encoding="utf-8"))
    manager = _manager(dev_logs=True, strict=True)
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2
    assert manager._best_config_snapshot.source == BestConfigSource.DEV_LOG.value
    # ... and a manager bound to another environment or function refuses / skips it.
    foreign = _manager(environment="staging", dev_logs=True, strict=True)
    with pytest.raises(ConfigurationError):
        foreign.maybe_auto_load_config()
    other = _manager(func=_other_agent, dev_logs=True, strict=True)
    other.maybe_auto_load_config()
    assert _is_default(other)


def test_d_the_orchestrator_hands_the_logger_the_head_identity(
    tmp_path, monkeypatch
) -> None:
    from traigent.config.types import TraigentConfig
    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.evaluators.base import Dataset

    from tests.unit.core.test_orchestrator_run_config_logging import (
        TinyEvaluator,
        TinyOptimizer,
    )

    monkeypatch.setenv("TRAIGENT_OPTIMIZATION_LOG_DIR", str(tmp_path / "ol"))
    orchestrator = OptimizationOrchestrator(
        optimizer=TinyOptimizer({"t": [1, 2]}, ["accuracy"]),
        evaluator=TinyEvaluator(),
        max_trials=1,
        config=TraigentConfig(execution_mode="local"),
    )
    orchestrator.head_config_id = "agent"
    orchestrator.head_function_ref = _REF
    orchestrator.head_environment = "prod"
    orchestrator._initialize_logger("session-1234567890", lambda: None, Dataset([]))
    orchestrator._logger.log_session_end(_result())
    data = _artifact(orchestrator._logger)
    for key, value in IDENTITY.items():
        assert data[key] == value


def test_d_the_optimized_function_wires_its_identity_into_the_orchestrator(
    monkeypatch, tmp_path
) -> None:
    """``optimize()`` hands the orchestrator the function's identity for the logger."""
    import asyncio

    import traigent.core.optimized_function as optimized_function_module
    from traigent.core.optimized_function import OptimizedFunction
    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.evaluators.base import Dataset, EvaluationExample

    seen: dict[str, Any] = {}

    class RecordingOrchestrator(OptimizationOrchestrator):
        async def optimize(self, func, dataset, function_name=None):  # type: ignore[override]
            seen["identity"] = self._best_config_identity()
            return _result()

    monkeypatch.setattr(
        optimized_function_module, "OptimizationOrchestrator", RecordingOrchestrator
    )
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    wrapped = OptimizedFunction(
        func=_agent,
        configuration_space={"t": [1, 2]},
        objectives=["accuracy"],
        eval_dataset=Dataset(
            [EvaluationExample({"text": "a"}, "a")], name="d", description="d"
        ),
        config_id="agent",
        best_config_environment="prod",
        max_trials=1,
    )
    asyncio.run(wrapped.optimize(max_trials=1))
    assert seen["identity"] == IDENTITY


# --- the explicit, value-in-hand APIs stay unbound, and say so --------------------------


@pytest.mark.parametrize(
    "target",
    [
        ConfigStateManager.apply_best_config,
        ConfigStateManager.load_optimization_results,
        ConfigStateManager.set_config,
        OptimizedFunction.apply_best_config,
        OptimizedFunction.load_optimization_results,
        OptimizedFunction.set_config,
    ],
)
def test_explicit_value_apis_document_that_they_do_not_verify_stored_identity(
    target,
) -> None:
    assert "does not verify" in (target.__doc__ or "")
    assert "identity" in (target.__doc__ or "")


# --- round F1f -------------------------------------------------------------------------
# sol F1e review (bare dictionaries) and terra SDK review (schema_version knob, paths).


# F1f item 1: a bare dictionary is not exempt from the contradiction rule (ruling a).
@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
@pytest.mark.parametrize("field", sorted(CONTRADICTIONS))
@pytest.mark.parametrize("strict", [False, True])
def test_f1f_a_bare_dict_with_contradicting_identity_is_refused(
    entry, field, strict, tmp_path, monkeypatch
) -> None:
    data = {**CONTRADICTIONS[field], "t": 2}
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=strict)
    if strict:
        with pytest.raises(ConfigurationError, match="identity mismatch"):
            manager.maybe_auto_load_config()
    else:
        manager.maybe_auto_load_config()
    assert _is_default(manager)


def test_f1f_sols_exact_bare_dict_does_not_install_t2(tmp_path, monkeypatch) -> None:
    data = {
        "config_id": "other",
        "function_ref": "other:agent",
        "environment": "staging",
        "t": 2,
    }
    for entry in EXPLICIT_ENTRY_POINTS:
        manager = _entry(entry, tmp_path, monkeypatch, data, strict=False)
        manager.maybe_auto_load_config()
        assert manager._best_config_snapshot.config["t"] == 1


@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
def test_f1f_a_bare_dict_that_names_the_requested_identity_still_loads(
    entry, tmp_path, monkeypatch
) -> None:
    data = {
        "config_id": "agent",
        "function_ref": _REF,
        "environment": "prod",
        "t": 2,
    }
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=True)
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2


def test_f1f_the_unbound_low_level_read_still_returns_the_bare_dict(tmp_path) -> None:
    path = _write(tmp_path / "bare.json", {"function_ref": "other:agent", "t": 2})
    assert ConfigStateManager._load_config_from_path(path)["t"] == 2


# F1f item 4: only a canonical envelope is refused for its schema_version.
@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
@pytest.mark.parametrize("version", ["2026-10", "v3", 7, "", None])
def test_f1f_a_schema_version_knob_in_a_bare_config_is_an_ordinary_key(
    entry, version, tmp_path, monkeypatch, caplog
) -> None:
    data = {"schema_version": version, "t": 2}
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=False)
    with caplog.at_level(logging.WARNING):
        manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2
    assert manager._best_config_snapshot.config["schema_version"] == version
    assert not any("schema_version" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
def test_f1f_strict_refuses_a_schema_version_knob_file_as_identity_free_not_as_a_version(
    entry, tmp_path, monkeypatch
) -> None:
    manager = _entry(
        entry,
        tmp_path,
        monkeypatch,
        {"schema_version": "2026-10", "t": 2},
        strict=True,
    )
    with pytest.raises(ConfigurationError) as caught:
        manager.maybe_auto_load_config()
    assert "identity" in str(caught.value)
    assert "schema_version" not in str(caught.value)


@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
def test_f1f_a_legacy_envelope_with_a_schema_version_key_is_identity_free_legacy(
    entry, tmp_path, monkeypatch
) -> None:
    """``schema_version`` + ``config`` but no ``config_id`` is not the envelope shape."""
    data = {"schema_version": "2026-10", "config": {"t": 2}}
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=False)
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2


@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        {"schema_version": "traigent.best_config.v2", "t": 2},
        {"schema_version": "traigent.best_config.", "t": 2},
        {"schema_version": "traigent.best_config.v2", "config": {"t": 2}},
        {"schema_version": "2026-10", "config_id": "agent", "config": {"t": 2}},
        {"schema_version": 7, "config_id": "agent", "config": {"t": 2}},
    ],
    ids=["ns_bare", "ns_prefix_only", "ns_config", "shape", "shape_non_string"],
)
def test_f1f_a_canonical_envelope_with_an_unsupported_version_is_still_refused(
    entry, strict, data, tmp_path, monkeypatch
) -> None:
    manager = _entry(entry, tmp_path, monkeypatch, data, strict=strict)
    if strict:
        with pytest.raises(ConfigurationError, match="schema_version"):
            manager.maybe_auto_load_config()
    else:
        manager.maybe_auto_load_config()
    assert _is_default(manager)


# F1f item 5: a customer-chosen path never reaches a log line or an exception text.
_PATH_CANARY = "acme-private-canary"


def _canary_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / _PATH_CANARY
    root.mkdir()
    monkeypatch.chdir(root)
    return root


def _assert_no_path(canary_text: str, caught=None) -> None:
    assert _PATH_CANARY not in canary_text
    if caught is not None:
        assert _PATH_CANARY not in str(caught.value) + repr(caught.value)


_REFUSED_FILES = {
    "contradiction": _legacy(config_id="other"),
    "identity_free": _IDENTITY_FREE,
    "bare_identity_free": {"t": 2},
    "unsupported_version": {"schema_version": "traigent.best_config.v2", "t": 2},
    "bad_canonical": {
        "schema_version": "traigent.best_config.v1",
        "config_id": "agent",
        "config": {"t": 2},
        "unknown_field": 1,
    },
}


@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
@pytest.mark.parametrize("kind", sorted(_REFUSED_FILES))
@pytest.mark.parametrize("strict", [False, True])
def test_f1f_a_refusal_never_prints_the_file_path(
    entry, kind, strict, tmp_path, monkeypatch, caplog
) -> None:
    root = _canary_root(tmp_path, monkeypatch)
    caplog.set_level(logging.DEBUG)
    manager = _entry(entry, root, monkeypatch, _REFUSED_FILES[kind], strict=strict)
    caught = None
    if strict:
        with pytest.raises(ConfigurationError) as caught:
            manager.maybe_auto_load_config()
        label = {"load_from": "load_from", "env": "TRAIGENT_CONFIG_PATH"}[entry]
        assert label in str(caught.value)
    else:
        manager.maybe_auto_load_config()
    _assert_no_path(caplog.text, caught)


@pytest.mark.parametrize("entry", EXPLICIT_ENTRY_POINTS)
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("problem", ["missing", "not_json", "not_object"])
def test_f1f_an_unreadable_file_never_prints_the_file_path(
    entry, strict, problem, tmp_path, monkeypatch, caplog
) -> None:
    root = _canary_root(tmp_path, monkeypatch)
    caplog.set_level(logging.DEBUG)
    path = root / "explicit.json"
    if problem == "not_json":
        path.write_text("{not json", encoding="utf-8")
    elif problem == "not_object":
        path.write_text("[1, 2]", encoding="utf-8")
    if entry == "load_from":
        manager = _manager(load_from=str(path), strict=strict)
    else:
        monkeypatch.setenv("TRAIGENT_CONFIG_PATH", str(path))
        manager = _manager(strict=strict)
    caught = None
    if strict:
        with pytest.raises(ConfigurationError) as caught:
            manager.maybe_auto_load_config()
    else:
        manager.maybe_auto_load_config()
    _assert_no_path(caplog.text, caught)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("kind", ["contradiction", "unsupported_version"])
def test_f1f_a_discovered_dev_log_refusal_never_prints_the_file_path(
    strict, kind, tmp_path, monkeypatch, caplog
) -> None:
    root = _canary_root(tmp_path, monkeypatch)
    caplog.set_level(logging.DEBUG)
    data = (
        _legacy(environment="staging")
        if kind == "contradiction"
        else _legacy(schema_version="traigent.best_config.v2")
    )
    manager = _entry("dev_log", root, monkeypatch, data, strict=strict)
    caught = None
    if strict:
        with pytest.raises(ConfigurationError) as caught:
            manager.maybe_auto_load_config()
        assert "dev-log" in str(caught.value)
    else:
        manager.maybe_auto_load_config()
    _assert_no_path(caplog.text, caught)


def test_f1f_a_loaded_file_does_not_log_its_path_either(
    tmp_path, monkeypatch, caplog
) -> None:
    root = _canary_root(tmp_path, monkeypatch)
    caplog.set_level(logging.DEBUG)
    for entry in ("load_from", "env", "dev_log"):
        manager = _entry(entry, root, monkeypatch, _legacy(), strict=False)
        manager.maybe_auto_load_config()
        assert manager._best_config_snapshot.config["t"] == 2
    _assert_no_path(caplog.text)


def test_f1f_the_path_stays_available_as_a_structured_attribute(
    tmp_path, monkeypatch
) -> None:
    root = _canary_root(tmp_path, monkeypatch)
    manager = _entry(
        "load_from", root, monkeypatch, _legacy(config_id="other"), strict=True
    )
    with pytest.raises(ConfigurationError) as caught:
        manager.maybe_auto_load_config()
    assert _PATH_CANARY in str(caught.value.details["path"])
    assert _PATH_CANARY not in str(caught.value)
