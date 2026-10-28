"""Capture: dataset, evaluator, agent, git, config, manifest."""

from __future__ import annotations

import json
import operator
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.unit.snapshot import helpers
from traigent.evaluators.base import EvaluationExample
from traigent.snapshot import (
    EvaluatorInputs,
    ExecutionSnapshot,
    build_execution_snapshot,
)
from traigent.snapshot.manifest import canonical_manifest_bytes, parse_manifest
from traigent.utils.artifact_fingerprints import (
    compute_dataset_fingerprint,
    effective_evaluator_code_identity,
)


def clock() -> datetime:
    return datetime(*helpers.FIXED_CLOCK_ARGS, 999, tzinfo=UTC)


def manifest_parts(snapshot: ExecutionSnapshot) -> dict[str, Any]:
    return dict(snapshot.manifest["parts"])


def reason(snapshot: ExecutionSnapshot, path: str) -> str | None:
    entry = manifest_parts(snapshot)[path]
    return entry.get("reason")


# ---------------------------------------------------------------- dataset


def test_dataset_fingerprint_after_decode_equals_original() -> None:
    dataset = [
        EvaluationExample(input_data={"q": "café   line"}, expected_output="a"),
        {
            "input": {"q": 2},
            "expected_output": [1, 2],
            "id": "x1",
            "metadata": {"k": "v"},
        },
        {"input_data": {"q": 3}, "answer": "z"},
    ]
    snapshot = build_execution_snapshot(dataset=dataset)
    rows = [
        json.loads(line)
        for line in snapshot.parts["dataset/examples.jsonl"].decode("utf-8").split("\n")
        if line
    ]
    assert len(rows) == 3
    assert compute_dataset_fingerprint(rows) == compute_dataset_fingerprint(dataset)
    assert snapshot.manifest["fingerprints"][
        "dataset_fp1"
    ] == compute_dataset_fingerprint(dataset)
    assert rows[1]["id"] == "x1" and rows[1]["metadata"] == {"k": "v"}


def test_dataset_order_preserved() -> None:
    dataset = [{"input": i, "expected": i} for i in range(5)]
    rows = build_execution_snapshot(dataset=dataset).parts["dataset/examples.jsonl"]
    assert [
        json.loads(line)["input"] for line in rows.decode().split("\n") if line
    ] == [0, 1, 2, 3, 4]


def test_generator_refused_and_not_consumed() -> None:
    generator = ({"input": i, "expected": i} for i in range(3))
    snapshot = build_execution_snapshot(dataset=generator)
    assert reason(snapshot, "dataset/examples.jsonl") == "DATASET_NOT_MATERIALIZED"
    assert next(generator)["input"] == 0  # never iterated


def test_non_json_dataset_value_is_not_serializable_not_stringified() -> None:
    snapshot = build_execution_snapshot(dataset=[{"input": {1, 2}, "expected": "x"}])
    assert reason(snapshot, "dataset/examples.jsonl") == "NOT_SERIALIZABLE"
    assert "dataset/examples.jsonl" not in snapshot.parts


def test_nan_dataset_value_is_not_serializable() -> None:
    snapshot = build_execution_snapshot(
        dataset=[{"input": float("nan"), "expected": 1}]
    )
    assert reason(snapshot, "dataset/examples.jsonl") == "NOT_SERIALIZABLE"


def test_dataset_not_provided() -> None:
    assert (
        reason(build_execution_snapshot(), "dataset/examples.jsonl") == "NOT_PROVIDED"
    )


# ---------------------------------------------------------------- config


def test_config_strict_json() -> None:
    ok = build_execution_snapshot(
        config={"temperature": 0.7}, config_space={"m": ["a"]}
    )
    assert json.loads(ok.parts["config/config.json"]) == {"temperature": 0.7}
    bad = build_execution_snapshot(config={"x": object()})
    assert reason(bad, "config/config.json") == "NOT_SERIALIZABLE"


def test_mutating_callers_dict_after_build_changes_nothing() -> None:
    config = {"nested": {"a": 1}}
    rows = [{"input": {"a": 1}, "expected": 1}]
    snapshot = build_execution_snapshot(config=config, dataset=rows)
    digest, config_bytes = (
        snapshot.manifest_digest,
        snapshot.parts["config/config.json"],
    )
    config["nested"]["a"] = 999
    rows[0]["input"]["a"] = 999
    assert snapshot.manifest_digest == digest
    assert snapshot.parts["config/config.json"] == config_bytes
    assert json.loads(config_bytes) == {"nested": {"a": 1}}


def test_parts_copied_at_construction() -> None:
    snapshot = build_execution_snapshot(config={"a": 1})
    parts = dict(snapshot.parts)
    original = snapshot.parts["config/config.json"]
    caller = dict(parts)
    rebuilt = ExecutionSnapshot(snapshot.manifest_bytes, caller)
    caller["config/config.json"] = b"{}"
    caller["extra"] = b"x"
    assert rebuilt.parts["config/config.json"] == original
    assert "extra" not in rebuilt.parts
    with pytest.raises(TypeError):
        operator.setitem(rebuilt.parts, "config/config.json", b"{}")


def test_repr_shows_digests_not_content() -> None:
    snapshot = build_execution_snapshot(config={"secret": "CONTENT-CANARY"})
    text = repr(snapshot)
    assert snapshot.manifest_digest in text and "CONTENT-CANARY" not in text


# ---------------------------------------------------------------- evaluator


def test_custom_evaluator_member_and_identity() -> None:
    snapshot = build_execution_snapshot(
        evaluator=EvaluatorInputs(custom_evaluator=helpers.custom)
    )
    path = "evaluator/custom_evaluator/custom.py"
    assert b"def custom" in snapshot.parts[path]
    fingerprints = snapshot.manifest["fingerprints"]
    expected = effective_evaluator_code_identity(custom_evaluator=helpers.custom)
    assert fingerprints["evaluator_code_identity"]["digest"] == expected["digest"]
    assert fingerprints["evaluator_code_identity"]["algorithm"] == "ecd1"
    assert fingerprints["evaluator_fp1"].startswith("fp1:")
    assert (
        fingerprints["evaluator_fp1"]
        != fingerprints["evaluator_code_identity"]["digest"]
    )


def test_metric_shadows_scoring_function() -> None:
    inputs = EvaluatorInputs(
        scoring_function=helpers.scorer,
        metric_functions={"accuracy": helpers.metric_a},
        objectives=["accuracy"],
    )
    snapshot = build_execution_snapshot(evaluator=inputs)
    identity = snapshot.manifest["fingerprints"]["evaluator_code_identity"]
    effective = effective_evaluator_code_identity(
        scoring_function=helpers.scorer,
        metric_functions={"accuracy": helpers.metric_a},
        objectives=["accuracy"],
    )
    unshadowed = effective_evaluator_code_identity(
        scoring_function=helpers.scorer, objectives=["accuracy"]
    )
    assert identity["digest"] == effective["digest"] != unshadowed["digest"]
    assert "evaluator/metric/accuracy.py" in snapshot.parts
    # legacy fp1 keeps its own (scoring-function-first) semantics, separately
    assert snapshot.manifest["fingerprints"]["evaluator_fp1"].startswith("fp1:")


def test_multiple_metrics_each_stored() -> None:
    snapshot = build_execution_snapshot(
        evaluator=EvaluatorInputs(
            metric_functions={"a": helpers.metric_a, "b": helpers.metric_b}
        )
    )
    assert b"def metric_a" in snapshot.parts["evaluator/metric/a.py"]
    assert b"def metric_b" in snapshot.parts["evaluator/metric/b.py"]
    assert (
        snapshot.manifest["fingerprints"]["evaluator_code_identity"]["origin"] == "code"
    )


def test_external_evaluator_has_no_source() -> None:
    snapshot = build_execution_snapshot(
        evaluator=EvaluatorInputs(
            external={"kind": "http", "endpoint": "https://e.test/x?token=S"}
        )
    )
    assert reason(snapshot, "evaluator/source") == "EXTERNAL_EVALUATOR"
    assert snapshot.manifest["fingerprints"]["evaluator_code_identity"] == {
        "origin": "external",
        "digest": None,
        "algorithm": None,
    }
    assert "token=S" not in snapshot.manifest_bytes.decode()


def test_builtin_and_absent_evaluator() -> None:
    builtin = build_execution_snapshot(evaluator=EvaluatorInputs())
    assert reason(builtin, "evaluator/source") == "BUILTIN_EVALUATOR"
    absent = build_execution_snapshot()
    assert reason(absent, "evaluator/source") == "NOT_PROVIDED"
    assert absent.manifest["fingerprints"]["evaluator_fp1"] is None


def test_unreadable_evaluator_source_is_unavailable() -> None:
    namespace: dict[str, Any] = {}
    exec("def dyn(a, b):\n    return 1\n", namespace)
    snapshot = build_execution_snapshot(
        evaluator=EvaluatorInputs(metric_functions={"dyn": namespace["dyn"]})
    )
    assert reason(snapshot, "evaluator/metric/dyn.py") == "SOURCE_UNAVAILABLE"


# ---------------------------------------------------------------- agent / git


def test_agent_assets_and_bound_state_stored_raw() -> None:
    snapshot = build_execution_snapshot(
        agent=helpers.sample_agent,
        declared_assets={"prompts": {"sys": "Be brief"}, "tool_definitions": {}},
    )
    assert b"def sample_agent" in snapshot.parts["agent/callable.py"]
    assert json.loads(snapshot.parts["agent/assets/prompts.json"]) == {
        "sys": "Be brief"
    }
    assert json.loads(snapshot.parts["agent/assets/tool_definitions.json"]) == {}
    assert snapshot.parts["agent/bound_state.json"] == b"{}"


def test_undeclared_assets_are_not_provided_and_partial() -> None:
    snapshot = build_execution_snapshot(agent=helpers.sample_agent)
    assert reason(snapshot, "agent/assets/prompts.json") == "NOT_PROVIDED"
    assert snapshot.manifest["snapshot_coverage"]["agent"] == "partial"
    assert (
        "prompts_not_provided" in snapshot.manifest["snapshot_coverage"]["agent_gaps"]
    )


def test_unknown_declared_asset_key_refused() -> None:
    with pytest.raises(ValueError):
        build_execution_snapshot(declared_assets={"secrets": {}})


def test_non_json_asset_is_not_serializable() -> None:
    snapshot = build_execution_snapshot(
        agent=helpers.sample_agent, declared_assets={"prompts": {"p": b"bytes"}}
    )
    assert reason(snapshot, "agent/assets/prompts.json") == "NOT_SERIALIZABLE"


def test_bound_state_unavailable_when_not_canonical() -> None:
    import functools

    bound = functools.partial(helpers.sample_agent, object())
    snapshot = build_execution_snapshot(agent=bound)
    assert reason(snapshot, "agent/bound_state.json") == "BOUND_STATE_UNAVAILABLE"
    assert snapshot.manifest["snapshot_coverage"]["agent"] == "partial"


def test_bound_state_stored_when_canonical() -> None:
    import functools

    snapshot = build_execution_snapshot(
        agent=functools.partial(helpers.sample_agent, "arg")
    )
    assert json.loads(snapshot.parts["agent/bound_state.json"]) == {
        "partial_args": ["arg"]
    }


def test_agent_build_fields_need_agent_id() -> None:
    without = build_execution_snapshot(agent=helpers.sample_agent)
    fingerprints = without.manifest["fingerprints"]
    assert fingerprints["agent_build_digest"] is None
    assert fingerprints["agent_build_unavailable"] == "NOT_PROVIDED"
    assert (
        b"def sample_agent" in without.parts["agent/callable.py"]
    )  # raw capture still happens
    with_id = build_execution_snapshot(agent=helpers.sample_agent, agent_id="my-agent")
    got = with_id.manifest["fingerprints"]
    assert got["agent_build_unavailable"] is None
    assert got["agent_build_digest"].startswith("sha256:")
    assert got["agent_build_coverage"] in {"complete", "partial"}
    assert got["agent_fp1"].startswith("fp1:")


def test_complete_agent_coverage_in_clean_git_repo(tmp_path: Path) -> None:
    repo, agent = helpers.make_git_agent(tmp_path, "snapagent_clean")
    try:
        snapshot = build_execution_snapshot(
            agent=agent,
            declared_assets={"prompts": {}, "tool_definitions": {}},
            agent_id="clean-agent",
        )
    finally:
        sys.modules.pop("snapagent_clean", None)
    assert snapshot.manifest["snapshot_coverage"] == {
        "agent": "complete",
        "agent_gaps": [],
    }
    assert b"upper" in snapshot.parts["agent/source/snapagent_clean.py"]
    git_info = json.loads(snapshot.parts["agent/git.json"])
    assert git_info["dirty"] is False and len(git_info["commit"]) == 40
    assert git_info["entry_path"] == "snapagent_clean.py"
    assert "agent/git_diff.patch" not in manifest_parts(
        snapshot
    )  # clean: nothing to diff
    assert str(repo) not in snapshot.manifest_bytes.decode()  # no absolute paths


def test_snapshot_coverage_is_computed_not_copied(tmp_path: Path) -> None:
    """Complete build coverage must not make snapshot coverage complete."""
    _, agent = helpers.make_git_agent(tmp_path, "snapagent_cov")
    try:
        snapshot = build_execution_snapshot(
            agent=agent,
            declared_assets={"prompts": {}},  # tool definitions missing
            agent_id="cov-agent",
        )
    finally:
        sys.modules.pop("snapagent_cov", None)
    assert snapshot.manifest["snapshot_coverage"]["agent"] == "partial"


def test_dirty_diff_not_requested_by_default(tmp_path: Path) -> None:
    repo, agent = helpers.make_git_agent(tmp_path, "snapagent_dirty")
    try:
        (repo / "snapagent_dirty.py").write_text(
            "def run(question):\n    return 'EDITED-SECRET-LINE'\n", encoding="utf-8"
        )
        (repo / "untracked.txt").write_text("UNTRACKED-CANARY", encoding="utf-8")
        snapshot = build_execution_snapshot(agent=agent)
    finally:
        sys.modules.pop("snapagent_dirty", None)
    assert reason(snapshot, "agent/git_diff.patch") == "NOT_REQUESTED"
    assert json.loads(snapshot.parts["agent/git.json"])["dirty"] is True
    assert "agent/git_diff.patch" not in snapshot.parts
    assert b"UNTRACKED-CANARY" not in b"".join(snapshot.parts.values())


def test_dirty_diff_opt_in(tmp_path: Path) -> None:
    repo, agent = helpers.make_git_agent(tmp_path, "snapagent_diff")
    try:
        (repo / "snapagent_diff.py").write_text(
            "def run(question):\n    return 'EDITED-LINE'\n", encoding="utf-8"
        )
        (repo / "untracked.txt").write_text("UNTRACKED-CANARY", encoding="utf-8")
        (repo / "other.bin").write_bytes(bytes(range(256)))
        helpers.git(repo, "add", "other.bin")  # staged binary is included
        snapshot = build_execution_snapshot(agent=agent, capture_dirty_diff=True)
    finally:
        sys.modules.pop("snapagent_diff", None)
    patch = snapshot.parts["agent/git_diff.patch"]
    assert b"EDITED-LINE" in patch and b"GIT binary patch" in patch
    assert b"UNTRACKED-CANARY" not in patch
    info = json.loads(snapshot.parts["agent/git.json"])
    assert info["untracked_files_excluded"] >= 1 and info["dirty"] is True


def test_dirty_diff_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from traigent.snapshot import capture

    repo, agent = helpers.make_git_agent(tmp_path, "snapagent_big")
    monkeypatch.setattr(capture, "MAX_DIRTY_DIFF_BYTES", 10)
    try:
        (repo / "snapagent_big.py").write_text("def run(q):\n    return 'x' * 50\n")
        snapshot = build_execution_snapshot(agent=agent, capture_dirty_diff=True)
    finally:
        sys.modules.pop("snapagent_big", None)
    assert reason(snapshot, "agent/git_diff.patch") == "TOO_LARGE"


def test_dirty_diff_reader_stops_early_instead_of_reading_it_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diff is streamed and git is stopped at the cap, not read to the end."""
    import subprocess

    from traigent.identity import agent_build

    repo, _ = helpers.make_git_agent(tmp_path, "snapagent_stream")
    sys.modules.pop("snapagent_stream", None)
    (repo / "snapagent_stream.py").write_text("x = 1\n" * 800_000, encoding="utf-8")
    full = helpers.git(repo, "diff", "HEAD")
    assert len(full) > 4_000_000

    read_bytes: list[int] = []
    real_popen = subprocess.Popen

    class CountingStream:
        def __init__(self, stream: Any) -> None:
            self._stream = stream

        def read(self, size: int) -> bytes:
            data = self._stream.read(size)
            read_bytes.append(len(data))
            return data

        def close(self) -> None:
            self._stream.close()

    def counting_popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        process.stdout = CountingStream(process.stdout)
        return process

    monkeypatch.setattr(agent_build.subprocess, "Popen", counting_popen)
    with pytest.raises(agent_build.GitOutputTooLarge):
        agent_build.git_output_bounded(repo, "diff", "HEAD", max_bytes=100_000)
    assert 100_000 < sum(read_bytes) <= 100_000 + 65536  # stopped within one chunk


def test_bounded_git_reader_returns_small_output_and_none_on_failure(
    tmp_path: Path,
) -> None:
    from traigent.identity import agent_build

    repo, _ = helpers.make_git_agent(tmp_path, "snapagent_small")
    sys.modules.pop("snapagent_small", None)
    head = agent_build.git_output_bounded(repo, "rev-parse", "HEAD", max_bytes=1000)
    assert head is not None and len(head.strip()) == 40
    assert agent_build.git_output_bounded(repo, "no-such-verb", max_bytes=1000) is None


def test_git_metadata_can_be_disabled(tmp_path: Path) -> None:
    _, agent = helpers.make_git_agent(tmp_path, "snapagent_nogit")
    try:
        snapshot = build_execution_snapshot(agent=agent, capture_git_metadata=False)
    finally:
        sys.modules.pop("snapagent_nogit", None)
    assert reason(snapshot, "agent/git.json") == "NOT_REQUESTED"


def test_git_unavailable_outside_a_repo(tmp_path: Path) -> None:
    import importlib.util

    path = tmp_path / "plain_agent.py"
    path.write_text("def run(q):\n    return q\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("plain_agent_x", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = build_execution_snapshot(agent=module.run)
    assert reason(snapshot, "agent/git.json") == "GIT_UNAVAILABLE"


def test_remote_urls_never_captured(tmp_path: Path) -> None:
    repo, agent = helpers.make_git_agent(tmp_path, "snapagent_remote")
    userinfo = "user:" + "TOKEN-CANARY"
    helpers.git(repo, "remote", "add", "origin", f"https://{userinfo}@host.test/r.git")
    try:
        snapshot = build_execution_snapshot(agent=agent, capture_dirty_diff=True)
    finally:
        sys.modules.pop("snapagent_remote", None)
    blob = snapshot.manifest_bytes + b"".join(snapshot.parts.values())
    assert b"TOKEN-CANARY" not in blob and b"host.test" not in blob


# ---------------------------------------------------------------- manifest


def test_manifest_canonical_bytes_deterministic() -> None:
    kwargs: dict[str, Any] = {
        "dataset": [{"input": 1, "expected": 2}],
        "config": {"b": 1.5, "a": 2},
        "evaluator": EvaluatorInputs(metric_functions={"a": helpers.metric_a}),
        "clock": clock,
    }
    first, second = (
        build_execution_snapshot(**kwargs),
        build_execution_snapshot(**kwargs),
    )
    assert first.manifest_bytes == second.manifest_bytes
    assert first.manifest_digest == second.manifest_digest
    assert first.manifest["created_at"] == "2026-10-27T12:30:45Z"
    parsed = parse_manifest(first.manifest_bytes)
    assert canonical_manifest_bytes(parsed) == first.manifest_bytes
    assert first.manifest["binding"] == {"kind": "unbound"}
    assert first.manifest["format"] == "tes1"
    assert b"1.5" not in first.manifest_bytes  # floats live in parts, not the manifest


def test_manifest_rejects_float_nan_and_noncanonical() -> None:
    with pytest.raises(ValueError):
        canonical_manifest_bytes({"a": 1.5})
    with pytest.raises(ValueError):
        canonical_manifest_bytes({"a": float("nan")})
    good = build_execution_snapshot(config={"a": 1}, clock=clock).manifest_bytes
    from traigent.snapshot import SnapshotStoreError

    for bad in (good + b" ", b"{}", b"not json", good.replace(b'"tes1"', b'"tes9"')):
        with pytest.raises(SnapshotStoreError):
            parse_manifest(bad)


def test_part_paths_validated() -> None:
    from traigent.snapshot.manifest import validate_part_path

    for bad in ("", "/abs", "a/../b", "a//b", "a\\b", "./a", "a/."):
        with pytest.raises(ValueError):
            validate_part_path(bad)
    assert validate_part_path("agent/source/pkg/mod.py") == "agent/source/pkg/mod.py"
