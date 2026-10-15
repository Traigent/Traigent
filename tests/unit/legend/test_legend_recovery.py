"""Legend part (b): ClientStore, exporters, conformance, capture, verified restore, activation."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest

from tests.unit.legend.conftest import make_dataset
from traigent.legend import (
    AgentStateSource,
    CaptureIndex,
    ConformanceOutcome,
    ConfigurationSource,
    DatasetSource,
    EvaluatorFilesSource,
    InMemoryTransport,
    LegendError,
    LocalDirectoryStore,
    SearchSpaceSource,
    activate,
    build_recovery_manifest,
    capture,
    compute_pillar,
    conformance_check,
    declare,
    export_agent_state,
    restore,
)
from traigent.legend import _contract as c
from traigent.legend.store import BUNDLE_NAME

PL = "pillar_legend_v0.json"
SSP = "ssp1:k" + "cd" * 8 + ":" + "ab" * 32
PASSED = ConformanceOutcome("passed")


@pytest.fixture
def store(tmp_path):
    return LocalDirectoryStore(tmp_path / "store")


@pytest.fixture
def idx():
    return CaptureIndex()


def _vref(transport, kind=c.KIND_CONFIGURATION, cfg=None):
    r = declare(kind, ConfigurationSource(cfg or {"seed": 1}), transport=transport)
    return r.version_ref


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        timeout=30,
    )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "agent"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "main.py").write_text("print('v1')\n")
    (root / "keep.txt").write_text("committed\n")
    (root / ".gitignore").write_text("secret.env\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(
        x for x in root.rglob("*") if x.is_file() and ".git" not in x.parts
    ):
        h.update(p.relative_to(root).as_posix().encode() + p.read_bytes())
    return h.hexdigest()


def _blob(store: LocalDirectoryStore, fp: str, kind: str) -> tuple[Path, Path]:
    return store._paths(kind, fp)


# ---------------------------------------------------------------- store
def test_store_refuses_overwrite_with_different_bytes(store):
    fp = c.MANIFEST_FINGERPRINT_PREFIX + "a" * 64
    assert store.put(c.KIND_AGENT, fp, io.BytesIO(b"one")) == "stored_unverified"
    assert store.put(c.KIND_AGENT, fp, io.BytesIO(b"one")) == "stored_unverified"
    with pytest.raises(LegendError) as err:
        store.put(c.KIND_AGENT, fp, io.BytesIO(b"two"))
    assert err.value.code == "manifest_mismatch"
    assert store.has(c.KIND_AGENT, fp) == "available"
    assert store.has(c.KIND_DATASET, fp) == "missing"


def test_store_has_no_delete_and_validates_keys(store):
    assert not hasattr(store, "delete") and not hasattr(store, "remove")
    with pytest.raises(LegendError):
        store.has(c.KIND_AGENT, "../../etc/passwd")
    with pytest.raises(LegendError):
        store.has("not_a_kind", c.MANIFEST_FINGERPRINT_PREFIX + "a" * 64)


# ---------------------------------------------------------------- conformance
class _ForgetfulStore:
    def put(self, kind, fp, stream):
        stream.read()
        return "stored_unverified"

    def get(self, kind, fp, dest):
        return "missing"

    def has(self, kind, fp):
        return "unknown"


class _OverwritingStore(LocalDirectoryStore):
    def put(self, kind, fp, stream):
        blob, meta = self._paths(kind, fp)
        blob.parent.mkdir(parents=True, exist_ok=True)
        data = stream.read()
        blob.write_bytes(data)
        meta.write_text(
            json.dumps(
                {"sha256": hashlib.sha256(data).hexdigest(), "length": len(data)}
            )
        )
        return "stored_unverified"


def test_conformance_passes_for_reference_store(store):
    assert conformance_check(store) == PASSED


@pytest.mark.parametrize("bad", ["forgetful", "overwriting"])
def test_conformance_failure_blocks_capture(bad, tmp_path, idx):
    store = (
        _ForgetfulStore() if bad == "forgetful" else _OverwritingStore(tmp_path / "s")
    )
    outcome = conformance_check(store)
    assert outcome.status == "conformance_failed" and outcome.kinds_failed
    t = InMemoryTransport()
    with pytest.raises(LegendError) as err:
        capture(
            _vref(t), ConfigurationSource({"a": 1}), store=store, transport=t, index=idx
        )
    assert err.value.code == "conformance_failed"
    assert not [e for e in t.events if e["event_type"] == "captured"]


# ---------------------------------------------------------------- capture
def test_capture_round_trip_verified_event_matches_contract(store, schemas, idx):
    t = InMemoryTransport()
    rec = capture(
        _vref(t), ConfigurationSource({"a": 1}), store=store, transport=t, index=idx
    )
    assert rec.recovery_state == "round_trip_verified" and rec.conformance_passed
    event = [e for e in t.events if e["event_type"] == "captured"][0]
    assert (
        event["evidence_source"] == "client_sdk_reported"
        and event["origin"] == "client"
    )
    schemas.validate(PL, "CapturedEventV0", event)
    schemas.validate(PL, "Coverage", rec.coverage)


def test_put_ack_without_round_trip_is_stored_unverified(store, tmp_path, idx, schemas):
    class WriteOnly(LocalDirectoryStore):
        def get(self, kind, fp, dest):
            return "missing"

    wo = WriteOnly(tmp_path / "wo")
    t = InMemoryTransport()
    rec = capture(
        _vref(t),
        ConfigurationSource({"a": 1}),
        store=wo,
        transport=t,
        conformance=PASSED,
        index=idx,
    )
    assert rec.recovery_state == "stored_unverified"
    event = [e for e in t.events if e["event_type"] == "captured"][0]
    assert event["recovery_state"] == "stored_unverified"
    schemas.validate(PL, "CapturedEventV0", event)


def test_capture_refuses_digest_only_sources(store, idx):
    from traigent.legend import AgentManifestSource

    with pytest.raises(LegendError) as err:
        capture(
            "versionref_" + "0" * 32,
            AgentManifestSource({}),
            store=store,
            conformance=PASSED,
            index=idx,
        )
    assert err.value.code == "coverage_insufficient"


# ---------------------------------------------------------------- restore grading
def test_exact_restore_of_configuration(store, tmp_path, idx, schemas):
    t = InMemoryTransport()
    rec = capture(
        _vref(t),
        ConfigurationSource({"a": 1, "b": [1, 2]}),
        store=store,
        transport=t,
        index=idx,
    )
    res = restore(rec, tmp_path / "out", store=store, transport=t, index=idx)
    assert (
        res.grade == "exact"
        and res.pillar_fingerprint_match
        and res.representation_digests_match
    )
    event = [e for e in t.events if e["event_type"] == "restore_verified"][0]
    assert event["outcome"] == "exact" and event["isolated_destination"] is True
    schemas.validate(PL, "RestoreVerifiedEventV0", event)


def _captured_config(store, idx):
    rec = capture(
        "versionref_" + "1" * 32,
        ConfigurationSource({"a": 1}),
        store=store,
        conformance=PASSED,
        index=idx,
    )
    return rec, _blob(store, rec.manifest_fingerprint, rec.kind)


def test_missing_bytes_fail(store, tmp_path, idx):
    rec, (blob, _meta) = _captured_config(store, idx)
    blob.unlink()
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    assert res.grade == "failed" and res.store_status == "missing" and res.dest is None


def test_truncated_bytes_fail(store, tmp_path, idx):
    rec, (blob, _meta) = _captured_config(store, idx)
    blob.write_bytes(blob.read_bytes()[:-5])
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    assert res.grade == "failed" and res.store_status == "truncated"


def test_wrong_bytes_fail_even_if_store_is_consistent(store, tmp_path, idx):
    rec, (blob, meta) = _captured_config(store, idx)
    data = bytearray(blob.read_bytes())
    data[-3] ^= 0xFF  # flips a byte in the last representation
    blob.write_bytes(bytes(data))
    meta.write_text(
        json.dumps(
            {"sha256": hashlib.sha256(bytes(data)).hexdigest(), "length": len(data)}
        )
    )
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    assert (
        res.grade == "failed"
        and not res.representation_digests_match
        and res.dest is None
    )
    assert not (tmp_path / "o").exists()


def test_store_level_corruption_fails(store, tmp_path, idx):
    rec, (blob, _meta) = _captured_config(store, idx)
    data = bytearray(blob.read_bytes())
    data[-1] ^= 0xFF
    blob.write_bytes(bytes(data))
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    assert res.grade == "failed" and res.store_status == "corrupted"


def test_reordered_dataset_rows_match_root_but_not_manifest(keys):
    a = compute_pillar(
        c.KIND_DATASET,
        DatasetSource(make_dataset([("a", 1), ("b", 2), ("a", 1)]), keys),
    )
    b = compute_pillar(
        c.KIND_DATASET,
        DatasetSource(make_dataset([("b", 2), ("a", 1), ("a", 1)]), keys),
    )
    assert a.fingerprint == b.fingerprint  # order-free multiset root
    ma, mb = build_recovery_manifest(a), build_recovery_manifest(b)
    assert ma.fingerprint != mb.fingerprint and ma.document["ordering"] == "preserved"


def test_dataset_capture_restores_exact_with_order(keys, store, tmp_path, idx):
    ds = make_dataset([("x", 1), ("y", 2), ("x", 1)])
    rec = capture(
        "versionref_" + "2" * 32,
        DatasetSource(ds, keys),
        store=store,
        conformance=PASSED,
        keys=keys,
        index=idx,
    )
    assert rec.recovery_state == "round_trip_verified"
    res = restore(rec, tmp_path / "o", store=store, keys=keys, index=idx)
    assert res.grade == "exact"


def test_dataset_without_keys_cannot_be_exact(keys, store, tmp_path, idx):
    from traigent.identity.keys import clear_content_identity_keys

    rec = capture(
        "versionref_" + "3" * 32,
        DatasetSource(make_dataset([("x", 1)]), keys),
        store=store,
        conformance=PASSED,
        index=idx,
    )
    clear_content_identity_keys()
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    assert res.grade == "substituted" and not res.pillar_fingerprint_match


def test_missing_dependency_is_partial(store, tmp_path, idx):
    dep = "versionref_" + "9" * 32
    rec = capture(
        "versionref_" + "4" * 32,
        ConfigurationSource({"a": 1}),
        store=store,
        conformance=PASSED,
        dependencies=[(dep, c.KIND_DATASET)],
        index=idx,
    )
    assert restore(rec, tmp_path / "a", store=store, index=idx).grade == "partial"
    gone = restore(
        rec,
        tmp_path / "b",
        store=store,
        index=idx,
        dependency_present=lambda r, k: False,
    )
    assert gone.grade == "partial" and not gone.dependencies_satisfied
    ok = restore(
        rec,
        tmp_path / "c",
        store=store,
        index=idx,
        dependency_present=lambda r, k: r == dep,
    )
    assert ok.grade == "exact"


def test_search_space_is_substituted_never_exact(store, tmp_path, idx):
    rec = capture(
        "versionref_" + "5" * 32,
        SearchSpaceSource(SSP, {"t": [0.1, 0.2]}),
        store=store,
        conformance=PASSED,
        index=idx,
    )
    assert rec.coverage["status"] == "complete"
    assert restore(rec, tmp_path / "o", store=store, index=idx).grade == "substituted"


def test_evaluator_files_exact(store, tmp_path, idx):
    root = tmp_path / "ev"
    root.mkdir()
    (root / "score.py").write_text("def score(x):\n    return 1\n")
    src = EvaluatorFilesSource(
        root,
        ["score.py"],
        [{"name": "accuracy", "orientation": "maximize", "weight": 1}],
        "ev_1",
    )
    rec = capture(
        "versionref_" + "6" * 32, src, store=store, conformance=PASSED, index=idx
    )
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    assert res.grade == "exact"
    assert (res.dest / "evaluator" / "score.py").read_text().startswith("def score")


# ---------------------------------------------------------------- agent code state
def test_dirty_tree_capture_and_restore(tmp_path, store, idx):
    root = _repo(tmp_path)
    (root / "main.py").write_text("print('dirty edit')\n")  # modified, uncommitted
    (root / "new_prompt.txt").write_text("untracked prompt\n")  # untracked
    (root / "secret.env").write_text("TOKEN=x\n")  # ignored
    (root / "keep.txt").unlink()  # tracked file deleted in the work tree
    lock = tmp_path / "uv.lock"
    lock.write_text("lock v1\n")
    before = _tree_hash(root)
    src = AgentStateSource(root, "agent_1", {"temperature": 0.2}, lockfile=lock)
    rec = capture(
        "versionref_" + "7" * 32, src, store=store, conformance=PASSED, index=idx
    )
    assert (
        rec.coverage["status"] == "complete"
        and "dependency_lock" in rec.coverage["covered"]
    )
    dest = tmp_path / "restored"
    res = restore(rec, dest, store=store, index=idx)
    assert res.grade == "exact"
    assert (dest / "tree" / "main.py").read_text() == "print('dirty edit')\n"
    assert (dest / "tree" / "new_prompt.txt").read_text() == "untracked prompt\n"
    assert not (dest / "tree" / "secret.env").exists()  # .gitignore respected
    assert not (dest / "tree" / "keep.txt").exists()
    assert not any(".git" in p.parts for p in (dest / "tree").rglob("*"))
    assert _tree_hash(root) == before  # restore never touched the source


def test_restore_refuses_destination_inside_source(tmp_path, store, idx):
    root = _repo(tmp_path)
    rec = capture(
        "versionref_" + "8" * 32,
        AgentStateSource(root, "a"),
        store=store,
        conformance=PASSED,
        index=idx,
    )
    before = _tree_hash(root)
    for bad in (root, root / "sub", root.parent):
        with pytest.raises(LegendError) as err:
            restore(rec, bad, store=store, index=idx)
        assert err.value.code == "invalid_request"
    assert _tree_hash(root) == before
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "f").write_text("x")
    with pytest.raises(LegendError):
        restore(rec, tmp_path / "busy", store=store, index=idx)


def test_size_limit_gives_partial_coverage_not_silence(tmp_path, store, idx):
    root = _repo(tmp_path)
    (root / "big.bin").write_bytes(b"0" * 5000)
    exp = export_agent_state(AgentStateSource(root, "a", max_file_bytes=1000))
    assert exp.skipped == 1
    assert exp.pillar.covered == ("applied_config",)
    rec = capture(
        "versionref_" + "a" * 32,
        AgentStateSource(root, "a", max_file_bytes=1000),
        store=store,
        conformance=PASSED,
        index=idx,
    )
    assert rec.coverage["status"] == "partial" and "code" in rec.coverage["missing"]
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    assert (
        res.grade == "partial" and res.pillar_fingerprint_match
    )  # content verified, coverage not


def test_agent_state_change_changes_manifest_and_fingerprint(tmp_path):
    root = _repo(tmp_path)
    a = export_agent_state(AgentStateSource(root, "a")).pillar
    (root / "main.py").write_text("print('v2')\n")
    b = export_agent_state(AgentStateSource(root, "a")).pillar
    assert a.fingerprint != b.fingerprint
    assert (
        build_recovery_manifest(a).fingerprint != build_recovery_manifest(b).fingerprint
    )


# ---------------------------------------------------------------- activation
def test_activate_requires_exact_or_explicit_accept(store, tmp_path, idx):
    dep = "versionref_" + "9" * 32
    rec = capture(
        "versionref_" + "b" * 32,
        ConfigurationSource({"a": 1}),
        store=store,
        conformance=PASSED,
        dependencies=[(dep, c.KIND_DATASET)],
        index=idx,
    )
    partial = restore(rec, tmp_path / "p", store=store, index=idx)
    assert partial.grade == "partial" and partial.activated is False
    with pytest.raises(LegendError):
        activate(partial, tmp_path / "live")
    with pytest.raises(LegendError):
        activate(partial, tmp_path / "live", accept="substituted")
    done = activate(partial, tmp_path / "live", accept="partial")
    assert done.grade == "partial" and done.accepted_inexact  # never labelled exact
    assert (tmp_path / "live").is_dir()

    exact_rec, _ = _captured_config(store, idx)
    exact = restore(exact_rec, tmp_path / "e", store=store, index=idx)
    out = activate(exact, tmp_path / "live2")
    assert out.grade == "exact" and not out.accepted_inexact
    with pytest.raises(LegendError):  # target must be empty unless overwrite
        activate(exact, tmp_path / "live2")


def test_activate_never_accepts_failed(store, tmp_path, idx):
    rec, (blob, _m) = _captured_config(store, idx)
    blob.unlink()
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    with pytest.raises(LegendError):
        activate(res, tmp_path / "live", accept="partial")


def test_restore_by_version_ref_and_list(store, tmp_path, idx):
    t = InMemoryTransport()
    ref = _vref(t)
    capture(ref, ConfigurationSource({"a": 1}), store=store, transport=t, index=idx)
    one = restore(ref, tmp_path / "one", store=store, index=idx)
    assert one.grade == "exact" and one.version_ref == ref
    many = restore([ref, ref], tmp_path / "many", store=store, index=idx)
    assert [r.grade for r in many] == ["exact", "exact"]
    assert BUNDLE_NAME not in {p.name for p in (tmp_path / "one").iterdir()}


def test_distinct_captures_do_not_overwrite(store, tmp_path, idx):
    r1 = capture(
        "versionref_" + "c" * 32,
        ConfigurationSource({"a": 1}),
        store=store,
        conformance=PASSED,
        index=idx,
    )
    r2 = capture(
        "versionref_" + "c" * 32,
        ConfigurationSource({"a": 2}),
        store=store,
        conformance=PASSED,
        index=idx,
    )
    assert r1.manifest_fingerprint != r2.manifest_fingerprint
    assert restore(r1, tmp_path / "1", store=store, index=idx).grade == "exact"
    assert restore(r2, tmp_path / "2", store=store, index=idx).grade == "exact"
