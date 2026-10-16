"""Legend part (b): ClientStore, exporters, conformance, capture, verified restore, activation."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.unit.legend.conftest import make_dataset
from traigent.legend import (
    AgentSource,
    AgentStateSource,
    CaptureIndex,
    ConformanceOutcome,
    ConfigurationSource,
    DatasetSource,
    EvaluatorFilesSource,
    EvaluatorSource,
    InMemoryTransport,
    LegendError,
    LocalDirectoryStore,
    SearchSpaceSource,
    activate,
    build_recovery_manifest,
    record_availability,
    capture,
    compute_pillar,
    conformance_check,
    declare,
    export_agent_state,
    recompute_fingerprint,
    restore,
)
from traigent.legend import _contract as c
from traigent.legend.manifest import (
    pillar_representation_commitment,
    representation_commitment,
    validate_coverage,
)
from traigent.legend.store import BUNDLE_NAME

PL = "pillar_legend_v0.json"
SSP = "ssp1:k" + "cd" * 8 + ":" + "ab" * 32
PASSED = ConformanceOutcome("passed")
PROOF = "eventref_" + "7" * 32


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
    assert store.put(c.KIND_AGENT, fp, io.BytesIO(b"two")) == "store_key_conflict"
    assert store.get(c.KIND_AGENT, fp, store.root / "chk") == "retrieved"
    assert (store.root / "chk" / BUNDLE_NAME).read_bytes() == b"one"
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
    src = SearchSpaceSource(SSP, {"t": [0.1, 0.2]})
    # without the Backend's pre-capture identity_verified proof the capture is partial
    bare = capture(
        "versionref_" + "5" * 32, src, store=store, conformance=PASSED, index=idx
    )
    assert bare.coverage["status"] == "partial" and not bare.identity_digests_reproduced
    rec = capture(
        "versionref_" + "5" * 32,
        src,
        store=store,
        conformance=PASSED,
        index=idx,
        identity_verification_ref=PROOF,
    )
    assert rec.coverage["status"] == "complete" and rec.backend_keyed
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    assert res.grade == "substituted" and res.verified_artifact_digest is None


# ---------------------------------------------------------------- agent code state
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


# ---------------------------------------------------------------- C9 / INV-17: identity digests
RM = "recovery_manifest_v0.json"
REQ = "legend_requests_v0.json"
HELPER = "def helper(x):\n    return x + 1\n"
PROMPT = "You are a careful assistant.\n"
TOOLS = {"name": "search", "params": ["q"]}
AGENT_SRC = "import helper_mod\n\n\ndef agent(q):\n    return helper_mod.helper(q)\n"


class Project:
    def __init__(self, tmp_path: Path, tag: str) -> None:
        self.root = tmp_path / f"proj_{tag}"
        self.root.mkdir()
        _git(self.root, "init", "-q")
        (self.root / ".gitignore").write_text("__pycache__/\n")
        (self.root / "helper_mod.py").write_text(HELPER)
        (self.root / "agent_mod.py").write_text(AGENT_SRC)
        (self.root / "prompts").mkdir()
        (self.root / "prompts" / "system.txt").write_text(PROMPT)
        (self.root / "tools.json").write_text(json.dumps(TOOLS))
        (self.root / "uv.lock").write_text("lock v1\n")
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "-m", "init")
        sys.path.insert(0, str(self.root))
        self.mod_name = f"agent_mod_{tag}"
        spec = importlib.util.spec_from_file_location(
            self.mod_name, self.root / "agent_mod.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.mod_name] = module
        spec.loader.exec_module(module)
        from traigent.identity.agent_build import declare_agent_assets

        self.func = declare_agent_assets(
            module.agent,
            prompts={"system": PROMPT},
            tool_definitions={"search": TOOLS},
            helper_modules={"helper_mod.py": HELPER},
            coverage="complete",
        )
        self.config = {"temperature": 0.2}

    def close(self) -> None:
        sys.modules.pop(self.mod_name, None)
        sys.modules.pop("helper_mod", None)
        if str(self.root) in sys.path:
            sys.path.remove(str(self.root))

    def source(self, **kw) -> AgentStateSource:
        base = {
            "root": self.root,
            "agent_id": "agent_1",
            "config": self.config,
            "func": self.func,
            "prompt_files": {"system": "prompts/system.txt"},
            "tool_definition_files": {"search": "tools.json"},
        }
        return AgentStateSource(**{**base, **kw})


@pytest.fixture
def project(tmp_path):
    p = Project(tmp_path, "a")
    yield p
    p.close()


def test_callable_declared_agent_and_captured_state_yield_same_build_digest(
    project, store, tmp_path, idx, schemas
):
    t = InMemoryTransport()
    declared = declare(
        c.KIND_AGENT, AgentSource(project.func, "agent_1", project.config), transport=t
    )
    fingerprint = t.versions[declared.version_ref]["fingerprint"]
    exp = export_agent_state(project.source(declared_fingerprint=fingerprint))
    assert exp.pillar.fingerprint == fingerprint  # callable-declared == captured state
    assert exp.pillar.reproduced and exp.skipped == 0
    # the digest recomputed from the CAPTURED representations alone is the same digest
    kinds = [r.kind for r in exp.pillar.representations]
    reps = [r.data for r in exp.pillar.representations]
    frame = {
        k: v
        for k, v in exp.manifest.items()
        if k not in ("asset_digests", "applied_config_digest", "dependency_lock_digest")
    }
    assert (
        recompute_fingerprint(c.KIND_AGENT, kinds, reps, {"frame": frame})
        == fingerprint
    )
    m = build_recovery_manifest(exp.pillar, source_entity_ref=declared.entity_ref)
    assert m.document["identity_digests_reproduced"] is True
    assert m.document["coverage"] == {
        "status": "complete",
        "covered": list(c.REQUIRED_BY_KIND[c.KIND_AGENT]),
        "missing": [],
    }
    schemas.validate(RM, "RecoveryManifestV0", m.document)
    rec = capture(
        declared.version_ref,
        project.source(declared_fingerprint=fingerprint),
        store=store,
        transport=t,
        conformance=PASSED,
        index=idx,
    )
    assert rec.coverage["status"] == "complete"
    res = restore(rec, tmp_path / "restored", store=store, index=idx)
    assert res.grade == "exact"
    assert (tmp_path / "restored" / "tree" / "helper_mod.py").read_text() == HELPER


def test_capture_requests_validate_against_request_schemas(
    project, store, tmp_path, idx, schemas
):
    t = InMemoryTransport()
    ref = declare(c.KIND_CONFIGURATION, ConfigurationSource({"a": 1}), transport=t)
    rec = capture(
        ref.version_ref,
        ConfigurationSource({"a": 1}),
        store=store,
        transport=t,
        conformance=PASSED,
        index=idx,
    )
    restore(rec, tmp_path / "o", store=store, transport=t, index=idx)
    cap, res = t.requests[-2], t.requests[-1]
    schemas.validate(REQ, "RecordCaptureRequestV0", cap)
    schemas.validate(REQ, "RecordRestoreRequestV0", res)
    for body in (cap, res):
        assert not {"origin", "evidence_source", "event_type"} & set(body)
        assert c.IDEMPOTENCY_KEY_RE.fullmatch(body["idempotency_key"])
    assert cap["idempotency_key"] != res["idempotency_key"]


@pytest.mark.parametrize(
    "path,text",
    [("helper_mod.py", HELPER + "# edited\n"), ("prompts/system.txt", PROMPT + "x\n")],
)
def test_changed_asset_file_means_identity_not_reproduced_and_partial(
    project, store, tmp_path, idx, path, text
):
    fp = export_agent_state(project.source()).pillar.fingerprint
    (project.root / path).write_text(text)
    exp = export_agent_state(project.source(declared_fingerprint=fp))
    assert exp.pillar.fingerprint == fp and not exp.pillar.reproduced
    m = build_recovery_manifest(exp.pillar)
    assert m.document["coverage"]["status"] == "partial"
    assert m.document["identity_digests_reproduced"] is False
    rec = capture(
        "versionref_" + "d" * 32,
        project.source(declared_fingerprint=fp),
        store=store,
        conformance=PASSED,
        index=idx,
    )
    res = restore(rec, tmp_path / "o", store=store, index=idx)
    assert res.grade == "partial" and not res.pillar_fingerprint_match


def test_wrong_declared_fingerprint_is_partial(project):
    exp = export_agent_state(project.source(declared_fingerprint="sha256:" + "0" * 64))
    assert not exp.pillar.reproduced
    assert (
        build_recovery_manifest(exp.pillar).document["coverage"]["status"] == "partial"
    )


def test_manifest_only_agent_cannot_be_complete(project):
    full = export_agent_state(project.source())
    exp = export_agent_state(project.source(func=None, declared_manifest=full.manifest))
    assert exp.pillar.fingerprint == full.pillar.fingerprint
    cov = build_recovery_manifest(exp.pillar).document["coverage"]
    assert cov["status"] == "partial" and "code" in cov["missing"]


def test_agent_without_identity_is_refused(project):
    with pytest.raises(LegendError) as err:
        export_agent_state(project.source(func=None))
    assert err.value.code == "coverage_insufficient"


def test_unmapped_prompt_file_is_skipped_and_forces_partial(project, schemas):
    exp = export_agent_state(project.source(prompt_files={}))
    assert exp.skipped >= 1
    doc = build_recovery_manifest(exp.pillar).document
    assert doc["rederivation"]["skipped_file_count"] == exp.skipped
    assert doc["coverage"]["status"] == "partial"
    assert "prompts" in doc["coverage"]["missing"]
    schemas.validate(RM, "RecoveryManifestV0", doc)


def test_size_limit_gives_partial_coverage_not_silence(project):
    (project.root / "big.bin").write_bytes(b"0" * 5000)
    exp = export_agent_state(project.source(max_file_bytes=1000))
    assert exp.skipped >= 1
    cov = build_recovery_manifest(exp.pillar).document["coverage"]
    assert cov["status"] == "partial" and "code" in cov["missing"]


def test_coverage_partitions_required_components_per_kind(keys):
    for kind, source in {
        c.KIND_CONFIGURATION: ConfigurationSource({"a": 1}),
        c.KIND_DATASET: DatasetSource(make_dataset([("a", 1)]), keys),
        c.KIND_SEARCH_SPACE: SearchSpaceSource(SSP, {"t": [1]}),
    }.items():
        cov = build_recovery_manifest(compute_pillar(kind, source)).document["coverage"]
        assert not set(cov["covered"]) & set(cov["missing"])
        assert set(cov["covered"]) | set(cov["missing"]) == set(
            c.REQUIRED_BY_KIND[kind]
        )
        assert (cov["status"] == "complete") == (cov["missing"] == [])
    # a declared-only agent: everything not read is missing, never silently dropped
    p = compute_pillar(c.KIND_AGENT, AgentSource(_fn, "agent_1", {}))
    cov = build_recovery_manifest(p).document["coverage"]
    assert set(cov["covered"]) | set(cov["missing"]) == set(
        c.REQUIRED_BY_KIND[c.KIND_AGENT]
    )
    assert cov["status"] == "partial" and "runtime" in cov["missing"]


def _fn():  # noqa: ANN202 - fixture function whose source is hashed
    return 1


EV_SRC = (
    "def score(output, expected):\n    return 1.0 if output == expected else 0.0\n\n\n"
    "class Ev:\n    def __init__(self):\n        self.scoring_function = score\n"
    "        self.metrics = ['accuracy']\n"
)


def _evaluator_project(tmp_path: Path):
    from traigent.identity.evaluator_version import declare_evaluator

    root = tmp_path / "evproj"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "ev_mod.py").write_text(EV_SRC)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    spec = importlib.util.spec_from_file_location("ev_mod_x", root / "ev_mod.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ev_mod_x"] = module
    spec.loader.exec_module(module)
    ev = declare_evaluator(
        module.Ev(),
        judge=None,
        dependency_versions={},
        config_digest="sha256:" + "1" * 64,
    )
    objectives = [{"name": "accuracy", "orientation": "maximize", "weight": 1}]
    return root, ev, objectives


def test_evaluator_files_reproduce_identity_and_restore_exact(
    store, tmp_path, idx, schemas
):
    from traigent.legend import EvaluatorManifestSource  # noqa: F401

    root, ev, objectives = _evaluator_project(tmp_path)
    try:
        src = EvaluatorFilesSource(root, evaluator=ev, objectives=objectives)
        pillar = export_pillar_of(src)
        assert pillar.reproduced
        # evaluators are Backend-keyed: no proof => partial, never client-reproduced
        bare = build_recovery_manifest(pillar).document
        assert bare["backend_keyed"] is True
        assert bare["coverage"]["status"] == "partial"
        assert bare["identity_digests_reproduced"] is False
        schemas.validate(RM, "RecoveryManifestV0", bare)
        doc = build_recovery_manifest(pillar, identity_verification_ref=PROOF).document
        assert doc["coverage"]["status"] == "complete"
        assert doc["identity_digests_reproduced"] is True
        assert doc["identity_verification_ref"] == PROOF
        schemas.validate(RM, "RecoveryManifestV0", doc)
        t = InMemoryTransport()
        vref = declare(
            c.KIND_EVALUATOR,
            EvaluatorSource(ev, objectives),
            transport=t,
        ).version_ref
        rec = capture(
            vref,
            src,
            store=store,
            transport=t,
            conformance=PASSED,
            index=idx,
            identity_verification_ref=PROOF,
        )
        assert rec.representation_commitment == pillar_representation_commitment(pillar)
        res = restore(rec, tmp_path / "o", store=store, transport=t, index=idx)
        # a client-origin restore of a backend-keyed pillar is at most substituted (INV-16)
        assert res.grade == "substituted"
        schemas.validate(REQ, "RecordRestoreRequestV0", t.requests[-1])
        assert t.requests[-1]["backend_keyed"] is True
        assert (res.dest / "evaluator" / "ev_mod.py").read_text() == EV_SRC
        # editing the helper file after declaration: not reproduced, partial
        fp = pillar.fingerprint
        (root / "ev_mod.py").write_text(EV_SRC + "# edit\n")
        changed = export_pillar_of(
            EvaluatorFilesSource(
                root, evaluator=ev, objectives=objectives, declared_fingerprint=fp
            )
        )
        assert not changed.reproduced
        assert (
            build_recovery_manifest(changed).document["coverage"]["status"] == "partial"
        )
    finally:
        sys.modules.pop("ev_mod_x", None)


def export_pillar_of(src):
    from traigent.legend import export_pillar

    return export_pillar(src)


def test_conformance_artifacts_use_reserved_namespace(tmp_path):
    seen: list[str] = []

    class Spy(LocalDirectoryStore):
        def put(self, kind, fp, stream):
            seen.append(fp)
            return super().put(kind, fp, stream)

    assert conformance_check(Spy(tmp_path / "s")) == PASSED
    assert seen and all(k.startswith(c.CONFORMANCE_KEY_PREFIX) for k in seen)
    assert not any(k.startswith(c.MANIFEST_FINGERPRINT_PREFIX) for k in seen)


def test_search_space_restore_never_exact_client_side(store, tmp_path, idx, schemas):
    t = InMemoryTransport()
    ref = declare(c.KIND_SEARCH_SPACE, SearchSpaceSource(SSP, {"t": [1]}), transport=t)
    rec = capture(
        ref.version_ref,
        SearchSpaceSource(SSP, {"t": [1]}),
        store=store,
        transport=t,
        conformance=PASSED,
        index=idx,
        identity_verification_ref=PROOF,
    )
    res = restore(rec, tmp_path / "o", store=store, transport=t, index=idx)
    assert res.grade == "substituted"
    schemas.validate(REQ, "RecordRestoreRequestV0", t.requests[-1])
    assert t.requests[-1]["outcome"] == "substituted"


def test_capture_store_key_conflict_is_a_closed_error(tmp_path, idx):
    class Conflicting(LocalDirectoryStore):
        def put(self, kind, fp, stream):
            if not fp.startswith(c.CONFORMANCE_KEY_PREFIX):
                return "store_key_conflict"
            return super().put(kind, fp, stream)

    with pytest.raises(LegendError) as err:
        capture(
            "versionref_" + "e" * 32,
            ConfigurationSource({"a": 1}),
            store=Conflicting(tmp_path / "s"),
            index=idx,
        )
    assert err.value.code == "store_key_conflict"


# ------------------------------------------------- contract 9b0451d additions
def test_representation_commitment_matches_the_schema_construction():
    digests = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
    raw = (
        b"traigent.legend.representation_commitment.v0\x00"
        + json.dumps(
            {"representations": digests}, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    assert (
        representation_commitment(digests)
        == "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    # order is part of the commitment
    assert representation_commitment(digests[::-1]) != representation_commitment(
        digests
    )


def test_capture_request_carries_commitment_and_validates(store, idx, schemas):
    t = InMemoryTransport()
    ref = _vref(t)
    rec = capture(
        ref,
        ConfigurationSource({"seed": 1}),
        store=store,
        transport=t,
        conformance=PASSED,
        index=idx,
    )
    req = t.requests[-1]
    schemas.validate(REQ, "RecordCaptureRequestV0", req)
    assert req["backend_keyed"] is False and req["identity_verification_ref"] is None
    assert req["identity_digests_reproduced"] is True
    assert req["representation_commitment"] == rec.representation_commitment
    event = [e for e in t.events if e["event_type"] == "captured"][0]
    schemas.validate(PL, "CapturedEventV0", event)
    assert event["ledger_position"] > 0


def test_coverage_partition_is_validated_per_kind():
    ok = {"status": "complete", "covered": ["applied_config"], "missing": []}
    validate_coverage(c.KIND_CONFIGURATION, ok)
    cases = {
        "overlap": (
            {
                "status": "partial",
                "covered": ["applied_config"],
                "missing": ["applied_config"],
            },
            "invalid_request",
        ),
        "status": (
            {"status": "complete", "covered": [], "missing": ["applied_config"]},
            "invalid_request",
        ),
        "foreign": (
            {
                "status": "complete",
                "covered": ["applied_config", "code"],
                "missing": [],
            },
            "invalid_request",
        ),
        "gap": (
            {"status": "complete", "covered": [], "missing": []},
            "coverage_insufficient",
        ),
    }
    for name, (cov, code) in cases.items():
        with pytest.raises(LegendError) as err:
            validate_coverage(c.KIND_CONFIGURATION, cov)
        assert err.value.code == code, name


def test_exact_restore_records_measurements_and_artifact_digest(
    store, tmp_path, idx, schemas
):
    t = InMemoryTransport()
    ref = _vref(t)
    rec = capture(
        ref,
        ConfigurationSource({"seed": 1}),
        store=store,
        transport=t,
        conformance=PASSED,
        index=idx,
    )
    res = restore(rec, tmp_path / "o", store=store, transport=t, index=idx)
    req = t.requests[-1]
    schemas.validate(REQ, "RecordRestoreRequestV0", req)
    assert (
        res.grade == "exact"
        and req["verified_artifact_digest"] == res.verified_artifact_digest
    )
    assert req["historical_key_available"] and req["dependencies_verified"]
    event = [e for e in t.events if e["event_type"] == "restore_verified"][0]
    schemas.validate(PL, "RestoreVerifiedEventV0", event)
    # a substituted staged artifact is refused at activation (manifest_mismatch)
    (res.dest / "00-config_document").write_bytes(b"{}")
    with pytest.raises(LegendError) as err:
        activate(res, tmp_path / "live")
    assert err.value.code == "manifest_mismatch"


def test_availability_request_validates_and_is_closed(store, idx, schemas):
    t = InMemoryTransport()
    ref = _vref(t)
    rec = capture(
        ref,
        ConfigurationSource({"seed": 1}),
        store=store,
        transport=t,
        conformance=PASSED,
        index=idx,
    )
    ev = record_availability(ref, rec.manifest_fingerprint, "missing", transport=t)
    schemas.validate(REQ, "RecordAvailabilityRequestV0", t.requests[-1])
    schemas.validate(REQ, None, t.requests[-1])
    assert ev is not None
    for bad in ("Missing", "gone", None):
        with pytest.raises(LegendError):
            record_availability(ref, rec.manifest_fingerprint, bad, transport=t)
    assert record_availability(ref, rec.manifest_fingerprint, "available") is None


def test_vendored_contract_files_equal_schema_9d71d82():
    schema = Path(
        "/home/nimrod/TraigentProjects/core_project/worktrees/TraigentSchema/"
        "pillar-legend/traigent_schema/schemas/legend"
    )
    head = (
        subprocess.run(
            ["git", "-C", str(schema), "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        if schema.is_dir()
        else ""
    )
    assert head in ("", "9d71d82") or head.startswith("9d71d8"), head
    if not schema.is_dir():
        pytest.skip("Schema worktree not present")
    fixtures = Path(__file__).parent / "fixtures"
    for f in fixtures.glob("*.json"):
        assert f.read_bytes() == (schema / f.name).read_bytes(), f.name


# ------------------------------------------------- review fixes (terra)
def _plain_evaluator(tmp_path):
    root, ev, objectives = _evaluator_project(tmp_path)
    return root, ev, objectives


def test_plain_evaluator_is_not_backend_keyed_and_restores_exact(
    store, tmp_path, idx, schemas
):
    root, ev, objectives = _plain_evaluator(tmp_path)
    try:
        src = EvaluatorFilesSource(
            root, evaluator=ev, objectives=objectives, authoritative_format="plain"
        )
        t = InMemoryTransport()
        vref = declare(
            c.KIND_EVALUATOR,
            EvaluatorSource(ev, objectives, authoritative_format="plain"),
            transport=t,
        ).version_ref
        rec = capture(
            vref, src, store=store, transport=t, conformance=PASSED, index=idx
        )
        assert rec.backend_keyed is False and rec.coverage["status"] == "complete"
        schemas.validate(REQ, "RecordCaptureRequestV0", t.requests[-1])
        res = restore(rec, tmp_path / "o", store=store, transport=t, index=idx)
        assert res.grade == "exact"
        schemas.validate(REQ, "RecordRestoreRequestV0", t.requests[-1])
        assert t.requests[-1]["backend_keyed"] is False
        # keyed and unknown formats are backend-keyed
        for fmt in ("keyed", "unknown"):
            assert c.backend_keyed(c.KIND_EVALUATOR, fmt)
        assert not c.backend_keyed(c.KIND_EVALUATOR, "plain")
        with pytest.raises(LegendError):
            compute_pillar(
                c.KIND_EVALUATOR,
                EvaluatorSource(ev, objectives, authoritative_format="x"),
            )
    finally:
        sys.modules.pop("ev_mod_x", None)


def test_builders_refuse_unknown_inputs_and_outputs_validate(schemas):
    from traigent.legend.payloads import (
        build_availability_payload,
        build_capture_payload,
        build_restore_payload,
    )

    vref, fp = "versionref_" + "1" * 32, "rmf0:sha256:" + "2" * 64
    cov = {"status": "complete", "covered": ["applied_config"], "missing": []}
    cap = {
        "version_ref": vref,
        "kind": c.KIND_CONFIGURATION,
        "manifest_fingerprint": fp,
        "coverage": cov,
        "recovery_state": "stored_unverified",
        "conformance_passed": True,
        "exporter_version": "0.1.0",
        "backend_keyed": False,
        "identity_verification_ref": None,
        "representation_commitment": "sha256:" + "3" * 64,
        "identity_digests_reproduced": True,
    }
    rst = {
        "version_ref": vref,
        "kind": c.KIND_CONFIGURATION,
        "manifest_fingerprint": fp,
        "outcome": "exact",
        "coverage": cov,
        "isolated_destination": True,
        "representation_digests_match": True,
        "pillar_fingerprint_match": True,
        "historical_key_available": True,
        "dependencies_verified": True,
        "verified_artifact_digest": "sha256:" + "4" * 64,
        "backend_keyed": False,
    }
    schemas.validate(REQ, "RecordCaptureRequestV0", build_capture_payload(**cap))
    schemas.validate(REQ, "RecordRestoreRequestV0", build_restore_payload(**rst))
    schemas.validate(
        REQ,
        "RecordAvailabilityRequestV0",
        build_availability_payload(vref, fp, "retired"),
    )
    bad_cov = {**cov, "extra": ["x"]}
    for override in (
        {"coverage": bad_cov},
        {"coverage": {**cov, "covered": ["applied_config", "bogus"]}},
        {"exporter_version": "1.0"},
        {"exporter_version": "v1.0.0"},
        {"exporter_version": "1.0.0\n"},
    ):
        with pytest.raises(LegendError):
            build_capture_payload(**{**cap, **override})
    for override in (
        {"coverage": bad_cov},
        {"isolated_destination": False},
        {"outcome": "great"},
    ):
        with pytest.raises(LegendError):
            build_restore_payload(**{**rst, **override})
    with pytest.raises(LegendError):
        build_restore_payload(
            **{**rst, "isolated_destination": False, "outcome": "partial"}
        )
    with pytest.raises(LegendError):
        build_availability_payload(vref, fp, "gone")
    with pytest.raises(LegendError):
        build_capture_payload(
            **{**cap, "kind": c.KIND_SEARCH_SPACE, "backend_keyed": False}
        )
