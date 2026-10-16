"""Pillar Legend client: payload validity, privacy canaries, idempotency, revert, manifests, aliases."""

from __future__ import annotations

import json

import pytest

from tests.unit.legend.conftest import make_dataset
from traigent.legend import (
    AgentManifestSource,
    AgentSource,
    AliasMap,
    ConfigurationSource,
    DatasetSource,
    EvaluatorManifestSource,
    InMemoryTransport,
    LegendError,
    SearchSpaceSource,
    build_recovery_manifest,
    compute_pillar,
    declare,
    query,
)
from traigent.legend import _contract as c
from traigent.legend import payloads

PL = "pillar_legend_v0.json"
RM = "recovery_manifest_v0.json"
REQ = "legend_requests_v0.json"
CANARY = "CANARY_name_Alice_has_diabetes"
SSP = "ssp1:k" + "cd" * 8 + ":" + "ab" * 32
DIGEST = "sha256:" + "11" * 32


def _agent_fn():  # noqa: ANN202 - fixture function whose source is hashed
    return 1


def _evaluator_manifest() -> dict:
    return {
        "manifest_version": 1,
        "evaluator_id": "ev_1",
        "code_digest": DIGEST,
        "config_digest": DIGEST,
        "helper_digests": {},
        "judge": None,
        "objectives": [{"name": "accuracy", "orientation": "maximize", "weight": 1}],
        "dependency_versions": {},
    }


def _agent_manifest() -> dict:
    return {
        "manifest_version": 1,
        "agent_id": "agent_1",
        "runtime": {"language": "python", "sdk_version": "1.0.0"},
        "source_digest": DIGEST,
        "asset_digests": {"helper_modules": {}, "prompts": {}, "tool_definitions": {}},
        "applied_config_digest": DIGEST,
        "coverage": "partial",
    }


def sources(keys):
    return {
        c.KIND_AGENT: AgentManifestSource(_agent_manifest()),
        c.KIND_CONFIGURATION: ConfigurationSource({"temperature": 0.2}),
        c.KIND_DATASET: DatasetSource(make_dataset([("a", 1), ("b", 2)]), keys),
        c.KIND_EVALUATOR: EvaluatorManifestSource(_evaluator_manifest()),
        c.KIND_SEARCH_SPACE: SearchSpaceSource(SSP, {"temperature": [0.1, 0.2]}),
    }


@pytest.mark.parametrize("kind", sorted(c.ENTITY_KINDS))
def test_declaration_validates_against_contract(kind, keys, schemas):
    t = InMemoryTransport()
    result = declare(kind, sources(keys)[kind], transport=t)
    payload = t.requests[0]
    version = t.versions[result.version_ref]
    schemas.validate(PL, "LegendVersionV0", version)
    schemas.validate(PL, "DeclaredEventV0", t.events[0])
    schemas.validate(REQ, "DeclareRequestV0", payload)
    props = schemas.docs[PL]["definitions"]["DeclaredEventV0"]["properties"]
    # every client-supplied event field is a contract field (closed shape)
    event_fields = set(payload) - {
        "entity_ref",
        "kind",
        "fingerprint_scheme",
        "key_id",
        "fingerprint",
        "initial_parent_version_ref",
    }
    assert event_fields <= set(props)
    assert payload["fingerprint_scheme"] == c.SCHEME_BY_KIND[kind]


def test_agent_callable_source_is_fingerprinted(schemas):
    t = InMemoryTransport()
    r = declare(c.KIND_AGENT, AgentSource(_agent_fn, "agent_1", {"x": 1}), transport=t)
    schemas.validate(PL, "LegendVersionV0", t.versions[r.version_ref])


def test_query_payload_matches_contract(keys, schemas):
    t = InMemoryTransport()
    r = declare(c.KIND_CONFIGURATION, ConfigurationSource({"a": 1}), transport=t)
    page = query(r.entity_ref, page_size=10, transport=t)
    assert page.version_refs == (r.version_ref,)
    schemas.validate(REQ, "QueryRequestV0", t.requests[-1])
    assert t.requests[-1]["kind"] is None and "instruction_ref" not in t.requests[-1]
    narrowed = query(r.entity_ref, kind=c.KIND_AGENT, transport=t)
    assert narrowed.version_refs == ()


def test_query_as_of_is_validated_sent_and_kept_across_pages(keys, schemas):
    t = InMemoryTransport()
    r = declare(c.KIND_CONFIGURATION, ConfigurationSource({"a": 1}), transport=t)
    declare(c.KIND_CONFIGURATION, ConfigurationSource({"a": 2}), transport=t)
    query(r.entity_ref, transport=t)
    assert "as_of_ledger_position" not in t.requests[-1]
    p1 = query(r.entity_ref, page_size=1, as_of_ledger_position=7, transport=t)
    schemas.validate(REQ, "QueryRequestV0", t.requests[-1])
    assert t.requests[-1]["as_of_ledger_position"] == 7
    if p1.next_page_cursor:
        query(
            r.entity_ref,
            page_size=1,
            page_cursor=p1.next_page_cursor,
            as_of_ledger_position=7,
            transport=t,
        )
        assert t.requests[-1]["as_of_ledger_position"] == 7
        assert t.requests[-1]["page_cursor"] == p1.next_page_cursor
    for bad in (-1, 1.5, "3", True):
        with pytest.raises(LegendError):
            query(r.entity_ref, as_of_ledger_position=bad, transport=t)
    query(r.entity_ref, as_of_ledger_position=0, transport=t)
    assert t.requests[-1]["as_of_ledger_position"] == 0


def test_no_names_or_paths_in_any_payload(keys, tmp_path):
    t = InMemoryTransport()
    ds = make_dataset([(CANARY, 1)])
    ds.name = CANARY
    ds.description = CANARY
    alias = AliasMap(tmp_path / "aliases.json")
    for kind, src in {**sources(keys), c.KIND_DATASET: DatasetSource(ds, keys)}.items():
        r = declare(kind, src, transport=t)
        alias.set(CANARY, kind, r.entity_ref)
    query(r.entity_ref, transport=t)
    blob = json.dumps(t.requests) + json.dumps(t.events) + json.dumps(t.versions)
    assert CANARY not in blob
    assert str(tmp_path) not in blob and "aliases.json" not in blob


def test_redeclare_is_idempotent(keys):
    t = InMemoryTransport()
    first = declare(c.KIND_DATASET, sources(keys)[c.KIND_DATASET], transport=t)
    again = declare(
        c.KIND_DATASET,
        sources(keys)[c.KIND_DATASET],
        entity=first.entity_ref,
        transport=t,
    )
    again2 = declare(
        c.KIND_DATASET,
        sources(keys)[c.KIND_DATASET],
        entity=first.entity_ref,
        transport=t,
    )
    assert first.version_ref == again.version_ref == again2.version_ref
    assert len(t.versions) == 1
    assert t.requests[1]["fingerprint"] == t.requests[0]["fingerprint"]
    # a fresh key per LOGICAL request; the digest (which excludes the key) is stable
    keys_ = {r["idempotency_key"] for r in t.requests}
    assert len(keys_) == 3
    assert t.requests[1]["request_digest"] == t.requests[2]["request_digest"]
    for r in t.requests:
        assert c.IDEMPOTENCY_KEY_RE.fullmatch(r["idempotency_key"])
    assert not again.created and first.created


def test_revert_declaration_payload(keys, schemas):
    t = InMemoryTransport()
    v1 = declare(c.KIND_CONFIGURATION, ConfigurationSource({"t": 1}), transport=t)
    v2 = declare(
        c.KIND_CONFIGURATION,
        ConfigurationSource({"t": 2}),
        entity=v1.entity_ref,
        previous_version=v1.version_ref,
        change_reason="configuration",
        transport=t,
    )
    v3 = declare(
        c.KIND_CONFIGURATION,
        ConfigurationSource({"t": 1}),
        entity=v1.entity_ref,
        previous_version=v2.version_ref,
        change_reason="rollback",
        transport=t,
    )
    assert v3.version_ref == v1.version_ref and not v3.created
    payload = t.requests[2]
    assert payload["previous_version_ref"] == v2.version_ref
    assert payload["change_reason"] == "rollback"
    revert_event = t.events[2]
    schemas.validate(PL, "DeclaredEventV0", revert_event)
    assert revert_event["previous_version_ref"] == v2.version_ref
    assert len(t.versions) == 2


def test_bad_refs_and_reasons_refused():
    cfg = ConfigurationSource({})
    for kwargs in (
        {"entity": "my-agent"},
        {"previous_version": "v1"},
        {"change_reason": "because"},
    ):
        with pytest.raises(LegendError):
            declare(c.KIND_CONFIGURATION, cfg, transport=InMemoryTransport(), **kwargs)
    with pytest.raises(LegendError):
        declare(
            c.KIND_AGENT, cfg, transport=InMemoryTransport()
        )  # wrong source for kind
    with pytest.raises(LegendError):
        declare(
            c.KIND_SEARCH_SPACE,
            SearchSpaceSource("not-a-ref"),
            transport=InMemoryTransport(),
        )


def test_conflicting_idempotency_payload_refused():
    t = InMemoryTransport()
    declare(c.KIND_CONFIGURATION, ConfigurationSource({"a": 1}), transport=t)
    first = t.requests[0]
    forged = payloads.seal(
        {
            k: v
            for k, v in {**first, "change_reason": "code"}.items()
            if k not in ("idempotency_key", "request_digest")
        },
        first["idempotency_key"],
    )
    with pytest.raises(LegendError) as err:
        t.declare(forged)
    assert err.value.code == "conflicting_idempotency_payload"
    # a body whose digest does not match is malformed, not a conflict
    with pytest.raises(LegendError) as bad:
        t.declare({**first, "change_reason": "code"})
    assert bad.value.code == "invalid_request"


class _Flaky(InMemoryTransport):
    def __init__(self) -> None:
        super().__init__()
        self.fail = 2

    def declare(self, payload):
        if self.fail:
            self.fail -= 1
            self.requests.append(dict(payload))
            raise LegendError("store_unavailable")
        return super().declare(payload)


def test_retries_reuse_key_and_digest_and_append_once():
    t = _Flaky()
    declare(c.KIND_CONFIGURATION, ConfigurationSource({"a": 1}), transport=t)
    assert len(t.requests) == 3 and len(t.events) == 1
    assert len({r["idempotency_key"] for r in t.requests}) == 1
    assert len({r["request_digest"] for r in t.requests}) == 1


def test_request_digest_excludes_key_and_digest():
    body = {"schema_version": "0", "x": 1}
    a, b = payloads.seal(body), payloads.seal(body)
    assert a["idempotency_key"] != b["idempotency_key"]
    assert a["request_digest"] == b["request_digest"] == payloads.request_digest(body)


def test_initial_parent_only_on_first_declaration_and_independent_of_previous():
    t = InMemoryTransport()
    v1 = declare(c.KIND_CONFIGURATION, ConfigurationSource({"t": 1}), transport=t)
    v2 = declare(
        c.KIND_CONFIGURATION,
        ConfigurationSource({"t": 2}),
        entity=v1.entity_ref,
        initial_parent=v1.version_ref,
        transport=t,
    )
    assert t.versions[v2.version_ref]["initial_parent_version_ref"] == v1.version_ref
    assert "previous_version_ref" not in t.requests[-1]
    # revert v2 -> v1 with previous set, initial parent null: legal, parent unchanged
    declare(
        c.KIND_CONFIGURATION,
        ConfigurationSource({"t": 2}),
        entity=v1.entity_ref,
        previous_version=v1.version_ref,
        transport=t,
    )
    assert t.versions[v2.version_ref]["initial_parent_version_ref"] == v1.version_ref
    with pytest.raises(LegendError) as err:  # a different non-null parent is refused
        declare(
            c.KIND_CONFIGURATION,
            ConfigurationSource({"t": 2}),
            entity=v1.entity_ref,
            initial_parent=v2.version_ref,
            transport=t,
        )
    assert err.value.code == "invalid_request"


def test_dataset_without_grant_refuses_to_mint():
    with pytest.raises(LegendError):
        compute_pillar(c.KIND_DATASET, DatasetSource(make_dataset([("a", 1)])))


@pytest.mark.parametrize("kind", sorted(c.ENTITY_KINDS))
def test_manifest_validates_and_fingerprint_format(kind, keys, schemas):
    pillar = compute_pillar(kind, sources(keys)[kind])
    m = build_recovery_manifest(pillar)
    schemas.validate(RM, "RecoveryManifestV0", m.document)
    schemas.validate(PL, "ManifestFingerprint", m.fingerprint)
    assert m.fingerprint.startswith("rmf0:sha256:")
    assert [r["ordinal"] for r in m.document["representations"]] == list(
        range(len(m.document["representations"]))
    )
    assert m.document["pillar_fingerprint"] == pillar.fingerprint
    assert build_recovery_manifest(pillar).fingerprint == m.fingerprint


def test_reordered_rows_same_root_different_manifest(keys, schemas):
    rows = [("a", 1), ("b", 2), ("a", 1), ("c", 3)]
    p1 = compute_pillar(c.KIND_DATASET, DatasetSource(make_dataset(rows), keys))
    p2 = compute_pillar(
        c.KIND_DATASET, DatasetSource(make_dataset(list(reversed(rows))), keys)
    )
    assert p1.fingerprint == p2.fingerprint  # order-free root
    m1, m2 = build_recovery_manifest(p1), build_recovery_manifest(p2)
    assert m1.fingerprint != m2.fingerprint
    assert (
        m1.document["representations"][0]["digest"]
        != m2.document["representations"][0]["digest"]
    )
    assert m1.document["ordering"] == "preserved"
    assert m1.document["coverage"]["status"] == "complete"
    schemas.validate(RM, "RecoveryManifestV0", m2.document)


def test_dataset_multiplicity_kept_in_manifest(keys):
    once = compute_pillar(c.KIND_DATASET, DatasetSource(make_dataset([("a", 1)]), keys))
    twice = compute_pillar(
        c.KIND_DATASET, DatasetSource(make_dataset([("a", 1), ("a", 1)]), keys)
    )
    assert (
        build_recovery_manifest(once).fingerprint
        != build_recovery_manifest(twice).fingerprint
    )


def test_partial_coverage_is_reported(keys, schemas):
    m = build_recovery_manifest(
        compute_pillar(c.KIND_EVALUATOR, sources(keys)[c.KIND_EVALUATOR])
    )
    assert m.document["coverage"]["status"] == "partial"
    assert "evaluator_code" in m.document["coverage"]["missing"]
    assert not schemas.is_valid(RM, "RecoveryManifestV0", {**m.document, "extra": 1})


def test_manifest_dependencies_are_refs_only(keys):
    p = compute_pillar(c.KIND_CONFIGURATION, ConfigurationSource({}))
    ref = "versionref_" + "0" * 32
    m = build_recovery_manifest(p, dependencies=[(ref, c.KIND_DATASET)])
    assert m.document["dependencies"] == [{"version_ref": ref, "kind": c.KIND_DATASET}]
    with pytest.raises(LegendError):
        build_recovery_manifest(p, dependencies=[("my-dataset", c.KIND_DATASET)])


def test_alias_round_trip(tmp_path):
    path = tmp_path / ".traigent" / "legend" / "aliases.json"
    a = AliasMap(path)
    ref = "entityref_" + "a" * 32
    a.set("prod agent", c.KIND_AGENT, ref)
    b = AliasMap(path)
    assert b.resolve("prod agent", c.KIND_AGENT) == ref
    assert b.resolve("prod agent", c.KIND_EVALUATOR) is None
    assert b.names(c.KIND_AGENT) == ["prod agent"]
    assert (
        b.remove("prod agent", c.KIND_AGENT)
        and AliasMap(path).names(c.KIND_AGENT) == []
    )
    with pytest.raises(LegendError):
        a.set("x", c.KIND_AGENT, "not-a-ref")


def test_store_failed_publish_leaves_nothing_visible(tmp_path, monkeypatch):
    import io
    import os

    from traigent.legend import LocalDirectoryStore

    store = LocalDirectoryStore(tmp_path / "s")
    fp = "rmf0:sha256:" + "ab" * 32
    real_link = os.link

    def crash(*_a, **_k):
        raise OSError("simulated crash at the publish step")

    monkeypatch.setattr(os, "link", crash)
    assert store.put(c.KIND_AGENT, fp, io.BytesIO(b"payload")) == "unreachable"
    monkeypatch.setattr(os, "link", real_link)
    # no entry, no header, no temp file: nothing of the failed put is visible
    assert [p for p in (tmp_path / "s").rglob("*") if p.is_file()] == []
    assert store.has(c.KIND_AGENT, fp) == "missing"
    assert store.get(c.KIND_AGENT, fp, tmp_path / "out") == "missing"
    # the retry, even with different bytes, is a clean first write
    assert store.put(c.KIND_AGENT, fp, io.BytesIO(b"other")) == "stored_unverified"
    assert store.get(c.KIND_AGENT, fp, tmp_path / "out") == "retrieved"
    assert store.put(c.KIND_AGENT, fp, io.BytesIO(b"payload")) == "store_key_conflict"


def test_store_concurrent_different_byte_puts_have_one_consistent_winner(tmp_path):
    import io
    import threading

    from traigent.legend import LocalDirectoryStore
    from traigent.legend.store import BUNDLE_NAME

    store = LocalDirectoryStore(tmp_path / "s")
    for n in range(150):
        fp = "rmf0:sha256:" + f"{n:064x}"
        barrier = threading.Barrier(2)
        out: dict[str, str] = {}

        def writer(
            tag: str,
            fp: str = fp,
            barrier: threading.Barrier = barrier,
            out: dict[str, str] = out,
        ) -> None:
            barrier.wait()
            out[tag] = store.put(c.KIND_AGENT, fp, io.BytesIO(tag.encode() * 64))

        threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert sorted(out.values()) == ["store_key_conflict", "stored_unverified"], out
        dest = tmp_path / f"o{n}"
        # the stored entry is exactly one writer's bytes, never a mix: it reads back clean
        assert store.get(c.KIND_AGENT, fp, dest) == "retrieved"
        winner = [t for t, st in out.items() if st == "stored_unverified"][0]
        assert (dest / BUNDLE_NAME).read_bytes() == winner.encode() * 64


def test_source_bundle_skips_sdk_runtime_dirs_and_agent_source_refuses_clearly(
    tmp_path,
):
    from traigent.legend import AgentSource
    from traigent.legend.exporters import _list_files, export

    root = tmp_path / "agent"
    (root / ".traigent" / "optimization_logs").mkdir(parents=True)
    (root / ".traigent" / "optimization_logs" / "run.json").write_text("{}")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "x").write_text("x")
    (root / "main.py").write_text("print(1)\n")
    assert _list_files(root) == [
        "main.py"
    ]  # non-git root: run logs never enter the bundle
    (root / ".traigent" / "optimization_logs" / "run2.json").write_text("{}")
    assert _list_files(root) == ["main.py"]  # and a new log does not move the file set

    with pytest.raises(LegendError) as err:
        export(AgentSource(lambda q: q, "agent_x", {}))
    assert err.value.code == "coverage_insufficient"
    assert "AgentStateSource" in str(err.value)
