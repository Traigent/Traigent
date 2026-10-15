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
    HttpLegendTransport,
    InMemoryTransport,
    LegendError,
    SearchSpaceSource,
    build_recovery_manifest,
    compute_pillar,
    declare,
    query,
)
from traigent.legend import _contract as c

PL = "pillar_legend_v0.json"
RM = "recovery_manifest_v0.json"
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
    instruction = {
        **t.requests[-1],
        "instruction_ref": "instructionref_" + "0" * 32,
        "operation": "query_versions",
    }
    schemas.validate(
        "director_legend_ops_v0.json", "QueryVersionsInstruction", instruction
    )


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
    assert t.requests[1]["idempotency_key"] == t.requests[2]["idempotency_key"]
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
    forged = {**t.requests[0], "change_reason": "code"}
    with pytest.raises(LegendError) as err:
        t.declare(forged)
    assert err.value.code == "conflicting_idempotency_payload"


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


def test_http_transport_is_off_by_default(monkeypatch):
    monkeypatch.delenv(c.LEGEND_ENABLED_ENV, raising=False)
    calls = []
    http = HttpLegendTransport(lambda path, payload: calls.append(path) or {})
    with pytest.raises(LegendError) as err:
        declare(c.KIND_CONFIGURATION, ConfigurationSource({}), transport=http)
    assert err.value.code == "flag_disabled" and calls == []
    monkeypatch.setenv(c.LEGEND_ENABLED_ENV, "1")
    ref = "versionref_" + "1" * 32
    ok = HttpLegendTransport(
        lambda p, b: {
            "version_ref": ref,
            "entity_ref": "entityref_" + "2" * 32,
            "created": True,
            "event_ref": "eventref_" + "3" * 32,
        }
    )
    assert (
        declare(c.KIND_CONFIGURATION, ConfigurationSource({}), transport=ok).version_ref
        == ref
    )
