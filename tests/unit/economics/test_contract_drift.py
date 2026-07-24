"""Contract-drift guard: SDK constants + emitted payloads vs the local Schema.

These tests are the reason the SDK may declare its own contract constants
without that becoming a competing contract: every value is bound to the
authoritative TraigentSchema economics schemas in the local worktree, and real
emitted batches are validated against the request schema. A schema change the
SDK does not track fails here.
"""

from __future__ import annotations

from traigent.economics import build_telemetry_request, funnel_eligible_event
from traigent.economics.contract import (
    CHARACTERIZATION_FIELD_NAMES,
    CONTRACT_ID,
    CONTRACT_VERSION,
    EVENT_TYPES,
    EVIDENCE_STATUSES,
    FUNNEL_STAGES,
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENCY_KEY_PATTERN,
    MAX_BATCH_EVENTS,
    PROVENANCE_VALUES,
    REJECTION_REASONS,
    SOURCE_KIND,
    SHARING_OUTCOMES,
    TELEMETRY_ENDPOINT,
)

_REQUEST_SCHEMA = "economics_telemetry_ingest_request_schema"


def _defs(local_contract, schema_name: str) -> dict:
    return local_contract.schema(schema_name)["definitions"]


def test_contract_id_and_version_match_schema(local_contract) -> None:
    common = _defs(local_contract, "economics_common_schema")
    assert CONTRACT_ID == common["ContractId"]["const"]
    assert CONTRACT_VERSION == common["ContractVersion"]["const"]


def test_source_kind_is_a_schema_member(local_contract) -> None:
    common = _defs(local_contract, "economics_common_schema")
    assert SOURCE_KIND in common["SourceKind"]["enum"]


def test_provenance_sharing_evidence_enums_match(local_contract) -> None:
    common = _defs(local_contract, "economics_common_schema")
    assert set(PROVENANCE_VALUES) == set(common["Provenance"]["enum"])
    assert set(SHARING_OUTCOMES) == set(common["SharingOutcome"]["enum"])
    assert set(EVIDENCE_STATUSES) == set(common["EvidenceStatus"]["enum"])


def test_event_types_match_the_closed_union(local_contract) -> None:
    funnel = local_contract.schema("economics_funnel_event_schema")
    run = local_contract.schema("economics_run_event_schema")
    receipt = local_contract.schema("economics_receipt_event_schema")
    schema_types = {
        funnel["properties"]["event_type"]["const"],
        run["properties"]["event_type"]["const"],
        receipt["properties"]["event_type"]["const"],
    }
    assert set(EVENT_TYPES) == schema_types


def test_funnel_stages_match_schema_order(local_contract) -> None:
    funnel = local_contract.schema("economics_funnel_event_schema")
    assert list(FUNNEL_STAGES) == funnel["properties"]["stage"]["enum"]


def test_characterization_allowlist_matches_vocabulary(local_contract) -> None:
    vocab = _defs(local_contract, "economics_characterization_vocabulary_schema")
    assert set(CHARACTERIZATION_FIELD_NAMES) == set(
        vocab["CharacterizationFieldName"]["enum"]
    )


def test_rejection_reasons_match_response_schema(local_contract) -> None:
    response = local_contract.schema("economics_telemetry_ingest_response_schema")
    reason_enum = response["properties"]["rejections"]["items"]["properties"]["reason"][
        "enum"
    ]
    assert set(REJECTION_REASONS) == set(reason_enum)


def test_batch_limit_matches_schema(local_contract) -> None:
    request = local_contract.schema(_REQUEST_SCHEMA)
    assert MAX_BATCH_EVENTS == request["properties"]["events"]["maxItems"]


def test_idempotency_key_grammar_matches_schema(local_contract) -> None:
    common = _defs(local_contract, "economics_common_schema")
    assert IDEMPOTENCY_KEY_PATTERN == common["IdempotencyKey"]["pattern"]


def test_endpoint_and_idempotency_header_match_schema(local_contract) -> None:
    endpoints = local_contract.endpoints()
    paths = endpoints["paths"]
    assert TELEMETRY_ENDPOINT in paths
    post = paths[TELEMETRY_ENDPOINT]["post"]
    header_params = [p for p in post["parameters"] if p["in"] == "header"]
    names = {p["name"] for p in header_params}
    assert IDEMPOTENCY_KEY_HEADER in names
    idem = next(p for p in header_params if p["name"] == IDEMPOTENCY_KEY_HEADER)
    assert idem["required"] is True
    assert idem["schema"]["pattern"] == IDEMPOTENCY_KEY_PATTERN


def test_replay_status_bindings_match_schema(local_contract) -> None:
    initial = local_contract.schema(
        "economics_telemetry_ingest_response_initial_schema"
    )
    replay = local_contract.schema("economics_telemetry_ingest_response_replay_schema")
    # 201 initial ingest binds replayed=false; 200 replay binds replayed=true.
    assert initial["properties"]["replayed"]["const"] is False
    assert replay["properties"]["replayed"]["const"] is True

    endpoints = local_contract.endpoints()
    responses = endpoints["paths"][TELEMETRY_ENDPOINT]["post"]["responses"]
    ok_ref = responses["200"]["content"]["application/json"]["schema"]["$ref"]
    created_ref = responses["201"]["content"]["application/json"]["schema"]["$ref"]
    assert ok_ref.endswith("economics_telemetry_ingest_response_replay_schema.json")
    assert created_ref.endswith(
        "economics_telemetry_ingest_response_initial_schema.json"
    )


def _response_ref_stem(response_entry: dict) -> str | None:
    ref = (
        response_entry.get("content", {})
        .get("application/json", {})
        .get("schema", {})
        .get("$ref")
    )
    if ref is None:
        return None
    return ref.rsplit("/", 1)[-1].removesuffix(".json")


def test_response_status_schema_map_matches_endpoint_bindings(local_contract) -> None:
    from traigent.economics import schema as schema_mod

    endpoints = local_contract.endpoints()
    responses = endpoints["paths"][TELEMETRY_ENDPOINT]["post"]["responses"]

    # 200 and 201 bind directly to the endpoint's response $ref.
    for status, expected_name in ((200, "replay"), (201, "initial")):
        stem = _response_ref_stem(responses[str(status)])
        mapped = schema_mod.RESPONSE_SCHEMA_BY_STATUS[status]
        assert stem == mapped
        assert expected_name in mapped

    # 422 is bound to the endpoint too, not a literal: the endpoint MUST declare
    # a 422, and the SDK's 422 schema must track it. Today the endpoint declares
    # 422 as a bare all-rejected status with no body $ref, so the SDK validates
    # its body against the same replayed=false initial shape the 201 uses; if the
    # Schema later gives 422 its own body $ref, this binds to it and fails on drift.
    entry_422 = responses["422"]  # KeyError here fails the test if 422 is dropped.
    ref_stem_422 = _response_ref_stem(entry_422)
    if ref_stem_422 is not None:
        assert schema_mod.RESPONSE_SCHEMA_BY_STATUS[422] == ref_stem_422
    else:
        assert (
            schema_mod.RESPONSE_SCHEMA_BY_STATUS[422]
            == schema_mod.RESPONSE_SCHEMA_BY_STATUS[201]
        )
        assert "rejected" in entry_422["description"].lower()


def test_runtime_fingerprint_binds_to_local_01f3e2a2_material(local_contract) -> None:
    from traigent.economics import schema as schema_mod

    computed = schema_mod.compute_economics_schema_fingerprint(
        local_contract.economics_dir
    )
    assert computed == schema_mod.EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT


def test_eligible_funnel_batch_validates_against_request_schema(local_contract) -> None:
    event = funnel_eligible_event(
        "proj-1",
        event_id="evt-1",
        occurred_at="2026-07-18T10:00:00.000Z",
        occurred_in_environment="production",
    )
    body = build_telemetry_request([event])
    assert local_contract.validate(body, _REQUEST_SCHEMA) == []


def test_full_run_economics_batch_validates_against_request_schema(
    local_contract,
) -> None:
    run_event = {
        "event_type": "run_economics",
        "event_id": "evt-2",
        "occurred_at": "2026-07-18T10:00:00.000Z",
        "project_ref": "proj-1",
        "run_id": "run-1",
        "archetype": "solo_coding_builder",
        "characterization": {
            "bands": {"value_channel": "save_expert_time"},
            "field_reports": [
                {
                    "field": "value_channel",
                    "provenance": "asked",
                    "confidence": 1.0,
                    "sharing_outcome": "shared",
                },
                {
                    "field": "error_cost_band",
                    "provenance": "asked",
                    "confidence": 1.0,
                    "sharing_outcome": "withheld_by_policy",
                },
            ],
        },
        "budget": {"authored_by": "backend", "recommended_daily_usd": 5, "cap_usd": 50},
        "actual_spend_usd": 0.0,
        "usage": {"input_tokens": 0, "output_tokens": 0, "model_calls": 0},
        "model_prices": [
            {
                "model_id": "gpt-4o-mini",
                "input_usd_per_mtok": 0.15,
                "output_usd_per_mtok": 0.6,
                "price_source": "provider_published",
                "as_of": "2026-07-18T10:00:00.000Z",
            }
        ],
        "evidence_identity": {
            "baseline_run_id": "run-0",
            "candidate_run_id": "run-1",
            "dataset_hash": "a" * 64,
            "evaluator_version": "exec-match-v2",
            "objective_weights": [{"objective": "accuracy", "weight": 1.0}],
            "effect_estimate": {
                "estimate": 0.1,
                "lower": 0.05,
                "upper": 0.15,
                "level": 0.95,
                "unit": "proportion",
            },
            "support": {"n_examples": 100, "n_paired": 100},
            "exclusions": [],
        },
        "advisory": {
            "advice_id": "adv-1",
            "recommended_action": "run_optimization",
            "client_action": "followed",
        },
        "labor_proxies": {},
    }
    body = build_telemetry_request([run_event])
    assert local_contract.validate(body, _REQUEST_SCHEMA) == []


def test_withheld_value_present_would_be_rejected_by_schema(local_contract) -> None:
    # Sanity: the schema itself makes withheld+present unrepresentable, matching
    # the client-side egress guard. Built directly (bypassing the egress guard)
    # to assert the schema's own rule.
    payload = {
        "contract": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "batch_id": "batch-1",
        "idempotency_key": "econ-tel-abcdefgh",
        "sent_at": "2026-07-18T10:00:00.000Z",
        "source": {
            "kind": SOURCE_KIND,
            "name": "traigent-python-sdk",
            "version": "0.0.0",
        },
        "events": [
            {
                "event_type": "run_economics",
                "event_id": "evt-2",
                "occurred_at": "2026-07-18T10:00:00.000Z",
                "project_ref": "proj-1",
                "run_id": "run-1",
                "archetype": "solo_coding_builder",
                "characterization": {
                    "bands": {"error_cost_band": "not_measured"},
                    "field_reports": [
                        {
                            "field": "error_cost_band",
                            "provenance": "asked",
                            "confidence": 1.0,
                            "sharing_outcome": "withheld_by_policy",
                        }
                    ],
                },
                "budget": {
                    "authored_by": "backend",
                    "recommended_daily_usd": 5,
                    "cap_usd": 50,
                },
                "actual_spend_usd": 0.0,
                "usage": {"input_tokens": 0, "output_tokens": 0, "model_calls": 0},
                "model_prices": [
                    {
                        "model_id": "gpt-4o-mini",
                        "input_usd_per_mtok": 0.15,
                        "output_usd_per_mtok": 0.6,
                        "price_source": "provider_published",
                        "as_of": "2026-07-18T10:00:00.000Z",
                    }
                ],
                "evidence_identity": {
                    "baseline_run_id": "run-0",
                    "candidate_run_id": "run-1",
                    "dataset_hash": "a" * 64,
                    "evaluator_version": "exec-match-v2",
                    "objective_weights": [{"objective": "accuracy", "weight": 1.0}],
                    "effect_estimate": {
                        "estimate": 0.1,
                        "lower": 0.05,
                        "upper": 0.15,
                        "level": 0.95,
                        "unit": "proportion",
                    },
                    "support": {"n_examples": 100, "n_paired": 100},
                    "exclusions": [],
                },
                "advisory": {
                    "advice_id": "adv-1",
                    "recommended_action": "run_optimization",
                    "client_action": "followed",
                },
                "labor_proxies": {},
            }
        ],
    }
    assert local_contract.validate(payload, _REQUEST_SCHEMA) != []
