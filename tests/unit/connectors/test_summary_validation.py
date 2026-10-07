import copy
import json
from pathlib import Path

import pytest

from traigent.connectors.models import ConnectionRef
from traigent.connectors.privacy import (
    CustomerSideMinter,
    serialize_summary,
    validate_summary,
)

SCHEMAS = Path(__file__).parents[3] / "traigent" / "connectors" / "schemas"

GOLDEN = {
    "connector_run_summary.json": {
        "schema_version": "1",
        "run_token": "tk_0123456789abcdefghjkmnpqrs",
        "connector_kind": "langfuse",
        "connection_token": "tk_1111111111aaaaaaaaaabbbbbb",
        "command": "bootstrap_dataset",
        "status": "partial",
        "started_at": "2026-10-08T10:00:00Z",
        "finished_at": "2026-10-08T10:00:42Z",
        "counts": {
            "observations_read": 120,
            "scores_read": 80,
            "rows_dropped_invalid": 3,
            "pages": 4,
            "items_written": 0,
        },
        "guarantees": [
            {
                "operation": "read_scores",
                "support": "emulated",
                "reason": "emulated_client_side",
            }
        ],
        "error_code": "rate_limited",
    },
    "dataset_revision_summary.json": {
        "schema_version": "1",
        "dataset_token": "tk_2222222222cccccccccceeeeee",
        "revision": 2,
        "source_connector_kind": "langfuse",
        "item_count": 200,
        "holdout_count": 40,
        "approved": True,
        "approval_at": "2026-10-08T11:00:00Z",
        "sampling_policy": {"kind": "random", "fraction": 0.25, "seed": 7},
        "score_semantics": [
            {
                "score_token": "tk_3333333333ddddddddddffffff",
                "type": "numeric",
                "direction": "higher_better",
            }
        ],
    },
    "correlation_summary.json": {
        "schema_version": "1",
        "run_token": "tk_0123456789abcdefghjkmnpqrs",
        "trials_total": 10,
        "trials_linked": 8,
        "trials_unknown": 2,
        "tier_counts": {
            "exact": 5,
            "commit_name": 2,
            "name_only": 1,
            "deployment": 0,
            "ambiguous": 0,
        },
        "agent_function_ref": "my_pkg.agents:answer_question",
        "agent_file_path": "src/my_pkg/agents.py",
    },
}


def _sample_agent() -> None:
    return None


def test_validate_summary_round_trips_golden_examples():
    for _name, payload in GOLDEN.items():
        assert validate_summary(payload) == payload


def test_models_validate_summary_round_trips_golden_examples():
    from traigent.connectors.models import validate_summary as model_validate_summary

    assert (
        model_validate_summary(GOLDEN["connector_run_summary.json"])
        == GOLDEN["connector_run_summary.json"]
    )


def test_summary_errors_contain_json_pointers_not_values():
    bad = copy.deepcopy(GOLDEN["connector_run_summary.json"])
    bad["error_code"] = "PRIVATE-CANARY-7931"
    with pytest.raises(ValueError) as exc:
        validate_summary(bad)
    assert "/error_code" in str(exc.value)
    assert "PRIVATE-CANARY-7931" not in str(exc.value)


def test_summary_serialization_excludes_content_canaries():
    minter = CustomerSideMinter(ConnectionRef("langfuse"), b"k" * 32)
    token = minter.mint("run", "CUSTOMER_CONTENT_CANARY")
    payload = {
        **GOLDEN["connector_run_summary.json"],
        "run_token": token,
        "connection_token": minter.connection_token,
    }
    encoded = json.dumps(serialize_summary(payload, minter=minter))
    assert "vendor-private-id" not in encoded
    assert "CUSTOMER_CONTENT_CANARY" not in encoded


def test_summary_serialization_rejects_token_not_owned_by_minter():
    from traigent.connectors.privacy import OpaqueToken

    minter = CustomerSideMinter(ConnectionRef("langfuse"), b"k" * 32)
    forged = object.__new__(OpaqueToken)
    object.__setattr__(forged, "_value", "tk_0123456789abcdefghjkmnpqrs")
    object.__setattr__(forged, "_connection_id", minter._connection_id)
    payload = {
        **GOLDEN["connector_run_summary.json"],
        "run_token": forged,
        "connection_token": minter.connection_token,
    }

    with pytest.raises(ValueError, match="^/run_token$"):
        serialize_summary(payload, minter=minter)


def test_summary_serialization_rejects_unverified_locator_strings():
    minter = CustomerSideMinter(ConnectionRef("langfuse"), b"k" * 32)
    payload = {
        **GOLDEN["correlation_summary.json"],
        "run_token": minter.mint("run", "synthetic"),
    }

    with pytest.raises(ValueError, match="^/agent_function_ref$"):
        serialize_summary(payload, minter=minter)


def test_summary_serialization_accepts_verified_code_facts():
    from traigent.connectors.models import VerifiedCodeFact

    minter = CustomerSideMinter(ConnectionRef("langfuse"), b"k" * 32)
    payload = {
        key: value
        for key, value in GOLDEN["correlation_summary.json"].items()
        if key not in {"agent_function_ref", "agent_file_path"}
    }
    payload["run_token"] = minter.mint("run", "synthetic")
    fact = VerifiedCodeFact.from_callable(
        _sample_agent, repository_root=Path(__file__).parents[3]
    )

    serialized = serialize_summary(payload, minter=minter, code_fact=fact)

    assert serialized["agent_function_ref"] == f"{__name__}:_sample_agent"
    assert (
        serialized["agent_file_path"]
        == "tests/unit/connectors/test_summary_validation.py"
    )


@pytest.mark.parametrize("name", GOLDEN)
def test_unknown_summary_fields_are_rejected(name):
    payload = copy.deepcopy(GOLDEN[name])
    payload["unexpected"] = "not part of the closed contract"
    with pytest.raises(ValueError) as exc:
        validate_summary(payload)
    assert str(exc.value).startswith("invalid summary at /")
    assert "not part of the closed contract" not in str(exc.value)


def test_enum_pattern_and_format_negatives_are_rejected():
    bad = copy.deepcopy(GOLDEN["connector_run_summary.json"])
    bad["status"] = "SUCCEEDED"
    with pytest.raises(ValueError):
        validate_summary(bad)
    bad = copy.deepcopy(GOLDEN["connector_run_summary.json"])
    bad["run_token"] = "tk_alice_has_diabetes"
    with pytest.raises(ValueError):
        validate_summary(bad)
    bad = copy.deepcopy(GOLDEN["connector_run_summary.json"])
    bad["started_at"] = "2026-10-08T10:00:00+02:00"
    with pytest.raises(ValueError):
        validate_summary(bad)


def test_tokens_are_minted_only_by_customer_side_minter():
    from traigent.connectors.privacy import OpaqueToken

    with pytest.raises((TypeError, ValueError)):
        OpaqueToken("tk_0123456789abcdefghjkmnpqrs", object())


def test_tokenizer_isolates_connections():
    a = CustomerSideMinter(ConnectionRef("langfuse"), b"a" * 32)
    b = CustomerSideMinter(ConnectionRef("langfuse"), b"b" * 32)
    assert (
        a.mint("observation", "same-id").value != b.mint("observation", "same-id").value
    )


def test_keyed_digests_are_connection_scoped():
    a = CustomerSideMinter(ConnectionRef("langfuse"), b"a" * 32)
    b = CustomerSideMinter(ConnectionRef("langfuse"), b"b" * 32)
    assert a.digest("same-id") != b.digest("same-id")
