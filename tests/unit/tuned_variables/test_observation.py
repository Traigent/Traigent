"""Tests for privacy-safe TVAR observations."""

from __future__ import annotations

import json

from traigent.tuned_variables.observation import (
    build_tvar_observation,
    merge_tvar_observation_metadata,
)


def test_build_tvar_observation_is_schema_valid_and_content_free() -> None:
    sentinel = "RAW_PROMPT_CANARY_DO_NOT_LEAK"

    observation = build_tvar_observation(
        session_id="session-1",
        trial_id="trial-1",
        config={
            "schema_context": "full_ddl_fk",
            "fewshot_k": 3,
            "raw_prompt": sentinel,
            "surrounding_context": {"prompt": sentinel},
        },
        metrics={"accuracy": 0.91, "latency": 12, "raw_text": sentinel},
        primary_metric="accuracy",
        comparability={"scope": "trial", "n": 20},
        catalog_entry_ids=("code_gen.schema_context.v1", "code_gen.fewshot_k.v1"),
        agent_type="code_gen",
        config_space_id="space-1",
        sdk_version="9.9.9",
    )

    payload = json.dumps(observation, sort_keys=True)
    assert observation["privacy"] == {"raw_content_included": False}
    assert observation["metrics"] == {"accuracy": 0.91, "latency": 12}
    assert observation["variables"] == [
        {"name": "schema_context", "value": "full_ddl_fk", "kind": "topology"},
        {"name": "fewshot_k", "value": 3, "kind": "cardinality"},
    ]
    assert sentinel not in payload


def test_merge_tvar_observation_metadata_is_non_destructive() -> None:
    observation = build_tvar_observation(
        session_id="session-1",
        trial_id="trial-1",
        config={"retrieval_k": 5},
        metrics={"answer_em": 0.4},
        primary_metric="answer_em",
        comparability={"scope": "trial", "n": 10},
        catalog_entry_ids=("rag.retrieval_k.v1",),
        agent_type="rag",
        config_space_id="space-1",
        sdk_version="9.9.9",
    )

    metadata = {"existing": {"value": 1}}
    merged = merge_tvar_observation_metadata(metadata, observation)

    assert merged["existing"] == {"value": 1}
    assert merged["tvar_observation_v1"] == observation
    assert "tvar_observation_v1" not in metadata


def test_privacy_canary_nested_context_never_appears() -> None:
    sentinel = "EXAMPLE_RESPONSE_SECRET_CANARY"

    observation = build_tvar_observation(
        session_id="session-2",
        trial_id="trial-2",
        config={
            "candidate_count": 2,
            "example_context": {"input": sentinel, "output": sentinel},
            "user_response": sentinel,
        },
        metrics={"execution_accuracy": 1.0},
        primary_metric="execution_accuracy",
        comparability={"scope": "trial", "n": 1},
        catalog_entry_ids=("code_gen.candidate_count.v1",),
        agent_type="code_gen",
        config_space_id="space-2",
        sdk_version="9.9.9",
    )

    assert observation["variables"] == [
        {"name": "candidate_count", "value": 2, "kind": "cardinality"}
    ]
    assert sentinel not in json.dumps(observation, sort_keys=True)
