"""Tests for privacy-safe TVAR observations."""

from __future__ import annotations

import json
import re

import pytest

from traigent.tuned_variables.observation import (
    build_tvar_observation,
    merge_tvar_observation_metadata,
)


@pytest.fixture(autouse=True)
def default_observation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIGENT_TVAR_OBSERVATION", raising=False)


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

    assert observation is not None
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

    assert observation is not None
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

    assert observation is not None
    assert observation["variables"] == [
        {"name": "candidate_count", "value": 2, "kind": "cardinality"}
    ]
    assert sentinel not in json.dumps(observation, sort_keys=True)


def test_default_hashed_hashes_free_text_and_keeps_enum_scalars() -> None:
    free_text = (
        "tenant support note with spaces and enough detail to exceed sixty four "
        "characters in the value"
    )

    observation = build_tvar_observation(
        session_id="session-3",
        trial_id="trial-3",
        config={
            "retrieval_profile": "linked_top6",
            "free_knob": free_text,
            "max_candidates": 8,
            "rerank_enabled": True,
            "temperature": 0.2,
        },
        metrics={"answer_em": 0.7},
        primary_metric="answer_em",
        comparability={"scope": "trial", "n": 5},
        sdk_version="9.9.9",
    )

    assert observation is not None
    values = {variable["name"]: variable["value"] for variable in observation["variables"]}
    assert re.fullmatch(r"sha256:[a-f0-9]{64}", values["free_knob"])
    assert values["free_knob"] != free_text
    assert values["retrieval_profile"] == "linked_top6"
    assert values["max_candidates"] == 8
    assert values["rerank_enabled"] is True
    assert values["temperature"] == 0.2
    assert free_text not in json.dumps(observation, sort_keys=True)


def test_observation_off_returns_none_and_merge_leaves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_TVAR_OBSERVATION", "off")

    observation = build_tvar_observation(
        session_id="session-4",
        trial_id="trial-4",
        config={"free_knob": "secret value with spaces"},
        metrics={"answer_em": 0.8},
        primary_metric="answer_em",
        comparability={"scope": "trial", "n": 2},
        sdk_version="9.9.9",
    )

    metadata = {"existing": {"value": 1}}
    merged = merge_tvar_observation_metadata(metadata, observation)

    assert observation is None
    assert merged == metadata
    assert "tvar_observation_v1" not in merged
    assert "tvar_observation_v1" not in metadata


def test_privacy_canary_free_text_never_appears_in_hashed_or_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "FREE_KNOB_SECRET_CANARY"
    free_text = f"{sentinel} embedded in a free string with spaces"

    hashed_observation = build_tvar_observation(
        session_id="session-6",
        trial_id="trial-6",
        config={"free_knob": free_text},
        metrics={"answer_em": 0.5},
        primary_metric="answer_em",
        comparability={"scope": "trial", "n": 1},
        sdk_version="9.9.9",
    )
    monkeypatch.setenv("TRAIGENT_TVAR_OBSERVATION", "off")
    off_observation = build_tvar_observation(
        session_id="session-6",
        trial_id="trial-7",
        config={"free_knob": free_text},
        metrics={"answer_em": 0.5},
        primary_metric="answer_em",
        comparability={"scope": "trial", "n": 1},
        sdk_version="9.9.9",
    )

    assert hashed_observation is not None
    assert sentinel not in json.dumps(hashed_observation, sort_keys=True)
    assert off_observation is None
    assert sentinel not in json.dumps(off_observation, sort_keys=True)


def test_effectuation_events_are_content_free_and_schema_valid() -> None:
    sentinel = "raw output should not appear"

    observation = build_tvar_observation(
        session_id="session-7",
        trial_id="trial-8",
        config={"candidate_count": 3},
        metrics={"answer_em": 0.5},
        primary_metric="answer_em",
        comparability={"scope": "trial", "n": 1},
        catalog_entry_ids=("code_gen.candidate_count.v1",),
        sdk_version="9.9.9",
        effectuation_events=[
            {
                "strategy": "self_consistency",
                "knob": "candidate_count",
                "calls": 3,
                "unique_outputs": 2,
                "aggregator": "majority_vote",
                "passthrough": False,
                "unsafe value": sentinel,
                "raw": sentinel,
            }
        ],
    )

    assert observation is not None
    assert observation["effectuation_events"] == [
        {
            "strategy": "self_consistency",
            "knob": "candidate_count",
            "calls": 3,
            "unique_outputs": 2,
            "aggregator": "majority_vote",
            "passthrough": False,
        }
    ]
    assert sentinel not in json.dumps(observation, sort_keys=True)
