"""Catalog tests for ACI and context-budget knob-pack TVar rows."""

from __future__ import annotations

import copy
import json

from jsonschema import Draft7Validator

from traigent.api.functions import (
    list_recommendation_agent_types,
    recommend_configuration_space,
)
from traigent.config_generator.catalog import _load_catalog_entry_schema, load_catalog

ACI_NAMES = {
    "repo_context_strategy",
    "file_view_window",
    "edit_granularity",
    "test_selection_strategy",
    "patch_review_mode",
}
CONTEXT_BUDGET_NAMES = {
    "context_selection_policy",
    "context_order",
    "summary_style",
    "compression_ratio",
    "citation_policy",
}
NEW_NAMES = ACI_NAMES | CONTEXT_BUDGET_NAMES
CVAR_VOCABULARY = {
    "confidence_to_edit",
    "min_test_signal",
    "evidence_coverage_min",
    "max_context_tokens",
}


def _new_entries() -> list[dict[str, object]]:
    entries = [
        entry
        for entry in load_catalog()
        if str(entry.get("name")) in NEW_NAMES
    ]
    assert {str(entry["name"]) for entry in entries} == NEW_NAMES
    return entries


def test_new_knob_pack_rows_validate_against_existing_schema() -> None:
    schema = _load_catalog_entry_schema()
    validator = Draft7Validator(schema)

    for entry in _new_entries():
        errors = sorted(validator.iter_errors(entry), key=lambda err: err.path)
        assert errors == []


def test_recommendations_include_aci_rows_for_code_gen_at_low_confidence() -> None:
    payload = recommend_configuration_space("code_gen", min_confidence="low")
    names = {row["name"] for row in payload["recommendations"]}

    assert ACI_NAMES <= names
    assert ACI_NAMES <= set(payload["configuration_space"])
    for row in payload["recommendations"]:
        if row["name"] in ACI_NAMES:
            assert row["confidence"] == "low"
            assert row["category"] == "agent_computer_interface"
            assert row["catalog_entry_id"].startswith("code_gen.")
            assert "software_engineering_agent" in row["apply_guidance"]


def test_recommendations_include_context_budget_rows_for_rag_at_low_confidence() -> None:
    payload = recommend_configuration_space("rag", min_confidence="low")
    names = {row["name"] for row in payload["recommendations"]}

    assert CONTEXT_BUDGET_NAMES <= names
    assert CONTEXT_BUDGET_NAMES <= set(payload["configuration_space"])
    for row in payload["recommendations"]:
        if row["name"] in CONTEXT_BUDGET_NAMES:
            assert row["confidence"] == "low"
            assert row["category"] == "context_budget"
            assert row["catalog_entry_id"].startswith("rag.")


def test_external_paper_evidence_derives_low_confidence_honestly() -> None:
    for entry in _new_entries():
        refs = entry["evidence_refs"]
        assert isinstance(refs, list) and refs
        for ref in refs:
            assert ref["scope"].startswith("external_paper:")
            assert ref["delta"] is None
            assert "external_paper_not_traigent_measured" in ref["limitations"]
            assert "not_catalog_metric_delta" in ref["limitations"]

    code_gen = recommend_configuration_space("code_gen", min_confidence="low")
    rag = recommend_configuration_space("rag", min_confidence="low")
    rows = [
        row
        for row in code_gen["recommendations"] + rag["recommendations"]
        if row["name"] in NEW_NAMES
    ]
    assert len(rows) == 10
    assert {row["confidence"] for row in rows} == {"low"}
    assert all("observational support; no measured delta" in row["evidence_note"] for row in rows)


def test_rows_do_not_expand_public_agent_types() -> None:
    assert list_recommendation_agent_types() == ("code_gen", "rag")


def test_cvar_vocabulary_only_appears_in_apply_guidance() -> None:
    for entry in _new_entries():
        assert "pack_id" not in entry
        assert "cvars" not in entry
        without_guidance = copy.deepcopy(entry)
        without_guidance.pop("apply_guidance", None)
        serialized = json.dumps(without_guidance, sort_keys=True)
        assert not any(term in serialized for term in CVAR_VOCABULARY)
