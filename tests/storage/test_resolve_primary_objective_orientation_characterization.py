"""Characterization tests pinning LocalStorageManager._resolve_primary_objective_orientation."""

from __future__ import annotations

import pytest

from traigent.storage.local_storage import LocalStorageManager


def resolve(cfg, name):
    return LocalStorageManager._resolve_primary_objective_orientation(None, cfg, name)


@pytest.mark.parametrize("cfg", [None, [], "x", 3])
def test_non_dict_config(cfg):
    assert resolve(cfg, "score") == "maximize"
    assert resolve(cfg, "acc") is None


def test_schema_definition_orientation_wins():
    cfg = {
        "objective_schema": {
            "objectives": [{"name": "acc", "orientation": "minimize"}]
        },
        "objective_orientations": {"acc": "maximize"},
    }
    assert resolve(cfg, "acc") == "minimize"


def test_schema_definition_non_string_orientation_is_stringified():
    cfg = {"objective_schema": {"objectives": [{"name": "acc", "orientation": 5}]}}
    assert resolve(cfg, "acc") == "5"


def test_schema_definition_without_orientation_returns_none_without_fallthrough():
    cfg = {
        "objective_schema": {"objectives": [{"name": "acc"}]},
        "objective_orientations": {"acc": "maximize"},
    }
    assert resolve(cfg, "acc") is None


def test_schema_first_match_wins_and_non_dict_definitions_skipped():
    cfg = {
        "objective_schema": {
            "objectives": [
                None,
                "x",
                {"name": "acc", "orientation": "minimize"},
                {"name": "acc", "orientation": "maximize"},
            ]
        }
    }
    assert resolve(cfg, "acc") == "minimize"


def test_orientations_mapping_fallback_stringified():
    assert resolve({"objective_orientations": {"acc": "minimize"}}, "acc") == "minimize"
    assert resolve({"objective_orientations": {"acc": 7}}, "acc") == "7"


def test_orientations_mapping_none_value_stringified():
    assert resolve({"objective_orientations": {"acc": None}}, "acc") == "None"


def test_orientations_not_dict_ignored():
    assert resolve({"objective_orientations": ["acc"]}, "acc") is None


def test_score_defaults_to_maximize_without_declared_objective():
    assert resolve({}, "score") == "maximize"
    assert resolve({"objective_schema": {"objectives": []}}, "score") == "maximize"


def test_score_with_declared_objectives_returns_none():
    assert resolve({"objectives": ["acc"]}, "score") is None
    assert (
        resolve({"objective_schema": {"objectives": [{"name": "acc"}]}}, "score")
        is None
    )


def test_malformed_schema_is_not_a_declaration():
    assert resolve({"objective_schema": "x"}, "score") == "maximize"
    assert resolve({"objective_schema": {"objectives": "abc"}}, "score") is None
    assert resolve({"objective_schema": {"objectives": "abc"}}, "acc") is None


def test_unknown_objective_none():
    assert resolve({}, "acc") is None
