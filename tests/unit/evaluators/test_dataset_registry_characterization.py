"""Characterization tests pinning ``dataset_registry._load_registry``."""

from __future__ import annotations

import json
import os

import pytest

from traigent.evaluators import dataset_registry as dr
from traigent.evaluators.dataset_registry import (
    DATASET_REGISTRY_ENV,
    DatasetRegistryEntry,
    clear_dataset_registry_cache,
)
from traigent.utils.exceptions import ValidationError


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_dataset_registry_cache()
    monkeypatch.delenv(DATASET_REGISTRY_ENV, raising=False)
    yield
    clear_dataset_registry_cache()


def _write(tmp_path, data, monkeypatch, raw=None):
    path = tmp_path / "reg.json"
    path.write_text(raw if raw is not None else json.dumps(data), encoding="utf-8")
    monkeypatch.setenv(DATASET_REGISTRY_ENV, str(path))
    return path


def _err(tmp_path, monkeypatch, data, raw=None):
    _write(tmp_path, data, monkeypatch, raw)
    with pytest.raises(ValidationError) as ei:
        dr._load_registry()
    return str(ei.value)


def test_no_env_returns_empty_and_resets_cache(monkeypatch, tmp_path):
    _write(tmp_path, {"a": "p"}, monkeypatch)
    assert dr._load_registry()
    monkeypatch.delenv(DATASET_REGISTRY_ENV)
    assert dr._load_registry() == {}
    assert dr._CACHE == {} and dr._CACHE_SOURCE is None and dr._CACHE_MTIME is None


def test_missing_file(monkeypatch, tmp_path):
    missing = tmp_path / "nope.json"
    monkeypatch.setenv(DATASET_REGISTRY_ENV, str(missing))
    with pytest.raises(ValidationError) as ei:
        dr._load_registry()
    assert str(ei.value) == f"Dataset registry file not found: {missing}"


def test_invalid_json(monkeypatch, tmp_path):
    msg = _err(tmp_path, monkeypatch, None, raw="{not json")
    assert msg.startswith("Invalid JSON in dataset registry: ")


@pytest.mark.parametrize("raw", ["[]", "3", '"s"', "null"])
def test_top_level_must_be_object(monkeypatch, tmp_path, raw):
    msg = _err(tmp_path, monkeypatch, None, raw=raw)
    assert msg == "Dataset registry must be a JSON object at the top level"


@pytest.mark.parametrize("datasets", [[], 3, "s", None])
def test_datasets_must_be_object(monkeypatch, tmp_path, datasets):
    msg = _err(tmp_path, monkeypatch, {"datasets": datasets})
    assert msg == "Dataset registry must map dataset identifiers to entries"


def test_empty_identifier_rejected(monkeypatch, tmp_path):
    msg = _err(tmp_path, monkeypatch, {"": "p"})
    assert msg == "Dataset identifiers must be non-empty strings"


@pytest.mark.parametrize("value", [1, [], None, 1.5, True])
def test_entry_must_be_string_or_object(monkeypatch, tmp_path, value):
    msg = _err(tmp_path, monkeypatch, {"d": value})
    assert msg == "Dataset 'd' entry must be either a string path or an object"


@pytest.mark.parametrize("entry", [{}, {"path": ""}, {"path": 3}, {"path": None}])
def test_path_required(monkeypatch, tmp_path, entry):
    msg = _err(tmp_path, monkeypatch, {"d": entry})
    assert msg == "Dataset 'd' must define a non-empty 'path'"


def test_name_must_be_string(monkeypatch, tmp_path):
    msg = _err(tmp_path, monkeypatch, {"d": {"path": "p", "name": 3}})
    assert msg == "Dataset 'd' field 'name' must be a string"


def test_description_must_be_string(monkeypatch, tmp_path):
    msg = _err(tmp_path, monkeypatch, {"d": {"path": "p", "description": []}})
    assert msg == "Dataset 'd' field 'description' must be a string"


def test_metadata_must_be_object(monkeypatch, tmp_path):
    msg = _err(tmp_path, monkeypatch, {"d": {"path": "p", "metadata": [1]}})
    assert msg == "Dataset 'd' field 'metadata' must be an object"


def test_validation_order_path_name_description_metadata(monkeypatch, tmp_path):
    entry = {"path": "p", "name": 1, "description": 1, "metadata": 1}
    assert "'name'" in _err(tmp_path, monkeypatch, {"d": entry})
    entry = {"path": "p", "description": 1, "metadata": 1}
    assert "'description'" in _err(tmp_path, monkeypatch, {"d": entry})
    entry = {"path": 1, "name": 1}
    assert "'path'" in _err(tmp_path, monkeypatch, {"d": entry})


def test_first_bad_entry_wins_in_order(monkeypatch, tmp_path):
    msg = _err(tmp_path, monkeypatch, {"a": 1, "b": {}})
    assert "Dataset 'a'" in msg


def test_parses_all_shapes(monkeypatch, tmp_path):
    data = {
        "datasets": {
            "s": "path/s.json",
            "full": {
                "path": "p",
                "name": "Nice",
                "description": "desc",
                "metadata": {"k": 1},
            },
            "bare": {"path": "q"},
            "emptyname": {"path": "r", "name": ""},
        }
    }
    _write(tmp_path, data, monkeypatch)
    reg = dr._load_registry()
    assert reg["s"] == DatasetRegistryEntry(name="s", path="path/s.json")
    assert reg["full"] == DatasetRegistryEntry("Nice", "p", "desc", {"k": 1})
    assert reg["bare"] == DatasetRegistryEntry("bare", "q", None, None)
    assert reg["emptyname"].name == "emptyname"
    assert list(reg) == ["s", "full", "bare", "emptyname"]


def test_top_level_mapping_without_datasets_key(monkeypatch, tmp_path):
    _write(tmp_path, {"a": "x"}, monkeypatch)
    assert dr._load_registry() == {"a": DatasetRegistryEntry("a", "x")}


def test_metadata_is_copied(monkeypatch, tmp_path):
    _write(tmp_path, {"d": {"path": "p", "metadata": {"k": 1}}}, monkeypatch)
    reg = dr._load_registry()
    assert reg["d"].metadata == {"k": 1}


def test_cache_hit_and_mtime_invalidation(monkeypatch, tmp_path):
    path = _write(tmp_path, {"a": "x"}, monkeypatch)
    first = dr._load_registry()
    assert dr._load_registry() is first
    assert dr._CACHE is first
    path.write_text(json.dumps({"b": "y"}), encoding="utf-8")
    os.utime(path, (1, 1))
    second = dr._load_registry()
    assert second is not first
    assert list(second) == ["b"]


def test_failed_parse_does_not_replace_cache(monkeypatch, tmp_path):
    path = _write(tmp_path, {"a": "x"}, monkeypatch)
    first = dr._load_registry()
    path.write_text(json.dumps({"b": 1}), encoding="utf-8")
    os.utime(path, (5, 5))
    with pytest.raises(ValidationError):
        dr._load_registry()
    assert dr._CACHE is first
