"""Golden characterization for connectors.manifest.load_manifest."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from traigent.connectors.manifest import (
    ConditionalUpdate,
    GuaranteeLevel,
    load_manifest,
)

BASE = {
    "schema_version": "1",
    "connector": "dummy",
    "version": "1.0",
    "operations": {
        "write": {
            "idempotency": "emulated",
            "conditional_update": "read_check",
            "limits": {"max_page": 10},
            "guarantees": {"ordering": "native"},
        },
        "read": {},
    },
}


def _mut(fn):
    d = copy.deepcopy(BASE)
    fn(d)
    return d


def test_valid_mapping_roundtrip():
    m = load_manifest(BASE)
    assert (m.schema_version, m.connector, m.version) == ("1", "dummy", "1.0")
    w = m.operations["write"]
    assert w.idempotency is GuaranteeLevel.EMULATED
    assert w.conditional_update is ConditionalUpdate.READ_CHECK
    assert w.limits == {"max_page": 10}
    assert w.guarantees == {"ordering": GuaranteeLevel.NATIVE}
    r = m.operations["read"]
    assert r.idempotency is GuaranteeLevel.NONE
    assert r.conditional_update is ConditionalUpdate.NONE
    assert r.limits == {} and r.guarantees == {}


def test_sources_str_json_yaml_path(tmp_path: Path):
    expected = load_manifest(BASE)
    assert load_manifest(json.dumps(BASE)) == expected
    assert load_manifest(yaml.safe_dump(BASE)) == expected
    f = tmp_path / "m.yml"
    f.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    assert load_manifest(f) == expected
    assert load_manifest(str(f)) == expected


def test_bad_source_type():
    with pytest.raises(TypeError, match="manifest source must be"):
        load_manifest(5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        load_manifest(None)  # type: ignore[arg-type]


def test_multiline_string_that_is_a_path_is_not_read(tmp_path: Path):
    f = tmp_path / "m.yml"
    f.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid closed manifest object: root"):
        load_manifest(str(f) + "\n")


CASES = {
    "root list": ('["x"]', "invalid closed manifest object: root"),
    "extra root key": (
        _mut(lambda d: d.update(x=1)),
        "invalid closed manifest object: root",
    ),
    "missing root key": (
        _mut(lambda d: d.pop("version")),
        "manifest requires schema_version 1 and all root fields",
    ),
    "schema 2": (
        _mut(lambda d: d.update(schema_version="2")),
        "manifest requires schema_version 1 and all root fields",
    ),
    "schema int": (
        _mut(lambda d: d.update(schema_version=1)),
        "manifest requires schema_version 1 and all root fields",
    ),
    "connector empty": (
        _mut(lambda d: d.update(connector="")),
        "connector and version must be nonempty strings",
    ),
    "version int": (
        _mut(lambda d: d.update(version=1)),
        "connector and version must be nonempty strings",
    ),
    "ops empty": (
        _mut(lambda d: d.update(operations={})),
        "manifest operations must be a nonempty mapping",
    ),
    "ops list": (
        _mut(lambda d: d.update(operations=[1])),
        "manifest operations must be a nonempty mapping",
    ),
    "op name int": (
        _mut(lambda d: d.update(operations={1: {}})),
        "operation names must be nonempty strings",
    ),
    "op name empty": (
        _mut(lambda d: d.update(operations={"": {}})),
        "operation names must be nonempty strings",
    ),
    "op not dict": (
        _mut(lambda d: d.update(operations={"w": []})),
        "invalid closed manifest object: operations.w",
    ),
    "op extra key": (
        _mut(lambda d: d.update(operations={"w": {"x": 1}})),
        "invalid closed manifest object: operations.w",
    ),
    "bad idempotency": (
        _mut(lambda d: d.update(operations={"w": {"idempotency": "atomic"}})),
        "unknown guarantee value for operation w",
    ),
    "bad conditional": (
        _mut(lambda d: d.update(operations={"w": {"conditional_update": "native"}})),
        "unknown guarantee value for operation w",
    ),
    "unhashable idempotency": (
        _mut(lambda d: d.update(operations={"w": {"idempotency": []}})),
        None,
    ),
    "limits list": (
        _mut(lambda d: d.update(operations={"w": {"limits": []}})),
        "invalid limits for operation w",
    ),
    "limits negative": (
        _mut(lambda d: d.update(operations={"w": {"limits": {"a": -1}}})),
        "invalid limits for operation w",
    ),
    "limits bool": (
        _mut(lambda d: d.update(operations={"w": {"limits": {"a": True}}})),
        "invalid limits for operation w",
    ),
    "limits key int": (
        _mut(lambda d: d.update(operations={"w": {"limits": {1: 1}}})),
        "invalid limits for operation w",
    ),
    "guarantees list": (
        _mut(lambda d: d.update(operations={"w": {"guarantees": []}})),
        "invalid guarantees for operation w",
    ),
    "guarantee key reserved": (
        _mut(lambda d: d.update(operations={"w": {"guarantees": {"limits": "none"}}})),
        "reserved guarantee key for operation w",
    ),
    "guarantee key int": (
        _mut(lambda d: d.update(operations={"w": {"guarantees": {1: "none"}}})),
        "reserved guarantee key for operation w",
    ),
    "guarantee value bad": (
        _mut(lambda d: d.update(operations={"w": {"guarantees": {"a": "atomic"}}})),
        "unknown guarantee value for operation w",
    ),
    "guarantee value int": (
        _mut(lambda d: d.update(operations={"w": {"guarantees": {"a": 1}}})),
        "unknown guarantee value for operation w",
    ),
    "limits checked before guarantees": (
        _mut(lambda d: d.update(operations={"w": {"limits": [], "guarantees": []}})),
        "invalid limits for operation w",
    ),
    "idempotency checked before limits": (
        _mut(lambda d: d.update(operations={"w": {"idempotency": "x", "limits": []}})),
        "unknown guarantee value for operation w",
    ),
    "first bad op wins": (
        _mut(
            lambda d: d.update(
                operations={"a": {"limits": []}, "b": {"guarantees": []}}
            )
        ),
        "invalid limits for operation a",
    ),
}


@pytest.mark.parametrize("label", list(CASES))
def test_refusals(label):
    doc, msg = CASES[label]
    if msg is None:
        with pytest.raises(TypeError):
            load_manifest(doc)
        return
    with pytest.raises(ValueError) as ei:
        load_manifest(doc)
    assert str(ei.value) == msg


def test_mapping_is_copied_not_mutated():
    doc = copy.deepcopy(BASE)
    load_manifest(doc)
    assert doc == BASE


def test_limits_copied():
    doc = copy.deepcopy(BASE)
    m = load_manifest(doc)
    doc["operations"]["write"]["limits"]["max_page"] = 99
    assert m.operations["write"].limits == {"max_page": 10}
