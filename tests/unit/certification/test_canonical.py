from __future__ import annotations

import pytest

from traigent.certification.canonical import canonicalize_artifact_document


def test_uses_fp2_for_plain_json_objects() -> None:
    assert canonicalize_artifact_document({"b": 2, "a": 1}) == '{"a":1,"b":2}'


@pytest.mark.parametrize(
    "document",
    [
        {"value": 0.5},
        {"value": (1, 2)},
        {"value": 2**53},
        {"value": float("nan")},
    ],
)
def test_rejects_values_outside_fp2(document: dict) -> None:
    with pytest.raises(TypeError):
        canonicalize_artifact_document(document)


def test_rejects_dict_subclasses_and_cycles() -> None:
    class Mapping(dict):
        pass

    with pytest.raises(TypeError):
        canonicalize_artifact_document(Mapping())
    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(TypeError):
        canonicalize_artifact_document(cyclic)
