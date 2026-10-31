"""Characterization tests pinning cloud.models.session_identity_v2_to_wire.

Requests are objects that record every attribute read into a shared trace, so
the number and order of reads are pinned along with the serialized identity.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.cloud import models

HEX = "0123456789abcdef"
DIGEST_A = "a" * 64
DIGEST_B = HEX * 4
CODE_DIGEST = "sha256:" + "c" * 64
UNKNOWN_VERSION: dict[str, Any] = {"schema": "fp1", "digest": None, "state": "unknown"}


class _Request:
    """Attribute bag recording reads; missing names raise AttributeError."""

    def __init__(self, trace: list[Any], **attrs: Any) -> None:
        self.__dict__["_trace"] = trace
        self.__dict__["_attrs"] = attrs

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        object.__getattribute__(self, "_trace").append(("read", name))
        attrs: dict[str, Any] = object.__getattribute__(self, "_attrs")
        if name not in attrs:
            raise AttributeError(name)
        value = attrs[name]
        if isinstance(value, BaseException):
            raise value
        return value


def _wire(**attrs: Any) -> dict[str, Any]:
    result: dict[str, Any] = models.session_identity_v2_to_wire(_Request([], **attrs))
    return result


def _versions(**overrides: dict[str, Any]) -> dict[str, Any]:
    base = {
        "agent": UNKNOWN_VERSION,
        "dataset": UNKNOWN_VERSION,
        "evaluator": UNKNOWN_VERSION,
        "config_space": UNKNOWN_VERSION,
    }
    base.update(overrides)
    return base


def _verified(digest: str) -> dict[str, Any]:
    return {"schema": "fp1", "digest": f"sha256:{digest}", "state": "verified"}


# ------------------------------------------------------------------ baseline


def test_empty_request_is_all_unknown_and_exact() -> None:
    result = _wire()
    assert result == {
        "identity_version": 2,
        "agent_id": None,
        "agent_id_source": "unknown",
        "dataset_id": None,
        "dataset_id_source": "unknown",
        "evaluator_id_source": "unknown",
        "artifact_versions": _versions(),
    }
    assert list(result) == [
        "identity_version",
        "agent_id",
        "agent_id_source",
        "dataset_id",
        "dataset_id_source",
        "evaluator_id_source",
        "artifact_versions",
    ]
    assert "evaluator_id" not in result
    assert "evaluator_code_digest" not in result


def test_object_without_any_attributes_is_the_same_as_none_attributes() -> None:
    assert _wire() == _wire(
        agent_key=None,
        dataset_id=None,
        artifact_fingerprints=None,
        evaluator_id=None,
        evaluator_definition_id=None,
        evaluator_id_source=None,
        evaluator_code_digest=None,
    )


# --------------------------------------------------------------------- agent


@pytest.mark.parametrize(
    ("agent_key", "expected"),
    [
        ("agent-1", "agent-1"),
        ("  agent-1\t", "agent-1"),
        ("", None),
        ("   ", None),
        ("\n\t", None),
        (None, None),
        (7, None),
        (b"agent", None),
        (["agent"], None),
    ],
)
def test_agent_key_is_stripped_and_declared(agent_key: Any, expected: Any) -> None:
    result = _wire(agent_key=agent_key)
    assert result["agent_id"] == expected
    assert result["agent_id_source"] == (
        "declared" if expected is not None else "unknown"
    )


def test_agent_key_strip_call_counts_follow_the_conditional_expression() -> None:
    calls: list[str] = []

    class Key(str):
        def strip(self, chars: str | None = None) -> str:
            calls.append("strip")
            return str.strip(self, chars)

    result = _wire(agent_key=Key(" a "))
    assert result["agent_id"] == "a"
    assert calls == ["strip", "strip"]
    calls.clear()
    result = _wire(agent_key=Key("  "))
    assert result["agent_id"] is None
    assert calls == ["strip"]


# ------------------------------------------------------------------- dataset


@pytest.mark.parametrize(
    ("dataset_id", "expected_id"),
    [("ds-1", "ds-1"), ("  ds-2 ", "ds-2"), (None, None), ("", None), (5, None)],
)
def test_dataset_id_comes_from_the_dataset_wire(
    dataset_id: Any, expected_id: Any
) -> None:
    result = _wire(dataset_id=dataset_id)
    assert result["dataset_id"] == expected_id
    assert result["dataset_id_source"] == ("declared" if expected_id else "unknown")


def test_dataset_wire_that_is_not_a_string_id_stays_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        models, "session_dataset_identity_to_wire", lambda request: {"dataset_id": 5}
    )
    result = _wire()
    assert result["dataset_id"] is None
    assert result["dataset_id_source"] == "unknown"


def test_empty_string_dataset_id_in_wire_counts_as_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        models, "session_dataset_identity_to_wire", lambda request: {"dataset_id": ""}
    )
    result = _wire()
    assert result["dataset_id"] == ""
    assert result["dataset_id_source"] == "declared"


def test_derived_label_is_used_when_no_explicit_dataset_id() -> None:
    result = _wire(dataset_metadata={"name": "my label"})
    assert result["dataset_id"] == "my label"
    assert result["dataset_id_source"] == "declared"


# ---------------------------------------------------------- artifact versions


def test_artifact_versions_map_valid_fp1_values() -> None:
    fingerprints = {
        "agent": f"fp1:{DIGEST_A}",
        "dataset": f"fp1:{DIGEST_B}",
        "evaluator": None,
        "config_space": f"fp1:{DIGEST_A}",
    }
    result = _wire(artifact_fingerprints=fingerprints)
    assert result["artifact_versions"] == _versions(
        agent=_verified(DIGEST_A),
        dataset=_verified(DIGEST_B),
        config_space=_verified(DIGEST_A),
    )
    assert list(result["artifact_versions"]) == [
        "agent",
        "dataset",
        "evaluator",
        "config_space",
    ]


def test_artifact_versions_consult_the_wire_object_once_per_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    wire_value = f"fp1:{DIGEST_A}"

    class Wire(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            trace.append(("get", key))
            return super().get(key, default)

        def __bool__(self) -> bool:
            trace.append(("wire_bool",))
            return super().__len__() > 0

    def fake(value: Any) -> Any:
        trace.append(("fingerprints_to_wire", value))
        return Wire(agent=wire_value, dataset="not-a-digest")

    monkeypatch.setattr(models, "artifact_fingerprints_to_wire", fake)
    result = models.session_identity_v2_to_wire(
        _Request(trace, artifact_fingerprints="RAW")
    )
    assert result["artifact_versions"] == _versions(agent=_verified(DIGEST_A))
    assert [event for event in trace if event[0] != "read"] == [
        ("fingerprints_to_wire", "RAW"),
        ("wire_bool",),
        ("get", "agent"),
        ("get", "dataset"),
        ("get", "evaluator"),
        ("get", "config_space"),
    ]


@pytest.mark.parametrize("empty", [None, {}])
def test_missing_or_empty_wire_gives_unknown_versions(
    monkeypatch: pytest.MonkeyPatch, empty: Any
) -> None:
    monkeypatch.setattr(models, "artifact_fingerprints_to_wire", lambda value: empty)
    assert _wire()["artifact_versions"] == _versions()


@pytest.mark.parametrize(
    "value",
    [
        f"fp1:{DIGEST_A.upper()}",
        f"fp1:{DIGEST_A[:-1]}",
        f"fp1:{DIGEST_A}0",
        f"fp2:{DIGEST_A}",
        f"fp1:{'g' * 64}",
        DIGEST_A,
        7,
        None,
    ],
)
def test_invalid_fingerprints_stay_unknown(value: Any) -> None:
    result = _wire(artifact_fingerprints={"agent": value})
    assert result["artifact_versions"]["agent"] == UNKNOWN_VERSION


def test_fingerprint_length_boundary_is_exactly_sixty_eight() -> None:
    ok = _wire(artifact_fingerprints={"agent": "fp1:" + "b" * 64})
    assert ok["artifact_versions"]["agent"] == _verified("b" * 64)
    short = _wire(artifact_fingerprints={"agent": "fp1:" + "b" * 63})
    assert short["artifact_versions"]["agent"] == UNKNOWN_VERSION
    long = _wire(artifact_fingerprints={"agent": "fp1:" + "b" * 65})
    assert long["artifact_versions"]["agent"] == UNKNOWN_VERSION


# ----------------------------------------------------------------- evaluator


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"evaluator_id": "ev-1"}, "ev-1"),
        ({"evaluator_id": "  ev-1 "}, "ev-1"),
        ({"evaluator_definition_id": "def-1"}, "def-1"),
        ({"evaluator_definition_id": " def-1 "}, "def-1"),
        ({"evaluator_id": "ev-1", "evaluator_definition_id": "def-1"}, "ev-1"),
        ({"evaluator_id": "", "evaluator_definition_id": "def-1"}, "def-1"),
        ({"evaluator_id": "   ", "evaluator_definition_id": "def-1"}, "def-1"),
        ({"evaluator_id": 5, "evaluator_definition_id": "def-1"}, "def-1"),
        ({"evaluator_id": None, "evaluator_definition_id": "def-1"}, "def-1"),
        ({"evaluator_id": "", "evaluator_definition_id": ""}, None),
        ({"evaluator_id": "", "evaluator_definition_id": "  "}, None),
        ({"evaluator_id": "ev", "evaluator_definition_id": 3}, "ev"),
        ({"evaluator_definition_id": 3}, None),
        ({"evaluator_id": 5}, None),
        ({}, None),
    ],
)
def test_evaluator_id_resolution(attrs: dict[str, Any], expected: Any) -> None:
    result = _wire(**attrs)
    if expected is None:
        assert "evaluator_id" not in result
    else:
        assert result["evaluator_id"] == expected
        assert result["evaluator_id_source"] == "registered"


def test_explicit_source_is_kept_even_with_an_id() -> None:
    result = _wire(evaluator_id="ev", evaluator_id_source="declared")
    assert result["evaluator_id"] == "ev"
    assert result["evaluator_id_source"] == "declared"


@pytest.mark.parametrize("source", ["registered", "declared", "unknown", "", 0, False])
def test_any_non_none_source_is_copied_verbatim(source: Any) -> None:
    assert _wire(evaluator_id_source=source)["evaluator_id_source"] == source
    assert (
        _wire(evaluator_id="ev", evaluator_id_source=source)["evaluator_id_source"]
        == source
    )


def test_source_defaults_follow_whether_an_id_exists() -> None:
    assert _wire(evaluator_id="ev")["evaluator_id_source"] == "registered"
    assert _wire()["evaluator_id_source"] == "unknown"


def test_evaluator_id_strip_call_counts_follow_the_conditional_expression() -> None:
    calls: list[str] = []

    class Text(str):
        def strip(self, chars: str | None = None) -> str:
            calls.append("strip")
            return str.strip(self, chars)

    result = _wire(evaluator_id=Text(" e "))
    assert result["evaluator_id"] == "e"
    # one strip in the fallback check, two in the normalization expression
    assert calls == ["strip", "strip", "strip"]
    calls.clear()
    result = _wire(evaluator_id=Text("  "), evaluator_definition_id=Text(" d "))
    assert result["evaluator_id"] == "d"
    # blank id: one strip; the fallback value is then normalized with two
    assert calls == ["strip", "strip", "strip"]


# --------------------------------------------------------------- code digest


def test_code_digest_is_emitted_only_without_an_evaluator_id() -> None:
    result = _wire(evaluator_code_digest=CODE_DIGEST)
    assert result["evaluator_code_digest"] == CODE_DIGEST
    assert result["evaluator_id_source"] == "unknown"
    assert "evaluator_id" not in result


@pytest.mark.parametrize(
    "attrs",
    [
        {"evaluator_id": "ev"},
        {"evaluator_definition_id": "def"},
        {"evaluator_id_source": "registered"},
        {"evaluator_id_source": "declared"},
        {"evaluator_id_source": ""},
        {"evaluator_ref": "ref"},
        {"evaluator_ref": 1},
    ],
)
def test_code_digest_is_withheld_by_any_competing_identity(
    attrs: dict[str, Any],
) -> None:
    result = _wire(evaluator_code_digest=CODE_DIGEST, **attrs)
    assert "evaluator_code_digest" not in result


@pytest.mark.parametrize("ref", [None, "", 0, [], {}])
def test_falsy_evaluator_ref_does_not_block_the_digest(ref: Any) -> None:
    result = _wire(evaluator_code_digest=CODE_DIGEST, evaluator_ref=ref)
    assert result["evaluator_code_digest"] == CODE_DIGEST


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:" + "c" * 63,
        "sha256:" + "c" * 65,
        "sha256:" + "C" * 64,
        "sha512:" + "c" * 64,
        "sha256:" + "c" * 64 + "\n",
        " sha256:" + "c" * 64,
        "c" * 64,
        "",
        None,
        5,
        b"sha256:" + b"c" * 64,
    ],
)
def test_malformed_code_digest_is_dropped(digest: Any) -> None:
    assert "evaluator_code_digest" not in _wire(evaluator_code_digest=digest)


def test_explicit_unknown_source_still_allows_the_digest() -> None:
    result = _wire(evaluator_code_digest=CODE_DIGEST, evaluator_id_source="unknown")
    assert result["evaluator_code_digest"] == CODE_DIGEST


def test_digest_key_comes_after_the_fixed_keys() -> None:
    result = _wire(evaluator_code_digest=CODE_DIGEST)
    assert list(result)[-1] == "evaluator_code_digest"
    result = _wire(evaluator_id="ev")
    assert list(result)[-1] == "evaluator_id"


# ------------------------------------------------------------ read order


def test_attribute_reads_without_any_identity() -> None:
    trace: list[Any] = []
    models.session_identity_v2_to_wire(_Request(trace))
    assert trace == [
        ("read", "agent_key"),
        ("read", "dataset_id"),
        ("read", "dataset_metadata"),
        ("read", "metadata"),
        ("read", "artifact_fingerprints"),
        ("read", "evaluator_id"),
        ("read", "evaluator_definition_id"),
        ("read", "evaluator_id_source"),
        ("read", "evaluator_code_digest"),
    ]


def test_attribute_reads_with_an_evaluator_id_skip_definition_and_digest() -> None:
    trace: list[Any] = []
    models.session_identity_v2_to_wire(
        _Request(trace, agent_key="a", dataset_id="d", evaluator_id="e")
    )
    assert trace == [
        ("read", "agent_key"),
        ("read", "dataset_id"),
        ("read", "artifact_fingerprints"),
        ("read", "evaluator_id"),
        ("read", "evaluator_id_source"),
    ]


def test_attribute_reads_with_a_valid_digest_reach_evaluator_ref() -> None:
    trace: list[Any] = []
    models.session_identity_v2_to_wire(
        _Request(trace, evaluator_code_digest=CODE_DIGEST)
    )
    assert trace[-3:] == [
        ("read", "evaluator_id_source"),
        ("read", "evaluator_code_digest"),
        ("read", "evaluator_ref"),
    ]


@pytest.mark.parametrize(
    "attrs",
    [
        {"evaluator_code_digest": "bad"},
        {"evaluator_code_digest": CODE_DIGEST, "evaluator_id_source": "declared"},
    ],
)
def test_evaluator_ref_is_not_read_when_an_earlier_condition_fails(
    attrs: dict[str, Any],
) -> None:
    trace: list[Any] = []
    models.session_identity_v2_to_wire(_Request(trace, **attrs))
    assert ("read", "evaluator_ref") not in trace


# ------------------------------------------------------------- raising paths


@pytest.mark.parametrize(
    "name",
    [
        "agent_key",
        "dataset_id",
        "artifact_fingerprints",
        "evaluator_id",
        "evaluator_id_source",
        "evaluator_code_digest",
        "evaluator_ref",
    ],
)
def test_raising_accessor_propagates(name: str) -> None:
    attrs: dict[str, Any] = {"evaluator_code_digest": CODE_DIGEST}
    attrs[name] = RuntimeError(f"{name} failed")
    with pytest.raises(RuntimeError, match=f"{name} failed"):
        _wire(**attrs)


def test_raising_definition_id_accessor_propagates_when_id_is_blank() -> None:
    with pytest.raises(RuntimeError, match="definition failed"):
        _wire(
            evaluator_id="", evaluator_definition_id=RuntimeError("definition failed")
        )


def test_raising_definition_id_accessor_is_not_read_when_id_is_valid() -> None:
    result = _wire(
        evaluator_id="ev", evaluator_definition_id=RuntimeError("never read")
    )
    assert result["evaluator_id"] == "ev"


def test_raising_dataset_wire_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: Any) -> Any:
        raise RuntimeError("dataset failed")

    monkeypatch.setattr(models, "session_dataset_identity_to_wire", boom)
    with pytest.raises(RuntimeError, match="dataset failed"):
        _wire()


def test_raising_fingerprint_wire_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(value: Any) -> Any:
        raise RuntimeError("fingerprints failed")

    monkeypatch.setattr(models, "artifact_fingerprints_to_wire", boom)
    with pytest.raises(RuntimeError, match="fingerprints failed"):
        _wire()


def test_raising_wire_get_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    class Wire(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("get failed")

    monkeypatch.setattr(
        models, "artifact_fingerprints_to_wire", lambda value: Wire(agent="x")
    )
    with pytest.raises(RuntimeError, match="get failed"):
        _wire()


def test_raising_wire_truth_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    class Wire(dict[str, Any]):
        def __bool__(self) -> bool:
            raise RuntimeError("truth failed")

    monkeypatch.setattr(models, "artifact_fingerprints_to_wire", lambda value: Wire())
    with pytest.raises(RuntimeError, match="truth failed"):
        _wire()


def test_evaluator_ref_truth_is_tested_once() -> None:
    truths: list[str] = []

    class Ref:
        def __bool__(self) -> bool:
            truths.append("ref")
            return False

    result = _wire(evaluator_code_digest=CODE_DIGEST, evaluator_ref=Ref())
    assert result["evaluator_code_digest"] == CODE_DIGEST
    assert truths == ["ref"]


def test_raising_evaluator_ref_truth_propagates() -> None:
    class Ref:
        def __bool__(self) -> bool:
            raise RuntimeError("ref truth failed")

    with pytest.raises(RuntimeError, match="ref truth failed"):
        _wire(evaluator_code_digest=CODE_DIGEST, evaluator_ref=Ref())


def test_digest_regex_is_the_shared_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    class Pattern:
        def fullmatch(self, value: Any) -> bool:
            seen.append(value)
            return isinstance(value, str) and value == "accepted"

    monkeypatch.setattr(models, "EVALUATOR_CODE_DIGEST_RE", Pattern())
    result = _wire(evaluator_code_digest="accepted")
    assert result["evaluator_code_digest"] == "accepted"
    assert seen == ["accepted"]
    seen.clear()
    assert "evaluator_code_digest" not in _wire(evaluator_code_digest="other")
    assert seen == ["other"]
    seen.clear()
    assert "evaluator_code_digest" not in _wire(evaluator_code_digest=5)
    assert seen == []


@pytest.mark.parametrize(
    "attrs",
    [
        {"evaluator_id": "ev"},
        {"evaluator_definition_id": "def"},
    ],
)
def test_digest_is_withheld_by_an_id_even_when_source_is_explicitly_unknown(
    attrs: dict[str, Any],
) -> None:
    result = _wire(
        evaluator_code_digest=CODE_DIGEST, evaluator_id_source="unknown", **attrs
    )
    assert "evaluator_code_digest" not in result
    assert result["evaluator_id_source"] == "unknown"
