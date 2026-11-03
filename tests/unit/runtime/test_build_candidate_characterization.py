"""Characterization of the closed candidate-fact builder, pinned before extraction.

Every refusal is the same closed error (code and path), so the tests pin which inputs are
refused, the full projected result, and the order and count of every mapping access.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from traigent.runtime.hosted import build_candidate
from traigent.runtime.model import RuntimeRequirementError

_DIGEST_A = "a" * 64
_DIGEST_B = "0123456789abcdef" * 4
_VERDICTS = [
    "MET",
    "BREACHED",
    "UNKNOWN",
    "UNKNOWN(inconclusive)",
    "UNKNOWN(insufficient)",
    "UNKNOWN(unresolved)",
    "UNKNOWN(unplaceable)",
    "UNKNOWN(late_evaluation)",
    "UNKNOWN(revision_gap)",
]


def _refused(value: Any) -> RuntimeRequirementError:
    with pytest.raises(RuntimeRequirementError) as info:
        build_candidate(value)
    return info.value


def _assert_closed(error: RuntimeRequirementError) -> None:
    assert error.code == "invalid_request"
    assert error.path == "candidate"
    assert str(error) == "invalid_request at candidate"


def _valid(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "evidence_valid": True,
        "legacy_decision": "Promote",
        "monitor_verdicts": ["MET"],
    }
    base.update(extra)
    return base


class _Trace(Mapping[str, Any]):
    """A mapping that records every access, in order."""

    def __init__(self, data: dict[str, Any], log: list[str] | None = None) -> None:
        self._data = data
        self.log: list[str] = [] if log is None else log

    def __getitem__(self, key: str) -> Any:
        self.log.append(f"getitem:{key}")
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        self.log.append("iter")
        return iter(list(self._data))

    def __len__(self) -> int:
        self.log.append("len")
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        self.log.append(f"contains:{key}")
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        self.log.append(f"get:{key}")
        return self._data.get(key, default)


# ------------------------------------------------------------------ top-level shape


@pytest.mark.parametrize("bad", [None, [], (), "text", 5, {"evidence_valid"}, b"x"])
def test_non_mapping_is_refused_with_the_closed_error(bad: Any) -> None:
    _assert_closed(_refused(bad))


def test_empty_mapping_is_refused() -> None:
    _assert_closed(_refused({}))


@pytest.mark.parametrize(
    "extra",
    [
        {"output_text": "free text"},
        {"Evidence_valid": True},
        {"evidence_valid_": True},
        {"": 1},
        {1: "x"},
    ],
)
def test_any_unknown_key_is_refused_never_dropped(extra: dict[Any, Any]) -> None:
    _assert_closed(_refused({"evidence_valid": False, **extra}))


def test_unknown_key_is_refused_even_when_everything_else_is_valid() -> None:
    _assert_closed(_refused(_valid(extra_key=1)))


def test_key_check_precedes_evidence_check() -> None:
    log: list[str] = []
    mapping = _Trace({"evidence_valid": "yes", "other": 1}, log)
    _assert_closed(_refused(mapping))
    assert log == ["iter"]


@pytest.mark.parametrize("value", [1, 0, "True", None, [], 1.0, "", object()])
def test_evidence_valid_must_be_a_real_bool(value: Any) -> None:
    _assert_closed(_refused({"evidence_valid": value}))


def test_evidence_valid_is_required() -> None:
    _assert_closed(_refused({"legacy_decision": "Promote", "monitor_verdicts": []}))


def test_invalid_evidence_with_only_the_flag_is_accepted() -> None:
    assert build_candidate({"evidence_valid": False}) == {"evidence_valid": False}


@pytest.mark.parametrize(
    "key,value",
    [
        ("legacy_decision", "Promote"),
        ("monitor_verdicts", []),
        ("monitor_verdicts", ["MET"]),
        ("fingerprint", _DIGEST_A),
        ("gate_result_digest", _DIGEST_A),
    ],
)
def test_invalid_evidence_carries_no_other_fact(key: str, value: Any) -> None:
    _assert_closed(_refused({"evidence_valid": False, key: value}))


def test_invalid_evidence_with_all_facts_is_refused() -> None:
    _assert_closed(
        _refused(
            _valid(
                evidence_valid=False,
                fingerprint=_DIGEST_A,
                gate_result_digest=_DIGEST_B,
            )
        )
    )


# ---------------------------------------------------------------- valid evidence


def test_valid_evidence_requires_both_join_inputs() -> None:
    _assert_closed(_refused({"evidence_valid": True}))
    _assert_closed(_refused({"evidence_valid": True, "legacy_decision": "Promote"}))
    _assert_closed(_refused({"evidence_valid": True, "monitor_verdicts": ["MET"]}))
    _assert_closed(
        _refused(
            {
                "evidence_valid": True,
                "fingerprint": _DIGEST_A,
                "gate_result_digest": _DIGEST_B,
            }
        )
    )


def test_minimal_valid_candidate_result_is_exact() -> None:
    assert build_candidate(_valid()) == {
        "evidence_valid": True,
        "legacy_decision": "Promote",
        "monitor_verdicts": ["MET"],
    }


def test_full_candidate_result_is_exact_and_ordered_regardless_of_input_order() -> None:
    out = build_candidate(
        {
            "gate_result_digest": _DIGEST_B,
            "fingerprint": _DIGEST_A,
            "monitor_verdicts": ("MET", "UNKNOWN(revision_gap)"),
            "legacy_decision": "NoDecision",
            "evidence_valid": True,
        }
    )
    assert out == {
        "evidence_valid": True,
        "legacy_decision": "NoDecision",
        "monitor_verdicts": ["MET", "UNKNOWN(revision_gap)"],
        "fingerprint": _DIGEST_A,
        "gate_result_digest": _DIGEST_B,
    }
    assert list(out) == [
        "evidence_valid",
        "legacy_decision",
        "monitor_verdicts",
        "fingerprint",
        "gate_result_digest",
    ]


def test_each_digest_alone_is_accepted_with_valid_evidence() -> None:
    assert build_candidate(_valid(fingerprint=_DIGEST_A))["fingerprint"] == _DIGEST_A
    assert (
        build_candidate(_valid(gate_result_digest=_DIGEST_B))["gate_result_digest"]
        == _DIGEST_B
    )


@pytest.mark.parametrize("decision", ["Promote", "Reject", "NoDecision"])
def test_legacy_decision_vocabulary_is_accepted(decision: str) -> None:
    assert (
        build_candidate(_valid(legacy_decision=decision))["legacy_decision"] == decision
    )


@pytest.mark.parametrize(
    "decision", ["promote", "PROMOTE", "Promote ", "", None, 1, ["Promote"], "Approve"]
)
def test_legacy_decision_outside_the_vocabulary_is_refused(decision: Any) -> None:
    _assert_closed(_refused(_valid(legacy_decision=decision)))


def test_every_verdict_in_the_vocabulary_is_accepted() -> None:
    out = build_candidate(_valid(monitor_verdicts=list(_VERDICTS)))
    assert out["monitor_verdicts"] == _VERDICTS


@pytest.mark.parametrize(
    "verdict",
    [
        "met",
        "UNKNOWN()",
        "UNKNOWN(other)",
        "UNKNOWN(inconclusive) ",
        "",
        None,
        1,
        ["MET"],
    ],
)
def test_verdict_outside_the_vocabulary_is_refused(verdict: Any) -> None:
    _assert_closed(_refused(_valid(monitor_verdicts=["MET", verdict])))


def test_verdict_count_boundary_is_32() -> None:
    assert (
        build_candidate(_valid(monitor_verdicts=["MET"] * 32))["monitor_verdicts"]
        == ["MET"] * 32
    )
    _assert_closed(_refused(_valid(monitor_verdicts=["MET"] * 33)))


def test_empty_verdict_list_counts_as_present_for_valid_evidence() -> None:
    assert build_candidate(_valid(monitor_verdicts=[]))["monitor_verdicts"] == []
    assert build_candidate(_valid(monitor_verdicts=()))["monitor_verdicts"] == []


@pytest.mark.parametrize(
    "bad", ["MET", b"MET", {"MET"}, {"a": "MET"}, None, 5, iter(["MET"])]
)
def test_verdicts_must_be_a_list_or_tuple(bad: Any) -> None:
    _assert_closed(_refused(_valid(monitor_verdicts=bad)))


def test_list_subclass_verdicts_are_accepted_as_a_list() -> None:
    class Verdicts(list[str]):
        pass

    out = build_candidate(_valid(monitor_verdicts=Verdicts(["MET", "BREACHED"])))
    assert out["monitor_verdicts"] == ["MET", "BREACHED"]
    assert type(out["monitor_verdicts"]) is list


@pytest.mark.parametrize(
    "bad", ["a" * 63, "a" * 65, "A" * 64, "g" * 64, "", None, 5, "a" * 64 + "\n"]
)
@pytest.mark.parametrize("field", ["fingerprint", "gate_result_digest"])
def test_digest_must_be_exactly_64_lowercase_hex(field: str, bad: Any) -> None:
    _assert_closed(_refused(_valid(**{field: bad})))


@pytest.mark.parametrize("field", ["fingerprint", "gate_result_digest"])
def test_digest_boundary_length_64_is_accepted(field: str) -> None:
    assert build_candidate(_valid(**{field: "f" * 64}))[field] == "f" * 64


def test_digests_on_valid_evidence_do_not_replace_the_join_requirement() -> None:
    _assert_closed(_refused({"evidence_valid": True, "fingerprint": _DIGEST_A}))


def test_input_mapping_is_not_mutated() -> None:
    data = _valid(fingerprint=_DIGEST_A)
    snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in data.items()}
    build_candidate(data)
    assert data == snapshot


def test_result_verdict_list_is_a_copy() -> None:
    verdicts = ["MET"]
    out = build_candidate(_valid(monitor_verdicts=verdicts))
    assert out["monitor_verdicts"] is not verdicts


# --------------------------------------------------------- access order and counts


def test_access_trace_for_minimal_valid_candidate() -> None:
    mapping = _Trace(_valid())
    build_candidate(mapping)
    assert mapping.log == [
        "iter",
        "get:evidence_valid",
        "getitem:evidence_valid",
        "contains:legacy_decision",
        "getitem:legacy_decision",
        "contains:monitor_verdicts",
        "getitem:monitor_verdicts",
        "contains:fingerprint",
        "contains:gate_result_digest",
    ]


def test_access_trace_for_full_valid_candidate() -> None:
    mapping = _Trace(_valid(fingerprint=_DIGEST_A, gate_result_digest=_DIGEST_B))
    build_candidate(mapping)
    assert mapping.log == [
        "iter",
        "get:evidence_valid",
        "getitem:evidence_valid",
        "contains:legacy_decision",
        "getitem:legacy_decision",
        "contains:monitor_verdicts",
        "getitem:monitor_verdicts",
        "contains:fingerprint",
        "getitem:fingerprint",
        "contains:gate_result_digest",
        "getitem:gate_result_digest",
    ]


def test_access_trace_for_invalid_evidence() -> None:
    mapping = _Trace({"evidence_valid": False})
    assert build_candidate(mapping) == {"evidence_valid": False}
    assert mapping.log == [
        "iter",
        "get:evidence_valid",
        "getitem:evidence_valid",
        "contains:legacy_decision",
        "contains:monitor_verdicts",
        "contains:fingerprint",
        "contains:gate_result_digest",
    ]


def test_invalid_legacy_decision_stops_before_later_fields_are_read() -> None:
    mapping = _Trace(_valid(legacy_decision="nope", fingerprint=_DIGEST_A))
    _assert_closed(_refused(mapping))
    assert mapping.log == [
        "iter",
        "get:evidence_valid",
        "getitem:evidence_valid",
        "contains:legacy_decision",
        "getitem:legacy_decision",
    ]


def test_invalid_verdicts_stop_before_digests_are_read() -> None:
    mapping = _Trace(_valid(monitor_verdicts="MET", fingerprint=_DIGEST_A))
    _assert_closed(_refused(mapping))
    assert mapping.log[-2:] == ["contains:monitor_verdicts", "getitem:monitor_verdicts"]
    assert "contains:fingerprint" not in mapping.log


def test_invalid_fingerprint_stops_before_gate_digest_is_read() -> None:
    mapping = _Trace(_valid(fingerprint="bad", gate_result_digest=_DIGEST_A))
    _assert_closed(_refused(mapping))
    assert mapping.log[-2:] == ["contains:fingerprint", "getitem:fingerprint"]
    assert "contains:gate_result_digest" not in mapping.log


def test_evidence_flag_is_read_by_get_and_again_by_getitem() -> None:
    class Flipping(Mapping[str, Any]):
        def __init__(self) -> None:
            self.reads = 0

        def __getitem__(self, key: str) -> Any:
            assert key == "evidence_valid"
            return False

        def get(self, key: str, default: Any = None) -> Any:
            self.reads += 1
            return True

        def __iter__(self) -> Iterator[str]:
            return iter(["evidence_valid"])

        def __len__(self) -> int:
            return 1

        def __contains__(self, key: object) -> bool:
            return False

    mapping = Flipping()
    # get says True (passes the bool check), the later subscript read says False
    assert build_candidate(mapping) == {"evidence_valid": False}
    assert mapping.reads == 1


def test_evidence_flag_read_by_getitem_true_after_get_true_requires_join_inputs() -> (
    None
):
    class Truthy(_Trace):
        def get(self, key: str, default: Any = None) -> Any:
            super().get(key, default)
            return True

    mapping = Truthy({"evidence_valid": True})
    _assert_closed(_refused(mapping))


def test_raising_membership_check_propagates_unchanged() -> None:
    class Boom(_Trace):
        def __contains__(self, key: object) -> bool:
            raise RuntimeError("contains failed")

    with pytest.raises(RuntimeError, match="contains failed"):
        build_candidate(Boom({"evidence_valid": False}))


def test_raising_item_read_propagates_unchanged() -> None:
    class Boom(_Trace):
        def __getitem__(self, key: str) -> Any:
            if key == "monitor_verdicts":
                raise RuntimeError("item failed")
            return super().__getitem__(key)

    with pytest.raises(RuntimeError, match="item failed"):
        build_candidate(Boom(_valid()))


def test_raising_iteration_during_key_check_propagates() -> None:
    class Boom(_Trace):
        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("iter failed")

    with pytest.raises(RuntimeError, match="iter failed"):
        build_candidate(Boom(_valid()))


def test_raise_once_item_read_fails_on_first_read_only() -> None:
    class Once(_Trace):
        armed = True

        def __getitem__(self, key: str) -> Any:
            if key == "legacy_decision" and self.armed:
                self.armed = False
                raise RuntimeError("first read failed")
            return super().__getitem__(key)

    mapping = Once(_valid())
    with pytest.raises(RuntimeError, match="first read failed"):
        build_candidate(mapping)
    assert build_candidate(mapping)["legacy_decision"] == "Promote"


def test_verdict_container_is_measured_then_iterated_once() -> None:
    events: list[str] = []

    class Verdicts(list[str]):
        def __len__(self) -> int:
            events.append("len")
            return super().__len__()

        def __iter__(self) -> Iterator[str]:
            events.append("iter")
            return super().__iter__()

    out = build_candidate(_valid(monitor_verdicts=Verdicts(["MET", "BREACHED"])))
    assert out["monitor_verdicts"] == ["MET", "BREACHED"]
    assert events == ["len", "iter"]


def test_lying_length_over_the_cap_is_refused_before_iteration() -> None:
    events: list[str] = []

    class Verdicts(list[str]):
        def __len__(self) -> int:
            events.append("len")
            return 33

        def __iter__(self) -> Iterator[str]:
            events.append("iter")
            return super().__iter__()

    _assert_closed(_refused(_valid(monitor_verdicts=Verdicts(["MET"]))))
    assert events == ["len"]


def test_raising_verdict_iteration_propagates() -> None:
    class Verdicts(list[str]):
        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("iter failed")

    with pytest.raises(RuntimeError, match="iter failed"):
        build_candidate(_valid(monitor_verdicts=Verdicts(["MET"])))


def test_key_str_subclass_with_scripted_equality_is_judged_by_that_equality() -> None:
    class NeverEqual(str):
        def __eq__(self, other: object) -> bool:
            return False

        def __ne__(self, other: object) -> bool:
            return True

        __hash__ = str.__hash__

    _assert_closed(_refused({NeverEqual("evidence_valid"): False}))


def test_verdict_str_subclass_is_returned_as_given() -> None:
    class Tagged(str):
        pass

    verdict = Tagged("MET")
    out = build_candidate(_valid(monitor_verdicts=[verdict]))
    assert out["monitor_verdicts"][0] is verdict


def test_digest_str_subclass_is_returned_as_given() -> None:
    class Tagged(str):
        pass

    digest = Tagged(_DIGEST_A)
    out = build_candidate(_valid(fingerprint=digest))
    assert out["fingerprint"] is digest
