"""Characterization of response field lookup used for observed provider versions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from traigent.identity import provider_versions as pv

Events = list[tuple[str, str]]
_NAMES = ("model", "model_name", "model_id")


class Probe:
    """An object whose attributes are read through logged accessors."""

    def __init__(
        self,
        label: str,
        events: Events,
        values: dict[str, Any],
        raising: dict[str, BaseException] | None = None,
        raise_once: set[str] | None = None,
    ) -> None:
        self._label = label
        self._events = events
        self._values = values
        self._raising = raising or {}
        self._raise_once = raise_once or set()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self._events.append((self._label, name))
        if name in self._raise_once:
            self._raise_once.discard(name)
            raise RuntimeError("once " + name)
        if name in self._raising:
            raise self._raising[name]
        if name in self._values:
            return self._values[name]
        raise AttributeError(name)


class MapProbe(Mapping[str, Any]):
    """A Mapping whose ``get`` calls are logged."""

    def __init__(
        self,
        label: str,
        events: Events,
        values: dict[str, Any],
        raising: dict[str, BaseException] | None = None,
    ) -> None:
        self._label = label
        self._events = events
        self._values = values
        self._raising = raising or {}

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str, default: Any = None) -> Any:
        self._events.append((self._label, key))
        if key in self._raising:
            raise self._raising[key]
        return self._values.get(key, default)


def test_plain_response_attribute_first_match_wins() -> None:
    events: Events = []
    response = Probe("r", events, {"model_name": "b", "model_id": "c"})
    assert pv._from_response(response, _NAMES) == "b"
    assert events == [
        ("r", "response_metadata"),
        ("r", "llm_output"),
        ("r", "model"),
        ("r", "model_name"),
    ]


def test_metadata_sources_are_scanned_after_the_response_in_order() -> None:
    events: Events = []
    meta = MapProbe("m", events, {"model_id": "from-meta"})
    out = MapProbe("o", events, {"model": "from-llm-output"})
    response = Probe("r", events, {"response_metadata": meta, "llm_output": out})
    assert pv._from_response(response, _NAMES) == "from-meta"
    assert events == [
        ("r", "response_metadata"),
        ("r", "llm_output"),
        ("r", "model"),
        ("r", "model_name"),
        ("r", "model_id"),
        ("m", "model"),
        ("m", "model_name"),
        ("m", "model_id"),
    ]


def test_llm_output_scanned_last_and_all_names_tried() -> None:
    events: Events = []
    meta = MapProbe("m", events, {})
    out = MapProbe("o", events, {"model_id": "late"})
    response = Probe("r", events, {"response_metadata": meta, "llm_output": out})
    assert pv._from_response(response, _NAMES) == "late"
    assert [e for e in events if e[0] == "o"] == [
        ("o", "model"),
        ("o", "model_name"),
        ("o", "model_id"),
    ]


def test_mapping_response_uses_get_for_metadata_and_names() -> None:
    events: Events = []
    meta = MapProbe("m", events, {"model": "meta-model"})
    response = MapProbe("r", events, {"response_metadata": meta})
    assert pv._from_response(response, _NAMES) == "meta-model"
    assert events == [
        ("r", "response_metadata"),
        ("r", "llm_output"),
        ("r", "model"),
        ("r", "model_name"),
        ("r", "model_id"),
        ("m", "model"),
    ]


def test_non_mapping_metadata_is_ignored() -> None:
    events: Events = []
    meta = Probe("m", events, {"model": "ignored"})
    response = Probe("r", events, {"response_metadata": meta, "llm_output": "text"})
    assert pv._from_response(response, _NAMES) is None
    assert all(label != "m" for label, _ in events)


def test_no_match_returns_none_and_names_are_exhausted() -> None:
    events: Events = []
    response = Probe("r", events, {})
    assert pv._from_response(response, ("a", "b")) is None
    assert events == [
        ("r", "response_metadata"),
        ("r", "llm_output"),
        ("r", "a"),
        ("r", "b"),
    ]


def test_empty_names_returns_none_after_metadata_reads() -> None:
    events: Events = []
    assert pv._from_response(Probe("r", events, {}), ()) is None
    assert events == [("r", "response_metadata"), ("r", "llm_output")]


@pytest.mark.parametrize("bad", [None, "", 5, b"x", ["a"], "x" * 257, {"a": 1}])
def test_invalid_values_are_skipped_in_favor_of_later_names(bad: Any) -> None:
    response = Probe("r", [], {"model": bad, "model_name": "ok"})
    assert pv._from_response(response, _NAMES) == "ok"


def test_text_length_boundaries() -> None:
    assert pv._from_response(Probe("r", [], {"model": "x"}), _NAMES) == "x"
    at_cap = "y" * 256
    assert pv._from_response(Probe("r", [], {"model": at_cap}), _NAMES) == at_cap
    assert pv._from_response(Probe("r", [], {"model": at_cap + "y"}), _NAMES) is None


def test_str_subclass_is_returned_unchanged() -> None:
    class S(str):
        pass

    value = S("sub")
    result = pv._from_response(Probe("r", [], {"model": value}), _NAMES)
    assert result is value


def test_value_access_exceptions_mean_unknown_and_scan_continues() -> None:
    events: Events = []
    response = Probe(
        "r",
        events,
        {"model_name": "after"},
        raising={"model": RuntimeError("bad property")},
    )
    assert pv._from_response(response, _NAMES) == "after"
    assert events[-2:] == [("r", "model"), ("r", "model_name")]


def test_mapping_get_exception_means_unknown() -> None:
    events: Events = []
    meta = MapProbe("m", events, {"model_id": "ok"}, raising={"model": ValueError("x")})
    response = Probe("r", events, {"response_metadata": meta})
    assert pv._from_response(response, _NAMES) == "ok"


def test_raise_once_value_accessor_is_not_retried() -> None:
    events: Events = []
    response = Probe("r", events, {"model": "second"}, raise_once={"model"})
    assert pv._from_response(response, ("model",)) is None
    assert events.count(("r", "model")) == 1
    # the accessor now succeeds, proving the first read failed inside the guard
    assert pv._from_response(response, ("model",)) == "second"


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_base_exceptions_from_value_access_propagate(exc: type[BaseException]) -> None:
    response = Probe("r", [], {}, raising={"model": exc()})
    with pytest.raises(exc):
        pv._from_response(response, _NAMES)


@pytest.mark.parametrize("attr", ["response_metadata", "llm_output"])
def test_metadata_accessor_exceptions_propagate_unguarded(attr: str) -> None:
    events: Events = []
    response = Probe("r", events, {"model": "m"}, raising={attr: RuntimeError("meta")})
    with pytest.raises(RuntimeError, match="meta"):
        pv._from_response(response, _NAMES)
    assert ("r", "model") not in events


@pytest.mark.parametrize("attr", ["response_metadata", "llm_output"])
def test_mapping_metadata_get_exceptions_propagate_unguarded(attr: str) -> None:
    events: Events = []
    response = MapProbe(
        "r", events, {"model": "m"}, raising={attr: RuntimeError("meta")}
    )
    with pytest.raises(RuntimeError, match="meta"):
        pv._from_response(response, _NAMES)
    assert ("r", "model") not in events


def test_raise_once_metadata_accessor_is_not_retried() -> None:
    events: Events = []
    response = Probe("r", events, {"model": "m"}, raise_once={"response_metadata"})
    with pytest.raises(RuntimeError, match="once"):
        pv._from_response(response, _NAMES)
    assert events == [("r", "response_metadata")]


def test_observation_key_full_results() -> None:
    response = Probe("r", [], {"model_id": "served", "system_fingerprint": "fp"})
    assert pv.observation_key(response, provider="OpenAI", requested_model="gpt") == (
        "openai",
        "gpt",
        "served",
        "fp",
    )
    assert pv.observation_key(
        Probe("r", [], {}), provider="p", requested_model="m"
    ) == (
        "p",
        "m",
        None,
        None,
    )


@pytest.mark.parametrize(
    ("provider", "requested"),
    [("", "m"), ("-x", "m"), ("p" * 65, "m"), (None, "m"), ("p", None), ("p", "")],
)
def test_observation_key_records_nothing_for_unrepresentable_inputs(
    provider: Any, requested: Any
) -> None:
    events: Events = []
    response = Probe("r", events, {"model": "a"})
    assert (
        pv.observation_key(response, provider=provider, requested_model=requested)
        is None
    )
    assert events == []


def test_observation_key_reads_model_names_then_fingerprint() -> None:
    events: Events = []
    response = Probe("r", events, {})
    pv.observation_key(response, provider="p", requested_model="m")
    assert events == [
        ("r", "response_metadata"),
        ("r", "llm_output"),
        ("r", "model"),
        ("r", "model_name"),
        ("r", "model_id"),
        ("r", "response_metadata"),
        ("r", "llm_output"),
        ("r", "system_fingerprint"),
    ]
