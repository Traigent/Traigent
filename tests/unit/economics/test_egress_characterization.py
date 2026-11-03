"""Characterization of characterization-egress enforcement: order, reads, messages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from traigent.economics.egress import enforce_characterization_egress
from traigent.economics.errors import EgressPolicyError


class TracedDict(Mapping[str, Any]):
    """A read-only mapping recording get/membership reads into a shared list."""

    def __init__(
        self,
        name: str,
        events: list[tuple[str, str, str]],
        data: dict[str, Any],
        raise_on: dict[str, Exception] | None = None,
    ) -> None:
        self._data = dict(data)
        self._name = name
        self._events = events
        self._raise_on = raise_on or {}

    def __getitem__(self, key: str) -> Any:
        self._events.append((self._name, "item", key))
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        self._events.append((self._name, "get", key))
        if key in self._raise_on:
            raise self._raise_on[key]
        return self._data.get(key, default)

    def __contains__(self, key: object) -> bool:
        self._events.append((self._name, "in", str(key)))
        return key in self._data


def _report(field: str = "value_channel", **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "field": field,
        "provenance": "asked",
        "sharing_outcome": "shared",
    }
    base.update(over)
    return base


def _error(characterization: Any) -> str:
    with pytest.raises(EgressPolicyError) as info:
        enforce_characterization_egress(characterization)
    return str(info.value)


def test_valid_inputs_return_none_and_do_not_mutate() -> None:
    chars: dict[str, Any] = {
        "bands": {"value_channel": "x"},
        "overrides": {"daily_volume_band": "y"},
        "field_reports": [
            _report("value_channel"),
            _report("daily_volume_band"),
            _report(
                "error_cost_band",
                sharing_outcome="withheld_by_policy",
                provenance="inferred",
                evidence_status="withheld_by_policy",
            ),
        ],
    }
    snapshot = repr(chars)
    assert enforce_characterization_egress(chars) is None
    assert repr(chars) == snapshot


@pytest.mark.parametrize(
    ("chars", "message"),
    [
        (None, "characterization must be an object"),
        ([], "characterization must be an object"),
        ({}, "characterization.field_reports must be a non-empty list"),
        (
            {"field_reports": ()},
            "characterization.field_reports must be a non-empty list",
        ),
        (
            {"field_reports": []},
            "characterization.field_reports must be a non-empty list",
        ),
        (
            {"field_reports": ["x"]},
            "each field report must be an object",
        ),
        (
            {"field_reports": [{}]},
            "a field report names a field outside the characterization allowlist",
        ),
        (
            {"field_reports": [{"field": 3}]},
            "a field report names a field outside the characterization allowlist",
        ),
        (
            {"field_reports": [{"field": "nope"}]},
            "a field report names a field outside the characterization allowlist",
        ),
        (
            {"field_reports": [{"field": "value_channel"}]},
            "field 'value_channel' has an invalid or missing sharing_outcome",
        ),
        (
            {"field_reports": [{"field": "value_channel", "sharing_outcome": "x"}]},
            "field 'value_channel' has an invalid or missing sharing_outcome",
        ),
        (
            {"field_reports": [_report(provenance="bogus")]},
            "field 'value_channel' has an invalid or missing provenance",
        ),
        (
            {"bands": 3, "field_reports": [_report()]},
            "characterization.bands must be an object",
        ),
        (
            {"overrides": [], "field_reports": [_report()]},
            "characterization.overrides must be an object",
        ),
        (
            {"bands": {"zzz": 1}, "field_reports": [_report()]},
            "characterization.bands carries a field outside the allowlist",
        ),
        (
            {"overrides": {"zzz": 1}, "field_reports": [_report()]},
            "characterization.overrides carries a field outside the allowlist",
        ),
        (
            {"field_reports": [_report(), _report()]},
            "field 'value_channel' has more than one field report "
            "(duplicate or contradictory reports are rejected locally)",
        ),
        (
            {"field_reports": [_report()]},
            "field 'value_channel' has a shared report but no transmitted value",
        ),
        (
            {
                "bands": {"value_channel": 1, "daily_volume_band": 2},
                "field_reports": [_report()],
            },
            "field 'daily_volume_band' is transmitted without a shared field report",
        ),
        (
            {
                "bands": {"value_channel": 1},
                "field_reports": [_report(sharing_outcome="withheld_by_policy")],
            },
            "field 'value_channel' is marked withheld but its value is present "
            "(withheld values must never leave the machine)",
        ),
    ],
)
def test_exact_messages(chars: Any, message: str) -> None:
    assert _error(chars) == message


def test_non_mapping_characterization_is_refused_before_any_read() -> None:
    class Reads(list[Any]):
        def get(self, *args: Any) -> Any:  # pragma: no cover - must not run
            raise AssertionError("read before type check")

    assert _error(Reads()) == "characterization must be an object"


# Provenance / evidence rules (inside the per-report check).
@pytest.mark.parametrize(
    ("over", "message"),
    [
        (
            {"provenance": "inferred"},
            "inferred field 'value_channel' must declare an evidence_status",
        ),
        (
            {"provenance": "inferred", "evidence_status": "zzz"},
            "field 'value_channel' has an invalid evidence_status",
        ),
        (
            {"provenance": "inferred", "evidence_status": "provided"},
            "field 'value_channel' declares provided evidence but carries no pointer",
        ),
        (
            {"evidence_status": "provided"},
            "asked field 'value_channel' must not claim inference evidence",
        ),
        (
            {"provenance": "defaulted", "evidence_pointer": "p"},
            "defaulted field 'value_channel' must not claim inference evidence",
        ),
        (
            {
                "provenance": "inferred",
                "evidence_status": "withheld_by_policy",
                "evidence_pointer": "p",
            },
            "field 'value_channel' carries an evidence pointer without provided evidence",
        ),
        (
            {
                "provenance": "inferred",
                "evidence_status": "provided",
                "evidence_pointer": "p",
                "sharing_outcome": "withheld_by_policy",
            },
            "withheld field 'value_channel' must not carry an evidence pointer",
        ),
        (
            {
                "provenance": "inferred",
                "evidence_status": "provided",
                "sharing_outcome": "withheld_by_policy",
            },
            "field 'value_channel' declares provided evidence but carries no pointer",
        ),
    ],
)
def test_provenance_and_evidence_messages(over: dict[str, Any], message: str) -> None:
    assert (
        _error({"bands": {"value_channel": 1}, "field_reports": [_report(**over)]})
        == message
    )


def test_withheld_report_with_withheld_evidence_status_is_accepted() -> None:
    chars = {
        "field_reports": [
            _report(
                sharing_outcome="withheld_by_policy",
                provenance="inferred",
                evidence_status="withheld_by_policy",
            )
        ]
    }
    assert enforce_characterization_egress(chars) is None


def test_order_present_names_checked_before_reports() -> None:
    chars = {"bands": {"zzz": 1}, "field_reports": [{"field": "nope"}]}
    # field_reports non-emptiness first, then present names, then reports.
    assert (
        _error(chars) == "characterization.bands carries a field outside the allowlist"
    )


def test_order_reports_empty_before_present() -> None:
    chars = {"bands": 3, "field_reports": []}
    assert _error(chars) == "characterization.field_reports must be a non-empty list"


def test_order_bands_before_overrides() -> None:
    chars = {"bands": 3, "overrides": 3, "field_reports": [_report()]}
    assert _error(chars) == "characterization.bands must be an object"


def test_order_report_type_before_field() -> None:
    assert _error({"field_reports": [{"field": "nope"}, "x"]}) == (
        "a field report names a field outside the characterization allowlist"
    )
    assert _error({"field_reports": ["x", {"field": "nope"}]}) == (
        "each field report must be an object"
    )


def test_order_duplicate_before_sharing_before_provenance() -> None:
    both_bad = {"field": "value_channel", "sharing_outcome": "bad", "provenance": "bad"}
    # first report is fine, the second is a duplicate AND has bad outcome.
    chars = {"field_reports": [_report(), dict(both_bad)]}
    assert _error(chars).startswith(
        "field 'value_channel' has more than one field report"
    )
    # a non-duplicate with bad sharing and bad provenance reports sharing.
    chars2 = {"field_reports": [dict(both_bad)]}
    assert (
        _error(chars2)
        == "field 'value_channel' has an invalid or missing sharing_outcome"
    )


def test_order_provenance_before_final_present_checks() -> None:
    chars = {
        "bands": {"daily_volume_band": 1},
        "field_reports": [_report(provenance="x")],
    }
    assert _error(chars).endswith("invalid or missing provenance")


def test_order_per_report_errors_before_later_reports_are_inspected() -> None:
    chars = {"field_reports": [{"field": "nope"}, _report(provenance="x")]}
    assert _error(chars) == (
        "a field report names a field outside the characterization allowlist"
    )


def test_order_present_checks_withheld_then_unreported_then_empty() -> None:
    # leaked wins over unreported.
    chars = {
        "bands": {"value_channel": 1, "daily_volume_band": 1},
        "field_reports": [_report(sharing_outcome="withheld_by_policy")],
    }
    assert "marked withheld" in _error(chars)
    # unreported wins over empty-shared.
    chars2 = {
        "bands": {"daily_volume_band": 1},
        "field_reports": [_report()],
    }
    assert "transmitted without a shared field report" in _error(chars2)


def test_sorted_first_field_named_when_several_violate() -> None:
    chars = {
        "bands": {"value_channel": 1, "daily_volume_band": 1, "error_cost_band": 1},
        "field_reports": [
            _report("value_channel", sharing_outcome="withheld_by_policy"),
            _report("daily_volume_band", sharing_outcome="withheld_by_policy"),
        ],
    }
    assert _error(chars).startswith("field 'daily_volume_band' is marked withheld")


def test_read_trace_valid_path() -> None:
    events: list[tuple[str, str, str]] = []
    r1 = TracedDict("r1", events, _report("value_channel"))
    r2 = TracedDict("r2", events, _report("daily_volume_band"))
    chars = TracedDict(
        "c",
        events,
        {
            "bands": {"value_channel": 1},
            "overrides": {"daily_volume_band": 1},
            "field_reports": [r1, r2],
        },
    )
    enforce_characterization_egress(chars)
    assert events == [
        ("c", "get", "field_reports"),
        ("c", "get", "bands"),
        ("c", "get", "overrides"),
        ("r1", "get", "field"),
        ("r1", "get", "sharing_outcome"),
        ("r1", "get", "provenance"),
        ("r1", "in", "evidence_status"),
        ("r1", "in", "evidence_pointer"),
        ("r1", "get", "evidence_status"),
        ("r1", "get", "sharing_outcome"),
        ("r2", "get", "field"),
        ("r2", "get", "sharing_outcome"),
        ("r2", "get", "provenance"),
        ("r2", "in", "evidence_status"),
        ("r2", "in", "evidence_pointer"),
        ("r2", "get", "evidence_status"),
        ("r2", "get", "sharing_outcome"),
    ]


def test_read_trace_stops_at_first_failure() -> None:
    events: list[tuple[str, str, str]] = []
    r1 = TracedDict("r1", events, {"field": "value_channel"})
    r2 = TracedDict("r2", events, _report("daily_volume_band"))
    chars = TracedDict("c", events, {"field_reports": [r1, r2]})
    _error(chars)
    assert events == [
        ("c", "get", "field_reports"),
        ("c", "get", "bands"),
        ("c", "get", "overrides"),
        ("r1", "get", "field"),
        ("r1", "get", "sharing_outcome"),
    ]


@pytest.mark.parametrize(
    "key,prefix",
    [
        ("field", ["field"]),
        ("sharing_outcome", ["field", "sharing_outcome"]),
        ("provenance", ["field", "sharing_outcome", "provenance"]),
    ],
)
def test_raising_report_accessor_propagates_with_partial_trace(
    key: str, prefix: list[str]
) -> None:
    events: list[tuple[str, str, str]] = []
    report = TracedDict(
        "r", events, _report(), raise_on={key: RuntimeError("boom " + key)}
    )
    chars = {"field_reports": [report]}
    with pytest.raises(RuntimeError, match="boom " + key):
        enforce_characterization_egress(chars)
    assert [e[2] for e in events] == prefix


def test_raise_once_report_accessor_is_not_retried() -> None:
    events: list[tuple[str, str, str]] = []
    calls = {"n": 0}

    class Flaky(TracedDict):
        def get(self, key: str, default: Any = None) -> Any:
            if key == "sharing_outcome":
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("once")
            return super().get(key, default)

    report = Flaky("r", events, _report())
    with pytest.raises(RuntimeError, match="once"):
        enforce_characterization_egress({"field_reports": [report]})
    assert calls["n"] == 1


def test_raising_characterization_accessors_propagate() -> None:
    for key in ("field_reports", "bands", "overrides"):
        events: list[tuple[str, str, str]] = []
        chars = TracedDict(
            "c",
            events,
            {"field_reports": [_report()]},
            raise_on={key: RuntimeError("bad " + key)},
        )
        with pytest.raises(RuntimeError, match="bad " + key):
            enforce_characterization_egress(chars)


class CountingStr(str):
    """A str recording hash/eq calls."""

    hashes = 0
    eqs = 0

    def __hash__(self) -> int:
        type(self).hashes += 1
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        type(self).eqs += 1
        return super().__eq__(other)


def test_field_and_sharing_hash_eq_counts_pinned() -> None:
    class Field(CountingStr):
        hashes = 0
        eqs = 0

    class Sharing(CountingStr):
        hashes = 0
        eqs = 0

    chars = {
        "bands": {"value_channel": 1},
        "field_reports": [
            {
                "field": Field("value_channel"),
                "provenance": "asked",
                "sharing_outcome": Sharing("shared"),
            }
        ],
    }
    enforce_characterization_egress(chars)
    counts = (Field.hashes, Field.eqs, Sharing.hashes, Sharing.eqs)
    assert counts == (4, 3, 1, 3)


def test_input_iterables_and_blocks_untouched() -> None:
    bands = {"value_channel": 1}
    reports = [_report()]
    enforce_characterization_egress({"bands": bands, "field_reports": reports})
    assert bands == {"value_channel": 1}
    assert reports == [_report()]
