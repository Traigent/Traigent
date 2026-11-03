"""Golden characterization for tuned_variables.discovery.discover_callables."""

from __future__ import annotations

import types

from traigent.tuned_variables.discovery import discover_callables


def _module():
    m = types.ModuleType("charmod")

    def retrieve_a(query: str, k: int = 1) -> list:
        return []

    def search_b(query: str) -> str:
        return ""

    def _private(query: str) -> int:
        return 0

    def other(x):
        return x

    def noann(query):
        return None

    for f in (retrieve_a, search_b, _private, other, noann):
        f.__module__ = "charmod"
        setattr(m, f.__name__, f)

    def imported(query):
        return None

    imported.__module__ = "elsewhere"
    m.imported = imported
    m.not_a_function = 5
    return m


def test_default_excludes_private_and_imported():
    out = discover_callables(_module())
    assert sorted(out) == ["noann", "other", "retrieve_a", "search_b"]
    assert out["retrieve_a"].module == "charmod"


def test_private_toggle_and_pattern_uses_match():
    m = _module()
    assert "_private" in discover_callables(m, include_private=True)
    out = discover_callables(m, pattern=r"^(retrieve|search)_")
    assert sorted(out) == ["retrieve_a", "search_b"]
    assert discover_callables(m, pattern="etrieve") == {}


def test_required_params_and_empty_list_is_no_filter():
    m = _module()
    assert sorted(discover_callables(m, required_params=["query"])) == [
        "noann",
        "retrieve_a",
        "search_b",
    ]
    assert sorted(discover_callables(m, required_params=["query", "k"])) == [
        "retrieve_a"
    ]
    assert len(discover_callables(m, required_params=[])) == 4


def test_return_type_filter_missing_annotation_passes():
    m = _module()
    # string annotations (postponed evaluation) cannot be issubclass-checked
    # and are treated as compatible
    assert sorted(discover_callables(m, return_type=list)) == [
        "noann",
        "other",
        "retrieve_a",
        "search_b",
    ]


def test_signature_failure_skips(monkeypatch):
    import inspect

    from traigent.tuned_variables import discovery

    real = inspect.signature

    def fake(obj, *a, **k):
        if getattr(obj, "__name__", "") == "other":
            raise ValueError("no sig")
        return real(obj, *a, **k)

    monkeypatch.setattr(discovery.inspect, "signature", fake)
    assert "other" not in discover_callables(_module())


def test_combined_filters_and_empty_pattern():
    m = _module()
    out = discover_callables(m, pattern="s", required_params=["query"], return_type=str)
    assert sorted(out) == ["search_b"]
    assert len(discover_callables(m, pattern="")) == 4
