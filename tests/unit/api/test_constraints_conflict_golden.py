"""Golden characterization tests for ``check_constraints_conflict``."""

from __future__ import annotations

from typing import Any

from traigent.api.constraints import ConstraintConflict, check_constraints_conflict


class _Fake:
    """Duck-typed constraint that records calls into a shared log."""

    def __init__(self, name: str, ok: Any, log: list[Any]) -> None:
        self.name = name
        self._ok = ok
        self._log = log

    def _collect_tvars(self, var_names: dict[int, str], seen: list[Any]) -> None:
        self._log.append(("collect", self.name))
        var_names[id(self)] = self.name

    def evaluate(self, config: dict[str, Any], var_names: dict[int, str]) -> Any:
        self._log.append(("eval", self.name, dict(config)))
        return self._ok(config) if callable(self._ok) else self._ok

    def explain(self, var_names: dict[int, str]) -> str:
        self._log.append(("explain", self.name))
        return f"explain-{self.name}"


def test_empty_inputs_return_none_without_calls() -> None:
    log: list[Any] = []
    c = _Fake("a", False, log)
    assert check_constraints_conflict([], [{"x": 1}], samples_exhaustive=True) is None
    assert check_constraints_conflict([c], [], samples_exhaustive=True) is None
    assert check_constraints_conflict([c], None, samples_exhaustive=True) is None
    assert log == []


def test_satisfying_sample_short_circuits_in_order() -> None:
    log: list[Any] = []
    a = _Fake("a", lambda cfg: cfg["x"] > 0, log)
    b = _Fake("b", True, log)
    result = check_constraints_conflict(
        [a, b], [{"x": 0}, {"x": 1}, {"x": 2}], samples_exhaustive=True
    )
    assert result is None
    assert log == [
        ("collect", "a"),
        ("collect", "b"),
        ("eval", "a", {"x": 0}),
        ("explain", "a"),
        ("eval", "b", {"x": 0}),
        ("eval", "a", {"x": 1}),
        ("eval", "b", {"x": 1}),
    ]


def test_non_exhaustive_returns_none_after_scanning_all_samples() -> None:
    log: list[Any] = []
    a = _Fake("a", False, log)
    b = _Fake("b", False, log)
    assert check_constraints_conflict([a, b], [{"x": 1}, {"x": 2}]) is None
    assert [e[0] for e in log].count("eval") == 4
    assert [e[0] for e in log].count("explain") == 4


def test_single_constraint_exhaustive_returns_none() -> None:
    log: list[Any] = []
    a = _Fake("a", False, log)
    assert check_constraints_conflict([a], [{"x": 1}], samples_exhaustive=True) is None


def test_exhaustive_conflict_aggregates_unique_constraints_in_first_seen_order() -> (
    None
):
    log: list[Any] = []
    a = _Fake("a", lambda cfg: cfg["x"] == 1, log)
    b = _Fake("b", lambda cfg: cfg["x"] == 2, log)
    c = _Fake("c", lambda cfg: cfg["x"] == 3, log)
    samples = [{"x": 1}, {"x": 2}, {"x": 9}]
    # No sample satisfies all three; the sample with the most violations wins.
    result = check_constraints_conflict([a, b, c], samples, samples_exhaustive=True)
    assert isinstance(result, ConstraintConflict)
    # x=1 violates b,c (2); x=2 violates a,c (2); x=9 violates a,b,c (3)
    assert result.config == {"x": 9}
    # First-seen order: sample x=1 yields b, c; then x=2 adds a.
    assert result.constraints == [b, c, a]
    assert result.messages == ["explain-b", "explain-c", "explain-a"]


def test_tie_picks_first_config_with_max_violations() -> None:
    log: list[Any] = []
    a = _Fake("a", False, log)
    b = _Fake("b", False, log)
    s1, s2 = {"x": 1}, {"x": 2}
    result = check_constraints_conflict([a, b], [s1, s2], samples_exhaustive=True)
    assert result is not None
    assert result.config is s1
    assert result.constraints == [a, b]
    assert result.messages == ["explain-a", "explain-b"]


def test_provided_var_names_skips_collection() -> None:
    log: list[Any] = []
    a = _Fake("a", False, log)
    b = _Fake("b", False, log)
    check_constraints_conflict(
        [a, b], [{"x": 1}], var_names={}, samples_exhaustive=True
    )
    assert not any(e[0] == "collect" for e in log)


def test_explain_called_for_every_violation_even_when_not_exhaustive() -> None:
    log: list[Any] = []
    a = _Fake("a", lambda cfg: 0, log)  # falsy non-bool
    assert check_constraints_conflict([a], [{"x": 1}]) is None
    assert ("explain", "a") in log
