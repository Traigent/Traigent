"""Characterize default experiment names before extracting their formatter."""

from collections.abc import Iterator

import pytest

from traigent.api.decorators import (
    _build_default_experiment_name,
    _resolve_objective_schema,
)


@pytest.mark.parametrize(
    ("func_name", "objectives", "configuration_space", "max_knobs", "max_total_length", "expected"),
    [
        ("worker", None, None, 4, 120, "worker"),
        ("worker", ["accuracy"], None, 4, 120, "worker[accuracy]"),
        ("worker", None, {"model": [], "temperature": []}, 4, 120, "worker[model,temperature]"),
        ("worker", None, {"a": [], "b": [], "c": [], "d": [], "e": []}, 4, 120, "worker[a,b,c,d,...]"),
        ("abc", None, None, 4, 3, "abc"),
        ("abc", None, None, 4, 2, "a~"),
        ("abc", None, None, 4, 1, "a"),
        ("worker", ["accuracy"], {"model": []}, 4, 17, "[accuracy][model]"),
        ("worker", ["accuracy"], {"model": []}, 4, 16, "[accuracy]"),
        ("worker", ["accuracy"], {"model": []}, 4, 0, ""),
        ("worker", ["accuracy"], {"model": []}, 4, -1, "[accuracy"),
        ("abcdefgh", ["accuracy"], None, 4, 11, "a[accuracy]"),
        ("🧪éabc", ["accuracy"], None, 4, 12, "🧪~[accuracy]"),
    ],
)
def test_default_experiment_name_exact_output(
    func_name: str,
    objectives: list[str] | None,
    configuration_space: dict[str, list[object]] | None,
    max_knobs: int,
    max_total_length: int,
    expected: str,
) -> None:
    schema = _resolve_objective_schema(objectives) if objectives is not None else None
    assert (
        _build_default_experiment_name(
            func_name,
            schema,
            configuration_space,
            max_knobs=max_knobs,
            max_total_length=max_total_length,
        )
        == expected
    )


class RepeatedKeysMapping(dict[str, object]):
    """Return different keys on successive reads to pin fallback read order."""

    def __init__(self, responses: list[list[str]]) -> None:
        super().__init__()
        self.responses = iter(responses)
        self.keys_calls = 0

    def __bool__(self) -> bool:
        return True

    def keys(self) -> Iterator[str]:
        self.keys_calls += 1
        return iter(next(self.responses))


def test_overbudget_suffix_reads_mapping_keys_again_for_fallback() -> None:
    config = RepeatedKeysMapping([["first"], ["second"]])
    schema = _resolve_objective_schema(["accuracy"])

    result = _build_default_experiment_name(
        "worker", schema, config, max_total_length=12
    )

    assert result == "[accuracy]"
    assert config.keys_calls == 2
