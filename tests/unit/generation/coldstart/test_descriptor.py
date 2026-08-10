"""Local, content-free descriptor construction from a callable's signature."""

from __future__ import annotations

from typing import Any

from traigent.generation.coldstart._descriptor import build_descriptor
from traigent.generation.coldstart._plan import validate_descriptor_arity


def _build(
    func: Any, verifier_kinds: tuple[str, ...] = ("executable_property",)
) -> dict:
    return build_descriptor(
        func, verifier_kinds=verifier_kinds, generation_capabilities=("customer_llm",)
    )


def test_arity_and_coarse_kinds_from_a_fully_typed_signature() -> None:
    def f(a: str, b: int, c: float, d: bool, e: dict, g: list) -> bool:
        return True

    descriptor = _build(f)
    assert descriptor["input_arity"] == 6
    assert descriptor["input_kinds"] == [
        "string",
        "integer",
        "number",
        "boolean",
        "object",
        "array",
    ]
    assert descriptor["output_kind"] == "boolean"


def test_output_kind_is_unknown_when_return_annotation_is_missing() -> None:
    def f(a: str):
        return a

    descriptor = _build(f)
    assert descriptor["output_kind"] == "unknown"


def test_output_kind_is_unknown_even_though_the_body_always_returns_bool() -> None:
    """Requirement: never GUESS output_kind from runtime behavior."""

    def f(a: str):
        # Deliberately not reflected in output_kind -- there is no return
        # annotation, so this must classify as "unknown" regardless.
        return True

    descriptor = _build(f)
    assert descriptor["output_kind"] == "unknown"


def test_unmappable_annotation_becomes_unknown() -> None:
    class CustomThing:
        pass

    def f(a: CustomThing) -> CustomThing:
        return a

    descriptor = _build(f)
    assert descriptor["input_kinds"] == ["unknown"]
    assert descriptor["output_kind"] == "unknown"


def test_missing_parameter_annotation_becomes_unknown() -> None:
    def f(a, b: int) -> int:
        return b

    descriptor = _build(f)
    assert descriptor["input_kinds"] == ["unknown", "integer"]


def test_optional_wraps_to_the_inner_coarse_kind() -> None:
    def f(a: int | None) -> str | None:
        return None

    descriptor = _build(f)
    assert descriptor["input_kinds"] == ["integer"]
    assert descriptor["output_kind"] == "string"


def test_pep604_union_with_none_wraps_to_the_inner_coarse_kind() -> None:
    def f(a: int | None) -> str | None:
        return None

    descriptor = _build(f)
    assert descriptor["input_kinds"] == ["integer"]
    assert descriptor["output_kind"] == "string"


def test_multi_type_union_is_unknown() -> None:
    def f(a: int | str) -> None:
        return None

    descriptor = _build(f)
    assert descriptor["input_kinds"] == ["unknown"]


def test_bool_is_never_conflated_with_integer() -> None:
    """bool is an int subclass at runtime; the descriptor must not conflate them."""

    def f(flag: bool) -> None:
        return None

    descriptor = _build(f)
    assert descriptor["input_kinds"] == ["boolean"]


def test_var_positional_and_var_keyword_do_not_count_toward_arity() -> None:
    def f(a: str, *args: Any, **kwargs: Any) -> None:
        return None

    descriptor = _build(f)
    assert descriptor["input_arity"] == 1
    assert descriptor["input_kinds"] == ["string"]


def test_own_descriptor_builder_never_produces_an_arity_mismatch() -> None:
    def f(a: str, b: int, c: bool) -> None:
        return None

    descriptor = _build(f)
    assert validate_descriptor_arity(descriptor) is None


def test_arity_mismatch_is_caught_client_side_without_a_round_trip() -> None:
    """The JSON schema can't express len(input_kinds) == input_arity; this
    is the client-side guard that catches it before a request is sent. A
    hand-built mismatched descriptor stands in for a would-be bug in
    build_descriptor -- it must never reach the network."""

    bad_descriptor = {
        "input_arity": 2,
        "input_kinds": ["string"],  # only 1 entry for arity 2
        "output_kind": "unknown",
        "verifier_kinds": [],
        "generation_capabilities": [],
    }
    gap = validate_descriptor_arity(bad_descriptor)
    assert gap is not None
    assert gap.reason == "descriptor_arity_mismatch"


def test_descriptor_carries_no_extra_keys() -> None:
    def f(a: str) -> bool:
        return True

    descriptor = _build(f)
    assert set(descriptor) == {
        "input_arity",
        "input_kinds",
        "output_kind",
        "verifier_kinds",
        "generation_capabilities",
    }
