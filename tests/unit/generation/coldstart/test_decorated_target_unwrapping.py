"""Passing an ``@traigent.optimize``-decorated function must describe the REAL function.

This is the documented entry point for the whole feature: "point cold start at
your decorated function". Before ``_unwrap_target``, doing exactly that produced

    {"input_arity": 0, "input_kinds": [], "output_kind": "unknown"}

because ``OptimizedFunction.__call__`` is declared ``(*args, **kwargs)`` and the
class sets no ``__wrapped__``.

That is the worst shape a bug can take: the descriptor is WELL-FORMED. It
validates against the schema, the backend issues a plan for a zero-argument
target, and candidates are then generated against a signature the real function
does not have. Nothing raises. The eval set is simply wrong, and every
downstream number computed from it is wrong with it.

Found by running the documented example rather than reading it.
"""

from __future__ import annotations

import warnings

import pytest

import traigent
from traigent.generation.coldstart._descriptor import build_descriptor

_KINDS = {
    "verifier_kinds": ("executable_property",),
    "generation_capabilities": ("customer_llm",),
}


def plain(question: str, count: int) -> bool:
    return True


@pytest.fixture
def decorated():
    # The decorator warns about CONTEXT-mode injection for a body that never
    # reads get_config(); irrelevant here -- we only inspect the signature.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        @traigent.optimize(
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"m": ["a", "b"]},
            offline=True,
        )
        def answer(question: str, count: int) -> bool:
            return True

    return answer


def test_the_wrapper_really_does_hide_the_signature(decorated) -> None:
    """Guard the premise. If this ever stops holding, the fix below is dead weight."""
    import inspect

    assert not hasattr(decorated, "__wrapped__")
    assert str(inspect.signature(decorated)).startswith("(*args")


def test_decorated_function_describes_the_underlying_signature(decorated) -> None:
    descriptor = build_descriptor(decorated, **_KINDS)

    assert descriptor["input_arity"] == 2
    assert descriptor["input_kinds"] == ["string", "integer"]
    assert descriptor["output_kind"] == "boolean"


def test_decorated_and_undecorated_agree(decorated) -> None:
    """The decorator must be invisible to the descriptor."""
    assert build_descriptor(decorated, **_KINDS) == build_descriptor(plain, **_KINDS)


def test_a_plain_function_is_unaffected() -> None:
    descriptor = build_descriptor(plain, **_KINDS)

    assert descriptor["input_arity"] == 2
    assert descriptor["output_kind"] == "boolean"


def test_functools_wraps_chains_are_followed() -> None:
    """__wrapped__ is the stdlib convention; honour it, not just Traigent's .func."""
    import functools

    @functools.wraps(plain)
    def wrapper(*args, **kwargs):
        return plain(*args, **kwargs)

    assert build_descriptor(wrapper, **_KINDS) == build_descriptor(plain, **_KINDS)


def test_a_self_referential_wrapper_chain_terminates() -> None:
    """Unwrapping must not spin on a pathological chain."""

    class Loop:
        def __call__(self, a: str) -> bool:
            return True

    loop = Loop()
    loop.func = loop  # points at itself

    descriptor = build_descriptor(loop, **_KINDS)
    assert descriptor["output_kind"] in {"boolean", "unknown"}


def test_unwrapping_never_leaks_the_wrapped_functions_name() -> None:
    """Unwrapping gives us MORE of the function; it must not give us its name."""
    serialized = str(build_descriptor(plain, **_KINDS))

    assert "question" not in serialized
    assert "count" not in serialized
    assert "plain" not in serialized
