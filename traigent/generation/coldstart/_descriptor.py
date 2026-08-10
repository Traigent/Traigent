"""Build a CONTENT-FREE cold-start descriptor from a callable's signature.

Only coarse type shape ever leaves this module: parameter NAMES, annotation
TEXT, docstrings, module paths, and default VALUES must never appear in the
returned descriptor dict. Inspection happens locally; only
``input_arity`` / ``input_kinds`` / ``output_kind`` (plus the
caller-provided ``verifier_kinds`` / ``generation_capabilities``) cross the
network boundary.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from typing import Any

# Exact-identity lookup on the annotation's type object itself (not an
# isinstance chain over runtime VALUES) -- bool and int are different type
# objects here even though bool subclasses int at runtime, so there is no
# ordering hazard to worry about.
_ANNOTATION_KIND_BY_TYPE: dict[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    dict: "object",
    list: "array",
    tuple: "array",
    set: "array",
    frozenset: "array",
}

# Fallback for annotations that only exist as text -- e.g. `from __future__
# import annotations` turns every annotation into a string, and
# typing.get_type_hints() can fail to resolve a forward reference the
# caller's module can't see (a locally defined class, a lambda with no
# __globals__, etc). This is a best-effort local classification aid; the
# resulting KIND still never leaves this module as raw text.
_TEXTUAL_KIND_HINTS: tuple[tuple[str, str], ...] = (
    ("bool", "boolean"),
    ("int", "integer"),
    ("float", "number"),
    ("str", "string"),
    ("dict", "object"),
    ("mapping", "object"),
    ("list", "array"),
    ("tuple", "array"),
    ("set", "array"),
    ("frozenset", "array"),
    ("sequence", "array"),
)

_UNION_ORIGINS = (typing.Union, types.UnionType)


def build_descriptor(
    func: Callable[..., Any],
    *,
    verifier_kinds: tuple[str, ...],
    generation_capabilities: tuple[str, ...],
) -> dict[str, Any]:
    """Build the descriptor dict for one target callable.

    Never inspects/serializes parameter names, docstrings, module paths, or
    default values -- only the callable's coarse arity and per-parameter /
    return coarse type classes.
    """
    signature = inspect.signature(func)
    resolved = _resolve_annotations(func)

    input_kinds: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # *args / **kwargs carry no fixed arity of their own; the
            # descriptor's input_arity only covers named parameters.
            continue
        annotation = resolved.get(name, inspect.Signature.empty)
        input_kinds.append(_classify(annotation))

    output_annotation = resolved.get("return", inspect.Signature.empty)
    output_kind = _classify(output_annotation)

    return {
        "input_arity": len(input_kinds),
        "input_kinds": input_kinds,
        "output_kind": output_kind,
        "verifier_kinds": list(verifier_kinds),
        "generation_capabilities": list(generation_capabilities),
    }


def _resolve_annotations(func: Callable[..., Any]) -> dict[str, Any]:
    try:
        return typing.get_type_hints(func)
    except Exception:
        # Unresolvable forward reference, or a callable with no meaningful
        # __globals__ (e.g. a runtime-built lambda). Fall back to the raw
        # __annotations__; classification below degrades anything it can't
        # confidently map to "unknown" rather than raising.
        return dict(getattr(func, "__annotations__", {}))


def _classify(annotation: Any) -> str:
    if (
        annotation is inspect.Signature.empty
        or annotation is None
        or annotation is type(None)
    ):
        return "unknown"
    if isinstance(annotation, type) and annotation in _ANNOTATION_KIND_BY_TYPE:
        return _ANNOTATION_KIND_BY_TYPE[annotation]

    origin = typing.get_origin(annotation)
    if origin is not None:
        if origin in _UNION_ORIGINS:
            return _classify_union(typing.get_args(annotation))
        if isinstance(origin, type) and origin in _ANNOTATION_KIND_BY_TYPE:
            return _ANNOTATION_KIND_BY_TYPE[origin]
        origin_name = getattr(origin, "__name__", "").lower()
        for needle, kind in _TEXTUAL_KIND_HINTS:
            if origin_name == needle:
                return kind
        return "unknown"

    if isinstance(annotation, str):
        return _classify_textual(annotation)

    return "unknown"


def _classify_union(args: tuple[Any, ...]) -> str:
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) != 1:
        return "unknown"
    return _classify(non_none[0])


def _classify_textual(annotation: str) -> str:
    text = annotation.strip().lower()
    for needle, kind in _TEXTUAL_KIND_HINTS:
        if (
            text == needle
            or text.startswith(needle + "[")
            or text.startswith(needle + "|")
            or text.startswith("optional[" + needle + "]")
        ):
            return kind
    return "unknown"
