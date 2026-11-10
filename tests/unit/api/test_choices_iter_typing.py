"""Choices is iterable with a typed element (Iterator[T]) and typed signatures."""

import typing

from traigent.api.parameter_ranges import Choices


def test_choices_iter_has_iterator_return_annotation() -> None:
    hints = typing.get_type_hints(Choices.__iter__)
    assert "return" in hints
    assert typing.get_origin(hints["return"]).__name__ == "Iterator"


def test_choices_contains_item_annotated_as_object() -> None:
    # __contains__ accepts any object (#3191); the old element-typed annotation
    # would reject a membership test against an unrelated type.
    assert typing.get_type_hints(Choices.__contains__)["item"] is object


def test_choices_iteration_behaviour_unchanged() -> None:
    assert list(Choices(["a", "b"])) == ["a", "b"]
    assert "a" in Choices(["a", "b"])
    assert "z" not in Choices(["a", "b"])
