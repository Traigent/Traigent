"""Tests for the materialization guard used by the typed/interactive
session-create paths (privacy_operations.py, interactive_optimizer.py).

The fingerprint is content-derived PROVENANCE, never identity --
never the dataset's name/label (owner decision, see traigent/cloud/privacy_operations.py
and traigent/optimizers/interactive_optimizer.py). These tests pin that
semantic and the "never drain a single-use iterator" safety guard that
`is_dataset_materialized` exists to enforce.
"""

from __future__ import annotations

from collections.abc import Iterator

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.artifact_fingerprints import (
    build_dataset_only_fingerprint_payload,
    compute_dataset_fingerprint,
    is_dataset_materialized,
)


def _example(input_data, expected_output):
    return EvaluationExample(input_data=input_data, expected_output=expected_output)


def test_same_content_different_names_yield_the_same_fingerprint() -> None:
    """The fingerprint tracks CONTENT, never the name/label."""
    examples = [
        _example({"question": "a"}, "answer-a"),
        _example({"question": "b"}, "answer-b"),
    ]

    dataset_a = Dataset(examples=list(examples), name="qa-dataset-v1")
    dataset_b = Dataset(examples=list(examples), name="a-totally-different-name")

    fp_a = compute_dataset_fingerprint(dataset_a)
    fp_b = compute_dataset_fingerprint(dataset_b)

    assert fp_a is not None
    assert fp_a == fp_b


def test_different_content_yields_different_fingerprint() -> None:
    dataset_a = Dataset(
        examples=[_example({"question": "a"}, "answer-a")], name="same-name"
    )
    dataset_b = Dataset(
        examples=[_example({"question": "a"}, "DIFFERENT-answer")], name="same-name"
    )

    fp_a = compute_dataset_fingerprint(dataset_a)
    fp_b = compute_dataset_fingerprint(dataset_b)

    assert fp_a is not None
    assert fp_b is not None
    assert fp_a != fp_b


def test_dataset_instance_is_materialized() -> None:
    """Dataset.examples is enforced to be a list at construction time
    (traigent/evaluators/base.py Dataset.__post_init__), so a Dataset is
    always safe to fingerprint."""
    dataset = Dataset(examples=[_example({"q": 1}, "a")])
    assert is_dataset_materialized(dataset) is True


def test_list_of_examples_is_materialized() -> None:
    assert is_dataset_materialized([_example({"q": 1}, "a")]) is True


def test_mapping_with_list_examples_is_materialized() -> None:
    assert is_dataset_materialized({"examples": [{"input": 1, "expected": 2}]}) is True


def test_none_and_empty_are_not_materialized_but_are_not_errors() -> None:
    assert is_dataset_materialized(None) is False
    assert compute_dataset_fingerprint(None) is None


def test_bare_generator_is_not_materialized() -> None:
    """A generator has no realized `.examples` -- treating it as
    materialized would let a caller drain it via
    `_extract_examples`'s `list(...)` fallback."""

    def gen() -> Iterator[EvaluationExample]:
        yield _example({"q": 1}, "a")
        yield _example({"q": 2}, "b")

    generator = gen()
    assert is_dataset_materialized(generator) is False


def test_bare_generator_is_never_consumed_by_the_materialization_check() -> None:
    """The guard itself must be side-effect-free: checking materialization
    must not advance the iterator the caller still needs to read."""

    def gen() -> Iterator[int]:
        yield 1
        yield 2
        yield 3

    generator = gen()

    # Call the guard several times -- it must never touch the generator.
    assert is_dataset_materialized(generator) is False
    assert is_dataset_materialized(generator) is False

    # The caller can still read every item afterwards.
    assert list(generator) == [1, 2, 3]


def test_mapping_with_generator_examples_is_not_materialized() -> None:
    def gen() -> Iterator[dict]:
        yield {"input": 1, "expected": 2}

    generator = gen()
    mapping = {"examples": generator}

    assert is_dataset_materialized(mapping) is False
    # Guard must not have consumed the generator either.
    assert list(generator) == [{"input": 1, "expected": 2}]


def test_metadata_only_mapping_produces_no_fingerprint() -> None:
    """A metadata-only descriptor is not content and must not become an identity.

    ``{"name": ..., "size": ...}`` is exactly the shape the typed session already
    sends as ``dataset_metadata``. It has no ``examples`` key, so it falls through
    ``_extract_examples``'s "wrap self as one example" branch and canonicalizes to
    an example whose input and expected output are both ``None`` -- meaning EVERY
    such descriptor hashes identically. As provenance that is useless off this
    digest, so emitting one here would merge unrelated datasets into a single
    optimization history.
    """
    assert (
        build_dataset_only_fingerprint_payload(
            {"name": "customer-support-v3", "size": 18}
        )
        is None
    )


def test_unrelated_metadata_only_mappings_do_not_collide_onto_one_identity() -> None:
    """The collision itself, asserted directly: two unrelated descriptors, no identity."""
    first = build_dataset_only_fingerprint_payload(
        {"name": "customer-support-v3", "size": 18}
    )
    second = build_dataset_only_fingerprint_payload(
        {"name": "sql-spider-eval", "size": 400}
    )
    assert first is None and second is None


def test_a_single_real_example_mapping_still_fingerprints() -> None:
    """The content check must not reject a legitimate one-example dataset."""
    payload = build_dataset_only_fingerprint_payload({"input": "q", "expected": "a"})
    assert payload is not None
    assert payload["artifact_fingerprints"]["dataset"].startswith("fp1:")


def test_examples_list_of_contentless_dicts_produces_no_fingerprint() -> None:
    """A populated list whose examples carry no input/expected is still not content."""
    assert (
        build_dataset_only_fingerprint_payload(
            {"examples": [{"note": "x"}, {"note": "y"}]}
        )
        is None
    )
