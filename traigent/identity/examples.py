"""Example, dataset and evaluated-set identity for SDK evaluation datasets.

Spec: TraigentSchema ``docs/identity/content-identity-v1.md`` sections 4.4
(projection), 5 (multiset root) and 10 (trial ``EvaluatedSetV1``).

What this module does for one optimization run:

* :func:`identify_dataset` projects every :class:`EvaluationExample` with the
  spec's SDK projection (``project_sdk_example``), mints its keyed
  ``(example_id, example_version)``, and computes the order-free multiset
  ``dataset_root``. Each example is stamped with its identity so every
  ``ExampleResult`` built from it can carry the content id instead of the old
  positional ``example_<i>`` fallback (:func:`result_identity_fields`).
* :mod:`traigent.identity.run` turns one trial's example results into the
  trial's ``EvaluatedSetV1`` (``evaluated_root`` + members), keyed to the run's
  ``dataset_root``.

**Fail closed.** Without a Backend purpose-key grant
(:func:`traigent.identity.keys.get_content_identity_keys` is ``None``) nothing
here mints an id: :func:`identify_dataset` returns ``None``, no example is
stamped, results keep their legacy row handle, and no roots are emitted. The
same happens when ANY example cannot be canonicalized (a non-JSON input, a
number outside +/-(2**53-1)): a dataset root over the examples that happened to
be hashable would claim a dataset that does not exist, so the run gets no root
at all rather than a wrong one.

**Row handle versus identity.** Two rows with identical content share one
``example_id`` (that is what makes the multiset count them). The SDK's internal
per-row correlation handle (``_example_correlation_key`` in
``traigent/evaluators/base.py``) therefore stays index-based: it is a
within-trial pointer, never an identity, and it is never sent as one.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from traigent.identity.content_identity import (
    SCHEME,
    ConflictingExampleVersionsWarning,
    ContentIdentityError,
    MultisetRoot,
    compute_multiset_root,
    identify_example,
    project_sdk_example,
)
from traigent.identity.keys import ContentIdentityKeys, get_content_identity_keys
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DatasetContentIdentity",
    "ExampleIdentity",
    "external_id_of",
    "identify_dataset",
    "identify_evaluation_example",
    "result_identity_fields",
    "stamped_identity",
]

#: Attribute stamped on an ``EvaluationExample`` holding ``(kid, ExampleIdentity)``.
_STAMP_ATTR = "_traigent_content_identity"
#: Attribute cached on a ``Dataset`` holding ``(kid, example object ids, identity)``.
_DATASET_CACHE_ATTR = "_traigent_content_identity_cache"
#: User-supplied id keys in example metadata, in precedence order. Both are
#: reserved annotation keys, so neither ever participates in example_version.
_EXTERNAL_ID_KEYS = ("external_id", "example_id")
_MAX_EXTERNAL_ID_LENGTH = 255


@dataclass(frozen=True)
class ExampleIdentity:
    """One example's content identity plus the user's own id (annotation only)."""

    example_id: str
    example_version: str
    external_id: str | None = None


@dataclass(frozen=True)
class DatasetContentIdentity:
    """A dataset's identity under one key version.

    ``examples`` is aligned with the dataset's example order; ``multiset`` is
    order-free.
    """

    key_id: str
    examples: tuple[ExampleIdentity, ...]
    multiset: MultisetRoot

    @property
    def dataset_root(self) -> str:
        return self.multiset.root

    def summary(self) -> dict[str, Any]:
        """Root and counts, without the member list (small enough for every trial)."""
        return {
            "scheme": SCHEME,
            "key_id": self.key_id,
            "dataset_root": self.multiset.root,
            "distinct_count": self.multiset.distinct_count,
            "total_count": self.multiset.total_count,
            "conflicting_example_ids": list(self.multiset.conflicting_example_ids),
        }


def external_id_of(metadata: Mapping[str, Any] | None) -> str | None:
    """The user's own example id from metadata (``external_id``, else ``example_id``).

    Kept as an annotation only: it never participates in ``example_id`` or
    ``example_version`` and is never sent in an id field.
    """
    if not isinstance(metadata, Mapping):
        return None
    for key in _EXTERNAL_ID_KEYS:
        value = metadata.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            value = str(value)
        if isinstance(value, str):
            value = value.strip()
            if value and len(value) <= _MAX_EXTERNAL_ID_LENGTH:
                return value
    return None


def identify_evaluation_example(
    example: Any, keys: ContentIdentityKeys
) -> ExampleIdentity:
    """Mint ``(example_id, example_version)`` for one ``EvaluationExample``.

    Applies the spec's SDK projection exactly (section 4.4): input :=
    ``input_data``; context := ``metadata["context"]`` when present and not
    null; expected := ``expected_output``; metadata := metadata minus the
    reserved annotation keys (empty means absent).

    Raises:
        ContentIdentityError: the example cannot be canonicalized.
    """
    metadata = getattr(example, "metadata", None)
    projection = project_sdk_example(
        getattr(example, "input_data", None),
        getattr(example, "expected_output", None),
        metadata if isinstance(metadata, Mapping) else None,
    )
    example_id, example_version = identify_example(keys.as_tenant_keys(), projection)
    return ExampleIdentity(
        example_id=example_id,
        example_version=example_version,
        external_id=external_id_of(metadata),
    )


def _set_stamp(example: Any, key_id: str, identity: ExampleIdentity | None) -> None:
    try:
        if identity is None:
            example.__dict__.pop(_STAMP_ATTR, None)
        else:
            example.__dict__[_STAMP_ATTR] = (key_id, identity)
    except AttributeError:  # __slots__ objects cannot be stamped
        pass


def stamped_identity(example: Any) -> ExampleIdentity | None:
    """The identity stamped on ``example`` under the CURRENTLY installed key.

    A stamp minted under a key that has since been removed or replaced is
    ignored (fail closed), so a rotated or cleared grant can never leave stale
    ids on new results.
    """
    keys = get_content_identity_keys()
    if keys is None:
        return None
    stamp = getattr(example, "__dict__", {}).get(_STAMP_ATTR)
    if not isinstance(stamp, tuple) or len(stamp) != 2:
        return None
    key_id, identity = stamp
    if key_id != keys.kid or not isinstance(identity, ExampleIdentity):
        return None
    return identity


def identify_dataset(
    dataset: Any, keys: ContentIdentityKeys | None = None
) -> DatasetContentIdentity | None:
    """Identify every example of ``dataset`` and compute its ``dataset_root``.

    ``keys`` defaults to the installed grant. Returns ``None`` -- and stamps
    nothing -- when there is no grant or any example cannot be identified.
    The result is cached on the dataset object for the same key and the same
    example objects, so repeated trials over one dataset hash it once.
    """
    if keys is None:
        keys = get_content_identity_keys()
    if keys is None:
        return None
    examples: Sequence[Any] = list(getattr(dataset, "examples", None) or [])
    fingerprint = (keys.kid, tuple(id(example) for example in examples))
    cached = getattr(dataset, "__dict__", {}).get(_DATASET_CACHE_ATTR)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == fingerprint:
        cached_identity = cached[1]
        if isinstance(cached_identity, DatasetContentIdentity):
            return cached_identity

    identities: list[ExampleIdentity] = []
    try:
        for example in examples:
            identities.append(identify_evaluation_example(example, keys))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConflictingExampleVersionsWarning)
            multiset = compute_multiset_root(
                [(ident.example_id, ident.example_version) for ident in identities],
                key_id=keys.kid,
            )
    except ContentIdentityError as error:
        # Content-free by construction: ContentIdentityError never echoes input.
        logger.warning(
            "Content identity disabled for this dataset: an example could not be "
            "identified (%s). No example ids or dataset root will be emitted.",
            error,
        )
        for example in examples:
            _set_stamp(example, keys.kid, None)
        return None

    if caught:
        logger.warning(
            "%d example id(s) in this dataset carry more than one expected "
            "output/metadata version (conflicting labels); they are listed in "
            "conflicting_example_ids and excluded from paired comparisons.",
            len(multiset.conflicting_example_ids),
        )

    for example, identity in zip(examples, identities, strict=True):
        _set_stamp(example, keys.kid, identity)
    result = DatasetContentIdentity(
        key_id=keys.kid, examples=tuple(identities), multiset=multiset
    )
    try:
        dataset.__dict__[_DATASET_CACHE_ATTR] = (fingerprint, result)
    except AttributeError:
        pass
    return result


def result_identity_fields(example: Any, fallback_example_id: str) -> dict[str, Any]:
    """Identity keyword arguments for an ``ExampleResult`` built from ``example``.

    With a stamped content identity: ``example_id`` is the keyed ``ex1`` id,
    ``example_version`` its ``exv1`` version, and a user-supplied id moves to
    ``external_id``. Without one (no key grant, or the dataset could not be
    identified): ``example_id`` is ``fallback_example_id`` -- the SDK's legacy
    per-row handle, which is NOT a content identity -- and ``example_version``
    stays ``None``.
    """
    metadata = getattr(example, "metadata", None)
    identity = stamped_identity(example)
    if identity is None:
        return {
            "example_id": fallback_example_id,
            "external_id": external_id_of(metadata),
        }
    return {
        "example_id": identity.example_id,
        "example_version": identity.example_version,
        "external_id": identity.external_id,
    }
