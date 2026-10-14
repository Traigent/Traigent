"""Resolve failing example ids back to the rows of their original dataset.

# Traceability: CONC-Layer-Data CONC-Quality-Observability FUNC-STORAGE FUNC-ANALYTICS

``OptimizationResult.failing_examples()`` and the outcome matrix carry only
example *ids*. To turn them into a regression dataset the ids must be joined to
the original rows — taking content from the dataset itself, never from the
(possibly redacted) per-example results.

The join uses the same handles the evaluators emit:

* the default local evaluator: ``metadata["example_id"]`` when present, else
  the positional ``example_<index>`` (``_example_correlation_key``);
* ``SimpleScoringEvaluator``: always the positional ``example_<index>``.

Candidates are stringified exactly as ``build_outcome_matrix`` does (``None``
skipped). Every requested id must resolve to exactly one row; anything
unresolved or ambiguous raises :class:`UnresolvedExampleIdError` before any
output is produced. Keyed content identities (``ex1:`` ids, minted under a
purpose-key grant) cannot be recomputed locally and are reported as unresolved.

Positional handles are only meaningful against the very file the run used:
pass the original dataset, unmodified and in its original order.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

from traigent.evaluators.base import Dataset, _example_correlation_key

_KEYED_ID_PREFIX = "ex1:"


class UnresolvedExampleIdError(ValueError):
    """Some example ids did not resolve to exactly one dataset row.

    Attributes:
        unresolved: Ids that matched no row.
        ambiguous: Ids that matched more than one row.
    """

    def __init__(self, unresolved: Sequence[str], ambiguous: Sequence[str]) -> None:
        self.unresolved = list(unresolved)
        self.ambiguous = list(ambiguous)
        parts: list[str] = []
        if self.unresolved:
            parts.append(
                f"{len(self.unresolved)} example id(s) match no dataset row: "
                f"{self.unresolved}"
            )
            if any(eid.startswith(_KEYED_ID_PREFIX) for eid in self.unresolved):
                parts.append(
                    "keyed content identities ('ex1:' ids) are not supported by "
                    "the local exporter"
                )
        if self.ambiguous:
            parts.append(
                f"{len(self.ambiguous)} example id(s) match more than one dataset "
                f"row: {self.ambiguous}"
            )
        super().__init__(
            "; ".join(parts)
            + ". Pass the original dataset the run evaluated, unmodified."
        )


def select_examples_by_id(dataset: Dataset, example_ids: Sequence[str]) -> Dataset:
    """Return a new dataset holding the rows whose ids are in ``example_ids``.

    Rows are emitted once each, in the original dataset order, with their
    parsed ``input_data``, ``expected_output`` and ``metadata`` copied
    unchanged.

    Args:
        dataset: The complete original dataset the run evaluated.
        example_ids: Ids as emitted in the outcome matrix.

    Returns:
        A new :class:`Dataset` of the matching rows.

    Raises:
        TypeError: When ``dataset`` is not a ``Dataset`` or an id is not a string.
        ValueError: When ``example_ids`` is empty.
        UnresolvedExampleIdError: When any id is unresolved or ambiguous.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(f"dataset must be a Dataset, got {type(dataset).__name__}")
    if isinstance(example_ids, (str, bytes)):
        raise TypeError("example_ids must be a sequence of ids, not a single string")
    requested = list(example_ids)
    if not requested:
        raise ValueError("example_ids is empty; nothing to select")
    for eid in requested:
        if not isinstance(eid, str):
            raise TypeError(f"example ids must be strings, got {eid!r}")

    # Enumerate the complete dataset before filtering: positional handles are
    # indices into the original row order.
    rows_by_id: dict[str, set[int]] = {}
    for index, example in enumerate(dataset.examples):
        candidates = {f"example_{index}"}
        key = _example_correlation_key(example, index)
        if key is not None:
            candidates.add(str(key))
        for candidate in candidates:
            rows_by_id.setdefault(candidate, set()).add(index)

    unresolved: list[str] = []
    ambiguous: list[str] = []
    chosen: set[int] = set()
    for eid in dict.fromkeys(requested):
        rows = rows_by_id.get(eid)
        if not rows:
            unresolved.append(eid)
        elif len(rows) > 1:
            ambiguous.append(eid)
        else:
            chosen.update(rows)
    if unresolved or ambiguous:
        raise UnresolvedExampleIdError(unresolved, ambiguous)

    return Dataset(
        examples=[copy.deepcopy(dataset.examples[i]) for i in sorted(chosen)],
        name=dataset.name,
        description=dataset.description,
        metadata=copy.deepcopy(dataset.metadata),
    )
