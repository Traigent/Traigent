"""Resolve failing example ids back to the rows of their original dataset.

# Traceability: CONC-Layer-Data CONC-Quality-Observability FUNC-STORAGE FUNC-ANALYTICS

``OptimizationResult.failing_examples()`` and the outcome matrix carry only
example *ids*. To turn them into a regression dataset each id must be joined to
one row of the local dataset file, and the exported content must come from that
file, never from the per-example results.

**The join is by recorded content, not by id scheme.** An example id is
whatever the evaluator emitted: the SDK's positional ``example_<i>``, a
``metadata["example_id"]``, a keyed ``ex1:`` content identity, or any string a
custom evaluator chose. None of those can be mapped to a row without assuming
which scheme produced it, and a wrong assumption exports the wrong row without
any error. Instead, for every requested id the run's own per-example records
(``trial.metadata["example_results"]``) supply the ``input_data`` and
``expected_output`` the run evaluated under that id, and the id resolves to the
single dataset row with that content.

**Both sides are compared in the form the run stores.** The SDK redacts
per-example records before keeping them (``trial_result_factory`` runs
``redact_sensitive_data`` over each ``to_dict()``), so an email or a 9-digit
number in a row is recorded as a placeholder. Each dataset row is therefore put
through the same ``to_dict``-shaped JSON projection and the same redaction
before comparison, and the record side is projected the same way (redaction is
idempotent). Redaction can map different rows to the same form; that is
detected over the WHOLE dataset and reported as ambiguity, never resolved by
picking one. The remaining limit is stated rather than hidden: the join proves
which row of the dataset you pass matches the run's record, so the dataset
must be the one the run evaluated -- a row edited only inside a redacted span
would still match. Export with the SDK version that ran the optimization: the
redaction patterns are applied again at export time, and a changed pattern set
could make a record match a row that literally contains a placeholder.

Every requested id must resolve, or :class:`UnresolvedExampleIdError` is raised
before any output is produced:

* **unverifiable** -- the run kept no record for the id, its records hold no
  ``input_data`` (example content was not logged), or its records in different
  trials disagree about the content (the id named different rows);
* **unresolved** -- no dataset row has that content (the dataset is not the one
  the run evaluated, or it changed since);
* **ambiguous** -- more than one row has that content (including rows that
  differ only inside a redacted span), or two ids resolve to the same row.

The returned dataset keeps the source rows' order, content and metadata. A row
whose metadata has no ``example_id`` gains one holding its id in the SOURCE
dataset (``example_<source index>``), so the subset's ids stay unique, stay
traceable to the source file, and the subset reloads through
``Dataset.from_jsonl``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Sequence
from typing import Any

from traigent.evaluators.base import Dataset
from traigent.security.redaction import redact_sensitive_data
from traigent.utils.persistence import _safe_json_value


class UnresolvedExampleIdError(ValueError):
    """Some example ids did not resolve to exactly one dataset row.

    Attributes:
        unresolved: Ids whose recorded content matches no dataset row.
        ambiguous: Ids whose content matches more than one row, or that resolve
            to the same row as another id.
        unverifiable: Ids the run's records cannot tie to one content: no
            record, no logged ``input_data``, or conflicting records.
    """

    def __init__(
        self,
        unresolved: Sequence[str],
        ambiguous: Sequence[str],
        unverifiable: Sequence[str] = (),
    ) -> None:
        self.unresolved = list(unresolved)
        self.ambiguous = list(ambiguous)
        self.unverifiable = list(unverifiable)
        parts: list[str] = []
        if self.unverifiable:
            parts.append(
                f"{len(self.unverifiable)} example id(s) have no single recorded "
                "input in the run (no per-example record, example content not "
                "logged, or the id named different inputs in different trials): "
                f"{self.unverifiable}"
            )
        if self.unresolved:
            parts.append(
                f"{len(self.unresolved)} example id(s) match no dataset row with "
                f"the input and expected_output the run evaluated: {self.unresolved}"
            )
        if self.ambiguous:
            parts.append(
                f"{len(self.ambiguous)} example id(s) match more than one dataset "
                f"row, or share a row with another id: {self.ambiguous}"
            )
        super().__init__(
            "; ".join(parts)
            + ". Pass the original dataset the run evaluated, unmodified; "
            "nothing was selected."
        )


def _field(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def _canonical(input_data: Any, expected_output: Any) -> str:
    """One comparable string for an example's input and expected output.

    The JSON projection ``ExampleResult.to_dict`` applies, then the redaction
    the SDK applies before storing a record, then sorted-key JSON.

    Raises:
        TypeError: When the value has no canonical JSON form (e.g. a mapping
            mixing int and str keys).
    """
    projected = redact_sensitive_data(
        [_safe_json_value(input_data), _safe_json_value(expected_output)]
    )
    return json.dumps(projected, sort_keys=True, ensure_ascii=False)


def _recorded_content(trials: Iterable[Any]) -> dict[str, set[str] | None]:
    """Map each example id to the canonical contents the run recorded for it.

    ``None`` marks an id with at least one record that carries no
    ``input_data`` (content withheld), so it cannot be verified. Ids are
    stringified exactly as ``build_outcome_matrix`` does (``None`` skipped).
    """
    contents: dict[str, set[str] | None] = {}
    for trial in trials:
        metadata = getattr(trial, "metadata", None)
        records = (
            metadata.get("example_results") if isinstance(metadata, dict) else None
        )
        if not isinstance(records, list):
            continue
        for record in records:
            example_id = _field(record, "example_id")
            if example_id is None:
                continue
            key = str(example_id)
            input_data = _field(record, "input_data")
            if input_data is None:
                contents[key] = None
                continue
            seen = contents.setdefault(key, set())
            if seen is None:
                continue
            try:
                seen.add(_canonical(input_data, _field(record, "expected_output")))
            except TypeError:
                contents[key] = None
    return contents


def _validated_requested_ids(dataset: Any, example_ids: Sequence[str]) -> list[str]:
    if not isinstance(dataset, Dataset):
        raise TypeError(f"dataset must be a Dataset, got {type(dataset).__name__}")
    if isinstance(example_ids, (str, bytes)):
        raise TypeError("example_ids must be a sequence of ids, not a single string")
    requested = list(dict.fromkeys(example_ids))
    if not requested:
        raise ValueError("example_ids is empty; nothing to select")
    for eid in requested:
        if not isinstance(eid, str):
            raise TypeError(f"example ids must be strings, got {eid!r}")
    return requested


def _rows_by_content(dataset: Dataset) -> dict[str, list[int]]:
    rows_by_content: dict[str, list[int]] = {}
    for index, example in enumerate(dataset.examples):
        try:
            content = _canonical(example.input_data, example.expected_output)
        except TypeError as exc:
            raise ValueError(
                f"Dataset row {index} has no canonical JSON form ({exc}); it "
                "cannot be compared with the run's records"
            ) from exc
        rows_by_content.setdefault(content, []).append(index)
    return rows_by_content


def select_examples_by_id(
    dataset: Dataset, example_ids: Sequence[str], run: Any
) -> Dataset:
    """Return a new dataset holding the rows the run evaluated under ``example_ids``.

    Args:
        dataset: The complete original dataset the run evaluated.
        example_ids: Ids as emitted in the run's outcome matrix.
        run: The run (an ``OptimizationResult``, or any object exposing
            ``trials`` whose ``metadata["example_results"]`` hold the
            per-example records) that produced the ids.

    Returns:
        A new :class:`Dataset` of the matching rows, in source order.

    Raises:
        TypeError: When ``dataset`` is not a ``Dataset`` or an id is not a string.
        ValueError: When ``example_ids`` is empty.
        UnresolvedExampleIdError: When any id is unverifiable, unresolved or
            ambiguous.
    """
    requested = _validated_requested_ids(dataset, example_ids)

    recorded = _recorded_content(list(getattr(run, "trials", None) or []))
    rows_by_content = _rows_by_content(dataset)

    unverifiable: list[str] = []
    unresolved: list[str] = []
    ambiguous: list[str] = []
    ids_by_row: dict[int, list[str]] = {}
    for eid in requested:
        contents = recorded.get(eid)
        if not contents or len(contents) != 1:
            unverifiable.append(eid)
            continue
        rows = rows_by_content.get(next(iter(contents)), [])
        if not rows:
            unresolved.append(eid)
        elif len(rows) > 1:
            ambiguous.append(eid)
        else:
            ids_by_row.setdefault(rows[0], []).append(eid)
    for ids in ids_by_row.values():
        if len(ids) > 1:
            ambiguous.extend(ids)
    if unverifiable or unresolved or ambiguous:
        raise UnresolvedExampleIdError(unresolved, ambiguous, unverifiable)

    examples = []
    for index in sorted(ids_by_row):
        example = copy.deepcopy(dataset.examples[index])
        metadata = dict(example.metadata or {})
        metadata.setdefault("example_id", f"example_{index}")
        example.metadata = metadata
        examples.append(example)
    return Dataset(
        examples=examples,
        name=dataset.name,
        description=dataset.description,
        metadata=copy.deepcopy(dataset.metadata),
    )
