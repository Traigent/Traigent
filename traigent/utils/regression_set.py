"""A committed, growing regression set built from the examples a run got wrong.

# Traceability: CONC-Layer-Data FUNC-STORAGE FUNC-ANALYTICS

The set lives in ``<dir>/regression.jsonl`` (default dir ``tests/regression``),
one JSON object per line::

    {"input": {...}, "output": "...",
     "metadata": {"example_id": "ex12", "content_digest": "sha256:...",
                  "regression": {"label": "seen_once", "fails": null,
                                 "reps": null,
                                 "provenance": [{"run": "...", "date": "...",
                                                 "config_digest": "...",
                                                 "metric": "..."}]}}}

Rows reload through ``Dataset.from_jsonl`` (``output`` is one of its
expected-output aliases) and through :func:`load_regression_dataset`.

The file holds the example content exactly as it is in the dataset: nothing in
it is redacted, so a regression test can rerun the exact example. Review it
before committing it to your own repository. Nothing is uploaded and nothing
here touches the network.

**Dedupe is by example id, guarded by a content digest.** SDK example ids can
be positional (``example_<i>``), so the same id may name different content in
different runs. A merge whose id matches an existing row but whose content
digest differs is refused and writes nothing, unless the caller names a policy:
``replace`` swaps the stored row for the new content, ``keep-both`` stores the
new content under a distinct id.

Error messages name ids and fields, never example content.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from traigent.evaluators.base import (
    Dataset,
    EvaluationExample,
    _example_correlation_key,
)
from traigent.utils.file_lock import (
    FileLockTimeoutError,
    FileLockUnsupportedError,
    exclusive_lock,
)

REGRESSION_FILENAME = "regression.jsonl"
DEFAULT_REGRESSION_DIR = "tests/regression"
LABEL_SEEN_ONCE = "seen_once"
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0

OnIdConflict = Literal["refuse", "replace", "keep-both"]
ON_ID_CONFLICT_CHOICES: tuple[str, ...] = ("refuse", "replace", "keep-both")


class RegressionSetError(ValueError):
    """The regression file is missing, unreadable or malformed."""


class IdConflictError(RegressionSetError):
    """Ids already stored name different content than the incoming rows.

    Attributes:
        example_ids: The conflicting ids (never their content).
    """

    def __init__(self, example_ids: Sequence[str]) -> None:
        self.example_ids = list(example_ids)
        super().__init__(
            f"{len(self.example_ids)} example id(s) already in the regression set "
            f"name different content (content_digest differs): {self.example_ids}. "
            "Example ids can be positional, so the dataset may have changed. "
            "Nothing was written; pass --on-id-conflict replace or keep-both to "
            "choose."
        )


class StaleDigestError(RegressionSetError):
    """A stored row's content no longer matches its stored ``content_digest``.

    The row was edited by hand without updating the digest. Names the row id
    and line number, never the content.

    Attributes:
        stale: ``(example_id, line_number)`` of every mismatching row.
    """

    def __init__(self, path: Path, stale: Sequence[tuple[str, int]]) -> None:
        self.stale = list(stale)
        named = ", ".join(f"{eid!r} (line {line})" for eid, line in self.stale)
        super().__init__(
            f"Regression file {path}: the content of {len(self.stale)} row(s) "
            f"does not match the stored content_digest: {named}. The row was "
            "edited by hand, so its provenance would describe different "
            "content. Nothing was written. To accept the edited content for a "
            "row whose example is failing now, re-run with --on-id-conflict "
            "replace; otherwise undo the edit, or set the row's content_digest "
            "to traigent.utils.regression_set.content_digest(row['input'], "
            "row['output'])."
        )


@dataclass
class MergeReport:
    """What a merge did."""

    path: Path
    rows_added: int = 0
    provenance_added: int = 0
    rows_replaced: int = 0
    added_ids: list[str] = field(default_factory=list)


def regression_file(directory: str | os.PathLike[str]) -> Path:
    """The regression file inside ``directory``."""
    return Path(directory) / REGRESSION_FILENAME


def content_digest(input_data: Any, expected_output: Any) -> str:
    """``sha256:<hex>`` over the canonical JSON of an example's content."""
    text = json.dumps(
        [input_data, expected_output],
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_digest(config: Any) -> str | None:
    """Short digest naming a configuration, or ``None`` when there is none."""
    if config is None:
        return None
    text = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------- reading


def _bad(path: Path, line_no: int, problem: str) -> RegressionSetError:
    return RegressionSetError(
        f"Regression file {path} is not valid (line {line_no}): {problem}. "
        "Fix or remove the file; it was not modified."
    )


def _validate_row(path: Path, line_no: int, row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise _bad(path, line_no, "row is not a JSON object")
    if "input" not in row or "output" not in row:
        raise _bad(path, line_no, "row lacks the 'input' or 'output' field")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise _bad(path, line_no, "'metadata' is missing or not an object")
    if not isinstance(metadata.get("example_id"), str):
        raise _bad(path, line_no, "'metadata.example_id' is missing or not a string")
    if not isinstance(metadata.get("content_digest"), str):
        raise _bad(
            path, line_no, "'metadata.content_digest' is missing or not a string"
        )
    regression = metadata.get("regression")
    if not isinstance(regression, dict):
        raise _bad(path, line_no, "'metadata.regression' is missing or not an object")
    if not isinstance(regression.get("label"), str):
        raise _bad(path, line_no, "'metadata.regression.label' is not a string")
    if not isinstance(regression.get("provenance"), list):
        raise _bad(path, line_no, "'metadata.regression.provenance' is not a list")
    return row


def _read_numbered_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Parse and structurally validate the file; rows come with line numbers."""
    try:
        # Bytes, not read_text(): text mode would translate a bare "\r" into a
        # line break, and the file's records are separated by "\n" only.
        text = path.read_bytes().decode("utf-8-sig")
    except FileNotFoundError as exc:
        raise RegressionSetError(
            f"Regression file {path} does not exist; create it with "
            "'traigent regression add'"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise RegressionSetError(
            f"Cannot read regression file {path}: {type(exc).__name__}"
        ) from exc
    rows: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    # JSONL semantics: records end at "\n" only. str.splitlines() would also
    # break at U+2028, U+2029, U+0085, "\x0b", "\x0c" and others, which may sit
    # inside a JSON string.
    for line_no, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _bad(path, line_no, "line is not valid JSON") from exc
        row = _validate_row(path, line_no, parsed)
        example_id = row["metadata"]["example_id"]
        if example_id in seen:
            raise _bad(path, line_no, "example_id repeats an earlier row")
        seen.add(example_id)
        rows.append((line_no, row))
    if not rows:
        raise RegressionSetError(
            f"Regression file {path} holds no rows; it was not modified"
        )
    return rows


def _stale_rows(rows: Sequence[tuple[int, dict[str, Any]]]) -> list[tuple[str, int]]:
    """``(example_id, line)`` of rows whose content no longer matches the digest."""
    stale: list[tuple[str, int]] = []
    for line_no, row in rows:
        try:
            current = content_digest(row["input"], row["output"])
        except (TypeError, ValueError):
            current = None
        if current != row["metadata"]["content_digest"]:
            stale.append((row["metadata"]["example_id"], line_no))
    return stale


def read_regression_rows(path: Path) -> list[dict[str, Any]]:
    """Parse and validate an existing regression file.

    Every row's ``content_digest`` is recomputed from its ``input`` and
    ``output``; a mismatch means the row was edited by hand.

    Raises:
        RegressionSetError: When the file is missing, empty, or any line is not
            a well-formed regression row (or ids repeat).
        StaleDigestError: A row's content does not match its stored digest.
    """
    numbered = _read_numbered_rows(path)
    stale = _stale_rows(numbered)
    if stale:
        raise StaleDigestError(path, stale)
    return [row for _, row in numbered]


# --------------------------------------------------------------------- writing


#: Characters that other tools treat as line breaks although JSON allows them
#: raw inside a string (``json.dumps(ensure_ascii=False)`` leaves them as is).
#: The writer escapes them, so a record is always exactly one "\n"-terminated
#: line for every reader, including ones that split on Unicode line breaks.
_LINE_BREAK_ESCAPES = {
    "\u0085": "\\u0085",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
}


def _dump_line(row: dict[str, Any]) -> str:
    """One JSONL record: readable UTF-8, with Unicode line breaks escaped."""
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    for raw, escaped in _LINE_BREAK_ESCAPES.items():
        text = text.replace(raw, escaped)
    return text + "\n"


def _atomic_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write ``rows`` as JSONL via a temp sibling and an atomic rename."""
    lines = [_dump_line(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        existing_mode: int | None = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        existing_mode = None
    fd = os.open(
        tmp_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o666 if existing_mode is None else 0o600,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if existing_mode is not None and hasattr(os, "fchmod"):
                os.fchmod(fd, existing_mode)
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------- merge


def _new_row(
    example: EvaluationExample,
    example_id: str,
    digest: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(example.metadata or {})
    metadata.update(
        {
            "example_id": example_id,
            "content_digest": digest,
            "regression": {
                "label": LABEL_SEEN_ONCE,
                "fails": None,
                "reps": None,
                "provenance": [provenance],
            },
        }
    )
    return {
        "input": example.input_data,
        "output": example.expected_output,
        "metadata": metadata,
    }


def _incoming_id(example: EvaluationExample, position: int) -> str:
    """The row's id: ``metadata.example_id``, else ``example_<position>``.

    ``None``, ``""`` and booleans count as absent, the same rule the dataset
    loader applies (``_example_correlation_key``). ``position`` is the index in
    ``examples``; callers that selected a subset of a larger dataset must stamp
    each row's run-recorded id first (``select_examples_with_run_ids``), which
    ``traigent regression add`` does.
    """
    return str(_example_correlation_key(example, position))


def _plan_row(
    example: EvaluationExample,
    example_id: str,
    digest: str,
    provenance: dict[str, Any],
    by_id: dict[str, int],
    rows: list[dict[str, Any]],
    on_conflict: str,
    report: MergeReport,
    conflicts: list[str],
) -> None:
    index = by_id.get(example_id)
    if index is None:
        rows.append(_new_row(example, example_id, digest, provenance))
        by_id[example_id] = len(rows) - 1
        report.rows_added += 1
        report.added_ids.append(example_id)
        return
    stored = rows[index]["metadata"]
    if stored["content_digest"] == digest:
        entries = stored["regression"]["provenance"]
        if provenance not in entries:
            entries.append(provenance)
            report.provenance_added += 1
        return
    if on_conflict == "replace":
        rows[index] = _new_row(example, example_id, digest, provenance)
        report.rows_replaced += 1
        return
    if on_conflict == "keep-both":
        alias = f"{example_id}~{digest.split(':', 1)[1][:8]}"
        _plan_row(
            example, alias, digest, provenance, by_id, rows, "refuse", report, conflicts
        )
        return
    conflicts.append(example_id)


def merge_failures(
    directory: str | os.PathLike[str],
    examples: Dataset,
    *,
    run: str,
    date: str,
    config: Any,
    metric: str,
    on_id_conflict: OnIdConflict = "refuse",
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> MergeReport:
    """Merge failing examples into ``<directory>/regression.jsonl``.

    A new id becomes a row labelled ``seen_once``; a repeat of a stored id with
    the same content digest gains a provenance entry (an identical entry is not
    added twice); an id whose content digest differs is handled per
    ``on_id_conflict``. Nothing is written unless the whole merge succeeds.

    The whole read-merge-replace runs under a cross-process lock on the hidden
    sibling ``.regression.jsonl.lock``, waiting at most ``lock_timeout`` seconds.

    Raises:
        RegressionSetError: The existing file is unparseable, the policy is
            unknown, or the lock could not be taken in time (nothing written).
        IdConflictError: Ids clash and ``on_id_conflict`` is ``refuse``.
    """
    if on_id_conflict not in ON_ID_CONFLICT_CHOICES:
        raise RegressionSetError(
            f"on_id_conflict must be one of {list(ON_ID_CONFLICT_CHOICES)}"
        )
    path = regression_file(directory)
    # The lock covers the whole read-merge-replace, and the file is read only
    # after the lock is held, so concurrent adds cannot lose each other's rows.
    try:
        with exclusive_lock(path, timeout=lock_timeout):
            return _merge_locked(
                path, examples, run, date, config, metric, on_id_conflict
            )
    except (FileLockTimeoutError, FileLockUnsupportedError) as exc:
        raise RegressionSetError(str(exc)) from exc


def _admit_stale_rows(
    path: Path,
    numbered: Sequence[tuple[int, dict[str, Any]]],
    rows: list[dict[str, Any]],
    examples: Dataset,
    on_id_conflict: str,
) -> None:
    """Refuse hand-edited rows, except those ``replace`` is about to overwrite."""
    stale = _stale_rows(numbered)
    if not stale:
        return
    if on_id_conflict == "replace":
        incoming = {_incoming_id(ex, pos) for pos, ex in enumerate(examples.examples)}
        if all(eid in incoming for eid, _ in stale):
            stale_ids = {eid for eid, _ in stale}
            for row in rows:
                if row["metadata"]["example_id"] in stale_ids:
                    # Forces the replace branch; the new row carries a fresh digest.
                    row["metadata"]["content_digest"] = ""
            return
    raise StaleDigestError(path, stale)


def _merge_locked(
    path: Path,
    examples: Dataset,
    run: str,
    date: str,
    config: Any,
    metric: str,
    on_id_conflict: str,
) -> MergeReport:
    numbered = _read_numbered_rows(path) if path.exists() else []
    rows = [row for _, row in numbered]
    _admit_stale_rows(path, numbered, rows, examples, on_id_conflict)
    by_id = {row["metadata"]["example_id"]: i for i, row in enumerate(rows)}
    report = MergeReport(path=path)
    conflicts: list[str] = []
    provenance = {
        "run": run,
        "date": date,
        "config_digest": config_digest(config),
        "metric": metric,
    }
    for position, example in enumerate(examples.examples):
        digest = content_digest(example.input_data, example.expected_output)
        _plan_row(
            example,
            _incoming_id(example, position),
            digest,
            provenance,
            by_id,
            rows,
            on_id_conflict,
            report,
            conflicts,
        )
    if conflicts:
        raise IdConflictError(conflicts)
    if report.rows_added or report.provenance_added or report.rows_replaced:
        _atomic_write(path, rows)
    return report


# --------------------------------------------------------------------- loading


def load_regression_dataset(
    directory: str | os.PathLike[str] = DEFAULT_REGRESSION_DIR,
    labels: Sequence[str] = (LABEL_SEEN_ONCE, "reliable", "intermittent"),
) -> Dataset:
    """Load the regression set as a :class:`Dataset`, keeping only ``labels``.

    Raises:
        RegressionSetError: The file is missing or unparseable, or no row has
            one of the requested labels (an empty dataset is never returned).
    """
    if isinstance(labels, str):
        raise TypeError("labels must be a sequence of label strings, not one string")
    path = regression_file(directory)
    rows = read_regression_rows(path)
    wanted = set(labels)
    kept = [r for r in rows if r["metadata"]["regression"]["label"] in wanted]
    if not kept:
        raise RegressionSetError(
            f"Regression file {path} has no rows with label in {sorted(wanted)}"
        )
    try:
        full = Dataset.from_jsonl(path)
    except Exception as exc:
        raise RegressionSetError(
            f"Regression file {path} does not load as a dataset: {type(exc).__name__}"
        ) from exc
    keep_ids = {r["metadata"]["example_id"] for r in kept}
    examples = [
        ex for ex in full.examples if (ex.metadata or {}).get("example_id") in keep_ids
    ]
    return Dataset(
        examples=examples,
        name="regression_set",
        description=f"Regression set from {path}",
        metadata=dict(full.metadata or {}),
    )
