"""A committed, growing regression set built from the examples a run got wrong.

# Traceability: CONC-Layer-Data FUNC-STORAGE FUNC-ANALYTICS

The set lives in ``<dir>/regression.jsonl`` (default dir ``tests/regression``),
one JSON object per line::

    {"input": {...}, "output": "...",
     "metadata": {"example_id": "ex12", "content_digest": "sha256:...",
                  "regression": {"schema_version": 1, "label": "seen_once",
                                 "fails": null, "reps": null,
                                 "provenance": [{
                                     "run_id": "<saved run's optimization id>",
                                     "run": "<saved result name>",
                                     "imported": "2026-11-20",
                                     "mode": "best_config",
                                     "threshold": 0.5,
                                     "config_digest": "<generate_config_hash>",
                                     "failed_by": "score",
                                     "metric_field": "accuracy"}]}}}

``schema_version`` is per row (a JSONL file has no header, and a row stays
self-describing when rows are copied or merged). A reader refuses a row whose
major version is newer than it understands, and refuses a row without the field
(a preview format whose provenance has a different shape), each with a message
naming the line. ``run_id`` and ``mode`` and ``threshold`` identify an
evidence entry: adding the same saved run again, on any date, under any name,
adds nothing. ``run`` and ``imported`` are informational only. ``config_digest``
is the SDK's ``generate_config_hash`` of the winning configuration, or ``null``
with ``mode: "all_configs"`` (no single configuration is meant).

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
new content under a distinct id. ``replace`` keeps the row's earlier
provenance entries, each marked with the digest of the content it described.

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
from traigent.utils.hashing import generate_config_hash

REGRESSION_FILENAME = "regression.jsonl"
DEFAULT_REGRESSION_DIR = "tests/regression"
LABEL_SEEN_ONCE = "seen_once"
#: Major version of the row format this module reads and writes.
SCHEMA_VERSION = 1
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
            "Nothing was written. Choose one: --on-id-conflict replace "
            "overwrites each stored row with the incoming content (its earlier "
            "provenance entries are kept, marked with the digest of the content "
            "they described); --on-id-conflict keep-both leaves the stored rows "
            "as they are and stores the incoming content under a new id "
            "'<id>~<8 hex of the content digest>'."
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
            "content. Nothing was written. Resolve it one of three ways: undo "
            "the edit; keep the edit by setting the row's content_digest to "
            "traigent.utils.regression_set.content_digest(row['input'], "
            "row['output']); or re-run with --on-id-conflict replace, which "
            "applies only when the row's example id is among the failing "
            "examples being added, and DISCARDS the edit by overwriting the "
            "row's input, output and metadata with the incoming dataset "
            "content (the row's earlier provenance entries are kept, marked "
            "with the digest of the content they described)."
        )


@dataclass
class MergeReport:
    """What a merge did."""

    path: Path
    rows_added: int = 0
    provenance_added: int = 0
    rows_replaced: int = 0
    added_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunEvidence:
    """The saved run behind a merge, and why each example counted as failing.

    Attributes:
        run_id: Stable identity of the saved run (its optimization id, or a
            digest of its outcome matrix when the id was not persisted). Part
            of the evidence identity.
        run_name: The name the result was saved under; informational.
        imported: Import date; informational, never part of the identity.
        mode: ``"best_config"`` or ``"all_configs"``.
        threshold: The per-example score below which a cell failed.
        config_digest: ``generate_config_hash`` of the winning configuration;
            ``None`` in ``all_configs`` mode.
        basis: ``example_id -> (failed_by, metric_field)`` as returned by
            ``select_failure_basis``; the field that was compared and whether
            the cells failed by ``score`` or by execution ``error``.
    """

    run_id: str
    run_name: str
    imported: str
    mode: str
    threshold: float
    config_digest: str | None
    basis: dict[str, tuple[str, str]] = field(default_factory=dict)

    def provenance_for(self, example_id: str) -> dict[str, Any]:
        failed_by, metric_field = self.basis.get(example_id, ("unknown", "unknown"))
        return {
            "run_id": self.run_id,
            "run": self.run_name,
            "imported": self.imported,
            "mode": self.mode,
            "threshold": self.threshold,
            "config_digest": self.config_digest,
            "failed_by": failed_by,
            "metric_field": metric_field,
        }


def config_digest_for(mode: str, config: Any) -> str | None:
    """The SDK's config hash for ``best_config`` runs; ``None`` for all_configs."""
    if mode == "all_configs" or config is None:
        return None
    return generate_config_hash(config)


def _evidence_key(entry: Any) -> tuple[Any, ...] | None:
    """Identity of a provenance entry: what was decided, not when it was imported."""
    if not isinstance(entry, dict) or not isinstance(entry.get("run_id"), str):
        return None
    return (entry["run_id"], entry.get("mode"), entry.get("threshold"))


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
    version = regression.get("schema_version")
    if version is None:
        raise _bad(
            path,
            line_no,
            "'metadata.regression.schema_version' is missing (a file from an "
            "earlier preview format; re-create it with 'traigent regression add')",
        )
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise _bad(
            path, line_no, "'metadata.regression.schema_version' is not a version"
        )
    if version > SCHEMA_VERSION:
        raise _bad(
            path,
            line_no,
            f"schema_version {version} is newer than this SDK reads "
            f"({SCHEMA_VERSION}); upgrade traigent",
        )
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
                "schema_version": SCHEMA_VERSION,
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


def _variant_id(example_id: str, digest: str) -> str:
    """The id keep-both gives incoming content that clashes with ``example_id``."""
    return f"{example_id}~{digest.split(':', 1)[1][:8]}"


def _mark_superseded(entry: Any, old_digest: str) -> Any:
    """A provenance entry kept across a replace, tagged with the content it was for."""
    if isinstance(entry, dict) and "superseded_content_digest" not in entry:
        return {**entry, "superseded_content_digest": old_digest}
    return entry


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
    stale_originals: dict[str, str],
) -> None:
    index = by_id.get(example_id)
    if index is None:
        rows.append(_new_row(example, example_id, digest, provenance))
        by_id[example_id] = len(rows) - 1
        report.rows_added += 1
        report.added_ids.append(example_id)
        return
    stored = rows[index]["metadata"]
    variant = _variant_id(example_id, digest)
    if stored["content_digest"] != digest and variant in by_id:
        # An earlier keep-both stored this content under its variant id: the
        # evidence belongs to that row, not to a new conflict on the original id.
        variant_index = by_id[variant]
        if rows[variant_index]["metadata"]["content_digest"] == digest:
            index, stored = variant_index, rows[variant_index]["metadata"]
    if stored["content_digest"] == digest:
        entries = stored["regression"]["provenance"]
        key = _evidence_key(provenance)
        if key is None or all(_evidence_key(entry) != key for entry in entries):
            entries.append(provenance)
            report.provenance_added += 1
        return
    if on_conflict == "replace":
        replacement = _new_row(example, example_id, digest, provenance)
        old_digest = stale_originals.get(example_id, stored["content_digest"])
        kept = [
            _mark_superseded(entry, old_digest)
            for entry in stored["regression"]["provenance"]
        ]
        replacement["metadata"]["regression"]["provenance"] = kept + [provenance]
        rows[index] = replacement
        report.rows_replaced += 1
        return
    if on_conflict == "keep-both":
        alias = _variant_id(example_id, digest)
        _plan_row(
            example,
            alias,
            digest,
            provenance,
            by_id,
            rows,
            "refuse",
            report,
            conflicts,
            stale_originals,
        )
        return
    conflicts.append(example_id)


def merge_failures(
    directory: str | os.PathLike[str],
    examples: Dataset,
    *,
    evidence: RunEvidence,
    on_id_conflict: OnIdConflict = "refuse",
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> MergeReport:
    """Merge failing examples into ``<directory>/regression.jsonl``.

    A new id becomes a row labelled ``seen_once``; a repeat of a stored id with
    the same content digest gains a provenance entry (an entry for the same
    ``run_id``, ``mode`` and ``threshold`` is not added twice, whatever the
    import date or saved name); an id whose content digest differs is handled per
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
            return _merge_locked(path, examples, evidence, on_id_conflict)
    except (FileLockTimeoutError, FileLockUnsupportedError) as exc:
        raise RegressionSetError(str(exc)) from exc


def _admit_stale_rows(
    path: Path,
    numbered: Sequence[tuple[int, dict[str, Any]]],
    rows: list[dict[str, Any]],
    examples: Dataset,
    on_id_conflict: str,
) -> dict[str, str]:
    """Refuse hand-edited rows, except those ``replace`` is about to overwrite.

    Returns ``{example_id: stored content_digest}`` for the rows it admitted,
    so the replace can mark the kept provenance with the digest it described.
    """
    stale = _stale_rows(numbered)
    if not stale:
        return {}
    if on_id_conflict == "replace":
        incoming = {_incoming_id(ex, pos) for pos, ex in enumerate(examples.examples)}
        if all(eid in incoming for eid, _ in stale):
            stale_ids = {eid for eid, _ in stale}
            originals: dict[str, str] = {}
            for row in rows:
                if row["metadata"]["example_id"] in stale_ids:
                    originals[row["metadata"]["example_id"]] = row["metadata"][
                        "content_digest"
                    ]
                    # Forces the replace branch; the new row carries a fresh digest.
                    row["metadata"]["content_digest"] = ""
            return originals
    raise StaleDigestError(path, stale)


def _merge_locked(
    path: Path,
    examples: Dataset,
    evidence: RunEvidence,
    on_id_conflict: str,
) -> MergeReport:
    numbered = _read_numbered_rows(path) if path.exists() else []
    rows = [row for _, row in numbered]
    stale_originals = _admit_stale_rows(path, numbered, rows, examples, on_id_conflict)
    by_id = {row["metadata"]["example_id"]: i for i, row in enumerate(rows)}
    report = MergeReport(path=path)
    conflicts: list[str] = []
    for position, example in enumerate(examples.examples):
        digest = content_digest(example.input_data, example.expected_output)
        incoming_id = _incoming_id(example, position)
        _plan_row(
            example,
            incoming_id,
            digest,
            evidence.provenance_for(incoming_id),
            by_id,
            rows,
            on_id_conflict,
            report,
            conflicts,
            stale_originals,
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

    The file is read, validated and digest-checked exactly once, and the
    dataset is built from those validated rows in memory, so a concurrent
    replacement of the file cannot make the returned rows disagree with the
    labels that were selected. The same duplicate-id rule the file loader
    applies is enforced by the :class:`Dataset` constructor.

    A valid file in which no row carries one of ``labels`` returns an **empty**
    dataset (for example a set that has no ``reliable`` rows yet): that is a
    normal state, not an error, and a caller can tell it from a broken file
    because a broken file raises. Callers that cannot run on zero examples
    should test ``len(dataset)``.

    Raises:
        RegressionSetError: The file is missing, unreadable, has no rows or
            holds a malformed row (or repeated ids).
        StaleDigestError: A row was edited by hand after it was written.
    """
    if isinstance(labels, str):
        raise TypeError("labels must be a sequence of label strings, not one string")
    path = regression_file(directory)
    rows = read_regression_rows(path)
    wanted = set(labels)
    examples = [
        EvaluationExample(
            input_data=row["input"],
            expected_output=row["output"],
            metadata=dict(row["metadata"]),
        )
        for row in rows
        if row["metadata"]["regression"]["label"] in wanted
    ]
    return Dataset(
        examples=examples,
        name="regression_set",
        description=f"Regression set from {path}",
        metadata={"source_path": str(path)},
    )
