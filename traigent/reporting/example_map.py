"""Local-only example content map generation for report enrichment."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jsonschema

from traigent.utils.example_id import LOCAL_MARKER_KEY, sdk_minted_example_id
from traigent.utils.exceptions import ValidationError

EXAMPLE_CONTENT_MAP_SCHEMA_VERSION = "1.0.0"


#: The only wire id shapes the SDK emits; the trailing number is the position of
#: the result within its trial (the dataset row, unless the trial ran a subset).
_WIRE_ID_RE = re.compile(r"^(?:example_(\d+)|ex_[0-9a-f]{8}_(\d+))$")


def _malformed(what: str) -> ValidationError:
    """A content-free error: names the structure, never echoes a value."""
    return ValidationError(f"Malformed run artifact: {what}")


def _local_marker(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """The reserved local-provenance dict; a non-dict value is a refusal.

    A custom metric named like the reserved key hides the subset/privacy record,
    and silently treating the trial as a full-dataset, non-privacy one could map
    ids to the wrong rows or claim ids the portal never had.
    """
    if LOCAL_MARKER_KEY not in metadata:
        return {}
    local = metadata[LOCAL_MARKER_KEY]
    if not isinstance(local, Mapping):
        raise ValidationError(
            f"This run has a metric named {LOCAL_MARKER_KEY}, which hides its "
            "subset/privacy record; the example map cannot be built safely"
        )
    return local


def _trial_subset(metadata: Mapping[str, Any], assume_full: bool) -> list[int] | None:
    """The dataset rows a trial's result positions refer to (``None`` = in order).

    ``rows`` is ``"full"``, the list of dataset rows the result positions refer to,
    or ``"unordered"`` (no row provenance from the evaluator). A trial recorded
    before row provenance existed has none of these. Both that and ``"unordered"``
    are refused unless ``assume_full`` (``--assume-full-dataset``).
    """
    rows = _local_marker(metadata).get("rows")
    if rows is None:
        if assume_full:
            return None
        raise ValidationError(
            "This run was recorded before row provenance existed; pass "
            "--assume-full-dataset if it did not use dataset subsets"
        )
    if rows == "full":
        return None
    if rows == "unordered":
        if assume_full:
            return None
        raise ValidationError(
            "This trial's example ids are not tied to dataset rows (it was "
            "pruned or partial, or its evaluator recorded no row provenance); "
            "pass --assume-full-dataset only if its results are in dataset order"
        )
    if not isinstance(rows, list) or not all(
        isinstance(j, int) and not isinstance(j, bool) and j >= 0 for j in rows
    ):
        raise _malformed("trial rows is neither 'full' nor a list of row indices")
    return rows


MAP_TARGETS = ("connected", "synced")


def resolve_map_target(session: Mapping[str, Any], target: str | None = None) -> str:
    """Which portal session a map is for: ``connected`` or ``synced``.

    Default: ``connected`` when the local session was mirrored from a connected
    backend session (``metadata.backend_session``), else ``synced`` (offline
    sessions reach the portal only through ``traigent sync``).
    """
    if target is not None:
        if target not in MAP_TARGETS:
            raise ValidationError(f"target must be one of {', '.join(MAP_TARGETS)}")
        return target
    session_meta = session.get("metadata")
    if session_meta is not None and not isinstance(session_meta, Mapping):
        raise _malformed("session metadata is not an object")
    return "connected" if (session_meta or {}).get("backend_session") else "synced"


def wire_rows_from_session(
    session: Mapping[str, Any],
    target: str | None = None,
    assume_full_dataset: bool = False,
) -> dict[str, int]:
    """Return ``{wire_example_id: dataset_row_index}`` for a run's local session.

    Reads ``trials[*].metadata.measures[*].example_id`` -- what the run
    resolved -- from a local session (``LocalStorageManager`` or a
    ``traigent local export`` JSON), then, uniformly for ALL trials:

    * ``target="connected"``: keeps the persisted ids, which are what
      ``_log_trial_to_backend`` forwards unchanged to the live run's portal
      session;
    * ``target="synced"``: applies the SAME rewrite ``traigent sync`` applies
      before upload (``SyncManager._content_free_example_measures``), so the
      keys equal the ids of the session sync creates. Sync replays every
      trial, so a connected run later recovered through sync appears in the
      portal as a SECOND session with these ids;
    * maps the id's result position to the dataset row through the trial's
      persisted ``rows`` when it ran a reordered subset.

    ``target=None`` picks per :func:`resolve_map_target`. Only the SDK's own id
    shapes are accepted, structure is validated, and errors never echo values.
    Conflicting rows for one id raise. ``synced`` describes what sync WOULD send:
    it omits the trials sync skips for an empty config (the shared predicate),
    but a partial sync (for example, no backend trial slot available) can still
    leave keys the portal never received. A ``connected`` map of a run recorded
    in privacy mode (``privacy_mode`` under the trial's reserved ``_traigent_local`` metadata key) is refused:
    that run never posted per-example ids. Sessions recorded before this field
    existed carry no such marker and behave as before.
    """
    from traigent.cloud.sync_manager import SyncManager

    trials = session.get("trials")
    if not isinstance(trials, list):
        raise _malformed("'trials' is missing or not a list")
    connected = resolve_map_target(session, target) == "connected"

    found: dict[str, int] = {}
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise _malformed("a trial is not an object")
        metadata = trial.get("metadata")
        if metadata is None:
            continue
        if not isinstance(metadata, Mapping):
            raise _malformed("trial metadata is not an object")
        measures = metadata.get("measures")
        if measures is None:
            continue
        if not isinstance(measures, list):
            raise _malformed("trial measures is not a list")
        for measure in measures:
            raw = measure.get("example_id") if isinstance(measure, Mapping) else None
            if not isinstance(raw, str) or _WIRE_ID_RE.fullmatch(raw) is None:
                raise _malformed("a measure has an unrecognised example id")
        if connected and _local_marker(metadata).get("privacy_mode") is True:
            raise ValidationError(
                "This run was submitted in privacy mode, so the portal has no "
                "per-example ids; use --target synced only if it was later synced"
            )
        subset = _trial_subset(metadata, assume_full_dataset)
        if not connected and not SyncManager.upload_config_for_trial(
            trial.get("config"), metadata
        ):
            continue  # `traigent sync` skips trials with an empty config
        ids = (
            [m["example_id"] for m in measures]
            if connected
            else [
                m["example_id"]
                for m in SyncManager._content_free_example_measures(metadata)
            ]
        )
        for wire_id in ids:
            match = _WIRE_ID_RE.fullmatch(wire_id)
            position = int(match.group(1) or match.group(2)) if match else -1
            if subset is not None:
                if position >= len(subset):
                    raise _malformed("a measure position is outside its trial subset")
                row = subset[position]
            else:
                row = position
            if found.setdefault(wire_id, row) != row:
                raise ValidationError(
                    "Run artifact maps one example id to different dataset rows "
                    "in different trials (reordered subsets); a map cannot "
                    "represent that"
                )
    if not found:
        raise ValidationError("Run artifact contains no per-example measures")
    return found


def build_example_content_map(
    dataset_path: str | Path,
    *,
    dataset_identifier: str | None = None,
    run_session: Mapping[str, Any] | None = None,
    target: str | None = None,
    assume_full_dataset: bool = False,
) -> dict[str, Any]:
    """Build a local-only example content map from a dataset file.

    Entries are keyed by the example ids the portal will see, so it can match
    them to results by exact id.

    * With ``run_session`` (a local session of the run, see
      :func:`wire_rows_from_session`) each id the run produced is keyed to the
      dataset row it evaluated. ``target`` (``connected`` or ``synced``, see
      :func:`resolve_map_target`) names which portal session the map is for: a
      connected run later recovered through ``traigent sync`` appears in the
      portal as a second session with different ids, and you pick it with
      ``target="synced"``. Exact for keyed runs, positional and custom
      evaluators, offline-then-synced runs and subset trials. Rows the run
      never reported (no per-example metrics) are left out.
    * Without it, rows are keyed ``example_<index>``, the id the default
      evaluators record. This is only a fallback: runs with content identity
      keys, or custom evaluators that supply their own ids, send other ids and
      will not match.

    ``assume_full_dataset`` is for artifacts recorded before row provenance
    existed (no ``rows`` in a trial's reserved local record): without it such an
    artifact is refused; with it, result positions are taken as dataset rows.

    ``dataset_identifier`` is accepted for backward compatibility and ignored.
    """
    del dataset_identifier
    path = Path(dataset_path)
    examples = _load_examples(path)
    if run_session is not None:
        rows = wire_rows_from_session(run_session, target, assume_full_dataset)
        if max(rows.values()) >= len(examples):
            raise ValidationError(
                "Run artifact refers to rows beyond the end of the dataset; "
                "it does not belong to this dataset"
            )
    else:
        rows = {sdk_minted_example_id(i): i for i in range(len(examples))}
    example_map: dict[str, Any] = {}

    for example_id, index in sorted(rows.items(), key=lambda item: item[1]):
        example = examples[index]
        example_map[example_id] = {
            "example_num": index + 1,
            "input": example["input"],
            "expected_output": example["expected_output"],
            # Keep stored entries aligned with fingerprint canonicalization.
            "metadata": example["metadata"],
        }

    return {
        "schema_version": EXAMPLE_CONTENT_MAP_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_fingerprint": compute_dataset_fingerprint(examples),
        "example_map": example_map,
    }


def compute_dataset_fingerprint(
    examples: list[dict[str, Any]],
) -> str:
    """Compute deterministic fingerprint over dataset content and order."""
    canonical_examples = [
        {
            "input": ex["input"],
            "expected_output": ex["expected_output"],
            "metadata": ex.get("metadata") or {},
        }
        for ex in examples
    ]
    serialized = json.dumps(
        canonical_examples,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def validate_example_content_map(payload: dict[str, Any]) -> list[str]:
    """Validate a generated example map against schema contract."""
    schema = _load_example_content_map_schema()
    try:
        jsonschema.validate(
            instance=payload,
            schema=schema,
            format_checker=jsonschema.FormatChecker(),
        )
    except jsonschema.SchemaError as exc:
        return [f"Schema definition error: {exc.message}"]
    except jsonschema.ValidationError as exc:
        return [str(exc)]
    return []


def _load_example_content_map_schema() -> dict[str, Any]:
    """Load schema from TraigentSchema package with fallback definition."""
    try:
        from traigent_schema import load_schema as tg_load_schema

        return cast(dict[str, Any], tg_load_schema("example_content_map_schema"))
    except (ImportError, FileNotFoundError):
        return _fallback_example_content_map_schema()


def _load_examples(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_examples_from_jsonl(path)
    if suffix == ".json":
        return _load_examples_from_json(path)
    raise ValidationError(f"Unsupported dataset format: {suffix}. Use .json or .jsonl")


def _load_examples_from_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"Invalid JSON on line {line_no} in {path}: {exc}"
                ) from exc
            examples.append(_normalize_example_item(data, source=f"{path}:{line_no}"))
    if not examples:
        raise ValidationError(f"No examples found in dataset: {path}")
    return examples


def _load_examples_from_json(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        items = payload["examples"]
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise ValidationError(f"Unsupported JSON dataset structure in {path}")

    examples = [
        _normalize_example_item(item, source=f"{path}:{index}")
        for index, item in enumerate(items)
    ]
    if not examples:
        raise ValidationError(f"No examples found in dataset: {path}")
    return examples


def _normalize_example_item(item: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValidationError(f"Dataset entry must be an object: {source}")
    if "input" not in item:
        raise ValidationError(f"Dataset entry missing required 'input' field: {source}")
    if "output" not in item and "expected_output" not in item:
        raise ValidationError(
            f"Dataset entry missing required 'output' or 'expected_output' field: {source}"
        )
    has_output = "output" in item
    has_expected_output = "expected_output" in item
    output_value = item.get("output")
    expected_output_value = item.get("expected_output")
    if has_output and has_expected_output and output_value != expected_output_value:
        raise ValidationError(
            "Dataset entry has conflicting 'output' and 'expected_output' values: "
            f"{source}"
        )
    expected_output = output_value if has_output else expected_output_value
    metadata = {
        key: value
        for key, value in item.items()
        if key not in {"input", "output", "expected_output"}
    }
    return {
        "input": item["input"],
        "expected_output": expected_output,
        "metadata": metadata,
    }


def _fallback_example_content_map_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Example Content Map Schema",
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "pattern": r"^\d+\.\d+\.\d+$",
            },
            "generated_at": {"type": "string", "format": "date-time"},
            "dataset_fingerprint": {"type": "string"},
            "example_map": {
                "type": "object",
                "patternProperties": {
                    "^.+$": {
                        "type": "object",
                        "properties": {
                            "example_num": {"type": "integer", "minimum": 1},
                            "input": {},
                            "expected_output": {},
                            "metadata": {"type": "object"},
                        },
                        "required": ["example_num", "input", "expected_output"],
                        "additionalProperties": False,
                    }
                },
                "additionalProperties": False,
            },
        },
        "required": [
            "schema_version",
            "generated_at",
            "dataset_fingerprint",
            "example_map",
        ],
        "additionalProperties": False,
    }
