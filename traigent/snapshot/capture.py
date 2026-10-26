"""Capture an execution snapshot: raw bytes of what an optimization used.

Everything here is local. The snapshot holds the dataset rows, configuration,
agent source, evaluator source and (opt in) the uncommitted git diff, so a
customer can store them in a bucket they own. Nothing in this module talks to a
Traigent endpoint.
"""

from __future__ import annotations

import functools
import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from traigent.identity.agent_build import (
    agent_bound_state,
    candidate_agent_version,
    collect_agent_build_base,
    git_output,
    innermost_callable,
    sdk_version,
    survey_agent_sources,
)
from traigent.snapshot.manifest import (
    FORMAT,
    ExecutionSnapshot,
    PartMediaType,
    UnavailableReason,
    canonical_manifest_bytes,
    digest_of,
    validate_part_path,
)
from traigent.utils import fp2
from traigent.utils.artifact_fingerprints import (
    compute_agent_fingerprint,
    compute_config_space_fingerprint,
    compute_dataset_fingerprint,
    compute_evaluator_fingerprint,
    effective_evaluator_code_identity,
    example_input_and_expected,
    materialized_examples,
)

__all__ = ["EvaluatorInputs", "build_execution_snapshot", "serialize_example"]

MAX_DIRTY_DIFF_BYTES = 10 * 1024 * 1024
_ECD_ALGORITHM = "ecd1"
_DECLARED_KINDS = ("prompts", "tool_definitions")
_NAME_CLEAN_RE = re.compile(r"[^A-Za-z0-9_-]")
_MAX_NAME = 64

DATASET_PATH = "dataset/examples.jsonl"
CONFIG_PATH = "config/config.json"
CONFIG_SPACE_PATH = "config/config_space.json"
CALLABLE_PATH = "agent/callable.py"
BOUND_STATE_PATH = "agent/bound_state.json"
GIT_PATH = "agent/git.json"
GIT_DIFF_PATH = "agent/git_diff.patch"
EVALUATOR_UNAVAILABLE_PATH = "evaluator/source"


def _asset_path(kind: str) -> str:
    return f"agent/assets/{kind}.json"


@dataclass(frozen=True)
class EvaluatorInputs:
    """The inputs ``effective_evaluator_code_identity`` and fp1 take."""

    custom_evaluator: Callable[..., Any] | None = None
    scoring_function: Callable[..., Any] | None = None
    metric_functions: Mapping[str, Callable[..., Any]] | None = None
    objectives: Iterable[str] | None = None
    external: Any = None


def _json_bytes(value: Any) -> bytes:
    """Strict JSON: a non-JSON value raises (never ``default=str``)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def serialize_example(example: Any) -> dict[str, Any]:
    """One dataset row as a JSON-able dict (``input``, ``expected``, ``id``, ``metadata``).

    ``input`` and ``expected`` are exactly what the dataset fp1 hashes, so
    ``compute_dataset_fingerprint`` of the decoded rows equals the original's.
    """
    input_data, expected = example_input_and_expected(example)
    row: dict[str, Any] = {"input": input_data, "expected": expected}
    if isinstance(example, Mapping):
        identifier = example.get("id")
        metadata = example.get("metadata")
    else:
        identifier = getattr(example, "id", None)
        metadata = getattr(example, "metadata", None)
    if identifier is not None:
        row["id"] = identifier
    if metadata:
        row["metadata"] = metadata
    return row


class _Parts:
    """Accumulates present and unavailable parts."""

    def __init__(self) -> None:
        self.present: dict[str, tuple[bytes, PartMediaType]] = {}
        self.unavailable: dict[str, UnavailableReason] = {}

    def add(self, path: str, data: bytes, media: PartMediaType) -> None:
        validate_part_path(path)
        self.present[path] = (bytes(data), media)

    def skip(self, path: str, reason: UnavailableReason) -> None:
        validate_part_path(path)
        self.unavailable[path] = reason

    def add_json(self, path: str, value: Any) -> None:
        try:
            self.add(path, _json_bytes(value), PartMediaType.JSON)
        except (TypeError, ValueError, RecursionError):
            self.skip(path, UnavailableReason.NOT_SERIALIZABLE)

    def declared(self, path: str) -> bool:
        return path in self.present or path in self.unavailable


def _capture_dataset(parts: _Parts, dataset: Any) -> str | None:
    if dataset is None:
        parts.skip(DATASET_PATH, UnavailableReason.NOT_PROVIDED)
        return None
    examples = materialized_examples(dataset)
    if examples is None:
        parts.skip(DATASET_PATH, UnavailableReason.DATASET_NOT_MATERIALIZED)
        return None
    original_fp: str | None = compute_dataset_fingerprint(dataset)
    try:
        rows = [serialize_example(example) for example in examples]
        data = "".join(_json_bytes(row).decode("utf-8") + "\n" for row in rows).encode(
            "utf-8"
        )
        decoded = [
            json.loads(line) for line in data.decode("utf-8").split("\n") if line
        ]
        if compute_dataset_fingerprint(decoded) != original_fp:
            raise ValueError("dataset does not round-trip")
    except (TypeError, ValueError, AttributeError, RecursionError):
        parts.skip(DATASET_PATH, UnavailableReason.NOT_SERIALIZABLE)
        return original_fp
    parts.add(DATASET_PATH, data, PartMediaType.JSONL)
    return original_fp


def _capture_config(parts: _Parts, config: Any, config_space: Any) -> None:
    for path, value in ((CONFIG_PATH, config), (CONFIG_SPACE_PATH, config_space)):
        if value is None:
            parts.skip(path, UnavailableReason.NOT_PROVIDED)
        else:
            parts.add_json(path, value)


def _callable_text(func: Any) -> str | None:
    target = func.func if isinstance(func, functools.partial) else func
    inner = innermost_callable(target)
    try:
        return inspect.getsource(getattr(inner, "__func__", inner))
    except (OSError, TypeError):
        return None


def _capture_git(
    parts: _Parts,
    survey: Any,
    *,
    capture_git_metadata: bool,
    capture_dirty_diff: bool,
) -> None:
    root = survey.project_root
    if not capture_git_metadata:
        parts.skip(GIT_PATH, UnavailableReason.NOT_REQUESTED)
    elif survey.commit is None or root is None:
        parts.skip(GIT_PATH, UnavailableReason.GIT_UNAVAILABLE)
    else:
        untracked = git_output(root, "ls-files", "--others", "--exclude-standard", "-z")
        entry = None
        if survey.entry_file is not None:
            try:
                entry = survey.entry_file.relative_to(root).as_posix()
            except ValueError:
                entry = None
        parts.add_json(
            GIT_PATH,
            {
                "commit": survey.commit,
                "dirty": survey.dirty,
                "entry_path": entry,
                "untracked_files_excluded": (
                    len([n for n in untracked.split("\0") if n])
                    if isinstance(untracked, str)
                    else None
                ),
            },
        )
    if not survey.dirty or root is None:
        return
    if not capture_dirty_diff:
        parts.skip(GIT_DIFF_PATH, UnavailableReason.NOT_REQUESTED)
        return
    diff = git_output(
        root,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--binary",
        "HEAD",
        "--",
        ".",
        binary=True,
    )
    if not isinstance(diff, bytes):
        parts.skip(GIT_DIFF_PATH, UnavailableReason.GIT_UNAVAILABLE)
    elif len(diff) > MAX_DIRTY_DIFF_BYTES:
        parts.skip(GIT_DIFF_PATH, UnavailableReason.TOO_LARGE)
    else:
        parts.add(GIT_DIFF_PATH, diff, PartMediaType.PATCH)


def _capture_agent(
    parts: _Parts,
    agent: Callable[..., Any] | None,
    declared_assets: Mapping[str, Any] | None,
    *,
    capture_git_metadata: bool,
    capture_dirty_diff: bool,
) -> tuple[str, tuple[str, ...]]:
    """Capture the agent; return ``(snapshot coverage, content-free gaps)``."""
    declared = dict(declared_assets or {})
    if agent is None:
        for path in (CALLABLE_PATH, BOUND_STATE_PATH, GIT_PATH):
            parts.skip(path, UnavailableReason.NOT_PROVIDED)
        for kind in _DECLARED_KINDS:
            parts.skip(_asset_path(kind), UnavailableReason.NOT_PROVIDED)
        return "partial", ("agent_not_provided",)

    gaps: list[str] = []
    source = _callable_text(agent)
    if source is None:
        parts.skip(CALLABLE_PATH, UnavailableReason.SOURCE_UNAVAILABLE)
        gaps.append("callable_source_unavailable")
    else:
        parts.add(CALLABLE_PATH, source.encode("utf-8"), PartMediaType.PYTHON)

    survey = survey_agent_sources(agent)
    gaps.extend(survey.gaps)
    for name, helper in survey.helper_files.items():
        member = f"agent/source/{name}"
        try:
            parts.add(member, helper.read_bytes(), PartMediaType.PYTHON)
        except OSError:
            parts.skip(member, UnavailableReason.SOURCE_UNAVAILABLE)
            gaps.append("helper_module_unreadable")

    for kind in _DECLARED_KINDS:
        if kind not in declared:
            parts.skip(_asset_path(kind), UnavailableReason.NOT_PROVIDED)
            gaps.append(f"{kind}_not_provided")
        else:
            parts.add_json(_asset_path(kind), declared[kind])

    bound = agent_bound_state(agent)
    bound_bytes: bytes | None = None
    if bound is not None:
        try:
            bound_bytes = fp2.canonicalize(bound).encode("utf-8")
        except fp2.Fp2UnsupportedValue:
            bound_bytes = None
    if bound_bytes is None:
        parts.skip(BOUND_STATE_PATH, UnavailableReason.BOUND_STATE_UNAVAILABLE)
        gaps.append("bound_state_unavailable")
    else:
        parts.add(BOUND_STATE_PATH, bound_bytes, PartMediaType.JSON)

    _capture_git(
        parts,
        survey,
        capture_git_metadata=capture_git_metadata,
        capture_dirty_diff=capture_dirty_diff,
    )
    stored_all = not [
        path
        for path in parts.unavailable
        if path.startswith("agent/") and path not in (GIT_PATH, GIT_DIFF_PATH)
    ]
    coverage = "complete" if stored_all and not gaps else "partial"
    return coverage, tuple(sorted(set(gaps)))


def _safe_name(name: str, taken: set[str]) -> str:
    base = _NAME_CLEAN_RE.sub("_", str(name))[:_MAX_NAME].strip("_") or "unnamed"
    candidate, index = base, 1
    while candidate in taken:
        index += 1
        candidate = f"{base}-{index}"
    taken.add(candidate)
    return candidate


def _capture_evaluator(
    parts: _Parts, evaluator: EvaluatorInputs | None
) -> tuple[str | None, dict[str, Any]]:
    if evaluator is None:
        parts.skip(EVALUATOR_UNAVAILABLE_PATH, UnavailableReason.NOT_PROVIDED)
        return None, {"origin": "not_provided", "digest": None, "algorithm": None}

    objectives = list(evaluator.objectives or ())
    metric_functions = dict(evaluator.metric_functions or {})
    fp1 = compute_evaluator_fingerprint(
        evaluator.custom_evaluator,
        evaluator.scoring_function,
        metric_functions or None,
        evaluator.external,
    )
    identity = effective_evaluator_code_identity(
        custom_evaluator=evaluator.custom_evaluator,
        scoring_function=evaluator.scoring_function,
        metric_functions=metric_functions or None,
        objectives=objectives,
        external=evaluator.external,
    )
    code_identity = {
        "origin": identity.get("origin"),
        "digest": identity.get("digest"),
        "algorithm": _ECD_ALGORITHM if identity.get("digest") else None,
    }

    members: list[tuple[str, str, Callable[..., Any]]] = []
    if evaluator.custom_evaluator is not None:
        members.append(
            (
                "custom_evaluator",
                getattr(evaluator.custom_evaluator, "__name__", "evaluator"),
                evaluator.custom_evaluator,
            )
        )
    if evaluator.scoring_function is not None:
        members.append(
            (
                "scoring_function",
                getattr(evaluator.scoring_function, "__name__", "scorer"),
                evaluator.scoring_function,
            )
        )
    members.extend(
        ("metric", name, fn) for name, fn in sorted(metric_functions.items())
    )
    if not members:
        reason = {
            "external": UnavailableReason.EXTERNAL_EVALUATOR,
            "builtin": UnavailableReason.BUILTIN_EVALUATOR,
        }.get(str(identity.get("origin")), UnavailableReason.SOURCE_UNAVAILABLE)
        parts.skip(EVALUATOR_UNAVAILABLE_PATH, reason)
        return fp1, code_identity
    taken: dict[str, set[str]] = {}
    for role, name, function in members:
        label = _safe_name(name, taken.setdefault(role, set()))
        path = f"evaluator/{role}/{label}.py"
        try:
            parts.add(
                path, inspect.getsource(function).encode("utf-8"), PartMediaType.PYTHON
            )
        except (OSError, TypeError):
            parts.skip(path, UnavailableReason.SOURCE_UNAVAILABLE)
    return fp1, code_identity


def _agent_build(
    agent: Callable[..., Any] | None, agent_id: str | None, config: Any
) -> dict[str, Any]:
    if agent is None or agent_id is None:
        return {
            "agent_build_digest": None,
            "agent_build_coverage": None,
            "agent_build_unavailable": UnavailableReason.NOT_PROVIDED.value,
        }
    base = collect_agent_build_base(agent, agent_id=agent_id)
    version = candidate_agent_version(
        base, config if isinstance(config, Mapping) else None
    )
    if base is None or version is None:
        return {
            "agent_build_digest": None,
            "agent_build_coverage": None,
            "agent_build_unavailable": UnavailableReason.SOURCE_UNAVAILABLE.value,
        }
    return {
        "agent_build_digest": version["build_digest"],
        "agent_build_coverage": base.coverage,
        "agent_build_unavailable": None,
    }


def _now() -> datetime:
    return datetime.now(UTC)


def build_execution_snapshot(
    *,
    dataset: Any = None,
    config: Mapping[str, Any] | None = None,
    config_space: Any = None,
    agent: Callable[..., Any] | None = None,
    evaluator: EvaluatorInputs | None = None,
    agent_id: str | None = None,
    capture_git_metadata: bool = True,
    capture_dirty_diff: bool = False,
    declared_assets: Mapping[str, Any] | None = None,
    clock: Callable[[], datetime] = _now,
) -> ExecutionSnapshot:
    """Capture a snapshot of the given inputs, entirely locally.

    Nothing is fetched or sent. A part that cannot be captured is recorded as
    unavailable with a closed reason code instead of being guessed. The result
    is an *unbound client declaration*: it says these bytes were captured, not
    that they are what ran.

    ``capture_dirty_diff`` is opt in: the diff of uncommitted changes can
    contain anything in the working tree (untracked files are never included
    and are only counted).
    """
    if declared_assets is not None:
        if not isinstance(declared_assets, Mapping):
            raise TypeError("declared_assets must be a mapping")
        if set(declared_assets) - set(_DECLARED_KINDS):
            raise ValueError(
                "declared_assets keys must be 'prompts' or 'tool_definitions'"
            )
    if agent_id is not None and not isinstance(agent_id, str):
        raise TypeError("agent_id must be a string")

    parts = _Parts()
    dataset_fp = _capture_dataset(parts, dataset)
    _capture_config(parts, config, config_space)
    coverage, gaps = _capture_agent(
        parts,
        agent,
        declared_assets,
        capture_git_metadata=capture_git_metadata,
        capture_dirty_diff=capture_dirty_diff,
    )
    evaluator_fp1, code_identity = _capture_evaluator(parts, evaluator)

    fixed = [
        DATASET_PATH,
        CONFIG_PATH,
        CONFIG_SPACE_PATH,
        CALLABLE_PATH,
        BOUND_STATE_PATH,
        GIT_PATH,
        *(_asset_path(kind) for kind in _DECLARED_KINDS),
    ]
    evaluator_paths = sorted(
        path
        for path in (*parts.present, *parts.unavailable)
        if path.startswith("evaluator/")
    )
    manifest_parts: dict[str, dict[str, Any]] = {}
    for path, (data, media) in parts.present.items():
        manifest_parts[path] = {
            "sha256": digest_of(data),
            "size": len(data),
            "media_type": media.value,
        }
    for path, reason in parts.unavailable.items():
        manifest_parts[path] = {"status": "unavailable", "reason": reason.value}
    when = clock().astimezone(UTC).replace(microsecond=0)
    manifest = {
        "format": FORMAT,
        "sdk_version": sdk_version(),
        "created_at": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "binding": {"kind": "unbound"},
        "parts": manifest_parts,
        "required_parts": sorted(
            {p for p in fixed if parts.declared(p)} | set(evaluator_paths)
        ),
        "fingerprints": {
            "dataset_fp1": dataset_fp,
            "config_space_fp1": compute_config_space_fingerprint(config_space),
            "agent_fp1": compute_agent_fingerprint(agent),
            "evaluator_fp1": evaluator_fp1,
            "evaluator_code_identity": code_identity,
            **_agent_build(agent, agent_id, config),
        },
        "snapshot_coverage": {"agent": coverage, "agent_gaps": list(gaps)},
    }
    return ExecutionSnapshot(
        canonical_manifest_bytes(manifest),
        {path: data for path, (data, _) in parts.present.items()},
    )
