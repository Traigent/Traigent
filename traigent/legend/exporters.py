"""Exporters that read ACTUAL content (contract delta C4) and the matching verifiers.

``export_pillar`` returns a :class:`Pillar` whose representations are canonical
byte streams; ``recompute_fingerprint`` re-derives the pillar fingerprint from
restored representations alone, which is what makes a restore verifiable.
Anything that cannot be exported or re-derived lowers coverage or makes the
fingerprint unrecomputable -- never silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # nosec B404 - fixed git argv, no shell
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.identity.content_identity import (
    ContentIdentityError,
    canonical_bytes,
    compute_agent_build_digest,
    compute_evaluator_version_digest,
)
from traigent.identity.examples import identify_dataset
from traigent.identity.keys import ContentIdentityKeys
from traigent.legend import _contract as c
from traigent.legend.bundle import pack_tree
from traigent.legend.errors import LegendError
from traigent.legend.pillars import (
    ConfigurationSource,
    DatasetSource,
    Pillar,
    Representation,
    SearchSpaceSource,
    compute_pillar,
)
from traigent.utils import fp2

DEFAULT_MAX_FILE_BYTES: Final = 5 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES: Final = 64 * 1024 * 1024
# What the whole source tree stands for once every file is in the bundle.
_TREE_COMPONENTS: Final = ("code", "prompts", "helper_modules", "tool_definitions")
_ALWAYS_SKIP_DIRS: Final = frozenset({".git", "__pycache__"})
_GIT_TIMEOUT_S: Final = 30


@dataclass(frozen=True)
class AgentStateSource:
    """The agent's code state: committed tree + uncommitted + untracked files.

    ``.gitignore`` is respected inside a git work tree. A file over
    ``max_file_bytes`` (or past ``max_total_bytes``) is skipped and coverage
    drops to partial; a symlink is skipped the same way.
    """

    root: str | os.PathLike[str]
    agent_id: str
    config: Mapping[str, Any] | None = None
    lockfile: str | os.PathLike[str] | None = None
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    sdk_version: str = "0"


@dataclass(frozen=True)
class EvaluatorFilesSource:
    """Evaluator source files (relative to ``root``) plus judge config and objectives."""

    root: str | os.PathLike[str]
    files: Sequence[str]
    objectives: Sequence[Mapping[str, Any]]
    evaluator_id: str
    judge: Mapping[str, Any] | None = None
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _list_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed argv
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        ).stdout
        return sorted({p for p in out.decode("utf-8").split("\0") if p})
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        found = []
        for base, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _ALWAYS_SKIP_DIRS]
            for name in names:
                found.append(Path(base, name).relative_to(root).as_posix())
        return sorted(found)


def _read_tree(
    root: Path, names: Sequence[str], max_file: int, max_total: int
) -> tuple[list[tuple[str, bool, bytes]], int]:
    entries: list[tuple[str, bool, bytes]] = []
    skipped = 0
    total = 0
    for name in names:
        path = root / name
        try:
            if path.is_symlink() or not path.is_file():
                # deleted tracked files simply are not part of the current state
                if path.is_symlink():
                    skipped += 1
                continue
            size = path.stat().st_size
            if size > max_file or total + size > max_total:
                skipped += 1
                continue
            data = path.read_bytes()
        except OSError:
            skipped += 1
            continue
        total += len(data)
        entries.append((name, bool(path.stat().st_mode & 0o111), data))
    return entries, skipped


def _agent_manifest(
    agent_id: str,
    sdk_version: str,
    tree: bytes,
    config: bytes,
    lock: bytes | None,
    skipped: int,
) -> dict[str, Any]:
    source = {"tree": _sha(tree), "lock": _sha(lock) if lock is not None else None}
    return {
        "manifest_version": 1,
        "agent_id": agent_id,
        "runtime": {"language": "python", "sdk_version": sdk_version},
        "source_digest": _sha(canonical_bytes(source)),
        "asset_digests": {"helper_modules": {}, "prompts": {}, "tool_definitions": {}},
        "applied_config_digest": _sha(config),
        "coverage": "partial" if skipped else "complete",
    }


@dataclass(frozen=True)
class AgentStateExport:
    pillar: Pillar
    manifest: dict[
        str, Any
    ]  # declare with AgentManifestSource(manifest) to get this fingerprint
    skipped: int


def export_agent_state(source: AgentStateSource) -> AgentStateExport:
    root = Path(source.root)
    if not root.is_dir():
        raise LegendError("invalid_request", "agent root is not a directory")
    entries, skipped = _read_tree(
        root, _list_files(root), source.max_file_bytes, source.max_total_bytes
    )
    tree = pack_tree(entries)
    config = canonical_bytes(dict(source.config or {}))
    reps = [
        Representation(c.REPR_SOURCE_BUNDLE, tree),
        Representation(c.REPR_CONFIG_DOCUMENT, config),
    ]
    covered = ["applied_config"]
    if not skipped:
        covered += list(_TREE_COMPONENTS)
    lock: bytes | None = None
    if source.lockfile is not None:
        try:
            lock = Path(source.lockfile).read_bytes()
        except OSError:
            raise LegendError("invalid_request", "lockfile could not be read") from None
        reps.append(Representation(c.REPR_DEPENDENCY_LOCK, lock))
        covered.append("dependency_lock")
    manifest = _agent_manifest(
        source.agent_id, source.sdk_version, tree, config, lock, skipped
    )
    try:
        digest = compute_agent_build_digest(manifest)
    except ContentIdentityError:
        raise LegendError(
            "invalid_request", "agent state could not be fingerprinted"
        ) from None
    pillar = Pillar(
        c.KIND_AGENT,
        c.SCHEME_BY_KIND[c.KIND_AGENT],
        None,
        digest,
        c.IDENTITY_SDK_FINGERPRINTED,
        tuple(reps),
        tuple(covered),
    )
    return AgentStateExport(pillar, manifest, skipped)


def _evaluator_manifest(
    evaluator_id: str, code: bytes, doc: bytes, judge: Any, objectives: Any
) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "evaluator_id": evaluator_id,
        "code_digest": _sha(code),
        "config_digest": _sha(doc),
        "helper_digests": {},
        "judge": judge,
        "objectives": objectives,
        "dependency_versions": {},
    }


def export_evaluator_files(source: EvaluatorFilesSource) -> Pillar:
    root = Path(source.root)
    entries, skipped = _read_tree(
        root, sorted(source.files), source.max_file_bytes, 1 << 62
    )
    if skipped or len(entries) != len(set(source.files)) or not entries:
        raise LegendError(
            "coverage_insufficient", "evaluator files could not all be read"
        )
    code = pack_tree(entries)
    objectives = sorted(
        (dict(o) for o in source.objectives), key=lambda o: str(o.get("name"))
    )
    judge = dict(source.judge) if source.judge is not None else None
    try:
        doc = canonical_bytes({"judge": judge, "objectives": objectives})
        manifest = _evaluator_manifest(
            source.evaluator_id, code, doc, judge, objectives
        )
        digest = compute_evaluator_version_digest(manifest)
    except ContentIdentityError:
        raise LegendError(
            "invalid_request", "evaluator could not be fingerprinted"
        ) from None
    return Pillar(
        c.KIND_EVALUATOR,
        c.SCHEME_BY_KIND[c.KIND_EVALUATOR],
        None,
        digest,
        c.IDENTITY_SDK_FINGERPRINTED,
        (
            Representation(c.REPR_EVALUATOR_BUNDLE, code),
            Representation(c.REPR_JUDGE_CONFIG, doc),
        ),
        ("evaluator_code", "judge_config", "objectives"),
    )


def export(source: Any) -> tuple[Pillar, dict[str, Any]]:
    """Export real content for ``source``: the pillar plus the non-content re-derivation inputs.

    Refuses sources that carry only digests (nothing recoverable).
    """
    if isinstance(source, AgentStateSource):
        exp = export_agent_state(source)
        return exp.pillar, {
            "agent_id": source.agent_id,
            "sdk_version": source.sdk_version,
            "skipped": exp.skipped,
        }
    if isinstance(source, EvaluatorFilesSource):
        return export_evaluator_files(source), {"evaluator_id": source.evaluator_id}
    if isinstance(source, ConfigurationSource):
        return compute_pillar(c.KIND_CONFIGURATION, source), {}
    if isinstance(source, DatasetSource):
        return compute_pillar(c.KIND_DATASET, source), {}
    if isinstance(source, SearchSpaceSource):
        if source.definition is None:
            raise LegendError(
                "coverage_insufficient", "search space definition missing"
            )
        return compute_pillar(c.KIND_SEARCH_SPACE, source), {}
    raise LegendError("coverage_insufficient", "source carries no recoverable content")


def export_pillar(source: Any) -> Pillar:
    return export(source)[0]


def _rep(reps: Sequence[bytes], kinds: Sequence[str], kind: str) -> bytes | None:
    for k, data in zip(kinds, reps, strict=False):
        if k == kind:
            return data
    return None


def recompute_fingerprint(
    kind: str,
    kinds: Sequence[str],
    reps: Sequence[bytes],
    inputs: Mapping[str, Any],
    keys: ContentIdentityKeys | None = None,
) -> str | None:
    """Re-derive the pillar fingerprint from representations; ``None`` = not derivable here."""
    try:
        if kind == c.KIND_AGENT:
            tree = _rep(reps, kinds, c.REPR_SOURCE_BUNDLE)
            config = _rep(reps, kinds, c.REPR_CONFIG_DOCUMENT)
            if tree is None or config is None:
                return None
            lock = _rep(reps, kinds, c.REPR_DEPENDENCY_LOCK)
            manifest = _agent_manifest(
                str(inputs["agent_id"]),
                str(inputs["sdk_version"]),
                tree,
                config,
                lock,
                int(inputs["skipped"]),
            )
            return str(compute_agent_build_digest(manifest))
        if kind == c.KIND_CONFIGURATION:
            raw = _rep(reps, kinds, c.REPR_CONFIG_DOCUMENT)
            return None if raw is None else str(fp2.digest(json.loads(raw)))
        if kind == c.KIND_DATASET:
            raw = _rep(reps, kinds, c.REPR_DATASET_ROWS)
            if raw is None:
                return None
            examples = [
                EvaluationExample(
                    input_data=row["input"],
                    expected_output=row["expected"],
                    metadata=row["metadata"],
                )
                for row in json.loads(raw)
            ]
            identity = identify_dataset(Dataset(examples=examples), keys)
            return None if identity is None else identity.dataset_root
        if kind == c.KIND_EVALUATOR:
            code = _rep(reps, kinds, c.REPR_EVALUATOR_BUNDLE)
            doc = _rep(reps, kinds, c.REPR_JUDGE_CONFIG)
            if code is None or doc is None:
                return None
            parsed = json.loads(doc)
            manifest = _evaluator_manifest(
                str(inputs["evaluator_id"]),
                code,
                doc,
                parsed["judge"],
                parsed["objectives"],
            )
            return str(compute_evaluator_version_digest(manifest))
    except (
        ContentIdentityError,
        KeyError,
        TypeError,
        ValueError,
        fp2.Fp2UnsupportedValue,
    ):
        return None
    return None  # search_space: the id is Backend-keyed, not derivable in the SDK
