"""Exporters that read ACTUAL content (contract delta C4) and the matching verifiers.

``export_pillar`` returns a :class:`Pillar` whose representations are canonical
byte streams; ``recompute_fingerprint`` re-derives the pillar fingerprint from
restored representations alone, which is what makes a restore verifiable.
Anything that cannot be exported or re-derived lowers coverage or makes the
fingerprint unrecomputable -- never silently.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import subprocess  # nosec B404 - fixed git argv, no shell
from collections.abc import Callable, Mapping, Sequence
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
from traigent.identity import agent_build as _ab
from traigent.identity.evaluator_version import build_evaluator_binding
from traigent.identity.examples import identify_dataset
from traigent.identity.keys import ContentIdentityKeys
from traigent.legend import _contract as c
from traigent.legend.bundle import pack_tree, unpack_tree
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
_ALWAYS_SKIP_DIRS: Final = frozenset({".git", "__pycache__"})
_GIT_TIMEOUT_S: Final = 30


@dataclass(frozen=True)
class AgentStateSource:
    """The agent's captured file state, tied to the identity the ledger version records.

    The identity side is ``func`` (the declared callable: the identity code derives the
    ``AgentBuildManifestV1`` from it) or ``declared_manifest`` (a ready manifest).
    The exporter then re-reads the SAME file set the identity code hashed -- the helper
    modules named in the manifest, the lockfile, the prompt / tool files you map to the
    declared asset names -- recomputes the build digest from those bytes and compares it
    with the declared one (INV-17). ``root`` must be the project root (the git top
    level), because declared asset names are relative to it.
    """

    root: str | os.PathLike[str]
    agent_id: str
    config: Mapping[str, Any] | None = None
    func: Callable[..., Any] | None = None
    declared_manifest: Mapping[str, Any] | None = None
    declared_fingerprint: str | None = None
    prompt_files: Mapping[str, str] | None = (
        None  # declared asset name -> path under root
    )
    tool_definition_files: Mapping[str, str] | None = None
    lockfile: str | os.PathLike[str] | None = None
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES


@dataclass(frozen=True)
class EvaluatorFilesSource:
    """An evaluator's captured helper files, tied to its declared evaluator version.

    ``evaluator`` (+ ``objectives``) lets the identity code derive the manifest; or pass
    ``declared_manifest``. The exporter re-reads the files named in the manifest's
    ``helper_digests`` under ``root`` (the git top level) and recomputes the version
    digest from them (INV-17). Without the evaluator object the code leaf cannot be
    re-derived, so ``evaluator_code`` stays missing.
    """

    root: str | os.PathLike[str]
    evaluator: Any = None
    objectives: Any = None
    evaluator_id: str | None = None
    declared_manifest: Mapping[str, Any] | None = None
    declared_fingerprint: str | None = None
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


def _read_declared(root: Path, name: str, max_file: int) -> bytes | None:
    """Bytes of the declared file ``root/name``; ``None`` if absent, unsafe, a symlink or too big."""
    parts = name.split("/")
    if not name or name.startswith("/") or any(p in ("", ".", "..") for p in parts):
        return None
    path = root / name
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > max_file:
            return None
        return path.read_bytes()
    except OSError:
        return None


def _asset_bytes(raw: bytes, declared_digest: str) -> bytes | None:
    """The bytes whose plain sha256 is the declared asset digest, derived from ``raw``.

    A text / bytes asset digests as its bytes; a structured specification digests as its
    fp2 canonical JSON. ``None`` means the file does not reproduce the declared digest.
    """
    if _sha(raw) == declared_digest:
        return raw
    try:
        canonical = fp2.canonicalize(json.loads(raw)).encode("utf-8")
    except (ValueError, TypeError, fp2.Fp2UnsupportedValue):
        return None
    return canonical if _sha(canonical) == declared_digest else None


def _pack_or_empty(entries: Sequence[tuple[str, bool, bytes]]) -> bytes:
    """An empty category is asserted by a zero-length representation, never skipped."""
    return pack_tree(entries) if entries else b""


def _unpack_or_empty(data: bytes | None) -> list[tuple[str, bool, bytes]] | None:
    if data is None:
        return None
    return unpack_tree(data) if data else []


_LOCK_NAMES: Final = ("uv.lock", "poetry.lock", "Pipfile.lock", "pdm.lock")
_NON_FILE_KEYS: Final = (
    "asset_digests",
    "applied_config_digest",
    "dependency_lock_digest",
)


def _declared_agent(source: AgentStateSource) -> tuple[dict[str, Any], str] | None:
    """``(AgentBuildManifestV1, build_digest)`` exactly as the identity code derives it."""
    if source.func is not None:
        base = _ab.collect_agent_build_base(source.func, agent_id=source.agent_id)
        version = _ab.candidate_agent_version(base, source.config)
        if version is None:
            return None
        return version["manifest"], version["build_digest"]
    if source.declared_manifest is not None:
        manifest = copy.deepcopy(dict(source.declared_manifest))
        try:
            return manifest, compute_agent_build_digest(manifest)
        except ContentIdentityError:
            return None
    return None


def _entry_file_matches(
    func: Callable[..., Any], root: Path, tree: Sequence[tuple[str, bool, bytes]]
) -> bool:
    """The callable's source text appears in the captured copy of its own source file."""
    target = _ab.innermost_callable(func)
    try:
        filename = inspect.getsourcefile(getattr(target, "__func__", target))
        text = _ab.normalize_source(
            inspect.getsource(getattr(target, "__func__", target))
        )
        relative = Path(filename).resolve().relative_to(root.resolve()).as_posix()  # type: ignore[arg-type]
    except (OSError, TypeError, ValueError):
        return False
    if text is None:
        return False
    wanted = [line.strip() for line in text.split("\n")]
    for name, _exe, data in tree:
        if name != relative:
            continue
        have = [line.strip() for line in data.decode("utf-8", "replace").split("\n")]
        return any(
            have[i : i + len(wanted)] == wanted
            for i in range(len(have) - len(wanted) + 1)
        )
    return False


@dataclass(frozen=True)
class AgentStateExport:
    pillar: Pillar
    manifest: dict[str, Any]  # the declared AgentBuildManifestV1
    skipped: int


def _asset_category(
    root: Path,
    declared: Mapping[str, str],
    files: Mapping[str, str] | None,
    max_file: int,
    *,
    by_name: bool,
) -> tuple[list[tuple[str, bool, bytes]], dict[str, str], bool, int]:
    """Re-read one asset category; ``(stored entries, recomputed digests, all_match, skipped)``."""
    entries: list[tuple[str, bool, bytes]] = []
    digests: dict[str, str] = {}
    ok, skipped = True, 0
    for name, want in sorted(declared.items()):
        path = name if by_name else (files or {}).get(name)
        raw = _read_declared(root, path, max_file) if path is not None else None
        if raw is None:
            ok, skipped = False, skipped + 1
            continue
        stored = _asset_bytes(raw, want)
        if stored is None:
            stored, ok = raw, False
        digests[name] = _sha(stored)
        entries.append((name, False, stored))
    return entries, digests, ok, skipped


def export_agent_state(source: AgentStateSource) -> AgentStateExport:
    root = Path(source.root)
    if not root.is_dir():
        raise LegendError("invalid_request", "agent root is not a directory")
    declared = _declared_agent(source)
    if declared is None:
        raise LegendError(
            "coverage_insufficient",
            "agent identity could not be derived (pass func or declared_manifest)",
        )
    manifest, declared_digest = declared
    fingerprint_ok = source.declared_fingerprint in (None, declared_digest)
    tree_entries, skipped = _read_tree(
        root, _list_files(root), source.max_file_bytes, source.max_total_bytes
    )
    tree = pack_tree(tree_entries)
    assets = manifest["asset_digests"]
    helper_e, helper_d, helper_ok, sk = _asset_category(
        root, assets["helper_modules"], None, source.max_file_bytes, by_name=True
    )
    skipped += sk
    prompt_e, prompt_d, prompts_ok, sk = _asset_category(
        root,
        assets["prompts"],
        source.prompt_files,
        source.max_file_bytes,
        by_name=False,
    )
    skipped += sk
    tool_e, tool_d, tools_ok, sk = _asset_category(
        root,
        assets["tool_definitions"],
        source.tool_definition_files,
        source.max_file_bytes,
        by_name=False,
    )
    skipped += sk
    lock: bytes | None = None
    lock_path = Path(source.lockfile) if source.lockfile is not None else None
    for name in _LOCK_NAMES if lock_path is None else ():
        if (root / name).is_file():
            lock_path = root / name
            break
    if lock_path is not None:
        try:
            lock = lock_path.read_bytes()
        except OSError:
            skipped += 1
    lock_digest = _sha(lock) if lock is not None else None
    lock_ok = lock_digest == manifest.get("dependency_lock_digest")
    try:
        config = canonical_bytes(dict(source.config or {}))
        applied = str(fp2.digest(dict(source.config or {})))
    except (ContentIdentityError, fp2.Fp2UnsupportedValue, TypeError, ValueError):
        raise LegendError("invalid_request", "config is not serializable") from None
    applied_ok = applied == manifest["applied_config_digest"]
    code_ok = (
        not skipped
        and source.func is not None
        and _ab.afp2_source_digest(source.func) == manifest.get("source_digest")
        and _entry_file_matches(source.func, root, tree_entries)
    )
    frame = {k: v for k, v in manifest.items() if k not in _NON_FILE_KEYS}
    rebuilt = _agent_manifest_from(
        frame,
        {"helper_modules": helper_d, "prompts": prompt_d, "tool_definitions": tool_d},
        applied,
        lock_digest,
    )
    try:
        recomputed: str | None = compute_agent_build_digest(rebuilt)
    except ContentIdentityError:
        recomputed = None
    reproduced = (
        recomputed == declared_digest
        and fingerprint_ok
        and helper_ok
        and prompts_ok
        and tools_ok
        and lock_ok
        and applied_ok
        and bool(code_ok)
    )
    reps = [
        Representation(c.REPR_SOURCE_BUNDLE, tree),
        Representation(c.REPR_HELPER_MODULES, _pack_or_empty(helper_e)),
        Representation(c.REPR_PROMPT_FILES, _pack_or_empty(prompt_e)),
        Representation(c.REPR_TOOL_DEFINITIONS, _pack_or_empty(tool_e)),
        Representation(c.REPR_CONFIG_DOCUMENT, config),
    ]
    if lock is not None:
        reps.append(Representation(c.REPR_DEPENDENCY_LOCK, lock))
    covered = [
        name
        for name, ok in (
            ("code", code_ok),
            ("prompts", prompts_ok),
            ("helper_modules", helper_ok),
            ("tool_definitions", tools_ok),
            ("applied_config", applied_ok),
            ("runtime", "runtime" in frame),
            ("dependency_lock", lock is not None and lock_ok),
        )
        if ok
    ]
    pillar = Pillar(
        c.KIND_AGENT,
        c.SCHEME_BY_KIND[c.KIND_AGENT],
        None,
        source.declared_fingerprint
        or declared_digest,  # the ledger's, never file-derived
        c.IDENTITY_SDK_FINGERPRINTED,
        tuple(reps),
        tuple(covered),
        reproduced=reproduced,
        skipped=skipped,
    )
    return AgentStateExport(pillar, manifest, skipped)


def _agent_manifest_from(
    frame: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, str]],
    applied: str,
    lock_digest: str | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = dict(frame)
    manifest["asset_digests"] = {k: dict(v) for k, v in assets.items()}
    manifest["applied_config_digest"] = applied
    if lock_digest is not None:
        manifest["dependency_lock_digest"] = lock_digest
    return manifest


def export_evaluator_files(source: EvaluatorFilesSource) -> Pillar:
    root = Path(source.root)
    if not root.is_dir():
        raise LegendError("invalid_request", "evaluator root is not a directory")
    if source.evaluator is not None:
        binding, _src, _reason = build_evaluator_binding(
            source.evaluator,
            objectives=source.objectives,
            evaluator_id=source.evaluator_id,
        )
        if binding is None:
            raise LegendError("invalid_request", "evaluator could not be fingerprinted")
        manifest, declared_digest = dict(binding["manifest"]), binding["version_digest"]
    elif source.declared_manifest is not None:
        manifest = copy.deepcopy(dict(source.declared_manifest))
        try:
            declared_digest = compute_evaluator_version_digest(manifest)
        except ContentIdentityError:
            raise LegendError(
                "invalid_request", "evaluator could not be fingerprinted"
            ) from None
    else:
        raise LegendError(
            "coverage_insufficient",
            "evaluator identity could not be derived (pass evaluator or declared_manifest)",
        )
    fingerprint_ok = source.declared_fingerprint in (None, declared_digest)
    entries, helper_d, helper_ok, skipped = _asset_category(
        root, manifest["helper_digests"], None, source.max_file_bytes, by_name=True
    )
    frame = {k: v for k, v in manifest.items() if k != "helper_digests"}
    try:
        recomputed: str | None = compute_evaluator_version_digest(
            {**frame, "helper_digests": helper_d}
        )
    except ContentIdentityError:
        recomputed = None
    code_ok = source.evaluator is not None and helper_ok
    reproduced = (
        recomputed == declared_digest and fingerprint_ok and helper_ok and code_ok
    )
    covered = ["judge_config", "objectives"]
    if helper_ok:
        covered.append("helper_modules")
    if code_ok:
        covered.append("evaluator_code")
    return Pillar(
        c.KIND_EVALUATOR,
        c.SCHEME_BY_KIND[c.KIND_EVALUATOR],
        None,
        source.declared_fingerprint or declared_digest,
        c.IDENTITY_SDK_FINGERPRINTED,
        (
            Representation(c.REPR_EVALUATOR_BUNDLE, _pack_or_empty(entries)),
            Representation(c.REPR_JUDGE_CONFIG, canonical_bytes(frame)),
        ),
        tuple(covered),
        reproduced=reproduced,
        skipped=skipped,
    )


def export(source: Any) -> tuple[Pillar, dict[str, Any]]:
    """Export real content for ``source``: the pillar plus the non-content re-derivation inputs.

    Refuses sources that carry only digests (nothing recoverable).
    """
    if isinstance(source, AgentStateSource):
        exp = export_agent_state(source)
        frame = {k: v for k, v in exp.manifest.items() if k not in _NON_FILE_KEYS}
        return exp.pillar, {"frame": frame}
    if isinstance(source, EvaluatorFilesSource):
        return export_evaluator_files(source), {}
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
            config = _rep(reps, kinds, c.REPR_CONFIG_DOCUMENT)
            helpers = _unpack_or_empty(_rep(reps, kinds, c.REPR_HELPER_MODULES))
            prompts = _unpack_or_empty(_rep(reps, kinds, c.REPR_PROMPT_FILES))
            tools = _unpack_or_empty(_rep(reps, kinds, c.REPR_TOOL_DEFINITIONS))
            if config is None or helpers is None or prompts is None or tools is None:
                return None
            lock = _rep(reps, kinds, c.REPR_DEPENDENCY_LOCK)
            manifest = _agent_manifest_from(
                inputs["frame"],
                {
                    "helper_modules": {n: _sha(d) for n, _e, d in helpers},
                    "prompts": {n: _sha(d) for n, _e, d in prompts},
                    "tool_definitions": {n: _sha(d) for n, _e, d in tools},
                },
                str(fp2.digest(json.loads(config))),
                _sha(lock) if lock is not None else None,
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
            files = _unpack_or_empty(_rep(reps, kinds, c.REPR_EVALUATOR_BUNDLE))
            doc = _rep(reps, kinds, c.REPR_JUDGE_CONFIG)
            if files is None or doc is None:
                return None
            manifest = {
                **json.loads(doc),
                "helper_digests": {n: _sha(d) for n, _e, d in files},
            }
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
