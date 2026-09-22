"""The agent build manifest (``AgentBuildManifestV1``) and its ``build_digest``.

Spec: TraigentSchema ``docs/identity/content-identity-v1.md`` section 8 and
``schemas/agents/agent_version_manifest_v1_schema.json``. The agent VERSION is
the digest of a manifest committing to the whole behaviour-affecting surface:

``runtime``
    Always: ``{"language": "python", "language_version", "sdk_version"}``.
``code_revision`` / ``source_digest``
    The git commit (plus ``dirty``) of the repository holding the agent's
    source, and the fp2 ``afp2`` digest of the callable plus bound state.
    **A dirty tree needs a source digest**: when the working tree differs from
    the commit and ``afp2`` cannot be computed (unreadable source, bound state
    that is not canonically serializable), NO manifest is produced -- two
    different uncommitted trees on one commit would otherwise share a digest.
``asset_digests``
    ``helper_modules`` -- the SDK ENUMERATES these itself: every module loaded
    in this process whose source file lives inside the agent's project
    (third-party packages under site-/dist-packages and virtualenvs, and the
    Traigent SDK itself, are excluded; ``dependency_lock_digest`` and
    ``runtime`` cover those). ``prompts`` and ``tool_definitions`` cannot be
    discovered in general, so they come only from an explicit declaration
    (:func:`declare_agent_assets`).
``applied_config_digest``
    fp2 digest of the trial's configuration (``{}`` for none).
``coverage``
    ``complete`` ONLY when the SDK actually covered every category: the code is
    pinned (clean commit, or a source digest), helper enumeration found every
    project file and could name and read it, no dirty file lies outside the
    enumerated helpers, and prompts AND tool definitions were declared
    (``{}`` is a valid declaration of "none"). Anything else is ``partial``,
    with the content-free reasons kept locally in :attr:`AgentBuildBase.gaps`.
    ``complete`` remains a producer declaration (spec section 15.11): it says
    the SDK looked everywhere it knows to look, not that nothing else exists.

Every manifest field is a digest, a revision, a version string or a
project-relative file name -- never file contents, prompts or configuration
values.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import platform
import re
import subprocess  # nosec B404 - fixed argv, no shell, see _git
import sys
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traigent.identity.content_identity import (
    ContentIdentityError,
    compute_agent_build_digest,
)
from traigent.utils import fp2
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "AgentBuildBase",
    "afp2_source_digest",
    "candidate_agent_version",
    "collect_agent_build_base",
    "declare_agent_assets",
    "innermost_callable",
]

_AGENT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_ASSET_NAME_RE = re.compile(r"[A-Za-z0-9_.:/-]{1,128}")
_SHA256_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_MAX_ASSETS_PER_CATEGORY = 256
_DECLARED_ASSETS_ATTR = "__traigent_agent_assets__"
_LOCK_FILES = ("uv.lock", "poetry.lock", "Pipfile.lock", "pdm.lock")
_EXCLUDED_PATH_PARTS = frozenset(
    {"site-packages", "dist-packages", ".venv", "venv", ".tox", "node_modules"}
)
_GIT_TIMEOUT_SECONDS = 10.0
_SDK_ROOT = Path(__file__).resolve().parents[1]  # the installed ``traigent`` package


# --------------------------------------------------------------------------
# Declared assets
# --------------------------------------------------------------------------


def _asset_digest(value: Any) -> str:
    """Digest one declared asset. Content is hashed locally and never kept."""
    if isinstance(value, str) and _SHA256_DIGEST_RE.fullmatch(value):
        return value
    if isinstance(value, str):
        return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    if isinstance(value, (bytes, bytearray)):
        return "sha256:" + hashlib.sha256(bytes(value)).hexdigest()
    try:
        # Structured tool / retriever specifications: the fp2 digest of the
        # JSON value, so key order never changes the version.
        return str(fp2.digest(value))
    except fp2.Fp2UnsupportedValue as error:
        raise ContentIdentityError(
            "declared asset must be text, bytes, a sha256:<hex> digest or a "
            "JSON-serializable specification"
        ) from error


def _declared_category(assets: Mapping[str, Any] | None) -> dict[str, str] | None:
    if assets is None:
        return None
    if not isinstance(assets, Mapping):
        raise ContentIdentityError("declared assets must be a mapping of name -> asset")
    if len(assets) > _MAX_ASSETS_PER_CATEGORY:
        raise ContentIdentityError(
            f"at most {_MAX_ASSETS_PER_CATEGORY} assets per category may be declared"
        )
    digests: dict[str, str] = {}
    for name, value in assets.items():
        if not isinstance(name, str) or not _ASSET_NAME_RE.fullmatch(name):
            raise ContentIdentityError(
                "asset names must fully match [A-Za-z0-9_.:/-]{1,128}"
            )
        digests[name] = _asset_digest(value)
    return digests


def declare_agent_assets(
    func: Callable[..., Any] | None = None,
    *,
    prompts: Mapping[str, Any] | None = None,
    tool_definitions: Mapping[str, Any] | None = None,
) -> Any:
    """Declare the prompts and tool definitions an agent's behaviour depends on.

    Usable as a decorator (``@declare_agent_assets(prompts=..., ...)``) or as a
    call (``declare_agent_assets(fn, prompts=...)``). Each value is the asset's
    text or bytes, a JSON-serializable specification (tool / function-calling /
    retriever schemas), or an already computed ``sha256:<hex>`` digest. Only
    digests are kept; content never leaves this call.

    Pass ``{}`` to declare that a category is empty. A category left
    undeclared keeps the agent's manifest at ``coverage: "partial"`` -- the SDK
    cannot discover prompts or tools on its own, and never assumes "none".
    """
    declared = {
        "prompts": _declared_category(prompts),
        "tool_definitions": _declared_category(tool_definitions),
    }

    def apply(target: Callable[..., Any]) -> Callable[..., Any]:
        existing = getattr(target, _DECLARED_ASSETS_ATTR, None) or {}
        merged = dict(existing)
        merged.update({k: v for k, v in declared.items() if v is not None})
        setattr(target, _DECLARED_ASSETS_ATTR, merged)
        return target

    if func is None:
        return apply
    return apply(func)


def _declared_assets_of(func: Any) -> dict[str, dict[str, str]]:
    """Merge declarations found on ``func`` and every layer it wraps."""
    found: dict[str, dict[str, str]] = {}
    for layer in _unwrap_chain(func):
        declared = getattr(layer, _DECLARED_ASSETS_ATTR, None)
        if isinstance(declared, dict):
            for category, digests in declared.items():
                if category not in found and isinstance(digests, dict):
                    found[category] = dict(digests)
    return found


# --------------------------------------------------------------------------
# afp2 (fp2 agent fingerprint)
# --------------------------------------------------------------------------


def _unwrap_chain(func: Any) -> list[Any]:
    chain: list[Any] = []
    seen: set[int] = set()
    current = func
    while current is not None and id(current) not in seen and len(chain) < 32:
        seen.add(id(current))
        chain.append(current)
        nxt = getattr(current, "__wrapped__", None)
        if nxt is None:
            # OptimizedFunction keeps the user callable on ``.func``.
            candidate = getattr(current, "func", None)
            if callable(candidate) and not isinstance(current, functools.partial):
                nxt = candidate
        current = nxt
    return chain


def _innermost(func: Any) -> Any:
    return _unwrap_chain(func)[-1]


def innermost_callable(func: Any) -> Any:
    """The user-authored callable under any decorator / OptimizedFunction layers."""
    return _innermost(func)


def _source_without_decorators(target: Any) -> str | None:
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return None
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "async def ")):
            text = textwrap.dedent("\n".join(lines[index:]))
            return text.rstrip()
    return None  # a lambda or other def-less source: not a digestible agent


def _bound_state(
    partial: functools.partial[Any] | None, target: Any
) -> dict[str, Any] | None:
    """fp2 ``bound``; ``None`` when bound state exists but cannot be observed.

    ``partial`` contributes its arguments, ``target``'s function its closure
    cells, and a bound method's ``__self__`` its instance attributes.
    """
    bound: dict[str, Any] = {}
    if partial is not None:
        if partial.args:
            bound["partial_args"] = list(partial.args)
        if partial.keywords:
            bound["partial_kwargs"] = dict(partial.keywords)
    function = getattr(target, "__func__", target)
    closure = getattr(function, "__closure__", None)
    if closure:
        code = getattr(function, "__code__", None)
        if code is None:
            return None
        cells: dict[str, Any] = {}
        for name, cell in zip(code.co_freevars, closure, strict=False):
            try:
                cells[name] = cell.cell_contents
            except ValueError:  # an empty cell: state we cannot observe
                return None
        bound["closure"] = cells
    instance = getattr(target, "__self__", None)
    if instance is not None and not inspect.ismodule(instance):
        attributes = getattr(instance, "__dict__", None)
        if attributes is None:
            return None
        if attributes:
            bound["instance"] = dict(attributes)
    return bound


def afp2_source_digest(func: Callable[..., Any]) -> str | None:
    """The fp2 ``afp2`` digest of ``func``'s callable plus bound state.

    ``None`` means *unknown* (source unreadable, not a ``def``, or bound state
    that cannot be observed or canonically serialized) -- never a digest that
    silently skipped something.
    """
    partial = func if isinstance(func, functools.partial) else None
    target = _innermost(partial.func if partial is not None else func)
    source = _source_without_decorators(getattr(target, "__func__", target))
    if source is None:
        return None
    bound = _bound_state(partial, target)
    if bound is None:
        return None
    manifest: dict[str, Any] = {"kind": "afp2", "runtime": "python", "source": source}
    if bound:
        manifest["bound"] = bound
    try:
        return str(fp2.digest(manifest))
    except fp2.Fp2UnsupportedValue:
        return None


# --------------------------------------------------------------------------
# Code revision and helper enumeration
# --------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_state(start: Path) -> tuple[Path, str, set[str]] | None:
    """``(toplevel, commit, dirty relative paths)`` or ``None`` outside git."""
    toplevel = _git(start, "rev-parse", "--show-toplevel")
    commit = _git(start, "rev-parse", "HEAD")
    status = _git(start, "status", "--porcelain=v1", "-z", "--untracked-files=normal")
    if toplevel is None or commit is None or status is None:
        return None
    commit = commit.strip()
    if not _COMMIT_RE.fullmatch(commit):
        return None
    dirty: set[str] = set()
    entries = status.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        paths = [path]
        if code[0] in "RC":  # rename/copy: the next field is the source path
            if index < len(entries) and entries[index]:
                paths.append(entries[index])
            index += 1
        dirty.update(p for p in paths if not _is_bytecode_cache(p))
    return Path(toplevel.strip()).resolve(), commit, dirty


def _is_bytecode_cache(relative_path: str) -> bool:
    """Interpreter bytecode caches are derived from sources, never behaviour inputs.

    Importing the agent writes ``__pycache__/`` next to its modules; in a
    repository that does not ignore it, counting it as a change would mark
    every clean checkout dirty the moment the agent runs.
    """
    path = relative_path.rstrip("/")
    return "__pycache__" in path.split("/") or path.endswith((".pyc", ".pyo"))


def _is_excluded(path: Path) -> bool:
    if any(part in _EXCLUDED_PATH_PARTS for part in path.parts):
        return True
    try:
        path.relative_to(_SDK_ROOT)
        return True
    except ValueError:
        return False


def _enumerate_helper_modules(
    project_root: Path,
) -> tuple[dict[str, str], set[str], list[str]]:
    """``(name -> digest, enumerated relative paths, gaps)`` for project modules."""
    files: dict[str, Path] = {}
    gaps: list[str] = []
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str) or not filename.endswith(".py"):
            continue
        try:
            path = Path(filename).resolve()
            relative = path.relative_to(project_root)
        except (OSError, ValueError):
            continue
        if _is_excluded(path):
            continue
        files[relative.as_posix()] = path
    digests: dict[str, str] = {}
    for name in sorted(files):
        if len(digests) >= _MAX_ASSETS_PER_CATEGORY:
            gaps.append("helper_modules_over_limit")
            break
        if not _ASSET_NAME_RE.fullmatch(name):
            gaps.append("helper_module_name_not_representable")
            continue
        try:
            data = files[name].read_bytes()
        except OSError:
            gaps.append("helper_module_unreadable")
            continue
        digests[name] = "sha256:" + hashlib.sha256(data).hexdigest()
    return digests, set(files), gaps


def _dependency_lock_digest(project_root: Path) -> str | None:
    for name in _LOCK_FILES:
        candidate = project_root / name
        try:
            if candidate.is_file():
                return "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            return None
    return None


def _sdk_version() -> str:
    try:
        from traigent._version import get_version

        return str(get_version())[:64] or "unknown"
    except Exception:  # noqa: BLE001 - the version string must never break a run
        return "unknown"


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentBuildBase:
    """Every manifest field except ``applied_config_digest`` (which is per trial).

    ``gaps`` lists content-free reasons for ``coverage == "partial"``. It is
    kept locally and is not part of the manifest or its digest.
    """

    agent_id: str
    runtime: dict[str, str]
    asset_digests: dict[str, dict[str, str]]
    coverage: str
    code_revision: dict[str, Any] | None = None
    source_digest: str | None = None
    dependency_lock_digest: str | None = None
    gaps: tuple[str, ...] = field(default=())

    def manifest(self, applied_config_digest: str) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "agent_id": self.agent_id,
            "runtime": dict(self.runtime),
            "asset_digests": {k: dict(v) for k, v in self.asset_digests.items()},
            "applied_config_digest": applied_config_digest,
            "coverage": self.coverage,
        }
        if self.code_revision is not None:
            manifest["code_revision"] = dict(self.code_revision)
        if self.source_digest is not None:
            manifest["source_digest"] = self.source_digest
        if self.dependency_lock_digest is not None:
            manifest["dependency_lock_digest"] = self.dependency_lock_digest
        return manifest


def collect_agent_build_base(
    func: Callable[..., Any], *, agent_id: str | None
) -> AgentBuildBase | None:
    """Collect the run-constant part of ``func``'s build manifest.

    Returns ``None`` -- no build version is claimed -- when there is no
    representable ``agent_id`` (the declared ``agent_key``; the Backend's
    project-owned agent id replaces it in milestone M3), or when neither a
    commit nor a source digest pins the code, or when the tree is dirty and no
    source digest exists.
    """
    if not isinstance(agent_id, str) or not _AGENT_ID_RE.fullmatch(agent_id):
        logger.debug("Agent build manifest skipped: no representable agent_id")
        return None
    gaps: list[str] = []
    runtime = {
        "language": "python",
        "language_version": platform.python_version()[:64],
        "sdk_version": _sdk_version(),
    }
    target = _innermost(func)
    source_file = None
    try:
        source_file = inspect.getsourcefile(getattr(target, "__func__", target))
    except TypeError:
        source_file = None
    source_digest = afp2_source_digest(func)
    if source_digest is None:
        gaps.append("source_digest_unknown")

    code_revision: dict[str, Any] | None = None
    dirty_paths: set[str] = set()
    project_root: Path | None = None
    if source_file:
        entry = Path(source_file).resolve()
        state = _git_state(entry.parent)
        if state is not None:
            project_root, commit, dirty_paths = state
            code_revision = {"vcs": "git", "commit": commit, "dirty": bool(dirty_paths)}
        else:
            project_root = entry.parent
            gaps.append("no_code_revision")
    else:
        gaps.append("entry_source_file_unknown")

    if code_revision is None and source_digest is None:
        logger.debug(
            "Agent build manifest skipped: code is neither committed nor digestible"
        )
        return None
    if code_revision is not None and code_revision["dirty"] and source_digest is None:
        logger.warning(
            "Agent build version withheld: the working tree has uncommitted "
            "changes and the agent's source digest could not be computed, so "
            "the commit alone does not identify what ran."
        )
        return None

    helper_modules: dict[str, str] = {}
    if project_root is not None:
        helper_modules, enumerated, helper_gaps = _enumerate_helper_modules(
            project_root
        )
        gaps.extend(helper_gaps)
        if dirty_paths - enumerated:
            gaps.append("dirty_files_outside_manifest")
    else:
        gaps.append("helper_modules_not_enumerated")

    declared = _declared_assets_of(func)
    for category in ("prompts", "tool_definitions"):
        if category not in declared:
            gaps.append(f"{category}_not_declared")

    dependency_lock = (
        _dependency_lock_digest(project_root) if project_root is not None else None
    )
    return AgentBuildBase(
        agent_id=agent_id,
        runtime=runtime,
        asset_digests={
            "helper_modules": helper_modules,
            "prompts": declared.get("prompts", {}),
            "tool_definitions": declared.get("tool_definitions", {}),
        },
        coverage="partial" if gaps else "complete",
        code_revision=code_revision,
        source_digest=source_digest,
        dependency_lock_digest=dependency_lock,
        gaps=tuple(sorted(set(gaps))),
    )


def candidate_agent_version(
    base: AgentBuildBase | None, config: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """``AgentVersionV1`` (``{agent_id, build_digest, manifest}``) for one configuration.

    ``None`` when there is no base or the configuration is not canonically
    serializable (its digest would otherwise have to skip something).
    """
    if base is None:
        return None
    try:
        applied = str(fp2.digest(dict(config or {})))
        manifest = base.manifest(applied)
        build_digest = compute_agent_build_digest(manifest)
    except (fp2.Fp2UnsupportedValue, ContentIdentityError, TypeError, ValueError):
        return None
    return {
        "agent_id": base.agent_id,
        "build_digest": build_digest,
        "manifest": manifest,
    }
