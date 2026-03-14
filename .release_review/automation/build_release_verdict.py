#!/usr/bin/env python3
"""Build canonical release verdict from checks, evidence, and waivers.

Gate policy (v3):
- strict mode enforces the full peer-review matrix and is the only release-ready mode.
- quick mode enforces a single-lane reduced-angle review for faster incremental sweeps.
- reusable prior per-file artifacts may satisfy a requirement when the reviewed file is
  unchanged since that artifact's commit and the run policy allows reuse.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

DEFAULT_REQUIRED_CHECKS = [
    "lint-type",
    "tests-unit",
    "tests-integration",
    "security",
    "dependency-review",
    "codeql",
    "release-review-consistency",
]

REQUIRED_REVIEW_ROLES = ("primary", "secondary", "tertiary", "reconciliation")
DEFAULT_FILE_REVIEW_AGENT_MATRIX: dict[str, tuple[str, ...]] = {
    "primary": ("codex_cli",),
    "secondary": ("claude_cli",),
    "tertiary": ("codex_cli", "copilot_cli"),
    "reconciliation": ("codex_cli",),
}
ALL_FILE_REVIEW_ANGLES = (
    "security_authz",
    "correctness_regression",
    "async_concurrency_performance",
    "dto_api_contract",
)
DEFAULT_QUICK_FILE_REVIEW_AGENT_MATRIX: dict[str, tuple[str, ...]] = {
    "primary": ("codex_cli", "claude_cli", "copilot_cli"),
}
DEFAULT_REVIEW_MODES: dict[str, dict[str, Any]] = {
    "strict": {
        "required_review_types": list(REQUIRED_REVIEW_ROLES),
        "required_angles": list(ALL_FILE_REVIEW_ANGLES),
        "require_component_evidence": True,
        "allow_previous_artifact_reuse": True,
        "skip_if_file_unchanged_and_artifact_exists": True,
        "single_model": False,
        "file_review_agent_matrix": {
            role: list(values) for role, values in DEFAULT_FILE_REVIEW_AGENT_MATRIX.items()
        },
    },
    "quick": {
        "required_review_types": ["primary"],
        "required_angles": [
            "security_authz",
            "correctness_regression",
            "async_concurrency_performance",
        ],
        "require_component_evidence": False,
        "allow_previous_artifact_reuse": True,
        "skip_if_file_unchanged_and_artifact_exists": True,
        "single_model": True,
        "file_review_agent_matrix": {
            role: list(values)
            for role, values in DEFAULT_QUICK_FILE_REVIEW_AGENT_MATRIX.items()
        },
    },
}


@dataclass(frozen=True)
class RequiredComponent:
    name: str
    priority: str
    requires_family_diversity: bool
    scope_globs: tuple[str, ...]


REQUIRED_COMPONENTS: list[RequiredComponent] = [
    RequiredComponent(
        name="Public API + Safety",
        priority="P0",
        requires_family_diversity=True,
        scope_globs=(
            "traigent/api/**/*.py",
            "traigent/security/**/*.py",
            "tests/**/api/**/*.py",
            "tests/**/security/**/*.py",
            "tests/**/*privacy*.py",
        ),
    ),
    RequiredComponent(
        name="Core Orchestration + Config",
        priority="P0",
        requires_family_diversity=True,
        scope_globs=(
            "traigent/core/**/*.py",
            "traigent/config/**/*.py",
            "traigent/config_generator/**/*.py",
            "traigent/cloud/**/*.py",
            "traigent/cli/**/*.py",
            "traigent/storage/**/*.py",
            "traigent/providers/**/*.py",
            "traigent/utils/**/*.py",
            "traigent/tuned_variables/**/*.py",
            "traigent/tvl/**/*.py",
            "traigent/analytics/**/*.py",
            "traigent/agents/**/*.py",
            "traigent/traigent_client.py",
            "traigent/__init__.py",
            "traigent/_version.py",
            "tests/**/core/**/*.py",
            "tests/**/config/**/*.py",
            "tests/**/config_generator/**/*.py",
            "tests/**/cloud/**/*.py",
            "tests/**/storage/**/*.py",
            "tests/**/tuned_variables/**/*.py",
            "tests/**/tvl/**/*.py",
            "tests/**/utils/**/*.py",
            "tests/test_backend_config.py",
            "tests/unit/test_traigent_client*.py",
        ),
    ),
    RequiredComponent(
        name="Integrations + Invokers",
        priority="P1",
        requires_family_diversity=True,
        scope_globs=(
            "traigent/integrations/**/*.py",
            "traigent/invokers/**/*.py",
            "traigent/hybrid/**/*.py",
            "traigent/wrapper/**/*.py",
            "traigent/hooks/**/*.py",
            "tests/**/integrations/**/*.py",
            "tests/**/invokers/**/*.py",
            "tests/**/hybrid/**/*.py",
            "tests/**/wrapper/**/*.py",
            "tests/**/hooks/**/*.py",
            "tests/unit/test_bridge_wrapper.py",
            "tests/unit/test_hybrid_protocol_privacy.py",
        ),
    ),
    RequiredComponent(
        name="Optimizers + Evaluators",
        priority="P1",
        requires_family_diversity=True,
        scope_globs=(
            "traigent/optimizers/**/*.py",
            "traigent/evaluators/**/*.py",
            "traigent/core/samplers/**/*.py",
            "traigent/core/stat_significance.py",
            "traigent/core/result_selection.py",
            "tests/**/optimizers/**/*.py",
            "tests/**/evaluators/**/*.py",
            "tests/**/core/samplers/**/*.py",
            "tests/**/core/test_stat_significance.py",
        ),
    ),
    RequiredComponent(
        name="Packaging + CI",
        priority="P1",
        requires_family_diversity=True,
        scope_globs=(
            ".github/workflows/*.yml",
            "pyproject.toml",
            "requirements/**/*.txt",
            "Makefile",
            "uv.lock",
            "pytest.ini",
            "mypy.ini",
            ".pre-commit-config.yaml",
            "MANIFEST.in",
            "scripts/**/*.py",
            "scripts/**/*.sh",
        ),
    ),
    RequiredComponent(
        name="Docs + Release Ops",
        priority="P2",
        requires_family_diversity=False,
        scope_globs=(
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "MAINTENANCE.md",
            "docs/**/*.md",
            "docs/**/*.yaml",
            "docs/**/*.yml",
            "docs/**/*.json",
            ".release_review/**/*",
            "tests/**/*.py",
            "traigent_validation/**/*.py",
            "examples/**/*",
            "walkthrough/**/*",
            ".archive/**/*",
            ".claude/**/*",
            "assets/**/*",
        ),
    ),
]

_COMPONENT_ALIAS_MAP = {
    "public api safety": "Public API + Safety",
    "public api + safety": "Public API + Safety",
    "public_api_safety": "Public API + Safety",
    "core orchestration config": "Core Orchestration + Config",
    "core orchestration + config": "Core Orchestration + Config",
    "core_orchestration_config": "Core Orchestration + Config",
    "integrations invokers": "Integrations + Invokers",
    "integrations + invokers": "Integrations + Invokers",
    "integrations_invokers": "Integrations + Invokers",
    "optimizers evaluators": "Optimizers + Evaluators",
    "optimizers + evaluators": "Optimizers + Evaluators",
    "optimizers_evaluators": "Optimizers + Evaluators",
    "packaging ci": "Packaging + CI",
    "packaging + ci": "Packaging + CI",
    "packaging_ci": "Packaging + CI",
    "docs release ops": "Docs + Release Ops",
    "docs + release ops": "Docs + Release Ops",
    "docs_release_ops": "Docs + Release Ops",
}

_CANONICAL_COMPONENTS = {c.name: c for c in REQUIRED_COMPONENTS}
_COMPONENT_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def resolve_within(base_dir: Path, candidate: Path, label: str) -> Path:
    """Resolve candidate path and ensure it stays under base_dir."""
    base_resolved = base_dir.resolve()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_resolved / candidate).resolve()

    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} must remain under {base_resolved}") from exc

    return resolved


def get_git_sha_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def parse_iso_utc(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


@dataclass
class Finding:
    finding_id: str
    severity: str
    file: str
    line: int
    title: str
    status: str


@dataclass
class Waiver:
    waiver_id: str
    finding_id: str
    severity: str
    expires_at: str
    approved_by: list[str]
    reason: str
    valid: bool


@dataclass
class EvidenceEntry:
    schema_version: int | None
    component: str
    review_type: str
    agent_type: str
    reviewer_model: str
    reviewer_family: str
    decision: str
    commit_sha: str
    timestamp_utc: str
    evidence_file: str
    files_reviewed: tuple[str, ...]
    review_summary_length: int
    strengths_count: int
    checks_count: int


@dataclass
class FileReviewArtifact:
    schema_version: int | None
    component: str
    review_type: str
    agent_type: str
    reviewer_model: str
    decision: str
    commit_sha: str
    timestamp_utc: str
    file_path: str
    artifact_file: str
    angles_reviewed: frozenset[str]
    source_run_id: str
    notes_length: int
    findings_count: int
    strengths_count: int
    checks_count: int
    is_reused: bool = False


@dataclass(frozen=True)
class ReviewModeConfig:
    name: str
    required_review_types: tuple[str, ...]
    required_angles: tuple[str, ...]
    require_component_evidence: bool
    allow_previous_artifact_reuse: bool
    skip_if_file_unchanged_and_artifact_exists: bool
    single_model: bool
    file_review_agent_matrix: dict[str, set[str]]


def normalize_check_results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("checks"), list):
        raw = raw["checks"]
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        status = str(item.get("status") or "unknown").strip().lower()
        normalized.append(
            {
                "key": key,
                "status": status,
                "required": bool(item.get("required", False)),
            }
        )
    return normalized


def is_check_pass(status: str) -> bool:
    return status in {"pass", "success"}


def normalize_component_label(value: str) -> str:
    value = _COMPONENT_TOKEN_RE.sub(" ", value.strip().lower())
    return " ".join(value.split())


def canonical_component_name(value: str) -> str | None:
    normalized = normalize_component_label(value)
    if not normalized:
        return None
    if normalized in _COMPONENT_ALIAS_MAP:
        return _COMPONENT_ALIAS_MAP[normalized]
    for component_name in _CANONICAL_COMPONENTS:
        if normalize_component_label(component_name) == normalized:
            return component_name
    return None


def normalize_repo_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def normalize_agent_type(value: str) -> str:
    normalized = _COMPONENT_TOKEN_RE.sub("_", value.strip().lower()).strip("_")
    if not normalized:
        return ""
    if any(token in normalized for token in ("codex", "gpt", "openai")):
        return "codex_cli"
    if any(token in normalized for token in ("claude", "anthropic", "opus", "sonnet", "haiku")):
        return "claude_cli"
    if any(token in normalized for token in ("copilot", "gemini", "github")):
        return "copilot_cli"
    return normalized


def glob_matches(path: str, pattern: str) -> bool:
    normalized_path = normalize_repo_path(path)
    normalized_pattern = normalize_repo_path(pattern)
    path_obj = PurePosixPath(normalized_path)
    if path_obj.match(normalized_pattern):
        return True
    if fnmatch.fnmatch(normalized_path, normalized_pattern):
        return True
    if "/**/" in normalized_pattern:
        compact_pattern = normalized_pattern.replace("/**/", "/")
        if path_obj.match(compact_pattern):
            return True
        if fnmatch.fnmatch(normalized_path, compact_pattern):
            return True
    return False


def path_matches_globs(path: str, globs: tuple[str, ...]) -> bool:
    normalized = normalize_repo_path(path)
    return any(glob_matches(normalized, pattern) for pattern in globs)


def classify_model_family(model: str) -> str:
    lowered = model.strip().lower()
    if any(token in lowered for token in ("codex", "gpt", "openai")):
        return "openai"
    if any(
        token in lowered for token in ("claude", "anthropic", "opus", "sonnet", "haiku")
    ):
        return "anthropic"
    if any(token in lowered for token in ("gemini", "google")):
        return "google"
    if "copilot" in lowered:
        return "github"
    return lowered.split()[0] if lowered else "unknown"


def sha_matches(reference_sha: str, commit_sha: str) -> bool:
    ref = reference_sha.strip().lower()
    commit = commit_sha.strip().lower()
    if not ref or not commit:
        return False
    return ref.startswith(commit) or commit.startswith(ref)


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = [normalize_repo_path(line) for line in path.read_text().splitlines()]
    return [line for line in lines if line]


def load_scope_config(scope_path: Path) -> dict[str, Any]:
    if not scope_path.exists():
        return {}
    try:
        data = yaml.safe_load(scope_path.read_text())
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    data = load_json(run_dir / "run_manifest.json", {})
    return data if isinstance(data, dict) else {}


def _normalize_review_types(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    normalized = []
    for item in raw:
        value = str(item).strip().lower()
        if value in REQUIRED_REVIEW_ROLES and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _normalize_angles(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    normalized = []
    for item in raw:
        value = str(item).strip()
        if value in ALL_FILE_REVIEW_ANGLES and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _resolve_mode_agent_matrix(raw: Any) -> dict[str, set[str]]:
    matrix: dict[str, set[str]] = {}
    if not isinstance(raw, dict):
        return matrix
    for role, values in raw.items():
        role_name = str(role).strip().lower()
        if role_name not in REQUIRED_REVIEW_ROLES:
            continue
        if isinstance(values, str):
            normalized = normalize_agent_type(values)
            if normalized:
                matrix[role_name] = {normalized}
            continue
        if not isinstance(values, list):
            continue
        normalized_values = {
            normalize_agent_type(str(item))
            for item in values
            if str(item).strip()
        }
        normalized_values.discard("")
        if normalized_values:
            matrix[role_name] = normalized_values
    return matrix


def resolve_review_mode(run_dir: Path) -> ReviewModeConfig:
    scope_config = load_scope_config(Path(".release_review/scope.yml"))
    manifest = load_run_manifest(run_dir)
    configured_modes = scope_config.get("review_modes")
    if not isinstance(configured_modes, dict):
        configured_modes = {}

    default_mode = str(scope_config.get("default_review_mode") or "strict").strip()
    requested_mode = str(manifest.get("review_mode") or default_mode or "strict").strip()
    if requested_mode not in DEFAULT_REVIEW_MODES:
        requested_mode = "strict"

    merged: dict[str, Any] = dict(DEFAULT_REVIEW_MODES[requested_mode])
    raw_mode = configured_modes.get(requested_mode)
    if isinstance(raw_mode, dict):
        merged.update(raw_mode)
        if "file_review_agent_matrix" in raw_mode:
            merged["file_review_agent_matrix"] = raw_mode["file_review_agent_matrix"]

    required_review_types = _normalize_review_types(
        merged.get("required_review_types")
    ) or tuple(DEFAULT_REVIEW_MODES[requested_mode]["required_review_types"])
    required_angles = _normalize_angles(
        merged.get("required_angles")
    ) or tuple(DEFAULT_REVIEW_MODES[requested_mode]["required_angles"])
    file_review_agent_matrix = _resolve_mode_agent_matrix(
        merged.get("file_review_agent_matrix")
    )
    if not file_review_agent_matrix:
        file_review_agent_matrix = _resolve_mode_agent_matrix(
            DEFAULT_REVIEW_MODES[requested_mode]["file_review_agent_matrix"]
        )

    return ReviewModeConfig(
        name=requested_mode,
        required_review_types=required_review_types,
        required_angles=required_angles,
        require_component_evidence=bool(merged.get("require_component_evidence", True)),
        allow_previous_artifact_reuse=bool(
            merged.get("allow_previous_artifact_reuse", True)
        ),
        skip_if_file_unchanged_and_artifact_exists=bool(
            merged.get("skip_if_file_unchanged_and_artifact_exists", True)
        ),
        single_model=bool(merged.get("single_model", False)),
        file_review_agent_matrix=file_review_agent_matrix,
    )


def git_file_unchanged_since(commit_sha: str, baseline_sha: str, file_path: str) -> bool:
    if sha_matches(baseline_sha, commit_sha):
        return True
    if not commit_sha.strip() or not baseline_sha.strip():
        return False
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", commit_sha, baseline_sha, "--", file_path],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def iter_previous_run_dirs(run_dir: Path) -> list[Path]:
    runs_root = run_dir.parent
    manifests: list[tuple[tuple[int, str], Path]] = []
    for manifest_file in runs_root.glob("*/run_manifest.json"):
        candidate_run_dir = manifest_file.parent
        if candidate_run_dir.resolve() == run_dir.resolve():
            continue
        manifest = load_json(manifest_file, {})
        if not isinstance(manifest, dict):
            continue
        ts = str(
            manifest.get("generated_at_utc")
            or manifest.get("created_at_utc")
            or ""
        ).strip()
        parsed = parse_iso_utc(ts) if ts else None
        sort_key = (1, parsed.isoformat()) if parsed is not None else (0, ts)
        manifests.append((sort_key, candidate_run_dir))
    manifests.sort(reverse=True)
    return [item[1] for item in manifests]


def file_in_scope(
    file_path: str,
    include: list[str],
    exclude: list[str],
    shared_files: list[str],
) -> bool:
    normalized = normalize_repo_path(file_path)
    if any(glob_matches(normalized, pattern) for pattern in exclude):
        return False
    if normalized in shared_files:
        return True
    if include:
        return any(glob_matches(normalized, pattern) for pattern in include)
    return True


def git_changed_files(base_branch: str) -> list[str]:
    try:
        merge_base_proc = subprocess.run(
            ["git", "merge-base", base_branch, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        merge_base = merge_base_proc.stdout.strip()
        if not merge_base:
            return []
        diff_proc = subprocess.run(
            ["git", "diff", "--name-only", f"{merge_base}..HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    files = [normalize_repo_path(line) for line in diff_proc.stdout.splitlines()]
    return [line for line in files if line]


def resolve_scope_files(run_dir: Path) -> list[str]:
    review_scope_file = run_dir / "inventories" / "review_scope_files.txt"
    if review_scope_file.exists():
        return sorted(set(read_lines(review_scope_file)))

    manifest = load_json(run_dir / "run_manifest.json", {})
    base_branch = "main"
    if isinstance(manifest, dict):
        candidate = str(manifest.get("base_branch") or "").strip()
        if candidate:
            base_branch = candidate

    scope_config = load_scope_config(Path(".release_review/scope.yml"))
    include = [str(x) for x in scope_config.get("include", [])]
    exclude = [str(x) for x in scope_config.get("exclude", [])]
    shared_files = [
        normalize_repo_path(str(x)) for x in scope_config.get("shared_files", [])
    ]

    changed_files = git_changed_files(base_branch)
    scoped = [
        file_path
        for file_path in changed_files
        if file_in_scope(file_path, include, exclude, shared_files)
    ]
    return sorted(set(scoped))


def resolve_component_scope_globs() -> dict[str, tuple[str, ...]]:
    defaults = {
        component.name: component.scope_globs for component in REQUIRED_COMPONENTS
    }
    scope_config = load_scope_config(Path(".release_review/scope.yml"))
    raw_map = scope_config.get("component_map")
    if not isinstance(raw_map, dict):
        return defaults

    resolved = defaults.copy()
    for raw_name, patterns in raw_map.items():
        canonical = canonical_component_name(str(raw_name))
        if canonical is None:
            continue
        if not isinstance(patterns, list):
            continue
        resolved[canonical] = tuple(str(item) for item in patterns if str(item).strip())
    return resolved


def load_file_review_agent_matrix(mode: ReviewModeConfig) -> dict[str, set[str]]:
    return {role: set(values) for role, values in mode.file_review_agent_matrix.items()}


def is_file_review_matrix_enforced() -> bool:
    scope_config = load_scope_config(Path(".release_review/scope.yml"))
    raw = scope_config.get("enforce_per_file_artifact_matrix")
    if isinstance(raw, bool):
        return raw
    return True


def map_scope_files_to_components(
    scope_files: list[str],
) -> tuple[dict[str, set[str]], list[str]]:
    globs_by_component = resolve_component_scope_globs()
    required_by_component: dict[str, set[str]] = {
        component.name: set() for component in REQUIRED_COMPONENTS
    }
    unmatched: list[str] = []

    for file_path in scope_files:
        mapped = False
        for component in REQUIRED_COMPONENTS:
            globs = globs_by_component.get(component.name, ())
            if path_matches_globs(file_path, globs):
                required_by_component[component.name].add(
                    normalize_repo_path(file_path)
                )
                mapped = True
                break
        if not mapped:
            unmatched.append(normalize_repo_path(file_path))

    return required_by_component, sorted(set(unmatched))


def collect_evidence_findings(evidence_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    if not evidence_root.exists():
        return findings

    for file in sorted(evidence_root.rglob("*.json")):
        data = load_json(file, {})
        if not isinstance(data, dict):
            continue
        raw_findings = data.get("findings", [])
        if not isinstance(raw_findings, list):
            continue
        for item in raw_findings:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity") or "").upper()
            finding_id = str(item.get("id") or "").strip()
            if not finding_id or severity not in {"P0", "P1", "P2", "P3"}:
                continue
            findings.append(
                Finding(
                    finding_id=finding_id,
                    severity=severity,
                    file=str(item.get("file") or ""),
                    line=int(item.get("line") or 0),
                    title=str(item.get("title") or ""),
                    status=str(item.get("status") or "open").lower(),
                )
            )
    return findings


def collect_peer_review_evidence(evidence_root: Path) -> list[EvidenceEntry]:
    entries: list[EvidenceEntry] = []
    if not evidence_root.exists():
        return entries

    for file in sorted(evidence_root.rglob("*.json")):
        data = load_json(file, {})
        if not isinstance(data, dict):
            continue

        component = canonical_component_name(str(data.get("component") or ""))
        review_type = str(data.get("review_type") or "").strip().lower()
        agent_type = normalize_agent_type(str(data.get("agent_type") or ""))
        reviewer_model = str(data.get("reviewer_model") or "").strip()
        decision = str(data.get("decision") or "").strip().lower()
        commit_sha = str(data.get("commit_sha") or "").strip()
        timestamp_utc = str(data.get("timestamp_utc") or "").strip()
        schema_version_raw = data.get("schema_version")
        schema_version = (
            schema_version_raw if isinstance(schema_version_raw, int) else None
        )
        review_summary = str(data.get("review_summary") or "").strip()
        raw_files_reviewed = data.get("files_reviewed", [])
        files_reviewed: list[str] = []
        if isinstance(raw_files_reviewed, list):
            files_reviewed = [
                normalize_repo_path(str(item))
                for item in raw_files_reviewed
                if str(item).strip()
            ]
        raw_strengths = data.get("strengths", [])
        strengths_count = len(raw_strengths) if isinstance(raw_strengths, list) else 0
        raw_checks = data.get("checks_performed", [])
        checks_count = len(raw_checks) if isinstance(raw_checks, list) else 0

        if component is None:
            continue
        if review_type not in {
            "primary",
            "secondary",
            "tertiary",
            "reconciliation",
            "captain",
        }:
            continue
        if not reviewer_model:
            continue
        if decision not in {"approved", "changes_required", "blocked"}:
            continue
        if not commit_sha:
            continue

        entries.append(
            EvidenceEntry(
                schema_version=schema_version,
                component=component,
                review_type=review_type,
                agent_type=agent_type,
                reviewer_model=reviewer_model,
                reviewer_family=classify_model_family(reviewer_model),
                decision=decision,
                commit_sha=commit_sha,
                timestamp_utc=timestamp_utc,
                evidence_file=str(file),
                files_reviewed=tuple(sorted(set(files_reviewed))),
                review_summary_length=len(review_summary),
                strengths_count=strengths_count,
                checks_count=checks_count,
            )
        )

    return entries


def latest_entry(entries: list[EvidenceEntry]) -> EvidenceEntry:
    def sort_key(entry: EvidenceEntry) -> tuple[int, str]:
        ts = parse_iso_utc(entry.timestamp_utc)
        if ts is None:
            return (0, entry.timestamp_utc)
        return (1, ts.isoformat())

    return sorted(entries, key=sort_key)[-1]


def latest_file_review_artifact(
    artifacts: list[FileReviewArtifact],
) -> FileReviewArtifact:
    def sort_key(entry: FileReviewArtifact) -> tuple[int, str]:
        ts = parse_iso_utc(entry.timestamp_utc)
        if ts is None:
            return (0, entry.timestamp_utc)
        return (1, ts.isoformat())

    return sorted(artifacts, key=sort_key)[-1]


def collect_file_review_artifacts(
    file_review_root: Path,
    *,
    source_run_id: str,
) -> list[FileReviewArtifact]:
    artifacts: list[FileReviewArtifact] = []
    if not file_review_root.exists():
        return artifacts

    for file in sorted(file_review_root.rglob("*.json")):
        data = load_json(file, {})
        if not isinstance(data, dict):
            continue

        component = canonical_component_name(str(data.get("component") or ""))
        review_type = str(data.get("review_type") or "").strip().lower()
        agent_type = normalize_agent_type(str(data.get("agent_type") or ""))
        reviewer_model = str(data.get("reviewer_model") or "").strip()
        decision = str(data.get("decision") or "").strip().lower()
        commit_sha = str(data.get("commit_sha") or "").strip()
        timestamp_utc = str(data.get("timestamp_utc") or "").strip()
        schema_version_raw = data.get("schema_version")
        schema_version = (
            schema_version_raw if isinstance(schema_version_raw, int) else None
        )
        notes = str(data.get("notes") or "").strip()
        raw_findings = data.get("findings", [])
        findings_count = len(raw_findings) if isinstance(raw_findings, list) else 0
        raw_strengths = data.get("strengths", [])
        strengths_count = len(raw_strengths) if isinstance(raw_strengths, list) else 0
        raw_checks = data.get("checks_performed", [])
        checks_count = len(raw_checks) if isinstance(raw_checks, list) else 0
        file_path = normalize_repo_path(
            str(data.get("file") or data.get("file_path") or "")
        )
        raw_angles = data.get("angles_reviewed", [])
        angles_reviewed = frozenset(
            angle
            for angle in (
                str(item).strip() for item in raw_angles if str(item).strip()
            )
            if angle in ALL_FILE_REVIEW_ANGLES
        )

        if component is None:
            continue
        if review_type not in REQUIRED_REVIEW_ROLES:
            continue
        if not agent_type:
            continue
        if not reviewer_model:
            continue
        if decision not in {"approved", "changes_required", "blocked"}:
            continue
        if not commit_sha:
            continue
        if not file_path:
            continue

        artifacts.append(
            FileReviewArtifact(
                schema_version=schema_version,
                component=component,
                review_type=review_type,
                agent_type=agent_type,
                reviewer_model=reviewer_model,
                decision=decision,
                commit_sha=commit_sha,
                timestamp_utc=timestamp_utc,
                file_path=file_path,
                artifact_file=str(file),
                angles_reviewed=angles_reviewed,
                source_run_id=source_run_id,
                notes_length=len(notes),
                findings_count=findings_count,
                strengths_count=strengths_count,
                checks_count=checks_count,
            )
        )
    return artifacts


def collect_available_file_review_artifacts(
    run_dir: Path,
    mode: ReviewModeConfig,
) -> list[FileReviewArtifact]:
    manifest = load_run_manifest(run_dir)
    release_id = str(manifest.get("release_id") or run_dir.name).strip() or run_dir.name
    artifacts = collect_file_review_artifacts(
        run_dir / "file_reviews",
        source_run_id=release_id,
    )
    if not mode.allow_previous_artifact_reuse:
        return artifacts

    for previous_run_dir in iter_previous_run_dirs(run_dir):
        previous_manifest = load_run_manifest(previous_run_dir)
        previous_release_id = (
            str(previous_manifest.get("release_id") or previous_run_dir.name).strip()
            or previous_run_dir.name
        )
        artifacts.extend(
            collect_file_review_artifacts(
                previous_run_dir / "file_reviews",
                source_run_id=previous_release_id,
            )
        )
    return artifacts


def artifact_matches_target_commit(
    artifact: FileReviewArtifact,
    *,
    baseline_sha: str,
    file_path: str,
    mode: ReviewModeConfig,
) -> bool:
    if sha_matches(baseline_sha, artifact.commit_sha):
        return True
    if not mode.allow_previous_artifact_reuse:
        return False
    if not mode.skip_if_file_unchanged_and_artifact_exists:
        return False
    return git_file_unchanged_since(artifact.commit_sha, baseline_sha, file_path)


def evaluate_required_peer_reviews(
    entries: list[EvidenceEntry],
    file_review_artifacts: list[FileReviewArtifact],
    baseline_sha: str,
    required_files_by_component: dict[str, set[str]],
    unmatched_scope_files: list[str],
    file_review_agent_matrix: dict[str, set[str]],
    enforce_file_review_matrix: bool,
    mode: ReviewModeConfig,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[EvidenceEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.component, []).append(entry)

    artifact_index: dict[tuple[str, str, str], list[FileReviewArtifact]] = {}
    for artifact in file_review_artifacts:
        key = (
            artifact.component,
            normalize_repo_path(artifact.file_path),
            artifact.review_type,
        )
        artifact_index.setdefault(key, []).append(artifact)

    failures: list[dict[str, Any]] = []

    if unmatched_scope_files:
        preview = unmatched_scope_files[:50]
        failures.append(
            {
                "component": "__scope__",
                "priority": "P0",
                "reason": "unmapped_in_scope_files",
                "details": "One or more in-scope changed files are not mapped to any required component.",
                "files": preview,
                "remaining_count": max(0, len(unmatched_scope_files) - len(preview)),
            }
        )

    components_to_evaluate = [
        required
        for required in REQUIRED_COMPONENTS
        if mode.require_component_evidence
        or required_files_by_component.get(required.name, set())
    ]

    for required in components_to_evaluate:
        required_files = required_files_by_component.get(required.name, set())
        component_entries = grouped.get(required.name, [])
        if mode.require_component_evidence and not component_entries:
            failures.append(
                {
                    "component": required.name,
                    "priority": required.priority,
                    "reason": "missing_component_evidence",
                    "details": "No evidence JSON found for component.",
                }
            )
            continue

        latest_by_role: dict[str, EvidenceEntry] = {}
        if mode.require_component_evidence:
            for role in mode.required_review_types:
                role_entries = [
                    item for item in component_entries if item.review_type == role
                ]
                if not role_entries:
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "missing_required_role",
                            "role": role,
                            "details": f"Missing {role} peer-review evidence.",
                        }
                    )
                    continue

                chosen = latest_entry(role_entries)
                latest_by_role[role] = chosen

                if not chosen.agent_type:
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "component_evidence_missing_agent_type",
                            "role": role,
                            "evidence_file": chosen.evidence_file,
                            "details": "Component evidence must include explicit agent_type.",
                        }
                    )

                allowed_agents = sorted(file_review_agent_matrix.get(role, set()))
                if (
                    chosen.agent_type
                    and allowed_agents
                    and chosen.agent_type not in allowed_agents
                ):
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "component_evidence_agent_mismatch",
                            "role": role,
                            "agent_type": chosen.agent_type,
                            "expected_agents": allowed_agents,
                            "evidence_file": chosen.evidence_file,
                            "details": "Component evidence agent_type does not match required lane.",
                        }
                    )

                if chosen.schema_version is None or chosen.schema_version < 2:
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "component_evidence_schema_version_invalid",
                            "role": role,
                            "evidence_file": chosen.evidence_file,
                            "details": "Component evidence must include schema_version >= 2.",
                        }
                    )

                if chosen.review_summary_length < 50:
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "component_evidence_summary_missing",
                            "role": role,
                            "evidence_file": chosen.evidence_file,
                            "details": "Component evidence review_summary must be at least 50 characters.",
                        }
                    )

                if chosen.strengths_count == 0:
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "component_evidence_missing_strengths",
                            "role": role,
                            "evidence_file": chosen.evidence_file,
                            "details": "Component evidence must include at least one positive finding (strength).",
                        }
                    )

                if chosen.checks_count == 0:
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "component_evidence_missing_checks_performed",
                            "role": role,
                            "evidence_file": chosen.evidence_file,
                            "details": "Component evidence must include checks_performed entries.",
                        }
                    )

                reviewed_files = set(chosen.files_reviewed)
                missing_files = sorted(required_files - reviewed_files)
                if missing_files:
                    preview = missing_files[:50]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "missing_file_peer_review_coverage",
                            "role": role,
                            "evidence_file": chosen.evidence_file,
                            "details": (
                                f"{role} evidence does not cover all in-scope files assigned "
                                "to this component."
                            ),
                            "files": preview,
                            "remaining_count": max(0, len(missing_files) - len(preview)),
                        }
                    )

                if chosen.decision != "approved":
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "non_approved_role_decision",
                            "role": role,
                            "decision": chosen.decision,
                            "evidence_file": chosen.evidence_file,
                            "details": (
                                f"Latest {role} evidence decision is {chosen.decision}, "
                                "must be approved."
                            ),
                        }
                    )

                if not sha_matches(baseline_sha, chosen.commit_sha):
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "reviewed_commit_mismatch",
                            "role": role,
                            "reviewed_commit": chosen.commit_sha,
                            "expected_commit": baseline_sha,
                            "evidence_file": chosen.evidence_file,
                            "details": "Evidence commit_sha does not match baseline commit under review.",
                        }
                    )

            primary = latest_by_role.get("primary")
            secondary = latest_by_role.get("secondary")
            tertiary = latest_by_role.get("tertiary")
            if primary and secondary:
                if (
                    primary.reviewer_model.strip().lower()
                    == secondary.reviewer_model.strip().lower()
                ):
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "primary_secondary_same_model",
                            "primary_model": primary.reviewer_model,
                            "secondary_model": secondary.reviewer_model,
                            "details": "Primary and secondary reviewers must be different models.",
                        }
                    )

                if (
                    required.requires_family_diversity
                    and primary.reviewer_family == secondary.reviewer_family
                ):
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "primary_secondary_same_family",
                            "primary_model": primary.reviewer_model,
                            "secondary_model": secondary.reviewer_model,
                            "primary_family": primary.reviewer_family,
                            "secondary_family": secondary.reviewer_family,
                            "details": (
                                "P0/P1 components require primary and secondary reviewers from "
                                "different model families."
                            ),
                        }
                    )

            if primary and tertiary:
                if (
                    primary.reviewer_model.strip().lower()
                    == tertiary.reviewer_model.strip().lower()
                ):
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "primary_tertiary_same_model",
                            "primary_model": primary.reviewer_model,
                            "tertiary_model": tertiary.reviewer_model,
                            "details": "Primary and tertiary reviewers must be different models.",
                        }
                    )

            if secondary and tertiary:
                if (
                    secondary.reviewer_model.strip().lower()
                    == tertiary.reviewer_model.strip().lower()
                ):
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "secondary_tertiary_same_model",
                            "secondary_model": secondary.reviewer_model,
                            "tertiary_model": tertiary.reviewer_model,
                            "details": "Secondary and tertiary reviewers must be different models.",
                        }
                    )

        if enforce_file_review_matrix:
            for role in mode.required_review_types:
                allowed_agents = sorted(file_review_agent_matrix.get(role, set()))
                missing_file_artifacts: list[str] = []
                mismatched_commit_files: list[str] = []
                non_approved_file_artifacts: list[str] = []
                agent_mismatch_rows: list[dict[str, Any]] = []
                schema_invalid_files: list[str] = []
                missing_strengths_files: list[str] = []
                missing_checks_files: list[str] = []
                missing_notes_for_clean_approval_files: list[str] = []
                missing_angle_rows: list[dict[str, Any]] = []

                for required_file in sorted(required_files):
                    artifacts_for_file = artifact_index.get(
                        (required.name, normalize_repo_path(required_file), role),
                        [],
                    )
                    if not artifacts_for_file:
                        missing_file_artifacts.append(required_file)
                        continue

                    qualifying = [
                        artifact
                        for artifact in artifacts_for_file
                        if artifact_matches_target_commit(
                            artifact,
                            baseline_sha=baseline_sha,
                            file_path=required_file,
                            mode=mode,
                        )
                    ]
                    if not qualifying:
                        mismatched_commit_files.append(required_file)
                        continue

                    approved = [
                        artifact for artifact in qualifying if artifact.decision == "approved"
                    ]
                    if not approved:
                        non_approved_file_artifacts.append(required_file)
                        continue

                    allowed_approved = approved
                    if allowed_agents:
                        allowed_approved = [
                            artifact
                            for artifact in approved
                            if artifact.agent_type in allowed_agents
                        ]
                        if not allowed_approved:
                            newest = latest_file_review_artifact(approved)
                            agent_mismatch_rows.append(
                                {
                                    "file": required_file,
                                    "expected_agents": allowed_agents,
                                    "found_agents": sorted(
                                        {artifact.agent_type for artifact in approved}
                                    ),
                                    "latest_artifact": newest.artifact_file,
                                }
                            )
                            continue

                    quality_approved = [
                        artifact
                        for artifact in allowed_approved
                        if artifact.schema_version and artifact.schema_version >= 2
                    ]
                    if not quality_approved:
                        schema_invalid_files.append(required_file)
                        continue

                    quality_with_strengths = [
                        artifact
                        for artifact in quality_approved
                        if artifact.strengths_count > 0
                    ]
                    if not quality_with_strengths:
                        missing_strengths_files.append(required_file)
                        continue

                    quality_with_checks = [
                        artifact
                        for artifact in quality_with_strengths
                        if artifact.checks_count > 0
                    ]
                    if not quality_with_checks:
                        missing_checks_files.append(required_file)
                        continue

                    if not any(
                        artifact.findings_count > 0 or artifact.notes_length >= 20
                        for artifact in quality_with_checks
                    ):
                        missing_notes_for_clean_approval_files.append(required_file)

                    covered_angles = set()
                    for artifact in quality_with_checks:
                        covered_angles.update(artifact.angles_reviewed)
                    missing_angles = sorted(set(mode.required_angles) - covered_angles)
                    if missing_angles:
                        missing_angle_rows.append(
                            {
                                "file": required_file,
                                "missing_angles": missing_angles,
                                "artifacts": [
                                    artifact.artifact_file for artifact in quality_with_checks
                                ][:5],
                            }
                        )

                if missing_file_artifacts:
                    preview = missing_file_artifacts[:50]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "missing_file_review_artifact",
                            "role": role,
                            "details": (
                                f"Missing per-file review artifact(s) for {role} role "
                                "on in-scope files."
                            ),
                            "files": preview,
                            "remaining_count": max(
                                0, len(missing_file_artifacts) - len(preview)
                            ),
                        }
                    )

                if mismatched_commit_files:
                    preview = mismatched_commit_files[:50]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "file_review_artifact_commit_mismatch",
                            "role": role,
                            "expected_commit": baseline_sha,
                            "details": (
                                "Per-file review artifact commit_sha does not match baseline "
                                "commit and the file changed since that artifact."
                            ),
                            "files": preview,
                            "remaining_count": max(
                                0, len(mismatched_commit_files) - len(preview)
                            ),
                        }
                    )

                if non_approved_file_artifacts:
                    preview = non_approved_file_artifacts[:50]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "file_review_artifact_non_approved",
                            "role": role,
                            "details": (
                                "Per-file review artifacts exist but latest qualifying "
                                "entries are not approved."
                            ),
                            "files": preview,
                            "remaining_count": max(
                                0, len(non_approved_file_artifacts) - len(preview)
                            ),
                        }
                    )

                if schema_invalid_files:
                    preview = schema_invalid_files[:50]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "file_review_artifact_schema_version_invalid",
                            "role": role,
                            "details": "Per-file review artifacts must include schema_version >= 2.",
                            "files": preview,
                            "remaining_count": max(
                                0, len(schema_invalid_files) - len(preview)
                            ),
                        }
                    )

                if missing_strengths_files:
                    preview = missing_strengths_files[:50]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "file_review_artifact_missing_strengths",
                            "role": role,
                            "details": (
                                "Per-file review artifacts must include at least one positive finding (strength)."
                            ),
                            "files": preview,
                            "remaining_count": max(
                                0, len(missing_strengths_files) - len(preview)
                            ),
                        }
                    )

                if missing_checks_files:
                    preview = missing_checks_files[:50]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "file_review_artifact_missing_checks_performed",
                            "role": role,
                            "details": "Per-file review artifacts must include checks_performed entries.",
                            "files": preview,
                            "remaining_count": max(
                                0, len(missing_checks_files) - len(preview)
                            ),
                        }
                    )

                if missing_notes_for_clean_approval_files:
                    preview = missing_notes_for_clean_approval_files[:50]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "file_review_artifact_missing_notes_for_clean_approval",
                            "role": role,
                            "details": (
                                "Approved per-file artifacts with no findings must include explanatory notes."
                            ),
                            "files": preview,
                            "remaining_count": max(
                                0,
                                len(missing_notes_for_clean_approval_files)
                                - len(preview),
                            ),
                        }
                    )

                if missing_angle_rows:
                    preview = missing_angle_rows[:25]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "file_review_artifact_missing_angles",
                            "role": role,
                            "details": (
                                "Per-file artifacts do not cover all angles required by the selected review mode."
                            ),
                            "samples": preview,
                            "remaining_count": max(
                                0, len(missing_angle_rows) - len(preview)
                            ),
                        }
                    )

                if agent_mismatch_rows:
                    preview = agent_mismatch_rows[:25]
                    failures.append(
                        {
                            "component": required.name,
                            "priority": required.priority,
                            "reason": "file_review_artifact_agent_mismatch",
                            "role": role,
                            "details": (
                                "Per-file artifacts do not match required agent lane "
                                "for this review role."
                            ),
                            "samples": preview,
                            "remaining_count": max(
                                0, len(agent_mismatch_rows) - len(preview)
                            ),
                        }
                    )

    return failures


def collect_waivers(waiver_dir: Path) -> list[Waiver]:
    waivers: list[Waiver] = []
    now = datetime.now(UTC)

    if not waiver_dir.exists():
        return waivers

    for file in sorted(waiver_dir.glob("*.json")):
        data = load_json(file, {})
        if not isinstance(data, dict):
            continue

        approved_by = data.get("approved_by", [])
        if not isinstance(approved_by, list):
            approved_by = []

        expires_at = str(data.get("expires_at") or "")
        expiry_dt = parse_iso_utc(expires_at)
        valid = (
            bool(str(data.get("finding_id") or "").strip())
            and len(approved_by) >= 2
            and expiry_dt is not None
            and expiry_dt > now
        )

        waivers.append(
            Waiver(
                waiver_id=str(data.get("waiver_id") or file.stem),
                finding_id=str(data.get("finding_id") or ""),
                severity=str(data.get("severity") or "").upper(),
                expires_at=expires_at,
                approved_by=[str(x) for x in approved_by],
                reason=str(data.get("reason") or ""),
                valid=valid,
            )
        )

    return waivers


def build_verdict_payload(
    release_id: str,
    baseline_sha: str,
    review_mode: str,
    checks: list[dict[str, Any]],
    findings: list[Finding],
    waivers: list[Waiver],
    required_checks: list[str],
    failed_required_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    waiver_by_finding = {w.finding_id: w for w in waivers if w.valid}

    unresolved_p0: list[dict[str, Any]] = []
    unresolved_p1: list[dict[str, Any]] = []

    for finding in findings:
        if finding.status == "resolved":
            continue
        if finding.finding_id in waiver_by_finding:
            continue

        row = {
            "id": finding.finding_id,
            "severity": finding.severity,
            "file": finding.file,
            "line": finding.line,
            "title": finding.title,
        }
        if finding.severity == "P0":
            unresolved_p0.append(row)
        elif finding.severity == "P1":
            unresolved_p1.append(row)

    check_map = {item["key"]: item for item in checks}
    failed_required_checks: list[dict[str, str]] = []
    for key in required_checks:
        item = check_map.get(key)
        if item is None:
            failed_required_checks.append({"key": key, "status": "missing"})
            continue
        if not is_check_pass(str(item.get("status", "unknown"))):
            failed_required_checks.append(
                {"key": key, "status": str(item.get("status", "unknown"))}
            )

    accepted_waivers = [
        {
            "waiver_id": w.waiver_id,
            "finding_id": w.finding_id,
            "severity": w.severity,
            "expires_at": w.expires_at,
            "approved_by": w.approved_by,
            "reason": w.reason,
        }
        for w in waivers
        if w.valid
    ]

    if (
        unresolved_p0
        or unresolved_p1
        or failed_required_checks
        or failed_required_reviews
    ):
        status = "NOT_READY"
    elif review_mode == "quick":
        status = "QUICK_READY"
    elif accepted_waivers:
        status = "READY_WITH_ACCEPTED_RISKS"
    else:
        status = "READY"

    return {
        "release_id": release_id,
        "baseline_sha": baseline_sha,
        "review_mode": review_mode,
        "status": status,
        "unresolved_p0": unresolved_p0,
        "unresolved_p1": unresolved_p1,
        "failed_required_checks": failed_required_checks,
        "failed_required_reviews": failed_required_reviews,
        "waivers": accepted_waivers,
        "generated_at_utc": utc_now(),
    }


def resolve_baseline_sha(run_dir: Path, explicit_baseline: str | None) -> str:
    if explicit_baseline:
        return explicit_baseline

    manifest = load_json(run_dir / "run_manifest.json", {})
    if isinstance(manifest, dict):
        baseline = str(manifest.get("baseline_sha") or "").strip()
        if baseline:
            return baseline

    return get_git_sha_short()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build release verdict JSON")
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--run-dir", help="Defaults to .release_review/runs/<release_id>"
    )
    parser.add_argument(
        "--checks-file", help="Defaults to <run_dir>/gate_results/check_results.json"
    )
    parser.add_argument(
        "--output", help="Defaults to <run_dir>/gate_results/verdict.json"
    )
    parser.add_argument(
        "--baseline-sha",
        help="Defaults to run_manifest baseline_sha then git HEAD short SHA",
    )
    parser.add_argument(
        "--required-check",
        action="append",
        default=[],
        help="Repeatable required check key override",
    )
    args = parser.parse_args(argv)

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(".release_review") / "runs" / args.release_id
    ).resolve()
    checks_candidate = (
        Path(args.checks_file)
        if args.checks_file
        else Path("gate_results") / "check_results.json"
    )
    output_candidate = (
        Path(args.output) if args.output else Path("gate_results") / "verdict.json"
    )

    try:
        checks_file = resolve_within(run_dir, checks_candidate, "--checks-file")
        output = resolve_within(run_dir, output_candidate, "--output")
    except ValueError as err:
        print(f"Invalid path argument: {err}")
        return 2

    checks_raw = load_json(checks_file, [])
    checks = normalize_check_results(checks_raw)

    baseline_sha = resolve_baseline_sha(run_dir, args.baseline_sha)
    review_mode = resolve_review_mode(run_dir)

    findings = collect_evidence_findings(run_dir / "components")
    peer_entries = collect_peer_review_evidence(run_dir / "components")
    enforce_file_review_matrix = is_file_review_matrix_enforced()
    file_review_artifacts = (
        collect_available_file_review_artifacts(run_dir, review_mode)
        if enforce_file_review_matrix
        else []
    )
    file_review_agent_matrix = (
        load_file_review_agent_matrix(review_mode) if enforce_file_review_matrix else {}
    )
    scope_files = resolve_scope_files(run_dir)
    required_files_by_component, unmatched_scope_files = map_scope_files_to_components(
        scope_files
    )
    failed_required_reviews = evaluate_required_peer_reviews(
        peer_entries,
        file_review_artifacts,
        baseline_sha,
        required_files_by_component,
        unmatched_scope_files,
        file_review_agent_matrix,
        enforce_file_review_matrix,
        review_mode,
    )
    waivers = collect_waivers(run_dir / "waivers")

    required_checks = (
        args.required_check if args.required_check else DEFAULT_REQUIRED_CHECKS
    )

    payload = build_verdict_payload(
        release_id=args.release_id,
        baseline_sha=baseline_sha,
        review_mode=review_mode.name,
        checks=checks,
        findings=findings,
        waivers=waivers,
        required_checks=required_checks,
        failed_required_reviews=failed_required_reviews,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote verdict: {output}")
    print(f"Status: {payload['status']}")
    print(f"Review mode: {review_mode.name}")
    if failed_required_reviews:
        print(f"Peer-review blockers: {len(failed_required_reviews)}")
    return 1 if payload["status"] == "NOT_READY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
