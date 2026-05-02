"""Static scan report generation for ``traigent optimizer scan``."""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from traigent._version import get_version
from traigent.tuned_variables.detection_types import (
    CandidateType,
    TunedVariableCandidate,
)
from traigent.tuned_variables.detector import TunedVariableDetector

_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}

_LLM_CONSTRUCTOR_NAMES = {
    "ChatOpenAI": "langchain",
    "AzureChatOpenAI": "langchain",
    "ChatAnthropic": "langchain",
    "OpenAI": "openai",
    "Anthropic": "anthropic",
}


@dataclass(frozen=True, slots=True)
class _FunctionInfo:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    qualified_name: str
    source: str


def scan_path(path: str | Path, function_name: str | None = None) -> dict[str, Any]:
    """Scan a Python file or directory and return a schema-compatible report.

    The scan is static only: it reads and parses source files, but never imports
    or executes user code.
    """

    root = Path(path).expanduser().resolve()
    if root.is_file():
        scan_root = root.parent
        files = [root]
    else:
        scan_root = root
        files = _iter_python_files(root)

    candidates: list[dict[str, Any]] = []
    detector = TunedVariableDetector()
    for file_path in files:
        candidates.extend(
            _scan_file(
                file_path=file_path,
                scan_root=scan_root,
                detector=detector,
                function_name=function_name,
            )
        )

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return {
        "report_version": "0.1.0",
        "runtime": "python",
        "scan_root": str(scan_root),
        "generated_at": _utc_now(),
        "tool_version": f"traigent=={get_version()}",
        "candidates": candidates,
    }


def _iter_python_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if not any(
            part in _SKIP_DIR_NAMES or part.startswith(".") for part in path.parts
        )
    ]


def _scan_file(
    *,
    file_path: Path,
    scan_root: Path,
    detector: TunedVariableDetector,
    function_name: str | None,
) -> list[dict[str, Any]]:
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    module_name = _module_name(file_path, scan_root)
    lines = source.splitlines()
    candidates: list[dict[str, Any]] = []
    for function in _collect_functions(tree, module_name, lines):
        if function_name and function.node.name != function_name:
            continue
        candidate = _build_candidate(file_path, scan_root, function, detector, lines)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _module_name(file_path: Path, scan_root: Path) -> str:
    try:
        relative_parts = list(file_path.relative_to(scan_root).with_suffix("").parts)
    except ValueError:
        relative_parts = [file_path.stem]
    return ".".join(part for part in relative_parts if part != "__init__")


def _collect_functions(
    tree: ast.AST,
    module_name: str,
    lines: list[str],
) -> list[_FunctionInfo]:
    collector = _FunctionCollector(module_name, lines)
    collector.visit(tree)
    return collector.functions


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, module_name: str, lines: list[str]) -> None:
        self._module_name = module_name
        self._lines = lines
        self._class_stack: list[str] = []
        self.functions: list[_FunctionInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        for child in node.body:
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node)

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        prefix_parts = [self._module_name, *self._class_stack]
        qualified_name = ".".join(part for part in [*prefix_parts, node.name] if part)
        self.functions.append(
            _FunctionInfo(
                node=node,
                qualified_name=qualified_name,
                source=_function_source(self._lines, node),
            )
        )


def _function_source(
    lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    start = max(node.lineno - 1, 0)
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(lines[start:end])


def _build_candidate(
    file_path: Path,
    scan_root: Path,
    function: _FunctionInfo,
    detector: TunedVariableDetector,
    lines: list[str],
) -> dict[str, Any] | None:
    function_source = textwrap.dedent(function.source)
    detection = detector.detect_from_source(function_source, function.node.name)
    line_offset = function.node.lineno - 1
    relative_file = _relative_file(file_path, scan_root)
    signals = _detect_signals(function.node, relative_file, lines)
    tvar_signals = [
        _candidate_to_tvar_signal(candidate, relative_file, lines, line_offset)
        for candidate in detection.candidates
        if _is_actionable_tvar_candidate(candidate)
    ]

    if not signals and not tvar_signals:
        return None

    source_hash = hashlib.sha256(function.source.encode("utf-8")).hexdigest()
    span_hash = hashlib.sha256(
        ast.dump(function.node, include_attributes=False).encode("utf-8")
    ).hexdigest()
    objective_candidates = _objective_candidates(signals)
    return {
        "fingerprint": {
            "candidate_id": _candidate_id(span_hash, function.qualified_name),
            "source_hash": source_hash,
            "source_span_hash": span_hash,
        },
        "function": {
            "file": relative_file,
            "line": function.node.lineno,
            "end_line": getattr(function.node, "end_lineno", function.node.lineno),
            "name": function.node.name,
            "qualified_name": function.qualified_name,
        },
        "score": _score(signals, tvar_signals),
        "signals": signals,
        "tvar_signals": tvar_signals,
        "objective_candidates": objective_candidates,
        "dataset_status": {"status": "stub_required"},
    }


def _relative_file(file_path: Path, scan_root: Path) -> str:
    try:
        return file_path.relative_to(scan_root).as_posix()
    except ValueError:
        return file_path.as_posix()


def _candidate_id(source_span_hash: str, qualified_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", qualified_name).strip("_")
    return f"{source_span_hash[:8]}-{slug}"


def _score(signals: list[dict[str, Any]], tvar_signals: list[dict[str, Any]]) -> float:
    raw = 0.35 + (0.15 * min(len(signals), 2)) + (0.1 * min(len(tvar_signals), 3))
    if any(signal["kind"] == "llm_call" for signal in signals):
        raw += 0.12
    return round(min(raw, 1.0), 2)


def _detect_signals(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    relative_file: str,
    lines: list[str],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        call_name = _call_name(child.func)
        if not call_name:
            continue
        framework = _llm_framework(call_name)
        if framework:
            signals.append(
                _signal(
                    kind="llm_call",
                    framework=framework,
                    node=child,
                    relative_file=relative_file,
                    lines=lines,
                    category=(
                        "framework_constructor_arg"
                        if _is_constructor_call(call_name)
                        else "framework_call_kwarg"
                    ),
                    notes=f"Recognized LLM call: {call_name}",
                )
            )
        elif _is_retrieval_call(call_name):
            signals.append(
                _signal(
                    kind="retrieval_call",
                    framework="",
                    node=child,
                    relative_file=relative_file,
                    lines=lines,
                    category="framework_call_kwarg",
                    notes=f"Recognized retrieval call: {call_name}",
                )
            )
    return _dedupe_signals(signals)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def _llm_framework(call_name: str) -> str:
    leaf = call_name.split(".")[-1]
    if leaf in _LLM_CONSTRUCTOR_NAMES:
        return _LLM_CONSTRUCTOR_NAMES[leaf]

    lower = call_name.lower()
    if lower.endswith(".chat.completions.create") or lower.endswith(
        ".responses.create"
    ):
        return "openai"
    if lower.endswith(".messages.create"):
        return "anthropic"
    if lower in {"litellm.completion", "litellm.acompletion"}:
        return "litellm"
    return ""


def _is_constructor_call(call_name: str) -> bool:
    return call_name.split(".")[-1] in _LLM_CONSTRUCTOR_NAMES


def _is_retrieval_call(call_name: str) -> bool:
    lower = call_name.lower()
    return lower.endswith(
        (
            ".similarity_search",
            ".asimilarity_search",
            ".get_relevant_documents",
            ".aget_relevant_documents",
            ".retrieve",
        )
    )


def _signal(
    *,
    kind: str,
    framework: str,
    node: ast.AST,
    relative_file: str,
    lines: list[str],
    category: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "framework": framework,
        "evidence": {
            "file": relative_file,
            "line": getattr(node, "lineno", 1),
            "end_line": getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            "snippet": _snippet(lines, getattr(node, "lineno", 1)),
            "category": category,
        },
        "notes": notes,
    }


def _dedupe_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for signal in signals:
        key = (
            signal["kind"],
            signal["evidence"]["line"],
            signal["evidence"]["snippet"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(signal)
    return deduped


def _candidate_to_tvar_signal(
    candidate: TunedVariableCandidate,
    relative_file: str,
    lines: list[str],
    line_offset: int,
) -> dict[str, Any]:
    absolute_line = max(candidate.location.line + line_offset, 1)
    return {
        "tvar": _candidate_to_tvar(candidate),
        "confidence": candidate.confidence.value,
        "evidence": {
            "file": relative_file,
            "line": absolute_line,
            "end_line": (
                candidate.location.end_line + line_offset
                if candidate.location.end_line is not None
                else absolute_line
            ),
            "snippet": _snippet(lines, absolute_line),
            "category": _evidence_category(candidate, _snippet(lines, absolute_line)),
        },
    }


def _is_actionable_tvar_candidate(candidate: TunedVariableCandidate) -> bool:
    """Keep only candidates that can become a concrete optimizer search knob."""
    return candidate.current_value is not None or candidate.suggested_range is not None


def _candidate_to_tvar(candidate: TunedVariableCandidate) -> dict[str, Any]:
    tvar: dict[str, Any] = {
        "name": candidate.name,
        "type": _candidate_type(candidate),
    }
    domain = _candidate_domain(candidate)
    if domain:
        tvar["domain"] = domain
    default = _candidate_default(candidate)
    if default is not None:
        tvar["default"] = default
    if candidate.suggested_range and candidate.suggested_range.range_type == "LogRange":
        tvar["scale"] = "log"
    return tvar


def _candidate_type(candidate: TunedVariableCandidate) -> str:
    if candidate.candidate_type == CandidateType.NUMERIC_CONTINUOUS:
        return "float"
    if candidate.candidate_type == CandidateType.NUMERIC_INTEGER:
        return "int"
    if candidate.candidate_type == CandidateType.BOOLEAN:
        return "bool"
    if candidate.suggested_range and candidate.suggested_range.range_type == "Choices":
        return "enum"
    return "str"


def _candidate_domain(candidate: TunedVariableCandidate) -> dict[str, Any]:
    suggested = candidate.suggested_range
    if suggested is None:
        return {}
    kwargs = suggested.kwargs
    if suggested.range_type in {"Range", "IntRange", "LogRange"}:
        low = kwargs.get("low")
        high = kwargs.get("high")
        if low is None or high is None:
            return {}
        domain: dict[str, Any] = {"range": [low, high]}
        step = kwargs.get("step")
        if step is not None:
            domain["resolution"] = step
        return domain
    if suggested.range_type == "Choices":
        values = kwargs.get("values")
        if isinstance(values, (list, tuple)) and values:
            return {"values": list(values)}
    return {}


def _candidate_default(candidate: TunedVariableCandidate) -> Any:
    if candidate.current_value is not None:
        return candidate.current_value
    if candidate.suggested_range is not None:
        return candidate.suggested_range.kwargs.get("default")
    return None


def _evidence_category(candidate: TunedVariableCandidate, snippet: str) -> str:
    if f"{candidate.name}=" in snippet or f"{candidate.name} =" in snippet:
        return "framework_call_kwarg"
    if "=" in snippet:
        return "literal_assignment"
    return "other"


def _objective_candidates(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objective_map: dict[str, dict[str, Any]] = {}
    if any(signal["kind"] == "llm_call" for signal in signals):
        objective_map["accuracy"] = _objective(
            name="accuracy",
            direction="maximize",
            confidence="medium",
            rationale=(
                "Function calls an LLM; a quality metric should be confirmed "
                "before optimization."
            ),
            required_dataset_fields=["input", "expected_output"],
            auto_measurable=False,
        )
        objective_map["cost"] = _objective(
            name="cost",
            direction="minimize",
            confidence="high",
            rationale="LLM call detected; cost is auto-measurable from token usage.",
            required_dataset_fields=[],
            auto_measurable=True,
        )
        objective_map["latency"] = _objective(
            name="latency",
            direction="minimize",
            confidence="high",
            rationale="LLM call detected; latency is auto-measurable from timing.",
            required_dataset_fields=[],
            auto_measurable=True,
        )

    if any(signal["kind"] == "retrieval_call" for signal in signals):
        objective_map["recall_at_k"] = _objective(
            name="recall_at_k",
            direction="maximize",
            confidence="medium",
            rationale=(
                "Retrieval call detected; recall_at_k is the conventional quality "
                "metric when relevance labels are available."
            ),
            required_dataset_fields=["input", "relevant_doc_ids"],
            auto_measurable=False,
        )
        objective_map.setdefault(
            "latency",
            _objective(
                name="latency",
                direction="minimize",
                confidence="high",
                rationale="Retrieval call detected; latency is auto-measurable.",
                required_dataset_fields=[],
                auto_measurable=True,
            ),
        )
    return list(objective_map.values())


def _objective(
    *,
    name: str,
    direction: str,
    confidence: str,
    rationale: str,
    required_dataset_fields: list[str],
    auto_measurable: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "direction": direction,
        "confidence": confidence,
        "rationale": rationale,
        "required_dataset_fields": required_dataset_fields,
        "auto_measurable": auto_measurable,
        "requires_confirmation": not auto_measurable,
    }


def _snippet(lines: list[str], line: int) -> str:
    if line <= 0 or line > len(lines):
        return ""
    return lines[line - 1].strip()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
