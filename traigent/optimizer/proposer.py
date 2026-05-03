"""Decorate-plan generation for ``traigent optimizer decorate``."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from traigent._version import get_version
from traigent.optimizer.agent_enrichment import AgentRunConfig, enrich_decorate_plan
from traigent.optimizer.scanner import infer_project_root, scan_path


def build_decorate_plan(
    path: str | Path,
    *,
    function_name: str,
    objective_names: Iterable[str] = (),
    dataset_ref: str | None = None,
    requested_emit_mode: str = "auto",
    agent_mode: str = "static",
    agent_budget_tokens: int = 8_000,
    agent_timeout_seconds: int = 120,
    agent_command: str | None = None,
    agent_model: str | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a dry-run decorate plan for one function.

    The plan is reviewable and schema-shaped, but this v1 slice does not apply
    patches. The caller can serialize it as JSON for agent/IDE workflows.
    """

    report = scan_path(path, function_name=function_name, agent_mode="static")
    candidate = _select_candidate(report["candidates"], function_name)
    selected_objectives = _selected_objectives(
        candidate["objective_candidates"],
        objective_names,
    )
    selected_names = {objective["name"] for objective in selected_objectives}
    expected_fields = _expected_fields(
        candidate["objective_candidates"],
        selected_names,
    )
    bindings = [
        _binding_from_tvar_signal(signal, candidate["function"]["name"])
        for signal in candidate["tvar_signals"]
    ]
    resolved_emit_mode = _resolve_emit_mode(
        requested_emit_mode=requested_emit_mode,
        bindings=bindings,
        selected_objectives=selected_objectives,
        objective_candidates=candidate["objective_candidates"],
    )
    dataset_plan = _dataset_plan(
        candidate["function"]["file"],
        candidate["function"]["name"],
        dataset_ref,
        expected_fields,
        candidate["dataset_status"],
    )

    plan = {
        "plan_version": "0.1.0",
        "runtime": "python",
        "tool_version": f"traigent=={get_version()}",
        "generated_at": _utc_now(),
        "target": {
            "file": candidate["function"]["file"],
            "function": candidate["function"]["qualified_name"],
            "line": candidate["function"]["line"],
            "candidate_id": candidate["fingerprint"]["candidate_id"],
            "source_hash": candidate["fingerprint"]["source_hash"],
            "source_span_hash": candidate["fingerprint"]["source_span_hash"],
        },
        "requested_emit_mode": requested_emit_mode,
        "resolved_emit_mode": resolved_emit_mode,
        "injection_mode": "context",
        "proposed_tvar_bindings": bindings,
        "selected_objectives": selected_objectives,
        "objective_candidates": candidate["objective_candidates"],
        "dataset_plan": dataset_plan,
        "agent_enrichment": None,
        "emitted_files": [],
        "confirmation_state": {
            "objectives_confirmed": bool(selected_objectives),
            "dataset_confirmed": dataset_ref is not None
            or candidate["dataset_status"]["status"] == "present",
            "write_authorized": False,
        },
        "warnings": _warnings(
            candidate,
            selected_objectives,
            dataset_plan,
            scan_root=Path(report["scan_root"]),
        ),
    }
    if agent_mode == "static":
        return plan

    source_path = Path(report["scan_root"]) / candidate["function"]["file"]
    return enrich_decorate_plan(
        plan,
        source_path=source_path,
        function_source=_read_function_source(source_path, candidate["function"]),
        config=AgentRunConfig(
            mode=agent_mode,
            project_root=(
                Path(project_root).expanduser().resolve()
                if project_root is not None
                else infer_project_root(report["scan_root"])
            ),
            timeout_seconds=agent_timeout_seconds,
            budget_tokens=agent_budget_tokens,
            command=agent_command,
            model=agent_model,
        ),
    )


def _select_candidate(
    candidates: list[dict[str, Any]],
    function_name: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError(f"No optimizer candidate found for function '{function_name}'")
    return candidates[0]


def _read_function_source(source_path: Path, function_info: dict[str, Any]) -> str:
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    start = max(int(function_info.get("line", 1)) - 1, 0)
    end = int(function_info.get("end_line", start + 1))
    return "\n".join(lines[start:end])


def _selected_objectives(
    objective_candidates: list[dict[str, Any]],
    objective_names: Iterable[str],
) -> list[dict[str, Any]]:
    by_name = {candidate["name"]: candidate for candidate in objective_candidates}
    selected: list[dict[str, Any]] = []
    for name in objective_names:
        candidate = by_name.get(name)
        if candidate is not None:
            selected.append(
                {
                    "name": candidate["name"],
                    "direction": candidate["direction"],
                }
            )
            continue
        selected.append(
            {
                "name": name,
                "direction": "minimize" if name in {"cost", "latency"} else "maximize",
            }
        )
    return selected


def _expected_fields(
    objective_candidates: list[dict[str, Any]],
    selected_names: set[str],
) -> list[str]:
    fields: list[str] = []
    for candidate in objective_candidates:
        if selected_names and candidate["name"] not in selected_names:
            continue
        for field in candidate["required_dataset_fields"]:
            if field not in fields:
                fields.append(field)
    return fields


def _binding_from_tvar_signal(
    signal: dict[str, Any],
    function_name: str,
) -> dict[str, Any]:
    tvar = signal["tvar"]
    evidence = signal["evidence"]
    return {
        "tvar": tvar,
        "confidence": signal["confidence"],
        "domain_source": signal["domain_source"],
        "evidence": evidence,
        "injection_mode": "context",
        "current_value": tvar.get("default"),
        "locator": {
            "kind": "line_col",
            "details": {
                "function": function_name,
                "line": evidence["line"],
                "tvar": tvar["name"],
            },
        },
    }


def _resolve_emit_mode(
    *,
    requested_emit_mode: str,
    bindings: list[dict[str, Any]],
    selected_objectives: list[dict[str, Any]],
    objective_candidates: list[dict[str, Any]],
) -> str:
    if requested_emit_mode != "auto":
        return requested_emit_mode
    return "tvl"


def _dataset_plan(
    target_file: str,
    function_name: str,
    dataset_ref: str | None,
    expected_fields: list[str],
    dataset_status: dict[str, Any],
) -> dict[str, Any]:
    if dataset_ref is not None:
        return {
            "status": "present",
            "dataset_ref": dataset_ref,
            "format": _dataset_format(dataset_ref),
            "expected_fields": expected_fields,
        }
    if dataset_status.get("status") == "present":
        inferred_ref = dataset_status["candidate_path"]
        return {
            "status": "present",
            "dataset_ref": inferred_ref,
            "format": _dataset_format(inferred_ref),
            "expected_fields": expected_fields,
        }

    target_path = Path(target_file)
    return {
        "status": "stub_required",
        "format": "jsonl",
        "stub_path": (target_path.parent / f"{function_name}_dataset.jsonl").as_posix(),
        "expected_fields": expected_fields,
    }


def _dataset_format(dataset_ref: str) -> str:
    suffix = Path(dataset_ref).suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    if dataset_ref.startswith("hf://"):
        return "hf_dataset"
    return "other"


def _warnings(
    candidate: dict[str, Any],
    selected_objectives: list[dict[str, Any]],
    dataset_plan: dict[str, Any],
    *,
    scan_root: Path,
) -> list[str]:
    warnings: list[str] = []
    if not selected_objectives:
        warnings.append(
            "No objectives selected; pass --objective or review objective_candidates before --write."
        )
    if not candidate["tvar_signals"]:
        warnings.append("No concrete TVARs were detected for this function.")
    for signal in candidate["tvar_signals"]:
        values = signal["tvar"].get("domain", {}).get("values")
        if isinstance(values, list) and len({repr(value) for value in values}) <= 1:
            warnings.append(
                f"TVAR {signal['tvar']['name']} has a singleton enum domain; "
                "review before treating it as a search space."
            )
    dataset_warning = _dataset_field_warning(dataset_plan, scan_root)
    if dataset_warning:
        warnings.append(dataset_warning)
    return warnings


def _dataset_field_warning(dataset_plan: dict[str, Any], scan_root: Path) -> str | None:
    if dataset_plan.get("status") != "present":
        return None
    expected_fields = dataset_plan.get("expected_fields", [])
    if not expected_fields:
        return None
    dataset_ref = dataset_plan.get("dataset_ref")
    if not isinstance(dataset_ref, str) or dataset_ref.startswith(("hf://", "s3://")):
        return None
    dataset_path = Path(dataset_ref)
    if not dataset_path.is_absolute():
        dataset_path = scan_root / dataset_path
    actual_fields = _read_dataset_fields(dataset_path)
    if not actual_fields:
        return None
    missing = [field for field in expected_fields if field not in actual_fields]
    if not missing:
        return None
    return (
        f"Dataset {dataset_ref} is present but missing expected fields: "
        f"{', '.join(missing)}. Found fields: {', '.join(actual_fields)}."
    )


def _read_dataset_fields(path: Path) -> list[str]:
    try:
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                return list(value) if isinstance(value, dict) else []
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                return next(reader, [])
    except (OSError, json.JSONDecodeError, StopIteration):
        return []
    return []


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
