"""Decorate-plan generation for ``traigent optimizer decorate``."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from traigent._version import get_version
from traigent.optimizer.scanner import scan_path


def build_decorate_plan(
    path: str | Path,
    *,
    function_name: str,
    objective_names: Iterable[str] = (),
    dataset_ref: str | None = None,
    requested_emit_mode: str = "auto",
) -> dict[str, Any]:
    """Build a dry-run decorate plan for one function.

    The plan is reviewable and schema-shaped, but this v1 slice does not apply
    patches. The caller can serialize it as JSON for agent/IDE workflows.
    """

    report = scan_path(path, function_name=function_name)
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

    return {
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
        "dataset_plan": _dataset_plan(
            candidate["function"]["file"],
            candidate["function"]["name"],
            dataset_ref,
            expected_fields,
        ),
        "emitted_files": [],
        "confirmation_state": {
            "objectives_confirmed": bool(selected_objectives),
            "dataset_confirmed": dataset_ref is not None,
            "write_authorized": False,
        },
        "warnings": _warnings(candidate, selected_objectives),
    }


def _select_candidate(
    candidates: list[dict[str, Any]],
    function_name: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError(f"No optimizer candidate found for function '{function_name}'")
    return candidates[0]


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

    confirmed_names = {objective["name"] for objective in selected_objectives}
    selected_candidates = [
        candidate
        for candidate in objective_candidates
        if candidate["name"] in confirmed_names
    ]
    selected_are_auto = all(
        candidate["auto_measurable"] for candidate in selected_candidates
    )
    if len(bindings) <= 3 and (not selected_candidates or selected_are_auto):
        return "inline"
    return "tvl"


def _dataset_plan(
    target_file: str,
    function_name: str,
    dataset_ref: str | None,
    expected_fields: list[str],
) -> dict[str, Any]:
    if dataset_ref is not None:
        return {
            "status": "present",
            "dataset_ref": dataset_ref,
            "format": _dataset_format(dataset_ref),
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
) -> list[str]:
    warnings: list[str] = []
    if not selected_objectives:
        warnings.append(
            "No objectives selected; pass --objective or review objective_candidates before --write."
        )
    if not candidate["tvar_signals"]:
        warnings.append("No concrete TVARs were detected for this function.")
    return warnings


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
