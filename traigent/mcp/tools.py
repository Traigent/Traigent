"""Tool implementations for the local Traigent MCP server."""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

from traigent.api.functions import (
    list_recommendation_agent_types as public_list_recommendation_agent_types,
)
from traigent.api.functions import (
    recommend_configuration_space as public_recommend_configuration_space,
)
from traigent.cloud.credential_manager import CredentialManager
from traigent.config.backend_config import BackendConfig
from traigent.config.project import read_optional_project_env
from traigent.config.tenant import TENANT_ENV_VAR, read_optional_env
from traigent.evaluators.base import DATASET_ROOT_ENV, _resolve_dataset_source
from traigent.hooks.validator import DEFAULT_TOKENS_PER_QUERY, MODEL_COST_PER_1K
from traigent.tuned_variables.detection_types import DetectionResult
from traigent.tuned_variables.detector import TunedVariableDetector
from traigent.utils.exceptions import ValidationError as TraigentValidationError
from traigent.utils.function_identity import sanitize_identifier
from traigent.utils.persistence import PersistenceManager
from traigent.utils.secure_path import PathTraversalError, validate_path
from traigent.utils.validation import ValidationResult, Validators

V1_TOOL_NAMES: tuple[str, ...] = (
    "auth_status",
    "list_recommendation_agent_types",
    "recommend_configuration_space",
    "detect_tvars",
    "validate_dataset",
    "estimate_cost",
    "run_optimization",
    "get_results",
)

_REAL_SPEND_MESSAGE = (
    "Real optimization spends provider tokens/money. Re-run with confirm=true "
    "and an explicit positive cost_limit to start a real run."
)


class ToolInputError(ValueError):
    """Raised for user-correctable MCP tool input errors."""


def _failure(message: str, *, code: str = "invalid_input") -> dict[str, Any]:
    return {"ok": False, "code": code, "message": message}


def _serialize_validation_result(result: ValidationResult) -> dict[str, Any]:
    return {
        "ok": result.is_valid,
        "errors": [
            {
                "field": error.field,
                "message": error.message,
                "error_code": error.error_code,
                "severity": error.severity,
                "suggestions": list(error.suggestions),
                "context": dict(error.context),
            }
            for error in result.errors
        ],
        "warnings": [
            {
                "field": warning.field,
                "message": warning.message,
                "error_code": warning.error_code,
                "severity": warning.severity,
                "suggestions": list(warning.suggestions),
                "context": dict(warning.context),
            }
            for warning in result.warnings
        ],
        "suggestions": list(result.suggestions),
        "feedback": result.get_feedback(),
    }


def _resolve_cwd_file(
    file_path: str | Path,
    *,
    field_name: str,
    suffix: str | None = None,
) -> Path:
    root = Path.cwd().resolve()
    try:
        resolved = validate_path(Path(file_path).expanduser(), root, must_exist=True)
    except FileNotFoundError as exc:
        raise ToolInputError(f"{field_name} does not exist: {file_path}") from exc
    except PathTraversalError as exc:
        raise ToolInputError(
            f"{field_name} must stay under the current working directory: {root}"
        ) from exc

    if not resolved.is_file():
        raise ToolInputError(f"{field_name} must be a file: {resolved}")
    if suffix is not None and resolved.suffix != suffix:
        raise ToolInputError(f"{field_name} must be a {suffix} file: {resolved}")
    return cast(Path, resolved)


def _dataset_root() -> Path:
    configured = os.environ.get(DATASET_ROOT_ENV)
    if configured:
        # resolve(strict=True) raises FileNotFoundError when the configured root
        # is missing; surface it as a structured path_rejected failure instead of
        # letting it crash the tool call.
        try:
            return Path(configured).expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise ToolInputError(
                f"{DATASET_ROOT_ENV} points at a directory that does not exist: "
                f"{configured}"
            ) from exc
    return Path.cwd().resolve()


def _resolve_dataset_path(path: str | Path) -> tuple[Path, Path]:
    root = _dataset_root()
    try:
        resolved_path, _registry_entry = _resolve_dataset_source(str(path))
    except TraigentValidationError as exc:
        raise ToolInputError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise ToolInputError(str(exc)) from exc

    # Resolve to the REAL path before the containment check. A lexical
    # relative_to() on a possibly-unresolved path would let a symlink inside the
    # dataset root point outside and pass. validate_path() resolves symlinks and
    # re-checks containment against the resolved root.
    try:
        contained = validate_path(resolved_path, root, must_exist=True)
    except FileNotFoundError as exc:
        raise ToolInputError(f"Dataset path does not exist: {resolved_path}") from exc
    except PathTraversalError as exc:
        raise ToolInputError(
            f"Dataset path must stay under {DATASET_ROOT_ENV or 'cwd'} ({root}): "
            f"{resolved_path}"
        ) from exc
    return cast(Path, contained), root


def _mask_api_key(api_key: str | None) -> dict[str, Any]:
    if not api_key:
        return {"present": False, "prefix": None, "last4": None, "masked": None}

    prefix = api_key[:4]
    last4 = api_key[-4:] if len(api_key) > 8 else None
    masked = f"{prefix}...{last4}" if last4 is not None else f"{prefix}..."
    return {"present": True, "prefix": prefix, "last4": last4, "masked": masked}


def _sanitize_backend_url(url: str | None) -> str | None:
    """Strip any userinfo (``user:pass@``) from a backend URL before returning it.

    Credentials embedded in the URL must never leak into the MCP payload.
    """
    if not url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.username or parts.password:
        host = parts.hostname or ""
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        parts = parts._replace(netloc=host)
    return urlunsplit(parts)


async def auth_status_tool(check: bool = False) -> dict[str, Any]:
    """Return local auth status with masked key metadata only."""
    credentials = CredentialManager.get_credentials()
    api_key = credentials.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        api_key = None

    tenant_id = credentials.get("tenant_id") or read_optional_env(TENANT_ENV_VAR)
    project_id = credentials.get("project_id") or read_optional_project_env()
    backend_url = credentials.get("backend_url") or BackendConfig.get_backend_url()
    authenticated = bool(api_key or credentials.get("jwt_token"))
    validity: dict[str, Any] = {
        "checked": False,
        "valid": None,
        "source": "not_checked",
    }

    if check:
        if not api_key:
            validity = {
                "checked": True,
                "valid": False,
                "source": "local",
                "message": "No API key is available to validate.",
            }
        else:
            try:
                from traigent.cli.auth_commands import TraigentAuthCLI

                auth_cli = TraigentAuthCLI(backend_url_override=cast(str, backend_url))
                metadata = await auth_cli._validate_api_key(api_key, verbose=False)
            except Exception:
                # Live validation must NEVER propagate an exception into the MCP
                # payload — its message could embed response bodies or URLs.
                # Return a structured failure with a generic message instead.
                validity = {
                    "checked": True,
                    "valid": None,
                    "source": "backend",
                    "message": "Live validation could not be completed.",
                }
            else:
                validity = {
                    "checked": True,
                    "valid": metadata is not None,
                    "source": "backend",
                }

    return {
        "ok": True,
        "authenticated": authenticated,
        "credential_source": credentials.get("source") or "none",
        "api_key": _mask_api_key(api_key),
        "auth_type": (
            "api_key"
            if api_key
            else ("jwt" if credentials.get("jwt_token") else "none")
        ),
        "tenant_id": tenant_id,
        "project_id": project_id,
        "backend_url": _sanitize_backend_url(cast("str | None", backend_url)),
        "validity": validity,
    }


def list_recommendation_agent_types_tool() -> dict[str, Any]:
    """Return the local public recommendation catalog's agent types."""
    return {
        "ok": True,
        "agent_types": list(public_list_recommendation_agent_types()),
    }


def recommend_configuration_space_tool(
    agent_type: str,
    min_impact: Literal["low", "medium", "high"] | None = None,
    min_confidence: Literal["low", "medium", "high"] | None = None,
) -> dict[str, Any]:
    """Return a versioned local public recommendation response."""
    try:
        data = public_recommend_configuration_space(
            agent_type,
            min_impact=min_impact,
            min_confidence=min_confidence,
        )
    except ValueError as exc:
        return _failure(str(exc))
    return {"ok": True, "recommendation": data}


def _serialize_detection_result(result: DetectionResult) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for candidate in result.candidates:
        suggested_range = candidate.suggested_range
        candidates.append(
            {
                "name": candidate.name,
                "type": candidate.candidate_type.value,
                "confidence": candidate.confidence.value,
                "line": candidate.location.line,
                "col_offset": candidate.location.col_offset,
                "current_value": candidate.current_value,
                "canonical_name": candidate.canonical_name,
                "reasoning": candidate.reasoning,
                "detection_source": candidate.detection_source,
                "suggested_range": (
                    {
                        "range_type": suggested_range.range_type,
                        "kwargs": dict(suggested_range.kwargs),
                        "code": suggested_range.to_parameter_range_code(),
                    }
                    if suggested_range
                    else None
                ),
            }
        )

    return {
        "function_name": result.function_name,
        "count": result.count,
        "source_hash": result.source_hash,
        "detection_strategies_used": list(result.detection_strategies_used),
        "warnings": list(result.warnings),
        "configuration_space": result.to_configuration_space(min_confidence="low"),
        "candidates": candidates,
    }


def detect_tvars_tool(
    file_path: str,
    function_name: str | None = None,
) -> dict[str, Any]:
    """Detect tuned variables in a Python file contained by cwd."""
    try:
        resolved = _resolve_cwd_file(file_path, field_name="file_path", suffix=".py")
    except ToolInputError as exc:
        return _failure(str(exc), code="path_rejected")

    detector = TunedVariableDetector()
    results = detector.detect_from_file(resolved, function_name)
    return {
        "ok": True,
        "file_path": str(resolved),
        "path_policy": "file_path must resolve under the current working directory.",
        "results": [_serialize_detection_result(result) for result in results],
    }


def validate_dataset_tool(path: str) -> dict[str, Any]:
    """Validate a dataset path contained by TRAIGENT_DATASET_ROOT/cwd."""
    try:
        resolved_path, root = _resolve_dataset_path(path)
    except ToolInputError as exc:
        return _failure(str(exc), code="path_rejected")

    result = Validators.validate_dataset(str(resolved_path))
    payload = _serialize_validation_result(result)
    payload.update(
        {
            "dataset_path": str(resolved_path),
            "dataset_root": str(root),
            "path_policy": (
                f"path must resolve under {DATASET_ROOT_ENV} when set; "
                "otherwise under the current working directory."
            ),
        }
    )
    return payload


def _count_dataset_examples(path: Path) -> int:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return len(payload["examples"])
    return 1


def _estimate_model_cost_per_call(model: str | None) -> tuple[float | None, str | None]:
    if not model:
        return None, None

    normalized = model.lower()
    cost_per_1k = MODEL_COST_PER_1K.get(normalized)
    if cost_per_1k is None:
        for known_model, known_cost in MODEL_COST_PER_1K.items():
            if known_model in normalized or normalized in known_model:
                cost_per_1k = known_cost
                break

    if cost_per_1k is None:
        return None, None
    return cost_per_1k * (DEFAULT_TOKENS_PER_QUERY / 1000), "sdk_estimation_table"


def estimate_cost_tool(
    dataset_path: str,
    max_trials: int,
    model: str | None = None,
) -> dict[str, Any]:
    """Estimate dry-run/real-run scale from trials multiplied by examples."""
    if max_trials <= 0:
        return _failure("max_trials must be a positive integer.")

    validation = validate_dataset_tool(dataset_path)
    if not validation.get("ok"):
        return validation

    resolved_path = Path(cast(str, validation["dataset_path"]))
    example_count = _count_dataset_examples(resolved_path)
    llm_calls_upper_bound = max_trials * example_count
    cost_per_call, pricing_source = _estimate_model_cost_per_call(model)
    estimated_total_cost = (
        llm_calls_upper_bound * cost_per_call if cost_per_call is not None else None
    )

    assumptions = [
        "Upper-bound LLM calls = max_trials * dataset example count.",
        "Assumes one model invocation per trial/example and no retries/tool fan-out.",
        (
            "When model pricing is available, the SDK hook pricing table is used "
            f"with {DEFAULT_TOKENS_PER_QUERY} tokens per call."
        ),
        "Actual provider billing depends on prompt size, output size, retries, tools, and provider pricing.",
    ]
    return {
        "ok": True,
        "dataset_path": str(resolved_path),
        "dataset_root": validation["dataset_root"],
        "dataset_examples": example_count,
        "max_trials": max_trials,
        "llm_calls_upper_bound": llm_calls_upper_bound,
        "model": model,
        "estimated_cost_per_call_usd": cost_per_call,
        "estimated_total_cost_usd": estimated_total_cost,
        "pricing_source": pricing_source,
        "assumptions": assumptions,
    }


def _is_optimizable_function(obj: Any) -> bool:
    if inspect.isfunction(obj) and hasattr(obj, "optimize"):
        return True
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "OptimizedFunction":
        return True
    optimize_method = getattr(obj, "optimize", None)
    return callable(optimize_method) and (hasattr(obj, "func") or callable(obj))


def _load_user_module(script_path: Path) -> ModuleType:
    module_name = (
        f"_traigent_mcp_{sanitize_identifier(script_path.stem)}_"
        f"{abs(hash(str(script_path)))}"
    )
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ToolInputError(f"Could not load Python module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    parent = str(script_path.parent)
    inserted = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(parent)
    return module


def _find_optimizable_functions(module: ModuleType) -> list[tuple[str, Any]]:
    functions: list[tuple[str, Any]] = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_") or inspect.ismodule(obj):
            continue
        if _is_optimizable_function(obj):
            functions.append((name, obj))
    return functions


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _serialize_optimization_result_summary(
    function_name: str,
    result_name: str,
    result: Any,
) -> dict[str, Any]:
    successful_trials = getattr(result, "successful_trials", None) or []
    trials = getattr(result, "trials", None) or []
    return {
        "function_name": function_name,
        "result_name": result_name,
        "algorithm": getattr(result, "algorithm", None),
        "best_config": getattr(result, "best_config", None),
        "best_score": getattr(result, "best_score", None),
        "best_metrics": getattr(result, "best_metrics", None),
        "total_trials": len(trials),
        "successful_trials": len(successful_trials),
        "stop_reason": getattr(result, "stop_reason", None),
        "duration": getattr(result, "duration", None),
    }


def _run_loaded_optimizations(
    script_path: Path,
    *,
    mode: Literal["mock", "real"],
    cost_limit: float | None,
    max_trials: int,
    algorithm: str,
) -> tuple[list[dict[str, Any]], str, str]:
    module = _load_user_module(script_path)
    optimizable_functions = _find_optimizable_functions(module)
    if not optimizable_functions:
        raise ToolInputError(f"No @traigent.optimize functions found in {script_path}.")

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    persistence = PersistenceManager(".traigent")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summaries: list[dict[str, Any]] = []
    with (
        contextlib.redirect_stdout(stdout_buffer),
        contextlib.redirect_stderr(stderr_buffer),
    ):
        for function_name, function in optimizable_functions:
            kwargs: dict[str, Any] = {
                "algorithm": algorithm,
                "max_trials": max_trials,
                "progress_bar": False,
            }
            if mode == "real":
                kwargs["cost_limit"] = cost_limit
            result = function.optimize_sync(**kwargs)
            result_name = f"{sanitize_identifier(function_name)}_{algorithm}_{max_trials}_{timestamp}"
            persistence.save_result(result, result_name)
            summaries.append(
                _serialize_optimization_result_summary(
                    function_name, result_name, result
                )
            )

    return summaries, stdout_buffer.getvalue(), stderr_buffer.getvalue()


def run_optimization_tool(
    script_path: str | None = None,
    mode: Literal["mock", "real"] = "mock",
    confirm: bool = False,
    cost_limit: float | None = None,
    max_trials: int | None = None,
    algorithm: str | None = None,
) -> dict[str, Any]:
    """Run local optimization; real mode is gated by confirm and cost_limit."""
    if mode not in {"mock", "real"}:
        return _failure("mode must be either 'mock' or 'real'.")

    if mode == "real":
        if not confirm:
            return {"ok": False, "refused": True, "message": _REAL_SPEND_MESSAGE}
        if cost_limit is None or cost_limit <= 0:
            return {
                "ok": False,
                "refused": True,
                "message": "Real optimization requires an explicit positive cost_limit.",
            }
        from traigent.utils.env_config import is_mock_llm

        if is_mock_llm():
            return {
                "ok": False,
                "refused": True,
                "message": (
                    "Real optimization cannot run while mock mode is active in this "
                    "server process. Restart the MCP server without mock mode for a "
                    "real spend run."
                ),
            }

    if script_path is None:
        return _failure("script_path is required for run_optimization v1.")

    if max_trials is not None and max_trials <= 0:
        return _failure("max_trials must be a positive integer when provided.")

    try:
        resolved_script = _resolve_cwd_file(
            script_path, field_name="script_path", suffix=".py"
        )
    except ToolInputError as exc:
        return _failure(str(exc), code="path_rejected")

    effective_trials = max_trials or (4 if mode == "mock" else 10)
    effective_algorithm = algorithm or "random"

    # v1 single-agent limitation: optimization runs synchronously below and
    # blocks the MCP stdio event loop for the full duration of the run. No other
    # tool requests are serviced until it returns. This is acceptable for the
    # local single-client v1 server; concurrency is deferred to a later version.
    try:
        if mode == "mock":
            env = {
                "TRAIGENT_MOCK_LLM": "true",
                "TRAIGENT_OFFLINE_MODE": "true",
                "TRAIGENT_API_KEY": None,
                "TRAIGENT_DEV_API_KEY": None,
            }
            with _temporary_env(env):
                summaries, stdout_text, stderr_text = _run_loaded_optimizations(
                    resolved_script,
                    mode=mode,
                    cost_limit=None,
                    max_trials=effective_trials,
                    algorithm=effective_algorithm,
                )
        else:
            summaries, stdout_text, stderr_text = _run_loaded_optimizations(
                resolved_script,
                mode=mode,
                cost_limit=cost_limit,
                max_trials=effective_trials,
                algorithm=effective_algorithm,
            )
    except Exception as exc:
        return _failure(str(exc), code="optimization_failed")

    if mode == "real":
        # Real runs touch live providers; captured stdout/stderr can contain
        # prompts, completions, or other sensitive provider data. Scrub the tails
        # rather than returning them. Mock mode output is canned and harmless.
        stdout_payload = (
            "<redacted: real-run stdout suppressed to avoid leaking provider data>"
        )
        stderr_payload = (
            "<redacted: real-run stderr suppressed to avoid leaking provider data>"
        )
    else:
        stdout_payload = stdout_text[-4000:]
        stderr_payload = stderr_text[-4000:]

    return {
        "ok": True,
        "mode": mode,
        "script_path": str(resolved_script),
        "max_trials": effective_trials,
        "algorithm": effective_algorithm,
        "path_policy": "script_path must resolve under the current working directory.",
        "results": summaries,
        "stdout": stdout_payload,
        "stderr": stderr_payload,
    }


def _serialize_loaded_result(result: Any) -> dict[str, Any]:
    trials = []
    for trial in getattr(result, "trials", []) or []:
        trials.append(
            {
                "trial_id": getattr(trial, "trial_id", None),
                "config": getattr(trial, "config", None),
                "metrics": getattr(trial, "metrics", None),
                "status": str(getattr(trial, "status", "")),
                "duration": getattr(trial, "duration", None),
            }
        )

    return {
        "algorithm": getattr(result, "algorithm", None),
        "objectives": list(getattr(result, "objectives", []) or []),
        "best_config": getattr(result, "best_config", None),
        "best_score": getattr(result, "best_score", None),
        "best_metrics": getattr(result, "best_metrics", None),
        "status": str(getattr(result, "status", "")),
        "duration": getattr(result, "duration", None),
        "metadata": dict(getattr(result, "metadata", {}) or {}),
        "trials": trials,
    }


def get_results_tool(result_name: str | None = None) -> dict[str, Any]:
    """List or show local optimization results from the .traigent store."""
    persistence = PersistenceManager(".traigent")
    if result_name is None:
        return {
            "ok": True,
            "storage_dir": str(persistence.base_dir),
            "results": persistence.list_results(),
        }

    try:
        result = persistence.load_result(result_name)
    except FileNotFoundError as exc:
        return _failure(str(exc), code="not_found")
    except (PathTraversalError, ValueError) as exc:
        return _failure(str(exc), code="path_rejected")

    return {
        "ok": True,
        "storage_dir": str(persistence.base_dir),
        "result_name": result_name,
        "result": _serialize_loaded_result(result),
    }
