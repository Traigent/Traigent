"""Optional TraceSync runtime instrumentation bootstrap (out-of-band).

This module monkey-patches Traigent entrypoints at runtime to emit TraceSync
actions without modifying SDK source files. It is safe to omit from packaged
artifacts; the SDK does not depend on it.

Usage:
    # Enable via environment and import side-effect
    TRAIGENT_TRACE_SYNC=1 python -m tools.trace_sync_bootstrap

    # Or programmatically
    from tools.trace_sync_bootstrap import enable_trace_sync
    enable_trace_sync()
"""

from __future__ import annotations

import functools
import os
from typing import Any, Awaitable, Callable, Optional

ENABLE_ENV = "TRAIGENT_TRACE_SYNC"


def _noop_decorator(
    *_args: Any, **_kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return _wrapper


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


try:
    from tools.trace_sync.runtime import (  # type: ignore
        flow_context,
        get_flow_id,
        record_action,
        set_flow_id,
        should_trace,
    )

    TRACE_RUNTIME_AVAILABLE = True
except Exception:
    record_action = _noop
    set_flow_id = _noop
    flow_context = _noop_decorator
    should_trace = lambda *_args, **_kwargs: False  # type: ignore[assignment]
    get_flow_id = lambda *_args, **_kwargs: None  # type: ignore[assignment]
    TRACE_RUNTIME_AVAILABLE = False

_PATCH_FLAG = "_trace_sync_patched"


def _mark_patched(obj: Any) -> Any:
    setattr(obj, _PATCH_FLAG, True)
    return obj


def _is_patched(obj: Any) -> bool:
    return bool(getattr(obj, _PATCH_FLAG, False))


def _safe_patch(target: Any, attr: str, wrapper_fn: Callable[[Any], Any]) -> bool:
    """Patch target.attr with wrapper_fn(target.attr) if not already patched."""
    original = getattr(target, attr, None)
    if original is None or _is_patched(original):
        return False

    wrapped = wrapper_fn(original)
    _mark_patched(wrapped)
    setattr(target, attr, wrapped)
    return True


def _build_concept_payload(
    concept_id: str,
    req_ids: list[str],
    func_ids: list[str],
    sync_ids: list[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "req_ids": req_ids,
        "func_ids": func_ids,
        "sync_ids": sync_ids,
        "details": details,
    }


def _emit_record(kind: str, payload: dict[str, Any]) -> None:
    """Emit a trace record if runtime is available and sampling passes."""
    flow_id = get_flow_id()
    if TRACE_RUNTIME_AVAILABLE and should_trace(flow_id):
        record_action(
            {
                "kind": kind,
                "flow_id": flow_id,
                **payload,
            }
        )


def _wrap_sync(
    fn: Callable[..., Any], concept_kwargs: dict[str, Any]
) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            _emit_record(
                "concept_action",
                {
                    "concept_id": concept_kwargs.get("concept_id"),
                    "action_name": concept_kwargs.get("details", {}).get(
                        "operation", fn.__name__
                    ),
                    "inputs": concept_kwargs,
                },
            )
        except Exception:
            pass
        return fn(*args, **kwargs)

    return wrapper


def _wrap_sync_dynamic(
    fn: Callable[..., Any],
    concept_kwargs_factory: Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]],
) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            payload = concept_kwargs_factory(args, kwargs)
            if payload:
                _emit_record(
                    "concept_action",
                    {
                        "concept_id": payload.get("concept_id"),
                        "action_name": payload.get("details", {}).get(
                            "operation", fn.__name__
                        ),
                        "inputs": payload,
                    },
                )
        except Exception:
            pass
        return fn(*args, **kwargs)

    return wrapper


def _wrap_async(
    fn: Callable[..., Awaitable[Any]],
    *,
    concept_kwargs_factory: Optional[
        Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]
    ] = None,
    event_details_factory: Optional[Callable[[Any], dict[str, Any]]] = None,
) -> Callable[..., Awaitable[Any]]:
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if concept_kwargs_factory:
            try:
                payload = concept_kwargs_factory(args, kwargs)
                if payload:
                    _emit_record(
                        "concept_action",
                        {
                            "concept_id": payload.get("concept_id"),
                            "action_name": payload.get("details", {}).get(
                                "operation", fn.__name__
                            ),
                            "inputs": payload,
                        },
                    )
            except Exception:
                pass

        result = await fn(*args, **kwargs)

        if event_details_factory:
            try:
                details = event_details_factory(result)
                if details is not None:
                    _emit_record(
                        "event",
                        {
                            "concept_id": details.get("concept_id"),
                            "action_name": details.get("action"),
                            "inputs": details,
                        },
                    )
            except Exception:
                pass
        return result

    return wrapper


def _patch_api_functions() -> None:
    import traigent.api.functions as api_functions

    def _concept_configure(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        return _build_concept_payload(
            "CONC-Layer-API",
            ["REQ-API-001"],
            ["FUNC-API-ENTRY"],
            ["SYNC-OptimizationFlow"],
            {
                "operation": "configure",
                "parallel_workers": kwargs.get("parallel_workers"),
                "cache_policy": kwargs.get("cache_policy"),
                "logging_level": kwargs.get("logging_level"),
                "parallel_config_provided": kwargs.get("parallel_config") is not None,
                "objectives_provided": kwargs.get("objectives") is not None,
            },
        )

    def _concept_initialize(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        return _build_concept_payload(
            "CONC-Layer-API",
            ["REQ-API-001"],
            ["FUNC-API-ENTRY"],
            ["SYNC-OptimizationFlow"],
            {
                "operation": "initialize",
                "api_url_provided": kwargs.get("api_url") is not None,
                "config_provided": kwargs.get("config") is not None,
            },
        )

    _safe_patch(
        api_functions,
        "configure",
        lambda fn: _wrap_sync_dynamic(fn, _concept_configure),
    )

    _safe_patch(
        api_functions,
        "initialize",
        lambda fn: _wrap_sync_dynamic(fn, _concept_initialize),
    )


def _patch_optimize_decorator() -> None:
    import traigent.api.decorators as api_decorators

    def _concept_kwargs(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        provided_objectives = kwargs.get("objectives") is not None
        provided_config_space = kwargs.get("configuration_space") is not None
        tvl_spec = kwargs.get("tvl_spec")
        return _build_concept_payload(
            "CONC-Layer-API",
            ["REQ-API-001"],
            ["FUNC-API-ENTRY"],
            ["SYNC-OptimizationFlow"],
            {
                "operation": "optimize_decorator",
                "objectives_provided": provided_objectives,
                "configuration_space_provided": provided_config_space,
                "tvl_spec": str(tvl_spec) if tvl_spec else None,
            },
        )

    _safe_patch(
        api_decorators,
        "optimize",
        lambda fn: _wrap_sync_dynamic(fn, _concept_kwargs),
    )


def _patch_optimized_function() -> None:
    import traigent.core.optimized_function as optimized_function

    def _concept_kwargs(
        args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        self = args[0]
        dataset = getattr(self, "eval_dataset", None)
        dataset_kind = type(dataset).__name__ if dataset is not None else None
        config_space = bool(getattr(self, "configuration_space", None))
        return _build_concept_payload(
            "CONC-Layer-Core",
            ["REQ-ORCH-003"],
            ["FUNC-ORCH-LIFECYCLE"],
            ["SYNC-OptimizationFlow"],
            {
                "execution_mode": getattr(self, "execution_mode", None),
                "dataset_kind": dataset_kind,
                "has_configuration_space": config_space,
            },
        )

    def _event_details(result: Any) -> dict[str, Any]:
        try:
            trials = getattr(result, "trials", []) or []
            best_score = getattr(result, "best_score", None)
            optimization_id = getattr(result, "optimization_id", None)
        except Exception:
            trials = []
            best_score = None
            optimization_id = None
        return {
            "action": "optimize_completed",
            "trial_count": len(trials),
            "best_score": best_score,
            "optimization_id": optimization_id,
        }

    _safe_patch(
        optimized_function.OptimizedFunction,
        "optimize",
        lambda fn: _wrap_async(
            fn,
            concept_kwargs_factory=_concept_kwargs,
            event_details_factory=_event_details,
        ),
    )


def _patch_orchestrator() -> None:
    import traigent.core.orchestrator as orchestrator

    def _concept_kwargs(
        args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        self = args[0]
        mode = getattr(getattr(self, "traigent_config", None), "execution_mode", None)
        parallel_trials = getattr(self, "parallel_trials", None)
        return _build_concept_payload(
            "CONC-Layer-Core",
            ["REQ-ORCH-003"],
            ["FUNC-ORCH-LIFECYCLE"],
            ["SYNC-OptimizationFlow"],
            {
                "execution_mode": mode,
                "parallel_trials": parallel_trials,
            },
        )

    def _event_details(result: Any) -> dict[str, Any]:
        trials = getattr(result, "trials", []) or []
        best_score = getattr(result, "best_score", None)
        status = getattr(result, "status", None)
        return {
            "action": "orchestrator_complete",
            "trial_count": len(trials),
            "best_score": best_score,
            "status": getattr(status, "name", status),
        }

    _safe_patch(
        orchestrator.OptimizationOrchestrator,
        "optimize",
        lambda fn: _wrap_async(
            fn,
            concept_kwargs_factory=_concept_kwargs,
            event_details_factory=_event_details,
        ),
    )


def _patch_invokers() -> None:
    import traigent.invokers.local as local_invokers
    import traigent.invokers.batch as batch_invokers

    def _wrap_local_invoke(
        fn: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        def _concept_kwargs(
            args: tuple[Any, ...], _kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            _self, func, config, input_data = args[:4]
            return _build_concept_payload(
                "CONC-Layer-Core",
                ["REQ-INV-006", "REQ-INJ-002"],
                ["FUNC-INVOKERS"],
                ["SYNC-OptimizationFlow"],
                {
                    "function_name": getattr(func, "__name__", "callable"),
                    "config_keys": (
                        sorted(list(config.keys()))
                        if isinstance(config, dict)
                        else None
                    ),
                    "input_type": type(input_data).__name__,
                },
            )

        def _event_details(result: Any) -> dict[str, Any]:
            status = "success" if getattr(result, "is_successful", False) else "error"
            return {
                "action": "invoke",
                "status": status,
                "execution_time": getattr(result, "execution_time", None),
            }

        return _wrap_async(
            fn,
            concept_kwargs_factory=_concept_kwargs,
            event_details_factory=_event_details,
        )

    _safe_patch(local_invokers.LocalInvoker, "invoke", _wrap_local_invoke)
    _safe_patch(local_invokers.LocalInvoker, "invoke_batch", lambda fn: _wrap_async(fn))
    _safe_patch(batch_invokers.BatchInvoker, "invoke_batch", lambda fn: _wrap_async(fn))


def _patch_evaluators() -> None:
    import traigent.evaluators.local as local_evaluators

    def _concept_kwargs(
        args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        self = args[0]
        config = args[2]
        dataset = args[3]
        metrics = getattr(self, "metrics", None)
        dataset_size = len(getattr(dataset, "examples", []) or [])
        return _build_concept_payload(
            "CONC-Layer-Core",
            ["REQ-EVAL-005"],
            ["FUNC-EVAL-METRICS"],
            ["SYNC-OptimizationFlow"],
            {
                "metrics": metrics,
                "dataset_size": dataset_size,
                "config_keys": (
                    sorted(list(config.keys())) if isinstance(config, dict) else None
                ),
            },
        )

    def _event_details(result: Any) -> dict[str, Any]:
        return {
            "action": "evaluation_complete",
            "successful_examples": getattr(result, "successful_examples", None),
            "total_examples": getattr(result, "total_examples", None),
        }

    _safe_patch(
        local_evaluators.LocalEvaluator,
        "evaluate",
        lambda fn: _wrap_async(
            fn,
            concept_kwargs_factory=_concept_kwargs,
            event_details_factory=_event_details,
        ),
    )


def _patch_persistence() -> None:
    try:
        from traigent.utils.persistence import PersistenceManager
    except Exception:
        return

    def _wrap_save(fn):
        def wrapper(*args: Any, **kwargs: Any):
            try:
                payload = _build_concept_payload(
                    "CONC-Layer-Infra",
                    ["REQ-STOR-007"],
                    ["FUNC-STORAGE"],
                    ["SYNC-StorageLogging"],
                    {
                        "operation": "save_result",
                        "base_dir": (
                            str(getattr(args[0], "base_dir", None)) if args else None
                        ),
                        "name": (
                            kwargs.get("name")
                            if kwargs.get("name")
                            else (args[2] if len(args) > 2 else None)
                        ),
                    },
                )
                _emit_record(
                    "concept_action",
                    {
                        "concept_id": payload.get("concept_id"),
                        "action_name": "save_result",
                        "inputs": payload,
                    },
                )
            except Exception:
                pass
            result = fn(*args, **kwargs)
            try:
                _emit_record(
                    "event",
                    {
                        "action_name": "storage_save",
                        "inputs": {"action": "storage_save"},
                    },
                )
            except Exception:
                pass
            return result

        return wrapper

    def _wrap_load(fn):
        def wrapper(*args: Any, **kwargs: Any):
            try:
                payload = _build_concept_payload(
                    "CONC-Layer-Infra",
                    ["REQ-STOR-007"],
                    ["FUNC-STORAGE"],
                    ["SYNC-StorageLogging"],
                    {
                        "operation": "load_result",
                        "base_dir": (
                            str(getattr(args[0], "base_dir", None)) if args else None
                        ),
                        "name": (
                            kwargs.get("name")
                            if kwargs.get("name")
                            else (args[1] if len(args) > 1 else None)
                        ),
                    },
                )
                _emit_record(
                    "concept_action",
                    {
                        "concept_id": payload.get("concept_id"),
                        "action_name": "load_result",
                        "inputs": payload,
                    },
                )
            except Exception:
                pass
            return fn(*args, **kwargs)

        return wrapper

    _safe_patch(PersistenceManager, "save_result", _wrap_save)
    _safe_patch(PersistenceManager, "load_result", _wrap_load)


def _patch_security() -> None:
    try:
        from traigent.security.headers import SecurityHeadersMiddleware
    except Exception:
        return

    def _concept_kwargs(
        args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        self = args[0]
        return _build_concept_payload(
            "CONC-Layer-Infra",
            ["REQ-SEC-010"],
            ["FUNC-SECURITY"],
            ["SYNC-CloudHybrid"],
            {
                "operation": "apply_headers",
                "enable_cors": getattr(self, "enable_cors", None),
                "enable_csp": getattr(self, "enable_csp", None),
                "enable_hsts": getattr(self, "enable_hsts", None),
            },
        )

    _safe_patch(
        SecurityHeadersMiddleware,
        "apply_headers",
        lambda fn: _wrap_sync_dynamic(fn, concept_kwargs_factory=_concept_kwargs),
    )


def _patch_analytics() -> None:
    try:
        from traigent.utils.local_analytics import LocalAnalytics
    except Exception:
        return

    def _concept_kwargs(
        args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        self = args[0]
        config = getattr(self, "config", None)
        exec_mode = getattr(config, "execution_mode", None) if config else None
        enabled = getattr(self, "enabled", None)
        return _build_concept_payload(
            "CONC-Layer-CrossCutting",
            ["REQ-ANLY-011"],
            ["FUNC-ANALYTICS"],
            ["SYNC-Observability"],
            {
                "operation": "collect_usage_stats",
                "execution_mode": exec_mode,
                "enabled": enabled,
            },
        )

    _safe_patch(
        LocalAnalytics,
        "collect_usage_stats",
        lambda fn: _wrap_sync_dynamic(fn, concept_kwargs_factory=_concept_kwargs),
    )


def _patch_tvl() -> None:
    try:
        import traigent.tvl.spec_loader as tvl_loader
    except Exception:
        return

    def _concept_kwargs(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        spec_path = (
            kwargs.get("spec_path")
            if kwargs.get("spec_path")
            else (args[0] if args else None)
        )
        environment = kwargs.get("environment")
        return _build_concept_payload(
            "CONC-Layer-Core",
            ["REQ-TVLSPEC-012"],
            ["FUNC-TVLSPEC"],
            ["SYNC-OptimizationFlow"],
            {
                "spec_path": str(spec_path) if spec_path else None,
                "environment": environment,
            },
        )

    _safe_patch(
        tvl_loader,
        "load_tvl_spec",
        lambda fn: _wrap_sync_dynamic(fn, concept_kwargs_factory=_concept_kwargs),
    )


def _patch_tvl() -> None:
    try:
        import traigent.tvl.spec_loader as tvl_loader
    except Exception:
        return

    def _concept_kwargs(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        spec_path = (
            kwargs.get("spec_path")
            if kwargs.get("spec_path")
            else (args[0] if args else None)
        )
        environment = kwargs.get("environment")
        return _build_concept_payload(
            "CONC-Layer-Core",
            ["REQ-TVLSPEC-012"],
            ["FUNC-TVLSPEC"],
            ["SYNC-OptimizationFlow"],
            {
                "spec_path": str(spec_path) if spec_path else None,
                "environment": environment,
            },
        )

    _safe_patch(
        tvl_loader,
        "load_tvl_spec",
        lambda fn: _wrap_sync_dynamic(fn, concept_kwargs_factory=_concept_kwargs),
    )


def _patch_cloud_auth() -> None:
    try:
        from traigent.cloud.auth import AuthManager
    except Exception:
        return

    def _concept_kwargs(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        api_key = (
            kwargs.get("api_key")
            if kwargs.get("api_key")
            else (args[1] if len(args) > 1 else None)
        )
        return _build_concept_payload(
            "CONC-Layer-Infra",
            ["REQ-CLOUD-009"],
            ["FUNC-CLOUD-HYBRID"],
            ["SYNC-CloudHybrid"],
            {
                "operation": "set_api_key",
                "api_key_provided": bool(api_key),
            },
        )

    _safe_patch(
        AuthManager,
        "_set_api_key_token",
        lambda fn: _wrap_sync_dynamic(fn, concept_kwargs_factory=_concept_kwargs),
    )


def _patch_optimizers() -> None:
    try:
        import traigent.optimizers.base as opt_base
    except Exception:
        return

    def _concept_kwargs(
        args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        self = args[0]
        history = args[1] if len(args) > 1 else []
        config_space = getattr(self, "config_space", None)
        objectives = getattr(self, "objectives", None)
        return _build_concept_payload(
            "CONC-Layer-Core",
            ["REQ-OPT-ALG-004"],
            ["FUNC-OPT-ALGORITHMS"],
            ["SYNC-OptimizationFlow"],
            {
                "history_len": len(history) if history is not None else 0,
                "config_space_keys": (
                    list(config_space.keys())
                    if isinstance(config_space, dict)
                    else None
                ),
                "objectives": objectives,
            },
        )

    _safe_patch(
        opt_base.BaseOptimizer,
        "generate_candidates",
        lambda fn: _wrap_sync_dynamic(fn, concept_kwargs_factory=_concept_kwargs),
    )


def _patch_integrations() -> None:
    try:
        import traigent.utils.langchain_interceptor as lc_interceptor
    except Exception:
        return

    def _concept_kwargs(
        _args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        return _build_concept_payload(
            "CONC-Layer-Integration",
            ["REQ-INT-008"],
            ["FUNC-INTEGRATIONS"],
            ["SYNC-IntegrationHook"],
            {
                "operation": kwargs.get("operation") or "capture_response",
            },
        )

    _safe_patch(
        lc_interceptor,
        "capture_langchain_response",
        lambda fn: _wrap_sync_dynamic(fn, concept_kwargs_factory=_concept_kwargs),
    )
    _safe_patch(
        lc_interceptor,
        "clear_captured_responses",
        lambda fn: _wrap_sync_dynamic(fn, concept_kwargs_factory=_concept_kwargs),
    )


def enable_trace_sync() -> bool:
    """Enable TraceSync runtime instrumentation by patching entrypoints."""
    if getattr(enable_trace_sync, "_enabled", False):
        return TRACE_RUNTIME_AVAILABLE

    _patch_api_functions()
    _patch_optimize_decorator()
    _patch_optimized_function()
    _patch_orchestrator()
    _patch_invokers()
    _patch_evaluators()
    _patch_persistence()
    _patch_security()
    _patch_analytics()
    _patch_tvl()
    _patch_cloud_auth()
    _patch_optimizers()
    _patch_integrations()

    enable_trace_sync._enabled = True  # type: ignore[attr-defined]
    return TRACE_RUNTIME_AVAILABLE


def _should_enable_from_env() -> bool:
    return os.getenv(ENABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    if _should_enable_from_env():
        enabled = enable_trace_sync()
        if not enabled:
            print(
                "TraceSync runtime not available; instrumentation running in no-op mode."
            )
    else:
        print(f"{ENABLE_ENV} not set; skipping TraceSync patching.")
