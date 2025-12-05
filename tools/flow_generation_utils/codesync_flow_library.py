"""CodeSync-prefixed flow library for runtime tracing demos.

These flows live outside SDK/tests and can be run via tools/run_trace_flows.py.
Each flow is small and deterministic to rapidly exercise instrumentation.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, Iterable

from traigent.api import functions as api_functions
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.invokers.local import LocalInvoker
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.optimizers.base import BaseOptimizer
from traigent.config.types import TraigentConfig
from traigent.api.types import TrialResult, TrialStatus
from traigent.utils.persistence import PersistenceManager
from traigent.api.types import OptimizationResult, OptimizationStatus
from datetime import datetime, timezone
from pathlib import Path

Flow = Callable[[], Any | Coroutine[Any, Any, Any]]


def flow_api_config_init() -> None:
    """Hit API configure + initialize entrypoints."""
    api_functions.configure(logging_level="WARNING", parallel_workers=1)
    api_functions.initialize(api_key=None, api_url=None)


async def flow_invoker_basic() -> None:
    """Single invocation through LocalInvoker."""

    def demo_function(config: dict, text: str) -> str:
        return f"{text} | model={config.get('model', 'default')}"

    invoker = LocalInvoker(injection_mode="parameter", config_param="config")
    config = {"model": "demo-model", "temperature": 0.5}
    input_data = {"text": "hello flow"}

    await invoker.invoke(demo_function, config, input_data)


async def flow_invoker_batch() -> None:
    """Batch invocation path."""

    def demo_function(config: dict, text: str) -> str:
        return f"{text} | model={config.get('model', 'default')}"

    invoker = LocalInvoker(injection_mode="parameter", config_param="config")
    config = {"model": "demo-model", "temperature": 0.1}
    batch = [
        {"text": "batch-1"},
        {"text": "batch-2"},
    ]
    await invoker.invoke_batch(demo_function, config, batch)


async def flow_evaluator_basic() -> None:
    """Evaluate a simple function over a tiny dataset to emit evaluator traces."""

    def scored_function(text: str) -> str:
        return text.upper()

    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"text": "foo"}, expected_output="FOO"),
            EvaluationExample(input_data={"text": "bar"}, expected_output="BAR"),
        ],
        name="codesync-demo",
        description="Tiny dataset for trace demo",
    )

    evaluator = LocalEvaluator(metrics=["accuracy"], timeout=5.0, detailed=True)
    await evaluator.evaluate(scored_function, config={}, dataset=dataset)


class _DummyOptimizer(BaseOptimizer):
    """Minimal optimizer to drive orchestrator flow."""

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        return {"candidate": len(history) + 1}

    def should_stop(self, history: list[TrialResult]) -> bool:
        return len(history) >= 1

    def tell(self, config: dict[str, Any], result: TrialResult) -> None:  # noqa: D401
        """Optional hook; no-op for dummy."""
        return None


async def flow_orchestrator_stub() -> None:
    """Run a tiny orchestrator loop (max_trials=1) to hit ORCH/EVAL traces."""

    def objective_func(sample: dict[str, Any]) -> str:
        return sample.get("text", "").upper()

    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"text": "orch"}, expected_output="ORCH"),
        ],
        name="orch-demo",
        description="Tiny orchestrator dataset",
    )

    evaluator = LocalEvaluator(metrics=["accuracy"], timeout=5.0, detailed=True)
    optimizer = _DummyOptimizer(
        config_space={"candidate": [1]}, objectives=["accuracy"]
    )
    config = TraigentConfig.edge_analytics_mode()

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=1,
        config=config,
        objectives=["accuracy"],
    )

    await orchestrator.optimize(
        func=objective_func, dataset=dataset, function_name="orch_func"
    )


def flow_storage_persistence() -> None:
    """Exercise PersistenceManager save/load for storage traces."""

    now = datetime.now(timezone.utc)
    trial = TrialResult(
        trial_id="trial-1",
        config={"candidate": 1},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=now,
        metadata={},
    )
    result = OptimizationResult(
        trials=[trial],
        best_config={"candidate": 1},
        best_score=1.0,
        optimization_id="demo-opt",
        duration=0.1,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="dummy",
        timestamp=now,
        metadata={"function_name": "storage_demo"},
    )

    pm = PersistenceManager(base_dir=Path("runtime") / "storage_demo")
    pm.save_result(result, name="storage_trace")


def flow_security_headers() -> None:
    """Apply security headers to a dummy response object."""
    from traigent.security.headers import SecurityHeadersMiddleware

    class DummyResponse:
        def __init__(self) -> None:
            self.headers = {}

    middleware = SecurityHeadersMiddleware(
        enable_cors=True, allowed_origins=["https://example.com"]
    )
    resp = DummyResponse()
    middleware.apply_headers(
        resp,
        request=type(
            "Req",
            (),
            {"headers": {"Origin": "https://example.com"}, "path": "/api/login"},
        )(),
    )


def flow_analytics_local() -> None:
    """Collect local analytics stats to emit analytics trace."""
    from traigent.utils.local_analytics import LocalAnalytics

    config = TraigentConfig.edge_analytics_mode()
    config.enable_usage_analytics = False  # keep no-op but still trace
    analytics = LocalAnalytics(config)
    analytics.collect_usage_stats()


def flow_integration_langchain() -> None:
    """Trigger integration hooks for LangChain interceptor."""
    from traigent.utils import langchain_interceptor as lc

    lc.capture_langchain_response({"content": "hello", "usage": {"tokens": 1}})
    lc.get_all_captured_responses()
    lc.clear_captured_responses()


def flow_tvl_spec_load() -> None:
    """Load a tiny TVL spec from a temp file to emit TVL traces."""
    import yaml
    from traigent.tvl.spec_loader import load_tvl_spec

    spec_path = Path("runtime") / "tvl_demo.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)

    spec = {
        "configuration_space": {
            "temperature": {
                "type": "float",
                "range": {"min": 0.1, "max": 0.5},
            },
        },
        "objectives": [
            {"name": "accuracy", "orientation": "maximize"},
        ],
        "optimization": {"max_trials": 1},
    }

    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    load_tvl_spec(spec_path=spec_path)


def flow_cloud_auth_api_key() -> None:
    """Configure AuthManager with a dummy API key to emit cloud trace."""
    from traigent.cloud.auth import AuthManager

    AuthManager(api_key="dummy_api_key_for_trace_sync_1234567890").get_api_key_preview()


def flow_optimizer_generate() -> None:
    """Call generate_candidates on a dummy optimizer to emit optimizer trace."""
    from traigent.optimizers.base import BaseOptimizer

    class DemoOptimizer(BaseOptimizer):
        def suggest_next_trial(self, history):
            return {"x": len(history)}

        def should_stop(self, history):
            return len(history) >= 1

    opt = DemoOptimizer(config_space={"x": [1, 2]}, objectives=["accuracy"])
    opt.generate_candidates(2)


CODE_SYNC_FLOWS: Dict[str, Flow] = {
    # API/CLI
    "api_config_init": flow_api_config_init,
    # Invokers
    "invoker_basic": flow_invoker_basic,
    "invoker_batch": flow_invoker_batch,
    # Evaluator
    "evaluator_basic": flow_evaluator_basic,
    # Orchestrator
    "orchestrator_stub": flow_orchestrator_stub,
    # Storage
    "storage_persistence": flow_storage_persistence,
    # Security
    "security_headers": flow_security_headers,
    # Analytics
    "analytics_local": flow_analytics_local,
    # Integrations
    "integration_langchain": flow_integration_langchain,
    # TVL/config
    "tvl_spec_load": flow_tvl_spec_load,
    # Cloud/Auth
    "cloud_auth_api_key": flow_cloud_auth_api_key,
    # Optimizer
    "optimizer_generate": flow_optimizer_generate,
}


def list_codesync_flows(selected: Iterable[str] | None = None) -> dict[str, Flow]:
    if not selected:
        return CODE_SYNC_FLOWS
    names = set(selected)
    return {name: fn for name, fn in CODE_SYNC_FLOWS.items() if name in names}
