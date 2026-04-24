from __future__ import annotations

import argparse
import asyncio

from walkthrough.demo import optimize_and_observe as demo


def test_optimize_and_observe_tiny_demo_emits_one_session(monkeypatch) -> None:
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    runtime = demo.build_runtime(
        argparse.Namespace(
            mode="mock",
            scale="tiny",
            observability="memory",
            post_runs=1,
        )
    )
    demo.configure_runtime(runtime, use_backend_observability=False)
    client, obs_mode, collector = demo.build_observability_client(runtime)
    assert obs_mode == "memory"
    assert collector is not None
    demo.set_default_observability_client(client)
    agent = demo.create_demo_agent(runtime, client)

    async def run_demo_flow() -> object:
        results = await agent.optimize(
            algorithm=runtime.scale.algorithm,
            max_trials=runtime.scale.max_trials,
            show_progress=False,
            random_seed=42,
        )
        client.flush()
        agent.apply_best_config(results)
        for example in runtime.eval_dataset[: runtime.post_runs]:
            question = str(example["input"]["question"])
            agent(question)
        client.flush()
        return results

    try:
        results = asyncio.run(run_demo_flow())
        summary = collector.summarize()
    finally:
        client.close()

    expected_optimization_traces = runtime.scale.max_trials * len(runtime.eval_dataset)
    assert results.best_metrics["accuracy"] >= 0.85
    assert summary["trace_count"] == demo.expected_trace_count(runtime)
    assert summary["optimization_traces"] == expected_optimization_traces
    assert summary["applied_config_traces"] == runtime.post_runs
    assert summary["session_ids"] == [runtime.session_id]
