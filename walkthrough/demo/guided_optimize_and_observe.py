#!/usr/bin/env python3
"""Guided phase runner for observing, optimizing, and re-running with best config.

This script is designed to be called by the interactive bash wrapper:

    walkthrough/demo/run_guided_optimize_and_observe_demo.sh

It supports three phases:
- baseline: emit observed traces using a known default config
- optimize: run optimization and persist results
- post: load persisted results, apply the best config, and emit observed traces
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import optimize_and_observe as shared

import traigent
from traigent.config.context import get_trial_context
from traigent.observability import (
    ObservabilityClient,
    observe,
    set_default_observability_client,
)

DEFAULT_BASELINE_CONFIG: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "response_style": "bullet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one guided demo phase for observe + optimize."
    )
    parser.add_argument(
        "--phase",
        choices=("baseline", "optimize", "post"),
        required=True,
        help="Which guided phase to execute.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Stable identifier shared across the three guided phases.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "mock", "real"),
        default="mock",
        help="Execution mode. Use mock first for quick FE validation.",
    )
    parser.add_argument(
        "--scale",
        choices=tuple(shared.SCALE_PRESETS.keys()),
        default="small",
        help="Optimization scale preset used for the optimization phase.",
    )
    parser.add_argument(
        "--observability",
        choices=("auto", "backend", "memory"),
        default="backend",
        help="Where observe() traces should be sent.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(SCRIPT_DIR / "artifacts" / "guided_optimize_observe"),
        help="Directory where per-run artifacts are stored.",
    )
    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=3,
        help="Number of direct observed runs to emit in the baseline phase.",
    )
    parser.add_argument(
        "--post-runs",
        type=int,
        default=3,
        help="Number of direct observed runs to emit in the post phase.",
    )
    parser.add_argument(
        "--frontend-url",
        default="http://localhost:3002",
        help="Frontend base URL printed in the phase summary.",
    )
    return parser.parse_args()


def make_run_dir(root: Path, run_id: str) -> Path:
    return root / run_id


def optimization_results_path(run_dir: Path) -> Path:
    return run_dir / "optimization_results.json"


def phase_summary_path(run_dir: Path, phase: str) -> Path:
    return run_dir / f"{phase}_summary.json"


def load_saved_optimization_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def get_saved_experiment_run_id(payload: dict[str, Any]) -> str | None:
    direct = as_optional_text(payload.get("experiment_run_id"))
    if direct:
        return direct

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None

    nested = as_optional_text(metadata.get("experiment_run_id"))
    if nested:
        return nested

    local_session_summary = metadata.get("local_session_summary")
    if isinstance(local_session_summary, dict):
        local_metadata = local_session_summary.get("metadata")
        if isinstance(local_metadata, dict):
            return as_optional_text(local_metadata.get("experiment_run_id"))
    return None


def build_runtime(args: argparse.Namespace, *, trace_name: str) -> shared.RuntimeSettings:
    runtime_args = argparse.Namespace(
        mode=args.mode,
        scale=args.scale,
        observability=args.observability,
        post_runs=0,
    )
    runtime = shared.build_runtime(runtime_args)
    tags = tuple(runtime.tags) + (
        "guided-demo",
        f"run:{args.run_id}",
        f"phase:{args.phase}",
    )
    return replace(
        runtime,
        trace_name=trace_name,
        environment=f"walkthrough-guided-{runtime.mode}",
        tags=tags,
        post_runs=0,
    )


def build_trace_name(phase: str, *, active_config: dict[str, Any] | None = None) -> str:
    if phase == "baseline":
        model = str((active_config or DEFAULT_BASELINE_CONFIG).get("model", "default"))
        return f"guided-optimize-observe-baseline-{model}"
    if phase == "optimize":
        return "guided-optimize-observe-optimization"
    model = str((active_config or {}).get("model", "best"))
    return f"guided-optimize-observe-post-best-{model}"


def select_questions(
    runtime: shared.RuntimeSettings,
    *,
    count: int,
) -> list[str]:
    return [
        str(example["input"]["question"])
        for example in runtime.eval_dataset[: max(0, count)]
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_phase_summary(
    *,
    args: argparse.Namespace,
    runtime: shared.RuntimeSettings,
    run_dir: Path,
    trace_name: str,
    active_config: dict[str, Any] | None,
    answers: list[dict[str, str]] | None,
    optimization_id: str | None,
    experiment_id: str | None,
    experiment_run_id: str | None,
    best_config: dict[str, Any] | None,
    best_metrics: dict[str, Any] | None,
    flush_success: bool,
    flush_errors: list[str],
) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "phase": args.phase,
        "mode": runtime.mode,
        "scale": runtime.scale_name,
        "trace_name": trace_name,
        "environment": runtime.environment,
        "tags": list(runtime.tags),
        "frontend_url": args.frontend_url,
        "frontend_observability_url": f"{args.frontend_url}/observability",
        "frontend_experiments_url": f"{args.frontend_url}/experiments",
        "frontend_experiment_url": (
            f"{args.frontend_url}/experiments/view/{experiment_id}"
            if experiment_id
            else None
        ),
        "optimization_results_path": str(optimization_results_path(run_dir)),
        "optimization_id": optimization_id,
        "experiment_id": experiment_id,
        "experiment_run_id": experiment_run_id,
        "active_config": active_config,
        "baseline_default_config": DEFAULT_BASELINE_CONFIG,
        "best_config": best_config,
        "best_metrics": best_metrics,
        "answers": answers or [],
        "flush_success": flush_success,
        "flush_errors": list(flush_errors),
    }


def create_guided_agent(
    runtime: shared.RuntimeSettings,
    client: ObservabilityClient,
    *,
    run_id: str,
    phase: str,
    trace_name: str,
    trace_metadata: dict[str, Any],
):
    @observe(name="record-execution-context", client=client)
    def record_execution_context(
        phase_label: str,
        config_source: str,
        model: str,
        temperature: float,
        response_style: str,
        trial_id: str | None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "phase": phase_label,
            "config_source": config_source,
            "model": model,
            "temperature": temperature,
            "response_style": response_style,
            "trial_id": trial_id,
        }

    @observe(name="prepare-prompt", client=client)
    def prepare_prompt(question: str, response_style: str) -> str:
        return shared.build_prompt(question, response_style)

    @observe(name="mock-llm-call", client=client)
    def invoke_mock(question: str, model: str, response_style: str) -> str:
        shared.set_mock_model(model)
        time.sleep(shared.get_mock_latency(model, "simple_qa") * 0.01)
        return shared.render_mock_answer(question, response_style)

    @observe(name="real-llm-call", client=client)
    def invoke_real(
        prompt: str,
        model: str,
        temperature: float,
        response_style: str,
    ) -> str:
        llm = shared.create_llm_client(model, temperature)
        response = llm.invoke(prompt)
        content = str(response.content)
        if response_style == "bullet" and not content.lstrip().startswith("-"):
            return f"- {content.strip()}"
        return content.strip()

    @observe(name="post-process-response", client=client)
    def post_process_response(output: str) -> str:
        return output.strip()

    @traigent.optimize(
        evaluation={"eval_dataset": runtime.eval_dataset},
        objectives=shared.OBJECTIVES,
        scoring_function=shared.score_response,
        configuration_space=runtime.config_space,
        default_config=DEFAULT_BASELINE_CONFIG,
        injection_mode="context",
        execution_mode="edge_analytics",
        mock_mode_config={
            "base_accuracy": 0.85,
            "variance": 0.0,
            "random_seed": 42,
        },
    )
    @observe(
        name=trace_name,
        client=client,
        session_id=f"guided-session:{run_id}:{phase}",
        user_id="guided-demo-user",
        custom_trace_id=f"guided-trace:{run_id}:{phase}",
        environment=runtime.environment,
        tags=list(runtime.tags),
        metadata=trace_metadata,
    )
    def answer_question(question: str) -> str:
        config = traigent.get_config()
        model = str(config.get("model", shared.DEFAULT_MOCK_MODEL))
        temperature = float(config.get("temperature", 0.0) or 0.0)
        response_style = str(config.get("response_style", "direct"))
        trial_ctx = get_trial_context() or {}
        trial_id = (
            str(trial_ctx.get("trial_id"))
            if trial_ctx.get("trial_id") is not None
            else None
        )

        if phase == "baseline":
            phase_label = "baseline-default-config"
            config_source = "default-config"
        elif phase == "post":
            phase_label = "post-best-config"
            config_source = "best-config"
        else:
            phase_label = "optimization-trial" if trial_id else "optimization-direct"
            config_source = "trial-config" if trial_id else "default-config"

        record_execution_context(
            phase_label=phase_label,
            config_source=config_source,
            model=model,
            temperature=temperature,
            response_style=response_style,
            trial_id=trial_id,
        )

        prompt = prepare_prompt(question, response_style)
        if runtime.mode == "mock":
            raw_output = invoke_mock(question, model, response_style)
        else:
            raw_output = invoke_real(prompt, model, temperature, response_style)
        return post_process_response(raw_output)

    return answer_question


def run_direct_calls(
    agent: Any,
    *,
    questions: list[str],
) -> list[dict[str, str]]:
    answers: list[dict[str, str]] = []
    for question in questions:
        answer = str(agent(question))
        print(f"Q: {question}")
        print(f"A: {answer}")
        answers.append({"question": question, "answer": answer})
    return answers


async def run_phase(args: argparse.Namespace) -> None:
    artifacts_dir = Path(args.artifacts_dir).resolve()
    run_dir = make_run_dir(artifacts_dir, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    persisted_results: dict[str, Any] | None = None
    best_config_for_post: dict[str, Any] | None = None
    persisted_optimization_id: str | None = None
    persisted_experiment_id: str | None = None
    persisted_experiment_run_id: str | None = None
    if args.phase == "post":
        results_path = optimization_results_path(run_dir)
        if not results_path.exists():
            raise SystemExit(
                f"Cannot run post phase because results file is missing: {results_path}"
            )
        persisted_results = load_saved_optimization_results(results_path)
        best_config_for_post = dict(persisted_results.get("best_config") or {})
        persisted_optimization_id = as_optional_text(persisted_results.get("optimization_id"))
        persisted_experiment_id = as_optional_text(persisted_results.get("experiment_id"))
        persisted_experiment_run_id = get_saved_experiment_run_id(persisted_results)

    trace_name = build_trace_name(args.phase, active_config=best_config_for_post)
    runtime = build_runtime(args, trace_name=trace_name)
    use_backend_observability = shared.should_use_backend_observability(
        runtime.observability_mode
    )
    shared.configure_runtime(
        runtime,
        use_backend_observability=use_backend_observability,
    )
    client, obs_mode, _collector = shared.build_observability_client(runtime)
    set_default_observability_client(client)

    summary_path = phase_summary_path(run_dir, args.phase)
    print(f"Guided demo phase: {args.phase}")
    print(f"Run id: {args.run_id}")
    print(f"Mode: {runtime.mode}")
    print(f"Scale: {runtime.scale_name}")
    print(f"Observability sink: {obs_mode}")
    print(f"Trace name: {trace_name}")
    print(f"Environment: {runtime.environment}")
    print(f"Results file: {optimization_results_path(run_dir)}")
    print(f"Phase summary: {summary_path}")

    trace_metadata: dict[str, Any] = {
        "demo": "guided-optimize-observe",
        "run_id": args.run_id,
        "phase": args.phase,
        "baseline_default_config": DEFAULT_BASELINE_CONFIG,
        "scale": runtime.scale_name,
        "mode": runtime.mode,
    }
    if args.phase == "baseline":
        trace_metadata["config_source"] = "default-config"
        trace_metadata["active_config"] = DEFAULT_BASELINE_CONFIG
    elif args.phase == "post" and best_config_for_post is not None:
        trace_metadata["config_source"] = "best-config"
        trace_metadata["active_config"] = best_config_for_post
        if persisted_optimization_id:
            trace_metadata["optimization_id"] = persisted_optimization_id
        if persisted_experiment_id:
            trace_metadata["experiment_id"] = persisted_experiment_id
        if persisted_experiment_run_id:
            trace_metadata["experiment_run_id"] = persisted_experiment_run_id
    else:
        trace_metadata["config_source"] = "trial-config"

    agent = create_guided_agent(
        runtime,
        client,
        run_id=args.run_id,
        phase=args.phase,
        trace_name=trace_name,
        trace_metadata=trace_metadata,
    )

    try:
        answers: list[dict[str, str]] | None = None
        optimization_id: str | None = persisted_optimization_id
        experiment_id: str | None = persisted_experiment_id
        experiment_run_id: str | None = persisted_experiment_run_id
        best_config: dict[str, Any] | None = None
        best_metrics: dict[str, Any] | None = None

        if args.phase == "baseline":
            agent.reset_optimization()
            agent.set_config(DEFAULT_BASELINE_CONFIG)
            print("\nBaseline config:")
            print(json.dumps(DEFAULT_BASELINE_CONFIG, indent=2, sort_keys=True))
            answers = run_direct_calls(
                agent,
                questions=select_questions(runtime, count=args.baseline_runs),
            )
        elif args.phase == "optimize":
            print("\nOptimization default config before search:")
            print(json.dumps(DEFAULT_BASELINE_CONFIG, indent=2, sort_keys=True))
            results = await agent.optimize(
                algorithm=runtime.scale.algorithm,
                max_trials=runtime.scale.max_trials,
                show_progress=True,
                random_seed=42,
            )
            optimization_id = as_optional_text(getattr(results, "optimization_id", None))
            experiment_id = as_optional_text(getattr(results, "experiment_id", None))
            experiment_run_id = get_saved_experiment_run_id(
                {
                    "experiment_run_id": getattr(results, "experiment_run_id", None),
                    "metadata": getattr(results, "metadata", {}) or {},
                }
            )
            best_config = dict(results.best_config or {})
            best_metrics = dict(results.best_metrics or {})
            agent.save_optimization_results(str(optimization_results_path(run_dir)))
            print("\nBest config:")
            print(json.dumps(best_config, indent=2, sort_keys=True))
            print("\nBest metrics:")
            print(json.dumps(best_metrics, indent=2, sort_keys=True))
            print(f"\nOptimization id: {optimization_id}")
        else:
            agent.load_optimization_results(str(optimization_results_path(run_dir)))
            agent.apply_best_config()
            best_config = dict(agent.current_config)
            print("\nApplied best config:")
            print(json.dumps(best_config, indent=2, sort_keys=True))
            answers = run_direct_calls(
                agent,
                questions=select_questions(runtime, count=args.post_runs),
            )

        flush = client.flush()
        print(
            "\nObservability flush: "
            f"success={flush.success} sent={flush.items_sent} "
            f"pending={flush.items_pending}"
        )
        if flush.errors:
            print(f"Flush errors: {flush.errors}")

        summary = build_phase_summary(
            args=args,
            runtime=runtime,
            run_dir=run_dir,
            trace_name=trace_name,
            active_config=(
                DEFAULT_BASELINE_CONFIG
                if args.phase == "baseline"
                else (best_config_for_post if args.phase == "post" else None)
            ),
            answers=answers,
            optimization_id=optimization_id,
            experiment_id=experiment_id,
            experiment_run_id=experiment_run_id,
            best_config=best_config,
            best_metrics=best_metrics,
            flush_success=flush.success,
            flush_errors=flush.errors,
        )
        write_json(summary_path, summary)

        if obs_mode == "backend" and not flush.success:
            error_message = "; ".join(flush.errors) or "unknown backend ingest failure"
            raise SystemExit(
                "Backend observability delivery failed. "
                f"Check TRAIGENT_API_KEY / backend auth and retry. Details: {error_message}"
            )
    finally:
        close_result = client.close()
        if not close_result.success and close_result.errors:
            print(f"Observability close warnings: {close_result.errors}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_phase(args))


if __name__ == "__main__":
    main()
