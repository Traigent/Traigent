#!/usr/bin/env python3
"""Demo: optimize a method and observe every trial plus post-optimization runs.

This script demonstrates the recommended composition:

    @traigent.optimize(...)
    @observe(...)
    def my_method(...):
        ...

It supports:
- mock mode for fast local experimentation
- real mode using OpenAI, Anthropic, or Gemini keys
- optimization scales from tiny to large
- backend observability when TRAIGENT_BACKEND_URL and TRAIGENT_API_KEY are set
- in-memory observability fallback when backend credentials are absent

Examples:
    python walkthrough/demo/optimize_and_observe.py --mode mock --scale tiny
    python walkthrough/demo/optimize_and_observe.py --mode mock --scale medium
    python walkthrough/demo/optimize_and_observe.py --mode real --scale small

Recommended local backend setup:
    export TRAIGENT_BACKEND_URL="http://localhost:5001"
    export TRAIGENT_API_KEY="your-local-api-key"  # pragma: allowlist secret
    export TRAIGENT_PROJECT_ID="your-project-id"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WALKTHROUGH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WALKTHROUGH_ROOT))

from utils.helpers import (
    print_cost_estimate,
    print_optimization_config,
    print_results_table,
    sanitize_traigent_api_key,
)
from utils.mock_answers import (
    ANSWERS,
    DEFAULT_MOCK_MODEL,
    get_mock_accuracy,
    get_mock_latency,
    normalize_text,
    set_mock_model,
)
from utils.scoring import token_match_score

import traigent
from traigent import TraigentConfig
from traigent.config.context import get_trial_context
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.observability import (
    ObservabilityClient,
    ObservabilityConfig,
    observe,
    set_default_observability_client,
)

DATASETS = Path(__file__).resolve().parent.parent / "datasets"
DEFAULT_DATASET_PATH = DATASETS / "simple_questions.jsonl"

ALL_MODEL_CANDIDATES = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-sonnet-4-20250514",
    "gemini-2.0-flash",
    "gemini-1.5-pro-latest",
]

OBJECTIVES = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.60),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.25),
        ObjectiveDefinition("latency", orientation="minimize", weight=0.15),
    ]
)


@dataclass(frozen=True)
class ScalePreset:
    dataset_examples: int
    max_trials: int
    algorithm: str
    max_models: int
    temperatures: tuple[float, ...]
    response_styles: tuple[str, ...]
    post_runs: int


SCALE_PRESETS: dict[str, ScalePreset] = {
    "tiny": ScalePreset(
        dataset_examples=4,
        max_trials=4,
        algorithm="grid",
        max_models=2,
        temperatures=(0.0, 0.3),
        response_styles=("direct",),
        post_runs=2,
    ),
    "small": ScalePreset(
        dataset_examples=6,
        max_trials=8,
        algorithm="grid",
        max_models=3,
        temperatures=(0.0, 0.3),
        response_styles=("direct", "reasoned"),
        post_runs=2,
    ),
    "medium": ScalePreset(
        dataset_examples=10,
        max_trials=12,
        algorithm="random",
        max_models=4,
        temperatures=(0.0, 0.3, 0.7),
        response_styles=("direct", "reasoned"),
        post_runs=3,
    ),
    "large": ScalePreset(
        dataset_examples=20,
        max_trials=20,
        algorithm="random",
        max_models=5,
        temperatures=(0.0, 0.3, 0.7),
        response_styles=("direct", "reasoned", "bullet"),
        post_runs=4,
    ),
}


@dataclass(frozen=True)
class RuntimeSettings:
    mode: str
    scale_name: str
    scale: ScalePreset
    observability_mode: str
    trace_name: str
    environment: str
    tags: tuple[str, ...]
    post_runs: int
    models: tuple[str, ...]
    config_space: dict[str, list[Any]]
    eval_dataset: list[dict[str, Any]]


class InMemoryTraceCollector:
    """Collect latest trace snapshots when backend delivery is not configured."""

    def __init__(self) -> None:
        self._latest_traces: dict[str, dict[str, Any]] = {}

    def send(self, traces: list[dict[str, Any]]) -> dict[str, Any]:
        for trace in traces:
            trace_id = str(trace.get("id", ""))
            if trace_id:
                self._latest_traces[trace_id] = trace
        return {"accepted": len(traces)}

    @staticmethod
    def _walk_observations(trace: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        stack = list(trace.get("observations", []))
        while stack:
            current = stack.pop(0)
            items.append(current)
            stack[0:0] = current.get("children", [])
        return items

    def summarize(self) -> dict[str, Any]:
        traces = list(self._latest_traces.values())
        observation_total = 0
        optimization_traces = 0
        applied_config_traces = 0

        for trace in traces:
            observations = self._walk_observations(trace)
            observation_total += len(observations)
            phases = {
                str((obs.get("output_data") or {}).get("phase"))
                for obs in observations
                if obs.get("name") == "record-execution-context"
            }
            if "optimization-trial" in phases:
                optimization_traces += 1
            if "applied-config-run" in phases:
                applied_config_traces += 1

        return {
            "trace_count": len(traces),
            "observation_count": observation_total,
            "optimization_traces": optimization_traces,
            "applied_config_traces": applied_config_traces,
            "trace_ids": sorted(self._latest_traces.keys()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize an observed Traigent method in mock or real mode."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "mock", "real"),
        default="auto",
        help="Execution mode. 'auto' prefers real mode when provider keys exist.",
    )
    parser.add_argument(
        "--scale",
        choices=tuple(SCALE_PRESETS.keys()),
        default="small",
        help="Optimization scale preset.",
    )
    parser.add_argument(
        "--observability",
        choices=("auto", "backend", "memory"),
        default="auto",
        help="Where observe() traces should be sent.",
    )
    parser.add_argument(
        "--post-runs",
        type=int,
        default=None,
        help="Override the number of direct post-optimization runs.",
    )
    return parser.parse_args()


def provider_for_model(model: str) -> str:
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "google"
    return "unknown"


def available_real_models() -> list[str]:
    available: list[str] = []
    if os.getenv("OPENAI_API_KEY"):
        available.extend(["gpt-4o-mini", "gpt-4o"])
    if os.getenv("ANTHROPIC_API_KEY"):
        available.append("claude-sonnet-4-20250514")
    if os.getenv("GOOGLE_API_KEY"):
        available.extend(["gemini-2.0-flash", "gemini-1.5-pro-latest"])
    return available


def resolve_mode(requested_mode: str) -> str:
    real_models = available_real_models()
    if requested_mode == "mock":
        return "mock"
    if requested_mode == "real":
        if not real_models:
            raise SystemExit(
                "Real mode requested but no provider keys are available. "
                "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY."
            )
        return "real"
    if real_models:
        return "real"
    print(
        "No provider key detected, switching to mock mode. "
        "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY for real mode."
    )
    return "mock"


def load_eval_dataset(limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with DEFAULT_DATASET_PATH.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            raw = json.loads(line)
            items.append(
                {
                    "input": dict(raw["input"]),
                    "expected_output": raw["output"],
                }
            )
    return items


def build_runtime(args: argparse.Namespace) -> RuntimeSettings:
    mode = resolve_mode(args.mode)
    scale = SCALE_PRESETS[args.scale]
    if mode == "mock":
        models = ALL_MODEL_CANDIDATES[: scale.max_models]
    else:
        models = available_real_models()[: scale.max_models]
        if not models:
            raise SystemExit("No real models are available for real mode.")

    post_runs = args.post_runs if args.post_runs is not None else scale.post_runs
    trace_name = f"walkthrough-optimize-observe-{mode}-{args.scale}"
    config_space = {
        "model": list(models),
        "temperature": list(scale.temperatures),
        "response_style": list(scale.response_styles),
    }
    return RuntimeSettings(
        mode=mode,
        scale_name=args.scale,
        scale=scale,
        observability_mode=args.observability,
        trace_name=trace_name,
        environment=f"walkthrough-{mode}",
        tags=("walkthrough", "optimize-observe", mode, f"scale:{args.scale}"),
        post_runs=max(0, post_runs),
        models=tuple(models),
        config_space=config_space,
        eval_dataset=load_eval_dataset(scale.dataset_examples),
    )


def explicit_backend_observability_available() -> bool:
    return bool(
        (os.getenv("TRAIGENT_BACKEND_URL") or os.getenv("TRAIGENT_API_URL"))
        and os.getenv("TRAIGENT_API_KEY")
    )


def should_use_backend_observability(requested_mode: str) -> bool:
    if requested_mode == "backend":
        return True
    if requested_mode == "auto" and explicit_backend_observability_available():
        return True
    return False


def configure_runtime(
    runtime: RuntimeSettings,
    *,
    use_backend_observability: bool,
) -> None:
    sanitize_traigent_api_key()
    os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")
    if not use_backend_observability:
        os.environ["TRAIGENT_OFFLINE_MODE"] = "true"
    elif not os.getenv("TRAIGENT_API_KEY"):
        os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")
    if runtime.mode == "mock":
        os.environ["TRAIGENT_MOCK_LLM"] = "true"
        os.environ.setdefault("OPENAI_API_KEY", "mock-key-for-demos")
        print(
            "MOCK MODE: Running optimize_and_observe.py with simulated LLM "
            "responses and real Traigent optimization/observability flow. "
            "Use --mode real with provider keys for live model calls."
        )
    else:
        os.environ.pop("TRAIGENT_MOCK_LLM", None)
    traigent.configure(
        logging_level=os.getenv("TRAIGENT_LOG_LEVEL", "WARNING").upper()
    )
    traigent.initialize(
        config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True)
    )


def build_observability_client(
    runtime: RuntimeSettings,
) -> tuple[ObservabilityClient, str, InMemoryTraceCollector | None]:
    collector: InMemoryTraceCollector | None = None
    explicit_backend = explicit_backend_observability_available()

    requested = runtime.observability_mode
    use_backend = requested == "backend" or (
        requested == "auto" and explicit_backend
    )

    if use_backend and not explicit_backend:
        raise SystemExit(
            "Backend observability requested but TRAIGENT_BACKEND_URL or "
            "TRAIGENT_API_KEY is missing."
        )

    if use_backend:
        client = ObservabilityClient(
            ObservabilityConfig(default_environment=runtime.environment)
        )
        return client, "backend", None

    collector = InMemoryTraceCollector()
    client = ObservabilityClient(
        ObservabilityConfig(default_environment=runtime.environment),
        sender=collector.send,
    )
    return client, "memory", collector


def build_prompt(question: str, response_style: str) -> str:
    base = (
        "Answer with the final answer only. Do not ask follow-up questions. "
        "Keep it concise and preserve the question's terminology. "
        "If the answer is numeric, use digits only, no commas, no units."
    )
    if response_style == "reasoned":
        return f"{base} Think silently first, then answer.\nQuestion: {question}"
    if response_style == "bullet":
        return f"{base} Return one bullet containing the answer.\nQuestion: {question}"
    return f"{base}\nQuestion: {question}"


def render_mock_answer(question: str, response_style: str) -> str:
    answer = ANSWERS.get(normalize_text(question), "I don't know")
    if response_style == "reasoned":
        return f"Reasoning complete. Final answer: {answer}"
    if response_style == "bullet":
        return f"- {answer}"
    return answer


def score_response(
    output: str,
    expected: str | None = None,
    expected_output: str | None = None,
    config: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    resolved_expected = expected_output if expected_output is not None else expected
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        config = config or {}
        model = str(config.get("model", DEFAULT_MOCK_MODEL))
        temperature = float(config.get("temperature", 0.0) or 0.0)
        response_style = str(config.get("response_style", "direct"))
        style_bonus = {
            "direct": 0.00,
            "reasoned": 0.02,
            "bullet": -0.01,
        }.get(response_style, 0.0)
        use_cot = response_style == "reasoned"
        accuracy = get_mock_accuracy(
            model,
            task_type="simple_qa",
            temperature=temperature,
            use_cot=use_cot,
        )
        return max(0.0, min(1.0, accuracy + style_bonus))
    return token_match_score(output, resolved_expected)


def create_llm_client(model: str, temperature: float) -> Any:
    provider = provider_for_model(model)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    raise ValueError(f"Unsupported provider for model '{model}'")


def create_demo_agent(runtime: RuntimeSettings, client: ObservabilityClient):
    @observe(name="record-execution-context", client=client)
    def record_execution_context(
        phase: str,
        model: str,
        temperature: float,
        response_style: str,
        trial_id: str | None,
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "model": model,
            "temperature": temperature,
            "response_style": response_style,
            "trial_id": trial_id,
        }

    @observe(name="prepare-prompt", client=client)
    def prepare_prompt(question: str, response_style: str) -> str:
        return build_prompt(question, response_style)

    @observe(name="mock-llm-call", client=client)
    def invoke_mock(question: str, model: str, response_style: str) -> str:
        set_mock_model(model)
        time.sleep(get_mock_latency(model, "simple_qa") * 0.01)
        return render_mock_answer(question, response_style)

    @observe(name="real-llm-call", client=client)
    def invoke_real(
        prompt: str,
        model: str,
        temperature: float,
        response_style: str,
    ) -> str:
        llm = create_llm_client(model, temperature)
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
        objectives=OBJECTIVES,
        scoring_function=score_response,
        configuration_space=runtime.config_space,
        injection_mode="context",
        execution_mode="edge_analytics",
        mock_mode_config={
            "base_accuracy": 0.85,
            "variance": 0.0,
            "random_seed": 42,
        },
    )
    @observe(
        name=runtime.trace_name,
        client=client,
        environment=runtime.environment,
        tags=list(runtime.tags),
        metadata={
            "demo": "optimize-and-observe",
            "scale": runtime.scale_name,
            "mode": runtime.mode,
        },
    )
    def answer_question(question: str) -> str:
        config = traigent.get_config()
        model = str(config.get("model", DEFAULT_MOCK_MODEL))
        temperature = float(config.get("temperature", 0.0) or 0.0)
        response_style = str(config.get("response_style", "direct"))
        trial_ctx = get_trial_context() or {}
        trial_id = (
            str(trial_ctx.get("trial_id"))
            if trial_ctx.get("trial_id") is not None
            else None
        )
        phase = "optimization-trial" if trial_id else "applied-config-run"

        record_execution_context(
            phase=phase,
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


def print_runtime_summary(runtime: RuntimeSettings, obs_mode: str) -> None:
    print("Traigent Demo: Optimize and Observe")
    print("=" * 50)
    print(f"Mode: {runtime.mode}")
    print(f"Scale: {runtime.scale_name}")
    print(f"Observability sink: {obs_mode}")
    print(f"Trace name: {runtime.trace_name}")
    print(f"Dataset size: {len(runtime.eval_dataset)} examples")
    print(f"Post-optimization runs: {runtime.post_runs}")
    print_optimization_config(OBJECTIVES, runtime.config_space)
    if runtime.mode == "real":
        print_cost_estimate(
            models=list(runtime.models),
            dataset_size=len(runtime.eval_dataset),
            task_type="simple_qa",
            num_trials=runtime.scale.max_trials,
        )


def print_mock_scaling_hint() -> None:
    print("\nSuggested mock-mode scaling runs:")
    print("  python walkthrough/demo/optimize_and_observe.py --mode mock --scale tiny")
    print("  python walkthrough/demo/optimize_and_observe.py --mode mock --scale small")
    print("  python walkthrough/demo/optimize_and_observe.py --mode mock --scale medium")
    print("  python walkthrough/demo/optimize_and_observe.py --mode mock --scale large")


async def main() -> None:
    args = parse_args()
    runtime = build_runtime(args)
    use_backend_observability = should_use_backend_observability(
        runtime.observability_mode
    )
    configure_runtime(
        runtime,
        use_backend_observability=use_backend_observability,
    )
    client, obs_mode, collector = build_observability_client(runtime)
    set_default_observability_client(client)

    print_runtime_summary(runtime, obs_mode)
    if runtime.mode == "mock":
        print_mock_scaling_hint()

    agent = create_demo_agent(runtime, client)

    try:
        optimization_started_at = time.perf_counter()
        results = await agent.optimize(
            algorithm=runtime.scale.algorithm,
            max_trials=runtime.scale.max_trials,
            show_progress=True,
            random_seed=42,
        )
        optimization_elapsed = time.perf_counter() - optimization_started_at

        print_results_table(
            results,
            runtime.config_space,
            OBJECTIVES,
            is_mock=runtime.mode == "mock",
            task_type="simple_qa",
            dataset_size=len(runtime.eval_dataset),
        )

        print("\nBest configuration:")
        print(f"  model: {results.best_config.get('model')}")
        print(f"  temperature: {results.best_config.get('temperature')}")
        print(f"  response_style: {results.best_config.get('response_style')}")
        print("\nBest metrics:")
        print(f"  accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
        print(f"  cost: ${results.best_metrics.get('cost', 0):.6f}")
        print(f"  latency: {results.best_metrics.get('latency', 0):.3f}s")
        print(f"\nOptimization runtime: {optimization_elapsed:.2f}s")

        optimize_flush = client.flush()
        print(
            "\nObservability flush after optimization: "
            f"success={optimize_flush.success} sent={optimize_flush.items_sent} "
            f"pending={optimize_flush.items_pending}"
        )
        if optimize_flush.errors:
            print(f"  errors: {optimize_flush.errors}")

        agent.apply_best_config(results)

        print("\nPost-optimization runs with applied best config:")
        for example in runtime.eval_dataset[: runtime.post_runs]:
            question = str(example["input"]["question"])
            answer = agent(question)
            print(f"  Q: {question}")
            print(f"  A: {answer}")

        post_flush = client.flush()
        print(
            "\nObservability flush after post-optimization runs: "
            f"success={post_flush.success} sent={post_flush.items_sent} "
            f"pending={post_flush.items_pending}"
        )
        if post_flush.errors:
            print(f"  errors: {post_flush.errors}")

        if collector is not None:
            summary = collector.summarize()
            print("\nIn-memory observability summary:")
            print(f"  traces captured: {summary['trace_count']}")
            print(f"  observations captured: {summary['observation_count']}")
            print(f"  optimization traces: {summary['optimization_traces']}")
            print(f"  applied-config traces: {summary['applied_config_traces']}")
            if summary["trace_ids"]:
                preview = ", ".join(summary["trace_ids"][:5])
                print(f"  trace ids: {preview}")
        else:
            print("\nBackend observability delivery was enabled.")
            print("Search for these filters in the frontend observability pages:")
            print(f"  trace name: {runtime.trace_name}")
            print(f"  environment: {runtime.environment}")
            print(f"  tags: {', '.join(runtime.tags)}")
    finally:
        close_result = client.close()
        if not close_result.success and close_result.errors:
            print(f"Observability close warnings: {close_result.errors}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
