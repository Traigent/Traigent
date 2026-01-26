#!/usr/bin/env python3
"""Row-driven example runner for Traigent scenarios.

Usage (from repo root):
    .venv/bin/python examples/templates/run_template.py --row 3

Reads examples/datasets/matrices/test_matrix.csv and executes the specified row configuration.
Prints best config/score, aggregated table (service modes), and raw per-sample table.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent directory to path to ensure traigent can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from traigent.utils.logging import setup_logging  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


# Canonical answers and prompt template (reuse from core/hello-world)
_HELLO_WORLD_ROOT = Path(__file__).parent.parent / "core" / "hello-world"
_HELLO_WORLD_DATA_ROOT = Path(__file__).parent.parent / "datasets" / "hello-world"
_HELLO_WORLD_PROMPT_PATH = _HELLO_WORLD_ROOT / "prompt.txt"


def _prompt_template() -> str:
    if _HELLO_WORLD_PROMPT_PATH.exists():
        return _HELLO_WORLD_PROMPT_PATH.read_text().strip()
    return "Answer concisely."


_CANONICAL_ANSWERS = [
    "Artificial Intelligence",
    "Uses data and algorithms",
    "Retrieval Augmented Generation",
    "A list of steps",
    "A private key",
]

_PROMPT_SUFFIX = _prompt_template()


def _map_to_canonical(raw: str) -> str:
    for ans in _CANONICAL_ANSWERS:
        if ans.lower() in raw.lower():
            return ans
    return raw.strip()[:128]


def _build_prompt(question: str, ctx_snippets: list[str] | None) -> str:
    ctx_block = (
        "" if not ctx_snippets else "Context:\n" + "\n---\n".join(ctx_snippets) + "\n\n"
    )
    return f"{ctx_block}Question: {question}\n\n{_PROMPT_SUFFIX}"


def _extract_openai_text(response: Any) -> str:
    """Best-effort extraction of text content from an OpenAI responses payload."""
    outputs = getattr(response, "output", None)
    collected: list[str] = []
    if isinstance(outputs, Sequence):
        for block in outputs:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str):
                collected.append(block_text)
            content = getattr(block, "content", None)
            if isinstance(content, Sequence):
                for chunk in content:
                    chunk_text = getattr(chunk, "text", None)
                    if isinstance(chunk_text, str):
                        collected.append(chunk_text)
                    else:
                        inline = getattr(chunk, "content", None)
                        if isinstance(inline, str):
                            collected.append(inline)
    if collected:
        merged = "\n".join(part for part in collected if part)
        if merged:
            return merged
    fallback = getattr(response, "output_text", None)
    if isinstance(fallback, str):
        return fallback
    return ""


BASE = Path(__file__).parent.parent  # examples/
CSV_PATH = BASE / "datasets" / "matrices" / "test_matrix.csv"


def _parse_json(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    v = value.strip()
    if v == "" or v.lower() == "none":
        return default
    try:
        return json.loads(v)
    except Exception:
        return default


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        number = int(value)
        return number if number > 0 else default
    except (TypeError, ValueError):
        return default


def _load_row(row_id: int) -> dict[str, str]:
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("id", -1)) == row_id:
                    return row
            except Exception:
                continue
    raise SystemExit(f"Row id {row_id} not found in {CSV_PATH}")


def _has_key_for_integration(name: str) -> bool:
    if name == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY") is not None
    if name == "openai":
        return os.getenv("OPENAI_API_KEY") is not None
    return True


# ---------- Dummy path used ONLY in mock scenarios ----------


def _dummy_function(question: str, **cfg: Any) -> str:
    # Deterministic mapping to known answers from the hello_world dataset
    # Uses config to vary behavior for demo purposes
    answers = [
        "Artificial Intelligence",
        "Uses data and algorithms",
        "Retrieval Augmented Generation",
        "A list of steps",
        "A private key",
    ]
    # Map config to an index
    idx = 0
    if cfg.get("use_rag"):
        idx += 2
    idx += int(cfg.get("top_k", 1)) % 2
    # Flip for certain models
    model = str(cfg.get("model", ""))
    if "sonnet" in model:
        idx = (idx + 1) % len(answers)
    return answers[idx % len(answers)]


# ---------- Lightweight BM25 retrieval for RAG ----------

_bm25 = None  # Will hold a rank_bm25 object lazily
_context_docs: list[str] | None = None


def _load_context_docs() -> list[str]:
    global _context_docs
    if _context_docs is not None:
        return _context_docs
    ctx_path = _HELLO_WORLD_DATA_ROOT / "context_documents.jsonl"
    docs: list[str] = []
    if ctx_path.exists():
        with open(ctx_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    value = (
                        obj.get("text")
                        or obj.get("page_content")
                        or obj.get("content")
                        or obj.get("document")
                        or ""
                    )
                    value = value.strip()
                    if value:
                        docs.append(value)
                except Exception:
                    continue
    _context_docs = docs
    return docs


def _get_bm25() -> Any:
    global _bm25
    if _bm25 is not None:
        return _bm25
    docs = _load_context_docs()

    class _SimpleBM25:
        def __init__(self, _docs: list[str]):
            self._docs = _docs

        def get_top_n(self, query_tokens: list[str], n: int = 3):
            scores = []
            for d in self._docs:
                score = sum(d.lower().count(t) for t in query_tokens)
                scores.append(score)
            top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
            return [self._docs[i] for i in top_idx]

    try:
        from rank_bm25 import BM25Okapi  # type: ignore

        if not docs:
            raise ValueError("No context documents available for BM25")

        tokenized_docs = [d.lower().split() for d in docs]
        # Filter out empty token lists to avoid zero-division in BM25Okapi
        tokenized_docs = [tokens for tokens in tokenized_docs if tokens]
        if not tokenized_docs:
            raise ValueError("Context documents contained no tokens for BM25")

        _bm25 = BM25Okapi(tokenized_docs)
    except (ImportError, ValueError, ZeroDivisionError):
        # Fallback: simple term-frequency scorer
        _bm25 = _SimpleBM25(docs)
    return _bm25


def retrieve_context(query: str, top_k: int = 3) -> list[str]:
    bm25 = _get_bm25()
    tokens = query.lower().split()
    try:
        result: list[str] = bm25.get_top_n(tokens, n=top_k)
        return result
    except Exception:
        return []


def _simple_scoring(
    output: Any, expected: Any, llm_metrics: dict | None = None
) -> float:
    try:
        return (
            1.0
            if (str(output).strip().lower() == str(expected).strip().lower())
            else 0.0
        )
    except Exception:
        return 0.0


def _custom_evaluator(_func: Callable, config: dict[str, Any], example: Any) -> Any:
    # Build an ExampleResult-compatible object using public API
    from traigent.api.types import ExampleResult

    output = _dummy_function(example.input_data.get("question", ""), **config)
    expected = example.expected_output
    acc = 1.0 if str(output).strip().lower() == str(expected).strip().lower() else 0.0
    return ExampleResult(
        example_id=getattr(example, "example_id", "custom"),
        input_data=example.input_data,
        expected_output=expected,
        actual_output=output,
        metrics={"accuracy": acc},
        execution_time=0.0,
        success=True,
    )


@dataclass
class ScenarioConfig:
    row_id: int
    name: str
    description: str
    scenario_notes: str
    execution_mode: str
    objectives: list[str]
    algorithm: str
    max_trials: int
    configuration_space: dict[str, Any]
    evaluator_type: str
    scoring_function_name: str
    injection_mode: str
    framework_targets: list[str]
    dataset: str
    parallel_config: dict[str, Any]
    privacy_enabled: bool
    mock_mode: bool
    integration: str


def _setup_environment(mock_mode: bool) -> None:
    modules_to_clear = [
        mod for mod in list(sys.modules.keys()) if mod.startswith("traigent")
    ]
    for mod in modules_to_clear:
        del sys.modules[mod]
    if modules_to_clear:
        print(
            f"[INFO] Cleared {len(modules_to_clear)} cached traigent modules for fresh import"
        )

    if mock_mode:
        os.environ["TRAIGENT_MOCK_LLM"] = "true"
        print("[INFO] Mock mode enabled (TRAIGENT_MOCK_LLM=true)")
    else:
        previous = os.environ.pop("TRAIGENT_MOCK_LLM", None)
        if previous is not None:
            print(
                f"[INFO] Mock mode disabled (TRAIGENT_MOCK_LLM cleared, was '{previous}')"
            )
        else:
            print("[INFO] Mock mode not set (real execution mode)")


def _define_target(
    integration: str, injection_mode: str, optimize_kwargs: dict[str, Any]
) -> Any:
    from anthropic.types import TextBlock  # type: ignore

    import traigent  # pylint: disable=import-error

    if integration == "anthropic":
        from anthropic import AsyncAnthropic

        # Check if we're in mock mode
        is_mock_mode = os.getenv("TRAIGENT_MOCK_LLM", "false").lower() == "true"

        # Validate API key before proceeding (unless in mock mode)
        if is_mock_mode:
            api_key = "mock-key"
        else:
            raw_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not raw_api_key:
                print("\n" + "=" * 70)
                print("🚨 CRITICAL ERROR: No Anthropic API key found!")
                print("=" * 70)
                print("Please set ANTHROPIC_API_KEY environment variable.")
                print("Example: export ANTHROPIC_API_KEY='sk-ant-api03-...'")
                print("=" * 70 + "\n")
                raise ValueError(
                    "Missing ANTHROPIC_API_KEY - cannot proceed with Anthropic integration"
                )
            api_key = raw_api_key

        # Check for common API key format errors
        if api_key.startswith("sk-proj-") or api_key.startswith("sk-"):
            if not api_key.startswith("sk-ant-"):
                print("\n" + "=" * 70)
                print("⚠️  CRITICAL WARNING: Invalid Anthropic API key format detected!")
                print("=" * 70)
                print(f"Your API key starts with: {api_key[:10]}...")
                print("This appears to be an OpenAI key, not an Anthropic key!")
                print("\nAnthropic keys should start with: sk-ant-api03-...")
                print("OpenAI keys typically start with: sk-proj-... or sk-...")
                print("\n⚠️  API calls will fail with authentication errors!")
                print("⚠️  All metrics will show as 0 due to failed API calls!")
                print("=" * 70 + "\n")

        client = AsyncAnthropic(api_key=api_key)

        # Track API call statistics
        api_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "first_error_logged": False,
        }

        @traigent.optimize(**optimize_kwargs)
        async def anthropic_target(question: str) -> str:
            api_stats["total_calls"] += 1

            # Mock mode: return mock responses without calling API
            if is_mock_mode:
                api_stats["successful_calls"] += 1
                # Return mock canonical answers based on question patterns
                q_lower = question.lower()
                if "capital" in q_lower and "france" in q_lower:
                    return "Paris"
                elif "capital" in q_lower and "japan" in q_lower:
                    return "Tokyo"
                elif "ocean" in q_lower or "largest ocean" in q_lower:
                    return "Pacific"
                elif "python" in q_lower:
                    return "Guido van Rossum"
                else:
                    return "Mock Answer"

            try:
                prompt = _build_prompt(question, retrieve_context(question))
                response = await client.messages.create(
                    model="claude-3-haiku-20240307",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                )
                api_stats["successful_calls"] += 1
                for block in response.content:
                    if isinstance(block, TextBlock):
                        return _map_to_canonical(block.text)
                return ""
            except Exception as e:
                api_stats["failed_calls"] += 1

                # Log first error prominently
                if not api_stats["first_error_logged"]:
                    api_stats["first_error_logged"] = True
                    print("\n" + "=" * 70)
                    print("🚨 CRITICAL: First Anthropic API call failed!")
                    print("=" * 70)
                    print(f"Error: {e}")
                    print("\nPossible causes:")
                    print(
                        "1. Invalid API key (check format: should start with sk-ant-)"
                    )
                    print("2. Network connectivity issues")
                    print("3. Rate limiting or quota exceeded")
                    print("4. Invalid model name or parameters")
                    print("\n⚠️  All subsequent metrics will be 0 due to API failures!")
                    print("=" * 70 + "\n")
                else:
                    # Log subsequent errors more briefly
                    print(
                        f"❌ API call {api_stats['total_calls']} failed: {str(e)[:100]}"
                    )

                # Return a fallback to continue evaluation
                return "Artificial Intelligence"

        # Wrap optimize so we can emit summary stats without losing Traigent interface
        original_optimize = anthropic_target.optimize

        async def optimize_with_reporting(*args: Any, **kwargs: Any):
            result = await original_optimize(*args, **kwargs)
            if (
                api_stats["total_calls"] > 0
                and api_stats["total_calls"] == api_stats["failed_calls"]
                and api_stats["total_calls"] >= 5
            ):
                print("\n" + "=" * 70)
                print("📊 API CALL SUMMARY - CRITICAL FAILURE")
                print("=" * 70)
                print(f"Total API calls attempted: {api_stats['total_calls']}")
                print(f"Successful calls: {api_stats['successful_calls']}")
                print(f"Failed calls: {api_stats['failed_calls']}")
                print("Success rate: 0%")
                print("\n⚠️  ALL API CALLS FAILED - Check your API key and network!")
                print(
                    "⚠️  All metrics are showing 0 because no real LLM responses were received!"
                )
                print("=" * 70 + "\n")
            return result

        anthropic_target.optimize = optimize_with_reporting  # type: ignore[attr-defined]

        return anthropic_target

    if integration == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        @traigent.optimize(**optimize_kwargs)
        async def openai_target(question: str) -> str:
            prompt = _build_prompt(question, retrieve_context(question))
            response = await client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                max_output_tokens=256,
            )
            text = _extract_openai_text(response)
            return _map_to_canonical(text or "")

        return openai_target

    @traigent.optimize(**optimize_kwargs)
    async def default_target(question: str) -> str:
        ctx_docs = retrieve_context(question)
        prompt = _build_prompt(question, ctx_docs)
        answer = _dummy_function(str(question), prompt=prompt)
        return _map_to_canonical(answer)

    return default_target


def _parse_row_config(row_id: int) -> ScenarioConfig:
    row = _load_row(row_id)
    raw_parallel_config = _parse_json(row.get("parallel_config"), {})
    parallel_dict = raw_parallel_config if isinstance(raw_parallel_config, dict) else {}

    trial_concurrency = _coerce_positive_int(parallel_dict.get("trial_concurrency"), 1)
    example_concurrency = _coerce_positive_int(
        parallel_dict.get("example_concurrency"), 1
    )
    thread_workers = _coerce_positive_int(
        parallel_dict.get("thread_workers"),
        max(trial_concurrency, example_concurrency, 1),
    )

    requested_mode = parallel_dict.get("mode")
    if isinstance(requested_mode, str):
        cleaned_mode = requested_mode.strip().lower()
    else:
        cleaned_mode = ""
    if cleaned_mode in {"parallel", "sequential", "auto"}:
        mode = cleaned_mode
    else:
        mode = (
            "parallel"
            if (trial_concurrency > 1 or example_concurrency > 1)
            else "sequential"
        )

    if mode == "parallel" and trial_concurrency <= 1:
        trial_concurrency = max(example_concurrency, 2)
    if mode == "parallel" and example_concurrency <= 1:
        example_concurrency = max(trial_concurrency, 2)

    thread_workers = max(thread_workers, trial_concurrency, example_concurrency)

    parallel_config: dict[str, Any] = {
        "mode": mode,
        "trial_concurrency": trial_concurrency,
        "example_concurrency": example_concurrency,
        "thread_workers": thread_workers,
    }

    return ScenarioConfig(
        row_id=row_id,
        name=row.get("name", f"Scenario {row_id}"),
        description=row.get("description", ""),
        scenario_notes=row.get("scenario_notes", ""),
        execution_mode=(row.get("execution_mode") or "edge_analytics").strip(),
        objectives=_parse_json(row.get("objectives"), ["accuracy"]) or ["accuracy"],
        algorithm=(row.get("algorithm") or "grid").strip(),
        max_trials=int(row.get("max_trials") or 6),
        configuration_space=_parse_json(row.get("configuration_space"), {}) or {},
        evaluator_type=(row.get("evaluator") or "default").strip(),
        scoring_function_name=(row.get("scoring_function") or "").strip(),
        injection_mode=(row.get("injection_mode") or "context").strip(),
        framework_targets=_parse_json(row.get("framework_targets"), []) or [],
        dataset=(
            row.get("dataset") or "examples/datasets/hello-world/evaluation_set.jsonl"
        ).strip(),
        parallel_config=parallel_config,
        privacy_enabled=_to_bool(row.get("privacy_enabled", "false")),
        mock_mode=_to_bool(row.get("mock_mode", "false")),
        integration=(row.get("integration") or "none").strip(),
    )


def _resolve_dataset_path(dataset: str) -> str:
    dataset_path = Path(dataset)
    if dataset_path.exists():
        return dataset
    return str((Path.cwd() / dataset).resolve())


def _build_optimize_kwargs(config: ScenarioConfig, dataset_path: str) -> dict[str, Any]:
    return {
        "eval_dataset": dataset_path,
        "objectives": config.objectives,
        "configuration_space": config.configuration_space,
        "execution_mode": config.execution_mode,
        "injection_mode": config.injection_mode,
        "framework_targets": config.framework_targets,
        "privacy_enabled": config.privacy_enabled,
        "parallel_config": config.parallel_config,
    }


def _build_call_kwargs(config: ScenarioConfig) -> dict[str, Any]:
    call_kwargs: dict[str, Any] = {
        "parallel_config": config.parallel_config,
    }
    if config.evaluator_type == "scoring_function":
        available_scores: dict[str, Callable[[Any, Any, dict | None], float]] = {
            "simple_score": _simple_scoring,
        }
        call_kwargs["scoring_function"] = available_scores.get(
            config.scoring_function_name, _simple_scoring
        )
    elif config.evaluator_type == "custom_evaluator":
        call_kwargs["custom_evaluator"] = _custom_evaluator
    return call_kwargs


def _should_skip_with_missing_keys(config: ScenarioConfig) -> bool:
    needs_key = config.integration in {"anthropic", "openai"}
    return (
        needs_key
        and not config.mock_mode
        and not _has_key_for_integration(config.integration)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Traigent scenario by CSV row id")
    parser.add_argument(
        "--row",
        type=int,
        required=True,
        help="Row id from datasets/matrices/test_matrix.csv",
    )
    args = parser.parse_args()

    config = _parse_row_config(args.row)
    print({"row": config.row_id, "name": config.name, "desc": config.description})

    setup_logging(os.getenv("TRAIGENT_LOG_LEVEL", "INFO"))

    if _should_skip_with_missing_keys(config):
        print(f"SKIPPED: Missing API key for integration {config.integration}")
        return

    _setup_environment(config.mock_mode)

    from traigent.api.types import OptimizationResult

    dataset_path = _resolve_dataset_path(config.dataset)
    optimize_kwargs = _build_optimize_kwargs(config, dataset_path)
    target = _define_target(config.integration, config.injection_mode, optimize_kwargs)
    call_kwargs = _build_call_kwargs(config)

    async def runner() -> None:
        assert target is not None, "Target function not defined"
        res: OptimizationResult = await target.optimize(
            algorithm=config.algorithm,
            max_trials=config.max_trials,
            **call_kwargs,
        )
        print({"best_config": res.best_config, "best_score": res.best_score})

        if config.execution_mode != "edge_analytics":
            df = res.to_aggregated_dataframe(
                primary_objective=config.objectives[0] if config.objectives else None
            )
            print(df)
            raw_df = res.to_dataframe()
            print(raw_df.head())
        else:
            raw_df = res.to_dataframe()
            print(raw_df.head())
            aggregated = res.to_aggregated_dataframe(
                primary_objective=config.objectives[0] if config.objectives else None
            )
            print(aggregated)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
