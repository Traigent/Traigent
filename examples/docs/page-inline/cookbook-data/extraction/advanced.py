#!/usr/bin/env python3
"""Cookbook Data - Structured Extraction (Advanced)

Adds few-shots, prompt style, and output format. Minimal configuration surface.
"""
import json
import os
import sys
from pathlib import Path

# --- Setup for running from repo without installation ---
# Add repo root to path so we can import examples.utils and traigent
_module_path = Path(__file__).resolve()
for _depth in range(1, 7):
    try:
        _repo_root = _module_path.parents[_depth]
        if (_repo_root / "traigent").is_dir() and (_repo_root / "examples").is_dir():
            if str(_repo_root) not in sys.path:
                sys.path.insert(0, str(_repo_root))
            break
    except IndexError:
        continue
from examples.utils.langchain_compat import ChatOpenAI, HumanMessage, extract_content

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

DATASET = os.path.join(os.path.dirname(__file__), "extraction_eval.jsonl")


def _json_accuracy(
    output: str | None, expected: dict | None, llm_metrics=None
) -> float:
    if not output or not expected:
        return 0.0
    try:
        data = json.loads(output)
    except Exception:
        return 0.0
    return (
        1.0
        if (
            data.get("company") == expected.get("company")
            and data.get("amount") == expected.get("amount")
        )
        else 0.0
    )


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.1, 0.3],
        "use_examples": [True, False],
        "prompt_style": ["direct", "reasoned"],
        "output_format": ["json", "kv"],
    },
    eval_dataset=DATASET,
    objectives=["accuracy", "cost", "response_time"],
    metric_functions={"accuracy": _json_accuracy},
    execution_mode="edge_analytics",
    max_trials=10,
)
def extract_fields(text: str) -> str:
    cfg = traigent.get_trial_config()
    guidance = {
        "direct": "Extract fields precisely.",
        "reasoned": "Identify key fields, then extract them precisely.",
    }[cfg.get("prompt_style", "direct")]

    fewshots = (
        """
Examples:
Text: "Invoice: Acme Corp billed $1200 on 2024-09-01." -> {"company": "Acme Corp", "amount": 1200}
Text: "Receipt: Beta LLC total charge was $45." -> {"company": "Beta LLC", "amount": 45}
"""
        if cfg.get("use_examples", False)
        else ""
    )

    if cfg.get("output_format") == "kv":
        guard = "Return 'company: <name> | amount: <number>'"
    else:
        guard = 'Return JSON: {"company": string, "amount": number}'

    prompt = f"""
{guidance}
{fewshots}
Text: {text}
{guard}
""".strip()

    llm = ChatOpenAI(
        model=cfg.get("model", "gpt-3.5-turbo"), temperature=cfg.get("temperature", 0.0)
    )
    raw = extract_content(llm.invoke([HumanMessage(content=prompt)]))

    if cfg.get("output_format") == "kv":
        # Normalize KV to JSON if needed for metric
        try:
            # Simple parse: company: X | amount: Y
            parts = [p.strip() for p in raw.split("|")]
            kv = {}
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    kv[k.strip().lower()] = v.strip()
            amt = kv.get("amount")
            if amt is not None:
                try:
                    kv["amount"] = int(str(amt).replace("$", "").split()[0])
                except Exception:
                    pass
            return json.dumps(
                {"company": kv.get("company", ""), "amount": kv.get("amount")}
            )
        except Exception:
            return raw
    return raw


if __name__ == "__main__":
    import asyncio

    async def _main():
        print("Optimizing extract_fields (advanced)…")
        res = await extract_fields.optimize(max_trials=10)
        print("Best config:", res.best_config)
        extract_fields.set_config(res.best_config)
        print("Test:", extract_fields("Invoice: Omega Inc billed $799 on 2024-05-01."))

    asyncio.run(_main())
