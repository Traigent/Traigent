from __future__ import annotations

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
from examples.utils.langchain_compat import ChatOpenAI

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

# Your existing function - no changes needed


def generate_summary(text: str) -> str:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # TraiGent will optimize this
        temperature=0.7,  # And this
        model_kwargs={"max_tokens": 150},  # And this
    )

    prompt = f"Summarize this text in 2-3 sentences: {text}"
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


# Add optimization with seamless injection


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_tokens": [100, 150, 200, 300],
    },
    eval_dataset="summaries.jsonl",
    objectives=["cost", "accuracy"],
)
def optimized_summary(text: str) -> str:
    # Exact same code - TraiGent seamlessly injects optimal parameters
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # Becomes optimal model automatically
        temperature=0.7,  # Becomes optimal temperature automatically
        model_kwargs={"max_tokens": 150},  # Becomes optimal token limit automatically
    )

    prompt = f"Summarize this text in 2-3 sentences: {text}"
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


# TraiGent finds: model="gpt-4o-mini", temperature=0.3, model_kwargs={"max_tokens": 200
# Your function automatically uses these optimal values!

if __name__ == "__main__":
    try:
        _res = optimized_summary("test input")
        print(getattr(_res, "content", _res))
    except Exception as e:
        print(e)
