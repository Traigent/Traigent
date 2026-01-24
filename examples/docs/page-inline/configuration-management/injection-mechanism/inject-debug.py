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

# Debug mode to see exactly what Traigent is doing


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5, 0.9],
    },
    debug=True,  # Enable debug mode
)
def debug_example(prompt: str) -> str:
    print("Original parameters in code:")
    print("- model: gpt-3.5-turbo")
    print("- temperature: 0.7")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Debug output will show:
    # [Traigent] Intercepted ChatOpenAI constructor
    # [Traigent] Original: model=gpt-3.5-turbo, temperature=0.7
    # [Traigent] Optimized: model=gpt-4o-mini, temperature=0.1
    # [Traigent] Injecting optimized parameters

    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


if __name__ == "__main__":
    try:
        _res = debug_example("test input")
        print(getattr(_res, "content", _res))
    except Exception as e:
        print(e)
