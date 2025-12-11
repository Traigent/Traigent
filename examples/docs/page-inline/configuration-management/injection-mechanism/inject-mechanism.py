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

# How TraiGent intercepts and modifies your LLM calls


# Your original code:
def original_function():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke("Hello")
    return getattr(response, "content", str(response))


# What TraiGent does internally:
@traigent.optimize(configuration_space={"model": [...], "temperature": [...]})
def my_function():
    # 1. TraiGent intercepts the ChatOpenAI constructor
    # 2. Gets optimal parameters from configuration space
    # 3. Overrides your parameters with optimal ones
    # 4. Creates LLM instance with optimal parameters

    optimal_config = traigent.get_config()
    if not isinstance(optimal_config, dict):
        optimal_config = {"model": "gpt-4o-mini", "temperature": 0.3}

    # Your code sees this, but TraiGent modified the parameters:
    llm = ChatOpenAI(
        model=optimal_config["model"],  # Was "gpt-3.5-turbo", now "gpt-4o-mini"
        temperature=optimal_config["temperature"],  # Was 0.7, now 0.3
    )
    response = llm.invoke("Hello")
    return getattr(response, "content", str(response))


# The injection happens at multiple levels:
# - Constructor parameter interception
# - Method call parameter overrides
# - Configuration context management
# - Runtime parameter substitution

if __name__ == "__main__":
    try:
        _res = my_function()
        print(getattr(_res, "content", _res))
    except Exception as e:
        print(e)
