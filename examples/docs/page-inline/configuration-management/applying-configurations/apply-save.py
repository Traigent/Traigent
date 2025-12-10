"""Demonstrate saving and reusing optimized Traigent configurations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar, cast

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

F = TypeVar("F")
CONFIG_PATH = Path(__file__).with_name("optimal_support_config.json")


def _load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config if it exists, otherwise return an empty dict."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _apply_saved_config(func: Any, config_path: Path) -> None:
    config = _load_config(config_path)
    setter = getattr(func, "set_config", None)
    if config and callable(setter):
        try:
            cast("Callable[[Dict[str, Any]], Any]", setter)(config)
        except Exception:
            # Safety net: examples should not crash if config contains unexpected values
            pass


def apply_saved_config(path: Path) -> Callable[[F], F]:
    """Local decorator that mirrors the intended Traigent helper."""

    def decorator(func: F) -> F:
        _apply_saved_config(func, path)
        return func

    return decorator


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "max_tokens": [100, 300, 500],
    },
    eval_dataset="customer_support.jsonl",
    objectives=["cost", "accuracy"],
    save_config=str(CONFIG_PATH),
)
def customer_support_agent(query: str) -> str:
    """Handle a support query using parameters injected by Traigent."""
    cfg = traigent.get_config()
    model = (
        cfg.get("model", "gpt-3.5-turbo") if isinstance(cfg, dict) else "gpt-3.5-turbo"
    )
    temperature = float(cfg.get("temperature", 0.7)) if isinstance(cfg, dict) else 0.7
    max_tokens = int(cfg.get("max_tokens", 300)) if isinstance(cfg, dict) else 300

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        model_kwargs={"max_tokens": max_tokens},
    )
    prompt = f"Customer query: {query}"
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


@apply_saved_config(CONFIG_PATH)
def production_support_agent(query: str) -> str:
    """Use the saved optimal configuration without re-running optimization."""
    cfg = traigent.get_config()
    model = (
        cfg.get("model", "gpt-3.5-turbo") if isinstance(cfg, dict) else "gpt-3.5-turbo"
    )
    temperature = float(cfg.get("temperature", 0.7)) if isinstance(cfg, dict) else 0.7
    max_tokens = int(cfg.get("max_tokens", 300)) if isinstance(cfg, dict) else 300

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        model_kwargs={"max_tokens": max_tokens},
    )
    response = llm.invoke(f"Customer query: {query}")
    return getattr(response, "content", str(response))


if __name__ == "__main__":
    print(customer_support_agent("How do I reset my password?"))
    print(production_support_agent("How quickly can I upgrade my plan?"))
