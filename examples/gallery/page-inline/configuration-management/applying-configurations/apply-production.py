"""Production deployment example that reuses saved Traigent configurations."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


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

_DEFAULT_CONFIG: dict[str, Any] = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 300,
    "response_style": "professional",
}


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


class OptimizedChatBot:
    """Simple chatbot that can refresh its Traigent configuration at runtime."""

    def __init__(self, config_path: str | None = None):
        self._base_dir = Path(__file__).resolve().parent
        self._config_path = Path(config_path) if config_path else None
        self.config: dict[str, Any] = self._load_configuration()
        self._response_fn: Callable[[str, str], str] | None = None
        self._setup_optimized_function()

    def _load_configuration(self) -> dict[str, Any]:
        candidates: list[Path] = []
        if self._config_path:
            candidates.append(self._config_path)

        env = os.getenv("ENVIRONMENT")
        configs_dir = self._base_dir / "configs"
        if env == "production":
            candidates.append(configs_dir / "prod_config.json")
        else:
            candidates.append(configs_dir / "dev_config.json")

        for candidate in candidates:
            if candidate.exists():
                loaded = _safe_read_json(candidate)
                if loaded:
                    return loaded
        return _DEFAULT_CONFIG.copy()

    def _setup_optimized_function(self) -> None:
        config = self.config

        @traigent.optimize(
            configuration_space={
                "model": [config.get("model", "gpt-3.5-turbo")],
                "temperature": [config.get("temperature", 0.7)],
                "max_tokens": [config.get("max_tokens", 300)],
            },
            objectives=["quality"],
            execution_mode="edge_analytics",
            max_trials=5,
        )
        def _generate_response(user_input: str, context: str = "") -> str:
            llm = ChatOpenAI(
                model=str(config.get("model", "gpt-3.5-turbo")),
                temperature=float(config.get("temperature", 0.7)),
                max_tokens=int(config.get("max_tokens", 300)),
            )
            prompt = f"Context: {context}\nUser: {user_input}\nAssistant:"
            reply = llm.invoke(prompt)
            return getattr(reply, "content", str(reply))

        self._response_fn = _generate_response
        self._apply_config_to_function()

    def _apply_config_to_function(self) -> None:
        if self._response_fn is None:
            return
        setter = getattr(self._response_fn, "set_config", None)
        if callable(setter):
            try:
                setter(self.config)
            except Exception:
                pass

    def generate_response(self, user_input: str, context: str = "") -> str:
        if self._response_fn is None:
            raise RuntimeError("Chat bot not initialized correctly")
        return self._response_fn(user_input, context)

    def update_config(self, new_config_path: str) -> None:
        self._config_path = Path(new_config_path)
        self.config = self._load_configuration()
        self._apply_config_to_function()


if __name__ == "__main__":
    try:
        bot = OptimizedChatBot()
        print(bot.generate_response("How do I reset my password?"))

        bot.update_config("customer_support_optimal.json")
        print(bot.generate_response("Do you support enterprise SSO?"))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
