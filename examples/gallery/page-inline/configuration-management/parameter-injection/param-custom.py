from __future__ import annotations

import sys
from pathlib import Path

# --- Setup for running from repo without installation ---
# Set TRAIGENT_SDK_PATH to override when running from outside the repo tree.
_sdk_override = os.environ.get("TRAIGENT_SDK_PATH")
if _sdk_override:
    if _sdk_override not in sys.path:
        sys.path.insert(0, _sdk_override)
else:
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

    _sdk = os.environ.get("TRAIGENT_SDK_PATH")
    if _sdk:
        sys.path.insert(0, _sdk)
    else:
        module_path = Path(__file__).resolve()
        for depth in (2, 3):
            try:
                sys.path.append(str(module_path.parents[depth]))
            except IndexError:
                continue
    traigent = importlib.import_module("traigent")

# Define custom parameters to optimize


@traigent.optimize(
    configuration_space={
        "llm_creativity": [0.1, 0.3, 0.5, 0.7, 0.9],  # Custom parameter
        "response_length": ["short", "medium", "long"],  # Custom parameter
        "tone": ["formal", "casual", "friendly"],  # Custom parameter
        "model_tier": ["fast", "balanced", "accurate"],  # Custom parameter
    }
)
def adaptive_chat_bot(user_message: str) -> str:
    # Get optimized parameters from Traigent
    config = traigent.get_config()
    if not isinstance(config, dict):
        config = {}

    # Map custom parameters to LLM settings
    model_map = {
        "fast": "gpt-3.5-turbo",
        "balanced": "gpt-4o-mini",
        "accurate": "gpt-4o",
    }

    length_map = {"short": 100, "medium": 300, "long": 500}

    # Use optimized values
    model_key = str(config.get("model_tier", "balanced"))
    length_key = str(config.get("response_length", "medium"))
    temperature = float(config.get("llm_creativity", 0.5))

    llm = ChatOpenAI(
        model=model_map.get(model_key, "gpt-4o-mini"),
        temperature=temperature,
        max_tokens=length_map.get(length_key, 300),
    )

    # Build prompt with optimized tone
    tone = str(config.get("tone", "casual"))
    tone_prompt = f"Respond in a {tone} tone."
    full_prompt = f"{tone_prompt}\n\nUser: {user_message}\nAssistant:"

    response = llm.invoke(full_prompt)
    return getattr(response, "content", str(response))


if __name__ == "__main__":
    try:
        try:
            _res = adaptive_chat_bot("test input")
            print(getattr(_res, "content", _res))
        except Exception as e:
            print(e)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
