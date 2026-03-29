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

# Advanced seamless injection with multiple LLM instances


@traigent.optimize(
    configuration_space={
        # Separate configs for different LLM roles
        "analyzer_model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "analyzer_temp": [0.0, 0.1, 0.2],
        "generator_model": ["gpt-3.5-turbo", "gpt-4o"],
        "generator_temp": [0.5, 0.7, 0.9],
        "max_context": [2000, 4000, 8000],
    }
)
def intelligent_content_system(topic: str) -> str:
    # Step 1: Analyze topic requirements
    analyzer = ChatOpenAI(
        model="gpt-3.5-turbo",  # Uses analyzer_model from optimization
        temperature=0.1,  # Uses analyzer_temp from optimization
        max_tokens=500,
    )

    analysis_result = analyzer.invoke(f"Analyze content requirements for: {topic}")
    analysis = getattr(analysis_result, "content", str(analysis_result))

    # Step 2: Generate content based on analysis
    generator = ChatOpenAI(
        model="gpt-4o",  # Uses generator_model from optimization
        temperature=0.7,  # Uses generator_temp from optimization
        max_tokens=2000,  # Uses max_context from optimization
    )

    content_result = generator.invoke(
        f"Based on this analysis: {analysis}\n\nCreate content about: {topic}"
    )
    return getattr(content_result, "content", str(content_result))


# Traigent optimizes each LLM call independently for best results

if __name__ == "__main__":
    try:
        try:
            _res = intelligent_content_system("test input")
            print(getattr(_res, "content", _res))
        except Exception as e:
            print(e)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
