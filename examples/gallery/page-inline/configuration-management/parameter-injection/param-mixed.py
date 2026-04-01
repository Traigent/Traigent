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
from langchain_openai import ChatOpenAI

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

# Mix seamless injection with custom parameters


@traigent.optimize(
    configuration_space={
        # Direct LLM parameters (seamless injection)
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        # Custom application parameters
        "summarization_style": ["bullet", "paragraph", "outline"],
        "detail_level": ["brief", "detailed", "comprehensive"],
        "include_metrics": [True, False],
    }
)
def smart_document_processor(document: str) -> str:
    config = traigent.get_config()
    if not isinstance(config, dict):
        config = {}

    # Seamless injection handles model and temperature
    model = str(config.get("model", "gpt-4o-mini"))
    temperature = float(config.get("temperature", 0.7))
    llm = ChatOpenAI(model=model, temperature=temperature)

    # Use custom parameters for application logic
    style_instructions = {
        "bullet": "Format as bullet points",
        "paragraph": "Format as flowing paragraphs",
        "outline": "Format as a hierarchical outline",
    }

    detail_instructions = {
        "brief": "Keep it concise, 2-3 sentences",
        "detailed": "Provide moderate detail, 1-2 paragraphs",
        "comprehensive": "Provide comprehensive analysis",
    }

    # Build optimized prompt
    prompt = f"""
    {style_instructions.get(str(config.get('summarization_style', 'bullet')), 'Format as bullet points')}
    {detail_instructions.get(str(config.get('detail_level', 'brief')), 'Keep it concise, 2-3 sentences')}
    {'Include key metrics and numbers.' if bool(config.get('include_metrics', False)) else ''}

    Document: {document}
    """

    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


if __name__ == "__main__":
    try:
        try:
            _res = smart_document_processor("test input")
            print(getattr(_res, "content", _res))
        except Exception as e:
            print(e)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
