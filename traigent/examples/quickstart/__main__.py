"""Find the best model and temperature for your task - in one decorator.

No API keys needed - runs in mock mode and simulates LLM responses
so you can see the optimization flow instantly.
For a real run with actual LLM calls, see walkthrough/real/01_tuning_qa.py.
"""

import asyncio
import os
from pathlib import Path

# Default to mock mode so the quickstart works without API keys.
# Override with: TRAIGENT_MOCK_LLM=false python -m traigent.examples.quickstart
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
if os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
    os.environ.setdefault("OPENAI_API_KEY", "mock-key-for-demos")
    # Only fall back to offline mode if no Traigent API key is configured.
    # When the user has set TRAIGENT_API_KEY, results should reach the portal
    # even though the LLM calls themselves are mocked.
    if not os.environ.get("TRAIGENT_API_KEY"):
        os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

# The bundled dataset lives next to this file (inside the installed package).
# Set TRAIGENT_DATASET_ROOT so the SDK's path validation accepts it regardless
# of which directory the user runs from.
_PACKAGE_DIR = str(Path(__file__).resolve().parent)
os.environ.setdefault("TRAIGENT_DATASET_ROOT", _PACKAGE_DIR)

# NOTE: This uses LangChain's ChatOpenAI rather than the litellm calls shown
# in the documentation. That is intentional and temporary: Traigent's mock
# interceptor (TRAIGENT_MOCK_LLM=true) only patches LangChain's ChatOpenAI at
# this point - it does not yet intercept raw litellm/openai calls.
# Once the interceptor adds raw litellm support this file will be updated to
# match the litellm style shown in the docs.
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import traigent
from traigent.api.decorators import EvaluationOptions

DATASET = str(Path(__file__).resolve().parent / "qa_samples.jsonl")
CONFIG_SPACE = {
    "model": ["gpt-4o-mini", "gpt-4o"],
    "temperature": [0.0, 0.7, 1.0],
}

_SYSTEM_PROMPT = "Answer in as few words as possible. Give only the answer itself, nothing else."

# In mock mode the LLM returns a fixed placeholder string, so contains_accuracy
# would always score 0.  Let the built-in mock simulation handle accuracy instead.
_mock_mode = os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes")


def contains_accuracy(output: str, expected: str) -> float:
    """1.0 if the expected answer appears anywhere in the output (case-insensitive)."""
    return 1.0 if expected.lower() in output.lower() else 0.0


@traigent.optimize(
    configuration_space=CONFIG_SPACE,
    objectives=["accuracy"],
    evaluation=EvaluationOptions(
        eval_dataset=DATASET,
        metric_functions=None if _mock_mode else {"accuracy": contains_accuracy},
    ),
    execution_mode="edge_analytics",
)
def answer(question: str) -> str:
    """Call an LLM with the current trial's config."""
    cfg = traigent.get_config()
    llm = ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
    return llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(question)]).content


if __name__ == "__main__":
    result = asyncio.run(answer.optimize(max_trials=6, algorithm="grid"))
