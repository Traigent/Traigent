"""Find the best model and temperature for your task - in one decorator.

No API keys needed - runs in mock mode and simulates LLM responses
so you can see the optimization flow instantly.
For a real run with actual LLM calls, see walkthrough/real/01_tuning_qa.py.
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

# Default to mock mode so the quickstart works without API keys.
# Override with: TRAIGENT_MOCK_LLM=false python -m traigent.examples.quickstart
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
if os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
    os.environ.setdefault("OPENAI_API_KEY", "mock-key-for-demos")
    os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

# NOTE: This uses LangChain's ChatOpenAI rather than the litellm calls shown
# in the documentation. That is intentional and temporary: Traigent's mock
# interceptor (TRAIGENT_MOCK_LLM=true) only patches LangChain's ChatOpenAI at
# this point — it does not yet intercept raw litellm/openai calls.
# Once the interceptor adds raw litellm support this file will be updated to
# match the litellm style shown in the docs.
from langchain_openai import ChatOpenAI

import traigent

_DATASET_SRC = Path(__file__).parent / "qa_samples.jsonl"
# The SDK requires dataset files to reside under the system temp directory.
# When installed via pip the package data is in site-packages, so copy it.
if str(_DATASET_SRC).startswith(tempfile.gettempdir()):
    DATASET = str(_DATASET_SRC)
else:
    _tmp = Path(tempfile.mkdtemp())
    DATASET = str(shutil.copy(_DATASET_SRC, _tmp / _DATASET_SRC.name))
OBJECTIVES = ["accuracy"]
CONFIG_SPACE = {
    "model": ["gpt-4o-mini", "gpt-4o"],
    "temperature": [0.0, 0.7, 1.0],
}


@traigent.optimize(
    configuration_space=CONFIG_SPACE,
    objectives=OBJECTIVES,
    eval_dataset=DATASET,
    execution_mode="edge_analytics",
)
def answer(question: str) -> str:
    """Call an LLM with the current trial's config."""
    cfg = traigent.get_config()
    llm = ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
    return llm.invoke(question).content


if __name__ == "__main__":
    result = asyncio.run(answer.optimize(max_trials=6, algorithm="grid"))
