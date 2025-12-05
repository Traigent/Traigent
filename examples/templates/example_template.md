# Example Folder Template

Use this template for every example to keep structure minimal, consistent, and powerful.

## Folder Layout

```
examples/
└── NN_slug/                     # One folder per example
    ├── run.py                   # Single decorated function + minimal main
    ├── evaluation_set.jsonl     # Evaluation dataset (JSONL of {input, output})
    ├── prompt.txt               # Prompt template (no code, plain text)
    ├── context_documents.jsonl  # Optional: RAG documents (JSONL of {page_content})
    └── example_set.jsonl        # Optional: few-shot examples (JSONL of {role, content})
```

Notes:
- Keep `run.py` focused on one feature (e.g., RAG parameter injection, custom metric).
- Do not duplicate code across examples. If helpers are needed repeatedly, place them in `examples/shared/` and import.

## Files

- `run.py`
  - Contains exactly one function decorated with `@traigent.optimize`.
  - Shows parameter injection clearly in `configuration_space`.
  - Reads `evaluation_set.jsonl`, `prompt.txt`, and optionally `context_documents.jsonl`/`example_set.jsonl`.
  - Prints a single concise result: `{"best_config": ..., "best_score": ...}`.

- `evaluation_set.jsonl`
  - Each line: `{ "inpugt": { ... }, "output": <ground_truth> }`.
  - Keep it tiny (3–10 lines) and deterministic.

- `prompt.txt`
  - Plain text template for the model instruction.
  - `run.py` constructs the final prompt by combining:
    - Optional RAG context
    - `Question: {question}` (or task input)
    - The contents of `prompt.txt`

- `context_documents.jsonl` (optional)
  - Each line: `{ "page_content": "..." }`.
  - Used for RAG examples; loaded into a retriever (e.g., BM25).

- `example_set.jsonl` (optional)
  - Few-shot examples for the prompt as lines like `{ "role": "user|assistant", "content": "..." }`.
  - `run.py` can prepend these to the prompt if relevant.

## Minimal `run.py` Skeleton

```python
#!/usr/bin/env python3
"""One-line summary of the example."""

import os, asyncio, json
from pathlib import Path
from typing import List

import traigent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

BASE = Path(__file__).parent
DATASET = str(BASE / "evaluation_set.jsonl")
PROMPT_PATH = BASE / "prompt.txt"
CONTEXT_PATH = BASE / "context_documents.jsonl"

def _load_prompt() -> str:
    return PROMPT_PATH.read_text().strip()

def _load_context_docs() -> List[Document]:
    if not CONTEXT_PATH.exists():
        return []
    docs: List[Document] = []
    with open(CONTEXT_PATH) as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(page_content=obj["page_content"]))
    return docs

_PROMPT = _load_prompt()
_RETRIEVER = BM25Retriever.from_documents(_load_context_docs()) if _load_context_docs() else None

def _build_prompt(question: str, context: str | None) -> str:
    ctx = f"Context:\n{context}\n\n" if context else ""
    return f"{ctx}Question: {question}\n\n{_PROMPT}"

@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy"],
    configuration_space={
        "model": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "temperature": [0.0, 0.2],
        "use_rag": [True, False],
        "top_k": [1, 2, 3],
    },
    execution_mode="edge_analytics",
)
def answer_question(question: str, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.2, use_rag: bool = True, top_k: int = 2) -> str:
    assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY"
    ctx = None
    if use_rag and _RETRIEVER is not None:
        _RETRIEVER.k = max(1, int(top_k))
        ctx = "\n\n".join(d.page_content for d in _RETRIEVER.get_relevant_documents(question))
    prompt = _build_prompt(question, ctx)
raw = str(
    ChatAnthropic(
        model_name=model,
        temperature=temperature,
        max_tokens=128,
        timeout=None,
        stop=None,
    ).invoke([HumanMessage(content=prompt)]).content
).strip()
    # Normalize to expected labels (keep minimal)
    for k in [
        "Artificial Intelligence",
        "Uses data and algorithms",
        "Retrieval Augmented Generation",
        "A list of steps",
        "A private key",
    ]:
        if k.lower() in raw.lower():
            return k
    return raw[:128]

async def main():
    r = await answer_question.optimize(max_trials=6)
    print({"best_config": r.best_config, "best_score": r.best_score})

if __name__ == "__main__":
    asyncio.run(main())
```

## Success Criteria
- Minimal code and prints; one decorated function; one result line.
- All data (dataset, context, prompt, few-shot) lives in files, not code.
- Works with `uv` extras install and `ANTHROPIC_API_KEY` set.
