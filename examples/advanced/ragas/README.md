# RAGAS Playground Examples

Three focused examples show how to evaluate retrieval-heavy pipelines with the RAGAS metric suite. Each script can be executed directly (see `run.py` files) and uses the new `ragas` integration in Traigent.

| Example | Highlights | When to Use |
| --- | --- | --- |
| `basics/` | Non-LLM metrics (`context_precision`, `context_recall`, `answer_similarity`). Configuration toggles grounded vs. vague strategies and direct vs. rambling tone. | Quick sanity checks without API keys. |
| `column_map/` | Demonstrates `ragas_column_map` to remap custom metadata (`gold_contexts`, `reference_answer`). Strategies compare lookup vs. narrative summaries to show mapping impact. | When datasets use bespoke keys or need column remapping. |
| `with_llm/` | Faithfulness and answer relevancy using a LangChain/OpenAI client. Switch between citing evidence and omitting it, plus succinct vs. speculative styles. | When you want full RAGAS fidelity with an LLM grader. |

## Running

```bash
# Install optional dependencies if you haven't already
pip install optuna ragas rapidfuzz langchain-openai

# Mock-friendly basics (no API keys required)
python examples/advanced/ragas/basics/run.py

# Column-map demonstration (still mock-friendly)
python examples/advanced/ragas/column_map/run.py

# Faithfulness/answer relevancy (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
python examples/advanced/ragas/with_llm/run.py
```
