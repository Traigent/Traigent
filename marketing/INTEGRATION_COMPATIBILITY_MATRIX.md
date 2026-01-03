# Integration Compatibility Matrix (Marketing-Level)

This is a **communication aid**, not a promise. Use it to qualify “low friction” claims and set expectations.

For authoritative details, point builders to the repo integrations and examples.

## Works well (low friction)

| Area | What “works well” means | Notes |
| --- | --- | --- |
| Plain Python functions | Decorate a function, run `.optimize()`, evaluate on a dataset | Best when key knobs are explicit or framework-overridable |
| Local execution | `execution_mode="edge_analytics"` | Default OSS mode |
| Mock mode | `TRAIGENT_MOCK_LLM=true` | Great for demos/CI dry runs (still validate your own code paths) |
| CI gates pattern | Evaluate baseline vs PR + optionally tune | Example runner exists under `examples/integrations/ci-cd/` |

## Supported integrations (typical adoption paths)

These are **in-repo integrations/plugins**; exact behavior depends on how your agent is written.

| Category | Examples | Where to look |
| --- | --- | --- |
| LLM providers / SDKs | OpenAI, Anthropic, Azure OpenAI, Bedrock, Gemini, Mistral, Cohere | `traigent/integrations/llms/` |
| Agent frameworks | LangChain, LlamaIndex | `traigent/integrations/llms/langchain/`, `traigent/integrations/llms/llamaindex_plugin.py` |
| Vector stores | ChromaDB, Pinecone, Weaviate | `traigent/integrations/vector_stores/` |
| Observability | MLflow, Weights & Biases | `traigent/integrations/observability/` |

## Where you may need adapter code (set expectations)

| Scenario | Why it needs work | Typical fix |
| --- | --- | --- |
| Custom in-house agent framework | No standard hook point for injection/overrides | Add a thin adapter or explicit config plumbing |
| Hidden configuration | Parameters not externally controllable | Make knobs explicit or use supported override targets |
| Non-standard evaluation harness | Scoring logic is bespoke | Use a custom evaluator function, keep the spec as the contract |

## Roadmap vs OSS (keep claims honest)

- OSS focuses on local execution (`edge_analytics`) with forward-compatible placeholders for cloud/hybrid.
- Don’t imply managed backend/hosted eval infrastructure unless it’s actually provisioned for the reader.

