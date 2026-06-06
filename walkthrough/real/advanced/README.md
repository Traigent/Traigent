# Advanced Real Walkthroughs

These examples are standalone companion walkthroughs for provider-specific or
multi-agent scenarios. Run them from the repository root.

## 04: AWS Bedrock Multi-Agent RAG

`04_multi_agent_bedrock.py` demonstrates a two-agent RAG workflow:

- Retriever agent: FAISS over a bundled offline corpus, using a tiny
  LangChain-compatible embeddings wrapper. Real mode calls Bedrock Titan
  embeddings through LiteLLM. Mock mode uses deterministic local vectors so the
  dry run never calls AWS.
- Generator agent: `litellm.acompletion(model="bedrock/<us inference profile>")`
  with a grid search of one trial per model.
- Observability: `execution_mode="edge_analytics"` plus declared retriever and
  generator agents. When `TRAIGENT_BACKEND_URL`, scoped `TRAIGENT_API_KEY`, and
  `TRAIGENT_TRACE_ENABLED=true` are configured, the public span API emits
  per-agent workflow trace boxes. Without those variables, the script still runs
  fully offline.

Dry run first:

```bash
PYTHONPATH=. \
TRAIGENT_MOCK_LLM=true \
TRAIGENT_OFFLINE_MODE=true \
TRAIGENT_COST_APPROVED=true \
.venv/bin/python walkthrough/real/advanced/04_multi_agent_bedrock.py
```

Real Bedrock run:

```bash
PYTHONPATH=. \
TRAIGENT_COST_APPROVED=true \
AWS_REGION=us-east-1 \
.venv/bin/python walkthrough/real/advanced/04_multi_agent_bedrock.py
```

Use the normal AWS credential chain (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, or role-based credentials). Do not
put credentials in committed files.

For hybrid trace delivery, add:

```bash
TRAIGENT_BACKEND_URL=<backend-url>
TRAIGENT_API_KEY=<scoped-traigent-key>
TRAIGENT_TRACE_ENABLED=true
```

The generator model IDs are sourced from
`tests/cost_coverage/test_model_price_coverage.py`:

- `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- `us.anthropic.claude-3-5-sonnet-20241022-v2:0`
- `us.anthropic.claude-3-haiku-20240307-v1:0`
- `us.amazon.nova-micro-v1:0`
- `us.amazon.nova-lite-v1:0`
- `us.meta.llama3-1-8b-instruct-v1:0`
- `us.meta.llama3-1-70b-instruct-v1:0`

### Bedrock Traps

- Use LiteLLM `bedrock/` model strings for the generator, not `ChatBedrock`, in
  this walkthrough. The LiteLLM path is what Traigent's metadata interceptor
  captures, and it is the path covered by the LiteLLM-shaped mock response
  safety fix.
- Prefer cross-region `us.*` inference profile IDs. Bare legacy IDs such as
  `claude-3-*` can be blocked by Bedrock access policy or regional availability,
  and they are not the profile IDs covered by the cost preflight catalog.
- Titan embeddings go through LiteLLM in a tiny local `Embeddings` wrapper. The
  walkthrough intentionally avoids `langchain-aws`.
- Exact-match accuracy can under-score verbose answers. Keep the real prompt
  terse, or use a contains/semantic scorer for RAG answers. This walkthrough
  uses `semantic_overlap_score`.
- Run mock mode first. `TRAIGENT_MOCK_LLM=true` costs nothing and now returns
  LiteLLM-shaped mock responses, so completion access patterns and token usage
  stay realistic enough for dry-run validation.
- The cost-coverage preflight is intentional. If you add a future Bedrock model
  ID before pricing exists in LiteLLM or the SDK catalog, the run should fail
  before making calls instead of silently treating it as free.

The issue author validated the real recipe with Bedrock Haiku 4.5
(`us.anthropic.claude-haiku-4-5`) at 20/20 for 20 generator calls. This
walkthrough keeps the current repository-curated full profile ID for that model
as the first grid entry.
