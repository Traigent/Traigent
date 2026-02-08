# What Can You Optimize?

Traigent optimizes any parameter you place in `configuration_space`. This includes LLM
settings like model and temperature, but also prompts, few-shot strategies, RAG retrieval
settings, output formats, and arbitrary application parameters.

This guide covers what you can tune today with the core SDK and how configuration
reaches your code.

---

## Four Ways Configuration Reaches Your Code

Traigent has three injection modes (context, parameter, seamless) plus one additive
integration path (framework overrides).

| Pattern | How It Works | You Need To | Notes |
| ------- | ------------ | ----------- | ----- |
| **Context** (default) | Traigent stores config in a thread-safe context variable | Call `traigent.get_config()` in your function | Works for all parameter types |
| **Parameter** | Traigent passes config as a function argument | Add a `config` parameter to your function | Set `injection_mode="parameter"` |
| **Seamless** | Traigent rewrites variable assignments via AST transform | Nothing — variable names matching config keys are replaced | Set `injection_mode="seamless"` |
| **Framework overrides** *(additive)* | Traigent monkey-patches LLM SDK constructors and methods | Set `framework_targets=["openai.OpenAI", ...]` in the decorator | Applies on top of any injection mode above |

Framework overrides are available when override behavior is enabled and explicit
framework targets are configured. Without `framework_targets`, overrides do not activate.

For full injection mode details, see [Injection Modes](injection_modes.md).

---

## LLM Parameters

The most common optimization target. Traigent searches across models, temperatures,
and generation settings to find the configuration that best meets your objectives.

```python
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [256, 512],
    },
    eval_dataset="eval/qa.jsonl",
    objectives=["accuracy", "cost"],
)
def answer(question: str) -> str:
    cfg = traigent.get_config()
    response = completion(model=cfg["model"], temperature=cfg["temperature"], ...)
    return response.choices[0].message.content
```

**Core presets:** `Choices.model()`, `Range.temperature()`, `Range.top_p()`,
`IntRange.max_tokens()`, `Range.frequency_penalty()`, `Range.presence_penalty()`

See [Configuration Spaces](configuration-spaces.md) for the full preset reference.

---

## Prompts and Instructions

Place prompt variants in `configuration_space` as `Choices`, then retrieve and
apply them in your function. This lets Traigent compare system prompts, reasoning
strategies, tone, or style across trials.

```python
@traigent.optimize(
    configuration_space={
        "system_prompt": [
            "Answer directly and concisely.",
            "Think step by step before answering.",
            "You are a domain expert. Be thorough.",
        ],
        "model": ["gpt-4o-mini", "gpt-4o"],
    },
    eval_dataset="eval/qa.jsonl",
    objectives=["accuracy"],
)
def qa_agent(question: str) -> str:
    cfg = traigent.get_config()
    return completion(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": question},
        ],
    ).choices[0].message.content
```

**Core preset:** `Choices.prompting_strategy()` returns
`["direct", "chain_of_thought", "react", "self_consistency"]` as a ready-made
configuration parameter.

**Examples:**
[prompt-ab-test](../../examples/core/prompt-ab-test/),
[prompt-style-optimization](../../examples/core/prompt-style-optimization/),
[walkthrough/advanced/02_prompt_optimization.py](../../walkthrough/real/advanced/02_prompt_optimization.py)

---

## Few-Shot Selection

Optimize how many examples to include in a prompt, and which selection strategy to use.

```python
@traigent.optimize(
    configuration_space={
        "k": [0, 2, 4],
        "selection_strategy": ["top_k", "diverse"],
        "model": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
    },
    eval_dataset="eval/sentiment.jsonl",
    objectives=["accuracy"],
)
def classify(text: str) -> str:
    cfg = traigent.get_config()
    examples = select_examples(k=cfg["k"], strategy=cfg["selection_strategy"])
    prompt = build_prompt(text, examples)
    return llm.invoke(prompt)
```

**Core preset:** `IntRange.few_shot_count(max_examples=10)` provides a ready-made
range for the number of in-context examples.

**Example:** [few-shot-classification](../../examples/core/few-shot-classification/)

---

## RAG Pipeline

Tune retrieval parameters — chunk size, retriever type, reranking model, embedding
model, and context formatting — alongside LLM settings in a single optimization run.

```python
@traigent.optimize(
    configuration_space={
        "chunk_size": [256, 512, 1024],
        "retriever": ["similarity", "mmr", "bm25"],
        "reranker": ["none", "cohere-rerank-v3"],
        "model": ["gpt-4o-mini"],
    },
    eval_dataset="eval/rag_questions.jsonl",
    objectives=["accuracy", "latency"],
)
def rag_answer(query: str) -> str:
    cfg = traigent.get_config()
    docs = retrieve(query, method=cfg["retriever"], top_k=5)
    if cfg["reranker"] != "none":
        docs = rerank(docs, model=cfg["reranker"])
    return generate(query, docs, model=cfg["model"])
```

**Core presets:**
`IntRange.chunk_size()`, `IntRange.k_retrieval()`,
`Choices.retriever_type()`, `Choices.reranker_model()`,
`Choices.embedding_model()`, `Choices.context_format()`

See [Configuration Spaces](configuration-spaces.md) for the values each preset returns.

**Examples:**
[chunking-long-context](../../examples/core/chunking-long-context/),
[context engineering (advanced)](../../examples/advanced/ai-engineering-tasks/p0_context_engineering/)

---

## Any Application Parameter

Traigent is not limited to LLM settings. Any parameter you place in
`configuration_space` becomes a tunable variable — routing strategies, risk factors,
thresholds, concurrency settings, or domain-specific knobs.

```python
@traigent.optimize(
    configuration_space={
        "strategy": ["conservative", "balanced", "aggressive"],
        "risk_factor": [0.1, 0.5, 0.9],
        "max_exposure": [1000, 5000, 10000],
    },
    eval_dataset="eval/trades.jsonl",
    objectives=["returns"],
)
def trading_algorithm(market_data):
    cfg = traigent.get_config()
    return compute_positions(market_data, cfg["strategy"], cfg["risk_factor"])
```

**Examples:**
[structured-output-json](../../examples/core/structured-output-json/),
[safety-guardrails](../../examples/core/safety-guardrails/),
[tool-use-calculator](../../examples/core/tool-use-calculator/)

---

## Optional Plugin Presets

The `traigent-tuned-variables` plugin adds domain-specific preset collections
(`RAGPresets`, `PromptingPresets`) and post-optimization variable analysis.
Install with:

```bash
pip install traigent-tuned-variables
```

See [Tuned Variables](tuned_variables.md) for preset details and usage examples.
