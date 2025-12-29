# Examples 02–10: Story-Driven Drafts (One Feature Each)

- All examples run in Edge Analytics mode (execution_mode: "edge_analytics").
- Each example tunes 2–3 dimensions via configuration_space.
- Injection modes showcased across the set: context, parameter, seamless.
- Batch/parallel settings are shown where relevant.
- Search strategies demonstrated: grid, random, bayesian.
- Single concise print at end of each `run.py`: `{ "best_config": ..., "best_score": ... }`.

---

## core/few-shot-classification
- Hook: "Ever fought over which k and selection strategy make your few-shot prompt actually work?"
- Use-case: Sentiment classification for ambiguous short texts.
- Feature: Few-shot prompting via parameter injection.
- Execution: Edge Analytics
- Injection: parameter (config_param: "config")
- Algorithm: grid
- Dimensions:
  - `k`: [0, 2, 4]
  - `selection_strategy`: ["top_k", "diverse"]
  - `temperature`: [0.0, 0.3]
- Files: `evaluation_set.jsonl`, `prompt.txt`, `example_set.jsonl`
- Eval: Exact label match accuracy (built-in)
- Batch/Parallel: sequential
- Notes: In `run.py`, function signature includes `config` param; inside, read `config["k"]`, etc.

---

## core/multi-objective-tradeoff
- Hook: "Balancing accuracy against cost and tokens—what’s the sweet spot for your QA?"
- Use-case: Short QA with budget sensitivity.
- Feature: Multi-objective optimization (accuracy vs cost).
- Execution: Edge Analytics
- Injection: seamless (auto_override_frameworks=True)
- Algorithm: bayesian
- Dimensions:
  - `model`: ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
  - `temperature`: [0.0, 0.2]
  - `max_tokens`: [64, 128]
- Files: `evaluation_set.jsonl`, `prompt.txt` (optional `context_documents.jsonl`)
- Eval: Objective weights (e.g., accuracy 0.7, cost 0.3) defined with an ObjectiveSchema
- Batch/Parallel: `parallel_config.trial_concurrency = 2` (light parallelism)
- Notes: Demonstrates multi-objective trade-off and shows final best weighted config.

---

## core/token-budget-summarization
- Hook: "Need tight summaries under a strict token budget without losing key facts?"
- Use-case: Meeting-note summarization with keyphrase retention.
- Feature: Token budget control (parameter injection of `max_tokens`).
- Execution: Edge Analytics
- Injection: context (default)
- Algorithm: random
- Dimensions:
  - `max_tokens`: [64, 96, 128]
  - `temperature`: [0.0, 0.2]
  - `style`: ["bulleted", "paragraph"]
- Files: `evaluation_set.jsonl` (transcripts + keyphrase truth), `prompt.txt`
- Eval: Keyphrase coverage accuracy
- Batch/Parallel: `parallel_config.example_concurrency` ∈ {2, 4} (demonstrate batched evaluation vs default sequential)
- Notes: Uses `parallel_config` with `example_concurrency` to process examples in small batches.

---

## core/structured-output-json
- Hook: "Struggling to get clean, valid JSON back from messy text?"
- Use-case: Field extraction from mini-invoice snippets.
- Feature: Custom measure (custom evaluator) for JSON validity and field accuracy.
- Execution: Edge Analytics
- Injection: seamless (auto framework override)
- Algorithm: grid
- Dimensions:
  - `temperature`: [0.0, 0.2]
  - `format_hint`: ["strict_json", "relaxed_json"]
  - `schema_rigidity`: ["strict", "lenient"]
- Files: `evaluation_set.jsonl` (text + expected fields), `prompt.txt`
- Eval: Custom evaluator that checks JSON parsability and field presence/accuracy (composite score)
- Batch/Parallel: sequential
- Notes: Shows `custom_evaluator=` in decorator; isolates the custom metric feature.

---

## core/tool-use-calculator
- Hook: "Tired of arithmetic slips? Toggle tool use and watch math answers stabilize."
- Use-case: Basic math QA that benefits from a calculator.
- Feature: Parameter injection (toggle tool use) + strategy API usage.
- Execution: Edge Analytics
- Injection: parameter (config_param: "config")
- Algorithm: random (via strategy object)
- Dimensions:
  - `use_tool`: [True, False]
  - `temperature`: [0.0, 0.2]
  - `max_tool_calls`: [1, 2]
- Files: `evaluation_set.jsonl` (expr + numeric truth), `prompt.txt`
- Eval: Exact numeric accuracy
- Batch/Parallel: `parallel_config.trial_concurrency = 2`
- Notes: Demonstrates `strategy = traigent.set_strategy(algorithm="random", parallel_workers=2)` and passing to `.optimize(strategy=strategy)`.

---

## core/prompt-style-optimization
- Hook: "Battling tone and style? Bullet points or narrative—let the data decide."
- Use-case: Email drafting with explicit style requirements.
- Feature: Seamless injection (style/tone parameters injected without function signature changes).
- Execution: Edge Analytics
- Injection: seamless
- Algorithm: bayesian
- Dimensions:
  - `style`: ["bulleted", "paragraph"]
  - `tone`: ["formal", "friendly"]
  - `temperature`: [0.0, 0.2]
- Files: `evaluation_set.jsonl` (briefs + expected style markers), `prompt.txt`, optional `example_set.jsonl`
- Eval: Style compliance accuracy (detect bullets or paragraph continuity and tone keywords)
- Batch/Parallel: sequential
- Notes: Highlights seamless override and how Traigent manipulates parameters without changing code.

---

## core/chunking-long-context
- Hook: "Long docs, shallow answers? Tune chunking and top_k to lift grounded QA."
- Use-case: Long-context QA over multiple documents.
- Feature: RAG pipeline chunking/windowing optimization.
- Execution: Edge Analytics
- Injection: context (default)
- Algorithm: grid
- Dimensions:
  - `chunk_size`: [256, 512, 768]
  - `overlap`: [16, 32]
  - `top_k`: [2, 3]
- Files: `evaluation_set.jsonl`, `prompt.txt`, `context_documents.jsonl` (larger doc set)
- Eval: Exact answer accuracy
- Batch/Parallel: Demonstrate `parallel_config.trial_concurrency` switching between 1 and 4 by running once with sequential and once with parallel in code comments or via strategy param
- Notes: Clear isolation of RAG-chunking feature knobs.

---

## core/safety-guardrails
- Hook: "Need consistent refusals on unsafe prompts without over-blocking legit queries?"
- Use-case: Refusal behavior on toxic/PII prompts.
- Feature: Custom measure (custom evaluator) for refusal correctness; policy strength tuning.
- Execution: Edge Analytics
- Injection: context (default)
- Algorithm: random
- Dimensions:
  - `safety_strength`: ["low", "medium", "high"]
  - `refusal_style`: ["brief", "policy_cite"]
  - `temperature`: [0.0]
- Files: `evaluation_set.jsonl` (unsafe prompts + expected refusal), `prompt.txt` (policy)
- Eval: Custom evaluator that checks for refusal markers and policy alignment
- Batch/Parallel: sequential
- Notes: Keeps the example focused on custom evaluation criteria.

---

## core/prompt-ab-test
- Hook: "Prompt A vs B—stop guessing; run an A/B that picks a winner."
- Use-case: Choosing the best performing prompt template for QA.
- Feature: Prompt template selection via parameter injection.
- Execution: Edge Analytics
- Injection: context (default)
- Algorithm: grid
- Dimensions:
  - `prompt_variant`: ["a", "b"]
  - `model`: ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"]
  - `temperature`: [0.0, 0.2]
- Files: `evaluation_set.jsonl`, `prompt_a.txt`, `prompt_b.txt`, optional `example_set.jsonl`
- Eval: Exact match accuracy
- Batch/Parallel: `parallel_config.trial_concurrency = 2` (small speed-up)
- Notes: Classic A/B prompt bake-off with a minimal, decisive output.

---

## Coverage Matrix (Quick Check)
- Execution Mode: All "edge_analytics" ✅
- Injection Modes: context (04,08,09,10), parameter (02,06), seamless (03,05,07) ✅
- Parallel/Batch: example_concurrency (04), trial_concurrency (03,06,08,10) ✅
- Multi-Objective: 03 ✅
- Custom Measure: 05, 09 ✅
- Search Strategies: grid (02,05,08,10), random (04,06,09), bayesian (03,07) ✅
- Optimize API Surface: decorator kwargs (algorithm, execution_mode, parallel_config), StrategyConfig via set_strategy (06) ✅
