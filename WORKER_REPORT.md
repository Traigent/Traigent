# Worker Report W3 - SDK metric/evaluator recommendation catalog

## Changes

- Added `traigent/evaluators/catalog/metric_eval_catalog.v1.json` with 14 active metric/evaluator recommendation entries across `code_gen`, `rag`, and `general`.
- Added `traigent/evaluators/catalog/schemas/metric_eval_catalog_entry_schema.json` with strict entry validation, canonical enum constraints, `additionalProperties: false`, and `llm_based` -> `cost_note` conditional requirement.
- Added `traigent/evaluators/catalog_loader.py` for cached catalog loading, per-entry JSON Schema validation, `catalog_entries()`, and `catalog_version()`.
- Added `traigent/evaluators/recommendations.py` with `EVAL_RECOMMENDATION_CAVEAT`, `list_eval_recommendation_task_types()`, `recommend_metrics()`, and `recommend_evaluator()`.
- Exported the recommendation API from `traigent.evaluators`.
- Added `traigent recommend-eval <task_type>` with `--list-types`, repeatable `--measure-type`, `--evaluator`, and `--json`.
- Added tests for schema validation, canonical enum locks, built-in metric registry binding, LLM cost notes, public API filters/errors/ranking, and CLI smoke coverage.

## Verification Commands

Command:
`PYTHONPATH=$PWD /tmp/venv-sdk/bin/python -m pytest tests/unit/evaluators/test_metric_eval_catalog.py tests/unit/evaluators/test_metric_eval_recommendations.py tests/unit/cli/test_recommend_eval_command.py -n0 -q`

Result:
`15 passed in 0.25s`

Command:
`PYTHONPATH=$PWD /tmp/venv-sdk/bin/python -m pytest tests/unit/config_generator/test_recommendations_api.py tests/unit/config_generator/test_knob_pack_rows.py tests/unit/cli/test_recommend_command.py tests/cli -n0 -q`

Result:
`71 passed in 0.55s`

Command:
`/tmp/venv-sdk/bin/ruff format traigent/evaluators/catalog_loader.py traigent/evaluators/recommendations.py tests/unit/evaluators/test_metric_eval_catalog.py tests/unit/evaluators/test_metric_eval_recommendations.py tests/unit/cli/test_recommend_eval_command.py && /tmp/venv-sdk/bin/ruff check traigent/evaluators/catalog_loader.py traigent/evaluators/recommendations.py tests/unit/evaluators/test_metric_eval_catalog.py tests/unit/evaluators/test_metric_eval_recommendations.py tests/unit/cli/test_recommend_eval_command.py`

Result:
`5 files left unchanged`
`All checks passed!`

Command:
`/tmp/venv-sdk/bin/ruff check traigent/cli/main.py traigent/evaluators/__init__.py`

Result:
`All checks passed!`

Command:
`PYTHONPATH=$PWD /tmp/venv-sdk/bin/mypy traigent/evaluators/recommendations.py`

Result:
`Success: no issues found in 1 source file`

Command:
`git diff --check`

Result:
No output.

## Catalog Entries For Captain Verification

- `code_gen.execution_accuracy.spider.v1` - Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task (Yu et al., 2018)
- `code_gen.pass_at_k.humaneval.v1` - Evaluating Large Language Models Trained on Code (Chen et al., 2021)
- `code_gen.syntax_validity.v1` - Compiler or parser preflight before executable code evaluation
- `code_gen.unit_test_success_rate.v1` - Executable test harness pass/fail aggregation
- `rag.answer_exact_match.squad.v1` - SQuAD: 100,000+ Questions for Machine Comprehension of Text (Rajpurkar et al., 2016)
- `rag.token_f1.squad.v1` - SQuAD: 100,000+ Questions for Machine Comprehension of Text (Rajpurkar et al., 2016)
- `rag.faithfulness.ragas.v1` - RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)
- `rag.answer_relevance.ragas.v1` - RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)
- `rag.latency_p95.v1` - Service-level latency percentile monitoring
- `general.bertscore_f1.v1` - BERTScore: Evaluating Text Generation with BERT (Zhang et al., 2020)
- `general.llm_judge_mt_bench.v1` - Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)
- `general.geval_quality.v1` - G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (Liu et al., 2023)
- `general.cost_per_success.v1` - Cost-normalized success tracking for model and configuration selection
- `general.safety_refusal_check.v1` - Policy regression suite for safety-case fixtures

## Deferred Items

- None.

## Reviewer Risks

- The catalog is recommendation metadata. Entries with `custom_template` or `llm_judge_template` still require callers to bind the described metric function or judge callback before runtime use.
- `max_cost_tier` in `recommend_evaluator()` is derived from evaluation method because the requested catalog shape did not include a cost-tier field.
- RAGAS and LLM-judge entries intentionally carry medium evidence-strength labels and explicit cost notes because evaluator fit depends on judge model, prompt, and dataset shape.
