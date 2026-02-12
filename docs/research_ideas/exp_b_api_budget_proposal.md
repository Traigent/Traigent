# EXP-B Live Cold Start: API Access Decision

## What This Is

We need to run the IRT-OptFind paper's **missing live validation experiment** — the one gap the paper itself acknowledges. All 7 existing experiments use either synthetic data or retrospective (offline) evaluation. No experiment runs the full Algorithm 3 (Phase A → Phase B → Phase C) prospectively against real LLMs with real API calls.

We've built the infrastructure to do this. It's ready to run. The blocker is an API rate limit.

## What's Ready

| Component | Status |
|-----------|--------|
| IRT core (fitting, identification, generation) | Cloned from TraigentPaper, imports verified |
| Full cold start pipeline (exp9_live_cold_start.py) | 1,375 lines, complete Algorithm 3 |
| OpenRouter client (exp8_live_validation.py) | Rate limiting, caching, retry logic |
| Model discovery script | 15 free models discovered and validated |
| Multi-run orchestrator | Resume support, per-run checkpointing |
| Analysis/aggregation code | Identification success rate, regret, Phase B elimination stats |

### Models Discovered (15 free-tier LLMs on OpenRouter)

| Model | Family | Size | Type |
|-------|--------|------|------|
| google/gemma-3-27b-it:free | Google Gemma | 27B | standard |
| google/gemma-3-12b-it:free | Google Gemma | 12B | standard |
| google/gemma-3n-e4b-it:free | Google Gemma | 4B | standard |
| google/gemma-3n-e2b-it:free | Google Gemma | 2B | standard |
| google/gemma-3-4b-it:free | Google Gemma | 4B | standard |
| arcee-ai/trinity-large-preview:free | Arcee AI | 30B | standard |
| liquid/lfm-2.5-1.2b-instruct:free | Liquid | 1.2B | standard |
| openrouter/free | OpenRouter | ~7B | standard |
| openrouter/aurora-alpha | OpenRouter | ~7B | standard |
| deepseek/deepseek-r1-0528:free | DeepSeek | 671B | reasoning |
| nvidia/nemotron-nano-9b-v2:free | Nvidia | 9B | reasoning |
| nvidia/nemotron-3-nano-30b-a3b:free | Nvidia | 30B | reasoning |
| nvidia/nemotron-nano-12b-v2-vl:free | Nvidia | 12B | reasoning |
| stepfun/step-3.5-flash:free | StepFun | ~7B | reasoning |
| z-ai/glm-4.5-air:free | Z-AI/Zhipu | ~7B | reasoning |

All models confirmed responding as of Feb 11, 2026. Six distinct model families. Size range: 1.2B to 671B parameters.

## The Blocker

**OpenRouter's free-tier rate limit: 50 API requests per day.**

One experiment run (Algorithm 3 + verification) requires ~844 API calls:

| Phase | What happens | API calls |
|-------|-------------|-----------|
| Phase A (Seed) | Generate 15 broad items, score on all 15 models | ~228 |
| Phase B (Elimination) | Generate 25 focused items, score on active models | ~255 |
| Phase C (Refinement) | 3 rounds of 10 items each, shrinking active set | ~146 |
| Verification | Fill remaining score matrix cells | ~215 |
| **Total per run** | | **~844** |

At 50 requests/day, one run takes **~17 days**. A statistically meaningful experiment (10 runs) would take **~170 days**. This is not feasible.

**All API calls are to $0-cost models.** No per-call charges. The only constraint is the daily request count.

## How OpenRouter's Credit System Works

OpenRouter uses a cumulative purchase threshold to unlock higher rate limits:

- **< $10 lifetime purchases**: 50 free-model requests/day
- **>= $10 lifetime purchases**: 1,000 free-model requests/day

Key details:
- The $10 is a **one-time purchase**, not a recurring charge
- Free model calls remain **$0 forever** — no credits are consumed
- The $10 in credits sits in the account (can be used for paid models later, or left idle)
- The rate limit upgrade is **permanent** once the threshold is met

## Options

### Option A: $10 Credits + Lean Scope (Recommended for fast turnaround)
- **Cost**: $10 one-time
- **Scope**: 15 models, 3 independent runs, skip verification phase
- **API calls**: ~1,887 total
- **Timeline**: ~2 days
- **What we get**: Identification success rate (with binomial CI), mean regret, Phase B elimination behavior

### Option B: $10 Credits + Full Scope
- **Cost**: $10 one-time
- **Scope**: 15 models, 10 independent runs, with verification
- **API calls**: ~8,440 total
- **Timeline**: ~9 days
- **What we get**: Statistically robust results matching paper's experimental rigor

### Option C: Free Tier, Minimal
- **Cost**: $0
- **Scope**: 9 models, 1 run, skip verification
- **API calls**: ~390
- **Timeline**: ~8 days
- **What we get**: Pipeline validation only. Not publishable — too few runs, no confidence intervals.

### Option D: Free Tier, Full
- **Cost**: $0
- **Scope**: 15 models, 10 runs, with verification
- **API calls**: ~8,440
- **Timeline**: ~170 days
- **Not feasible.**

## What We Get From This Experiment

This is the only experiment that validates the paper's core contribution end-to-end with real systems:

1. **Identification success rate**: Does Algorithm 3 actually find the epsilon-optimal LLM config? (Paper claims >=95% with delta=0.05)
2. **Sample efficiency**: How many items/evaluations does cold start actually need vs. the theoretical bound?
3. **Phase B effectiveness**: Does rapid elimination actually cut configs in practice, or does it stall (as EXP-A suggests at C=15)?
4. **Real-world noise**: LLM responses have formatting variance, API errors, and non-IRT behavior. Does IRT still work?
5. **Model ranking sanity**: Do larger models (27B, 30B) actually rank higher than smaller ones (1.2B, 2B)?

No existing experiment in the paper or supplementary material answers these questions with live API calls.

## Summary

| | Cost | Timeline | Statistical power |
|-|------|----------|-------------------|
| **Option A** | $10 | 2 days | Moderate (3 runs) |
| **Option B** | $10 | 9 days | High (10 runs) |
| **Option C** | $0 | 8 days | None (1 run) |
| **Option D** | $0 | 170 days | N/A |

The $10 is not consumed — it unlocks a gate. All actual LLM calls are free.
