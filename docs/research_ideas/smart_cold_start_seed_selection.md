# Research Idea: Smart Cold Start Seed Selection

## Problem

Phase A cold start currently uses a **fixed** difficulty split (5 easy / 5 medium / 5 hard) to generate 15 seed items. This is static and naive — if the LLM generator produces items that don't actually match the requested difficulty, or if the difficulty distribution doesn't suit the specific config space, Phase A can get "unlucky":

- All items too easy → every config scores ~100%, no discrimination, IRT can't differentiate
- All items too hard → every config scores ~0%, same problem
- Items cluster in one difficulty region → wastes information budget on uninformative comparisons
- Phase B then stalls (EXP-A showed 64-94% stall rates across all arms)

## Idea

Replace the static 5-5-5 split with a **smarter, adaptive seed selection system**. Two complementary approaches:

### Approach 1: Embedding-Based Diversity Selection

Before scoring, use an embedding model to ensure seed items are **semantically diverse** — not just difficulty-diverse:

- Embed candidate items (generated or from a pool) into a vector space
- Select items that maximize coverage (e.g., max-min distance, DPP, or k-medoids)
- This prevents redundant items that test the same skill/topic

Lightweight — embedding is cheap compared to full LLM scoring of all configs.

### Approach 2: Lightweight Evaluator Pre-Screening

Use a small, fast model (or heuristic) to **pre-estimate item difficulty** before committing to full scoring:

- Generate N candidate items (e.g., 30-50)
- Run a cheap pre-screen: single fast model scores them, or use perplexity as a difficulty proxy
- Select the 15 that best span the difficulty spectrum based on pre-screen results
- Then proceed with full scoring on all C configs

This is like a "Phase 0" that costs almost nothing but dramatically improves Phase A quality.

### Approach 3: Adaptive Phase A (Mini-Rounds)

Instead of generating all 15 items at once, do Phase A in mini-rounds:

1. Generate 5 items (broad difficulty)
2. Score all configs on these 5
3. Fit preliminary IRT model
4. Analyze: are items too easy? too hard? clustered?
5. Generate next 5 items **targeting the detected gap**
6. Repeat once more
7. Total: still 15 items, but adaptively targeted

This is essentially applying focused generation (currently Phase B/C only) to Phase A itself.

### Approach 4: Seed Quality Gate

After Phase A scoring but before Phase B, run a quality check:

- Compute score variance across configs: if < threshold, items aren't discriminating
- Check difficulty distribution: if all items have pass rate > 0.9 or < 0.1, flag as poor
- If quality gate fails: generate replacement items targeting the gap, re-score, before proceeding

This prevents the downstream cascade where bad Phase A → Phase B stall → wasted rounds.

## What Already Exists

| Component | Status | Location |
|-----------|--------|----------|
| Fixed 5-5-5 difficulty split | Implemented | `cold_start.py` Phase A |
| Difficulty-stratified sampling from pool | Implemented | `cold_start.py` (pass-rate binning) |
| Static anchor items | Experimental (EXP-A) | Tests mixing pre-calibrated items |
| Focused generation (gap analysis) | Implemented for Phase B/C | `focused_generation.py` |
| Embedding-based selection | Not implemented | — |
| Lightweight pre-screening | Not implemented | — |
| Adaptive Phase A | Not implemented | — |
| Seed quality gate | Not implemented | — |

## Expected Impact

- Reduce Phase B stall rate from ~80% to ~30-40%
- Improve Phase A theta correlation from r=0.91 to r=0.95+
- Reduce total rounds to convergence by 1-2 rounds
- Reduce total cost by 15-25%

## Relation to EXP-A Results

EXP-A showed that static anchors barely improved calibration (r=0.906 → 0.909). This suggests the problem isn't about *where* the items come from (static vs generated) but about *how well they cover the difficulty space*. Smart selection addresses the root cause — poor difficulty coverage — regardless of item source.

## Next Steps

- [ ] Prototype Approach 3 (Adaptive Phase A) on synthetic data — lowest implementation cost, reuses existing focused generation code
- [ ] Benchmark against current fixed 5-5-5 baseline
- [ ] If promising, test Approach 2 (pre-screening) as an alternative
- [ ] Approach 1 (embeddings) is most relevant for the live/online setting where items are text
