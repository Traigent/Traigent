# Phase B Open Questions: Five Investigations

## Expands: exp_a_when_phase_b_works.md

These five questions emerged from analyzing EXP-A's high stall rates (64-94%) and the paper's Experiment 4 cold start results. Each targets a specific gap in our understanding of when Phase B (Rapid Elimination) is cost-effective.

---

### 1. Elimination Count as a Function of C

**Question**: Does Phase B elimination scale as predicted with configuration count?

Theory predicts Phase B should eliminate configs whose ability is more than ~(CI_best + CI_c + 2×epsilon) below the best. With uniform spacing Delta = R/(C-1), the number eliminated should grow roughly linearly with C. At C=15, ~10 eliminated (paper Experiment 4). At C=100, theory predicts ~65.

**What to measure**: Run Phase B in isolation across C ∈ {10, 15, 25, 50, 100} with fixed epsilon=0.5 and N_A=15. Record elimination count, and compare to theoretical prediction. EXP-C has partial data (C=50, C=100) but doesn't isolate Phase B from Phase C.

**Why it matters**: If the scaling holds, Phase B becomes increasingly essential as C grows — the ROI improves quadratically (more configs eliminated × more rounds saved). If it doesn't scale (e.g., due to IRT estimation degradation at large C), the cold start protocol needs rethinking for production-scale config spaces.

---

### 2. Phase B Sensitivity to Phase A Item Quality

**Question**: How much does Phase A item discrimination (a-values) affect Phase B's ability to eliminate?

CI width after n items ≈ 1/sqrt(n × I_min), where I_min = a²/4 for the worst-case config. With a=2.5, I=1.56 → SE≈0.13 after 40 items. With a=0.5, I=0.0625 → SE≈0.63. The difference is 5× in CI width, which determines whether Phase B can eliminate anything.

**What to measure**: Fix C=15, epsilon=0.5, N_A=15. Vary Phase A item discrimination: a ∈ [0.3, 0.8] (low), [0.8, 1.5] (medium), [1.5, 2.5] (high). Record Phase B elimination count per condition.

**Why it matters**: Ties directly to the generation quality grid idea. If Phase B is viable only with high-a items, then the smart seed selection strategy (pre-screening items for discrimination) becomes a prerequisite for Phase B, not just an optimization.

---

### 3. Optimal Phase B Threshold

**Question**: Is the 2×epsilon relaxation the right elimination threshold for Phase B?

The paper uses 2×epsilon as a design choice — aggressive enough to cut obvious losers, conservative enough to protect near-optimal configs. But this is not derived from a theorem. A tighter threshold (1.5×epsilon) eliminates more configs but risks cutting ones within epsilon of the best. A looser threshold (3×epsilon) is safer but eliminates fewer.

**What to measure**: Run Phase B with thresholds in {1.0, 1.5, 2.0, 2.5, 3.0} × epsilon. Record: (a) elimination count, (b) false elimination rate (did we cut a config within epsilon of the best?), (c) downstream Phase C cost. Find the Pareto frontier of elimination aggressiveness vs. correctness.

**Why it matters**: An adaptive threshold — one that adjusts based on observed CI widths after Phase A — could improve Phase B's ROI across different conditions. For example: if Phase A CIs are tight, use a tighter threshold; if they're wide, use a looser one or skip Phase B entirely.

---

### 4. Phase B with More Phase A Items

**Question**: Would a larger Phase A (25-30 items instead of 15) make Phase B viable at C=15?

CI width scales as 1/sqrt(n). Going from 15 to 30 items tightens CIs by ~30% (1/sqrt(30) vs 1/sqrt(15)). The cost is 15 extra items × 15 configs = 225 additional evaluations. If this enables Phase B to eliminate 5+ configs instead of 0-2, the downstream Phase C savings (5 × 10 × 5 = 250 evaluations) justify the investment.

**What to measure**: Vary N_A ∈ {10, 15, 20, 25, 30, 40} with fixed C=15, epsilon=0.5. Record: (a) Phase A theta correlation (r), (b) Phase B elimination count, (c) total pipeline cost (Phase A + Phase B + Phase C). Find the N_A that minimizes total cost.

**Why it matters**: The paper fixed N_A=15 as a reasonable default, but this may not be optimal. If N_A=25 consistently makes Phase B work and reduces total pipeline cost, the cold start protocol should be updated. This is a low-risk experiment with high practical impact.

---

### 5. Phase B Skip Decision

**Question**: Can the algorithm predict after Phase A whether Phase B will be profitable, and skip it if not?

After Phase A, we have theta estimates and CI widths for all configs. A simple heuristic: compute the maximum gap `max_gap = UCB_worst - LCB_best`. If `max_gap < 2×epsilon`, no config can possibly be eliminated by Phase B — skip it. More sophisticated: estimate the expected number of eliminations from the CI distribution, and compare the expected savings against Phase B's cost (N_B × |A| evaluations).

**What to measure**: Run the full cold start pipeline on many trials. After Phase A, compute the skip heuristic. Compare: (a) total cost with Phase B always on, (b) total cost with Phase B always off, (c) total cost with the skip heuristic. The heuristic should match or beat both fixed policies.

**Why it matters**: EXP-A showed Phase B stalls 64-94% of the time at C=15. Each stall wastes N_B × C = 25 × 15 = 375 evaluations. A skip decision that correctly predicts stalls would save those evaluations in the majority of runs, while still capturing Phase B's benefit in the minority of runs where it works.

---

## Estimated Effort

| Question | Implementation | Compute | Analysis |
|----------|---------------|---------|----------|
| 1. Elimination vs C | 0.5 day | 2 hours | 0.5 day |
| 2. Item quality sensitivity | 0.5 day | 2 hours | 0.5 day |
| 3. Optimal threshold | 0.5 day | 4 hours | 1 day |
| 4. More Phase A items | 0.5 day | 2 hours | 0.5 day |
| 5. Skip decision | 1 day | 2 hours | 1 day |
| **Total** | **3 days** | **12 hours** | **3.5 days** |

All experiments are synthetic (no API cost). Questions 1, 2, and 4 can share infrastructure. Question 5 depends on results from 1-4.
