# Discussion 13 — Reminders & Follow-ups

## REMINDER: Sunday 2026-03-16

### Review EXP-C Scaling Results

EXP-C (`paper_experiments/exp_c_scaling`) completed on 2026-03-14.

**Key findings:**
- `irt_rank` is best overall (lowest k95 across all C values)
- `irt_optfind` advantage ratio vs rank: **0.55x–0.67x** (rank beats optfind here)
- Scaling fit: all methods show negative beta — k95 decreases as C grows (expected)
- **Monotonicity**: FAIL — 18/304 violations (minor)
- **Max-k all above 90%**: PASS
- Results in: `paper_experiments/exp_c_scaling/results/`

**TODO Sunday:**
- [ ] Analyze / interpret results for paper
- [ ] Decide if monotonicity violations need addressing
- [ ] Check other exps (exp_b, exp_e, exp_f…) for next to run
