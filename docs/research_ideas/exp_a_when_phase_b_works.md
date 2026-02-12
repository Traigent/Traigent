# Research Idea: When Does Phase B Actually Work?

## Context

Phase B (Rapid Elimination) is supposed to quickly cut obviously bad configs after Phase A's initial calibration. EXP-A showed stall rates of 64-94% — Phase B almost always fails to eliminate ≥2 configs at C=15 with 15 seed items. But the conditions tested may not represent the cases where Phase B is most valuable.

## Phase B Works When

- **Wide ability spread**: Configs range from clearly bad to clearly good. After 15 items, confidence intervals are tight enough to rule out the bottom.
- **Large C**: With C=50 or C=100, there are many obvious losers. Phase B can wipe 50%+ of configs in one pass, making Phase C dramatically cheaper.
- **High discrimination items in Phase A**: If seed items happen to have high `a`, CIs are tighter, more configs can be confidently eliminated.
- **Clustered bottom, separated top**: Most configs are mediocre, a few are clearly strong. The mediocre pile gets cut fast.

## Phase B Stalls When

- **Tightly packed thetas**: All configs have similar ability. CIs overlap, nobody can be ruled out.
- **Low discrimination items**: Noisy items → wide CIs → everything looks the same.
- **Small C**: Fewer obvious losers to eliminate.
- **Mismatched difficulty**: Phase A items don't match the config ability range (all too easy or all too hard), so theta estimates are unreliable.
- **Only 15 items**: Not enough data for tight CIs at the 2×epsilon threshold.

## Why This Matters

Phase B's value is proportional to how many configs it eliminates. Each config eliminated saves `items_per_round` evaluations in every subsequent Phase C round. At C=100 with 5 Phase C rounds of 15 items each, eliminating 50 configs in Phase B saves 50 × 15 × 5 = 3,750 evaluations. At C=15, eliminating 2 configs saves 2 × 15 × 4 = 120 evaluations. The ROI scales with C.

## What To Investigate

The current experiments don't systematically map Phase B effectiveness across conditions. Worth exploring:

1. **Phase B elimination count as a function of C** — does it scale as expected? (EXP-C has some data but doesn't isolate Phase B)
2. **Phase B sensitivity to Phase A item quality** — ties to the generation quality grid idea
3. **Optimal Phase B threshold** — is 2×epsilon the right relaxation? Could adaptive thresholds improve elimination?
4. **Phase B with more items** — what if Phase A used 25 or 30 items? Would the tighter CIs make Phase B viable at C=15?
5. **Phase B skip decision** — if Phase B is predicted to stall (e.g., Phase A CIs are too wide), should the algorithm skip it entirely and go straight to Phase C?

## Relation to Other Ideas

- **Smart seed selection**: Better Phase A items → tighter CIs → Phase B works more often
- **Generation quality grid**: Maps the item quality threshold below which Phase B becomes useless
- **EXP-A results**: Showed Phase B is marginal at C=15/15 items — this idea asks "where exactly does it become useful?"
