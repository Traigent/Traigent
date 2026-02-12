# Research Idea: Generation Quality Grid — When Does the Algorithm Actually Work?

## Expands: EXP-A (Static Anchors) + Paper Experiment 4 (Cold Start)

## Core Logic

All synthetic experiments assume a **perfect item pool** — uniform coverage of difficulty, discrimination always in [0.5, 2.5], any (a, b) combination available on demand. This is unrealistic. In production, items are LLM-generated, and the generator may:

- Miss the requested difficulty target
- Produce items with low discrimination (many items that don't cleanly separate configs)
- Only cover part of the difficulty spectrum
- Degrade in quality over rounds (repetitive or formulaic items)

EXP-A tested whether *static vs generated* items matter. The answer was "barely." But it tested this under ideal synthetic conditions where both static and generated items come from the same perfect pool. The real question isn't where items come from — it's **how good the items are**, and under what quality degradation the algorithm breaks down.

This experiment creates a systematic grid of generation quality conditions to map the boundary between "algorithm works" and "algorithm fails."

## Design

### Independent Variables (Generation Quality Grid)

| Variable | Levels | What it simulates |
|----------|--------|-------------------|
| **Difficulty targeting accuracy** | perfect, ±0.5, ±1.0, ±2.0 | Generator asked for b=1.0, actually produces b∈[0.5,1.5] vs b∈[-1.0,3.0] |
| **Discrimination range** | high [1.5, 2.5], medium [0.8, 1.5], low [0.3, 0.8] | Some domains naturally produce sharper or noisier items |
| **Pool difficulty coverage** | full [-2, 2], narrow [-1, 1], shifted [0, 2], sparse (gaps) | Not all difficulty regions may be reachable in a given domain |
| **Round degradation** | none, 10%/round, 20%/round | Generator produces worse items in later rounds (runs out of ideas) |

Full grid: 4 × 3 × 4 × 3 = 144 cells.

### Implementation

Instead of selecting columns from a uniformly generated pool, **constrain the pool** per condition:

```
For "difficulty targeting accuracy = ±1.0":
    When algorithm requests items near b=1.5,
    actually draw from items with b ∈ [0.5, 2.5] (target ± 1.0 noise)

For "discrimination range = low":
    Generate item pool with a ∈ [0.3, 0.8] only

For "pool coverage = shifted [0, 2]":
    All items have b ∈ [0, 2], no easy items available

For "round degradation = 10%/round":
    Each round, shrink the available a range by 10%
    Round 1: a ∈ [0.5, 2.5], Round 2: a ∈ [0.5, 2.3], etc.
```

### Dependent Variables (Per Phase)

Measure separately for Phase A, Phase B, and Phase C:

- **Phase A**: theta correlation with ground truth
- **Phase B**: elimination count, stall rate
- **Phase C**: rounds to convergence, total evaluations
- **Overall**: identification success rate, total cost

This reveals which phase is most sensitive to which quality dimension.

### Controls
- C=15, epsilon=0.5, delta=0.05
- 100 trials per cell (or 200 for key cells)
- Synthetic 2PL with known ground truth

## Key Questions This Answers

1. **How bad can generation be before Phase B becomes useless?** If targeting accuracy is ±2.0, does Phase B still eliminate anyone?
2. **Is discrimination or difficulty coverage more important?** If you can only improve one aspect of generation quality, which gives more bang?
3. **Does focused generation (Phase C) compensate for poor Phase A?** Or does garbage in → garbage out?
4. **What's the minimum viable generation quality for 90% success?** Directly informs product requirements for the LLM generator.
5. **Where's the cliff?** Is degradation gradual or is there a sharp threshold where the algorithm collapses?

## Expected Findings

- Difficulty targeting accuracy is probably the most important variable — if you can't place items near the right b, focused generation's entire premise breaks
- Low discrimination (a < 0.8) likely hurts Phase B most, because CIs become wide and elimination thresholds aren't met → stalls
- Shifted pool coverage should show the "unlucky Phase A" problem clearly — if all items are hard (b ∈ [0, 2]) but some configs live at theta=-1.0, those configs look identical
- Round degradation probably matters less than initial quality — Phase A sets the trajectory

## Relation to EXP-A

EXP-A asked: "Do static anchor items improve cold start?" Answer: barely, under ideal conditions. This experiment asks the **deeper question**: "Under what generation conditions does the cold start algorithm work at all?" It reframes the problem from item *source* (static vs generated) to item *quality* (how good are they, regardless of source).

If this experiment shows that success drops sharply below a certain generation quality threshold, it strengthens the case for both:
- The smart seed selection idea (pre-screen items before committing)
- Static anchors as a safety net (guaranteed quality floor even if generation is poor)

## Estimated Cost

| Item | Estimate |
|------|----------|
| Implementation | 1-2 days (modify pool generation, add quality constraints) |
| Compute | ~4-8 hours CPU (144 cells × 100 trials, all synthetic) |
| Analysis | 1 day (heatmaps, phase-level breakdowns, threshold identification) |
| API cost | $0 |
