# TVL 0.9 Video Scripts

These demo scripts showcase the new TVL 0.9 features in TraiGent. They follow the same format as the existing `demo-optimize.sh` and `demo-hooks.sh` scripts.

## Overview

| Script | Duration | Topic |
|--------|----------|-------|
| `demo-tvl-basics.sh` | ~45 sec | TVL 0.9 basics: typed TVARs, domain specs, multi-objective |
| `demo-tvl-promotion.sh` | ~60 sec | Promotion policy: epsilon-Pareto, statistical testing, BH correction |
| `demo-tvl-banded.sh` | ~50 sec | Banded objectives: TOST equivalence testing |

## Video 1: TVL 0.9 Basics

**File**: `demo-tvl-basics.sh`

**Key concepts**:
- TVL as a declarative optimization language
- Typed TVARs: `bool`, `int`, `float`, `enum[str]`, `tuple[...]`, `callable[Proto]`
- Domain specifications: ranges, enums, resolution
- Multi-objective optimization
- Loading specs with `load_tvl_spec()`
- Using `@traigent.optimize(tvl_spec="...")`

**Target audience**: New TraiGent users learning TVL

---

## Video 2: Promotion Policy

**File**: `demo-tvl-promotion.sh`

**Key concepts**:
- Why statistical testing matters for LLM optimization
- Epsilon-Pareto dominance (meaningful improvement thresholds)
- Alpha level and confidence (typically 0.05 for 95% confidence)
- Benjamini-Hochberg correction for multiple comparisons
- Chance constraints: P(metric >= threshold) >= confidence
- Using `PromotionGate` for deployment decisions

**Target audience**: ML engineers deploying optimized configs to production

---

## Video 3: Banded Objectives

**File**: `demo-tvl-banded.sh`

**Key concepts**:
- When "good enough" is better than "maximum"
- Banded objectives: target ranges instead of directions
- Two specification formats: `{low, high}` or `{center, tol}`
- TOST (Two One-Sided Tests) for equivalence testing
- Using `BandedObjectiveSpec` for evaluation
- Real-world use cases: SLOs, budgets, compliance

**Target audience**: Teams with specific performance requirements/SLOs

---

## Generating Demo Videos

### Option 1: Run Scripts Directly
```bash
cd demos
./scripts/demo-tvl-basics.sh
./scripts/demo-tvl-promotion.sh
./scripts/demo-tvl-banded.sh
```

### Option 2: Generate Animated SVGs

Update `demos/scripts/generate-cast.py` to include the new scripts:

```python
demos = [
    ('demo-optimize.sh', 'optimize.cast', 'Traigent LLM Optimization'),
    ('demo-hooks.sh', 'hooks.cast', 'Traigent Hooks & Callbacks'),
    ('demo-tvl-basics.sh', 'tvl-basics.cast', 'TVL 0.9 Typed Variables'),
    ('demo-tvl-promotion.sh', 'tvl-promotion.cast', 'TVL 0.9 Promotion Policy'),
    ('demo-tvl-banded.sh', 'tvl-banded.cast', 'TVL 0.9 Banded Objectives'),
]
```

Then run:
```bash
cd demos
./record-demos.sh
```

### Option 3: Record with asciinema

```bash
asciinema rec -c "./scripts/demo-tvl-basics.sh" output/tvl-basics.cast
svg-term --in output/tvl-basics.cast --out output/tvl-basics.svg
```

---

## Embedding in Documentation

```markdown
## TVL 0.9 Features

### Typed Tuned Variables
![TVL Basics](demos/output/tvl-basics.svg)

### Production Promotion Policy
![TVL Promotion](demos/output/tvl-promotion.svg)

### Banded Objectives
![TVL Banded](demos/output/tvl-banded.svg)
```

---

## Script Structure

Each script follows the standard format:
1. **Clear** the screen and show title
2. **Explain** the problem/concept
3. **Show** the TVL YAML spec
4. **Demonstrate** Python usage
5. **Simulate** running output
6. **Summarize** key takeaways

Sleep timings are calibrated for readability when converted to animated SVG.
