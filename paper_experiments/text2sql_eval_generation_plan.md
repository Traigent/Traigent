# POC: Evaluation Set Generation Improves Tuning Results

## Objective
Demonstrate that generating a quality evaluation set from an anchor set **before** tuning leads to better optimization results than direct tuning on a random sample.

## User Decisions
- **Spider**: User has it locally with full execution evaluation (SQLite DBs)
- **Budget**: $20-50 for real LLM experiments
- **Algorithm**: Greedy selection from candidate pool (not LLM-generated variations)

---

## Experimental Design

### Hypothesis
Given a fixed tuning budget:
1. **Baseline**: Tune agent on randomly sampled evaluation set → Test Accuracy A₁
2. **Treatment**: Generate optimized eval set from anchor, then tune → Test Accuracy A₂

**Expected**: A₂ > A₁ (treatment outperforms baseline)

### Data Splits (Stratified by Hardness: easy/medium/hard/extra)
```
Spider Dataset (~1,000+ queries)
    ├── Anchor Set A (100 examples) → Used for eval set generation scoring
    ├── Candidate Pool C (~800 examples) → Source for selection
    └── Test Set T (200 examples) → Held-out for final evaluation
```

### Key Metrics
1. **Primary**: Test set execution accuracy (baseline vs treatment)
2. **Eval Set Quality**: Separation power, anchor alignment
3. **Cost**: Total tuning cost, eval generation overhead

---

## File Structure

```
paper_experiments/text2sql_eval_generation/
├── __init__.py
├── config.py                    # Experiment configuration
├── spider_loader.py             # Spider dataset + SQLite execution
├── text2sql_agent.py            # LLM agent wrapper
├── eval_set_generator.py        # Greedy anchor-guided selection
├── quality_metrics.py           # Separation power, alignment
├── experiment_runner.py         # Main orchestration
├── results_analyzer.py          # Analysis + reporting
└── run_poc.py                   # Entry point
```

---

## Implementation Plan

### Phase 1: Data Setup (30 min)

**File: `spider_loader.py`**
```python
@dataclass
class SpiderExample:
    question: str
    query: str  # Gold SQL
    db_id: str
    hardness: str  # easy, medium, hard, extra

class SpiderDataset:
    def __init__(self, spider_path: Path):
        # Load train.json, get hardness from metadata
        pass

    def get_stratified_split(
        self, anchor_size=100, candidate_size=800, test_size=200
    ) -> tuple[list, list, list]:
        # Balanced sampling across hardness levels
        pass

    def execute_sql(self, predicted: str, gold: str, db_id: str) -> float:
        # Execute both, compare result sets → 0.0 or 1.0
        pass
```

### Phase 2: Eval Set Generator (45 min)

**File: `eval_set_generator.py`**

Core algorithm (greedy selection):
```python
class AnchorGuidedSelector:
    def __init__(self, anchor_set, candidate_pool):
        self.anchor = anchor_set
        self.candidates = candidate_pool
        # Pre-compute TF-IDF features for efficiency
        self._vectorizer = TfidfVectorizer()
        self._anchor_features = self._extract_features(anchor_set)
        self._candidate_features = self._extract_features(candidate_pool)

    def select_greedy(
        self, target_size: int,
        separation_weight: float = 0.6,
        alignment_weight: float = 0.4
    ) -> list[EvaluationExample]:
        """
        Greedy selection combining:
        - Separation: Maximize diversity (1 - max_similarity to selected)
        - Alignment: Maintain anchor distribution (avg_similarity to anchors)
        """
        selected = []
        remaining = set(range(len(self.candidates)))

        for _ in range(target_size):
            best_idx = max(remaining, key=lambda i:
                separation_weight * self._separation_score(i, selected) +
                alignment_weight * self._alignment_score(i)
            )
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [self.candidates[i] for i in selected]
```

### Phase 3: Text2SQL Agent (30 min)

**File: `text2sql_agent.py`**
```python
# Configuration space for tuning
CONFIG_SPACE = {
    "model": ["gpt-4o-mini", "gpt-3.5-turbo"],
    "temperature": [0.0, 0.3, 0.7],
    "prompt_style": ["minimal", "cot", "few_shot"],
}

def text2sql_agent(question: str, db_id: str, config: dict) -> str:
    """Generate SQL from natural language."""
    llm = ChatOpenAI(model=config["model"], temperature=config["temperature"])
    schema = load_schema(db_id)
    prompt = build_prompt(question, schema, config["prompt_style"])
    return str(llm.invoke(prompt).content)
```

### Phase 4: Experiment Runner (45 min)

**File: `experiment_runner.py`**
```python
async def run_experiment(
    eval_set: list[SpiderExample],
    test_set: list[SpiderExample],
    config_space: dict,
    n_trials: int = 20,
    condition: str = "baseline"
) -> ExperimentResult:
    """Run tuning on eval_set, evaluate on test_set."""

    # Tune on eval_set
    best_config, trial_history = await tune_agent(
        eval_set, config_space, n_trials
    )

    # Evaluate best config on test_set
    test_accuracy = evaluate_on_test(best_config, test_set)

    return ExperimentResult(
        condition=condition,
        best_config=best_config,
        test_accuracy=test_accuracy,
        total_cost=sum(t.cost for t in trial_history),
        trial_history=trial_history
    )
```

### Phase 5: Run & Analyze (30 min)

**File: `run_poc.py`**
```python
async def main():
    # 1. Load and split data
    spider = SpiderDataset(SPIDER_PATH)
    anchor, candidates, test = spider.get_stratified_split()

    # 2. Generate eval sets
    selector = AnchorGuidedSelector(anchor, candidates)
    generated_eval = selector.select_greedy(target_size=75)
    random_eval = random.sample(candidates, 75)

    # 3. Run experiments
    baseline = await run_experiment(random_eval, test, CONFIG_SPACE, condition="baseline")
    treatment = await run_experiment(generated_eval, test, CONFIG_SPACE, condition="treatment")

    # 4. Report
    print_comparison(baseline, treatment)
```

---

## Timeline (4 hours)

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Spider loader + SQLite execution | 30 min |
| 2 | Greedy eval set generator | 45 min |
| 3 | Text2SQL agent wrapper | 30 min |
| 4 | Experiment runner | 45 min |
| 5 | Run experiments + analysis | 1h 30min |

---

## Success Criteria

1. ✅ Generated eval set has measurably higher separation power
2. ✅ Treatment test accuracy > Baseline test accuracy
3. ✅ Total cost under $50
4. ✅ Improvement is meaningful (>5% relative)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SQLite setup issues | Pre-test with 5 databases before full run |
| Cost overrun | Start with 10 trials, scale up if budget allows |
| Slow execution | Parallelize SQL execution, cache results |
| No improvement | Document negative result, analyze why |

---

## Critical Files to Reference

- `traigent/evaluators/base.py` - Dataset, EvaluationExample patterns
- `traigent/core/orchestrator.py` - Optimization loop integration
- `traigent/cloud/subset_selection.py` - Existing selection algorithms
- `examples/advanced/ai-engineering-tasks/` - Custom evaluator patterns
