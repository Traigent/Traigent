# Traigent Algorithmic Foundations: Agile Implementation Roadmap

**Status**: Approved
**Created**: 2026-01-26
**Reviewed**: 2026-01-26 by Codex GPT-5.2 (xhigh reasoning)
**Priority Framework**: Value/Effort Matrix (Highest Value, Lowest Effort First)

---

## Executive Summary

This roadmap prioritizes 67 algorithms from `docs/design/algorithmic-foundations.md` for integration into Traigent SDK. Features are organized into 5 phases based on value/effort ratio.

**Key Insight**: Traigent already has solid infrastructure (BaseOptimizer, BaseSampler, MetricRegistry, Multi-objective support). Most algorithms can plug into existing extension points.

---

## Codex GPT-5.2 Review Summary (2026-01-26)

### Critical Findings

1. **Missing Primitives**: Phase 1 items assume evaluation harness, confidence scoring, and cost telemetry exist. Without these, algorithms won't show value.

2. **Bloom Filter Underspecified**: Semantic cache via Bloom Filter requires embeddings + backing store. Effort should be 2.5-3.0, not 1.5. **Move to Phase 2**

3. **Ski Rental & Early Voting**: Both need reliable accuracy/cost estimators. Integration effort is 2-3x current estimates. **Move to Phase 2**

4. **Timeline Optimistic**: Expect 35-50 sprints (not 26) once infra, docs, and testing are added.

### Missed High-Value/Low-Effort Features (Add to Phase 1)

| Feature | Value | Effort | Ratio | Why |
|---------|-------|--------|-------|-----|
| **ASHA/Successive Halving** | 4.5 | 1.5 | 3.0 | High-ROI early stopping, Optuna has it |
| **Epsilon-Greedy** | 3.5 | 1.0 | 3.5 | Simple baseline bandit, enables comparisons |
| **Deterministic Hash Cache** | 4.0 | 1.0 | 4.0 | Simpler than Bloom, immediate cost savings |
| **Sequential Testing (SPRT)** | 4.0 | 1.5 | 2.7 | Basic A/B with early stop |

### Revised Phase 1 Composition

| Keep | Move to Phase 2 | Add |
|------|-----------------|-----|
| Reservoir Sampling | Bloom Filter | ASHA/Successive Halving |
| Thompson Sampling | Ski Rental | Epsilon-Greedy |
| - | Early Voting | Deterministic Hash Cache |
| - | - | Sequential Testing (SPRT) |

### Hidden Dependencies Identified

- **Bandits (Thompson/UCB/EXP3/Hedge)**: Require reward normalization, cold-start priors, non-stationarity handling
- **HTN/MCTS/Branch&Bound**: Need formal state/action model and heuristic costs
- **Raft/BFT**: Need membership/identity, transport, fault handling (cross-service)
- **PID Controllers**: Require stable telemetry and consistent control intervals

---

## Business Architecture: SDK (Freemium) vs Backend (Premium)

### Strategic Principle

**Maximize backend logic to create upgrade incentives.** The SDK provides basic optimization capabilities for freemium users. Advanced algorithms, cross-run learning, and team features require backend connectivity (paid tiers).

### Tier Architecture

```
+-----------------------------------------------------------------------------+
|                           ENTERPRISE TIER (Backend)                         |
|  Multi-agent coordination, Raft/BFT consensus, Shapley attribution          |
|  Cross-experiment meta-learning, Team collaboration, LATS/ToT/GoT           |
|-----------------------------------------------------------------------------|
|                           PRO TIER (Backend)                                |
|  Bayesian optimization (GP), MCTS, HTN Planning, PAC-Bayes bounds          |
|  Cross-run warm-start, Hedge/EXP3 with history, MinHash dedup              |
|-----------------------------------------------------------------------------|
|                           TEAM TIER (Backend)                               |
|  Optimal parallelism, Quorum consensus, T-Digest analytics                  |
|  PID controllers with cloud telemetry, Count-Min/HLL analytics              |
|-----------------------------------------------------------------------------|
|                           FREEMIUM (SDK-only)                               |
|  Random search, Grid search, Thompson Sampling (stateless)                  |
|  Reservoir sampling, Deterministic cache, Epsilon-greedy, ASHA             |
+-----------------------------------------------------------------------------+
```

### Feature Tier Assignment

| Feature | Tier | Location | Upgrade Incentive |
|---------|------|----------|-------------------|
| **Random/Grid Search** | Free | SDK | Baseline, always available |
| **Reservoir Sampling** | Free | SDK | Basic dataset sampling |
| **Deterministic Hash Cache** | Free | SDK | Simple cost savings |
| **Epsilon-Greedy** | Free | SDK | Baseline bandit |
| **ASHA/Successive Halving** | Free | SDK | Early stopping (Optuna parity) |
| **Thompson Sampling** | Free | SDK | Stateless per-run only |
| **Sequential Testing (SPRT)** | Free | SDK | Basic A/B stopping |
| --- | --- | --- | --- |
| **Thompson w/ History** | Team | **Backend** | Cross-run prior learning |
| **UCB with Warm-Start** | Team | **Backend** | Learns from past experiments |
| **Optimal Parallelism** | Team | **Backend** | Queueing theory requires fleet telemetry |
| **Quorum Consensus** | Team | **Backend** | Multi-agent voting coordination |
| **PID Controllers** | Team | **Backend** | Requires cloud telemetry loop |
| **Bloom Filter Cache** | Team | **Backend** | Shared embedding index |
| **Count-Min Sketch** | Team | **Backend** | Aggregated token analytics |
| **HyperLogLog** | Team | **Backend** | Cross-run cardinality |
| **T-Digest** | Team | **Backend** | Aggregated percentiles |
| --- | --- | --- | --- |
| **EXP3 (Adversarial)** | Pro | **Backend** | Non-stationarity detection |
| **Hedge Algorithm** | Pro | **Backend** | Online weight learning |
| **Bayesian Optimizer (GP)** | Pro | **Backend** | Surrogate model training |
| **Beam Search** | Pro | **Backend** | Parallel config exploration |
| **HTN Planning** | Pro | **Backend** | Agent workflow orchestration |
| **MinHash/LSH Dedup** | Pro | **Backend** | Cross-experiment similarity |
| **MCTS** | Pro | **Backend** | Tree search with value network |
| **PAC-Bayes Bounds** | Pro | **Backend** | Sample complexity guarantees |
| **Ski Rental Cascade** | Pro | **Backend** | Cross-run cost modeling |
| --- | --- | --- | --- |
| **Raft Leader Election** | Enterprise | **Backend** | Multi-agent coordination |
| **Byzantine Fault Tolerance** | Enterprise | **Backend** | Fault-tolerant consensus |
| **Shapley Attribution** | Enterprise | **Backend** | Fair credit assignment |
| **Multi-Agent PID** | Enterprise | **Backend** | Team-level control loops |
| **Tree of Thought** | Enterprise | **Backend** | Inference-time search |
| **LATS** | Enterprise | **Backend** | Agent tree search |
| **Graph of Thought** | Enterprise | **Backend** | DAG reasoning |
| **Branch & Bound** | Enterprise | **Backend** | Optimal search |
| **Formal Contracts** | Enterprise | **Backend** | Runtime verification |

### Why Backend for Premium Features?

1. **Cross-Run Learning**: Bandits, Bayesian optimization, and meta-learning need historical data stored in backend
2. **Shared State**: Consensus, coordination, and caching require centralized state management
3. **Compute Intensity**: GP fitting, MCTS rollouts, LSH indexing are better suited for backend compute
4. **Telemetry Aggregation**: Analytics features (T-Digest, HLL, Count-Min) aggregate across all users/experiments
5. **Security**: Sensitive optimization logic protected from reverse engineering
6. **Upgrade Path**: Clear value proposition for each tier

### SDK Stub Pattern

For backend features, SDK provides thin stubs that:
1. Check tier/license
2. Submit request to backend API
3. Receive optimized result
4. Apply locally

```python
# Example: Bayesian Optimizer stub in SDK
class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization via GP surrogate. Requires Pro tier."""

    def __init__(self, cloud_client: CloudClient):
        self.cloud_client = cloud_client
        self._check_tier("pro")

    async def suggest_next_trial(self, history: list[TrialResult]) -> dict:
        # All GP logic lives in backend
        response = await self.cloud_client.post(
            "/optimizers/bayesian/suggest",
            {"experiment_id": self.experiment_id, "history": history}
        )
        return response["suggested_config"]
```

### Implementation Priority (Revised)

**Phase 1 (SDK Freemium)**: Ship basic value, low backend dependency
**Phase 2 (Backend Team)**: Core backend infrastructure + Team tier features
**Phase 3 (Backend Pro)**: Advanced optimization algorithms
**Phase 4 (Backend Enterprise)**: Multi-agent and research features

---

## 1. Value/Effort Scoring Framework

### Value Criteria (1-5 scale)
| Factor | Weight | Description |
|--------|--------|-------------|
| **User Impact** | 40% | Direct improvement to user optimization outcomes |
| **Differentiation** | 30% | Competitive advantage vs alternatives (Optuna, Ray Tune) |
| **Foundation** | 20% | Enables other high-value features |
| **Revenue** | 10% | Unlocks enterprise/premium tiers |

### Effort Criteria (1-5 scale, lower = easier)
| Factor | Weight | Description |
|--------|--------|-------------|
| **Lines of Code** | 30% | Simple (<100 LoC)=1, Medium (100-500)=3, Complex (500+)=5 |
| **Dependencies** | 25% | None=1, Internal=2, External lib=4, New service=5 |
| **Integration** | 25% | Uses existing interface=1, Extends interface=3, New architecture=5 |
| **Testing** | 20% | Unit test only=1, Integration=3, E2E+infrastructure=5 |

---

## 2. Phase 1: SDK Freemium Foundation (2-3 Sprints)
**Theme**: Ship basic SDK value, establish backend API patterns
**Tier**: Freemium (SDK-only)

### 2.1 Thompson Sampling (Stateless)
**Value: 4.5 | Effort: 1.5 | Ratio: 3.0 | Tier: Free**

```
Location: traigent/optimizers/thompson.py (SDK)
Upgrade Path: Team tier adds cross-run prior persistence
LOC: ~80
```

**What**: Auto-select between Random, TPE, Bayesian. Stateless per-run (no history persistence).

**Upgrade Hook**: Backend stores posteriors across experiments for warm-start.

```python
class ThompsonOptimzerSelector:
    """Select optimizer using Thompson Sampling (Beta posteriors)."""

    def __init__(self, optimizers: list[BaseOptimizer]):
        self.optimizers = optimizers
        self.successes = {opt.name: 1 for opt in optimizers}  # Prior
        self.failures = {opt.name: 1 for opt in optimizers}

    def select(self) -> BaseOptimizer:
        samples = {
            name: random.betavariate(self.successes[name], self.failures[name])
            for name in self.optimizers
        }
        return max(samples, key=samples.get)

    def update(self, optimizer_name: str, improved: bool):
        if improved:
            self.successes[optimizer_name] += 1
        else:
            self.failures[optimizer_name] += 1
```

**Files to modify**:
- `traigent/optimizers/adaptive_selector.py` (NEW)
- `traigent/core/orchestrator.py` (add selector option)
- `tests/unit/optimizers/test_adaptive_selector.py` (NEW)

---

### 2.2 Reservoir Sampling for Evaluation Dataset
**Value: 3.8 | Effort: 1.0 | Ratio: 3.8**

```
Location: traigent/core/samplers/reservoir.py (NEW)
Integration: BaseSampler interface
LOC: ~50
```

**What**: Unbiased sampling from streaming/large datasets without knowing size.

```python
class ReservoirSampler(BaseSampler):
    """Algorithm R: Unbiased k-sample from unknown-size stream."""

    def __init__(self, k: int, seed: int = None):
        super().__init__(seed=seed)
        self.k = k
        self.reservoir = []
        self.n_seen = 0

    def add(self, item: Any) -> bool:
        """Add item to stream. Returns True if item is in reservoir."""
        self.n_seen += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
            return True
        else:
            j = self._rng.randint(0, self.n_seen - 1)
            if j < self.k:
                self.reservoir[j] = item
                return True
        return False
```

**Files to modify**:
- `traigent/core/samplers/reservoir.py` (NEW)
- `traigent/core/samplers/__init__.py` (register)

---

### Phase 1 Summary

| Feature | Value | Effort | Ratio | Sprints |
|---------|-------|--------|-------|---------|
| Deterministic Hash Cache | 4.0 | 1.0 | **4.0** | 0.5 |
| Reservoir Sampling | 3.8 | 1.0 | **3.8** | 0.5 |
| Epsilon-Greedy | 3.5 | 1.0 | **3.5** | 0.5 |
| Thompson Sampling | 4.5 | 1.5 | **3.0** | 1 |
| ASHA/Successive Halving | 4.5 | 1.5 | **3.0** | 1 |
| Sequential Testing (SPRT) | 4.0 | 1.5 | **2.7** | 0.5 |

**Total Phase 1**: ~3 sprints, 6 features

---

## 3. Phase 2: Backend Team Tier (4-5 Sprints)
**Theme**: Backend infrastructure + Team tier monetization
**Tier**: Team (Backend-required)

### Phase 2 Summary

| Feature | Value | Effort | Ratio | Sprints |
|---------|-------|--------|-------|---------|
| UCB Bandit | 4.3 | 2.0 | **2.15** | 1 |
| Optimal Parallelism | 4.5 | 2.5 | **1.8** | 1 |
| Count-Min Sketch | 3.5 | 2.0 | **1.75** | 0.5 |
| Quorum Consensus | 4.2 | 2.5 | **1.68** | 1 |
| HyperLogLog | 3.3 | 2.0 | **1.65** | 0.5 |
| PID Controller | 4.0 | 2.5 | **1.6** | 1 |
| Bloom Filter Cache | 4.2 | 2.5 | **1.68** | 1 |
| Ski Rental Cascade | 4.0 | 2.5 | **1.6** | 0.5 |
| Early Voting | 4.0 | 2.5 | **1.6** | 0.5 |

**Total Phase 2**: ~5 sprints, 9 features

---

## 4. Phase 3: Backend Pro Tier (5-6 Sprints)
**Theme**: Advanced algorithms for competitive differentiation
**Tier**: Pro (Backend-required)

### Phase 3 Summary

| Feature | Value | Effort | Ratio | Sprints |
|---------|-------|--------|-------|---------|
| EXP3 Bandit | 4.0 | 2.5 | **1.6** | 1 |
| T-Digest | 3.8 | 2.5 | **1.52** | 1 |
| Hedge Weights | 4.5 | 3.0 | **1.5** | 1 |
| Beam Search | 4.2 | 3.0 | **1.4** | 1 |
| HTN Planning | 4.8 | 3.5 | **1.37** | 1.5 |
| MinHash/LSH | 4.0 | 3.0 | **1.33** | 1 |

**Total Phase 3**: ~5 sprints, 6 features

---

## 5. Phase 4: Backend Enterprise Tier (6-8 Sprints)
**Theme**: Multi-agent coordination for enterprise customers
**Tier**: Enterprise (Backend-required, SSO, dedicated support)

### Phase 4 Summary

| Feature | Value | Effort | Ratio | Sprints |
|---------|-------|--------|-------|---------|
| Raft Leader Election | 4.5 | 4.0 | **1.13** | 1.5 |
| PAC-Bayes Bounds | 4.3 | 4.0 | **1.08** | 1 |
| MCTS Search | 4.5 | 4.5 | **1.0** | 1.5 |
| Shapley Attribution | 4.0 | 4.0 | **1.0** | 1 |
| Multi-Agent PID | 4.0 | 4.0 | **1.0** | 1 |
| BFT Consensus | 4.2 | 4.5 | **0.93** | 1.5 |

**Total Phase 4**: ~6 sprints, 6 features

---

## 6. Phase 5: Research & Cutting-Edge (6+ Sprints)
**Theme**: Next-Generation Capabilities

### Phase 5 Summary

| Feature | Value | Effort | Ratio | Sprints |
|---------|-------|--------|-------|---------|
| LATS | 5.0 | 5.0 | **1.0** | 2 |
| ToT | 4.8 | 5.0 | **0.96** | 1.5 |
| CMA-ES Enhance | 4.2 | 4.5 | **0.93** | 1 |
| B&B Optimizer | 4.5 | 5.0 | **0.9** | 1.5 |
| GoT | 4.5 | 5.5 | **0.82** | 2 |
| Formal Contracts | 4.0 | 5.0 | **0.8** | 1 |

**Total Phase 5**: ~8 sprints, 6 features

---

## 7. Roadmap Timeline

```
                    2026
    Q1          |      Q2          |      Q3          |      Q4
    -------------------------------------------------------------

    Phase 1: Quick Wins (3 sprints)
    |-- Sprint 1: Thompson Sampling + Reservoir
    |-- Sprint 2: Epsilon-Greedy + ASHA + Hash Cache
    +-- Sprint 3: SPRT

                Phase 2: Foundation (5 sprints)
                |-- Sprint 4: UCB + Bloom Filter
                |-- Sprint 5: PID Controller
                |-- Sprint 6: Optimal Parallelism
                |-- Sprint 7: Quorum Consensus
                +-- Sprint 8: Count-Min + HyperLogLog + Ski Rental + Early Voting

                            Phase 3: Differentiation (5 sprints)
                            |-- Sprint 9: Hedge + EXP3
                            |-- Sprint 10: HTN Planning
                            |-- Sprint 11: Beam Search
                            |-- Sprint 12: T-Digest
                            +-- Sprint 13: MinHash/LSH

                                        Phase 4: Multi-Agent (6 sprints)
                                        |-- Sprint 14-15: Raft + MCTS
                                        |-- Sprint 16-17: Shapley + PAC
                                        +-- Sprint 18-19: BFT + Multi-PID

                                                    Phase 5: Research
                                                    +-- Ongoing...
```

---

## 8. Dependencies Graph

```
Phase 1 (Independent - can parallelize)
|-- Thompson Sampling --------------------------------------------+
|-- Reservoir Sampling -------------------------------------------+
|-- Epsilon-Greedy -----------------------------------------------+
|-- ASHA/Successive Halving --------------------------------------+
|-- Deterministic Hash Cache -------------------------------------+
+-- Sequential Testing (SPRT) ------------------------------------+
                                                                  |
Phase 2                                                           v
|-- UCB -----------------------------> EXP3 ------> Hedge (Phase 3)
|-- PID Controller ------------------> Multi-Agent PID (Phase 4)
|-- Optimal Parallelism -------------> Queueing-aware scheduling
|-- Quorum Consensus ----------------> Raft ------> BFT (Phase 4)
|-- Bloom Filter --------------------> Semantic cache
+-- Ski Rental + Early Voting -------> Model cascade optimization

Phase 3
|-- HTN Planning --------------------> LATS (Phase 5)
|-- Beam Search ---------------------> MCTS (Phase 4)
+-- T-Digest ------------------------> Analytics Dashboard

Phase 4
|-- MCTS ----------------------------> Branch & Bound (Phase 5)
|-- PAC-Bayes -----------------------> Formal Verification (Phase 5)
+-- Shapley -------------------------> Fair Multi-Agent Attribution

Phase 5
|-- ToT -----------------------------> GoT (later)
+-- LATS ----------------------------> Full Agent Orchestration
```

---

## 9. Integration Points

### Existing Extension Points (Use These)
| Component | Location | How to Extend |
|-----------|----------|---------------|
| Optimizers | `traigent/optimizers/base.py` | Extend `BaseOptimizer` |
| Samplers | `traigent/core/samplers/` | Extend `BaseSampler` |
| Metrics | `traigent/metrics/registry.py` | Register new metrics |
| Stop Conditions | `traigent/core/stop_conditions.py` | Add new conditions |
| Patterns | `traigent/patterns/` | Add to `PatternCatalog` |

### New Extension Points (Create These in Phase 2)
| Component | Purpose |
|-----------|---------|
| `traigent/bandits/` | Bandit algorithms (UCB, Thompson, EXP3, Hedge) |
| `traigent/consensus/` | Voting and consensus algorithms |
| `traigent/control/` | Feedback control (PID, adaptive) |
| `traigent/analytics/probabilistic/` | Sketches, HLL, T-Digest |
| `traigent/planning/` | HTN, search algorithms |

---

## 10. Testing Strategy

### Per-Feature Testing
1. **Unit tests**: Algorithm correctness (mock data)
2. **Property tests**: Invariant verification (Hypothesis)
3. **Integration tests**: Plugs into existing orchestrator
4. **Benchmark tests**: Performance vs baseline

### Key Test Patterns
```python
# Property-based testing example for Thompson Sampling
from hypothesis import given, strategies as st

@given(st.lists(st.floats(0, 1), min_size=10))
def test_thompson_sampling_converges(rewards):
    """Thompson sampling should converge to best arm."""
    selector = ThompsonSampler(n_arms=3)
    for reward in rewards:
        arm = selector.select()
        selector.update(arm, reward)

    # Should increasingly select best arm
    selections = [selector.select() for _ in range(100)]
    assert max(set(selections), key=selections.count) == np.argmax(rewards)
```

---

## 11. Success Metrics

### Phase 1 Success Criteria
- [ ] Thompson Sampling improves optimizer selection by >10%
- [ ] Deterministic Hash Cache reduces duplicate calls by >50%
- [ ] ASHA reduces wasted trials by >30%

### Phase 2 Success Criteria
- [ ] PID Controller achieves <5% steady-state error
- [ ] UCB outperforms random selection by >15%
- [ ] Quorum reduces total agent calls by >25%

### Phase 3 Success Criteria
- [ ] HTN enables complex workflow optimization
- [ ] Hedge achieves sublinear regret on objective weights
- [ ] T-Digest provides p99 within 1% error

### Phase 4 Success Criteria
- [ ] MCTS outperforms Bayesian on structured spaces
- [ ] BFT handles 1/3 faulty agents correctly
- [ ] Shapley provides fair attribution (+/-5% of true value)

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Algorithm complexity underestimated | Start with simplified versions, iterate |
| Integration conflicts | Use existing interfaces, avoid new architecture |
| Performance regression | Benchmark tests in CI |
| Thread-safety issues | Follow existing patterns (locks, DI) |
| External dependency issues | Prefer pure Python, fallback implementations |

---

## 13. Next Steps

1. **Validate priorities** with stakeholders
2. **Create Phase 1 tickets** in project tracker
3. **Set up new module structure** (`traigent/bandits/`, `traigent/consensus/`)
4. **Implement Thompson Sampling** as proof-of-concept
5. **Review and iterate** on roadmap quarterly

---

## Appendix: Full Feature Inventory

### Phase 1 Features (6) - SDK Freemium
1. Deterministic Hash Cache
2. Reservoir Sampling (dataset sampling)
3. Epsilon-Greedy (baseline bandit)
4. Thompson Sampling (optimizer selection)
5. ASHA/Successive Halving (early stopping)
6. Sequential Testing/SPRT (A/B stopping)

### Phase 2 Features (9) - Backend Team
7. UCB Bandit (prompt selection)
8. Bloom Filter Cache (semantic cache)
9. PID Controller (parameter tuning)
10. Optimal Parallelism (queueing theory)
11. Quorum Consensus (multi-agent voting)
12. Count-Min Sketch (frequency estimation)
13. HyperLogLog (cardinality)
14. Ski Rental Cascade (model escalation)
15. Early Voting (consensus termination)

### Phase 3 Features (6) - Backend Pro
16. Hedge Algorithm (objective weighting)
17. HTN Planning (agent workflows)
18. EXP3 Bandit (adversarial)
19. Beam Search (configuration)
20. T-Digest (percentiles)
21. MinHash/LSH (deduplication)

### Phase 4 Features (6) - Backend Enterprise
22. Raft Leader Election
23. Byzantine Fault Tolerance
24. Shapley Attribution
25. MCTS Search
26. PAC-Bayes Bounds
27. Multi-Agent PID

### Phase 5 Features (6) - Research
28. Tree of Thought
29. LATS
30. Branch and Bound
31. CMA-ES Enhancement
32. Formal Contracts
33. Graph of Thought

**Total: 33 prioritized features** from 67 algorithms in algorithmic foundations.
