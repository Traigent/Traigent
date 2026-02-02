# Bazak × Traigent POC

## Overview

Traigent optimizes AI agent configurations to improve accuracy, reduce latency, and lower costs—without requiring code changes. This POC demonstrates Traigent's Hybrid Mode optimization on Bazak's travel booking agent.

**Selected Use Case**: Improve accuracy when handling flight/vacation requests with children, where the agent should ask for child ages but currently fails to do so in ~20% of cases.

---

## Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Accuracy (child age prompting) | ~80% | ~90% |
| Cost | Current | Same or lower |
| SDLC Time Saved | - | Reduced tuning effort |

**Secondary metrics to monitor:**
- P95 Latency (baseline: 47s)
- Time to First Token (baseline: 6.4s avg, 17s P95)

---

## Integration Architecture

```
┌─────────────────┐         ┌─────────────────┐
│    Traigent     │  HTTP   │  Bazak Service  │
│   Optimizer     │◄───────►│   (Tunables)    │
└─────────────────┘         └─────────────────┘
        │                           │
        │ 1. Discover tunables      │
        │ 2. Execute with configs   │
        │ 3. Evaluate outputs       │
        │ 4. Optimize               │
        └───────────────────────────┘
```

Traigent calls Bazak's endpoints to:
1. **Discover** tunable parameters (model, temperature, prompts, etc.)
2. **Execute** the agent with different configurations on a dataset
3. **Evaluate** outputs against expected behaviors
4. **Iterate** to find optimal configuration

---

## API Contract

Bazak implements 4 REST endpoints under `/traigent/v1/`:

| Endpoint | Method | Required | Purpose |
|----------|--------|----------|---------|
| `/capabilities` | GET | Yes | Service feature discovery |
| `/config-space` | GET | Yes | Tunable parameter definitions |
| `/execute` | POST | Yes | Run agent with config on inputs |
| `/evaluate` | POST | Yes | Score outputs against targets |

**Full API specification**: [hybrid-mode-api-contract.md](./hybrid-mode-api-contract.md)

### Privacy-Preserving Mode (Default)

Traigent only observes:
- Tunable definitions and values
- Metrics (accuracy, cost, latency)

**Not transmitted**: Input queries, agent responses, or user data.

---

## Bazak Responsibilities

### 1. Evaluation Dataset
Create 50-100 test cases for the child-age-prompting scenario:

```json
{
  "input_id": "case_001",
  "query": "I need a flight to Paris for me and my two kids",
  "expected_behavior": "should_ask_ages",
  "context": {"has_children": true, "child_count": 2}
}
```

**Requirements:**
- Mix of positive cases (should ask) and negative cases (no children mentioned)
- Representative of real user queries
- Clear expected behaviors for evaluation

### 2. Tunable Parameters
Define parameters Traigent can optimize:

```json
{
  "tunables": [
    {"name": "model", "type": "enum", "domain": {"values": ["gpt-4o", "gpt-4o-mini"]}},
    {"name": "temperature", "type": "float", "domain": {"range": [0.0, 1.0]}},
    {"name": "system_prompt_version", "type": "enum", "domain": {"values": ["v1", "v2", "v3"]}},
    {"name": "max_retries", "type": "int", "domain": {"range": [0, 3]}}
  ]
}
```

### 3. REST Endpoints
Implement the 4 Traigent API endpoints:
- Flask, FastAPI, Express, or any HTTP framework
- Follow the [API contract](./hybrid-mode-api-contract.md)
- See [Flask demo](../examples/hybrid_mode_demo/) for reference

### 4. Evaluation Logic
Implement scoring in the `/evaluate` endpoint:

```python
def evaluate(output, expected):
    if expected["expected_behavior"] == "should_ask_ages":
        asked_ages = detect_age_question(output["response"])
        return {"accuracy": 1.0 if asked_ages else 0.0}
    # ... other evaluation logic
```

---

## Traigent Responsibilities

1. **Optimization Engine**: Run trials with different configurations
2. **Cost Tracking**: Monitor API costs per configuration
3. **Analytics Dashboard**: Visualize optimization progress
4. **Best Config Recommendation**: Identify optimal settings
5. **Technical Support**: Integration assistance

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Integration | 1 week | Bazak endpoints live, dataset ready |
| Optimization | 1-2 weeks | 50+ trials, best config identified |
| Validation | 3-5 days | A/B test best config vs baseline |
| Report | 2 days | Final metrics, recommendations |

---

## Quick Start

```bash
# 1. Clone the reference implementation
cd examples/hybrid_mode_demo

# 2. Start the demo server (reference)
python app.py

# 3. Run test client to verify endpoints
python test_client.py

# 4. Run optimization
python run_optimization.py
```

---

## Contact

- **Traigent**: [integration support contact]
- **Bazak**: [technical contact]

---

## References

- [API Contract](./hybrid-mode-api-contract.md) - Full endpoint specifications
- [Client Guide](./hybrid-mode-client-guide.md) - Implementation patterns
- [OpenAPI Spec](./hybrid-mode-openapi.yaml) - Machine-readable API definition
- [Flask Demo](../examples/hybrid_mode_demo/) - Working example
