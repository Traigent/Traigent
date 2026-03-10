# Arkia Sales Agent Demo

Traigent optimization demo for a travel sales agent inspired by [Mastra.ai](https://mastra.ai) patterns.

> **Note:** This demo uses a **MOCK implementation** that simulates LLM behavior for cost-free testing. No API keys required. See [Real Integration](#real-mastraai-integration) for production patterns.

## Business Context

[Arkia](https://www.arkia.co.il) is an Israeli airline that uses AI sales agents to help resellers sell flights and vacation packages. They make money on **margins**, so optimizing the cost/quality tradeoff is critical.

## Key Optimization Insights

**Hidden cost drivers that Traigent can optimize:**

1. **Memory turns** - More conversation history = better context = higher quality BUT more input tokens = higher cost
2. **Tool set** - More tools = better capability BUT tool descriptions consume tokens
3. **Model selection** - Groq Llama can match GPT-4o quality at 5-10x lower cost

## File Structure

```
arkia-sales-agent/
├── src/
│   ├── agent.ts          # MOCK agent + per-example batch scoring
│   ├── tools.ts          # Travel agent tools (flight search, booking, etc.)
│   ├── mastra-agent.ts   # Reference: REAL Mastra.ai patterns (not used)
│   ├── dataset.ts        # Curated sales conversations (Hebrew + English)
│   ├── trial.ts          # Shared trial logic + native optimize() wrapper
│   └── standalone.ts     # Local SDK-driven optimization demo
├── run_with_python.py    # Python orchestrator with @traigent.optimize
├── dataset.jsonl         # Dataset for Python optimizer
└── package.json
```

## Configuration Space

| Parameter | Options | Impact |
|-----------|---------|--------|
| **model** | gpt-3.5-turbo, gpt-4o-mini, gpt-4o, groq/llama-3.3-70b-versatile, groq/llama-3.1-8b-instant | Quality vs Cost |
| **temperature** | 0.0, 0.3, 0.5, 0.7 | Consistency vs Creativity |
| **system_prompt** | sales_aggressive, consultative, informative | Conversion style |
| **memory_turns** | 2, 5, 10, 15 | Context quality vs Token cost |
| **tool_set** | minimal, standard, enhanced, full | Capability vs Token cost |

### Tool Sets

| Set | Tools | Token Cost | Conversion Boost |
|-----|-------|------------|------------------|
| minimal | search, price, book | ~300 tokens | Baseline |
| standard | + availability, promotions | ~500 tokens | +8% |
| enhanced | + customer history, destination info | ~700 tokens | +12% |
| full | + packages, installments, upgrades | ~1000 tokens | +15% |

## Model Comparison (Simulated)

| Model | Quality | Cost/1M tokens | Latency | Best For |
|-------|---------|----------------|---------|----------|
| GPT-4o | Highest | $2.50/$10.00 | ~450ms | Maximum quality |
| GPT-4o-mini | Good | $0.15/$0.60 | ~250ms | Balanced |
| Groq Llama 70B | High | $0.59/$0.79 | ~80ms | Quality + Speed |
| Groq Llama 8B | Moderate | $0.05/$0.08 | ~30ms | High volume |

> **Note:** Costs are simulated values for demonstration. Actual pricing may vary.

## Running the Demo

### Execution Time

| Mode | Duration | Cost |
|------|----------|------|
| Local demo (`npm run dev`) | ~2 seconds | Free (mock) |
| Python optimization (24 trials) | ~5 minutes | Free (mock) |

### Local Demo (Traigent JS SDK chooses the configs)

```bash
cd demos/arkia-sales-agent
npm install
npm run build
npm run dev
```

What this now does:

- evaluates the current default Arkia config on a demo subset
- runs `optimize(...).optimize(...)` from the JS SDK over the Arkia search space
- replays the best config so you can compare baseline vs optimizer-selected config

### With Python Orchestrator / Bridge

```bash
cd demos/arkia-sales-agent
npm run build
TRAIGENT_COST_APPROVED=true python run_with_python.py
```

## Sample Output

Running `npm run dev` now produces a baseline-vs-optimizer summary:

```
**********************************************************************
*  ARKIA SALES AGENT - TRAIGENT JS SDK DEMO
*  SDK selects configurations; Arkia code scores each chosen config.
**********************************************************************

Dataset Statistics:
  Total examples: 20
  Demo subset: 10

======================================================================
RUNNING TRAIGENT NATIVE OPTIMIZATION
======================================================================
Trials run: 12
Stop reason: maxTrials
Optimizer best config: {"model":"gpt-4o-mini","temperature":0.3,"system_prompt":"consultative","memory_turns":5,"tool_set":"standard"}

Baseline vs Best:
  Conversion: 72.0% -> 88.0%
  Margin efficiency: 0.08 -> 0.61
  Total cost: $0.012400 -> $0.001920
  Avg latency: 410ms -> 230ms
```

## Metrics

### Key Metric: Margin Efficiency

```
margin_efficiency = conversion_score / (cost_per_conversation * 10000)
```

Higher margin efficiency = better ROI. The formula scales by 10,000 for readability.

### Quality Metrics (Mastra-compatible)
- `relevancy` - Answer relevancy (0-1)
- `completeness` - Response completeness (0-1)
- `tone_consistency` - Style consistency (0-1)

### Business Metrics
- `conversion_score` - Sales conversion rate (0-1)
- `margin_efficiency` - Conversion per dollar spent (higher is better)

### Cost Metrics
- `cost` - Total LLM cost in USD
- `cost_per_conversation` - Average cost per conversation
- `input_tokens` / `output_tokens` - Token usage

## Non-Determinism

This demo includes simulated variance to model real LLM behavior:
- Quality scores vary ±5% between runs
- Token counts have slight randomness
- Latency varies ±20%

For local optimization runs, `trial.ts` derives a deterministic seed from the
chosen config unless you pass `random_seed` explicitly. That keeps the mock
optimizer examples reproducible while preserving real-mode behavior.

You can still set `random_seed` in the agent configuration directly:

```typescript
const config: AgentConfig = {
  model: 'gpt-4o-mini',
  temperature: 0.3,
  system_prompt: 'consultative',
  memory_turns: 5,
  tool_set: 'standard',
  random_seed: 12345,  // Set for reproducible results
};
```

## Real Mastra.ai Integration

The `agent.ts` mock simulates what a real Mastra.ai agent would do. For production, see `src/mastra-agent.ts` which shows the patterns:

```typescript
import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';

const agent = new Agent({
  id: 'arkia-sales-agent',
  instructions: SYSTEM_PROMPTS[config.system_prompt],
  model: 'groq/llama-3.3-70b-versatile',
  memory: new Memory({
    options: {
      lastMessages: config.memory_turns * 2,
    },
  }),
});

const response = await agent.generate(messages, { resourceId });
```

> **Note:** `mastra-agent.ts` contains reference code only. To use real Mastra.ai, install `@mastra/core` and `@mastra/memory` packages.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No* | OpenAI API key (only for real integration) |
| `GROQ_API_KEY` | No* | Groq API key (only for real integration) |
| `TRAIGENT_COST_APPROVED` | No | Set to `true` to skip cost approval prompt |
| `TRAIGENT_EXECUTION_MODE` | No | `edge_analytics` (default) or `mock` |

\* This demo uses a mock implementation. API keys are only needed if you integrate real LLM calls.
