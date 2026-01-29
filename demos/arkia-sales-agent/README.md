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
│   ├── agent.ts          # MOCK agent (simulates LLM, no real API calls)
│   ├── tools.ts          # Travel agent tools (flight search, booking, etc.)
│   ├── mastra-agent.ts   # Reference: REAL Mastra.ai patterns (not used)
│   ├── dataset.ts        # Curated sales conversations (Hebrew + English)
│   ├── trial.ts          # Entry point for Python orchestrator
│   └── standalone.ts     # Local demo runner
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

### Local Demo (No API keys needed)

```bash
cd demos/arkia-sales-agent
npm install
npm run build
npm run dev
```

### With Python Orchestrator

```bash
cd demos/arkia-sales-agent
npm run build
TRAIGENT_COST_APPROVED=true python run_with_python.py
```

## Sample Output

Running `npm run dev` produces:

```
**********************************************************************
*  ARKIA SALES AGENT - LOCAL DEMO
*  Margin Optimization Demonstration
**********************************************************************

Dataset Statistics:
  Total examples: 20
  By intent: { flight_inquiry: 7, price_negotiation: 4, booking_intent: 4, support: 3, complaint: 2 }

======================================================================
COMPARISON: OpenAI SOTA vs OpenAI Optimized vs Groq
======================================================================

                      GPT-4o      GPT-4o-mini   Groq 70B    Groq 8B
----------------------------------------------------------------------
Conversion:           99.3%        88.5%          98.4%        68.6%
Cost:                 $0.04543    $0.00161      $0.00528    $0.00036
Latency (avg):        437ms         244ms          82ms         30ms
Margin Efficiency:    0.02          0.55           0.19         1.90
----------------------------------------------------------------------

[WINNER] Groq Llama 8B has the best margin efficiency: 1.90
         8588% better than GPT-4o SOTA!
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

**Results may differ between runs.** For reproducible results, set `random_seed` in the agent configuration:

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
