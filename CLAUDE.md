# @traigent/sdk

TypeScript SDK for integrating JavaScript/TypeScript LLM applications with Traigent's optimization platform.

## Overview

This SDK enables your LLM applications to participate in Traigent's optimization loop, allowing automated hyperparameter tuning for models, temperatures, prompts, and more. The SDK communicates with a Python orchestrator via NDJSON protocol.

## Installation

```bash
npm install @traigent/sdk
# or
pnpm add @traigent/sdk
# or
yarn add @traigent/sdk
```

## Usage Examples

### Basic Trial Function

Create a trial function that Traigent will call with different configurations:

```typescript
import { TrialConfig, getTrialConfig, getTrialParam } from '@traigent/sdk';

export async function runTrial(config: TrialConfig) {
  // Access the full config
  const trialConfig = getTrialConfig();
  console.log('Running trial:', config.trial_id);

  // Get specific parameters with defaults
  const model = getTrialParam('model', 'gpt-4o-mini');
  const temperature = getTrialParam<number>('temperature', 0.7);
  const maxTokens = getTrialParam<number>('max_tokens', 1000);

  // Your LLM logic here...
  const result = await yourLLMCall(model, temperature, maxTokens);

  // Return metrics
  return {
    metrics: {
      accuracy: result.accuracy,
      latency_ms: result.latency,
      custom_score: result.score,
    },
  };
}
```

### LangChain Integration

The SDK provides a built-in callback handler for LangChain that automatically captures metrics:

```typescript
import { TrialConfig, getTrialParam } from '@traigent/sdk';
import { TraigentHandler } from '@traigent/sdk/langchain';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

export async function runTrial(config: TrialConfig) {
  // Create the Traigent handler to capture metrics
  const handler = new TraigentHandler();

  // Get trial parameters
  const model = getTrialParam('model', 'gpt-4o-mini');
  const temperature = getTrialParam<number>('temperature', 0.7);

  // Initialize LangChain with the handler
  const llm = new ChatOpenAI({
    model,
    temperature,
    callbacks: [handler],
  });

  // Make LLM calls - metrics are automatically captured
  const response = await llm.invoke([
    new HumanMessage('What is the capital of France?'),
  ]);

  // Get captured metrics (tokens, cost, latency)
  const llmMetrics = handler.toMeasuresDict();

  // Combine with your custom metrics
  return {
    metrics: {
      ...llmMetrics,
      response_length: response.content.length,
      custom_accuracy: 0.95,
    },
  };
}
```

### Multiple LLM Calls

The handler aggregates metrics across multiple calls:

```typescript
import { TraigentHandler } from '@traigent/sdk/langchain';
import { ChatOpenAI } from '@langchain/openai';

export async function runTrial(config: TrialConfig) {
  const handler = new TraigentHandler();

  const llm = new ChatOpenAI({
    model: 'gpt-4o',
    callbacks: [handler],
  });

  // Multiple calls are aggregated
  const response1 = await llm.invoke('First prompt');
  const response2 = await llm.invoke('Second prompt');
  const response3 = await llm.invoke('Third prompt');

  // Metrics include totals across all calls
  const metrics = handler.toMeasuresDict();
  // {
  //   langchain_total_tokens: 450,
  //   langchain_input_tokens: 150,
  //   langchain_output_tokens: 300,
  //   langchain_total_cost: 0.015,
  //   langchain_total_latency_ms: 2500,
  //   langchain_llm_call_count: 3,
  // }

  console.log(`Made ${handler.callCount} LLM calls`);

  return { metrics };
}
```

### Using Trial Context

Access trial information from anywhere in your code using the context API:

```typescript
import {
  TrialContext,
  getTrialConfig,
  getTrialParam,
  isInTrial,
} from '@traigent/sdk';

// Check if running within a trial
function processData() {
  if (isInTrial()) {
    const config = getTrialConfig();
    console.log('Processing with trial config:', config);
  } else {
    console.log('Running outside of trial context');
  }
}

// Access specific parameters safely
function getModelConfig() {
  return {
    model: getTrialParam('model', 'gpt-4o-mini'),
    temperature: getTrialParam<number>('temperature', 0.7),
    maxTokens: getTrialParam<number>('max_tokens', 1000),
  };
}
```

### Preserving Context in Callbacks

When using APIs that don't automatically propagate async context, use `wrapCallback`:

```typescript
import { wrapCallback, getTrialConfig } from '@traigent/sdk';

export async function runTrial(config: TrialConfig) {
  // Without wrapping, getTrialConfig() would fail inside the callback
  const wrappedCallback = wrapCallback(() => {
    const cfg = getTrialConfig(); // Works!
    return processWithConfig(cfg);
  });

  // Use with setTimeout, event handlers, etc.
  await new Promise((resolve) => {
    setTimeout(async () => {
      const result = await wrappedCallback();
      resolve(result);
    }, 100);
  });

  return { metrics: { success: 1 } };
}
```

### Working with Dataset Subsets

Traigent can send specific dataset indices to evaluate:

```typescript
import { TrialConfig } from '@traigent/sdk';

export async function runTrial(config: TrialConfig) {
  const { indices } = config.dataset_subset;
  const myDataset = await loadDataset();

  // Evaluate only the specified subset
  const results = await Promise.all(
    indices.map(async (index) => {
      const example = myDataset[index];
      return evaluateExample(example);
    })
  );

  // Aggregate metrics
  const avgAccuracy = results.reduce((a, b) => a + b.accuracy, 0) / results.length;

  return {
    metrics: {
      accuracy: avgAccuracy,
      examples_evaluated: results.length,
    },
  };
}
```

### Custom Metric Prefixes

Add prefixes to organize metrics from different sources:

```typescript
import { TraigentHandler } from '@traigent/sdk/langchain';
import { prefixMeasures, mergeMeasures } from '@traigent/sdk';

export async function runTrial(config: TrialConfig) {
  // Handler with custom prefix
  const handler = new TraigentHandler({ metricPrefix: 'llm_' });

  // ... make LLM calls ...

  const llmMetrics = handler.toMeasuresDict();
  // { llm_total_cost: 0.01, llm_total_tokens: 100, ... }

  // Add prefix to custom metrics
  const customMetrics = prefixMeasures({
    accuracy: 0.95,
    f1_score: 0.92,
  }, 'eval_');
  // { eval_accuracy: 0.95, eval_f1_score: 0.92 }

  // Merge all metrics
  const allMetrics = mergeMeasures(llmMetrics, customMetrics);

  return { metrics: allMetrics };
}
```

### Error Handling

Return proper error information when trials fail:

```typescript
import { TrialConfig, createSuccessResult, createFailureResult } from '@traigent/sdk';

export async function runTrial(config: TrialConfig) {
  try {
    const result = await riskyOperation();

    return {
      metrics: {
        accuracy: result.accuracy,
        latency_ms: result.latency,
      },
    };
  } catch (error) {
    // Log the error
    console.error('Trial failed:', error);

    // Return failure result with retry hint
    throw {
      message: error.message,
      code: 'LLM_API_ERROR',
      retryable: true,
    };
  }
}
```

### Complete Example: RAG Pipeline Trial

```typescript
import { TrialConfig, getTrialParam, mergeMeasures } from '@traigent/sdk';
import { TraigentHandler } from '@traigent/sdk/langchain';
import { ChatOpenAI } from '@langchain/openai';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

export async function runTrial(config: TrialConfig) {
  const handler = new TraigentHandler();

  // Get hyperparameters from Traigent
  const model = getTrialParam('model', 'gpt-4o-mini');
  const temperature = getTrialParam<number>('temperature', 0.3);
  const topK = getTrialParam<number>('retrieval_top_k', 5);
  const chunkSize = getTrialParam<number>('chunk_size', 500);

  // Initialize components
  const llm = new ChatOpenAI({
    model,
    temperature,
    callbacks: [handler],
  });

  const embeddings = new OpenAIEmbeddings();

  // Build or load vector store
  const vectorStore = await MemoryVectorStore.fromDocuments(
    await loadAndChunkDocuments(chunkSize),
    embeddings
  );

  // Evaluate on dataset subset
  const { indices } = config.dataset_subset;
  const testQuestions = await loadTestQuestions();

  let totalScore = 0;

  for (const idx of indices) {
    const question = testQuestions[idx];

    // Retrieve context
    const docs = await vectorStore.similaritySearch(question.query, topK);
    const context = docs.map(d => d.pageContent).join('\n\n');

    // Generate answer
    const response = await llm.invoke(
      `Context: ${context}\n\nQuestion: ${question.query}\n\nAnswer:`
    );

    // Score the answer
    totalScore += scoreAnswer(response.content, question.expected);
  }

  // Calculate final metrics
  const accuracy = totalScore / indices.length;
  const llmMetrics = handler.toMeasuresDict();

  return {
    metrics: mergeMeasures(llmMetrics, {
      accuracy,
      retrieval_top_k: topK,
      chunk_size: chunkSize,
      examples_evaluated: indices.length,
    }),
  };
}
```

### Running Locally for Development

Test your trial function locally before connecting to Traigent:

```typescript
import { TrialContext } from '@traigent/sdk';
import { runTrial } from './my-trial';

// Mock trial config for local testing
const mockConfig = {
  trial_id: 'local-test-001',
  trial_number: 1,
  experiment_run_id: 'local-exp-001',
  config: {
    model: 'gpt-4o-mini',
    temperature: 0.7,
  },
  dataset_subset: {
    indices: [0, 1, 2],
    total_size: 100,
  },
};

// Run with context
async function testLocally() {
  const result = await TrialContext.run(mockConfig, async () => {
    return runTrial(mockConfig);
  });

  console.log('Trial result:', result);
}

testLocally();
```

## API Reference

### Core Exports

```typescript
import {
  // Context management
  TrialContext,
  getTrialConfig,
  getTrialParam,
  wrapCallback,
  bindContext,

  // Type checking
  isInTrial,

  // DTOs
  TrialConfig,
  TrialResultPayload,
  MeasuresDict,

  // Helpers
  createSuccessResult,
  createFailureResult,
  sanitizeMeasures,
  mergeMeasures,
  prefixMeasures,

  // Errors
  TrialContextError,
} from '@traigent/sdk';
```

### LangChain Integration

```typescript
import { TraigentHandler } from '@traigent/sdk/langchain';

const handler = new TraigentHandler({ metricPrefix?: string });

// Properties
handler.callCount;        // Number of LLM calls made

// Methods
handler.getMetrics();     // Get detailed metrics object
handler.toMeasuresDict(); // Get metrics as MeasuresDict
handler.reset();          // Clear collected metrics
```

### Captured Metrics (LangChain)

| Metric                       | Description              |
| ---------------------------- | ------------------------ |
| `langchain_total_tokens`     | Total tokens used        |
| `langchain_input_tokens`     | Input/prompt tokens      |
| `langchain_output_tokens`    | Output/completion tokens |
| `langchain_total_cost`       | Estimated cost in USD    |
| `langchain_input_cost`       | Input token cost         |
| `langchain_output_cost`      | Output token cost        |
| `langchain_total_latency_ms` | Total latency in ms      |
| `langchain_llm_call_count`   | Number of LLM calls      |

## Supported Models (Cost Estimation)

The SDK includes pricing for these models:

- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet

Unknown models fall back to gpt-4o-mini pricing.

## Python SDK Integration

The Python SDK can orchestrate JavaScript agents using the `runtime="node"` option. This enables:

- Automatic subprocess management for Node.js trials
- NDJSON protocol for communication (stdout for results, stderr for logs)
- Trial cancellation support via the `cancel` action
- Configurable timeouts and module paths

### Python Decorator Configuration

```python
from traigent import optimize

@optimize(
    execution={
        "runtime": "node",                 # Execute in Node.js subprocess
        "js_module": "./dist/trial.js",   # Path to JS module
        "js_function": "runTrial",         # Function to call (default)
        "js_timeout": 300.0,               # 5 minute timeout (default)
        "js_parallel_workers": 4,          # Parallel workers (optional)
    },
    configuration_space={                  # Note: 'configuration_space', not 'search_space'
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7],
        "system_prompt": ["concise", "detailed"],
    },
    eval_dataset="./dataset.jsonl",        # Required for JS runtime
    objectives=["accuracy", "cost"],
    max_trials=20,
    plateau_window=5,
)
async def optimize_agent(text: str) -> str:
    pass  # JS module handles execution

# Run optimization
import asyncio
result = asyncio.run(optimize_agent.optimize())
```

### JS Module Requirements

The JS module must export a function matching `js_function`:

```typescript
import { TrialConfig, getTrialParam } from '@traigent/sdk';

export async function runTrial(config: TrialConfig) {
  const model = getTrialParam('model', 'gpt-4o-mini');
  const temperature = getTrialParam<number>('temperature', 0.7);

  // Your agent logic here...

  return {
    metrics: {
      accuracy: 0.92,
      cost: 0.015,
      latency_ms: 450,
    },
  };
}
```

## Demo Application

A complete demo is available in `demos/agent-app/` showing:

- Mock sentiment classification agent
- Configuration optimization across model, temperature, and prompt type
- Session-based API integration with Traigent backend
- Detailed logging of configuration changes between trials

### Running the Demo

```bash
cd demos/agent-app
npm install
npm run dev
```

### Demo Output

The demo runs 6 trials with random configurations and displays:

```text
**********************************************************************
*  TRAIGENT OPTIMIZATION DEMO - JS AGENT                              *
**********************************************************************

Configuration Space:
  Models:      gpt-3.5-turbo, gpt-4o-mini, gpt-4o
  Temperature: 0, 0.3, 0.5, 0.7, 1
  Prompts:     concise, detailed, cot

######################################################################
# TRIAL 1 (trial_1737612345_1)
######################################################################
[TRIAL] Evaluating 10 examples from dataset

============================================================
AGENT CONFIGURATION:
  Model:       gpt-4o-mini
  Temperature: 0.3
  Prompt Type: detailed
============================================================
  [OK] "This product is amazing!..." => positive (expected: positive)
  [X]  "Terrible experience..." => neutral (expected: negative)
  ...

RESULTS:
  Accuracy:    80.0% (8/10)
  Total Cost:  $0.000110
  Avg Latency: 312ms
============================================================
```

## Session API

When running standalone (without Python orchestrator), use the session API:

```typescript
// Create optimization session
const session = await createSession({
  problem_statement: 'sentiment_agent_optimization',
  search_space: CONFIGURATION_SPACE,
  optimization_config: {
    algorithm: 'random',
    max_trials: 10,
  },
});

// Submit trial results
await submitTrialResult(session.session_id, {
  trial_id: 'trial_001',
  status: 'COMPLETED',
  config: { model: 'gpt-4o', temperature: 0.3 },
  metrics: { accuracy: 0.92, cost: 0.015 },
});
```

## CLI Protocol

The SDK includes a CLI that communicates via NDJSON protocol:

### Supported Actions

| Action       | Description                                  |
| ------------ | -------------------------------------------- |
| `run_trial`  | Execute a trial with the given configuration |
| `ping`       | Health check / keepalive                     |
| `shutdown`   | Graceful termination                         |
| `cancel`     | Cancel an in-flight trial                    |

### Request Format

```json
{
  "version": "1.0",
  "request_id": "req-001",
  "action": "run_trial",
  "payload": {
    "trial_id": "trial-001",
    "trial_number": 1,
    "config": { "model": "gpt-4o", "temperature": 0.3 },
    "dataset_subset": { "indices": [0, 1, 2], "total_size": 100 }
  }
}
```

### Response Format

```json
{
  "version": "1.0",
  "request_id": "req-001",
  "status": "success",
  "payload": {
    "metrics": { "accuracy": 0.92, "cost": 0.015 }
  }
}
```
