# @traigent/sdk

TypeScript SDK for Traigent LLM optimization platform.

This SDK enables JavaScript/TypeScript LLM applications to participate in Traigent's optimization loop via a Python-to-Node.js bridge.

## Installation

```bash
npm install @traigent/sdk
```

## Quick Start

### Within a Traigent Trial

```typescript
import { TrialContext, getTrialConfig } from '@traigent/sdk';

// Access the current trial configuration
const config = getTrialConfig();
const model = config.model as string;
const temperature = config.temperature as number;
```

### With LangChain.js

```typescript
import { TraigentHandler } from '@traigent/sdk/langchain';
import { ChatOpenAI } from '@langchain/openai';

const handler = new TraigentHandler();
const llm = new ChatOpenAI({ callbacks: [handler] });

await llm.invoke("Hello!");

// Get metrics for optimization
const metrics = handler.toMeasuresDict();
// { langchain_total_cost: 0.001, langchain_input_tokens: 5, ... }
```

## Framework Integrations

### LangChain.js

```typescript
import { TraigentHandler } from '@traigent/sdk/langchain';
```

### Vercel AI SDK (Coming Soon)

```typescript
import { withTraigent } from '@traigent/sdk/vercel-ai';
```

### OpenAI Direct (Coming Soon)

```typescript
import { createTraigentOpenAI } from '@traigent/sdk/openai';
```

## API Reference

### Core

- `TrialContext` - Manage trial context using AsyncLocalStorage
- `getTrialConfig()` - Get current trial's configuration parameters
- `getTrialParam(key, defaultValue?)` - Get a specific config parameter
- `TrialContext.isCancelled()` - Check if current trial is cancelled
- `TrialContext.checkCancellation()` - Throw if cancelled (for cooperative cancellation)
- `bindContext(fn)` - Bind a function to the current trial context
- `wrapCallback(fn)` - Wrap a callback to preserve trial context

### DTOs

- `TrialConfig` - Trial configuration from orchestrator
- `TrialResultPayload` - Trial result to return
- `MeasuresDict` - Validated metrics dictionary

## CLI Runner

The SDK includes a CLI runner for executing trials via the Python-to-JS bridge:

```bash
npx traigent-runner --module ./dist/my-agent.js --function runTrial
```

The runner communicates with the Python orchestrator via NDJSON over stdin/stdout.

See [agent.md](./agent.md) for detailed bridge architecture and protocol documentation.

## Requirements

- Node.js >= 18.0.0
- TypeScript >= 5.0 (for development)

## License

MIT
