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

### DTOs

- `TrialConfig` - Trial configuration from orchestrator
- `TrialResultPayload` - Trial result to return
- `MeasuresDict` - Validated metrics dictionary

## Requirements

- Node.js >= 18.0.0
- TypeScript >= 5.0 (for development)

## License

MIT
