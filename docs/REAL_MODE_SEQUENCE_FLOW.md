# Real Production Mode Execution - Sequence Flow

> Legacy/orchestrated reference. This document describes the Python-to-Node bridge and CLI runner flow, not the current recommended JS-native `wrapped.optimize(...)` or hybrid spec-authoring flow.

For the current JS-facing API, use [README.md](../README.md) and [CLIENT_CODE_GUIDE.md](./CLIENT_CODE_GUIDE.md).

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Python Side                                     │
│  ┌──────────────┐   ┌────────────────────┐   ┌──────────────────────┐       │
│  │ User Script  │ → │ @traigent.optimize │ → │ OptimizationOrchestrator │   │
│  │ (run_with_   │   │ (decorators.py)    │   │ (orchestrator.py)     │       │
│  │  python.py)  │   └────────────────────┘   └──────────┬───────────┘       │
│  └──────────────┘                                       │                    │
│                                           ┌─────────────▼────────────┐       │
│                                           │     JSEvaluator          │       │
│                                           │   (js_evaluator.py)      │       │
│                                           └─────────────┬────────────┘       │
│                                           ┌─────────────▼────────────┐       │
│                                           │    JSProcessPool         │       │
│                                           │   (process_pool.py)      │       │
│                                           └─────────────┬────────────┘       │
│                                           ┌─────────────▼────────────┐       │
│                                           │      JSBridge            │       │
│                                           │    (js_bridge.py)        │       │
│                                           └─────────────┬────────────┘       │
└─────────────────────────────────────────────────────────┼───────────────────┘
                                                          │ NDJSON over
                                                          │ stdin/stdout
┌─────────────────────────────────────────────────────────┼───────────────────┐
│                              Node.js Side               │                    │
│                                           ┌─────────────▼────────────┐       │
│                                           │     CLI Runner           │       │
│                                           │    (runner.ts)           │       │
│                                           └─────────────┬────────────┘       │
│                                           ┌─────────────▼────────────┐       │
│                                           │    User Trial Function   │       │
│                                           │      (trial.ts)          │       │
│                                           └─────────────┬────────────┘       │
│                                           ┌─────────────▼────────────┐       │
│                                           │      Agent Logic         │       │
│                                           │     (agent.ts/mastra)    │       │
│                                           └──────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Mermaid Sequence Diagram

```mermaid
sequenceDiagram
    participant User as User Script
    participant Dec as @traigent.optimize
    participant OF as OptimizedFunction
    participant Orch as Orchestrator
    participant Opt as Optimizer
    participant Eval as JSEvaluator
    participant Pool as JSProcessPool
    participant Bridge as JSBridge
    participant Runner as CLI Runner (Node.js)
    participant Trial as runTrial()
    participant Agent as Agent/LLM
    participant Backend as Backend API

    %% Initialization
    User->>Dec: @traigent.optimize(execution={runtime:"node"})
    Dec->>OF: OptimizedFunction(func, config_space, ...)
    User->>OF: await optimize()

    %% Setup
    OF->>Opt: get_optimizer("random", max_trials=24)
    OF->>Pool: JSProcessPool(max_workers=4, module_path)
    Pool->>Bridge: JSBridge(config) [x4 workers]

    %% Worker startup
    loop For each worker
        Bridge->>Runner: subprocess_exec("npx traigent-js --module ...")
        Runner->>Runner: loadTrialFunction(module, "runTrial")
        Runner->>Runner: startParentPidWatcher()
        Runner->>Runner: rl = createInterface(stdin)
        Bridge->>Runner: ping request (NDJSON)
        Runner-->>Bridge: pong response (NDJSON)
    end

    OF->>Eval: JSEvaluator(process_pool=pool)
    OF->>Orch: OptimizationOrchestrator(optimizer, evaluator)

    %% Trial Loop
    loop While not should_stop
        Orch->>Opt: suggest() → config
        Note over Opt: config = {model, temperature, ...}

        Orch->>Eval: evaluate(func, config, dataset)
        Eval->>Eval: _build_trial_config(trial_id, config, indices)
        Eval->>Pool: run_trial(trial_config)
        Pool->>Bridge: acquire() → worker

        %% NDJSON Protocol
        Bridge->>Runner: run_trial request (NDJSON stdin)
        Note over Bridge,Runner: {"action":"run_trial","payload":{...}}

        Runner->>Runner: parseRequest() + Zod validation
        Runner->>Runner: acquireTrialLock()
        Runner->>Runner: currentTrialId = trial_id
        Runner->>Trial: TrialContext.run(config, trialFn)

        Trial->>Trial: getDatasetSubset(indices)
        Trial->>Agent: runSalesAgent(examples, agentConfig)

        alt REAL_MODE=true
            Agent->>Agent: realLLMCall(model, messages)
            Note over Agent: Actual Groq/OpenAI API call
        else Mock Mode
            Agent->>Agent: simulateMetrics(config)
        end

        Agent-->>Trial: {metrics, latency, tokens, cost}
        Trial-->>Runner: {metrics, duration, metadata}

        Runner->>Runner: sanitizeMeasures(metrics)
        Runner->>Runner: createSuccessResult(trial_id, metrics)
        Runner-->>Bridge: success response (NDJSON stdout)
        Note over Runner,Bridge: {"status":"success","payload":{metrics}}

        Bridge->>Bridge: _parse_trial_response() → JSTrialResult
        Pool->>Bridge: release(worker)
        Bridge-->>Eval: JSTrialResult

        Eval->>Eval: _convert_js_result() → EvaluationResult
        Eval-->>Orch: EvaluationResult

        Orch->>Opt: report(config, metrics)
        Orch->>Orch: _update_best(config, result)
        Orch->>Backend: submit_trial(trial_result)
    end

    %% Shutdown
    Orch-->>OF: OptimizationResult
    OF->>Pool: shutdown()
    Pool->>Bridge: stop() [x4 workers]
    Bridge->>Runner: shutdown request (NDJSON)
    Runner->>Runner: process.exit(0)
    OF-->>User: OptimizationResult(best_config, best_metrics)
```

---

## Detailed Sequence Flow with Code References

### Phase 1: Initialization

**Entry Point:** [run_with_python.py:166-179](../demos/arkia-sales-agent/run_with_python.py#L166-L179)

```python
async def main():
    setup_node_path()
    check_node_version()
    js_module_path = verify_build()
    dataset_path = verify_dataset()

    traigent.initialize(
        execution_mode=os.getenv("TRAIGENT_EXECUTION_MODE", "edge_analytics"),
    )
```

**SDK Initialize:** [traigent/api/functions.py:167-218](../../../Traigent/traigent/api/functions.py#L167-L218)

---

### Phase 2: Decorator Processing

**Decorator Definition:** [run_with_python.py:187-233](../demos/arkia-sales-agent/run_with_python.py#L187-L233)

```python
@traigent.optimize(
    execution={
        "runtime": "node",
        "js_module": js_module_path,
        "js_function": "runTrial",
        "js_timeout": 60.0,
        "js_parallel_workers": 4,
    },
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", ...],
        "temperature": [0.0, 0.3, 0.5, 0.7],
        ...
    },
    objectives=["margin_efficiency", "conversion_score", "cost"],
    max_trials=24,
)
def arkia_sales_agent(customer_query: str, config: dict = None) -> str:
    pass
```

**Creates:** `OptimizedFunction` via [traigent/api/decorators.py](../../../Traigent/traigent/api/decorators.py)

---

### Phase 3: Optimization Start

**Trigger:** [run_with_python.py:285](../demos/arkia-sales-agent/run_with_python.py#L285)

```python
result = await arkia_sales_agent.optimize()
```

**OptimizedFunction.optimize():** [traigent/core/optimized_function.py](../../../Traigent/traigent/core/optimized_function.py)

**Creates:**
| Component | Class | File |
|-----------|-------|------|
| Optimizer | `RandomOptimizer` | [traigent/optimizers/random.py](../../../Traigent/traigent/optimizers/random.py) |
| Process Pool | `JSProcessPool` | [traigent/bridges/process_pool.py:89-431](../../../Traigent/traigent/bridges/process_pool.py#L89-L431) |
| Evaluator | `JSEvaluator` | [traigent/evaluators/js_evaluator.py:55-282](../../../Traigent/traigent/evaluators/js_evaluator.py#L55-L282) |
| Orchestrator | `OptimizationOrchestrator` | [traigent/core/orchestrator.py:105-200](../../../Traigent/traigent/core/orchestrator.py#L105-L200) |

---

### Phase 4: Process Pool Startup

**Pool Start:** [traigent/bridges/process_pool.py:155-207](../../../Traigent/traigent/bridges/process_pool.py#L155-L207)

```python
async def _start_workers(self) -> None:
    for _ in range(self._config.max_workers):
        bridge = JSBridge(bridge_config)
        self._workers.append(bridge)
        await bridge.start()
```

**Bridge Start:** [traigent/bridges/js_bridge.py:173-276](../../../Traigent/traigent/bridges/js_bridge.py#L173-L276)

```python
async def start(self) -> None:
    cmd = ["npx", "traigent-js", "--module", module_path, "--function", "runTrial"]

    self._process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    self._reader_task = asyncio.create_task(self._read_responses())
    await self.ping(timeout=10.0)  # Health check
```

---

### Phase 5: JS Runner Initialization (Node.js Side)

**Main Entry:** [src/cli/runner.ts:646-728](../src/cli/runner.ts#L646-L728)

```typescript
async function main(): Promise<void> {
  const args = parseCLIArgs();

  trialFn = await loadTrialFunction(args.module, args.function ?? 'runTrial');

  startParentPidWatcher();
  resetIdleTimer();

  rl = createInterface({ input: process.stdin, crlfDelay: Infinity });

  rl.on('line', (line) => {
    processRequest(line, trialFn);
  });
}
```

**Key Functions:**
| Function | Location | Purpose |
|----------|----------|---------|
| `loadTrialFunction` | [runner.ts:169-190](../src/cli/runner.ts#L169-L190) | Dynamic import of user module |
| `startParentPidWatcher` | [runner.ts:539-572](../src/cli/runner.ts#L539-L572) | Orphan detection |
| `resetIdleTimer` | [runner.ts:526-537](../src/cli/runner.ts#L526-L537) | Idle timeout management |

---

### Phase 6: Trial Loop (Orchestrator)

**Main Loop:** [traigent/core/orchestrator.py](../../../Traigent/traigent/core/orchestrator.py) (simplified)

```python
async def run(self, dataset: Dataset) -> OptimizationResult:
    while not self._should_stop():
        config = await self.optimizer.suggest()
        result = await self.evaluator.evaluate(func, config, dataset)
        await self.optimizer.report(config, result.aggregated_metrics)
        self._update_best(config, result)
        await self._session_manager.submit_trial(trial_result)
```

---

### Phase 7: JS Evaluation

**Evaluate Method:** [traigent/evaluators/js_evaluator.py:223-281](../../../Traigent/traigent/evaluators/js_evaluator.py#L223-L281)

```python
async def evaluate(self, func, config, dataset) -> EvaluationResult:
    trial_id = f"js_trial_{uuid.uuid4().hex[:8]}"

    trial_config = {
        "trial_id": trial_id,
        "trial_number": self._trial_counter,
        "config": config,
        "dataset_subset": {"indices": indices, "total": len(dataset)},
    }

    result = await self._process_pool.run_trial(trial_config)
    return self._convert_js_result(result, config, indices)
```

**Build Trial Config:** [js_evaluator.py:135-163](../../../Traigent/traigent/evaluators/js_evaluator.py#L135-L163)

---

### Phase 8: NDJSON Protocol Communication

**Send Request:** [traigent/bridges/js_bridge.py:499-558](../../../Traigent/traigent/bridges/js_bridge.py#L499-L558)

```python
async def _send_request(self, action: str, payload: dict, timeout: float):
    request = {
        "version": "1.0",
        "request_id": str(uuid.uuid4()),
        "action": action,
        "payload": payload,
    }
    request_line = json.dumps(request) + "\n"
    self._process.stdin.write(request_line.encode())
    await self._process.stdin.drain()
```

**Protocol Schema:** [src/cli/protocol.ts:62-73](../src/cli/protocol.ts#L62-L73)

```typescript
export const CLIRequestSchema = z.object({
  version: ProtocolVersionSchema,
  request_id: z.string().min(1),
  action: ActionSchema,
  payload: z.unknown(),
});
```

**NDJSON Request Example:**
```json
{
  "version": "1.0",
  "request_id": "abc123-uuid",
  "action": "run_trial",
  "payload": {
    "trial_id": "js_trial_4ffe457b",
    "trial_number": 1,
    "config": {
      "model": "groq/llama-3.1-8b-instant",
      "temperature": 0.0,
      "system_prompt": "consultative",
      "memory_turns": 2,
      "tool_set": "enhanced"
    },
    "dataset_subset": {
      "indices": [0, 1, 2, "...", 49],
      "total": 50
    },
    "timeout_ms": 60000
  }
}
```

---

### Phase 9: JS Request Processing

**Process Request:** [src/cli/runner.ts:578-641](../src/cli/runner.ts#L578-L641)

```typescript
async function processRequest(line: string, trialFn: UserTrialFunction): Promise<void> {
  const lineBytes = Buffer.byteLength(line, 'utf8');
  if (lineBytes > MAX_PAYLOAD_SIZE) {
    sendResponse(createErrorResponse('unknown', 'Payload too large', {errorCode: 'PAYLOAD_TOO_LARGE'}));
    return;
  }

  const request = parseRequest(line);  // Zod validation

  switch (request.action) {
    case 'run_trial':
      response = await handleRunTrial(request, trialFn);
      break;
    // ...
  }

  sendResponse(response);
}
```

**Parse Request:** [src/cli/protocol.ts:275-284](../src/cli/protocol.ts#L275-L284)

---

### Phase 10: Trial Execution with Lock

**Handle Run Trial:** [src/cli/runner.ts:267-401](../src/cli/runner.ts#L267-L401)

```typescript
async function handleRunTrial(request: CLIRequest, trialFn: UserTrialFunction): Promise<CLIResponse> {
  const trialStartTime = Date.now();

  // 1. Acquire lock
  const unlock = await acquireTrialLock();

  // 2. Check busy
  if (currentTrialId !== null) {
    unlock();
    return createErrorResponse(request.request_id, new BusyError(...));
  }

  // 3. Validate with Zod
  const parseResult = TrialConfigSchema.safeParse(request.payload);

  // 4. Set state IMMEDIATELY
  currentTrialId = config.trial_id;
  currentTrialAbortController = new AbortController();

  try {
    // 5. Execute with context
    const result = await TrialContext.run(config, async () => {
      return await trialFn(config);
    }, abortSignal);

    // 6. Sanitize & respond
    const metrics = sanitizeMeasures(result.metrics);
    return createSuccessResponse(request.request_id, createSuccessResult(...));
  } finally {
    currentTrialId = null;
    unlock();
  }
}
```

**Key Functions:**
| Function | Location | Purpose |
|----------|----------|---------|
| `acquireTrialLock` | [runner.ts:76-84](../src/cli/runner.ts#L76-L84) | Race condition prevention |
| `TrialContext.run` | [src/core/context.ts](../src/core/context.ts) | AsyncLocalStorage context |
| `sanitizeMeasures` | [src/dtos/measures.ts](../src/dtos/measures.ts) | Metric validation |

---

### Phase 11: User Trial Function

**Trial Function:** [demos/arkia-sales-agent/src/trial.ts:104-258](../demos/arkia-sales-agent/src/trial.ts#L104-L258)

```typescript
export async function runTrial(trialConfig: TrialConfig): Promise<TrialResult> {
  const agentConfig: AgentConfig = {
    model: trialConfig.config.model ?? 'gpt-4o-mini',
    temperature: trialConfig.config.temperature ?? 0.5,
    system_prompt: trialConfig.config.system_prompt ?? 'consultative',
    memory_turns: trialConfig.config.memory_turns ?? 5,
    tool_set: trialConfig.config.tool_set ?? 'standard',
  };

  const examples = getDatasetSubset(trialConfig.dataset_subset.indices);
  const result = await runSalesAgent(examples, agentConfig, console.error);

  return {
    metrics: {
      accuracy: (result.avg_relevancy + result.avg_completeness) / 2,
      response_time: result.avg_latency_ms,
      conversion_score: result.avg_conversion_score,
      cost: result.total_cost,
      margin_efficiency: result.margin_efficiency,
      // ...
    },
    duration: Date.now() - startTime,
  };
}
```

---

### Phase 12: Agent Execution (REAL_MODE)

**Agent Runner:** [demos/arkia-sales-agent/src/agent.ts](../demos/arkia-sales-agent/src/agent.ts)

**Real LLM Call:** [demos/arkia-sales-agent/src/real-llm.ts](../demos/arkia-sales-agent/src/real-llm.ts)

```typescript
// When REAL_MODE=true
export async function realLLMCall(config: LLMConfig): Promise<LLMResponse> {
  const response = await fetch('https://api.groq.com/...', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${GROQ_API_KEY}` },
    body: JSON.stringify({
      model: config.model,  // "groq/llama-3.1-8b-instant"
      temperature: config.temperature,
      messages: config.messages,
    }),
  });

  return {
    content: response.choices[0].message.content,
    usage: { input_tokens, output_tokens, cost },
    latency: responseTime,
  };
}
```

---

### Phase 13: Response Back to Python

**Send Response:** [src/cli/runner.ts:213-235](../src/cli/runner.ts#L213-L235)

```typescript
function sendResponse(response: CLIResponse): void {
  const line = serializeResponse(response) + '\n';

  if (draining) {
    writeQueue.push(line);
    return;
  }

  const ok = process.stdout.write(line);
  if (!ok) {
    draining = true;
    rl?.pause();  // Backpressure
    process.stdout.once('drain', flushWriteQueue);
  }
}
```

**NDJSON Response Example:**
```json
{
  "version": "1.1",
  "request_id": "abc123-uuid",
  "status": "success",
  "payload": {
    "trial_id": "js_trial_4ffe457b",
    "status": "completed",
    "metrics": {
      "accuracy": 0.758,
      "response_time": 335.12,
      "relevancy": 0.776,
      "completeness": 0.74,
      "conversion_score": 0.578,
      "cost": 0.001231,
      "input_tokens": 5976,
      "output_tokens": 3088,
      "margin_efficiency": 2.348
    },
    "duration": 32.04,
    "metadata": {
      "examples_processed": 50
    }
  }
}
```

**Python Receives:** [traigent/bridges/js_bridge.py:579-597](../../../Traigent/traigent/bridges/js_bridge.py#L579-L597)

**Parse Response:** [traigent/bridges/js_bridge.py:448-497](../../../Traigent/traigent/bridges/js_bridge.py#L448-L497)

---

### Phase 14: Backend Submission

**Submit Trial:** [traigent/core/backend_session_manager.py](../../../Traigent/traigent/core/backend_session_manager.py)

```python
async def submit_trial(self, trial_result: TrialResult):
    await self._client.post(
        f"/experiments/{experiment_id}/runs/{run_id}/configurations",
        json={
            "trial_id": trial_result.trial_id,
            "status": "COMPLETED",
            "config": trial_result.config,
            "metrics": trial_result.metrics,
        }
    )
```

---

## Complete Call Chain Summary

```
User Script (run_with_python.py:285)
└── arkia_sales_agent.optimize()
    └── OptimizedFunction.optimize() (optimized_function.py)
        ├── get_optimizer("random") (optimizers/__init__.py)
        ├── JSProcessPool(config) (bridges/process_pool.py:114)
        │   └── _start_workers() (process_pool.py:155)
        │       └── JSBridge.start() [x4] (js_bridge.py:173)
        │           ├── subprocess_exec("npx traigent-js ...") (js_bridge.py:243)
        │           │   └── [Node.js] main() (runner.ts:646)
        │           │       ├── loadTrialFunction() (runner.ts:169)
        │           │       ├── startParentPidWatcher() (runner.ts:544)
        │           │       └── rl.on('line', processRequest) (runner.ts:697)
        │           └── ping() (js_bridge.py:369)
        │               └── [Node.js] handlePing() (runner.ts:406)
        ├── JSEvaluator(pool) (js_evaluator.py:79)
        └── OptimizationOrchestrator.run() (orchestrator.py)
            └── [LOOP] while not should_stop:
                ├── optimizer.suggest() → config
                ├── evaluator.evaluate(config) (js_evaluator.py:223)
                │   ├── _build_trial_config() (js_evaluator.py:135)
                │   └── process_pool.run_trial() (process_pool.py:340)
                │       ├── acquire() (process_pool.py:223)
                │       ├── worker.run_trial() (js_bridge.py:405)
                │       │   └── _send_request("run_trial") (js_bridge.py:499)
                │       │       ├── stdin.write(NDJSON)
                │       │       │   └── [Node.js] processRequest() (runner.ts:578)
                │       │       │       ├── parseRequest() (protocol.ts:275)
                │       │       │       ├── handleRunTrial() (runner.ts:271)
                │       │       │       │   ├── acquireTrialLock() (runner.ts:76)
                │       │       │       │   ├── TrialContext.run() (context.ts)
                │       │       │       │   │   └── runTrial() (trial.ts:104)
                │       │       │       │   │       └── runSalesAgent() (agent.ts)
                │       │       │       │   │           └── realLLMCall() [REAL_MODE] (real-llm.ts)
                │       │       │       │   ├── sanitizeMeasures() (measures.ts)
                │       │       │       │   └── createSuccessResult() (trial.ts)
                │       │       │       └── sendResponse() (runner.ts:217)
                │       │       └── await response_future
                │       ├── _parse_trial_response() (js_bridge.py:448)
                │       └── release(worker) (process_pool.py:269)
                ├── _convert_js_result() (js_evaluator.py:179)
                ├── optimizer.report(metrics)
                └── session_manager.submit_trial()
```

---

## Key Data Objects

### TrialConfig (Python → JS)

```typescript
// Defined in: src/dtos/trial.ts
interface TrialConfig {
  trial_id: string;           // "js_trial_4ffe457b"
  trial_number: number;       // 1
  experiment_run_id: string;  // "uuid"
  config: {
    model: string;            // "groq/llama-3.1-8b-instant"
    temperature: number;      // 0.0
    system_prompt: string;    // "consultative"
    memory_turns: number;     // 2
    tool_set: string;         // "enhanced"
  };
  dataset_subset: {
    indices: number[];        // [0, 1, 2, ..., 49]
    total: number;            // 50
  };
}
```

### JSTrialResult (JS → Python)

```python
# Defined in: traigent/bridges/js_bridge.py:108-131
@dataclass
class JSTrialResult:
    trial_id: str           # "js_trial_4ffe457b"
    status: str             # "completed" | "failed" | "cancelled"
    metrics: dict           # {accuracy: 0.758, cost: 0.001231, ...}
    duration: float         # 32.04 (seconds)
    error_message: str      # None for success
    error_code: str         # "VALIDATION_ERROR", "TIMEOUT", etc.
    retryable: bool         # True/False
    metadata: dict          # {examples_processed: 50}
```

### EvaluationResult (Internal Python)

```python
# Defined in: traigent/evaluators/base.py
@dataclass
class EvaluationResult:
    config: dict                    # The tested config
    example_results: list           # Per-example results (empty for JS)
    aggregated_metrics: dict        # {accuracy: 0.758, cost: 0.001231}
    total_examples: int             # 50
    successful_examples: int        # 50
    duration: float                 # 32.04
    errors: list[str]               # [] for success
    summary_stats: dict             # Metadata
```

---

## Error Codes Reference

| Code | Location | Meaning |
|------|----------|---------|
| `TIMEOUT` | [runner.ts](../src/cli/runner.ts) | Trial exceeded timeout_ms |
| `VALIDATION_ERROR` | [runner.ts](../src/cli/runner.ts) | Zod schema validation failed |
| `BUSY` | [runner.ts](../src/cli/runner.ts) | Another trial already running |
| `CANCELLED` | [runner.ts](../src/cli/runner.ts) | Trial cancelled via cancel action |
| `USER_FUNCTION_ERROR` | [runner.ts](../src/cli/runner.ts) | User's runTrial() threw |
| `MODULE_LOAD_ERROR` | [runner.ts](../src/cli/runner.ts) | Failed to load JS module |
| `PROTOCOL_ERROR` | [protocol.ts](../src/cli/protocol.ts) | NDJSON parse/validation failed |
| `PAYLOAD_TOO_LARGE` | [runner.ts](../src/cli/runner.ts) | Request > 10MB |

---

## File Index

### Python SDK (Traigent)

| File | Purpose |
|------|---------|
| [traigent/api/decorators.py](../../../Traigent/traigent/api/decorators.py) | `@traigent.optimize` decorator |
| [traigent/core/optimized_function.py](../../../Traigent/traigent/core/optimized_function.py) | `OptimizedFunction` class |
| [traigent/core/orchestrator.py](../../../Traigent/traigent/core/orchestrator.py) | Trial loop orchestration |
| [traigent/evaluators/js_evaluator.py](../../../Traigent/traigent/evaluators/js_evaluator.py) | JS runtime evaluator |
| [traigent/bridges/js_bridge.py](../../../Traigent/traigent/bridges/js_bridge.py) | Node.js subprocess manager |
| [traigent/bridges/process_pool.py](../../../Traigent/traigent/bridges/process_pool.py) | Worker pool for parallelism |

### TypeScript SDK (traigent-js)

| File | Purpose |
|------|---------|
| [src/cli/runner.ts](../src/cli/runner.ts) | CLI entry point, request handling |
| [src/cli/protocol.ts](../src/cli/protocol.ts) | NDJSON schema definitions |
| [src/core/context.ts](../src/core/context.ts) | Trial context (AsyncLocalStorage) |
| [src/dtos/trial.ts](../src/dtos/trial.ts) | Trial config/result types |
| [src/dtos/measures.ts](../src/dtos/measures.ts) | Metric sanitization |

### Demo Application

| File | Purpose |
|------|---------|
| [demos/arkia-sales-agent/run_with_python.py](../demos/arkia-sales-agent/run_with_python.py) | Python orchestrator script |
| [demos/arkia-sales-agent/src/trial.ts](../demos/arkia-sales-agent/src/trial.ts) | Trial function entry |
| [demos/arkia-sales-agent/src/agent.ts](../demos/arkia-sales-agent/src/agent.ts) | Agent logic (mock/real) |
| [demos/arkia-sales-agent/src/real-llm.ts](../demos/arkia-sales-agent/src/real-llm.ts) | Real LLM API calls |
