# Bazak Server <-> Traigent Client — Sequence Diagram

## Full Optimization Loop (Two-Phase Mode)

Shows the actual flow between Traigent SDK (`HybridAPIEvaluator` + `OptimizationOrchestrator`)
and the Bazak server (`child-age-capability.js` via `server.js`), as implemented in
`run_mastra_js_optimization.py`.

```mermaid
sequenceDiagram
    box rgb(40,40,60) Traigent SDK (Python)
        participant Orch as OptimizationOrchestrator
        participant Eval as HybridAPIEvaluator
        participant HTTP as HTTPTransport
    end

    box rgb(60,40,40) Bazak Server (Node.js)
        participant Server as server.js<br/>(Express router)
        participant Cap as child-age-capability.js
        participant Agent as Mastra Agent<br/>(@mastra/core)
        participant Store as outputStore<br/>(in-memory Map)
        participant DS as child-age-dataset.json<br/>(100 cases)
    end

    participant OpenAI as OpenAI API

    Note over Orch,DS: Phase 0 — Discovery & Setup

    Orch->>Eval: discover_config_space()
    Eval->>HTTP: capabilities()
    HTTP->>Server: GET /traigent/v1/capabilities
    Server->>Cap: handleCapabilities()
    Cap-->>Server: {version, supports_evaluate: true, max_batch_size: 100, ...}
    Server-->>HTTP: 200 ServiceCapabilities
    HTTP-->>Eval: ServiceCapabilities

    Eval->>HTTP: discover_config_space()
    HTTP->>Server: GET /traigent/v1/config-space
    Server->>Cap: getConfigSpace()

    Note right of Cap: Reads tunables.json:<br/>model: [gpt-4o, gpt-4o-mini]<br/>temperature: [0.0 .. 1.0]<br/>system_prompt_version: [v1,v2,v3]<br/>max_retries: [0..3]

    Cap-->>Server: {schema_version: "0.9", tunable_id: "child-age-agent", tunables: [...]}
    Server-->>HTTP: 200 ConfigSpaceResponse
    HTTP-->>Eval: ConfigSpaceResponse
    Eval-->>Orch: config_space dict

    Note over Orch: Creates RandomSearchOptimizer<br/>with discovered config_space

    Note over Orch,DS: Phase 1 — Optimization Loop (repeats per trial)

    loop Trial 1..N (max_trials or budget exhausted)
        Note over Orch: Optimizer samples a config, e.g.:<br/>{model: "gpt-4o-mini", temperature: 0.3,<br/>system_prompt_version: "v2", max_retries: 1}

        Orch->>Eval: evaluate(config, dataset)

        loop Batch 1..M (dataset / batch_size)
            Note over Eval: Prepares batch of example_ids:<br/>[{example_id: "case_001"}, {example_id: "case_002"}, ...]

            Eval->>HTTP: execute(HybridExecuteRequest)
            HTTP->>Server: POST /traigent/v1/execute
            Note right of HTTP: {tunable_id: "child-age-agent",<br/>config: {model, temperature, ...},<br/>examples: [{example_id: "case_001"}, ...],<br/>timeout_ms: 60000}

            Server->>Cap: execute(body)

            loop For each input in batch
                Cap->>DS: Lookup query by example_id
                DS-->>Cap: {query: "Cruise to Bahamas for me and my son?", expected_behavior: "should_ask_ages"}

                Cap->>Agent: agent.generate(query, {instructions, modelSettings: {temperature}})
                Agent->>OpenAI: Chat Completion (model, messages, temperature)
                OpenAI-->>Agent: {text, usage: {input_tokens, output_tokens}}
                Agent-->>Cap: generationResult

                Note right of Cap: estimateUsageCostUsd()<br/>→ cost per input

                Cap->>Store: Store output by output_id<br/>key: "out_case_001_exec_uuid"<br/>val: {response, cost_usd, latency_ms, ...}
            end

            Cap-->>Server: {request_id, execution_id: "exec_...",<br/>status: "completed",<br/>outputs: [{example_id, output_id, cost_usd, latency_ms, tokens_used}, ...],<br/>operational_metrics: {total_cost_usd, latency_ms, ...},<br/>quality_metrics: null}
            Server-->>HTTP: 200 ExecuteResponse
            HTTP-->>Eval: HybridExecuteResponse

            Note over Eval: quality_metrics is null →<br/>supports_evaluate is true →<br/>Two-phase mode: call /evaluate

            Eval->>HTTP: evaluate(HybridEvaluateRequest)
            HTTP->>Server: POST /traigent/v1/evaluate
            Note right of HTTP: {tunable_id: "child-age-agent",<br/>execution_id: "exec_...",<br/>evaluations: [{example_id: "case_001",<br/>output_id: "out_case_001_exec_..."}, ...]}

            Server->>Cap: evaluate(body)

            loop For each evaluation item
                Cap->>Store: Lookup response by output_id
                Store-->>Cap: {response: "How old is your son?"}
                Cap->>DS: Lookup expected_behavior by example_id
                DS-->>Cap: expected_behavior: "should_ask_ages"
                Note right of Cap: detectAgeQuestion(response)<br/>→ true → accuracy = 1<br/><br/>If "should_ask_ages" &<br/>askedAges → accuracy=1<br/>If "should_not_ask_ages" &<br/>!askedAges → accuracy=1
            end

            Note right of Cap: aggregate: mean(accuracyValues),<br/>std(accuracyValues), n

            Cap-->>Server: {request_id, status: "completed",<br/>results: [{example_id, metrics: {accuracy: 0 or 1}}, ...],<br/>aggregate_metrics: {accuracy: {mean, std, n}}}
            Server-->>HTTP: 200 EvaluateResponse
            HTTP-->>Eval: HybridEvaluateResponse
        end

        Note over Eval: Merges execute metrics (cost, latency)<br/>with evaluate metrics (accuracy)<br/>per example

        Eval-->>Orch: EvaluationResult {aggregated_metrics, total_examples}

        Note over Orch: Records trial:<br/>config → {accuracy: 0.82, cost: $0.003, latency: 2400ms}<br/>Updates best_config if score improved
    end

    Note over Orch,DS: Phase 2 — Results

    Note over Orch: Returns OptimizationResult:<br/>- best_config: {model, temperature, ...}<br/>- best_score: 0.93<br/>- best_metrics: {accuracy, cost, latency}<br/>- all trials with convergence info<br/>- stop_reason: "max_trials" | "budget_exhausted"
```

## Single Trial Detail (Privacy Mode — Bazak Default)

Zoomed-in view of one execute+evaluate cycle in privacy mode,
where only `example_id` and `output_id` cross the wire (no actual query/response text).

```mermaid
sequenceDiagram
    participant SDK as Traigent SDK
    participant Bazak as Bazak Server
    participant Store as outputStore (Map)
    participant Dataset as dataset.json
    participant LLM as OpenAI API

    Note over SDK,LLM: Execute Phase

    SDK->>Bazak: POST /traigent/v1/execute
    Note right of SDK: Only IDs sent (privacy mode):<br/>{tunable_id: "child-age-agent",<br/>config: {model: "gpt-4o", temp: 0.2, ...},<br/>examples: [{example_id: "case_042"}]}

    Bazak->>Dataset: Lookup case_042
    Dataset-->>Bazak: query: "Trip for me and my 2 kids aged 5 and 8"

    Bazak->>LLM: generate(query, instructions, temperature)
    LLM-->>Bazak: "Great! I can help with your family trip..."

    Bazak->>Store: Store: out_case_042_exec_abc → {response, cost, latency}

    Bazak-->>SDK: {execution_id: "exec_abc",<br/>outputs: [{example_id: "case_042",<br/>output_id: "out_case_042_exec_abc",<br/>cost_usd: 0.00015, latency_ms: 820}],<br/>quality_metrics: null}

    Note over SDK: No response text returned!<br/>Only output_id + operational metrics

    Note over SDK,LLM: Evaluate Phase

    SDK->>Bazak: POST /traigent/v1/evaluate
    Note right of SDK: References only IDs:<br/>{execution_id: "exec_abc",<br/>evaluations: [{example_id: "case_042",<br/>output_id: "out_case_042_exec_abc"}]}

    Bazak->>Store: Lookup out_case_042_exec_abc
    Store-->>Bazak: response: "Great! I can help..."

    Bazak->>Dataset: Lookup case_042 expected_behavior
    Dataset-->>Bazak: "should_not_ask_ages"

    Note over Bazak: detectAgeQuestion("Great! I can help...")<br/>→ false (no age question)<br/>expected: should_not_ask_ages<br/>→ accuracy = 1 (correct!)

    Bazak-->>SDK: {results: [{example_id: "case_042",<br/>metrics: {accuracy: 1}}],<br/>aggregate_metrics: {accuracy: {mean: 1.0, std: 0, n: 1}}}

    Note over SDK: SDK sees accuracy=1 and cost=$0.00015<br/>Never saw the query or response text
```

## Health Check (used by credentials test and pre-flight)

```mermaid
sequenceDiagram
    participant Client as Traigent SDK / n8n Node
    participant Server as Bazak Server

    Client->>Server: GET /traigent/v1/health
    Server-->>Client: {status: "healthy", version: "1.0.0", uptime_seconds: 3600.5}
```
