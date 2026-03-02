# n8n Agent Evaluation Matrix

Use this matrix to select which n8n agents should be optimized first, with objective and dataset clarity.

## How to Use

1. List candidate agents from n8n workflows.
2. Fill dataset source and objective fields.
3. Score priority and implementation readiness.
4. Start with the highest-priority agent that is fully ready.

## Candidate Matrix

| Agent ID | Workflow Name | Business Use Case | Dataset Source | Dataset Size | Objective Weights (accuracy/cost/latency) | Baseline Accuracy | Baseline Cost | Baseline Latency | Readiness (Y/N) | Priority (1-5) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qa_agent_factual | Demo 2: Simple Agent — Execute & Evaluate | short factual Q&A | inline in `n8n-nodes-traigent/examples/demo-2-simple-agent.json` | 2 | 0.70 / 0.20 / 0.10 | TBD | TBD | TBD | N | 4 | tunable_id=`qa_agent`; good starter for baseline flow |
| rag_agent_grounded | Demo 4: RAG Agent — Retrieval-Augmented Q&A | grounded answers from provided documents | inline in `n8n-nodes-traigent/examples/demo-4-rag-agent.json` | 2 | 0.80 / 0.10 / 0.10 | TBD | TBD | TBD | N | 5 | tunable_id=`rag_agent`; best candidate for meaningful eval signal |
| qa_agent_ab_compare | Demo 5: Optimization Loop — Try Two Configs & Compare | config trade-off analysis (quality vs speed/cost) | inline in `n8n-nodes-traigent/examples/demo-5-optimization-loop.json` | 2 | 0.60 / 0.20 / 0.20 | TBD | TBD | TBD | N | 5 | same prompts under config A/B for direct comparison |
| qa_agent_discovery | Demo 3: Discover Tunables, Then Run | capability + config-space handshake validation | inline in `n8n-nodes-traigent/examples/demo-3-discover-and-run.json` | 1 | 0.55 / 0.20 / 0.25 | TBD | TBD | TBD | N | 3 | good preflight; not enough samples for optimization claims |
| yossi_agent_01 | Yossi production workflow (pending) | production business use case | Yossi workspace dataset (pending mapping) | TBD | TBD | TBD | TBD | TBD | N | 5 | fill once #203 access is complete |

## Initial Extraction Notes

The first four rows above were extracted from existing n8n workflow examples on 2026-02-28.
They are useful for local validation but remain `Readiness=N` for production optimization
until larger datasets and owner-approved objective weights are provided.

## Readiness Rules

An agent is `Readiness=Y` only if all are true:

1. Tunable ID is known and reachable from n8n.
2. Dataset source is available and stable.
3. Objective weights are agreed with product/owner.
4. Baseline metrics can be measured on at least 20 examples.

## Minimum Evaluation Spec per Agent

For each selected agent, define:

1. Input schema (what fields are required).
2. Expected output schema.
3. Quality metric definition (how accuracy is computed).
4. Failure policy (timeouts, partial failures, retries).
5. Success threshold to accept optimized config.

## Selection Gate for #206

Do not start #206 until:

1. At least one candidate has `Readiness=Y`.
2. Objective weights and baseline metrics are filled.
3. Dataset and tunable ownership are confirmed.
