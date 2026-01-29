/**
 * Real LLM Integration using Mastra.ai
 *
 * Uses Mastra Agent for real LLM calls and LLM-as-judge evaluation.
 *
 * Usage:
 *   REAL_MODE=true npm run dev
 *
 * Required env vars:
 *   OPENAI_API_KEY - For OpenAI models
 *   GROQ_API_KEY   - For Groq models
 */

import { Agent } from '@mastra/core/agent';
import type { ConversationExample } from './dataset.js';
import type { AgentConfig } from './agent.js';

// Check if running in real mode
export const REAL_MODE = process.env.REAL_MODE === 'true';

const SYSTEM_PROMPTS: Record<string, string> = {
  sales_aggressive: `You are Arkia's top sales agent. CLOSE DEALS.
- Push for immediate booking
- Emphasize urgency and limited availability
- Highlight exclusive deals
- Ask for the sale directly
- Keep responses to 2-3 sentences`,

  consultative: `You are Arkia's trusted travel consultant.
- Listen to customer needs
- Suggest matching options
- Explain value, not just price
- Keep responses to 2-3 sentences`,

  informative: `You are Arkia's travel advisor.
- Give detailed flight information
- Compare options objectively
- Answer questions thoroughly
- Keep responses to 2-3 sentences`,
};

const MODEL_COSTS: Record<string, [number, number]> = {
  'gpt-3.5-turbo': [0.5, 1.5],
  'gpt-4o-mini': [0.15, 0.6],
  'gpt-4o': [2.5, 10],
  'groq/llama-3.3-70b-versatile': [0.59, 0.79],
  'groq/llama-3.1-8b-instant': [0.05, 0.08],
};

// Map our model names to Mastra model format
function getMastraModel(model: string): string {
  if (model.startsWith('groq/')) {
    return model; // groq/llama-3.3-70b-versatile -> groq/llama-3.3-70b-versatile
  }
  return `openai/${model}`; // gpt-4o -> openai/gpt-4o
}

// Cache for Mastra agents
const agentCache = new Map<string, Agent>();

function getAgent(config: AgentConfig): Agent {
  const cacheKey = `${config.model}-${config.system_prompt}`;

  if (!agentCache.has(cacheKey)) {
    const agent = new Agent({
      id: `arkia-${config.model}-${config.system_prompt}`,
      name: 'arkia-sales-agent',
      instructions: SYSTEM_PROMPTS[config.system_prompt],
      model: getMastraModel(config.model),
    });
    agentCache.set(cacheKey, agent);
  }

  return agentCache.get(cacheKey)!;
}

export async function realLLMCall(
  conversation: ConversationExample,
  config: AgentConfig
): Promise<{
  response: string;
  metrics: { latency_ms: number; input_tokens: number; output_tokens: number; cost: number };
}> {
  const agent = getAgent(config);

  // Build messages as strings for Mastra
  const messages: string[] = [];

  // Add conversation history (limited by memory_turns)
  const history = conversation.messages.slice(-config.memory_turns * 2);
  for (const msg of history) {
    const role = msg.role === 'customer' ? 'Customer' : 'Agent';
    messages.push(`${role}: ${msg.content}`);
  }

  // Add current query if not already included
  const lastMessage = conversation.messages[conversation.messages.length - 1];
  if (!lastMessage || lastMessage.content !== conversation.customer_query) {
    messages.push(`Customer: ${conversation.customer_query}`);
  }

  const prompt = messages.join('\n');
  const startTime = Date.now();

  try {
    const result = await agent.generate(prompt);
    const latency = Date.now() - startTime;

    const response = result.text ?? '';
    const inputTokens = result.usage?.inputTokens ?? 0;
    const outputTokens = result.usage?.outputTokens ?? 0;

    const costs = MODEL_COSTS[config.model] ?? [1, 3];
    const cost = (inputTokens * costs[0] + outputTokens * costs[1]) / 1_000_000;

    return {
      response,
      metrics: { latency_ms: latency, input_tokens: inputTokens, output_tokens: outputTokens, cost },
    };
  } catch (error) {
    console.error(`[MASTRA ERROR] ${error instanceof Error ? error.message : error}`);
    return {
      response: `[Error]`,
      metrics: { latency_ms: Date.now() - startTime, input_tokens: 0, output_tokens: 0, cost: 0 },
    };
  }
}

// Eval agent for LLM-as-judge
let evalAgent: Agent | null = null;

function getEvalAgent(judgeModel: string): Agent {
  if (!evalAgent) {
    evalAgent = new Agent({
      id: 'eval-judge',
      name: 'eval-judge',
      instructions: `You are an evaluation judge. Score agent responses on a scale of 0.0 to 1.0.
Always respond with ONLY a JSON object, no explanation.`,
      model: getMastraModel(judgeModel),
    });
  }
  return evalAgent;
}

export async function evaluateResponse(
  query: string,
  response: string,
  judgeModel: string = 'groq/llama-3.1-8b-instant'
): Promise<{
  relevancy: number;
  completeness: number;
  tone_consistency: number;
  conversion_score: number;
  eval_cost: number;
}> {
  const agent = getEvalAgent(judgeModel);

  const evalPrompt = `Score this travel agent response (0.0-1.0 for each). Reply with ONLY JSON.

Customer: "${query}"
Agent: "${response}"

{"relevancy": 0.0, "completeness": 0.0, "tone_consistency": 0.0, "conversion_score": 0.0}`;

  try {
    const result = await agent.generate(evalPrompt);
    const content = result.text ?? '{}';

    const jsonMatch = content.match(/\{[\s\S]*\}/);
    const scores = jsonMatch ? JSON.parse(jsonMatch[0]) : {};

    const costs = MODEL_COSTS[judgeModel] ?? [0.05, 0.08];
    const inputTokens = result.usage?.inputTokens ?? 0;
    const outputTokens = result.usage?.outputTokens ?? 0;
    const evalCost = (inputTokens * costs[0] + outputTokens * costs[1]) / 1_000_000;

    return {
      relevancy: Math.max(0, Math.min(1, scores.relevancy ?? 0.7)),
      completeness: Math.max(0, Math.min(1, scores.completeness ?? 0.7)),
      tone_consistency: Math.max(0, Math.min(1, scores.tone_consistency ?? 0.7)),
      conversion_score: Math.max(0, Math.min(1, scores.conversion_score ?? 0.5)),
      eval_cost: evalCost,
    };
  } catch (error) {
    console.error(`[MASTRA EVAL ERROR] ${error instanceof Error ? error.message : error}`);
    return { relevancy: 0.7, completeness: 0.7, tone_consistency: 0.7, conversion_score: 0.5, eval_cost: 0 };
  }
}
