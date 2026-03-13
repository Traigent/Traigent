/**
 * Real Mastra.ai Agent Implementation for Arkia
 *
 * This file shows how to integrate Traigent with a REAL Mastra.ai agent.
 * The mock in agent.ts simulates this for cost-free optimization testing.
 *
 * NOTE: This file contains REFERENCE CODE showing the patterns.
 * Actual Mastra integration requires installing @mastra packages.
 *
 * Prerequisites:
 *   npm install @mastra/core @mastra/memory @mastra/libsql
 *
 * Environment:
 *   OPENAI_API_KEY=sk-...
 *   GROQ_API_KEY=gsk_...  (for Groq models)
 */

// ============================================================================
// MASTRA AGENT CODE - Reference implementation
// ============================================================================

/**
 * Example: Creating a Mastra agent with configurable memory
 *
 * ```typescript
 * import { Agent } from '@mastra/core/agent';
 * import { Memory } from '@mastra/memory';
 *
 * const SYSTEM_PROMPTS = {
 *   sales_aggressive: `You are Arkia's top sales agent...`,
 *   consultative: `You are Arkia's trusted travel consultant...`,
 *   informative: `You are Arkia's knowledgeable travel advisor...`,
 * };
 *
 * export function createMastraAgent(config: AgentConfig): Agent {
 *   // Memory configuration - THE KEY COST DRIVER
 *   const memory = new Memory({
 *     options: {
 *       lastMessages: config.memory_turns * 2, // Each turn = user + assistant
 *     },
 *   });
 *
 *   return new Agent({
 *     id: 'arkia-sales-agent',
 *     instructions: SYSTEM_PROMPTS[config.system_prompt],
 *     model: config.model,  // e.g., 'groq/llama-3.3-70b-versatile'
 *     memory,
 *     modelOptions: {
 *       temperature: config.temperature,
 *     },
 *   });
 * }
 *
 * // Using the agent
 * const agent = createMastraAgent({
 *   model: 'groq/llama-3.3-70b-versatile',
 *   temperature: 0.3,
 *   system_prompt: 'consultative',
 *   memory_turns: 5,
 * });
 *
 * const response = await agent.generate([
 *   { role: 'user', content: 'Looking for flights to Barcelona' },
 * ], { resourceId: 'session-123' });
 * ```
 */

// ============================================================================
// MASTRA EVAL MODELS - Configurable judge models for quality scoring
// ============================================================================

/**
 * Available models for running Mastra evals (LLM-as-judge).
 * The judge model is ALSO an optimization dimension - cheaper judges = lower eval cost.
 */
export const EVAL_MODELS = {
  // OpenAI models for judging
  'gpt-4o': {
    id: 'openai/gpt-4o',
    quality: 'highest',
    cost_per_1m: 2.5,
    description: 'Best quality judgments, highest cost',
  },
  'gpt-4o-mini': {
    id: 'openai/gpt-4o-mini',
    quality: 'high',
    cost_per_1m: 0.15,
    description: 'Good quality, balanced cost',
  },
  // Groq models for judging - FAST and CHEAP
  'groq/llama-3.3-70b-versatile': {
    id: 'groq/llama-3.3-70b-versatile',
    quality: 'high',
    cost_per_1m: 0.59,
    description: 'High quality + fast, good for production',
  },
  'groq/llama-3.1-8b-instant': {
    id: 'groq/llama-3.1-8b-instant',
    quality: 'moderate',
    cost_per_1m: 0.05,
    description: 'Ultra cheap + fast, good for development',
  },
} as const;

export type EvalModelName = keyof typeof EVAL_MODELS;

/**
 * Mastra built-in scorers configuration.
 * These are the quality metrics Arkia cares about.
 */
export const MASTRA_SCORERS = {
  // Accuracy and Reliability
  'answer-relevancy': {
    description: 'How well the response addresses the input query',
    scale: '0-1 (higher is better)',
    use_case: 'Essential for sales - is the agent answering the question?',
  },
  completeness: {
    description: 'Does the response include all necessary information?',
    scale: '0-1 (higher is better)',
    use_case: 'Important for travel sales - includes all flight details?',
  },
  faithfulness: {
    description: 'Is the response grounded in provided context?',
    scale: '0-1 (higher is better)',
    use_case: 'Prevents hallucinated flight times or prices',
  },
  hallucination: {
    description: 'Detects factual contradictions and unsupported claims',
    scale: '0-1 (lower is better)',
    use_case: 'Critical for travel - no fake deals or availability',
  },
  // Output Quality
  'tone-consistency': {
    description: 'Measures formality and style consistency',
    scale: '0-1 (higher is better)',
    use_case: 'Maintains brand voice across conversations',
  },
  toxicity: {
    description: 'Detects harmful or inappropriate content',
    scale: '0-1 (lower is better)',
    use_case: 'Customer safety and brand protection',
  },
  // Context Quality (for RAG)
  'context-relevance': {
    description: 'Evaluates relevance of retrieved context',
    scale: '0-1 (higher is better)',
    use_case: 'If using flight database retrieval',
  },
  // Tool Usage
  'tool-call-accuracy': {
    description: 'Did the agent call the right tool?',
    scale: '0-1 (higher is better)',
    use_case: 'Essential when using booking/search tools',
  },
} as const;

export type MastraScorerName = keyof typeof MASTRA_SCORERS;

/**
 * Configuration for running evals in production.
 * This is what Arkia would configure in their Traigent optimization.
 */
export interface EvalConfig {
  // Which scorers to run (affects eval cost)
  scorers: MastraScorerName[];
  // Judge model (affects eval quality and cost)
  judge_model: EvalModelName;
  // Run evals on every conversation or sample?
  sample_rate: number; // 0.0 - 1.0
}

/** Default eval config for development */
export const DEFAULT_EVAL_CONFIG: EvalConfig = {
  scorers: ['answer-relevancy', 'completeness', 'tone-consistency'],
  judge_model: 'groq/llama-3.1-8b-instant', // Cheap for dev
  sample_rate: 1.0,
};

/** Production eval config */
export const PRODUCTION_EVAL_CONFIG: EvalConfig = {
  scorers: [
    'answer-relevancy',
    'completeness',
    'tone-consistency',
    'hallucination',
    'tool-call-accuracy',
  ],
  judge_model: 'groq/llama-3.3-70b-versatile', // Higher quality
  sample_rate: 0.1, // Sample 10% for cost efficiency
};

/**
 * Example: Running Mastra evals
 *
 * ```typescript
 * import { evaluate } from '@mastra/evals';
 *
 * // Evaluate a response using Mastra's built-in evals
 * const results = await evaluate({
 *   model: 'groq/llama-3.1-8b-instant',  // Cheap judge model
 *   scorers: ['answer-relevancy', 'completeness', 'tone-consistency'],
 *   input: customerQuery,
 *   output: agentResponse,
 *   context: flightData,  // Optional context for faithfulness
 * });
 *
 * console.log(results);
 * // {
 * //   'answer-relevancy': 0.92,
 * //   'completeness': 0.85,
 * //   'tone-consistency': 0.88,
 * //   cost: 0.00003,  // Eval cost in USD
 * // }
 * ```
 */

/**
 * Mock implementation of Mastra evals for testing.
 * In production, this would use the actual @mastra/evals package.
 */
export async function evaluateWithMastraEvals(
  _input: string,
  _output: string,
  _context?: string,
  judgeModel: EvalModelName = 'groq/llama-3.1-8b-instant'
): Promise<{
  relevancy: number;
  completeness: number;
  tone_consistency: number;
  hallucination: number;
  judge_model: string;
  eval_cost: number;
}> {
  const model = EVAL_MODELS[judgeModel];

  // Mock implementation - simulates quality based on model capability
  const qualityFactor = model.quality === 'highest' ? 1.0 : model.quality === 'high' ? 0.95 : 0.85;

  // Estimate eval cost (~500 tokens per eval)
  const evalCost = (500 * model.cost_per_1m) / 1_000_000;

  return {
    relevancy: 0.85 * qualityFactor,
    completeness: 0.82 * qualityFactor,
    tone_consistency: 0.88 * qualityFactor,
    hallucination: 0.05 / qualityFactor, // Lower is better
    judge_model: model.id,
    eval_cost: evalCost,
  };
}
