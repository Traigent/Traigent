/**
 * Arkia Travel Sales Agent - MOCK IMPLEMENTATION
 *
 * This is a MOCK implementation that simulates LLM behavior for cost-free
 * optimization testing with Traigent. It models realistic quality/cost
 * tradeoffs without making actual API calls.
 *
 * For REAL Mastra.ai integration, see: ./mastra-agent.ts
 *
 * Why use a mock?
 * - Test optimization strategies without LLM costs
 * - Iterate quickly on configuration space design
 * - Validate Traigent integration before production
 *
 * The mock simulates:
 * - Quality differences between models (GPT-4o vs GPT-4o-mini vs Groq)
 * - Cost calculation based on real pricing
 * - Memory turn impact on token usage
 * - Latency differences between providers
 *
 * Optimization Goals:
 * - Maximize sales conversion (quality)
 * - Minimize token costs (margins)
 * - Maintain professional tone
 *
 * Key Insight: Memory configuration is a hidden cost driver.
 * More context = better responses but higher token costs.
 *
 * Reproducibility:
 * - Set random_seed in AgentConfig for deterministic results
 * - Without seed, results vary ±5% due to simulated LLM variance
 */

import type { ConversationExample } from './dataset.js';
import { TOOL_SET_TOKEN_COSTS, TOOL_SET_CONVERSION_BOOST, type ToolSetName } from './tools.js';
import { REAL_MODE, realLLMCall, evaluateResponse } from './real-llm.js';

// ============================================================================
// NAMED CONSTANTS
// ============================================================================

/**
 * Token estimation multiplier.
 * Words -> tokens conversion factor (~1.3 tokens per word on average).
 * Based on typical English text tokenization with GPT tokenizers.
 */
const TOKEN_MULTIPLIER = 1.3;

/**
 * Average tokens per conversation turn.
 * Typical customer query + assistant response averages ~50 tokens.
 * This affects memory cost calculation.
 */
const TOKENS_PER_TURN = 50;

/**
 * Margin efficiency scaling factor.
 * Scales the raw conversion/cost ratio to readable values (0.01 to 10 range).
 * Without scaling, values would be in the thousands.
 */
const MARGIN_SCALE = 10000;

/**
 * Temperature impact on consistency.
 * At temperature=1.0, quality is reduced by this factor.
 * Based on observation that higher temp = more variance = lower avg quality.
 */
const TEMP_CONSISTENCY_FACTOR = 0.08;

/**
 * Memory quality boost per turn (capped).
 * Each additional memory turn improves quality by 2%, up to 10% max.
 * Reflects that more context helps but has diminishing returns.
 */
const MEMORY_QUALITY_BOOST_PER_TURN = 0.02;
const MEMORY_QUALITY_BOOST_MAX = 0.1;

/**
 * Quality score variance range (±).
 * Simulates natural LLM output variation between identical prompts.
 * Value of 0.1 means ±5% variance (0.5 * 0.1 = 0.05).
 */
const QUALITY_VARIANCE = 0.1;

/**
 * Output token range [min, variance].
 * Mock responses generate 40-100 tokens.
 */
const OUTPUT_TOKENS_MIN = 40;
const OUTPUT_TOKENS_VARIANCE = 60;

/**
 * Latency variance range.
 * Actual latency varies 0.8x to 1.2x of base latency.
 */
const LATENCY_VARIANCE_MIN = 0.8;
const LATENCY_VARIANCE_RANGE = 0.4;

// ============================================================================
// SEEDED RANDOM NUMBER GENERATOR
// ============================================================================

/**
 * Simple seeded random number generator (Linear Congruential Generator).
 * Provides reproducible random numbers when seed is set.
 */
class SeededRandom {
  private seed: number;

  constructor(seed?: number) {
    this.seed = seed ?? Math.random() * 2147483647;
  }

  /**
   * Returns a random number between 0 and 1 (like Math.random()).
   */
  next(): number {
    // LCG parameters (same as glibc)
    this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
    return this.seed / 0x7fffffff;
  }

  /**
   * Returns a random number in range [min, max).
   */
  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }
}

// Global RNG instance (can be seeded per-run)
let rng = new SeededRandom();

/**
 * Sets the random seed for reproducible results.
 * Call this before running trials if you want deterministic output.
 */
export function setRandomSeed(seed: number): void {
  rng = new SeededRandom(seed);
}

/**
 * Resets RNG to use Math.random() (non-deterministic).
 */
export function resetRandomSeed(): void {
  rng = new SeededRandom();
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/** Configuration space for the agent */
export interface AgentConfig {
  model:
    // OpenAI models
    | 'gpt-3.5-turbo'
    | 'gpt-4o-mini'
    | 'gpt-4o'
    // Groq models (via LiteLLM)
    | 'groq/llama-3.3-70b-versatile'
    | 'groq/llama-3.1-8b-instant';
  temperature: number;
  system_prompt: 'sales_aggressive' | 'consultative' | 'informative';
  memory_turns: number;  // How many previous turns to include (affects cost!)
  tool_set: ToolSetName; // Which tools to enable (affects cost AND capability!)
  random_seed?: number;  // Optional seed for reproducible results
}

/**
 * Cost per 1M tokens [input, output].
 * Based on published pricing (simulated for demo - actual prices may vary).
 * Source: OpenAI and Groq pricing pages (January 2025)
 */
const MODEL_COSTS: Record<string, [number, number]> = {
  // OpenAI pricing
  'gpt-3.5-turbo': [0.5, 1.5],
  'gpt-4o-mini': [0.15, 0.6],
  'gpt-4o': [2.5, 10.0],
  // Groq pricing - significantly cheaper!
  'groq/llama-3.3-70b-versatile': [0.59, 0.79],  // High quality, good price
  'groq/llama-3.1-8b-instant': [0.05, 0.08],      // Ultra cheap, fast
};

/**
 * Simulated quality scores based on model and prompt type.
 *
 * These values are calibrated based on:
 * - Typical benchmark performance differences between models
 * - Observed conversion rate differences with different prompt styles
 * - The assumption that larger models generally perform better
 *
 * Note: In production, use Mastra evals to measure actual performance.
 */
const MODEL_QUALITY: Record<string, Record<string, {
  relevancy: number;      // Mastra: answer-relevancy
  completeness: number;   // Mastra: completeness
  tone: number;           // Mastra: tone-consistency
  conversion: number;     // Business metric: sales conversion rate
}>> = {
  // OpenAI models
  'gpt-3.5-turbo': {
    sales_aggressive: { relevancy: 0.72, completeness: 0.68, tone: 0.65, conversion: 0.45 },
    consultative:     { relevancy: 0.75, completeness: 0.70, tone: 0.78, conversion: 0.52 },
    informative:      { relevancy: 0.78, completeness: 0.75, tone: 0.82, conversion: 0.48 },
  },
  'gpt-4o-mini': {
    sales_aggressive: { relevancy: 0.82, completeness: 0.78, tone: 0.75, conversion: 0.58 },
    consultative:     { relevancy: 0.85, completeness: 0.82, tone: 0.88, conversion: 0.65 },
    informative:      { relevancy: 0.88, completeness: 0.85, tone: 0.90, conversion: 0.60 },
  },
  'gpt-4o': {
    sales_aggressive: { relevancy: 0.90, completeness: 0.88, tone: 0.82, conversion: 0.68 },
    consultative:     { relevancy: 0.94, completeness: 0.92, tone: 0.95, conversion: 0.78 },
    informative:      { relevancy: 0.95, completeness: 0.94, tone: 0.96, conversion: 0.72 },
  },
  // Groq models - Llama 3.3 70B is comparable to GPT-4o quality
  'groq/llama-3.3-70b-versatile': {
    sales_aggressive: { relevancy: 0.88, completeness: 0.85, tone: 0.80, conversion: 0.65 },
    consultative:     { relevancy: 0.91, completeness: 0.89, tone: 0.92, conversion: 0.74 },
    informative:      { relevancy: 0.93, completeness: 0.91, tone: 0.94, conversion: 0.70 },
  },
  // Groq Llama 3.1 8B - fast and cheap, lower quality
  'groq/llama-3.1-8b-instant': {
    sales_aggressive: { relevancy: 0.70, completeness: 0.65, tone: 0.62, conversion: 0.42 },
    consultative:     { relevancy: 0.73, completeness: 0.68, tone: 0.75, conversion: 0.50 },
    informative:      { relevancy: 0.76, completeness: 0.72, tone: 0.78, conversion: 0.46 },
  },
};

/** Average latency by model (ms) */
const MODEL_LATENCY: Record<string, number> = {
  // OpenAI latencies
  'gpt-3.5-turbo': 180,
  'gpt-4o-mini': 250,
  'gpt-4o': 450,
  // Groq latencies - extremely fast due to LPU hardware!
  'groq/llama-3.3-70b-versatile': 80,   // ~5x faster than GPT-4o
  'groq/llama-3.1-8b-instant': 30,      // Ultra fast for simple tasks
};

/** System prompt templates */
const SYSTEM_PROMPTS: Record<string, string> = {
  sales_aggressive: `You are Arkia's top sales agent. Your goal is to CLOSE DEALS.
- Always push for immediate booking
- Emphasize urgency and limited availability
- Highlight exclusive deals and upgrades
- Ask for the sale directly`,

  consultative: `You are Arkia's trusted travel consultant. Help customers find their perfect trip.
- Listen to customer needs carefully
- Suggest options that match their preferences
- Explain value, not just price
- Build long-term relationships`,

  informative: `You are Arkia's knowledgeable travel advisor. Provide comprehensive information.
- Give detailed flight and destination information
- Compare options objectively
- Answer all questions thoroughly
- Let customers make informed decisions`,
};

// ============================================================================
// MOCK LLM IMPLEMENTATION
// ============================================================================

/**
 * Mock LLM call that simulates a travel sales conversation turn.
 * Includes memory cost calculation based on context window.
 */
function mockLLMCall(
  conversation: ConversationExample,
  config: AgentConfig
): {
  response: string;
  metrics: {
    relevancy: number;
    completeness: number;
    tone_consistency: number;
    conversion_score: number;
    latency_ms: number;
    input_tokens: number;
    output_tokens: number;
    cost: number;
  };
} {
  const baseQuality = MODEL_QUALITY[config.model]?.[config.system_prompt] ?? {
    relevancy: 0.7, completeness: 0.7, tone: 0.7, conversion: 0.5
  };

  // Temperature affects consistency - lower is more deterministic
  const tempFactor = 1 - (config.temperature * TEMP_CONSISTENCY_FACTOR);

  // Memory affects quality - more context = better understanding
  const memoryQualityBoost = Math.min(
    config.memory_turns * MEMORY_QUALITY_BOOST_PER_TURN,
    MEMORY_QUALITY_BOOST_MAX
  );

  // Tool set affects conversion - more tools = better capability
  const toolConversionBoost = TOOL_SET_CONVERSION_BOOST[config.tool_set] ?? 1.0;

  // Calculate quality scores with variance (using seeded RNG)
  // Clamp to [0, 1] to prevent invalid negative values from variance
  const variance = () => (rng.next() - 0.5) * QUALITY_VARIANCE;
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const relevancy = clamp((baseQuality.relevancy + memoryQualityBoost) * tempFactor + variance());
  const completeness = clamp((baseQuality.completeness + memoryQualityBoost) * tempFactor + variance());
  const tone = clamp(baseQuality.tone * tempFactor + variance());
  // Conversion is boosted by tool set
  const conversion = clamp((baseQuality.conversion + memoryQualityBoost * 2) * tempFactor * toolConversionBoost + variance());

  // Calculate token costs
  // System prompt tokens (fixed per config)
  const systemTokens = SYSTEM_PROMPTS[config.system_prompt].split(/\s+/).length * TOKEN_MULTIPLIER;

  // Tool descriptions tokens - ANOTHER HIDDEN COST DRIVER!
  const toolTokens = TOOL_SET_TOKEN_COSTS[config.tool_set] ?? 0;

  // Memory tokens - THIS IS THE HIDDEN COST DRIVER
  const memoryTokens = config.memory_turns * TOKENS_PER_TURN;

  // Current message tokens (guard against empty messages array)
  const lastMessage = conversation.messages[conversation.messages.length - 1];
  const currentMessageTokens = lastMessage
    ? lastMessage.content.split(/\s+/).length * TOKEN_MULTIPLIER
    : TOKENS_PER_TURN;  // Fallback to default tokens if no messages

  // Total input tokens (system + tools + memory + current message)
  const inputTokens = Math.round(systemTokens + toolTokens + memoryTokens + currentMessageTokens);

  // Output tokens (response) - using seeded RNG
  const outputTokens = Math.round(OUTPUT_TOKENS_MIN + rng.next() * OUTPUT_TOKENS_VARIANCE);

  // Calculate cost
  const costs = MODEL_COSTS[config.model] ?? [1, 3];
  const cost = (inputTokens * costs[0] + outputTokens * costs[1]) / 1_000_000;

  // Latency with variance (using seeded RNG)
  const baseLatency = MODEL_LATENCY[config.model] ?? 300;
  const latency = baseLatency * (LATENCY_VARIANCE_MIN + rng.next() * LATENCY_VARIANCE_RANGE);

  // Generate mock response
  const response = generateMockResponse(conversation, config);

  return {
    response,
    metrics: {
      relevancy,
      completeness,
      tone_consistency: tone,
      conversion_score: conversion,
      latency_ms: latency,
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      cost,
    },
  };
}

function generateMockResponse(conversation: ConversationExample, config: AgentConfig): string {
  const intent = conversation.intent;

  const responses: Record<string, Record<string, string>> = {
    flight_inquiry: {
      sales_aggressive: "Great choice! I have the PERFECT flight for you - and it's selling fast! Let me book it now before prices go up. Ready to secure your seat?",
      consultative: "I'd love to help you find the right flight. Can you tell me more about your travel dates and preferences? Are you flexible with times?",
      informative: "Here are the available flights for your route. The morning flight offers the best connection times, while the evening flight is more economical.",
    },
    price_negotiation: {
      sales_aggressive: "This is already our BEST price! But because you're a valued customer, I can throw in priority boarding. Deal?",
      consultative: "I understand budget is important. Let me show you some options that might work better for your needs without compromising on quality.",
      informative: "The price breakdown includes base fare, taxes, and fees. Here's how it compares to similar dates and flexible ticket options.",
    },
    booking_intent: {
      sales_aggressive: "Excellent decision! I'm booking this RIGHT NOW before someone else takes it. Just need your payment details!",
      consultative: "Wonderful! I'll walk you through the booking process step by step. Do you have any questions before we proceed?",
      informative: "To complete the booking, I'll need passenger details, contact information, and payment method. The process takes about 5 minutes.",
    },
    support: {
      sales_aggressive: "I'll get that sorted immediately! And while I have you, have you considered upgrading to flexible tickets for your next trip?",
      consultative: "I completely understand. Let me look into this for you right away and find the best solution.",
      informative: "Here's the information you need. For flight changes, the policy allows modifications up to 24 hours before departure.",
    },
    complaint: {
      sales_aggressive: "I sincerely apologize! Let me make this right AND offer you a discount on your next booking!",
      consultative: "I'm truly sorry to hear about your experience. Let me understand what happened and see how we can make it right.",
      informative: "I apologize for the inconvenience. Here are your options for compensation according to our passenger rights policy.",
    },
  };

  return responses[intent]?.[config.system_prompt] ?? "How can I help you with your travel plans today?";
}

// ============================================================================
// MAIN AGENT RUNNER
// ============================================================================

/**
 * Run the sales agent on a batch of conversation examples.
 *
 * @param examples - Conversation examples to process
 * @param config - Agent configuration (model, temperature, etc.)
 * @param logger - Logging function (defaults to console.log)
 * @returns Aggregated metrics for all examples
 */
export async function runSalesAgent(
  examples: ConversationExample[],
  config: AgentConfig,
  logger: (msg: string) => void = console.log
): Promise<{
  // Mastra-compatible metrics
  avg_relevancy: number;
  avg_completeness: number;
  avg_tone_consistency: number;
  // Business metrics
  avg_conversion_score: number;
  // Cost metrics (for margin optimization)
  total_cost: number;
  avg_cost_per_conversation: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_latency_ms: number;
  // Composite score: quality vs cost tradeoff
  margin_efficiency: number;  // conversion / cost (higher is better)
  // Processing stats
  examples_processed: number;
  examples_failed: number;
}> {
  // Set random seed if provided for reproducibility, otherwise reset to non-deterministic
  // This prevents cross-run contamination where a previous seeded run affects subsequent runs
  if (config.random_seed !== undefined) {
    setRandomSeed(config.random_seed);
  } else {
    resetRandomSeed();
  }

  logger(`\n${'='.repeat(70)}`);
  logger(`ARKIA SALES AGENT - CONFIGURATION:`);
  logger(`  Model:         ${config.model}`);
  logger(`  Temperature:   ${config.temperature}`);
  logger(`  Prompt Style:  ${config.system_prompt}`);
  logger(`  Memory Turns:  ${config.memory_turns} (affects token cost!)`);
  logger(`  Tool Set:      ${config.tool_set}`);
  if (config.random_seed !== undefined) {
    logger(`  Random Seed:   ${config.random_seed} (reproducible)`);
  }
  logger(`${'='.repeat(70)}`);

  let totalRelevancy = 0;
  let totalCompleteness = 0;
  let totalTone = 0;
  let totalConversion = 0;
  let totalCost = 0;
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let totalLatency = 0;
  let successCount = 0;
  let failedCount = 0;

  for (let i = 0; i < examples.length; i++) {
    const example = examples[i];

    try {
      let relevancy: number, completeness: number, tone: number, conversion: number;
      let cost: number, inputTokens: number, outputTokens: number, latency: number;

      if (REAL_MODE) {
        // Real LLM API call
        const llmResult = await realLLMCall(example, config);

        // Evaluate response quality using LLM-as-judge
        const evalResult = await evaluateResponse(
          example.customer_query,
          llmResult.response,
          'groq/llama-3.1-8b-instant'  // Cheap eval model
        );

        relevancy = evalResult.relevancy;
        completeness = evalResult.completeness;
        tone = evalResult.tone_consistency;
        conversion = evalResult.conversion_score;
        cost = llmResult.metrics.cost + evalResult.eval_cost;  // Include eval cost
        inputTokens = llmResult.metrics.input_tokens;
        outputTokens = llmResult.metrics.output_tokens;
        latency = llmResult.metrics.latency_ms;
      } else {
        // Mock LLM call (no API costs)
        const result = mockLLMCall(example, config);
        relevancy = result.metrics.relevancy;
        completeness = result.metrics.completeness;
        tone = result.metrics.tone_consistency;
        conversion = result.metrics.conversion_score;
        cost = result.metrics.cost;
        inputTokens = result.metrics.input_tokens;
        outputTokens = result.metrics.output_tokens;
        latency = result.metrics.latency_ms;
      }

      totalRelevancy += relevancy;
      totalCompleteness += completeness;
      totalTone += tone;
      totalConversion += conversion;
      totalCost += cost;
      totalInputTokens += inputTokens;
      totalOutputTokens += outputTokens;
      totalLatency += latency;
      successCount++;

      // Log each conversation result
      const conversionIcon = conversion > 0.6 ? '[SALE]' : '[    ]';
      logger(`  ${conversionIcon} "${example.customer_query.substring(0, 45)}..." | conv: ${(conversion * 100).toFixed(0)}% | cost: $${cost.toFixed(6)}`);
    } catch (error) {
      failedCount++;
      const errorMsg = error instanceof Error ? error.message : String(error);
      logger(`  [FAIL] Example ${i + 1}: ${errorMsg}`);
    }
  }

  // Handle case where all examples failed
  if (successCount === 0) {
    logger(`\n[ERROR] All ${examples.length} examples failed!`);
    return {
      avg_relevancy: 0,
      avg_completeness: 0,
      avg_tone_consistency: 0,
      avg_conversion_score: 0,
      total_cost: 0,
      avg_cost_per_conversation: 0,
      total_input_tokens: 0,
      total_output_tokens: 0,
      avg_latency_ms: 0,
      margin_efficiency: 0,
      examples_processed: 0,
      examples_failed: failedCount,
    };
  }

  const n = successCount;
  const avgRelevancy = totalRelevancy / n;
  const avgCompleteness = totalCompleteness / n;
  const avgTone = totalTone / n;
  const avgConversion = totalConversion / n;
  const avgCostPerConv = totalCost / n;
  const avgLatency = totalLatency / n;

  // Margin efficiency: how much conversion do we get per dollar spent?
  // Higher is better - want high conversion with low cost
  // Formula: conversion_rate / (cost_per_conv * MARGIN_SCALE)
  // Example: 0.7 conversion / ($0.001 * 10000) = 0.7 / 10 = 0.07
  const marginEfficiency = avgConversion / (avgCostPerConv * MARGIN_SCALE);

  logger(`\n${'─'.repeat(70)}`);
  logger(`RESULTS SUMMARY:`);
  logger(`  Quality Metrics (Mastra-compatible):`);
  logger(`    Answer Relevancy:    ${(avgRelevancy * 100).toFixed(1)}%`);
  logger(`    Completeness:        ${(avgCompleteness * 100).toFixed(1)}%`);
  logger(`    Tone Consistency:    ${(avgTone * 100).toFixed(1)}%`);
  logger(`  Business Metrics:`);
  logger(`    Conversion Score:    ${(avgConversion * 100).toFixed(1)}%`);
  logger(`    Margin Efficiency:   ${marginEfficiency.toFixed(2)} (conversion/cost)`);
  logger(`  Cost Metrics:`);
  logger(`    Total Cost:          $${totalCost.toFixed(6)}`);
  logger(`    Avg Cost/Conv:       $${avgCostPerConv.toFixed(6)}`);
  logger(`    Total Tokens:        ${totalInputTokens + totalOutputTokens} (in: ${totalInputTokens}, out: ${totalOutputTokens})`);
  logger(`    Avg Latency:         ${avgLatency.toFixed(0)}ms`);
  if (failedCount > 0) {
    logger(`  Processing:`);
    logger(`    Processed:           ${successCount}/${examples.length} examples`);
    logger(`    Failed:              ${failedCount} examples`);
  }
  logger(`${'='.repeat(70)}\n`);

  return {
    avg_relevancy: avgRelevancy,
    avg_completeness: avgCompleteness,
    avg_tone_consistency: avgTone,
    avg_conversion_score: avgConversion,
    total_cost: totalCost,
    avg_cost_per_conversation: avgCostPerConv,
    total_input_tokens: totalInputTokens,
    total_output_tokens: totalOutputTokens,
    avg_latency_ms: avgLatency,
    margin_efficiency: marginEfficiency,
    examples_processed: successCount,
    examples_failed: failedCount,
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

/** Default configuration (current SOTA defaults they mentioned) */
export const DEFAULT_CONFIG: AgentConfig = {
  model: 'gpt-4o',           // SOTA model (expensive!)
  temperature: 0.7,
  system_prompt: 'informative',
  memory_turns: 10,          // Default memory (expensive!)
  tool_set: 'full',          // All tools enabled (expensive!)
};

/** Configuration space for optimization */
export const CONFIGURATION_SPACE = {
  model: [
    // OpenAI models
    'gpt-3.5-turbo',
    'gpt-4o-mini',
    'gpt-4o',
    // Groq models - great for margin optimization!
    'groq/llama-3.3-70b-versatile',  // High quality + cheap + fast
    'groq/llama-3.1-8b-instant',      // Ultra cheap + ultra fast
  ] as const,
  temperature: [0.0, 0.3, 0.5, 0.7] as const,
  system_prompt: ['sales_aggressive', 'consultative', 'informative'] as const,
  memory_turns: [2, 5, 10, 15] as const,  // Key for margin optimization!
  tool_set: ['minimal', 'standard', 'enhanced', 'full'] as const, // Tool selection!
};
