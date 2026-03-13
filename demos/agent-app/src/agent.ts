/**
 * Mock Sentiment Classification Agent
 *
 * This agent demonstrates how Traigent optimizes LLM configurations.
 * Instead of calling a real LLM, it simulates responses based on config.
 */

import type { DatasetExample } from './dataset.js';

/** Configuration space for the agent */
export interface AgentConfig {
  model: 'gpt-3.5-turbo' | 'gpt-4o-mini' | 'gpt-4o';
  temperature: number;
  system_prompt: 'concise' | 'detailed' | 'cot';
}

/** Cost per 1M tokens [input, output] */
const MODEL_COSTS: Record<string, [number, number]> = {
  'gpt-3.5-turbo': [0.5, 1.5],
  'gpt-4o-mini': [0.15, 0.6],
  'gpt-4o': [2.5, 10.0],
};

/** Simulated accuracy based on model and prompt type */
const MODEL_ACCURACY: Record<string, Record<string, number>> = {
  'gpt-3.5-turbo': { concise: 0.7, detailed: 0.75, cot: 0.78 },
  'gpt-4o-mini': { concise: 0.8, detailed: 0.85, cot: 0.88 },
  'gpt-4o': { concise: 0.88, detailed: 0.92, cot: 0.95 },
};

/** Simulated latency in ms */
const MODEL_LATENCY: Record<string, number> = {
  'gpt-3.5-turbo': 200,
  'gpt-4o-mini': 300,
  'gpt-4o': 500,
};

/**
 * Mock LLM call that simulates sentiment classification.
 * Returns predicted sentiment based on model accuracy.
 */
function mockLLMCall(
  text: string,
  expectedOutput: string,
  config: AgentConfig
): { prediction: string; correct: boolean; latency: number; cost: number } {
  const baseAccuracy = MODEL_ACCURACY[config.model]?.[config.system_prompt] ?? 0.7;

  // Temperature affects consistency - lower is more deterministic
  const tempFactor = 1 - config.temperature * 0.1;
  const accuracy = baseAccuracy * tempFactor;

  // Add some randomness for realistic simulation
  const random = Math.random();
  const correct = random < accuracy;

  // Simulate prediction
  const prediction = correct ? expectedOutput : getWrongPrediction(expectedOutput);

  // Calculate mock cost (assume ~100 input tokens, ~5 output tokens)
  const costs = MODEL_COSTS[config.model] ?? [1, 3];
  const cost = (100 * costs[0] + 5 * costs[1]) / 1_000_000;

  // Get latency with some variance
  const baseLatency = MODEL_LATENCY[config.model] ?? 300;
  const latency = baseLatency * (0.8 + Math.random() * 0.4);

  return { prediction, correct, latency, cost };
}

function getWrongPrediction(correct: string): string {
  const options = ['positive', 'negative', 'neutral'];
  const wrong = options.filter((o) => o !== correct);
  return wrong[Math.floor(Math.random() * wrong.length)];
}

/**
 * Sentiment Classification Agent
 *
 * Processes examples and returns accuracy metrics.
 */
export async function runSentimentAgent(
  examples: DatasetExample[],
  config: AgentConfig,
  logger: (msg: string) => void = console.log
): Promise<{
  accuracy: number;
  total_cost: number;
  avg_latency_ms: number;
  predictions: string[];
}> {
  logger(`\n${'='.repeat(60)}`);
  logger(`AGENT CONFIGURATION:`);
  logger(`  Model:       ${config.model}`);
  logger(`  Temperature: ${config.temperature}`);
  logger(`  Prompt Type: ${config.system_prompt}`);
  logger(`${'='.repeat(60)}`);

  let correctCount = 0;
  let totalCost = 0;
  let totalLatency = 0;
  const predictions: string[] = [];

  for (let i = 0; i < examples.length; i++) {
    const example = examples[i];
    const result = mockLLMCall(example.input.text, example.output, config);

    predictions.push(result.prediction);
    if (result.correct) correctCount++;
    totalCost += result.cost;
    totalLatency += result.latency;

    // Log each prediction
    const status = result.correct ? '[OK]' : '[X] ';
    logger(
      `  ${status} "${example.input.text.substring(0, 40)}..." => ${result.prediction} (expected: ${example.output})`
    );
  }

  const accuracy = correctCount / examples.length;
  const avgLatency = totalLatency / examples.length;

  logger(`\nRESULTS:`);
  logger(`  Accuracy:    ${(accuracy * 100).toFixed(1)}% (${correctCount}/${examples.length})`);
  logger(`  Total Cost:  $${totalCost.toFixed(6)}`);
  logger(`  Avg Latency: ${avgLatency.toFixed(0)}ms`);
  logger(`${'='.repeat(60)}\n`);

  return {
    accuracy,
    total_cost: totalCost,
    avg_latency_ms: avgLatency,
    predictions,
  };
}

/** Default configuration */
export const DEFAULT_CONFIG: AgentConfig = {
  model: 'gpt-3.5-turbo',
  temperature: 0.7,
  system_prompt: 'concise',
};

/** Configuration space for optimization */
export const CONFIGURATION_SPACE = {
  model: ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o'] as const,
  temperature: [0.0, 0.3, 0.5, 0.7, 1.0] as const,
  system_prompt: ['concise', 'detailed', 'cot'] as const,
};
