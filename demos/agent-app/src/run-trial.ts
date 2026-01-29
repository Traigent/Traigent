/**
 * Trial Entry Point for Python-Orchestrated Optimization
 *
 * This module exports the `runTrial` function that is called by the
 * Python SDK's JS Bridge when using `runtime="node"` in the decorator.
 *
 * The Python SDK handles:
 * - Configuration sampling (random, grid, bayesian, etc.)
 * - Dataset subset selection
 * - Trial scheduling and parallelism
 * - Result aggregation and optimization
 * - Budget and early stopping
 *
 * This function handles:
 * - Running the agent with the provided configuration
 * - Computing metrics on the provided dataset subset
 * - Returning results to Python
 */

import type { TrialConfig, TrialFunctionResult } from '@traigent/sdk';
import { TrialContext, getTrialParam } from '@traigent/sdk';
import { SENTIMENT_DATASET, getDatasetSubset, type DatasetExample } from './dataset.js';
import { runSentimentAgent, type AgentConfig } from './agent.js';

/**
 * Run a single optimization trial.
 *
 * This function is called by the Python SDK via the JS Bridge.
 * It receives the trial configuration and dataset subset indices,
 * runs the agent, and returns metrics.
 *
 * @param config - Trial configuration from Python orchestrator
 * @returns Trial result with metrics
 */
export async function runTrial(config: TrialConfig): Promise<TrialFunctionResult> {
  // Extract agent configuration from trial config with proper defaults
  const model = getTrialParam<string>('model') ?? 'gpt-3.5-turbo';
  const temperature = getTrialParam<number>('temperature') ?? 0.7;
  const systemPrompt = getTrialParam<string>('system_prompt') ?? 'concise';

  const agentConfig: AgentConfig = {
    model: model as AgentConfig['model'],
    temperature,
    system_prompt: systemPrompt as AgentConfig['system_prompt'],
  };

  // Get dataset subset (indices come from Python orchestrator)
  const indices = config.dataset_subset?.indices ?? [];
  let examples: DatasetExample[];

  if (indices.length > 0) {
    examples = getDatasetSubset(indices);
  } else {
    // Fallback: use all examples if no subset specified
    examples = SENTIMENT_DATASET;
  }

  // Check for cancellation before starting
  if (TrialContext.isCancelled()) {
    return {
      metrics: { accuracy: 0, cost: 0, latency_ms: 0 },
      metadata: { cancelled: true, reason: 'cancelled_before_start' },
    };
  }

  // Run the agent - use stderr for logging (stdout is reserved for protocol)
  const logger = (msg: string) => console.error(`[agent] ${msg}`);
  const startTime = Date.now();

  const result = await runSentimentAgent(examples, agentConfig, logger);

  // Check for cancellation after completion
  const cancelled = TrialContext.isCancelled();

  return {
    metrics: {
      accuracy: result.accuracy,
      cost: result.total_cost,
      latency_ms: result.avg_latency_ms,
    },
    metadata: {
      examples_processed: examples.length,
      duration_seconds: (Date.now() - startTime) / 1000,
      model: agentConfig.model,
      temperature: agentConfig.temperature,
      system_prompt: agentConfig.system_prompt,
      cancelled,
    },
  };
}

// Export for use with npx traigent-js run
export default runTrial;
