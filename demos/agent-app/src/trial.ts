/**
 * Trial function for Traigent CLI Runner
 *
 * This module exports a runTrial function that can be invoked by the Python
 * orchestrator via the traigent-js CLI runner. It enables Python-orchestrated
 * optimization of the JS sentiment agent.
 *
 * Usage from Python:
 *   @traigent.optimize(
 *       execution={
 *           "runtime": "node",
 *           "js_module": "./demos/agent-app/dist/trial.js",
 *           "js_function": "runTrial",
 *           "js_parallel_workers": 4,  # Enable parallel execution
 *       }
 *   )
 */

import { type TrialConfig } from '@traigent/sdk';
import { runSentimentAgent, type AgentConfig } from './agent.js';
import { getDatasetSubset } from './dataset.js';

/**
 * Result type expected by the CLI runner.
 * The runner handles wrapping this into the full protocol response.
 */
interface TrialResult {
  metrics: Record<string, number | null>;
  duration?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Trial function invoked by Python orchestrator.
 *
 * Receives trial config from Python, runs the sentiment agent,
 * and returns metrics for the optimizer.
 */
export async function runTrial(trialConfig: TrialConfig): Promise<TrialResult> {

  // Extract agent config from trial parameters (with type assertions)
  const config = trialConfig.config;
  const agentConfig: AgentConfig = {
    model: (config.model as AgentConfig['model']) ?? 'gpt-3.5-turbo',
    temperature: (config.temperature as number) ?? 0.7,
    system_prompt: (config.system_prompt as AgentConfig['system_prompt']) ?? 'concise',
  };

  // Get dataset subset from orchestrator's sampling
  const datasetSubset = trialConfig.dataset_subset;
  const examples = getDatasetSubset(datasetSubset.indices);

  // Log trial info (redirected to stderr by CLI runner)
  console.error(
    `[Trial ${trialConfig.trial_number}] Running with config:`,
    JSON.stringify(agentConfig)
  );
  console.error(
    `[Trial ${trialConfig.trial_number}] Processing ${examples.length} examples`
  );

  // Run the agent - use stderr for logging to avoid NDJSON corruption
  const result = await runSentimentAgent(examples, agentConfig, (msg) =>
    console.error(msg)
  );

  // Return metrics for the optimizer
  return {
    metrics: {
      accuracy: result.accuracy,
      cost: result.total_cost,
      latency_ms: result.avg_latency_ms,
    },
    // Optionally include metadata
    metadata: {
      examples_processed: examples.length,
      correct_predictions: Math.round(result.accuracy * examples.length),
    },
  };
}

// Default export for CLI runner compatibility
export default runTrial;
