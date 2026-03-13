/**
 * Trial function for Traigent CLI Runner - Arkia Sales Agent
 *
 * This module exports a runTrial function that can be invoked by the Python
 * orchestrator via the traigent-js CLI runner. It enables Python-orchestrated
 * optimization of the Arkia travel sales agent.
 *
 * Key Optimization Dimensions:
 * - Model selection (cost vs quality)
 * - Memory turns (hidden cost driver!)
 * - System prompt style (conversion rate)
 * - Temperature (consistency vs creativity)
 * - Tool set (capability vs token cost)
 *
 * Usage from Python:
 *   @traigent.optimize(
 *       execution={
 *           "runtime": "node",
 *           "js_module": "./dist/trial.js",
 *           "js_function": "runTrial",
 *       },
 *       configuration_space={
 *           "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
 *           "temperature": [0.0, 0.3, 0.5, 0.7],
 *           "system_prompt": ["sales_aggressive", "consultative", "informative"],
 *           "memory_turns": [2, 5, 10, 15],
 *           "tool_set": ["minimal", "standard", "enhanced", "full"],
 *       },
 *       objectives=["margin_efficiency", "avg_conversion_score"],
 *   )
 */

import {
  optimize,
  param,
  type NativeOptimizeOptions,
  type OptimizationResult,
  type OptimizationSpec,
  type TrialConfig,
} from '@traigent/sdk';
import { runSalesAgent, CONFIGURATION_SPACE, type AgentConfig } from './agent.js';
import { SALES_DATASET, getDatasetSubset, getDatasetStats } from './dataset.js';

/**
 * Result type expected by the CLI runner.
 */
interface TrialResult {
  metrics: Record<string, number | null>;
  duration?: number;
  metadata?: Record<string, unknown>;
  error?: string;
}

const DEFAULT_DEMO_DATASET_SIZE = 10;
const DEFAULT_MAX_COST_USD = 5;
const DEFAULT_MAX_TRIALS = 12;

function hashStringToSeed(value: string): number {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash === 0 ? 1 : hash;
}

export function deriveDeterministicSeed(config: TrialConfig['config']): number {
  return hashStringToSeed(
    JSON.stringify({
      model: config.model,
      temperature: config.temperature,
      system_prompt: config.system_prompt,
      memory_turns: config.memory_turns,
      tool_set: config.tool_set,
    })
  );
}

export function resolveDemoDatasetSize(): number {
  const requested = Number(process.env.ARKIA_DEMO_DATASET_SIZE ?? DEFAULT_DEMO_DATASET_SIZE);
  if (!Number.isFinite(requested) || requested <= 0) {
    return Math.min(DEFAULT_DEMO_DATASET_SIZE, SALES_DATASET.length);
  }
  return Math.max(1, Math.min(SALES_DATASET.length, Math.floor(requested)));
}

function resolveMaxCostUsd(): number {
  const requested = Number(process.env.ARKIA_MAX_COST_USD ?? DEFAULT_MAX_COST_USD);
  if (!Number.isFinite(requested) || requested <= 0) {
    return DEFAULT_MAX_COST_USD;
  }
  return requested;
}

function resolveMaxTrials(): number {
  const requested = Number(process.env.ARKIA_MAX_TRIALS ?? DEFAULT_MAX_TRIALS);
  if (!Number.isInteger(requested) || requested <= 0) {
    return DEFAULT_MAX_TRIALS;
  }
  return requested;
}

/**
 * Validates that a config value is in the allowed set.
 * @throws Error if validation fails
 */
function validateConfigValue<T>(name: string, value: T, allowed: readonly T[]): void {
  if (!allowed.includes(value)) {
    throw new Error(`Invalid ${name}: "${value}". ` + `Allowed values: ${allowed.join(', ')}`);
  }
}

/**
 * Validates the agent configuration against CONFIGURATION_SPACE.
 * @throws Error if any config value is invalid
 */
function validateAgentConfig(config: AgentConfig): void {
  validateConfigValue('model', config.model, CONFIGURATION_SPACE.model);
  validateConfigValue('system_prompt', config.system_prompt, CONFIGURATION_SPACE.system_prompt);
  validateConfigValue('tool_set', config.tool_set, CONFIGURATION_SPACE.tool_set);

  // Validate numeric ranges
  if (typeof config.temperature !== 'number' || config.temperature < 0 || config.temperature > 1) {
    throw new Error(
      `Invalid temperature: ${config.temperature}. Must be a number between 0 and 1.`
    );
  }

  if (
    typeof config.memory_turns !== 'number' ||
    config.memory_turns < 1 ||
    config.memory_turns > 50
  ) {
    throw new Error(
      `Invalid memory_turns: ${config.memory_turns}. Must be a number between 1 and 50.`
    );
  }
}

/**
 * Trial function invoked by Python orchestrator.
 *
 * Receives trial config from Python, runs the sales agent,
 * and returns metrics for the optimizer.
 *
 * @param trialConfig - Configuration from Traigent orchestrator containing:
 *   - config: Agent configuration parameters (model, temperature, etc.)
 *   - dataset_subset: { indices: number[], total: number } for sampling
 *   - trial_number: Sequential trial number
 *   - trial_id: Unique trial identifier
 *
 * @returns TrialResult with metrics for optimization:
 *   - metrics.margin_efficiency: Primary optimization target (higher = better)
 *   - metrics.conversion_score: Business metric (0-1)
 *   - metrics.cost: Total simulated LLM cost in USD
 *   - duration: Trial execution time in milliseconds
 *   - metadata: Additional context for debugging
 *
 * @throws Error if configuration is invalid (caught and returned as error field)
 */
export async function runTrial(trialConfig: TrialConfig): Promise<TrialResult> {
  const startTime = Date.now();

  // Guard: Validate required trialConfig fields exist
  if (!trialConfig.config) {
    console.error(`[TRIAL ${trialConfig.trial_number}] ERROR: Missing required field 'config'`);
    return {
      metrics: { margin_efficiency: null, conversion_score: null, cost: null },
      duration: Date.now() - startTime,
      metadata: {},
      error: "Missing required field 'config' in trialConfig",
    };
  }

  if (!trialConfig.dataset_subset) {
    console.error(
      `[TRIAL ${trialConfig.trial_number}] ERROR: Missing required field 'dataset_subset'`
    );
    return {
      metrics: { margin_efficiency: null, conversion_score: null, cost: null },
      duration: Date.now() - startTime,
      metadata: {},
      error: "Missing required field 'dataset_subset' in trialConfig",
    };
  }

  // Extract agent config from trial parameters with defaults
  const config = trialConfig.config;
  const agentConfig: AgentConfig = {
    model: (config.model as AgentConfig['model']) ?? 'gpt-4o-mini',
    temperature: (config.temperature as number) ?? 0.5,
    system_prompt: (config.system_prompt as AgentConfig['system_prompt']) ?? 'consultative',
    memory_turns: (config.memory_turns as number) ?? 5,
    tool_set: (config.tool_set as AgentConfig['tool_set']) ?? 'standard',
    random_seed:
      typeof config.random_seed === 'number' ? config.random_seed : deriveDeterministicSeed(config),
  };

  // Validate configuration before running
  try {
    validateAgentConfig(agentConfig);
  } catch (validationError) {
    const error =
      validationError instanceof Error ? validationError.message : String(validationError);
    console.error(`[TRIAL ${trialConfig.trial_number}] VALIDATION ERROR: ${error}`);
    return {
      metrics: {
        margin_efficiency: null,
        conversion_score: null,
        cost: null,
      },
      duration: Date.now() - startTime,
      metadata: { config: agentConfig },
      error: `Configuration validation failed: ${error}`,
    };
  }

  // Get dataset subset from orchestrator's sampling
  const datasetSubset = trialConfig.dataset_subset;
  const examples = getDatasetSubset(datasetSubset.indices);

  if (examples.length === 0) {
    console.error(
      `[TRIAL ${trialConfig.trial_number}] ERROR: No examples found for indices: ${datasetSubset.indices}`
    );
    return {
      metrics: {
        margin_efficiency: null,
        conversion_score: null,
        cost: null,
      },
      duration: Date.now() - startTime,
      metadata: { indices: datasetSubset.indices },
      error: 'No examples found in dataset for given indices',
    };
  }

  // Log trial info (redirected to stderr by CLI runner)
  console.error(`\n${'#'.repeat(70)}`);
  console.error(`# TRIAL ${trialConfig.trial_number} (${trialConfig.trial_id})`);
  console.error(`${'#'.repeat(70)}`);
  console.error(`[CONFIG] Model: ${agentConfig.model}`);
  console.error(`[CONFIG] Temperature: ${agentConfig.temperature}`);
  console.error(`[CONFIG] Prompt Style: ${agentConfig.system_prompt}`);
  console.error(`[CONFIG] Memory Turns: ${agentConfig.memory_turns}`);
  console.error(`[CONFIG] Tool Set: ${agentConfig.tool_set}`);
  console.error(`[DATASET] Processing ${examples.length} of ${datasetSubset.total} examples`);

  // Run the sales agent with error handling
  try {
    const result = await runSalesAgent(examples, agentConfig, (msg) => console.error(msg));

    const duration = Date.now() - startTime;

    // Log summary
    console.error(`\n[TRIAL COMPLETE] Duration: ${duration}ms`);
    console.error(`[METRICS] Conversion: ${(result.avg_conversion_score * 100).toFixed(1)}%`);
    console.error(`[METRICS] Margin Efficiency: ${result.margin_efficiency.toFixed(2)}`);
    console.error(`[METRICS] Total Cost: $${result.total_cost.toFixed(6)}`);

    // Return metrics for the optimizer
    // Traigent will use these to find the optimal configuration
    return {
      metrics: {
        // Standard Traigent metrics (mapped for UI display)
        accuracy: (result.avg_relevancy + result.avg_completeness) / 2, // Composite quality score
        response_time: result.avg_latency_ms, // Maps to Response Time in UI

        // Mastra-compatible quality metrics
        relevancy: result.avg_relevancy,
        completeness: result.avg_completeness,
        tone_consistency: result.avg_tone_consistency,

        // Business metrics
        conversion_score: result.avg_conversion_score,

        // Cost metrics (for margin optimization)
        cost: result.total_cost,
        cost_per_conversation: result.avg_cost_per_conversation,
        input_tokens: result.total_input_tokens,
        output_tokens: result.total_output_tokens,
        latency_ms: result.avg_latency_ms,

        // Composite metric: THE KEY OPTIMIZATION TARGET
        // Higher margin_efficiency = better ROI
        margin_efficiency: result.margin_efficiency,
      },
      duration,
      metadata: {
        examples_processed: examples.length,
        model: agentConfig.model,
        memory_turns: agentConfig.memory_turns,
        prompt_style: agentConfig.system_prompt,
        tool_set: agentConfig.tool_set,
      },
    };
  } catch (runError) {
    const error = runError instanceof Error ? runError.message : String(runError);
    const duration = Date.now() - startTime;

    console.error(`\n[TRIAL ${trialConfig.trial_number}] EXECUTION ERROR: ${error}`);
    console.error(`[CONFIG] ${JSON.stringify(agentConfig)}`);

    return {
      metrics: {
        accuracy: null,
        response_time: null,
        margin_efficiency: null,
        conversion_score: null,
        cost: null,
      },
      duration,
      metadata: {
        config: agentConfig,
        examples_attempted: examples.length,
      },
      error: `Trial execution failed: ${error}`,
    };
  }
}

function assertNumericMetric(metrics: TrialResult['metrics'], name: string): void {
  if (typeof metrics[name] !== 'number' || !Number.isFinite(metrics[name])) {
    throw new Error(`Arkia optimization trial requires numeric metric "${name}".`);
  }
}

async function runNativeArkiaTrial(trialConfig: TrialConfig) {
  const result = await runTrial(trialConfig);
  if (result.error) {
    throw new Error(result.error);
  }

  assertNumericMetric(result.metrics, 'margin_efficiency');
  assertNumericMetric(result.metrics, 'conversion_score');
  assertNumericMetric(result.metrics, 'cost');

  return {
    metrics: result.metrics,
    metadata: result.metadata,
    duration: typeof result.duration === 'number' ? result.duration / 1000 : undefined,
  };
}

export const arkiaOptimizationSpec: OptimizationSpec = {
  configurationSpace: {
    model: param.enum(CONFIGURATION_SPACE.model),
    temperature: param.enum(CONFIGURATION_SPACE.temperature),
    system_prompt: param.enum(CONFIGURATION_SPACE.system_prompt),
    memory_turns: param.enum(CONFIGURATION_SPACE.memory_turns),
    tool_set: param.enum(CONFIGURATION_SPACE.tool_set),
  },
  objectives: [
    { metric: 'margin_efficiency', direction: 'maximize', weight: 2 },
    { metric: 'conversion_score', direction: 'maximize', weight: 1 },
    { metric: 'cost', direction: 'minimize', weight: 1 },
  ] as const,
  budget: {
    maxCostUsd: resolveMaxCostUsd(),
  },
  evaluation: {
    loadData: async () => SALES_DATASET.slice(0, resolveDemoDatasetSize()),
  },
};

export const optimizeArkiaSalesAgent = optimize(arkiaOptimizationSpec)(runNativeArkiaTrial);

export async function runArkiaOptimization(
  options: Partial<NativeOptimizeOptions> = {}
): Promise<OptimizationResult> {
  return optimizeArkiaSalesAgent.optimize({
    algorithm: 'random',
    maxTrials: resolveMaxTrials(),
    randomSeed: 7,
    ...options,
  });
}

// Default export for CLI runner compatibility
export default runTrial;

// Also export dataset stats for debugging
export { getDatasetStats, runNativeArkiaTrial };
