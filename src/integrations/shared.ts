import { TrialContext } from '../core/context.js';
import { recordRuntimeMetrics } from '../core/runtime-metrics.js';
import type { FrameworkTarget } from '../optimization/types.js';

const MODEL_COSTS: Record<string, [number, number]> = {
  'gpt-4o': [2.5, 10],
  'gpt-4o-mini': [0.15, 0.6],
  'gpt-4-turbo': [10, 30],
  'gpt-4': [30, 60],
  'gpt-3.5-turbo': [0.5, 1.5],
  'claude-3-opus': [15, 75],
  'claude-3-sonnet': [3, 15],
  'claude-3-haiku': [0.25, 1.25],
  'claude-3-5-sonnet': [3, 15],
};

type FrameworkOverrideMap = Record<string, string>;

const FRAMEWORK_PARAM_MAPS: Record<FrameworkTarget, FrameworkOverrideMap> = {
  openai: {
    model: 'model',
    temperature: 'temperature',
    maxTokens: 'max_tokens',
    topP: 'top_p',
    frequencyPenalty: 'frequency_penalty',
    presencePenalty: 'presence_penalty',
  },
  langchain: {
    model: 'model',
    temperature: 'temperature',
    maxTokens: 'maxTokens',
    topP: 'topP',
    frequencyPenalty: 'frequencyPenalty',
    presencePenalty: 'presencePenalty',
  },
  'vercel-ai': {
    model: 'modelId',
    temperature: 'temperature',
    maxTokens: 'maxTokens',
    topP: 'topP',
    frequencyPenalty: 'frequencyPenalty',
    presencePenalty: 'presencePenalty',
  },
};

export function getFrameworkOverrides(
  target: FrameworkTarget,
): Record<string, unknown> {
  const config = TrialContext.getConfigOrUndefined()?.config;
  if (!config) {
    return {};
  }

  const overrideMap = FRAMEWORK_PARAM_MAPS[target];
  const overrides: Record<string, unknown> = {};

  for (const [configKey, targetKey] of Object.entries(overrideMap)) {
    if (config[configKey] !== undefined) {
      overrides[targetKey] = config[configKey];
    }
  }

  return overrides;
}

export function estimateModelCostBreakdown(
  model: string,
  inputTokens: number,
  outputTokens: number,
): { inputCost: number; outputCost: number; totalCost: number } {
  const modelLower = model.toLowerCase();
  const sortedPrefixes = Object.keys(MODEL_COSTS).sort(
    (left, right) => right.length - left.length,
  );

  for (const prefix of sortedPrefixes) {
    if (!modelLower.includes(prefix)) {
      continue;
    }

    const [inputCost, outputCost] = MODEL_COSTS[prefix]!;
    const normalizedInputCost = (inputTokens * inputCost) / 1_000_000;
    const normalizedOutputCost = (outputTokens * outputCost) / 1_000_000;
    return {
      inputCost: normalizedInputCost,
      outputCost: normalizedOutputCost,
      totalCost: normalizedInputCost + normalizedOutputCost,
    };
  }

  const fallbackInputCost = inputTokens / 1_000_000;
  const fallbackOutputCost = (outputTokens * 3) / 1_000_000;
  return {
    inputCost: fallbackInputCost,
    outputCost: fallbackOutputCost,
    totalCost: fallbackInputCost + fallbackOutputCost,
  };
}

export function estimateModelCost(
  model: string,
  inputTokens: number,
  outputTokens: number,
): number {
  return estimateModelCostBreakdown(model, inputTokens, outputTokens).totalCost;
}

export function recordProviderUsage(
  model: string,
  inputTokens: number,
  outputTokens: number,
  latencySeconds: number,
): void {
  const totalTokens = inputTokens + outputTokens;
  const { inputCost, outputCost, totalCost } = estimateModelCostBreakdown(
    model,
    inputTokens,
    outputTokens,
  );
  recordRuntimeMetrics({
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: totalTokens,
    input_cost: inputCost,
    output_cost: outputCost,
    total_cost: totalCost,
    cost: totalCost,
    latency: latencySeconds,
  });
}

export function ensurePositiveDuration(valueMs: number): number {
  return Math.max(valueMs / 1000, 0);
}
