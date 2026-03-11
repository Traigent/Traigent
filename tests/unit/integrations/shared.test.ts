import { describe, expect, it } from 'vitest';

import { TrialContext } from '../../../src/core/context.js';
import type { TrialConfig } from '../../../src/dtos/trial.js';
import {
  ensurePositiveDuration,
  estimateModelCost,
  estimateModelCostBreakdown,
  getFrameworkOverrides,
} from '../../../src/integrations/shared.js';

function createTrialConfig(config: Record<string, unknown>): TrialConfig {
  return {
    trial_id: 'trial-shared',
    trial_number: 1,
    experiment_run_id: 'exp-shared',
    config,
    dataset_subset: {
      indices: [0],
      total: 1,
    },
  };
}

describe('integration shared helpers', () => {
  it('returns no framework overrides when no trial context is active', () => {
    expect(getFrameworkOverrides('openai')).toEqual({});
    expect(getFrameworkOverrides('langchain')).toEqual({});
    expect(getFrameworkOverrides('vercel-ai')).toEqual({});
  });

  it('maps active trial config keys to framework-specific override keys', async () => {
    const overrides = await TrialContext.run(
      createTrialConfig({
        model: 'gpt-4o-mini',
        temperature: 0.2,
        maxTokens: 64,
        topP: 0.9,
        frequencyPenalty: 0.1,
        presencePenalty: 0.05,
      }),
      async () => ({
        openai: getFrameworkOverrides('openai'),
        langchain: getFrameworkOverrides('langchain'),
        vercel: getFrameworkOverrides('vercel-ai'),
      }),
    );

    expect(overrides.openai).toEqual({
      model: 'gpt-4o-mini',
      temperature: 0.2,
      max_tokens: 64,
      top_p: 0.9,
      frequency_penalty: 0.1,
      presence_penalty: 0.05,
    });
    expect(overrides.langchain).toEqual({
      model: 'gpt-4o-mini',
      temperature: 0.2,
      maxTokens: 64,
      topP: 0.9,
      frequencyPenalty: 0.1,
      presencePenalty: 0.05,
    });
    expect(overrides.vercel).toEqual({
      modelId: 'gpt-4o-mini',
      temperature: 0.2,
      maxTokens: 64,
      topP: 0.9,
      frequencyPenalty: 0.1,
      presencePenalty: 0.05,
    });
  });

  it('uses a known pricing table when the model is recognized and falls back otherwise', () => {
    expect(estimateModelCostBreakdown('gpt-4o-mini', 1000, 500)).toEqual({
      inputCost: 0.00015,
      outputCost: 0.0003,
      totalCost: 0.00045,
    });

    expect(estimateModelCostBreakdown('unknown-model', 1000, 500)).toEqual({
      inputCost: 0.001,
      outputCost: 0.0015,
      totalCost: 0.0025,
    });

    expect(estimateModelCost('unknown-model', 1000, 500)).toBe(0.0025);
  });

  it('normalizes negative durations to zero seconds', () => {
    expect(ensurePositiveDuration(1500)).toBe(1.5);
    expect(ensurePositiveDuration(-250)).toBe(0);
  });
});
