import { experimental_wrapLanguageModel, type LanguageModel } from 'ai';

import { registerFrameworkTarget } from '../registry.js';
import {
  ensurePositiveDuration,
  getFrameworkOverrides,
  recordProviderUsage,
} from '../shared.js';

const TRAIGENT_WRAPPED = Symbol.for('traigent.vercel-ai.wrapped');

function getUsage(result: unknown): {
  promptTokens: number;
  completionTokens: number;
} {
  if (!result || typeof result !== 'object') {
    return {
      promptTokens: 0,
      completionTokens: 0,
    };
  }

  const usage = (result as Record<string, unknown>)['usage'];
  if (!usage || typeof usage !== 'object') {
    return {
      promptTokens: 0,
      completionTokens: 0,
    };
  }

  const usageRecord = usage as Record<string, unknown>;
  return {
    promptTokens:
      typeof usageRecord['promptTokens'] === 'number'
        ? (usageRecord['promptTokens'] as number)
        : 0,
    completionTokens:
      typeof usageRecord['completionTokens'] === 'number'
        ? (usageRecord['completionTokens'] as number)
        : 0,
  };
}

function getModelName(
  params: Record<string, unknown>,
  model: LanguageModel,
): string {
  return String(params['modelId'] ?? model.modelId);
}

export function withTraigent<T extends LanguageModel>(model: T): T {
  if ((model as T & { [TRAIGENT_WRAPPED]?: true })[TRAIGENT_WRAPPED]) {
    return model;
  }

  registerFrameworkTarget('vercel-ai');

  const wrapped = experimental_wrapLanguageModel({
    model,
    middleware: {
      async transformParams({ params }) {
        const overrides = getFrameworkOverrides('vercel-ai');
        return {
          ...params,
          ...overrides,
        };
      },
      async wrapGenerate({ doGenerate, params }) {
        const startedAt = Date.now();
        const result = await doGenerate();
        const { promptTokens, completionTokens } = getUsage(result);
        recordProviderUsage(
          getModelName(params as Record<string, unknown>, model),
          promptTokens,
          completionTokens,
          ensurePositiveDuration(Date.now() - startedAt),
        );
        return result;
      },
      async wrapStream({ doStream, params }) {
        const startedAt = Date.now();
        const result = await doStream();
        const { promptTokens, completionTokens } = getUsage(result);
        recordProviderUsage(
          getModelName(params as Record<string, unknown>, model),
          promptTokens,
          completionTokens,
          ensurePositiveDuration(Date.now() - startedAt),
        );
        return result;
      },
    },
  }) as T;

  Object.defineProperty(wrapped, TRAIGENT_WRAPPED, {
    value: true,
    enumerable: false,
    configurable: false,
    writable: false,
  });

  return wrapped;
}
