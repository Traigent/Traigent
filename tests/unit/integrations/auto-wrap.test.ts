import { beforeEach, describe, expect, it, vi } from 'vitest';

import { TrialContext } from '../../../src/core/context.js';
import {
  autoWrapFrameworkTarget,
  autoWrapFrameworkTargets,
} from '../../../src/integrations/auto-wrap.js';
import { clearRegisteredFrameworkTargets } from '../../../src/integrations/registry.js';
import type { TrialConfig } from '../../../src/dtos/trial.js';

function createTrialConfig(config: Record<string, unknown>): TrialConfig {
  return {
    trial_id: 'trial-auto-wrap',
    trial_number: 1,
    experiment_run_id: 'exp-auto-wrap',
    config,
    dataset_subset: {
      indices: [0],
      total: 1,
    },
  };
}

describe('autoWrapFrameworkTarget(s)', () => {
  beforeEach(() => {
    clearRegisteredFrameworkTargets();
  });

  it('returns non-object values unchanged', () => {
    expect(autoWrapFrameworkTargets('plain-value')).toBe('plain-value');
    expect(autoWrapFrameworkTargets(42)).toBe(42);
    expect(autoWrapFrameworkTargets(null)).toBeNull();
  });

  it('wraps supported targets inside arrays', async () => {
    const create = vi.fn(async (params: Record<string, unknown>) => ({
      model: params.model,
      usage: {
        prompt_tokens: 2,
        completion_tokens: 1,
      },
    }));

    const [client, untouched] = autoWrapFrameworkTargets([
      {
        chat: {
          completions: {
            create,
          },
        },
      },
      { label: 'plain-object' },
    ]);

    expect(untouched).toEqual({ label: 'plain-object' });

    await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini', temperature: 0.1 }),
      async () => {
        await client.chat?.completions?.create({
          model: 'gpt-3.5-turbo',
        });
      },
    );

    expect(create).toHaveBeenCalledWith({
      model: 'gpt-4o-mini',
      temperature: 0.1,
    });
  });

  it('wraps a direct LangChain-like target passed to autoWrapFrameworkTargets', async () => {
    const invoke = vi.fn(async () => 'ok');
    const bind = vi.fn(() => ({ invoke }));

    const wrapped = autoWrapFrameworkTargets({
      modelName: 'gpt-4o-mini',
      bind,
      async invoke() {
        return 'fallback';
      },
    });

    await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini', temperature: 0.3 }),
      async () => {
        await wrapped.invoke?.('hello');
      },
    );

    expect(bind).toHaveBeenCalledWith({
      model: 'gpt-4o-mini',
      temperature: 0.3,
    });
    expect(invoke).toHaveBeenCalledWith('hello');
  });

  it('wraps Vercel AI-like targets via autoWrapFrameworkTarget', async () => {
    const doGenerate = vi.fn(async (params: Record<string, unknown>) => ({
      usage: {
        promptTokens: 5,
        completionTokens: 2,
      },
      modelId: params.modelId,
    }));

    const wrapped = autoWrapFrameworkTarget({
      specificationVersion: 'v1',
      provider: 'openai',
      modelId: 'gpt-3.5-turbo',
      defaultObjectGenerationMode: 'json',
      supportedUrls: {},
      async doGenerate(params: Record<string, unknown>) {
        return doGenerate(params);
      },
    });

    await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini' }),
      async () => {
        await wrapped.doGenerate?.({
          modelId: 'gpt-3.5-turbo',
        });
      },
    );

    expect(doGenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        modelId: 'gpt-4o-mini',
      }),
    );
  });
});
