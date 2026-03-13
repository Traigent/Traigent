import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { TrialConfig } from '../../../src/dtos/trial.js';
import {
  autoWrapFrameworkTarget,
  autoWrapFrameworkTargets,
  discoverFrameworkTargets,
  prepareFrameworkTargets,
} from '../../../src/integrations/auto-wrap.js';
import { clearRegisteredFrameworkTargets } from '../../../src/integrations/registry.js';
import { TrialContext } from '../../../src/core/context.js';

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
    expect(autoWrapFrameworkTarget('plain-value')).toBe('plain-value');
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
      }
    );

    expect(create).toHaveBeenCalledWith({
      model: 'gpt-4o-mini',
      temperature: 0.1,
    });
  });

  it('wraps supported values inside a plain object map and leaves untouched entries alone', async () => {
    const create = vi.fn(async () => ({
      usage: {
        prompt_tokens: 1,
        completion_tokens: 1,
      },
    }));

    const wrapped = autoWrapFrameworkTargets({
      openaiClient: {
        chat: {
          completions: {
            create,
          },
        },
      },
      untouched: 'plain-value',
    });

    expect(wrapped.untouched).toBe('plain-value');

    await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini', temperature: 0.4 }),
      async () => {
        await wrapped.openaiClient.chat?.completions?.create({
          model: 'gpt-3.5-turbo',
        });
      }
    );

    expect(create).toHaveBeenCalledWith({
      model: 'gpt-4o-mini',
      temperature: 0.4,
    });
  });

  it('recursively wraps supported targets inside nested plain-object graphs', async () => {
    const create = vi.fn(async () => ({
      usage: {
        prompt_tokens: 3,
        completion_tokens: 1,
      },
    }));

    const wrapped = autoWrapFrameworkTargets({
      services: {
        llm: {
          primary: {
            chat: {
              completions: {
                create,
              },
            },
          },
        },
      },
    });

    await TrialContext.run(createTrialConfig({ model: 'gpt-4o', temperature: 0.2 }), async () => {
      await wrapped.services.llm.primary.chat?.completions?.create({
        model: 'gpt-3.5-turbo',
      });
    });

    expect(create).toHaveBeenCalledWith({
      model: 'gpt-4o',
      temperature: 0.2,
    });
  });

  it('preserves cycles and repeated references while recursively wrapping', async () => {
    const create = vi.fn(async () => ({
      usage: {
        prompt_tokens: 1,
        completion_tokens: 1,
      },
    }));
    const sharedClient = {
      chat: {
        completions: {
          create,
        },
      },
    };
    const graph: {
      primary: typeof sharedClient;
      secondary?: typeof sharedClient;
      self?: unknown;
    } = {
      primary: sharedClient,
    };
    graph.secondary = sharedClient;
    graph.self = graph;

    const wrapped = autoWrapFrameworkTargets(graph);

    expect(wrapped.self).toBe(wrapped);
    expect(wrapped.primary).toBe(wrapped.secondary);

    await TrialContext.run(createTrialConfig({ model: 'gpt-4o-mini' }), async () => {
      await wrapped.secondary?.chat?.completions?.create({
        model: 'gpt-3.5-turbo',
      });
    });

    expect(create).toHaveBeenCalledWith({
      model: 'gpt-4o-mini',
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
      }
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

    await TrialContext.run(createTrialConfig({ model: 'gpt-4o-mini' }), async () => {
      await wrapped.doGenerate?.({
        modelId: 'gpt-3.5-turbo',
      });
    });

    expect(doGenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        modelId: 'gpt-4o-mini',
      })
    );
  });

  it('documents referential behavior: OpenAI is in-place, LangChain/Vercel return wrapped objects', () => {
    const openaiClient = {
      chat: {
        completions: {
          create: vi.fn(async () => ({ usage: {} })),
        },
      },
    };
    const langchainModel = {
      modelName: 'gpt-4o-mini',
      bind: vi.fn(() => ({
        invoke: vi.fn(async () => 'ok'),
      })),
      async invoke() {
        return 'fallback';
      },
    };
    const vercelModel = {
      specificationVersion: 'v1',
      provider: 'openai',
      modelId: 'gpt-4o-mini',
      defaultObjectGenerationMode: 'json',
      supportedUrls: {},
      async doGenerate() {
        return { usage: {} };
      },
    };

    expect(autoWrapFrameworkTarget(openaiClient)).toBe(openaiClient);
    expect(autoWrapFrameworkTarget(langchainModel)).not.toBe(langchainModel);
    expect(autoWrapFrameworkTarget(vercelModel as never)).not.toBe(vercelModel);
  });

  it('discovers supported targets inside explicit object graphs with stable paths', () => {
    const graph = {
      openai: {
        chat: {
          completions: {
            create: vi.fn(async () => ({ usage: {} })),
          },
        },
      },
      nested: {
        models: [
          {
            modelName: 'gpt-4o-mini',
            bind: vi.fn(() => ({ invoke: vi.fn(async () => 'ok') })),
            async invoke() {
              return 'fallback';
            },
          },
        ],
      },
      untouched: {
        provider: 'custom',
      },
    };

    expect(discoverFrameworkTargets(graph)).toEqual([
      { path: 'openai', target: 'openai' },
      { path: 'nested.models[0]', target: 'langchain' },
    ]);
  });

  it('discovers a direct root target once and ignores repeated cycle traversal', () => {
    const root = {
      chat: {
        completions: {
          create: vi.fn(async () => ({ usage: {} })),
        },
      },
    };
    const graph: { root: typeof root; loop?: unknown } = { root };
    graph.loop = graph;

    expect(discoverFrameworkTargets(root)).toEqual([{ path: '<root>', target: 'openai' }]);
    expect(discoverFrameworkTargets(graph)).toEqual([{ path: 'root', target: 'openai' }]);
  });

  it('does not recurse into non-plain container instances', () => {
    class RuntimeContainer {
      constructor(readonly client: unknown) {}
    }

    const create = vi.fn(async () => ({
      usage: {
        prompt_tokens: 1,
        completion_tokens: 1,
      },
    }));
    const container = new RuntimeContainer({
      chat: {
        completions: {
          create,
        },
      },
    });

    expect(discoverFrameworkTargets(container)).toEqual([]);
    expect(autoWrapFrameworkTargets(container)).toBe(container);
  });

  it('prepares framework targets by discovering, wrapping, and reporting auto-override status', async () => {
    const create = vi.fn(async (params: Record<string, unknown>) => ({
      usage: {
        prompt_tokens: 2,
        completion_tokens: 1,
      },
      model: params.model,
    }));

    const prepared = prepareFrameworkTargets({
      providers: {
        primary: {
          chat: {
            completions: {
              create,
            },
          },
        },
      },
    });

    expect(prepared.discovered).toEqual([{ path: 'providers.primary', target: 'openai' }]);
    expect(prepared.autoOverrideStatus).toMatchObject({
      enabled: true,
      activeTargets: ['openai'],
      selectedTargets: ['openai'],
    });

    await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini', temperature: 0.2 }),
      async () => {
        await prepared.wrapped.providers.primary.chat?.completions?.create({
          model: 'gpt-3.5-turbo',
        });
      }
    );

    expect(create).toHaveBeenCalledWith({
      model: 'gpt-4o-mini',
      temperature: 0.2,
    });
  });

  it('reports disabled auto-override when preparing targets with autoOverrideFrameworks=false', () => {
    const prepared = prepareFrameworkTargets(
      {
        chat: {
          completions: {
            create: vi.fn(async () => ({ usage: {} })),
          },
        },
      },
      {
        autoOverrideFrameworks: false,
      }
    );

    expect(prepared.discovered).toEqual([{ path: '<root>', target: 'openai' }]);
    expect(prepared.autoOverrideStatus).toMatchObject({
      autoOverrideFrameworks: false,
      enabled: false,
      activeTargets: ['openai'],
      selectedTargets: [],
    });
  });
});
