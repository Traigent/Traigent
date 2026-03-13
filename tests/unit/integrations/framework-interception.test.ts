import { beforeEach, describe, expect, it, vi } from 'vitest';

import { TrialContext } from '../../../src/core/context.js';
import { withRuntimeMetricsCollector } from '../../../src/core/runtime-metrics.js';
import { clearRegisteredFrameworkTargets } from '../../../src/integrations/registry.js';
import { createTraigentOpenAI } from '../../../src/integrations/openai/index.js';
import { withTraigentModel } from '../../../src/integrations/langchain/index.js';
import { withTraigent } from '../../../src/integrations/vercel-ai/index.js';
import type { TrialConfig } from '../../../src/dtos/trial.js';

function createTrialConfig(config: Record<string, unknown>): TrialConfig {
  return {
    trial_id: 'trial-test',
    trial_number: 1,
    experiment_run_id: 'exp-test',
    config,
    dataset_subset: {
      indices: [0],
      total: 1,
    },
  };
}

describe('framework interception', () => {
  beforeEach(() => {
    clearRegisteredFrameworkTargets();
  });

  it('overrides OpenAI request params from trial context and records usage metrics', async () => {
    const create = vi.fn(async (params: Record<string, unknown>) => ({
      model: params.model,
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
      },
    }));

    const client = createTraigentOpenAI({
      chat: {
        completions: {
          create,
        },
      },
    });

    const { metrics } = await TrialContext.run(
      createTrialConfig({
        model: 'gpt-4o-mini',
        temperature: 0.1,
        maxTokens: 128,
      }),
      async () =>
        withRuntimeMetricsCollector(async () => {
          await client.chat?.completions?.create({
            model: 'gpt-3.5-turbo',
            temperature: 0.8,
            max_tokens: 32,
          });
        })
    );

    expect(create).toHaveBeenCalledWith({
      model: 'gpt-4o-mini',
      temperature: 0.1,
      max_tokens: 128,
    });
    expect(metrics.input_tokens).toBe(10);
    expect(metrics.output_tokens).toBe(5);
    expect(metrics.total_tokens).toBe(15);
    expect(metrics.total_cost).toBe(metrics.cost);
  });

  it('overrides LangChain model params via bind() and records runtime metrics', async () => {
    const invoke = vi.fn(async () => ({
      usage_metadata: {
        input_tokens: 11,
        output_tokens: 7,
      },
    }));
    const bind = vi.fn((config: Record<string, unknown>) => ({
      invoke,
      config,
    }));

    const model = withTraigentModel({
      modelName: 'gpt-4o-mini',
      bind,
      async invoke() {
        return 'fallback';
      },
    });

    const { metrics } = await TrialContext.run(
      createTrialConfig({
        model: 'gpt-4o-mini',
        temperature: 0.2,
      }),
      async () =>
        withRuntimeMetricsCollector(async () => {
          await model.invoke?.('hello');
        })
    );

    expect(bind).toHaveBeenCalledWith({
      model: 'gpt-4o-mini',
      temperature: 0.2,
    });
    expect(invoke).toHaveBeenCalledWith('hello');
    expect(metrics.input_tokens).toBe(11);
    expect(metrics.output_tokens).toBe(7);
    expect(metrics.total_tokens).toBe(18);
  });

  it('records LangChain usage from response_metadata and llmOutput fallbacks', async () => {
    const model = withTraigentModel({
      modelName: 'gpt-4o-mini',
      bind: vi.fn(() => ({
        async batch() {
          return [
            {
              response_metadata: {
                usage: {
                  input_tokens: 5,
                  output_tokens: 2,
                },
              },
            },
            {
              llmOutput: {
                tokenUsage: {
                  promptTokens: 7,
                  completionTokens: 3,
                },
              },
            },
          ];
        },
      })),
      async batch() {
        return [];
      },
    });

    const { metrics } = await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini' }),
      async () =>
        withRuntimeMetricsCollector(async () => {
          await model.batch?.(['hello', 'world']);
        })
    );

    expect(metrics.input_tokens).toBe(12);
    expect(metrics.output_tokens).toBe(5);
    expect(metrics.total_tokens).toBe(17);
  });

  it('records LangChain usage from response_metadata.tokenUsage', async () => {
    const model = withTraigentModel({
      modelName: 'gpt-4o-mini',
      bind: vi.fn(() => ({
        async invoke() {
          return {
            response_metadata: {
              tokenUsage: {
                promptTokens: 9,
                completionTokens: 4,
              },
            },
          };
        },
      })),
      async invoke() {
        return {
          content: 'fallback',
        };
      },
    });

    const { metrics } = await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini' }),
      async () =>
        withRuntimeMetricsCollector(async () => {
          await model.invoke?.('hello');
        })
    );

    expect(metrics.input_tokens).toBe(9);
    expect(metrics.output_tokens).toBe(4);
    expect(metrics.total_tokens).toBe(13);
    expect(metrics.total_cost).toBe(metrics.cost);
  });

  it('falls back to the original LangChain model when bind is unavailable and preserves passthrough members', async () => {
    const invoke = vi.fn(async () => ({
      content: 'ok',
    }));

    const model = withTraigentModel({
      modelName: 'gpt-4o-mini',
      invoke,
      stream: 'not-a-function' as never,
      extra: 'value',
    } as any);

    expect((model as Record<string, unknown>).extra).toBe('value');
    expect(model.stream).toBe('not-a-function');

    const { metrics } = await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini' }),
      async () =>
        withRuntimeMetricsCollector(async () => {
          await model.invoke?.('hello');
        })
    );

    expect(invoke).toHaveBeenCalledWith('hello');
    expect(metrics.total_tokens).toBe(0);
  });

  it('handles camelCase usage metadata and malformed LangChain usage payloads', async () => {
    const camelCaseModel = withTraigentModel({
      modelName: 'fallback-model',
      bind: vi.fn(() => ({
        async invoke() {
          return {
            usage_metadata: {
              inputTokens: 6,
              outputTokens: 4,
            },
          };
        },
      })),
      async invoke() {
        return 'unused';
      },
    });

    const camelCaseMetrics = await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini' }),
      async () =>
        withRuntimeMetricsCollector(async () => {
          await camelCaseModel.invoke?.('hello');
        })
    );

    expect(camelCaseMetrics.metrics.input_tokens).toBe(6);
    expect(camelCaseMetrics.metrics.output_tokens).toBe(4);
    expect(camelCaseMetrics.metrics.total_tokens).toBe(10);

    const malformedModel = withTraigentModel({
      modelId: 'model-from-id',
      async batch() {
        return [
          {
            usage_metadata: {
              input_tokens: 'bad',
              output_tokens: null,
            },
          },
          {
            response_metadata: {
              tokenUsage: {
                promptTokens: 'bad',
                completionTokens: null,
              },
            },
          },
          {
            response_metadata: {
              usage: {
                inputTokens: 'bad',
                outputTokens: null,
              },
            },
          },
          {
            llmOutput: {
              tokenUsage: {
                input_tokens: 'bad',
                output_tokens: null,
              },
            },
          },
        ];
      },
    });

    const { metrics } = await TrialContext.run(createTrialConfig({}), async () =>
      withRuntimeMetricsCollector(async () => {
        await malformedModel.batch?.(['a', 'b']);
      })
    );

    expect(metrics.input_tokens).toBe(0);
    expect(metrics.output_tokens).toBe(0);
    expect(metrics.total_tokens).toBe(0);
  });

  it('overrides Vercel AI generation params through middleware', async () => {
    const doGenerate = vi.fn(async (params: Record<string, unknown>) => ({
      text: 'ok',
      usage: {
        promptTokens: 10,
        completionTokens: 4,
      },
      params,
    }));

    const model = withTraigent({
      specificationVersion: 'v1',
      provider: 'test',
      modelId: 'gpt-3.5-turbo',
      defaultObjectGenerationMode: 'json',
      supportsImageUrls: false,
      supportsUrl: false,
      supportsStructuredOutputs: false,
      async doGenerate(params: Record<string, unknown>) {
        return doGenerate(params);
      },
      async doStream() {
        return {
          usage: {
            promptTokens: 0,
            completionTokens: 0,
          },
        };
      },
    });

    const { metrics } = await TrialContext.run(
      createTrialConfig({
        model: 'gpt-4o-mini',
        temperature: 0.4,
        maxTokens: 256,
      }),
      async () =>
        withRuntimeMetricsCollector(async () => {
          await model.doGenerate({
            modelId: 'gpt-3.5-turbo',
            temperature: 0.9,
            maxTokens: 32,
          });
        })
    );

    expect(doGenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        modelId: 'gpt-4o-mini',
        temperature: 0.4,
        maxTokens: 256,
      })
    );
    expect(metrics.input_tokens).toBe(10);
    expect(metrics.output_tokens).toBe(4);
    expect(metrics.total_tokens).toBe(14);
  });

  it('does not double-wrap framework targets', async () => {
    const create = vi.fn(async () => ({
      model: 'gpt-4o-mini',
      usage: {
        prompt_tokens: 2,
        completion_tokens: 1,
      },
    }));

    const openaiClient = {
      chat: {
        completions: {
          create,
        },
      },
    };
    createTraigentOpenAI(openaiClient);
    createTraigentOpenAI(openaiClient);

    const invoke = vi.fn(async () => 'ok');
    const bind = vi.fn(() => ({ invoke }));
    const wrappedOnce = withTraigentModel({
      modelName: 'gpt-4o-mini',
      bind,
      async invoke() {
        return 'fallback';
      },
    });
    const wrappedTwice = withTraigentModel(wrappedOnce);

    expect(wrappedTwice).toBe(wrappedOnce);

    const baseModel = {
      specificationVersion: 'v1',
      provider: 'openai',
      modelId: 'gpt-3.5-turbo',
      defaultObjectGenerationMode: 'json',
      supportedUrls: {},
      async doGenerate(params: Record<string, unknown>) {
        return {
          usage: {
            promptTokens: 1,
            completionTokens: 1,
          },
          modelId: params.modelId,
        };
      },
    };
    const vercelOnce = withTraigent(baseModel as never);
    const vercelTwice = withTraigent(vercelOnce);
    expect(vercelTwice).toBe(vercelOnce);

    await TrialContext.run(
      createTrialConfig({ model: 'gpt-4o-mini', temperature: 0.2 }),
      async () => {
        await openaiClient.chat.completions.create({ model: 'x' });
        await wrappedTwice.invoke?.('hello');
      }
    );

    expect(create).toHaveBeenCalledTimes(1);
    expect(bind).toHaveBeenCalledTimes(1);
    expect(invoke).toHaveBeenCalledTimes(1);
  });
});
