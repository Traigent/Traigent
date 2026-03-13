import { describe, expect, it } from 'vitest';

import { TrialContext } from '../../../src/core/context.js';
import { recordRuntimeMetrics } from '../../../src/core/runtime-metrics.js';
import {
  createAgentTrialFunction,
  invokeFunctionWithConfig,
  resolveEvaluationRows,
} from '../../../src/optimization/agent.js';
import { optimize, param } from '../../../src/optimization/spec.js';
import type { TrialConfig } from '../../../src/dtos/trial.js';

function createTrialConfig(
  config: Record<string, unknown>,
  overrides: Partial<TrialConfig> = {}
): TrialConfig {
  return {
    trial_id: 'trial-agent',
    trial_number: 1,
    experiment_run_id: 'exp-agent',
    config,
    dataset_subset: {
      indices: [],
      total: 2,
      ...(overrides.dataset_subset ?? {}),
    },
    ...overrides,
  };
}

describe('agent optimization helpers', () => {
  it('resolves evaluation rows from data or loadData and fails when missing', async () => {
    await expect(
      resolveEvaluationRows({
        evaluation: {
          data: [{ input: 'x' }],
        },
      })
    ).resolves.toEqual([{ input: 'x' }]);

    await expect(
      resolveEvaluationRows({
        evaluation: {
          loadData: async () => [{ input: 'y' }],
        },
      })
    ).resolves.toEqual([{ input: 'y' }]);

    await expect(resolveEvaluationRows({ evaluation: undefined })).rejects.toThrow(
      /requires evaluation\.data or evaluation\.loadData/i
    );
  });

  it('supports parameter injection for zero and one-argument functions and rejects invalid config args', () => {
    expect(
      invokeFunctionWithConfig(
        (config?: Record<string, unknown>) => config,
        undefined,
        [],
        { model: 'best' },
        'parameter'
      )
    ).toEqual({ model: 'best' });

    expect(
      invokeFunctionWithConfig(
        (input: string, config?: Record<string, unknown>) => [input, config],
        undefined,
        ['hello'],
        { tone: 'friendly' },
        'parameter'
      )
    ).toEqual(['hello', { tone: 'friendly' }]);

    expect(() =>
      invokeFunctionWithConfig(
        () => 'never',
        undefined,
        ['hello', 'bad-config'],
        { tone: 'friendly' },
        'parameter'
      )
    ).toThrow(/second argument to be an object config/i);
  });

  it('passes a frozen config object during parameter injection', () => {
    expect(() =>
      invokeFunctionWithConfig(
        (_input: string, config?: Record<string, unknown>) => {
          expect(Object.isFrozen(config)).toBe(true);
          (config as Record<string, unknown>)['tone'] = 'mutated';
          return 'never';
        },
        undefined,
        ['hello'],
        { tone: 'friendly', nested: { retries: 2 } },
        'parameter'
      )
    ).toThrow(TypeError);
  });

  it('rejects non-plain objects as parameter config values', () => {
    expect(() =>
      invokeFunctionWithConfig(
        () => 'never',
        undefined,
        ['hello', new Date()],
        { tone: 'friendly' },
        'parameter'
      )
    ).toThrow(/second argument to be an object config/i);
  });

  it('aggregates example metrics using default and explicit strategies', async () => {
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(['quiet', 'loud']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [
          {
            prompt: 'hi',
            expected: 'ignored',
            score: 0.25,
            quality: 0.4,
            cost: 0.1,
            inputCost: 0.03,
            outputCost: 0.07,
            latency: 3,
          },
          {
            prompt: 'bye',
            expected: 'ignored',
            score: 0.75,
            quality: 0.9,
            cost: 0.2,
            inputCost: 0.08,
            outputCost: 0.12,
            latency: 1,
          },
        ],
        inputField: 'prompt',
        expectedField: 'expected',
        customEvaluator: async ({ row }) => {
          const record = row as {
            score: number;
            quality: number;
            cost: number;
            inputCost: number;
            outputCost: number;
            latency: number;
          };
          return {
            accuracy: record.score,
            quality: record.quality,
            cost: record.cost,
            input_cost: record.inputCost,
            output_cost: record.outputCost,
            total_cost: record.inputCost + record.outputCost,
            latency: record.latency,
            optional: undefined as unknown as number,
          };
        },
        aggregation: {
          quality: 'max',
          cost: 'sum',
          input_cost: 'sum',
          output_cost: 'sum',
          total_cost: 'sum',
          latency: 'min',
          default: 'mean',
        },
      },
    })(async (input: string) => input);

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.bestMetrics).toMatchObject({
      accuracy: 0.5,
      quality: 0.9,
      cost: 0.30000000000000004,
      input_cost: 0.11,
      output_cost: 0.19,
      total_cost: 0.30000000000000004,
      latency: 1,
      optional: null,
    });
  });

  it('supports python-style customEvaluator(agentFn, config, row)', async () => {
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(['quiet', 'loud']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [
          { input: 'hello', output: 'HELLO', score: 1 },
          { input: 'bye', output: 'BYE', score: 0.5 },
        ],
        customEvaluator: async (agentFn, config, row) => {
          const record = row as { input: string; output: string; score: number };
          const prediction = await agentFn(record.input);
          return {
            accuracy: prediction === record.output ? record.score : 0,
            latency: config.tone === 'loud' ? 2 : 1,
          };
        },
      },
      injection: {
        mode: 'parameter',
      },
    })(async (input: string, config?: { tone?: string }) =>
      config?.tone === 'loud' ? `${String(input).toUpperCase()}!` : String(input).toUpperCase()
    );

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ tone: 'quiet' });
    expect(result.bestMetrics).toMatchObject({
      accuracy: 0.75,
      latency: 1,
    });
  });

  it('uses inline_rows when provided and preserves scoringFunction as the primary metric', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'unused', output: 'UNUSED' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
        metricFunctions: {
          accuracy: () => 0,
          auxiliary: (_output, _expectedOutput, _runtimeMetrics, row) =>
            (row as { score: number }).score,
        },
      },
    })(async (input: string) => input.toUpperCase());

    const normalizedSpec = (await import('../../../src/optimization/spec.js')).getOptimizationSpec(
      wrapped
    )!;

    const trialFn = createAgentTrialFunction(
      async (input: string) => input.toUpperCase(),
      normalizedSpec,
      [{ input: 'unused', output: 'UNUSED' }]
    );

    const result = await TrialContext.run(
      createTrialConfig(
        { model: 'a' },
        {
          dataset_subset: {
            indices: [0],
            total: 2,
            inline_rows: [
              { input: 'first', output: 'FIRST', score: 0.2 },
              { input: 'second', output: 'SECOND', score: 0.8 },
            ],
          },
        }
      ),
      () =>
        trialFn(
          createTrialConfig(
            { model: 'a' },
            {
              dataset_subset: {
                indices: [0],
                total: 2,
                inline_rows: [
                  { input: 'first', output: 'FIRST', score: 0.2 },
                  { input: 'second', output: 'SECOND', score: 0.8 },
                ],
              },
            }
          )
        )
    );

    expect(result.metrics.accuracy).toBe(1);
    expect(result.metrics.auxiliary).toBe(0.5);
  });

  it('fails on missing input or expected fields when they are configured', async () => {
    const missingInput = optimize({
      configurationSpace: {
        tone: param.enum(['quiet']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ question: 'hi', output: 'HI' }],
        inputField: 'prompt',
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (input: string) => input.toUpperCase());

    await expect(
      missingInput.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      })
    ).rejects.toThrow(/missing input field "prompt"/i);

    const missingExpected = optimize({
      configurationSpace: {
        tone: param.enum(['quiet']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'hi', answer: 'HI' }],
        expectedField: 'output',
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (input: string) => input.toUpperCase());

    await expect(
      missingExpected.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      })
    ).rejects.toThrow(/missing expected field "output"/i);
  });

  it('supports primitive rows when no inputField is configured', async () => {
    const wrapped = optimize({
      configurationSpace: {
        format: param.enum(['upper']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: ['hello'],
        customEvaluator: async ({ output }) => ({
          accuracy: output === 'HELLO' ? 1 : 0,
        }),
      },
    })(async (input: string) => input.toUpperCase());

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(result.bestMetrics?.accuracy).toBe(1);
  });

  it('supports maxTotalExamples and repetition aggregation on the high-level agent contract', async () => {
    let callCount = 0;
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(['best']),
      },
      objectives: ['accuracy'],
      execution: {
        maxTotalExamples: 2,
        repsPerTrial: 2,
        repsAggregation: 'max',
      },
      evaluation: {
        data: [
          { input: 'a', output: 'A' },
          { input: 'b', output: 'B' },
          { input: 'c', output: 'C' },
        ],
        scoringFunction: () => {
          callCount += 1;
          return callCount === 1 ? 0.2 : 0.9;
        },
      },
    })(async (input: string) => input.toUpperCase());

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 3,
    });

    expect(result.stopReason).toBe('maxExamples');
    expect(result.trials).toHaveLength(1);
    expect(result.bestMetrics?.accuracy).toBe(0.9);
    expect(result.trials[0]?.metadata).toMatchObject({
      evaluatedRows: 2,
      repsPerTrial: 2,
      repsAggregation: 'max',
    });
  });

  it('supports opt-in concurrent example evaluation with stable aggregation and isolated runtime metrics', async () => {
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(['quiet']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [
          {
            payload: { text: 'slow', delayMs: 20, tokens: 1 },
            score: 1,
          },
          {
            payload: { text: 'fast', delayMs: 0, tokens: 10 },
            score: 0.5,
          },
        ],
        inputField: 'payload',
        customEvaluator: async ({ row, runtimeMetrics }) => ({
          accuracy: (row as { score: number }).score,
          observed_tokens: runtimeMetrics.input_tokens ?? 0,
        }),
      },
      execution: {
        exampleConcurrency: 2,
      },
    })(async (input: { text: string; delayMs: number; tokens: number }) => {
      await new Promise((resolve) => setTimeout(resolve, input.delayMs));
      recordRuntimeMetrics({ input_tokens: input.tokens });
      return input.text.toUpperCase();
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(result.bestMetrics).toMatchObject({
      accuracy: 0.75,
      observed_tokens: 5.5,
      input_tokens: 11,
    });
  });
});
