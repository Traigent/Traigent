import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  getOptimizationSpec,
  getTrialParam,
  optimize,
  param,
  toHybridConfigSpace,
} from '../../../src/index.js';
import { TrialContext } from '../../../src/core/context.js';
import { ValidationError } from '../../../src/core/errors.js';
import { clearRegisteredFrameworkTargets } from '../../../src/integrations/registry.js';
import { createTraigentOpenAI } from '../../../src/integrations/openai/index.js';

describe('optimization spec helpers', () => {
  beforeEach(() => {
    clearRegisteredFrameworkTargets();
  });

  it('attaches metadata and keeps the wrapped callable behavior intact', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-4o-mini', 'gpt-4o']),
      },
      objectives: ['accuracy'],
      defaultConfig: {
        prompt_style: 'grounded',
      },
      constraints: [() => true],
      safetyConstraints: [(_config, metrics) => Number(metrics.accuracy) >= 0],
      evaluation: {
        data: [{ input: 'ok', output: 'OK' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (value: string) => value.toUpperCase());

    expect(await wrapped('ok')).toBe('OK');
    expect(getOptimizationSpec(wrapped)).toMatchObject({
      configurationSpace: {
        model: {
          type: 'enum',
          values: ['gpt-4o-mini', 'gpt-4o'],
        },
      },
      objectives: [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
      defaultConfig: {
        prompt_style: 'grounded',
      },
      constraints: [expect.any(Function)],
      safetyConstraints: [expect.any(Function)],
      injection: { mode: 'context' },
      execution: {
        mode: 'native',
        contract: 'agent',
        repsPerTrial: 1,
        repsAggregation: 'mean',
      },
    });
    expect(typeof wrapped.optimize).toBe('function');
    expect(typeof wrapped.applyBestConfig).toBe('function');
    expect(typeof wrapped.currentConfig).toBe('function');
  });

  it('serializes config space to the hybrid tunables shape', () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-4o-mini', 'gpt-4o']),
        temperature: param.float({ min: 0, max: 1, scale: 'linear' }),
        max_retries: param.int({ min: 0, max: 3, scale: 'linear' }),
      },
      objectives: ['accuracy', 'cost'],
      evaluation: {
        data: [{ input: 'hello', output: 'hello' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (value: string) => value);

    expect(toHybridConfigSpace(wrapped)).toEqual({
      tunables: [
        {
          name: 'model',
          type: 'enum',
          domain: { values: ['gpt-4o-mini', 'gpt-4o'] },
        },
        {
          name: 'temperature',
          type: 'float',
          domain: { range: [0, 1] },
          scale: 'linear',
        },
        {
          name: 'max_retries',
          type: 'int',
          domain: { range: [0, 3] },
          scale: 'linear',
        },
      ],
      constraints: {},
    });
  });

  it('rejects unknown string objectives', () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['gpt-4o-mini']),
        },
        objectives: ['quality'],
      })
    ).toThrow(ValidationError);
  });

  it('accepts explicit objective objects for custom metrics', () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-4o-mini']),
      },
      objectives: [{ metric: 'quality_score', direction: 'maximize' }],
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        metricFunctions: {
          quality_score: () => 0.9,
        },
      },
    })(async (value: string) => value);

    expect(getOptimizationSpec(wrapped)?.objectives).toEqual([
      { metric: 'quality_score', direction: 'maximize', weight: 1 },
    ]);
  });

  it('normalizes promotionPolicy for native specs', () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'accurate']),
      },
      objectives: ['accuracy'],
      promotionPolicy: {
        dominance: 'epsilon_pareto',
        alpha: 0.05,
        minEffect: {
          accuracy: 0.02,
        },
        tieBreakers: {
          cost: 'minimize',
        },
      },
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (value: string) => value);

    expect(getOptimizationSpec(wrapped)?.promotionPolicy).toEqual({
      dominance: 'epsilon_pareto',
      alpha: 0.05,
      minEffect: {
        accuracy: 0.02,
      },
      tieBreakers: {
        cost: 'minimize',
      },
    });
  });

  it('rejects invalid promotionPolicy fields', () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['cheap']),
        },
        objectives: ['accuracy'],
        promotionPolicy: {
          alpha: 1,
        },
      })
    ).toThrow(/promotionPolicy\.alpha must be in \(0, 1\)/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['cheap']),
        },
        objectives: ['accuracy'],
        promotionPolicy: {
          tieBreakers: {
            cost: 'lower',
          },
        },
      })
    ).toThrow(/promotionPolicy\.tieBreakers\.cost must be "maximize" or "minimize"/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['cheap']),
        },
        objectives: ['accuracy'],
        promotionPolicy: {
          chanceConstraints: [
            {
              name: 'latency',
              threshold: 0.1,
              confidence: 2,
            },
          ],
        },
      })
    ).toThrow(/promotionPolicy\.chanceConstraints\[0\]\.confidence must be in \(0, 1\]/i);
  });

  it('keeps evaluation and injection config in separate normalized fields', () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-4o-mini']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ question: 'x', answer: 'y' }],
        inputField: 'question',
        expectedField: 'answer',
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
      injection: {
        mode: 'parameter',
      },
    })(async (value: string, config?: { model?: string }) => `${config?.model}:${value}`);

    expect(getOptimizationSpec(wrapped)).toMatchObject({
      evaluation: {
        inputField: 'question',
        expectedField: 'answer',
      },
      injection: {
        mode: 'parameter',
      },
    });
  });

  it('rejects non-string, non-object objectives', () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['gpt-4o-mini']),
        },
        objectives: [123 as unknown as never],
      })
    ).toThrow(ValidationError);
  });

  it('injects applied best config into future normal calls via context mode', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'x', output: 'best' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async () => String(getTrialParam('model', 'fallback')));

    expect(await wrapped()).toBe('fallback');

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    wrapped.applyBestConfig(result);
    expect(await wrapped()).toBe('best');
  });

  it('injects defaultConfig into normal calls before optimization', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      defaultConfig: {
        model: 'best',
        region: 'eu',
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'x', output: 'best:eu' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(
      async () =>
        `${String(getTrialParam('model', 'fallback'))}:${String(getTrialParam('region', 'missing'))}`
    );

    expect(wrapped.currentConfig()).toEqual({ model: 'best', region: 'eu' });
    expect(await wrapped()).toBe('best:eu');
  });

  it('injects applied best config into future normal calls via parameter mode', async () => {
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(['concise', 'helpful']),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'parameter',
      },
      evaluation: {
        data: [{ input: 'x', output: 'HELPFUL' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (_input: string, config?: { tone?: string }) =>
      String(config?.tone ?? 'missing').toUpperCase()
    );

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    wrapped.applyBestConfig(result);
    expect(await wrapped('hello')).toBe('HELPFUL');
  });

  it('allows seamless injection without framework targets so transformed code can run', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'seamless',
      },
      evaluation: {
        data: [{ input: 'x', output: 'best' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async function answer() {
      const model = getTrialParam('model', 'cheap');
      return String(model);
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ model: 'best' });
  });

  it('warns when the deprecated low-level trial contract is used', async () => {
    const warningSpy = vi.spyOn(process, 'emitWarning').mockImplementation(() => {});

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(warningSpy).toHaveBeenCalledWith(
      expect.stringContaining('execution.contract="trial"'),
      'DeprecationWarning'
    );

    warningSpy.mockRestore();
  });

  it('rejects customEvaluator when combined with scoringFunction or metricFunctions', () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['gpt-4o-mini']),
        },
        objectives: ['accuracy'],
        evaluation: {
          data: [{ input: 'x', output: 'y' }],
          customEvaluator: async () => ({ accuracy: 1 }),
          scoringFunction: () => 1,
        },
      })
    ).toThrow(/customEvaluator/i);
  });

  it('rejects invalid defaultConfig, constraints, and safetyConstraints values', () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        defaultConfig: 'bad' as unknown as never,
        objectives: ['accuracy'],
      })
    ).toThrow(/defaultConfig/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        constraints: [true] as unknown as never,
        objectives: ['accuracy'],
      })
    ).toThrow(/constraints/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        safetyConstraints: [true] as unknown as never,
        objectives: ['accuracy'],
      })
    ).toThrow(/safetyConstraints/i);
  });

  it('rejects log-scaled parameters with non-positive ranges or non-multiplicative steps', () => {
    expect(() => param.float({ min: 0, max: 1, scale: 'log' })).toThrow(/require min\/max > 0/i);

    expect(() => param.float({ min: 0.001, max: 1, scale: 'log', step: 1 })).toThrow(
      /require step to be greater than 1/i
    );
  });

  it('rejects invalid top-level spec shapes and validation edge cases', () => {
    expect(() => optimize(null as unknown as never)).toThrow(/must be an object/i);

    expect(() =>
      optimize({
        configurationSpace: {},
        objectives: ['accuracy'],
      })
    ).toThrow(/at least one configuration parameter/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: [],
      })
    ).toThrow(/at least one objective/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: ['accuracy'],
        budget: {
          maxCostUsd: 0,
        },
      })
    ).toThrow(/positive number/i);
  });

  it('rejects invalid evaluation, injection, and execution config', () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: ['accuracy'],
        evaluation: {
          data: [{ input: 'x', output: 'x' }],
          inputField: '',
          scoringFunction: () => 1,
        },
      })
    ).toThrow(/inputField/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: ['accuracy'],
        injection: {
          mode: 'invalid' as never,
        },
      })
    ).toThrow(/context", "parameter", or "seamless/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: ['accuracy'],
        execution: {
          mode: 'offline' as never,
        },
      })
    ).toThrow(/execution\.mode/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: ['accuracy'],
        execution: {
          contract: 'unknown' as never,
        },
      })
    ).toThrow(/execution\.contract/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: ['accuracy'],
        execution: {
          maxTotalExamples: 0,
        },
      })
    ).toThrow(/maxTotalExamples/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: ['accuracy'],
        execution: {
          exampleConcurrency: 0,
        },
      })
    ).toThrow(/exampleConcurrency/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: ['accuracy'],
        execution: {
          repsPerTrial: 0,
        },
      })
    ).toThrow(/repsPerTrial/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['a']),
        },
        objectives: ['accuracy'],
        execution: {
          repsAggregation: 'sum' as never,
        },
      })
    ).toThrow(/repsAggregation/i);
  });

  it('normalizes plain-object specs and rejects invalid hybrid config space inputs', () => {
    expect(
      getOptimizationSpec({
        configurationSpace: {
          retries: param.int({ min: 0, max: 2 }),
        },
        objectives: ['cost'],
      })
    ).toMatchObject({
      configurationSpace: {
        retries: {
          type: 'int',
        },
      },
    });

    expect(getOptimizationSpec({ foo: 'bar' })).toBeUndefined();

    expect(() => toHybridConfigSpace({ foo: 'bar' })).toThrow(/requires a wrapped function/i);
  });

  it('clears applied config when applyBestConfig receives no bestConfig and rejects hybrid mode at optimize time', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (value: string) => value);

    expect(wrapped.currentConfig()).toBeUndefined();
    expect(
      wrapped.applyBestConfig({
        bestConfig: null,
        bestMetrics: null,
        trials: [],
        stopReason: 'error',
        totalCostUsd: 0,
      })
    ).toBeUndefined();
    expect(wrapped.currentConfig()).toBeUndefined();

    const hybridWrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      execution: {
        mode: 'hybrid',
      },
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (value: string) => value);

    await expect(
      hybridWrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      })
    ).rejects.toThrow(/not supported in this checkout/i);
  });

  it('rejects optimize() calls when loadData resolves to an empty dataset and reports no seamless path by default', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      evaluation: {
        loadData: async () => [],
        scoringFunction: () => 1,
      },
    })(async (value: string) => value);

    expect(wrapped.seamlessResolution()).toBeUndefined();
    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      })
    ).rejects.toThrow(/requires evaluation data to be a non-empty array/i);
  });

  it('returns currentConfig after applyBestConfig and rejects invalid parameter definitions', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (value: string) => value);

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    wrapped.applyBestConfig(result);
    expect(wrapped.currentConfig()).toEqual({ model: 'a' });

    expect(() =>
      optimize({
        configurationSpace: {
          'bad-name!': param.enum(['a']),
        },
        objectives: ['accuracy'],
      })
    ).toThrow(/valid identifier-like key/i);

    expect(() =>
      optimize({
        configurationSpace: {
          retries: {
            type: 'int',
            min: 0,
            max: 2,
            step: 0.5,
          },
        },
        objectives: ['accuracy'],
      })
    ).toThrow(/step to be an integer/i);
  });

  it('uses the active trial context without wrapping again and clones seamless target metadata', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async () => String(getTrialParam('model', 'missing')));

    await expect(
      TrialContext.run(
        {
          trial_id: 'trial-context',
          trial_number: 1,
          experiment_run_id: 'exp',
          config: { model: 'b' },
          dataset_subset: { indices: [0], total: 1 },
        },
        () => wrapped()
      )
    ).resolves.toBe('b');

    createTraigentOpenAI({
      chat: {
        completions: {
          create: vi.fn(),
        },
      },
    });

    const seamlessWrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'seamless',
        frameworkTargets: ['openai'],
      },
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (value: string) => value);

    seamlessWrapped.applyBestConfig({
      bestConfig: { model: 'best' },
      bestMetrics: { accuracy: 1 },
      trials: [],
      stopReason: 'completed',
      totalCostUsd: 0,
    });

    const firstResolution = seamlessWrapped.seamlessResolution();
    expect(firstResolution).toMatchObject({
      path: 'framework',
      targets: ['openai'],
    });
    firstResolution?.targets?.push('langchain');
    expect(seamlessWrapped.seamlessResolution()?.targets).toEqual(['openai']);
  });

  it('defaults seamless autoOverrideFrameworks to true and uses active wrapped targets', () => {
    createTraigentOpenAI({
      chat: {
        completions: {
          create: vi.fn(),
        },
      },
    });

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'seamless',
      },
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (value: string) => value);

    expect(getOptimizationSpec(wrapped).injection).toMatchObject({
      mode: 'seamless',
      autoOverrideFrameworks: true,
    });

    wrapped.applyBestConfig({
      bestConfig: { model: 'best' },
      bestMetrics: { accuracy: 1 },
      trials: [],
      stopReason: 'completed',
      totalCostUsd: 0,
    });

    expect(wrapped.seamlessResolution()).toMatchObject({
      path: 'framework',
      targets: ['openai'],
    });
  });

  it('can disable seamless auto framework overrides explicitly', async () => {
    vi.stubEnv('TRAIGENT_ENABLE_EXPERIMENTAL_RUNTIME_SEAMLESS', '1');

    createTraigentOpenAI({
      chat: {
        completions: {
          create: vi.fn(),
        },
      },
    });

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'seamless',
        autoOverrideFrameworks: false,
      },
      evaluation: {
        data: [{ input: 'x', output: 'best:x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async function answer(value: string) {
      const model = 'cheap';
      return `${model}:${value}`;
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ model: 'best' });
    expect(wrapped.seamlessResolution()).toMatchObject({
      path: 'runtime',
      experimental: true,
    });
  });
});
