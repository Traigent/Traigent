import { describe, expect, it } from "vitest";

import { TrialContext } from "../../../src/core/context.js";
import {
  createAgentTrialFunction,
  invokeFunctionWithConfig,
  resolveEvaluationRows,
} from "../../../src/optimization/agent.js";
import { getOptimizationSpec, optimize, param } from "../../../src/optimization/spec.js";
import type { TrialConfig } from "../../../src/dtos/trial.js";

function createTrialConfig(
  config: Record<string, unknown>,
  overrides: Partial<TrialConfig> = {},
): TrialConfig {
  return {
    trial_id: "trial-agent",
    trial_number: 1,
    experiment_run_id: "exp-agent",
    config,
    dataset_subset: {
      indices: [],
      total: 2,
      ...(overrides.dataset_subset ?? {}),
    },
    ...overrides,
  };
}

describe("agent optimization helpers", () => {
  it("resolves evaluation rows from data or loadData and fails when missing", async () => {
    await expect(
      resolveEvaluationRows({
        evaluation: {
          data: [{ input: "x" }],
        },
      }),
    ).resolves.toEqual([{ input: "x" }]);

    await expect(
      resolveEvaluationRows({
        evaluation: {
          loadData: async () => [{ input: "y" }],
        },
      }),
    ).resolves.toEqual([{ input: "y" }]);

    await expect(resolveEvaluationRows({ evaluation: undefined })).rejects.toThrow(
      /requires evaluation\.data or evaluation\.loadData/i,
    );
  });

  it("supports parameter injection for zero and one-argument functions and rejects invalid config args", () => {
    expect(
      invokeFunctionWithConfig(
        (config?: Record<string, unknown>) => config,
        undefined,
        [],
        { model: "best" },
        "parameter",
      ),
    ).toEqual({ model: "best" });

    expect(
      invokeFunctionWithConfig(
        (input: string, config?: Record<string, unknown>) => [input, config],
        undefined,
        ["hello"],
        { tone: "friendly" },
        "parameter",
      ),
    ).toEqual(["hello", { tone: "friendly" }]);

    expect(() =>
      invokeFunctionWithConfig(
        () => "never",
        undefined,
        ["hello", "bad-config"],
        { tone: "friendly" },
        "parameter",
      ),
    ).toThrow(/second argument to be an object config/i);
  });

  it("aggregates example metrics using default and explicit strategies", async () => {
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(["quiet", "loud"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [
          {
            prompt: "hi",
            expected: "ignored",
            score: 0.25,
            quality: 0.4,
            cost: 0.1,
            inputCost: 0.03,
            outputCost: 0.07,
            latency: 3,
          },
          {
            prompt: "bye",
            expected: "ignored",
            score: 0.75,
            quality: 0.9,
            cost: 0.2,
            inputCost: 0.08,
            outputCost: 0.12,
            latency: 1,
          },
        ],
        inputField: "prompt",
        expectedField: "expected",
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
          quality: "max",
          cost: "sum",
          input_cost: "sum",
          output_cost: "sum",
          total_cost: "sum",
          latency: "min",
          default: "mean",
        },
      },
    })(async (input: string) => input);

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
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

  it("supports python-style customEvaluator(agentFn, config, row)", async () => {
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(["quiet", "loud"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [
          { input: "hello", output: "HELLO", score: 1 },
          { input: "bye", output: "BYE", score: 0.5 },
        ],
        customEvaluator: async (agentFn, config, row) => {
          const record = row as { input: string; output: string; score: number };
          const prediction = await agentFn(record.input);
          return {
            accuracy: prediction === record.output ? record.score : 0,
            latency: config.tone === "loud" ? 2 : 1,
          };
        },
      },
      injection: {
        mode: "parameter",
      },
      execution: {
        mode: "native",
      },
    })(async (input: string, config?: { tone?: string }) =>
      config?.tone === "loud"
        ? `${String(input).toUpperCase()}!`
        : String(input).toUpperCase(),
    );

    const result = await wrapped.optimize({
      algorithm: "grid",
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ tone: "quiet" });
    expect(result.bestMetrics).toMatchObject({
      accuracy: 0.75,
      latency: 1,
    });
  });

  it("uses inline_rows when provided and preserves scoringFunction as the primary metric", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ input: "unused", output: "UNUSED" }],
        scoringFunction: (output, expectedOutput) =>
          output === expectedOutput ? 1 : 0,
        metricFunctions: {
          accuracy: () => 0,
          auxiliary: (_output, _expectedOutput, _runtimeMetrics, row) =>
            (row as { score: number }).score,
        },
      },
      execution: {
        mode: "native",
      },
    })(async (input: string) => input.toUpperCase());

    const normalizedSpec = getOptimizationSpec(wrapped)!;
    const trialFn = createAgentTrialFunction(
      async (input: string) => input.toUpperCase(),
      normalizedSpec,
      [{ input: "unused", output: "UNUSED" }],
    );

    const inlineRows = [
      { input: "first", output: "FIRST", score: 0.2 },
      { input: "second", output: "SECOND", score: 0.8 },
    ];

    const result = await TrialContext.run(
      createTrialConfig(
        { model: "a" },
        {
          dataset_subset: {
            indices: [0],
            total: 2,
            inline_rows: inlineRows,
          },
        },
      ),
      () =>
        trialFn(
          createTrialConfig(
            { model: "a" },
            {
              dataset_subset: {
                indices: [0],
                total: 2,
                inline_rows: inlineRows,
              },
            },
          ),
        ),
    );

    expect(result.metrics).toMatchObject({
      accuracy: 1,
      auxiliary: 0.5,
    });
  });

  it("rejects out-of-range dataset indices instead of evaluating undefined rows", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ input: "hello", output: "HELLO" }],
        scoringFunction: (output, expectedOutput) =>
          output === expectedOutput ? 1 : 0,
      },
      execution: {
        mode: "native",
      },
    })(async (input: string) => input.toUpperCase());

    const normalizedSpec = getOptimizationSpec(wrapped)!;
    const trialFn = createAgentTrialFunction(
      async (input: string) => input.toUpperCase(),
      normalizedSpec,
      [{ input: "hello", output: "HELLO" }],
    );

    await expect(
      TrialContext.run(
        createTrialConfig({
          model: "a",
        }, {
          dataset_subset: {
            indices: [3],
            total: 1,
          },
        }),
        () =>
          trialFn(
            createTrialConfig(
              { model: "a" },
              {
                dataset_subset: {
                  indices: [3],
                  total: 1,
                },
              },
            ),
          ),
      ),
    ).rejects.toThrow(/out of bounds/i);
  });

  it("rejects missing input and expected fields when the evaluation contract requires them", async () => {
    const missingInput = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ question: "hello", output: "HELLO" }],
        inputField: "input",
        scoringFunction: (output, expectedOutput) =>
          output === expectedOutput ? 1 : 0,
      },
      execution: {
        mode: "native",
      },
    })(async (input: string) => input.toUpperCase());

    await expect(
      missingInput.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
      }),
    ).rejects.toThrow(/missing input field "input"/i);

    const missingExpected = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ input: "hello" }],
        scoringFunction: () => 1,
      },
      execution: {
        mode: "native",
      },
    })(async (input: string) => input.toUpperCase());

    await expect(
      missingExpected.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
      }),
    ).rejects.toThrow(/must include an expected output field/i);
  });

  it("supports context-style custom evaluators and rejects missing objective metrics", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["cheap"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ input: "hello", output: "HELLO" }],
        customEvaluator: async ({
          output,
          expectedOutput,
          runtimeMetrics,
          config,
        }) => ({
          accuracy: output === expectedOutput ? 1 : 0,
          cost: runtimeMetrics.cost ?? null,
          model_seen: config.model === "cheap" ? 1 : 0,
        }),
      },
      execution: {
        mode: "native",
      },
    })(async (input: string) => input.toUpperCase());

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 1,
    });

    expect(result.bestMetrics).toMatchObject({
      accuracy: 1,
      model_seen: 1,
    });

    const missingObjective = optimize({
      configurationSpace: {
        model: param.enum(["cheap"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ input: "hello", output: "HELLO" }],
        metricFunctions: {
          auxiliary: () => 0.5,
        },
      },
      execution: {
        mode: "native",
      },
    })(async (input: string) => input.toUpperCase());

    await expect(
      missingObjective.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
      }),
    ).rejects.toThrow(/objective metric "accuracy"/i);
  });
});
