import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { ValidationError } from "../../../src/core/errors.js";
import { runNativeOptimization } from "../../../src/optimization/native.js";
import { normalizeOptimizationSpec } from "../../../src/optimization/spec.js";
import {
  TrialContext,
  getOptimizationSpec,
  getTrialConfig,
  getTrialParam,
  optimize,
  param,
} from "../../../src/index.js";

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function branin(x: number, y: number): number {
  const a = 1;
  const b = 5.1 / (4 * Math.PI ** 2);
  const c = 5 / Math.PI;
  const r = 6;
  const s = 10;
  const t = 1 / (8 * Math.PI);
  return a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * Math.cos(x) + s;
}

describe("native optimize()", () => {
  it("runs grid optimization and selects the best config from built-in objectives", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["cheap", "best"]),
      },
      objectives: ["accuracy", "cost"],
      evaluation: {
        data: [{ id: 1 }, { id: 2 }],
      },
    })(async (trialConfig) => {
      const model = trialConfig.config.model;
      return {
        metrics: {
          accuracy: model === "best" ? 0.95 : 0.6,
          cost: model === "best" ? 0.8 : 0.1,
        },
        metadata: {
          total: trialConfig.dataset_subset.total,
        },
      };
    });

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 10,
    });

    expect(result.bestConfig).toEqual({ model: "cheap" });
    expect(result.bestMetrics).toEqual({ accuracy: 0.6, cost: 0.1 });
    expect(result.trials).toHaveLength(2);
    expect(result.stopReason).toBe("completed");
    expect(result.totalCostUsd).toBeCloseTo(0.9, 10);
  });

  it("uses explicit objective objects for custom metrics", async () => {
    const wrapped = optimize({
      configurationSpace: {
        prompt: param.enum(["short", "long"]),
      },
      objectives: [{ metric: "quality_score", direction: "maximize" }],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        quality_score: trialConfig.config.prompt === "long" ? 0.9 : 0.5,
      },
    }));

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 10,
    });

    expect(result.bestConfig).toEqual({ prompt: "long" });
    expect(result.bestMetrics).toEqual({ quality_score: 0.9 });
  });

  it("produces deterministic random sampling when randomSeed is fixed", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a", "b", "c"]),
        retries: param.int({ min: 0, max: 2 }),
      },
      objectives: ["cost"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        cost:
          trialConfig.config.model === "a"
            ? 0.1
            : trialConfig.config.model === "b"
              ? 0.2
              : 0.3,
      },
    }));

    const first = await wrapped.optimize({
      mode: "native",
      algorithm: "random",
      maxTrials: 5,
      randomSeed: 42,
    });
    const second = await wrapped.optimize({
      mode: "native",
      algorithm: "random",
      maxTrials: 5,
      randomSeed: 42,
    });

    expect(first.trials.map((trial) => trial.config)).toEqual(
      second.trials.map((trial) => trial.config),
    );
  });

  it("supports log-scaled float grid search", async () => {
    const wrapped = optimize({
      configurationSpace: {
        learning_rate: param.float({
          min: 0.00001,
          max: 0.1,
          scale: "log",
          step: 10,
        }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: Number(trialConfig.config.learning_rate),
      },
    }));

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 10,
    });

    expect(result.trials.map((trial) => trial.config.learning_rate)).toEqual([
      0.00001, 0.0001, 0.001, 0.01, 0.1,
    ]);
    expect(result.stopReason).toBe("completed");
  });

  it("covers stepped grid edge cases and log-scale validation branches", async () => {
    const steppedWrapped = optimize({
      configurationSpace: {
        retries: param.int({ min: 0, max: 5, step: 2 }),
        temperature: param.float({ min: 0, max: 1, step: 0.3 }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy:
          Number(trialConfig.config.retries) +
          Number(trialConfig.config.temperature),
      },
    }));

    const steppedResult = await steppedWrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 100,
    });

    expect(
      new Set(steppedResult.trials.map((trial) => trial.config.retries)),
    ).toEqual(new Set([0, 2, 4, 5]));
    expect(
      new Set(steppedResult.trials.map((trial) => trial.config.temperature)),
    ).toEqual(new Set([0, 0.3, 0.6, 0.9, 1]));

    const invalidLogInt = optimize({
      configurationSpace: {
        retries: param.int({ min: 1, max: 10, scale: "log" }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    await expect(
      invalidLogInt.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 4,
      }),
    ).rejects.toThrow(/log-scaled int parameters to define a multiplicative step/i);

    expect(() =>
      optimize({
        configurationSpace: {
          learningRate: param.float({
            min: 0,
            max: 1,
            scale: "log",
            step: 10,
          }),
        },
        objectives: ["accuracy"],
        evaluation: {
          data: [{ id: 1 }],
        },
      })(async () => ({ metrics: { accuracy: 1 } })),
    ).toThrow(/require min\/max > 0/i);

    expect(() =>
      optimize({
        configurationSpace: {
          learningRate: param.float({
            min: 0.1,
            max: 1,
            scale: "log",
            step: 1,
          }),
        },
        objectives: ["accuracy"],
        evaluation: {
          data: [{ id: 1 }],
        },
      })(async () => ({ metrics: { accuracy: 1 } })),
    ).toThrow(/step to be greater than 1/i);
  });

  it("requires step for float grid search", async () => {
    const wrapped = optimize({
      configurationSpace: {
        temperature: param.float({ min: 0, max: 1, scale: "linear" }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 2,
      }),
    ).rejects.toThrow(/float parameters to define step/i);
  });

  it("throws when budget.maxCostUsd is used without numeric metrics.cost", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      budget: {
        maxCostUsd: 1,
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
      }),
    ).rejects.toThrow(
      /budget\.maxCostUsd requires every trial to return numeric metrics\.cost/i,
    );
  });

  it("throws when optimize() is called on a wrapped function that does not return trial metrics", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => "not-a-trial-result");

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
      }),
    ).rejects.toThrow(ValidationError);
  });

  it("stops early when the accumulated cost budget is reached", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a", "b", "c"]),
      },
      objectives: ["cost"],
      budget: {
        maxCostUsd: 0.25,
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        cost:
          trialConfig.config.model === "a"
            ? 0.1
            : trialConfig.config.model === "b"
              ? 0.2
              : 0.3,
      },
    }));

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 10,
    });

    expect(result.stopReason).toBe("budget");
    expect(result.trials).toHaveLength(2);
    expect(result.totalCostUsd).toBeCloseTo(0.3, 10);
  });

  it("returns a timeout stop reason when a trial exceeds timeoutMs", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["slow"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => {
      await delay(50);
      return {
        metrics: {
          accuracy: 1,
        },
      };
    });

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 1,
      timeoutMs: 10,
    });

    expect(result.stopReason).toBe("timeout");
    expect(result.errorMessage).toMatch(/timeout/i);
    expect(result.trials).toHaveLength(0);
  });

  it("returns an error stop reason when the trial function throws", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => {
      throw new Error("boom");
    });

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 1,
    });

    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toContain("boom");
    expect(result.trials).toHaveLength(0);
  });

  it("uses explicit duration values, falls back from invalid duration, and reports non-Error throws", async () => {
    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    const explicitDuration = await runNativeOptimization(
      async () => ({
        metrics: {
          accuracy: 1,
        },
        duration: 0.25,
      }),
      spec,
      {
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
      },
    );

    expect(explicitDuration.trials[0]?.duration).toBe(0.25);

    const fallbackDuration = await runNativeOptimization(
      async () => {
        await delay(5);
        return {
          metrics: {
            accuracy: 1,
          },
          duration: -1,
        };
      },
      spec,
      {
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
      },
    );

    expect(fallbackDuration.trials[0]?.duration).toBeGreaterThan(0);

    const stringError = await runNativeOptimization(
      async () => {
        throw "native boom";
      },
      spec,
      {
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
      },
    );

    expect(stringError.stopReason).toBe("error");
    expect(stringError.errorMessage).toBe("native boom");
  });

  it("supports sequential bayesian optimization for smooth objectives", async () => {
    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: -5, max: 10 }),
        y: param.float({ min: 0, max: 15 }),
      },
      objectives: [{ metric: "score", direction: "maximize" }],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      const x = Number(trialConfig.config.x);
      const y = Number(trialConfig.config.y);
      return {
        metrics: {
          score: -branin(x, y),
        },
      };
    });

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "bayesian",
      maxTrials: 24,
      randomSeed: 7,
    });

    expect(result.trials).toHaveLength(24);
    expect(result.stopReason).toBe("maxTrials");
    expect(Number(result.bestConfig?.x)).toBeGreaterThanOrEqual(-5);
    expect(Number(result.bestConfig?.x)).toBeLessThanOrEqual(10);
    expect(Number(result.bestConfig?.y)).toBeGreaterThanOrEqual(0);
    expect(Number(result.bestConfig?.y)).toBeLessThanOrEqual(15);
    expect(Number(result.bestMetrics?.score)).toBeGreaterThan(-5);
  });

  it("rejects conditional parameters in native optimization mode", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-3.5", "gpt-4"]),
        max_tokens: param.int({
          min: 256,
          max: 768,
          step: 256,
          conditions: { model: "gpt-4" },
          default: 512,
        }),
        temperature: param.float({
          min: 0.1,
          max: 0.5,
          step: 0.2,
          conditions: { model: "gpt-3.5" },
          default: 0.3,
        }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 10,
      }),
    ).rejects.toThrow(/does not support conditional parameter "max_tokens"/i);
  });

  it("rejects weighted objectives in native optimization mode", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["cheap", "best"]),
      },
      objectives: [{ metric: "quality", direction: "maximize", weight: 2 }],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        quality: 1,
      },
    }));

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "bayesian",
        maxTrials: 8,
        randomSeed: 5,
      }),
    ).rejects.toThrow(/does not support weighted objective "quality"/i);
  });

  it("rejects hybrid-only budgets and constraints in native optimization mode", async () => {
    const constrained = optimize({
      configurationSpace: {
        model: param.enum(["cheap", "best"]),
      },
      objectives: ["accuracy"],
      budget: {
        maxTrials: 4,
        maxWallclockMs: 5_000,
      },
      constraints: {
        structural: [
          {
            require: 'params.model == "cheap"',
          },
        ],
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      constrained.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 2,
      }),
    ).rejects.toThrow(/does not support structural or derived constraints/i);
  });

  it("rejects bayesian trialConcurrency > 1", async () => {
    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: 0, max: 1 }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "bayesian",
        maxTrials: 4,
        trialConcurrency: 2,
      }),
    ).rejects.toThrow(/does not support trialConcurrency > 1/i);
  });

  it("preserves TrialContext isolation across concurrent trials", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a", "b", "c", "d"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      const before = getTrialParam("model");
      await delay(trialConfig.config.model === "a" ? 20 : 5);
      const currentConfig = getTrialConfig();
      const after = getTrialParam("model");
      return {
        metrics: {
          accuracy: before === after ? 1 : 0,
        },
        metadata: {
          before,
          after,
          current: currentConfig.model,
          trialId: TrialContext.getTrialId(),
        },
      };
    });

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 4,
      trialConcurrency: 4,
    });

    expect(result.stopReason).toBe("completed");
    for (const trial of result.trials) {
      expect(trial.metrics.accuracy).toBe(1);
      expect(trial.metadata).toMatchObject({
        before: trial.config.model,
        after: trial.config.model,
        current: trial.config.model,
      });
    }
  });

  it("supports checkpoint/resume with the same final result as an uninterrupted run", async () => {
    const checkpointDir = await mkdtemp(join(tmpdir(), "traigent-checkpoint-"));
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a", "b", "c"]),
        retries: param.int({ min: 0, max: 2 }),
      },
      objectives: ["accuracy", "cost"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      await delay(15);
      return {
        metrics: {
          accuracy:
            Number(trialConfig.config.retries) * 0.2 +
            (trialConfig.config.model === "c" ? 0.6 : 0.3),
          cost: trialConfig.config.model === "c" ? 0.5 : 0.1,
        },
      };
    });

    const controller = new AbortController();
    setTimeout(() => controller.abort(), 60);

    const partial = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 9,
      signal: controller.signal,
      checkpoint: {
        key: "resume-grid",
        dir: checkpointDir,
      },
    });

    const resumed = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 9,
      checkpoint: {
        key: "resume-grid",
        dir: checkpointDir,
        resume: true,
      },
    });

    const uninterrupted = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 9,
    });

    expect(partial.stopReason).toBe("cancelled");
    expect(partial.trials.length).toBeGreaterThan(0);
    expect(resumed.bestConfig).toEqual(uninterrupted.bestConfig);
    expect(resumed.bestMetrics).toEqual(uninterrupted.bestMetrics);
    expect(resumed.stopReason).toEqual(uninterrupted.stopReason);
    expect(resumed.trials.map((trial) => trial.config)).toEqual(
      uninterrupted.trials.map((trial) => trial.config),
    );

    await rm(checkpointDir, { recursive: true, force: true });
  });

  it("stops with plateau when improvements stay below the configured threshold", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a", "b", "c", "d", "e"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 5,
      plateau: {
        window: 2,
        minImprovement: 0.01,
      },
    });

    expect(result.stopReason).toBe("plateau");
    expect(result.trials).toHaveLength(3);
  });

  it("rejects empty evaluation datasets", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
      }),
    ).rejects.toThrow(/evaluation data to be a non-empty array/i);
  });

  it("supports high-level agent optimization with scoringFunction and parameter injection", async () => {
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(["quiet", "loud"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ input: "hello", output: "HELLO!" }],
        scoringFunction: (output, expectedOutput) =>
          output === expectedOutput ? 1 : 0,
      },
      injection: {
        mode: "parameter",
      },
    })(async (input: string, config?: { tone?: string }) =>
      config?.tone === "loud"
        ? `${String(input).toUpperCase()}!`
        : String(input).toUpperCase(),
    );

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ tone: "loud" });
    expect(result.bestMetrics).toEqual(
      expect.objectContaining({ accuracy: 1 }),
    );
  });

  it("supports async loadData and custom input/expected field resolution", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        loadData: async () => [{ question: "hello", answer: "HELLO" }],
        inputField: "question",
        expectedField: "answer",
        scoringFunction: (output, expectedOutput) =>
          output === expectedOutput ? 1 : 0,
      },
    })(async (input: string) => input.toUpperCase());

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 1,
    });

    expect(result.bestMetrics).toEqual(
      expect.objectContaining({ accuracy: 1 }),
    );
  });

  it("allows metricFunctions and customEvaluator without expected output fields", async () => {
    const metricWrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: [{ metric: "quality", direction: "maximize" }],
      evaluation: {
        data: [{ input: "hello" }],
        metricFunctions: {
          quality: (output) => (String(output).length > 0 ? 1 : 0),
        },
      },
    })(async (input: string) => input.toUpperCase());

    const metricResult = await metricWrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 1,
    });

    expect(metricResult.bestMetrics).toEqual(
      expect.objectContaining({ quality: 1 }),
    );

    const evaluatorWrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ input: "hello" }],
        customEvaluator: async ({ output }) => ({
          accuracy: String(output) === "HELLO" ? 1 : 0,
        }),
      },
    })(async (input: string) => input.toUpperCase());

    const evaluatorResult = await evaluatorWrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 1,
    });

    expect(evaluatorResult.bestMetrics).toEqual(
      expect.objectContaining({ accuracy: 1 }),
    );
  });

  it("rejects invalid native optimize() runtime options", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      runNativeOptimization(
        async () => ({
          metrics: {
            accuracy: 1,
          },
        }),
        getOptimizationSpec(wrapped)!,
        {
        mode: "hybrid" as never,
        algorithm: "grid",
        maxTrials: 1,
        },
      ),
    ).rejects.toThrow(/native mode only accepts mode: "native"/i);

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
        randomSeed: -1,
      }),
    ).rejects.toThrow(/randomSeed must be a non-negative integer/i);

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
        timeoutMs: 0,
      }),
    ).rejects.toThrow(/timeoutMs must be a positive integer/i);

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
        trialConcurrency: 0,
      }),
    ).rejects.toThrow(/trialConcurrency must be a positive integer/i);

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
        plateau: {
          window: 0,
          minImprovement: 0,
        },
      }),
    ).rejects.toThrow(/plateau\.window must be a positive integer/i);

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
        plateau: {
          window: 1,
          minImprovement: -0.1,
        },
      }),
    ).rejects.toThrow(/plateau\.minImprovement must be a finite number >= 0/i);

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
        checkpoint: {
          key: "",
        },
      }),
    ).rejects.toThrow(/checkpoint\.key must be non-empty/i);

    await expect(
      wrapped.optimize({
        mode: "native",
        algorithm: "grid",
        maxTrials: 1,
        checkpoint: {
          key: "ok",
          dir: "",
        },
      }),
    ).rejects.toThrow(/checkpoint\.dir must be non-empty/i);
  });

  it("rejects invalid metrics returned from runNativeOptimization and handles pre-aborted signals", async () => {
    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    await expect(
      runNativeOptimization(
        async () =>
          ({
            metrics: {
              accuracy: Number.NaN,
            },
          }) as never,
        spec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
        },
      ),
    ).rejects.toThrow(/trial metrics are invalid/i);

    const controller = new AbortController();
    controller.abort();

    const result = await runNativeOptimization(
      async () => ({
        metrics: {
          accuracy: 1,
        },
      }),
      spec,
      {
        mode: "native",
        algorithm: "grid",
        maxTrials: 2,
        signal: controller.signal,
      },
    );

    expect(result.stopReason).toBe("cancelled");
    expect(result.trials).toHaveLength(0);
  });

  it("validates native options and evaluation prerequisites eagerly", async () => {
    const validSpec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        validSpec,
        undefined as never,
      ),
    ).rejects.toThrow(/options are required/i);

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        validSpec,
        {
          mode: "hybrid" as never,
          algorithm: "grid",
          maxTrials: 1,
        },
      ),
    ).rejects.toThrow(/native mode only accepts mode: "native"/i);

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        validSpec,
        {
          mode: "native",
          algorithm: "optuna" as never,
          maxTrials: 1,
        },
      ),
    ).rejects.toThrow(/only supports algorithm "grid", "random", or "bayesian"/i);

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        validSpec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 0,
        },
      ),
    ).rejects.toThrow(/requires maxTrials to be a positive integer/i);

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        validSpec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
          randomSeed: -1,
        },
      ),
    ).rejects.toThrow(/randomSeed must be a non-negative integer/i);

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        validSpec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
          timeoutMs: 0,
        },
      ),
    ).rejects.toThrow(/timeoutMs must be a positive integer/i);

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        validSpec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
          trialConcurrency: 0,
        },
      ),
    ).rejects.toThrow(/trialConcurrency must be a positive integer/i);

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        validSpec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
          plateau: {
            window: 0,
            minImprovement: 0,
          },
        },
      ),
    ).rejects.toThrow(/plateau\.window must be a positive integer/i);

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        validSpec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
          plateau: {
            window: 1,
            minImprovement: -1,
          },
        },
      ),
    ).rejects.toThrow(/plateau\.minImprovement must be a finite number >= 0/i);

    const missingEvaluationSpec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
    });

    await expect(
      runNativeOptimization(
        async () => ({ metrics: { accuracy: 1 } }),
        missingEvaluationSpec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
        },
      ),
    ).rejects.toThrow(/requires spec\.evaluation\.data or spec\.evaluation\.loadData/i);
  });

  it("rejects invalid native trial result shapes and objective metrics", async () => {
    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    await expect(
      runNativeOptimization(
        async () => "bad-result" as never,
        spec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
        },
      ),
    ).rejects.toThrow(/must resolve to an object containing metrics/i);

    await expect(
      runNativeOptimization(
        async () => ({
          metrics: {
            cost: 0.1,
          },
        }),
        spec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
        },
      ),
    ).rejects.toThrow(/missing numeric metric "accuracy"/i);
  });

  it("stops bayesian search with plateau when objective improvements flatten out", async () => {
    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: 0, max: 1 }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "bayesian",
      maxTrials: 10,
      randomSeed: 3,
      plateau: {
        window: 2,
        minImprovement: 0.01,
      },
    });

    expect(result.stopReason).toBe("plateau");
    expect(result.trials.length).toBeGreaterThanOrEqual(3);
  });

  it("merges defaultConfig into native trial configs and supports banded objectives", async () => {
    const seenTemperatures: number[] = [];
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
      },
      objectives: [
        {
          metric: "response_length",
          band: { low: 120, high: 180 },
        },
      ],
      defaultConfig: {
        temperature: 0.7,
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      seenTemperatures.push(Number(trialConfig.config.temperature));
      return {
        metrics: {
          response_length: trialConfig.config.model === "accurate" ? 150 : 90,
        },
      };
    });

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 2,
    });

    expect(seenTemperatures).toEqual([0.7, 0.7]);
    expect(result.bestConfig).toEqual({
      temperature: 0.7,
      model: "accurate",
    });
  });

  it("prefers in-band results over values above the target range for banded objectives", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["balanced", "verbose"]),
      },
      objectives: [
        {
          metric: "response_length",
          band: { low: 120, high: 180 },
        },
      ],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        response_length:
          trialConfig.config.model === "balanced" ? 150 : 240,
      },
    }));

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ model: "balanced" });
  });

  it("supports conditional sampling paths when runNativeOptimization() is invoked directly", async () => {
    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
        maxTokens: param.int({
          min: 64,
          max: 128,
          step: 64,
          conditions: { model: "accurate" },
          default: 64,
        }),
        temperature: param.float({
          min: 0.1,
          max: 0.3,
          step: 0.2,
          conditions: { model: "cheap" },
          default: 0.1,
        }),
      },
      objectives: ["accuracy"],
      evaluation: {
        loadData: async () => [{ id: 1 }, { id: 2 }],
      },
    });

    const result = await runNativeOptimization(
      async (trialConfig) => ({
        metrics: {
          accuracy:
            trialConfig.config.model === "accurate" &&
            trialConfig.config.maxTokens === 128
              ? 1
              : 0.5,
        },
      }),
      spec,
      {
        mode: "native",
        algorithm: "grid",
        maxTrials: 10,
      },
    );

    expect(result.trials.map((trial) => trial.config)).toEqual([
      { model: "cheap", maxTokens: 64, temperature: 0.1 },
      { model: "cheap", maxTokens: 64, temperature: 0.3 },
      { model: "accurate", maxTokens: 64, temperature: 0.1 },
      { model: "accurate", maxTokens: 128, temperature: 0.1 },
    ]);
    expect(result.bestConfig).toEqual({
      model: "accurate",
      maxTokens: 128,
      temperature: 0.1,
    });
  });

  it("covers random conditional sampling and exhausts a discrete space without duplicates", async () => {
    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
        retries: param.int({ min: 0, max: 1, step: 1 }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    const result = await runNativeOptimization(
      async (trialConfig) => ({
        metrics: {
          accuracy:
            trialConfig.config.model === "accurate" &&
            trialConfig.config.retries === 1
              ? 1
              : 0.2,
        },
      }),
      spec,
      {
        mode: "native",
        algorithm: "random",
        maxTrials: 10,
        randomSeed: 5,
      },
    );

    expect(result.trials).toHaveLength(4);
    expect(
      new Set(result.trials.map((trial) => JSON.stringify(trial.config))).size,
    ).toBe(4);
    expect(result.stopReason).toBe("completed");
  });

  it("supports continuous random sampling spaces without discrete exhaustion tracking", async () => {
    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        temperature: param.float({ min: 0, max: 1 }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    const result = await runNativeOptimization(
      async (trialConfig) => ({
        metrics: {
          accuracy: 1 - Math.abs(Number(trialConfig.config.temperature) - 0.5),
        },
      }),
      spec,
      {
        mode: "native",
        algorithm: "random",
        maxTrials: 3,
        randomSeed: 7,
      },
    );

    expect(result.trials).toHaveLength(3);
    expect(result.stopReason).toBe("maxTrials");
  });

  it("rejects invalid metrics returned from a trial function", async () => {
    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    await expect(
      runNativeOptimization(
        async () =>
          ({
            metrics: {
              accuracy: Number.NaN,
            },
          }) as never,
        spec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
        },
      ),
    ).rejects.toThrow(/trial metrics are invalid/i);
  });

  it("returns cancelled immediately when the optimization signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();

    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a", "b"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    const result = await runNativeOptimization(
      async () => ({
        metrics: {
          accuracy: 1,
        },
      }),
      spec,
      {
        mode: "native",
        algorithm: "grid",
        maxTrials: 2,
        signal: controller.signal,
      },
    );

    expect(result.stopReason).toBe("cancelled");
    expect(result.errorMessage).toMatch(/cancelled/i);
    expect(result.trials).toHaveLength(0);
  });

  it("rejects incompatible checkpoint state during resume", async () => {
    const checkpointDir = await mkdtemp(join(tmpdir(), "traigent-native-checkpoint-"));
    const checkpointKey = "native-grid";
    const checkpointPath = join(
      checkpointDir,
      `${createHash("sha256").update(checkpointKey).digest("hex")}.json`,
    );

    await writeFile(
      checkpointPath,
      JSON.stringify({
        version: 1,
        algorithm: "grid",
        specHash: "wrong",
        optionsHash: "wrong",
        experimentRunId: "native_resume",
        completedTrials: [],
        totalCostUsd: 0,
      }),
      "utf8",
    );

    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    await expect(
      runNativeOptimization(
        async () => ({
          metrics: {
            accuracy: 1,
          },
        }),
        spec,
        {
          mode: "native",
          algorithm: "grid",
          maxTrials: 1,
          checkpoint: {
            key: checkpointKey,
            dir: checkpointDir,
            resume: true,
          },
        },
      ),
    ).rejects.toThrow(/checkpoint does not match/i);

    await rm(checkpointDir, { recursive: true, force: true });
  });

  it("stops bayesian search on budget and on discrete exhaustiveness", async () => {
    const budgetSpec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
        temperature: param.float({ min: 0.1, max: 0.9 }),
      },
      objectives: ["accuracy"],
      budget: {
        maxCostUsd: 0.11,
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    const budgetResult = await runNativeOptimization(
      async (trialConfig) => ({
        metrics: {
          accuracy: trialConfig.config.model === "accurate" ? 0.9 : 0.5,
          cost: trialConfig.config.model === "accurate" ? 0.2 : 0.1,
        },
      }),
      budgetSpec,
      {
        mode: "native",
        algorithm: "bayesian",
        maxTrials: 8,
        randomSeed: 3,
      },
    );

    expect(budgetResult.stopReason).toBe("budget");

    const exhaustiveSpec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    const exhaustiveResult = await runNativeOptimization(
      async (trialConfig) => ({
        metrics: {
          accuracy: trialConfig.config.model === "accurate" ? 1 : 0.5,
        },
      }),
      exhaustiveSpec,
      {
        mode: "native",
        algorithm: "bayesian",
        maxTrials: 8,
        randomSeed: 1,
      },
    );

    expect(exhaustiveResult.stopReason).toBe("completed");
    expect(exhaustiveResult.trials).toHaveLength(2);
  });

  it("resumes bayesian search from checkpointed sampler state", async () => {
    const checkpointDir = await mkdtemp(join(tmpdir(), "traigent-bayes-checkpoint-"));
    const controller = new AbortController();
    let seen = 0;

    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: 0.1, max: 1 }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      seen += 1;
      if (seen === 2) {
        controller.abort();
      }
      return {
        metrics: {
          accuracy: Number(trialConfig.config.x),
        },
      };
    });

    const first = await wrapped.optimize({
      mode: "native",
      algorithm: "bayesian",
      maxTrials: 5,
      randomSeed: 4,
      signal: controller.signal,
      checkpoint: {
        key: "bayes-resume",
        dir: checkpointDir,
      },
    });

    expect(first.stopReason).toBe("cancelled");

    const resumed = await wrapped.optimize({
      mode: "native",
      algorithm: "bayesian",
      maxTrials: 5,
      randomSeed: 4,
      checkpoint: {
        key: "bayes-resume",
        dir: checkpointDir,
        resume: true,
      },
    });

    expect(resumed.trials).toHaveLength(5);
    await rm(checkpointDir, { recursive: true, force: true });
  });

  it("handles bayesian cancellation and bayesian trial errors", async () => {
    const cancelledSignal = new AbortController();
    cancelledSignal.abort();

    const spec = normalizeOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    });

    const cancelled = await runNativeOptimization(
      async () => ({
        metrics: {
          accuracy: 1,
        },
      }),
      spec,
      {
        mode: "native",
        algorithm: "bayesian",
        maxTrials: 3,
        signal: cancelledSignal.signal,
      },
    );

    expect(cancelled.stopReason).toBe("cancelled");

    const errored = await runNativeOptimization(
      async () => {
        throw new Error("bayes boom");
      },
      spec,
      {
        mode: "native",
        algorithm: "bayesian",
        maxTrials: 3,
      },
    );

    expect(errored.stopReason).toBe("error");
    expect(errored.errorMessage).toContain("bayes boom");
  });
});
