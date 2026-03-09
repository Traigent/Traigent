import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  TrialContext,
  getTrialConfig,
  getTrialParam,
} from "../../../src/core/context.js";
import {
  normalizeBackendApiBase,
  objectiveScoreValue,
  serializeSessionConfigurationSpace,
} from "../../../src/optimization/hybrid.js";
import { getOptimizationSpec, optimize, param } from "../../../src/index.js";

function jsonResponse(status: number, payload: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  };
}

function textResponse(status: number, body: string) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      throw new Error("no json body");
    },
    text: async () => body,
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

describe("hybrid optimize()", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("normalizes backend URLs for the interactive session API", () => {
    expect(normalizeBackendApiBase("http://localhost:5000")).toBe(
      "http://localhost:5000/api/v1",
    );
    expect(normalizeBackendApiBase("http://localhost:5000/api/v1")).toBe(
      "http://localhost:5000/api/v1",
    );
    expect(normalizeBackendApiBase("http://localhost:5000/api/v1/")).toBe(
      "http://localhost:5000/api/v1",
    );
    expect(() =>
      normalizeBackendApiBase("http://localhost:5000/custom/path"),
    ).toThrow(/backendUrl must be a backend origin or the \/api\/v1 base URL/i);
  });

  it("serializes spec parameters to the typed backend session configuration shape", () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini", "gpt-4o"]),
        retries: param.int({ min: 1, max: 5, step: 2, scale: "log" }),
        temperature: param.float({ min: 0.1, max: 1, step: 0.3 }),
      },
      objectives: ["accuracy"],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(
      serializeSessionConfigurationSpace(getOptimizationSpec(wrapped)!),
    ).toEqual({
      model: { type: "categorical", choices: ["gpt-4o-mini", "gpt-4o"] },
      retries: { type: "int", low: 1, high: 5, step: 2, log: true },
      temperature: { type: "float", low: 0.1, high: 1, step: 0.3 },
    });
  });

  it("serializes conditional parameters with required default fallbacks", () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
        maxTokens: param.int({
          min: 64,
          max: 256,
          step: 64,
          conditions: { model: "accurate" },
          default: 64,
        }),
      },
      objectives: ["accuracy"],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(
      serializeSessionConfigurationSpace(getOptimizationSpec(wrapped)!),
    ).toEqual({
      model: { type: "categorical", choices: ["cheap", "accurate"] },
      maxTokens: {
        type: "int",
        low: 64,
        high: 256,
        step: 64,
        conditions: { model: "accurate" },
        default: 64,
      },
    });
  });

  it("encodes tuple-valued categorical domains for the typed backend session API", () => {
    const wrapped = optimize({
      configurationSpace: {
        retrievalPair: param.enum([
          ["dense", "rerank"],
          ["bm25", "none"],
        ]),
      },
      objectives: ["accuracy"],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(
      serializeSessionConfigurationSpace(getOptimizationSpec(wrapped)!),
    ).toEqual({
      retrievalPair: {
        type: "categorical",
        choices: ["choice_0", "choice_1"],
        value_map: {
          choice_0: ["dense", "rerank"],
          choice_1: ["bm25", "none"],
        },
      },
    });
  });

  it("matches complex categorical defaults using stable equality when encoding tuple-like domains", () => {
    const wrapped = optimize({
      configurationSpace: {
        mode: param.enum(["structured"]),
        retrievalPair: {
          type: "enum",
          values: [{ provider: "dense", stage: "rerank" }],
          conditions: { mode: "structured" },
          default: { stage: "rerank", provider: "dense" },
        } as any,
      },
      objectives: ["accuracy"],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(
      serializeSessionConfigurationSpace(getOptimizationSpec(wrapped)!),
    ).toEqual({
      mode: { type: "categorical", choices: ["structured"] },
      retrievalPair: {
        type: "categorical",
        choices: ["choice_0"],
        value_map: {
          choice_0: { provider: "dense", stage: "rerank" },
        },
        conditions: { mode: "structured" },
        default: "choice_0",
      },
    });
  });

  it("includes hybrid budget and constraints in the typed session create payload", async () => {
    vi.stubEnv("TRAIGENT_BACKEND_URL", "http://localhost:5000");
    vi.stubEnv("TRAIGENT_API_KEY", "env-key");

    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-budget-constraints",
          status: "active",
          optimization_strategy: { algorithm: "optuna" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          reason: "budget_exhausted",
          stop_reason: "budget_exhausted",
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-budget-constraints",
          best_config: {},
          best_metrics: {},
          total_trials: 0,
          successful_trials: 0,
          total_duration: 0,
          cost_savings: 0,
          stop_reason: "budget_exhausted",
          metadata: {},
        }),
      );

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
      },
      objectives: [{ metric: "quality", direction: "maximize", weight: 2 }],
      budget: {
        maxCostUsd: 2,
        maxTrials: 8,
        maxWallclockMs: 45_000,
      },
      constraints: {
        structural: [
          {
            when: 'params.model == "accurate"',
            then: "True",
          },
        ],
        derived: [
          {
            require: "metrics.quality >= 0.8",
          },
        ],
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        quality: 0.9,
        cost: 0.3,
      },
    }));

    await wrapped.optimize({
      algorithm: "optuna",
      maxTrials: 12,
    });

    const createPayload = JSON.parse(
      String(fetchMock.mock.calls[0]?.[1]?.body),
    );
    expect(createPayload.budget).toEqual({
      max_cost_usd: 2,
      max_trials: 8,
      max_wallclock_ms: 45_000,
    });
    expect(createPayload.constraints).toEqual({
      structural: [
        {
          when: 'params.model == "accurate"',
          then: "True",
        },
      ],
      derived: [
        {
          require: "metrics.quality >= 0.8",
        },
      ],
    });
  });

  it("serializes banded objectives, defaultConfig, and promotionPolicy in the create payload", async () => {
    vi.stubEnv("TRAIGENT_BACKEND_URL", "http://localhost:5000");
    vi.stubEnv("TRAIGENT_API_KEY", "env-key");

    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-policy",
          status: "active",
          optimization_strategy: { algorithm: "optuna" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          reason: "search_complete",
          stop_reason: "search_complete",
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-policy",
          best_config: {},
          best_metrics: {},
          total_trials: 0,
          successful_trials: 0,
          total_duration: 0,
          cost_savings: 0,
          stop_reason: "search_complete",
          metadata: {},
        }),
      );

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
      },
      objectives: [
        {
          metric: "response_length",
          band: { low: 120, high: 180 },
          weight: 2,
        },
      ],
      defaultConfig: {
        temperature: 0.6,
      },
      promotionPolicy: {
        dominance: "epsilon_pareto",
        alpha: 0.05,
        adjust: "BH",
        minEffect: {
          response_length: 0,
        },
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        response_length: 150,
      },
    }));

    await wrapped.optimize({
      algorithm: "optuna",
      maxTrials: 3,
    });

    const createPayload = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(createPayload.objectives).toEqual([
      {
        metric: "response_length",
        band: { low: 120, high: 180 },
        test: "TOST",
        weight: 2,
      },
    ]);
    expect(createPayload.default_config).toEqual({ temperature: 0.6 });
    expect(createPayload.promotion_policy).toEqual({
      dominance: "epsilon_pareto",
      alpha: 0.05,
      adjust: "BH",
      min_effect: {
        response_length: 0,
      },
    });
  });

  it("defaults optimize() to hybrid mode, binds TrialContext, and returns backend finalization data", async () => {
    vi.stubEnv("TRAIGENT_BACKEND_URL", "http://localhost:5000");
    vi.stubEnv("TRAIGENT_API_KEY", "env-key");

    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-123",
          status: "active",
          optimization_strategy: { algorithm: "optuna" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: {
            trial_id: "trial-001",
            session_id: "session-123",
            trial_number: 1,
            config: { model: "gpt-4o", temperature: 0.3 },
            dataset_subset: {
              indices: [0, 2],
              selection_strategy: "diverse_sampling",
              confidence_level: 0.8,
              estimated_representativeness: 0.75,
              metadata: {},
            },
            exploration_type: "exploration",
            priority: 1,
          },
          should_continue: true,
          session_status: "active",
        }),
      )
      .mockResolvedValueOnce(jsonResponse(201, { success: true }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          reason: "Max trials reached",
          stop_reason: "max_trials_reached",
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-123",
          best_config: { model: "gpt-4o", temperature: 0.3 },
          best_metrics: { accuracy: 0.91, cost: 0.2 },
          total_trials: 1,
          successful_trials: 1,
          total_duration: 0.2,
          cost_savings: 0.1,
          stop_reason: "max_trials_reached",
          metadata: { finalized_by: "backend" },
        }),
      );

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini", "gpt-4o"]),
        temperature: param.float({ min: 0, max: 1, step: 0.1 }),
      },
      objectives: ["accuracy", "cost"],
      evaluation: {
        data: [{ id: 1 }, { id: 2 }, { id: 3 }],
      },
    })(async (trialConfig) => {
      expect(TrialContext.getTrialId()).toBe("trial-001");
      expect(getTrialParam("model")).toBe("gpt-4o");
      expect(getTrialConfig()).toEqual({
        model: "gpt-4o",
        temperature: 0.3,
      });
      expect(trialConfig.dataset_subset).toEqual({
        indices: [0, 2],
        total: 3,
      });

      return {
        metrics: {
          accuracy: 0.91,
          cost: 0.2,
        },
        metadata: {
          observedTrial: trialConfig.trial_id,
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: "optuna",
      maxTrials: 5,
    });

    expect(result.mode).toBe("hybrid");
    expect(result.sessionId).toBe("session-123");
    expect(result.stopReason).toBe("maxTrials");
    expect(result.bestConfig).toEqual({ model: "gpt-4o", temperature: 0.3 });
    expect(result.bestMetrics).toEqual({ accuracy: 0.91, cost: 0.2 });
    expect(result.trials).toHaveLength(1);
    expect(result.totalCostUsd).toBeCloseTo(0.2, 10);
    expect(result.metadata).toMatchObject({
      backendReason: "max_trials_reached",
      optimizationStrategy: { algorithm: "optuna" },
      finalization: { finalized_by: "backend" },
    });

    expect(fetchMock).toHaveBeenCalledTimes(5);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "http://localhost:5000/api/v1/sessions",
    );
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({
      method: "POST",
      headers: expect.objectContaining({
        Authorization: "Bearer env-key",
      }),
    });

    const createPayload = JSON.parse(
      String(fetchMock.mock.calls[0]?.[1]?.body),
    );
    expect(createPayload).toMatchObject({
      function_name: "anonymous_js_trial",
      objectives: ["accuracy", "cost"],
      dataset_metadata: { size: 3 },
      optimization_strategy: { algorithm: "optuna" },
      metadata: { sdk: "js", mode: "hybrid" },
    });
    expect(createPayload.configuration_space).toEqual({
      model: { type: "categorical", choices: ["gpt-4o-mini", "gpt-4o"] },
      temperature: { type: "float", low: 0, high: 1, step: 0.1 },
    });

    const nextPayload = JSON.parse(String(fetchMock.mock.calls[3]?.[1]?.body));
    expect(nextPayload.previous_results).toEqual([
      {
        session_id: "session-123",
        trial_id: "trial-001",
        metrics: { accuracy: 0.91, cost: 0.2 },
        duration: expect.any(Number),
        status: "completed",
        error_message: null,
        metadata: {
          observedTrial: "trial-001",
        },
      },
    ]);

    const finalizePayload = JSON.parse(
      String(fetchMock.mock.calls[4]?.[1]?.body),
    );
    expect(finalizePayload).toEqual({
      session_id: "session-123",
      include_full_history: false,
      metadata: {
        sdk: "js",
        mode: "hybrid",
      },
    });
  });

  it("round-trips tuple-valued backend suggestions into the local JS trial context", async () => {
    vi.stubEnv("TRAIGENT_BACKEND_URL", "http://localhost:5000");
    vi.stubEnv("TRAIGENT_API_KEY", "env-key");

    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-tuples",
          status: "active",
          optimization_strategy: { algorithm: "optuna" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: {
            trial_id: "trial-tuple-1",
            session_id: "session-tuples",
            trial_number: 1,
            config: { retrievalPair: ["dense", "rerank"] },
            dataset_subset: {
              indices: [0],
              metadata: {},
            },
          },
          should_continue: true,
          session_status: "active",
        }),
      )
      .mockResolvedValueOnce(jsonResponse(201, { success: true }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          stop_reason: "search_complete",
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-tuples",
          best_config: { retrievalPair: ["dense", "rerank"] },
          best_metrics: { accuracy: 1 },
          total_trials: 1,
          successful_trials: 1,
          total_duration: 0.01,
          cost_savings: 0,
          stop_reason: "search_complete",
          metadata: {},
        }),
      );

    const wrapped = optimize({
      configurationSpace: {
        retrievalPair: param.enum([
          ["dense", "rerank"],
          ["bm25", "none"],
        ]),
      },
      objectives: ["accuracy"],
      evaluation: { data: [{ id: 1 }] },
    })(async () => {
      expect(getTrialParam("retrievalPair")).toEqual(["dense", "rerank"]);
      expect(getTrialConfig()).toEqual({
        retrievalPair: ["dense", "rerank"],
      });

      return {
        metrics: { accuracy: 1 },
      };
    });

    const result = await wrapped.optimize({
      algorithm: "optuna",
      maxTrials: 1,
    });

    expect(result.bestConfig).toEqual({
      retrievalPair: ["dense", "rerank"],
    });
  });

  it("rejects invalid hybrid trial metrics before result submission", async () => {
    vi.stubEnv("TRAIGENT_BACKEND_URL", "http://localhost:5000");
    vi.stubEnv("TRAIGENT_API_KEY", "env-key");

    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-invalid-metrics",
          status: "active",
          optimization_strategy: { algorithm: "optuna" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: {
            trial_id: "trial-invalid-1",
            session_id: "session-invalid-metrics",
            trial_number: 1,
            config: { model: "gpt-4o" },
            dataset_subset: {
              indices: [0],
              metadata: {},
            },
          },
          should_continue: true,
          session_status: "active",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-invalid-metrics",
          best_config: {},
          best_metrics: {},
          total_trials: 0,
          successful_trials: 0,
          total_duration: 0,
          cost_savings: 0,
          stop_reason: "finalized",
          metadata: {},
        }),
      );

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o"]),
      },
      objectives: ["accuracy"],
      evaluation: { data: [{ id: 1 }] },
    })(async () => ({
      metrics: {
        accuracy: Number.NaN,
      },
    }));

    await expect(
      wrapped.optimize({
        algorithm: "optuna",
        maxTrials: 1,
      }),
    ).rejects.toThrow(/trial metrics are invalid/i);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(String(fetchMock.mock.calls[2]?.[0])).toContain("/finalize");
  });

  it("keeps banded objective scoring aligned for representative cases", () => {
    const objective = {
      kind: "banded" as const,
      metric: "response_length",
      band: { low: 120, high: 180 },
      bandTest: "TOST" as const,
      bandAlpha: 0.05,
      weight: 1,
    };

    expect(objectiveScoreValue(150, objective)).toBe(0);
    expect(objectiveScoreValue(100, objective)).toBe(-20);
    expect(objectiveScoreValue(220, objective)).toBe(-40);
  });

  it("falls back to TRAIGENT_API_URL and forwards optional session metadata", async () => {
    vi.stubEnv("TRAIGENT_API_URL", "http://localhost:5000");
    vi.stubEnv("TRAIGENT_API_KEY", "env-key");

    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-metadata",
          status: "active",
          optimization_strategy: { algorithm: "optuna", sampler: "tpe" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          reason: "search complete",
          stop_reason: "search_complete",
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-metadata",
          stop_reason: "search_complete",
          metadata: { finalized_by: "backend" },
        }),
      );

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }, { id: 2 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    const result = await wrapped.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 2,
      userId: "user-123",
      billingTier: "enterprise",
      optimizationStrategy: { sampler: "tpe", startupTrials: 5 },
      datasetMetadata: { source: "unit-test" },
    });

    expect(result.stopReason).toBe("completed");
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "http://localhost:5000/api/v1/sessions",
    );
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({
      method: "POST",
      headers: expect.objectContaining({
        Authorization: "Bearer env-key",
      }),
    });

    const createPayload = JSON.parse(
      String(fetchMock.mock.calls[0]?.[1]?.body),
    );
    expect(createPayload).toMatchObject({
      function_name: "anonymous_js_trial",
      user_id: "user-123",
      billing_tier: "enterprise",
      dataset_metadata: {
        size: 2,
        source: "unit-test",
      },
      optimization_strategy: {
        algorithm: "optuna",
        sampler: "tpe",
        startup_trials: 5,
      },
    });

    expect(fetchMock.mock.calls[2]?.[0]).toBe(
      "http://localhost:5000/api/v1/sessions/session-metadata/finalize",
    );
    expect(fetchMock.mock.calls[2]?.[1]).toMatchObject({
      method: "POST",
      headers: expect.objectContaining({
        Authorization: "Bearer env-key",
      }),
    });
  });

  it("falls back to finalization stop_reason when next-trial omits it", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-finalize-stop",
          status: "active",
          optimization_strategy: { algorithm: "optuna" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-finalize-stop",
          stop_reason: "max_trials_reached",
          metadata: { finalized_by: "backend" },
        }),
      );

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    const result = await wrapped.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 1,
      backendUrl: "http://localhost:5000",
      apiKey: "key",
    });

    expect(result.stopReason).toBe("maxTrials");
    expect(result.metadata).toMatchObject({
      backendReason: "max_trials_reached",
      finalization: { finalized_by: "backend" },
    });
  });

  it("rejects dataset metadata size that does not match the loaded evaluation size", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }, { id: 2 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    await expect(
      wrapped.optimize({
        mode: "hybrid",
        algorithm: "optuna",
        maxTrials: 1,
        backendUrl: "http://localhost:5000",
        apiKey: "key",
        datasetMetadata: { size: 3 },
      }),
    ).rejects.toThrow(
      /datasetMetadata\.size must match the loaded evaluation dataset size/i,
    );
  });

  it("rejects dataset metadata size when it is not a positive number", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }, { id: 2 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    await expect(
      wrapped.optimize({
        mode: "hybrid",
        algorithm: "optuna",
        maxTrials: 1,
        backendUrl: "http://localhost:5000",
        apiKey: "key",
        datasetMetadata: { size: 0 },
      }),
    ).rejects.toThrow(
      /datasetMetadata\.size must be a positive number when provided/i,
    );
  });

  it("names the missing backend configuration when hybrid mode is the default", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    await expect(
      wrapped.optimize({
        algorithm: "optuna",
        maxTrials: 1,
      }),
    ).rejects.toThrow(
      /requires backendUrl, TRAIGENT_BACKEND_URL, or TRAIGENT_API_URL/i,
    );
  });

  it("returns an error result when env-backed TRAIGENT_API_URL requests fail", async () => {
    vi.stubEnv("TRAIGENT_API_URL", "http://localhost:5000/api/v1/");
    vi.stubEnv("TRAIGENT_API_KEY", "env-key");
    fetchMock.mockResolvedValueOnce(textResponse(503, "backend unavailable"));

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    const result = await wrapped.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 1,
    });

    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toMatch(/http 503/i);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "http://localhost:5000/api/v1/sessions",
    );
  });

  it("fails fast when the backend exposes the legacy TraiGent /sessions contract", async () => {
    vi.stubEnv("TRAIGENT_API_URL", "http://localhost:5000");
    vi.stubEnv("TRAIGENT_API_KEY", "env-key");
    fetchMock.mockResolvedValueOnce(
      jsonResponse(400, {
        error:
          "Missing required fields: problem_statement, dataset, search_space, optimization_config",
        message:
          "Missing required fields: problem_statement, dataset, search_space, optimization_config",
        success: false,
      }),
    );

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    await expect(
      wrapped.optimize({
        mode: "hybrid",
        algorithm: "optuna",
        maxTrials: 1,
      }),
    ).rejects.toThrow(/pointed at a legacy TraiGent \/sessions API/i);
  });

  it("serializes weighted objectives and still rejects native-only options in hybrid mode", async () => {
    const weighted = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: [{ metric: "quality", direction: "maximize", weight: 2 }],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({ metrics: { quality: 1 } }));

    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-weighted",
          status: "active",
          optimization_strategy: { algorithm: "optuna" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          stop_reason: "search_complete",
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-weighted",
          stop_reason: "search_complete",
        }),
      );

    await weighted.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 1,
      backendUrl: "http://localhost:5000",
      apiKey: "key",
    });

    const weightedPayload = JSON.parse(
      String(fetchMock.mock.calls[0]?.[1]?.body),
    );
    expect(weightedPayload.objectives).toEqual([
      { metric: "quality", direction: "maximize", weight: 2 },
    ]);

    const conditional = optimize({
      configurationSpace: {
        model: param.enum(["a", "b"]),
        max_tokens: param.int({
          min: 128,
          max: 512,
          step: 128,
          conditions: { model: "b" },
          default: 256,
        }),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    fetchMock.mockReset();
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-conditional",
          status: "active",
          optimization_strategy: { algorithm: "optuna" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          stop_reason: "search_complete",
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-conditional",
          stop_reason: "search_complete",
        }),
      );

    await conditional.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 1,
      backendUrl: "http://localhost:5000",
      apiKey: "key",
    });

    const conditionalPayload = JSON.parse(
      String(fetchMock.mock.calls[0]?.[1]?.body),
    );
    expect(conditionalPayload.configuration_space.max_tokens).toEqual({
      type: "int",
      low: 128,
      high: 512,
      step: 128,
      conditions: { model: "b" },
      default: 256,
    });

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    await expect(
      wrapped.optimize({
        mode: "hybrid",
        algorithm: "optuna",
        maxTrials: 1,
        backendUrl: "http://localhost:5000",
        apiKey: "key",
        trialConcurrency: 2,
      } as never),
    ).rejects.toThrow(/does not support native option "trialConcurrency"/i);
  });

  it("maps a timed-out local trial to a failed submission and timeout stop reason", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-timeout",
          status: "active",
          optimization_strategy: {},
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: {
            trial_id: "trial-timeout",
            session_id: "session-timeout",
            trial_number: 1,
            config: { model: "slow" },
            dataset_subset: {
              indices: [0],
              selection_strategy: "random",
              confidence_level: 0.5,
              estimated_representativeness: 0.5,
              metadata: {},
            },
          },
          should_continue: true,
          session_status: "active",
        }),
      )
      .mockResolvedValueOnce(jsonResponse(201, { success: true }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-timeout",
          metadata: { finalized_by: "backend" },
        }),
      );

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
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 1,
      backendUrl: "http://localhost:5000",
      apiKey: "key",
      timeoutMs: 10,
    });

    expect(result.stopReason).toBe("timeout");
    expect(result.trials).toHaveLength(0);

    const submitPayload = JSON.parse(
      String(fetchMock.mock.calls[2]?.[1]?.body),
    );
    expect(submitPayload).toMatchObject({
      session_id: "session-timeout",
      trial_id: "trial-timeout",
      status: "failed",
      error_message: expect.stringMatching(/timeout/i),
      metadata: {
        timeout: true,
      },
    });
  });

  it("submits cancellation as a failed trial, finalizes the session, and stops locally", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-cancel",
          status: "active",
          optimization_strategy: {},
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: {
            trial_id: "trial-cancel",
            session_id: "session-cancel",
            trial_number: 1,
            config: { model: "slow" },
            dataset_subset: {
              indices: [0],
              selection_strategy: "random",
              confidence_level: 0.5,
              estimated_representativeness: 0.5,
              metadata: {},
            },
          },
          should_continue: true,
          session_status: "active",
        }),
      )
      .mockResolvedValueOnce(jsonResponse(201, { success: true }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-cancel",
          metadata: { finalized_by: "backend" },
        }),
      );

    const controller = new AbortController();
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["slow"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => {
      await delay(100);
      return {
        metrics: {
          accuracy: 1,
        },
      };
    });

    const promise = wrapped.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 2,
      backendUrl: "http://localhost:5000",
      apiKey: "key",
      signal: controller.signal,
    });

    setTimeout(() => controller.abort(), 10);

    const result = await promise;

    expect(result.stopReason).toBe("cancelled");
    expect(result.sessionId).toBe("session-cancel");
    expect(fetchMock).toHaveBeenCalledTimes(4);

    const submitPayload = JSON.parse(
      String(fetchMock.mock.calls[2]?.[1]?.body),
    );
    expect(submitPayload).toMatchObject({
      session_id: "session-cancel",
      trial_id: "trial-cancel",
      status: "failed",
      metadata: {
        cancelled: true,
      },
    });
    expect(fetchMock.mock.calls[3]?.[0]).toBe(
      "http://localhost:5000/api/v1/sessions/session-cancel/finalize",
    );
  });

  it("returns an error result for backend HTTP failures and still finalizes an active session", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-error",
          status: "active",
          optimization_strategy: {},
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: {
            trial_id: "trial-error",
            session_id: "session-error",
            trial_number: 1,
            config: { model: "a" },
            dataset_subset: {
              indices: [0],
              selection_strategy: "random",
              confidence_level: 0.5,
              estimated_representativeness: 0.5,
              metadata: {},
            },
          },
          should_continue: true,
          session_status: "active",
        }),
      )
      .mockResolvedValueOnce(textResponse(503, "backend unavailable"))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-error",
          metadata: { finalized_by: "backend" },
        }),
      );

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
        accuracy: 0.8,
      },
    }));

    const result = await wrapped.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 2,
      backendUrl: "http://localhost:5000",
      apiKey: "key",
    });

    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toMatch(/http 503/i);
    expect(result.sessionId).toBe("session-error");
    expect(result.trials).toHaveLength(1);
    expect(result.metadata).toMatchObject({
      finalization: { finalized_by: "backend" },
    });
    expect(fetchMock.mock.calls[3]?.[0]).toBe(
      "http://localhost:5000/api/v1/sessions/session-error/finalize",
    );
  });

  it("returns an error result for create-session auth failures", async () => {
    fetchMock.mockResolvedValueOnce(textResponse(403, "forbidden"));

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

    const result = await wrapped.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 1,
      backendUrl: "http://localhost:5000",
      apiKey: "key",
    });

    expect(result.stopReason).toBe("error");
    expect(result.sessionId).toBeUndefined();
    expect(result.errorMessage).toMatch(/http 403/i);
  });

  it("names the missing api key when hybrid becomes the default mode", async () => {
    vi.stubEnv("TRAIGENT_API_KEY", "");
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
      wrapped.optimize({
        algorithm: "optuna",
        maxTrials: 1,
        backendUrl: "http://localhost:5000",
      }),
    ).rejects.toThrow(/requires apiKey or TRAIGENT_API_KEY/i);
  });

  it("returns an error result when finalization fails after a terminal next-trial response", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-finalize-error",
          status: "active",
          optimization_strategy: {},
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          session_status: "completed",
        }),
      )
      .mockRejectedValueOnce(new Error("finalize timeout"));

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

    const result = await wrapped.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 1,
      backendUrl: "http://localhost:5000",
      apiKey: "key",
    });

    expect(result.stopReason).toBe("completed");
    expect(result.errorMessage).toBeUndefined();
    expect(result.metadata).toMatchObject({
      finalizeError: "finalize timeout",
    });
  });

  it("rejects explicit objectives whose direction does not match backend inference", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: [
        { metric: "quality_score", direction: "minimize", weight: 3 },
      ],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        quality_score: 0.9,
      },
    }));

    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(201, {
          session_id: "session-objective",
          status: "active",
          optimization_strategy: { algorithm: "optuna" },
          metadata: {},
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          suggestion: null,
          should_continue: false,
          stop_reason: "search_complete",
          session_status: "completed",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: "session-objective",
          stop_reason: "search_complete",
        }),
      );

    await wrapped.optimize({
      mode: "hybrid",
      algorithm: "optuna",
      maxTrials: 1,
      backendUrl: "http://localhost:5000",
      apiKey: "key",
    });

    const createPayload = JSON.parse(
      String(fetchMock.mock.calls[0]?.[1]?.body),
    );
    expect(createPayload.objectives).toEqual([
      { metric: "quality_score", direction: "minimize", weight: 3 },
    ]);
  });
});
