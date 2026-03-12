import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  autoWrapFrameworkTarget: vi.fn((target) => target),
  createTraigentOpenAI: vi.fn((target) => target),
  deleteOptimizationSession: vi.fn(),
  finalizeOptimizationSession: vi.fn(),
  getOptimizationSessionStatus: vi.fn(),
  getTrialParam: vi.fn((_name, fallback) => fallback),
  listOptimizationSessions: vi.fn(),
  optimize: vi.fn(),
  param: { enum: vi.fn((values) => values) },
}));

vi.mock("openai", () => ({
  default: class MockOpenAI {},
}));

vi.mock("@langchain/openai", () => ({
  ChatOpenAI: class MockChatOpenAI {},
}));

vi.mock("../../../dist/index.js", () => ({
  autoWrapFrameworkTarget: mocks.autoWrapFrameworkTarget,
  deleteOptimizationSession: mocks.deleteOptimizationSession,
  finalizeOptimizationSession: mocks.finalizeOptimizationSession,
  getOptimizationSessionStatus: mocks.getOptimizationSessionStatus,
  getTrialParam: mocks.getTrialParam,
  listOptimizationSessions: mocks.listOptimizationSessions,
  optimize: mocks.optimize,
  param: mocks.param,
}));

vi.mock("../../../dist/integrations/openai/index.js", () => ({
  createTraigentOpenAI: mocks.createTraigentOpenAI,
}));

let collectSessionHelpers: typeof import("../../../examples/core/online-showcase/shared.mjs").collectSessionHelpers;
let summarizeSessionEvidence: typeof import("../../../examples/core/online-showcase/shared.mjs").summarizeSessionEvidence;

beforeAll(async () => {
  const module = await import("../../../examples/core/online-showcase/shared.mjs");
  collectSessionHelpers = module.collectSessionHelpers;
  summarizeSessionEvidence = module.summarizeSessionEvidence;
});

describe("online showcase shared helpers", () => {
  beforeEach(() => {
    vi.unstubAllEnvs();
    Object.values(mocks).forEach((value) => {
      if (typeof value === "function" && "mockReset" in value) {
        value.mockReset();
      }
    });
  });

  it("keeps partial helper evidence when session listing fails", async () => {
    mocks.listOptimizationSessions.mockRejectedValueOnce(
      new Error("listing unavailable"),
    );
    mocks.getOptimizationSessionStatus.mockResolvedValueOnce({
      session_id: "session-1",
      status: "COMPLETED",
      progress: { completed: 5, total: 5, failed: 0 },
      metadata: {
        experiment_id: "exp-123",
        experiment_run_id: "run-456",
      },
    });
    mocks.finalizeOptimizationSession.mockResolvedValueOnce({
      success: true,
      session_id: "session-1",
    });
    mocks.deleteOptimizationSession.mockResolvedValueOnce({
      success: true,
    });
    vi.stubEnv("TRAIGENT_SHOWCASE_DELETE_AFTER_SECTION", "1");
    vi.stubEnv("TRAIGENT_SHOWCASE_FE_URL", "http://localhost:3001");

    const helpers = await collectSessionHelpers("session-1", {
      backendUrl: "http://localhost:5000/api/v1",
      apiKey: "demo-key",
    });
    const summary = summarizeSessionEvidence(helpers);

    expect(helpers.listed).toBeNull();
    expect(helpers.status?.status).toBe("COMPLETED");
    expect(helpers.finalized).toEqual({
      success: true,
      session_id: "session-1",
    });
    expect(helpers.deleted).toEqual({ attempted: true, deleted: true });
    expect(helpers.helperErrors).toEqual([
      { helper: "listed", message: "listing unavailable" },
    ]);

    expect(summary).toMatchObject({
      listedSessionCount: 0,
      listedTotal: null,
      status: "COMPLETED",
      experimentId: "exp-123",
      experimentRunId: "run-456",
      experimentUrl: "http://localhost:3001/experiments/view/exp-123",
      helperErrors: [{ helper: "listed", message: "listing unavailable" }],
    });
  });

  it("returns empty helper errors when all helper calls succeed", async () => {
    mocks.listOptimizationSessions.mockResolvedValueOnce({
      sessions: [{ session_id: "session-2", status: "COMPLETED" }],
      total: 1,
    });
    mocks.getOptimizationSessionStatus.mockResolvedValueOnce({
      session_id: "session-2",
      status: "COMPLETED",
    });
    mocks.finalizeOptimizationSession.mockResolvedValueOnce({
      success: true,
      session_id: "session-2",
    });
    vi.stubEnv("TRAIGENT_SHOWCASE_DELETE_AFTER_SECTION", "0");

    const helpers = await collectSessionHelpers("session-2", undefined);
    const summary = summarizeSessionEvidence(helpers);

    expect(helpers.deleted).toEqual({ attempted: false, deleted: false });
    expect(helpers.helperErrors).toEqual([]);
    expect(summary.helperErrors).toEqual([]);
    expect(summary.listedSessionCount).toBe(1);
    expect(summary.listedTotal).toBe(1);
  });
});
