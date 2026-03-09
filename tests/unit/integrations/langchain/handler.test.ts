import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TraigentHandler } from "../../../../src/integrations/langchain/handler.js";

describe("TraigentHandler", () => {
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-09T00:00:00Z"));
    errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.useRealTimers();
    errorSpy.mockRestore();
  });

  it("tracks llm metrics, prefixes measures, and prefers the most specific model pricing", async () => {
    const handler = new TraigentHandler();

    await handler.handleLLMStart({} as never, ["hello"], "run-1");
    vi.advanceTimersByTime(15);
    await handler.handleLLMEnd(
      {
        llmOutput: {
          modelName: "gpt-4o-mini-2025-01-01",
          tokenUsage: {
            promptTokens: 100,
            completionTokens: 50,
            totalTokens: 150,
          },
        },
      } as never,
      "run-1",
    );

    expect(handler.callCount).toBe(1);
    expect(handler.getMetrics()).toMatchObject({
      totalCost: (100 * 0.15 + 50 * 0.6) / 1_000_000,
      totalLatencyMs: 15,
      inputTokens: 100,
      outputTokens: 50,
      totalTokens: 150,
      llmCallCount: 1,
    });
    expect(handler.toMeasuresDict()).toEqual({
      langchain_total_cost: (100 * 0.15 + 50 * 0.6) / 1_000_000,
      langchain_total_latency_ms: 15,
      langchain_input_tokens: 100,
      langchain_output_tokens: 50,
      langchain_total_tokens: 150,
      langchain_llm_call_count: 1,
    });
  });

  it("supports chat start tracking, unknown models, error cleanup, and reset", async () => {
    const handler = new TraigentHandler({ metricPrefix: "custom_" });

    await handler.handleChatModelStart({} as never, [] as never, "chat-1");
    vi.advanceTimersByTime(5);
    await handler.handleLLMEnd(
      {
        llmOutput: {
          modelName: "some-unknown-model",
          tokenUsage: {
            promptTokens: 10,
            completionTokens: 5,
          },
        },
      } as never,
      "chat-1",
    );

    await handler.handleLLMStart({} as never, ["x"], "run-error");
    await handler.handleLLMError(new Error("network"), "run-error");
    await handler.handleLLMEnd({ llmOutput: {} } as never, "after-error");

    expect(errorSpy).toHaveBeenCalledWith(
      "[TraigentHandler] LLM error in run run-error:",
      "network",
    );
    expect(handler.getMetrics()).toMatchObject({
      totalCost: ((10 * 1 + 5 * 3) / 1_000_000),
      inputTokens: 10,
      outputTokens: 5,
      totalTokens: 15,
      llmCallCount: 2,
    });
    expect(handler.toMeasuresDict()).toMatchObject({
      custom_llm_call_count: 2,
    });

    handler.reset();
    expect(handler.callCount).toBe(0);
    expect(handler.getMetrics()).toMatchObject({
      totalCost: 0,
      totalLatencyMs: 0,
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      llmCallCount: 0,
      calls: [],
    });
  });
});
