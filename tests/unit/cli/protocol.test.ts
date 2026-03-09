import { describe, expect, it } from "vitest";

import {
  CLIRequestSchema,
  CancelRequestSchema,
  ErrorResponseSchema,
  PingRequestSchema,
  PROTOCOL_VERSION,
  RunTrialRequestSchema,
  ShutdownRequestSchema,
  createErrorResponse,
  createSuccessResponse,
  parseRequest,
  serializeResponse,
} from "../../../src/cli/protocol.js";

describe("CLI protocol helpers", () => {
  it("parses typed requests for all supported actions", () => {
    const runTrial = parseRequest(
      JSON.stringify({
        version: PROTOCOL_VERSION,
        request_id: "req-run",
        action: "run_trial",
        payload: {
          trial_id: "trial-1",
          trial_number: 1,
          experiment_run_id: "exp-1",
          config: { model: "gpt-4o-mini" },
          dataset_subset: { indices: [0], total: 1 },
        },
      }),
    );

    const ping = parseRequest(
      JSON.stringify({
        version: PROTOCOL_VERSION,
        request_id: "req-ping",
        action: "ping",
        payload: {},
      }),
    );

    const shutdown = parseRequest(
      JSON.stringify({
        version: PROTOCOL_VERSION,
        request_id: "req-stop",
        action: "shutdown",
        payload: {},
      }),
    );

    const cancel = parseRequest(
      JSON.stringify({
        version: PROTOCOL_VERSION,
        request_id: "req-cancel",
        action: "cancel",
        payload: { trial_id: "trial-1" },
      }),
    );

    expect(RunTrialRequestSchema.parse(runTrial).payload.trial_id).toBe("trial-1");
    expect(PingRequestSchema.parse(ping).action).toBe("ping");
    expect(ShutdownRequestSchema.parse(shutdown).action).toBe("shutdown");
    expect(CancelRequestSchema.parse(cancel).payload?.trial_id).toBe("trial-1");
    expect(CLIRequestSchema.parse(cancel).request_id).toBe("req-cancel");
  });

  it("creates and serializes success and error responses", () => {
    const success = createSuccessResponse("req-1", {
      metrics: { accuracy: 1 },
    });
    const fromError = createErrorResponse("req-2", new Error("boom"), {
      errorCode: "INTERNAL_ERROR",
      retryable: true,
    });
    const fromString = createErrorResponse("req-3", "bad request");

    expect(JSON.parse(serializeResponse(success))).toEqual(success);

    const parsedError = ErrorResponseSchema.parse(fromError);
    expect(parsedError.payload.error).toBe("boom");
    expect(parsedError.payload.error_code).toBe("INTERNAL_ERROR");
    expect(parsedError.payload.retryable).toBe(true);
    expect(parsedError.payload.stack).toContain("Error: boom");

    expect(ErrorResponseSchema.parse(fromString).payload).toMatchObject({
      error: "bad request",
      retryable: false,
    });
  });

  it("rejects malformed requests through schema validation", () => {
    expect(() =>
      parseRequest(
        JSON.stringify({
          version: PROTOCOL_VERSION,
          request_id: "",
          action: "ping",
        }),
      ),
    ).toThrow();

    expect(() =>
      parseRequest(
        JSON.stringify({
          version: "2.0",
          request_id: "req-1",
          action: "unknown",
          payload: {},
        }),
      ),
    ).toThrow();
  });
});
