import { describe, expect, it } from "vitest";

import {
  BusyError,
  CancelledError,
  DatasetMismatchError,
  PayloadTooLargeError,
  TimeoutError,
  TraigentError,
  UnsupportedActionError,
  ValidationError,
  getErrorCode,
  isRetryable,
  isTraigentError,
} from "../../../src/core/errors.js";

describe("Traigent errors", () => {
  it("creates typed errors with stable code and retryability", () => {
    const timeout = new TimeoutError("slow", 25);
    const cancelled = new CancelledError();
    const busy = new BusyError("busy", "trial-1");

    expect(timeout).toBeInstanceOf(TraigentError);
    expect(timeout.message).toContain("25ms");
    expect(timeout.code).toBe("TIMEOUT");
    expect(timeout.retryable).toBe(true);

    expect(cancelled.code).toBe("CANCELLED");
    expect(cancelled.retryable).toBe(false);

    expect(busy.code).toBe("INTERNAL_ERROR");
    expect(busy.retryable).toBe(true);
    expect(busy.currentTrialId).toBe("trial-1");
  });

  it("stores validation-related metadata on richer error types", () => {
    const validation = new ValidationError("invalid", {
      issues: [{ path: ["x"] }],
      summary: "bad input",
    });
    const mismatch = new DatasetMismatchError("mismatch", "expected", "actual");
    const payload = new PayloadTooLargeError("too big", 200, 100);
    const unsupported = new UnsupportedActionError("replay");

    expect(validation.issues).toEqual([{ path: ["x"] }]);
    expect(validation.summary).toBe("bad input");
    expect(mismatch.expectedHash).toBe("expected");
    expect(mismatch.actualHash).toBe("actual");
    expect(payload.size).toBe(200);
    expect(payload.maxSize).toBe(100);
    expect(unsupported.message).toContain("replay");
    expect(unsupported.action).toBe("replay");
  });

  it("classifies typed and untyped errors consistently", () => {
    const validation = new ValidationError("invalid");
    const unknown = new Error("boom");

    expect(isTraigentError(validation)).toBe(true);
    expect(isRetryable(validation)).toBe(false);
    expect(getErrorCode(validation)).toBe("VALIDATION_ERROR");

    expect(isTraigentError(unknown)).toBe(false);
    expect(isRetryable(unknown)).toBe(false);
    expect(getErrorCode(unknown)).toBe("USER_FUNCTION_ERROR");
  });
});
