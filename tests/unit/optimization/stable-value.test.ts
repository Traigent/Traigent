import { describe, expect, it } from "vitest";

import { ValidationError } from "../../../src/core/errors.js";
import {
  stableValueEquals,
  stableValueKey,
} from "../../../src/optimization/stable-value.js";

describe("stable value helpers", () => {
  it("treats structurally equal objects as equal regardless of key order", () => {
    expect(
      stableValueEquals(
        { b: 2, a: { y: true, x: [1, 2, 3] } },
        { a: { x: [1, 2, 3], y: true }, b: 2 },
      ),
    ).toBe(true);
  });

  it("distinguishes special numeric values and undefined payloads", () => {
    expect(stableValueKey(Number.NaN)).not.toBe(stableValueKey(null));
    expect(stableValueKey(Number.POSITIVE_INFINITY)).not.toBe(
      stableValueKey(Number.NEGATIVE_INFINITY),
    );
    expect(stableValueKey(-0)).not.toBe(stableValueKey(0));
    expect(stableValueKey(undefined)).toContain("undefined");
  });

  it("supports bigint serialization", () => {
    expect(stableValueKey(12n)).toContain("bigint");
  });

  it("rejects unsupported function and symbol values", () => {
    expect(() => stableValueKey(Symbol("x"))).toThrow(ValidationError);
    expect(() => stableValueKey(() => "x")).toThrow(ValidationError);
  });

  it("rejects circular arrays and objects", () => {
    const circularArray: unknown[] = [];
    circularArray.push(circularArray);

    const circularObject: Record<string, unknown> = {};
    circularObject["self"] = circularObject;

    expect(() => stableValueKey(circularArray)).toThrow(/circular/i);
    expect(() => stableValueKey(circularObject)).toThrow(/circular/i);
  });

  it("rejects non-plain objects", () => {
    class CustomValue {
      constructor(readonly value: string) {}
    }

    expect(() => stableValueKey(new CustomValue("x"))).toThrow(
      /unsupported object value/i,
    );
  });
});
