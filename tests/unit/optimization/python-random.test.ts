import { describe, expect, it } from "vitest";

import { PythonRandom } from "../../../src/optimization/python-random.js";

describe("PythonRandom", () => {
  it("restores deterministic state from serialize()", () => {
    const first = new PythonRandom(42);
    first.random();
    first.random();
    const serialized = first.serialize();

    const restored = new PythonRandom(serialized);
    expect(restored.random()).toBe(first.random());
    expect(restored.randint(1, 10)).toBe(first.randint(1, 10));
  });

  it("handles zero-seed and single-choice branches", () => {
    const random = new PythonRandom(0);
    expect(random.randBelow(1)).toBe(0);
    expect(random.choice(["only"])).toBe("only");
  });

  it("rejects invalid serialized states and invalid bounds", () => {
    expect(
      () =>
        new PythonRandom({
          index: 0,
          state: [1, 2, 3],
        } as never),
    ).toThrow(/invalid serialized python random state/i);

    const random = new PythonRandom(7);
    expect(() => random.randBelow(0)).toThrow(/positive safe integer/i);
    expect(() => random.choice([])).toThrow(/non-empty array/i);
    expect(() => random.randint(3, 2)).toThrow(/max >= min/i);
  });
});
