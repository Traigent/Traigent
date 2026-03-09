import { ValidationError } from "../core/errors.js";

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }

  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function canonicalizeNumber(value: number): number | Record<string, string> {
  if (Number.isNaN(value)) {
    return { __number__: "NaN" };
  }
  if (value === Number.POSITIVE_INFINITY) {
    return { __number__: "Infinity" };
  }
  if (value === Number.NEGATIVE_INFINITY) {
    return { __number__: "-Infinity" };
  }
  if (Object.is(value, -0)) {
    return { __number__: "-0" };
  }
  return value;
}

function canonicalizeValue(
  value: unknown,
  seen: WeakSet<object>,
  path: string,
): unknown {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return value;
  }

  if (typeof value === "number") {
    return canonicalizeNumber(value);
  }

  if (typeof value === "undefined") {
    return { __type__: "undefined" };
  }

  if (typeof value === "bigint") {
    return { __type__: "bigint", value: value.toString() };
  }

  if (typeof value === "function" || typeof value === "symbol") {
    throw new ValidationError(
      `Unsupported ${typeof value} in optimization value at ${path}.`,
    );
  }

  if (Array.isArray(value)) {
    if (seen.has(value)) {
      throw new ValidationError(
        `Circular optimization values are not supported at ${path}.`,
      );
    }
    seen.add(value);
    const canonical = value.map((entry, index) =>
      canonicalizeValue(entry, seen, `${path}[${index}]`),
    );
    seen.delete(value);
    return canonical;
  }

  if (!isPlainObject(value)) {
    throw new ValidationError(
      `Unsupported object value in optimization metadata at ${path}.`,
    );
  }

  if (seen.has(value)) {
    throw new ValidationError(
      `Circular optimization values are not supported at ${path}.`,
    );
  }

  seen.add(value);
  const canonical = Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonicalizeValue(value[key], seen, `${path}.${key}`)]),
  );
  seen.delete(value);
  return canonical;
}

export function stableValueKey(value: unknown): string {
  return JSON.stringify(canonicalizeValue(value, new WeakSet(), "$"));
}

export function stableValueEquals(left: unknown, right: unknown): boolean {
  return stableValueKey(left) === stableValueKey(right);
}
