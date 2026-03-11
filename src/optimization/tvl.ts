import { readFile } from "node:fs/promises";

import { parse as parseYaml } from "yaml";

import { ValidationError } from "../core/errors.js";
import { normalizeOptimizationSpec } from "./spec.js";
import type {
  BandTarget,
  LoadedTvlOptimizationSpec,
  ObjectiveInput,
  OptimizationBudget,
  OptimizationConstraints,
  PromotionPolicy,
  OptimizationSpec,
  ParameterDefinition,
  TvlLoadOptions,
} from "./types.js";

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function toFiniteNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ValidationError(`${field} must be a finite number.`);
  }
  return value;
}

function toPositiveInteger(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new ValidationError(`${field} must be a positive integer.`);
  }
  return value;
}

function parseTvarDomain(
  name: string,
  type: string,
  domain: unknown,
): ParameterDefinition {
  const domainObject = isPlainObject(domain) ? domain : {};
  const scale =
    domainObject["log"] === true || domainObject["scale"] === "log"
      ? "log"
      : "linear";

  if (type.startsWith("tuple")) {
    const values = Array.isArray(domain) ? domain : domainObject["values"];
    if (!Array.isArray(values) || values.length === 0) {
      throw new ValidationError(`tvars.${name}.domain.values must be a non-empty array for tuple variables.`);
    }
    for (const [index, value] of values.entries()) {
      if (!Array.isArray(value) || value.length === 0) {
        throw new ValidationError(
          `tvars.${name}.domain.values[${index}] must be a non-empty tuple-like array.`,
        );
      }
    }
    return {
      type: "enum",
      values: values.map((value) => [...(value as unknown[])]),
    };
  }

  if (type.startsWith("callable")) {
    const values = Array.isArray(domain) ? domain : domainObject["values"];
    if (!Array.isArray(values) || values.length === 0) {
      throw new ValidationError(
        `tvars.${name}.domain.values must be a non-empty array for callable variables.`,
      );
    }
    for (const [index, value] of values.entries()) {
      if (typeof value !== "string" || value.trim().length === 0) {
        throw new ValidationError(
          `tvars.${name}.domain.values[${index}] must be a non-empty string for callable variables.`,
        );
      }
    }
    return {
      type: "enum",
      values: values.map((value) => String(value)),
    };
  }

  switch (type) {
    case "bool":
      return {
        type: "enum",
        values: [false, true],
      };
    case "enum":
    case "enum[str]": {
      const values = domainObject["values"];
      if (!Array.isArray(values) || values.length === 0) {
        throw new ValidationError(`tvars.${name}.domain.values must be a non-empty array.`);
      }
      return {
        type: "enum",
        values: [...values],
      };
    }
    case "int":
    case "float": {
      const range = domainObject["range"];
      if (!Array.isArray(range) || range.length !== 2) {
        throw new ValidationError(`tvars.${name}.domain.range must contain exactly two numeric values.`);
      }
      const min = toFiniteNumber(range[0], `tvars.${name}.domain.range[0]`);
      const max = toFiniteNumber(range[1], `tvars.${name}.domain.range[1]`);
      const step =
        domainObject["step"] === undefined
          ? undefined
          : toFiniteNumber(domainObject["step"], `tvars.${name}.domain.step`);

      if (type === "int") {
        if (!Number.isInteger(min) || !Number.isInteger(max)) {
          throw new ValidationError(`tvars.${name}.domain.range must use integers for int variables.`);
        }
        if (step !== undefined && !Number.isInteger(step)) {
          throw new ValidationError(`tvars.${name}.domain.step must be an integer for int variables.`);
        }
        return {
          type: "int",
          min,
          max,
          scale,
          ...(step !== undefined ? { step } : {}),
        };
      }

      return {
        type: "float",
        min,
        max,
        scale,
        ...(step !== undefined ? { step } : {}),
      };
    }
    default:
      throw new ValidationError(
        `TVL variable "${name}" type "${type}" is not supported by the JS SDK.`,
      );
  }
}

function parseTvars(raw: unknown): Record<string, ParameterDefinition> {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new ValidationError("TVL tvars must be a non-empty array.");
  }

  return Object.fromEntries(
    raw.map((entry, index) => {
      if (!isPlainObject(entry)) {
        throw new ValidationError(`tvars[${index}] must be an object.`);
      }

      const name = entry["name"];
      const type = entry["type"];
      if (typeof name !== "string" || name.trim().length === 0) {
        throw new ValidationError(`tvars[${index}].name must be a non-empty string.`);
      }
      if (typeof type !== "string" || type.trim().length === 0) {
        throw new ValidationError(`tvars[${index}].type must be a non-empty string.`);
      }

      return [
        name,
        parseTvarDomain(name, type.trim().toLowerCase(), entry["domain"]),
      ] as const;
    }),
  );
}

function parseObjectives(raw: unknown): ObjectiveInput[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new ValidationError("TVL objectives must be a non-empty array.");
  }

  return raw.map((entry, index) => {
    if (!isPlainObject(entry)) {
      throw new ValidationError(`objectives[${index}] must be an object.`);
    }

    const metric = entry["name"];
    const band = entry["band"];
    if (typeof metric !== "string" || metric.trim().length === 0) {
      throw new ValidationError(`objectives[${index}].name must be a non-empty string.`);
    }

    if (band !== undefined) {
      if (!isPlainObject(band)) {
        throw new ValidationError(`objectives[${index}].band must be an object.`);
      }

      const normalizedBand = normalizeBandTarget(
        band,
        `objectives[${index}].band`,
      );
      const alpha =
        entry["alpha"] === undefined
          ? 0.05
          : toFiniteNumber(entry["alpha"], `objectives[${index}].alpha`);
      if (alpha <= 0 || alpha >= 1) {
        throw new ValidationError(`objectives[${index}].alpha must be in (0, 1).`);
      }
      const test = entry["test"] === undefined ? "TOST" : entry["test"];
      if (test !== "TOST") {
        throw new ValidationError(`objectives[${index}].test must be "TOST".`);
      }

      return {
        metric: metric.trim(),
        band: normalizedBand,
        test: "TOST",
        alpha,
        ...(entry["weight"] !== undefined
          ? { weight: toFiniteNumber(entry["weight"], `objectives[${index}].weight`) }
          : {}),
      };
    }

    const direction = entry["direction"];
    if (direction !== "maximize" && direction !== "minimize") {
      throw new ValidationError(
        `objectives[${index}].direction must be "maximize" or "minimize".`,
      );
    }

    return {
      metric: metric.trim(),
      direction,
      ...(entry["weight"] !== undefined
        ? { weight: toFiniteNumber(entry["weight"], `objectives[${index}].weight`) }
        : {}),
    };
  });
}

function normalizeBandTarget(
  band: Record<string, unknown>,
  field: string,
): BandTarget {
  if (band["low"] !== undefined || band["high"] !== undefined) {
    const low = toFiniteNumber(band["low"], `${field}.low`);
    const high = toFiniteNumber(band["high"], `${field}.high`);
    if (low >= high) {
      throw new ValidationError(`${field}.low must be less than ${field}.high.`);
    }
    return { low, high };
  }

  if (band["center"] !== undefined || band["tol"] !== undefined) {
    const center = toFiniteNumber(band["center"], `${field}.center`);
    const tol = toFiniteNumber(band["tol"], `${field}.tol`);
    if (tol <= 0) {
      throw new ValidationError(`${field}.tol must be positive.`);
    }
    return { center, tol };
  }

  throw new ValidationError(`${field} must provide low/high or center/tol.`);
}

function parseStructuralConstraints(
  raw: unknown,
): OptimizationConstraints["structural"] {
  if (raw === undefined) {
    return undefined;
  }
  if (!Array.isArray(raw)) {
    throw new ValidationError("constraints.structural must be an array when provided.");
  }

  return raw.map((entry, index) => {
    if (!isPlainObject(entry)) {
      throw new ValidationError(`constraints.structural[${index}] must be an object.`);
    }

    const errorMessage =
      typeof entry["error_message"] === "string"
        ? entry["error_message"]
        : typeof entry["errorMessage"] === "string"
          ? entry["errorMessage"]
          : undefined;

    if (typeof entry["require"] === "string") {
      return {
        ...(typeof entry["id"] === "string" ? { id: entry["id"] } : {}),
        require: entry["require"],
        ...(errorMessage ? { errorMessage } : {}),
      };
    }

    if (typeof entry["when"] === "string" && typeof entry["then"] === "string") {
      return {
        ...(typeof entry["id"] === "string" ? { id: entry["id"] } : {}),
        when: entry["when"],
        then: entry["then"],
        ...(errorMessage ? { errorMessage } : {}),
      };
    }

    if (typeof entry["expr"] === "string") {
      return {
        ...(typeof entry["id"] === "string" ? { id: entry["id"] } : {}),
        require: entry["expr"],
        ...(errorMessage ? { errorMessage } : {}),
      };
    }

    throw new ValidationError(
      `constraints.structural[${index}] must provide require, expr, or when/then.`,
    );
  });
}

function parseDerivedConstraints(
  raw: unknown,
): OptimizationConstraints["derived"] {
  if (raw === undefined) {
    return undefined;
  }
  if (!Array.isArray(raw)) {
    throw new ValidationError("constraints.derived must be an array when provided.");
  }

  return raw.map((entry, index) => {
    if (!isPlainObject(entry)) {
      throw new ValidationError(`constraints.derived[${index}] must be an object.`);
    }

    const requireExpression =
      typeof entry["require"] === "string"
        ? entry["require"]
        : typeof entry["expr"] === "string"
          ? entry["expr"]
          : undefined;
    if (!requireExpression) {
      throw new ValidationError(
        `constraints.derived[${index}] must provide require or expr.`,
      );
    }

    return {
      ...(typeof entry["id"] === "string" ? { id: entry["id"] } : {}),
      require: requireExpression,
      ...((typeof entry["error_message"] === "string"
        ? { errorMessage: entry["error_message"] }
        : typeof entry["errorMessage"] === "string"
          ? { errorMessage: entry["errorMessage"] }
          : {}) as Record<string, string>),
    };
  });
}

function parseConstraints(raw: unknown): OptimizationConstraints | undefined {
  if (raw === undefined) {
    return undefined;
  }
  if (!isPlainObject(raw)) {
    throw new ValidationError("constraints must be an object when provided.");
  }

  const structural = parseStructuralConstraints(raw["structural"]);
  const derived = parseDerivedConstraints(raw["derived"]);

  if ((structural?.length ?? 0) === 0 && (derived?.length ?? 0) === 0) {
    return undefined;
  }

  return {
    ...(structural ? { structural } : {}),
    ...(derived ? { derived } : {}),
  };
}

function parseBudget(exploration: unknown): OptimizationBudget | undefined {
  if (!isPlainObject(exploration) || !isPlainObject(exploration["budgets"])) {
    return undefined;
  }

  const budgets = exploration["budgets"];
  const budget: OptimizationBudget = {};

  if (budgets["max_trials"] !== undefined) {
    budget.maxTrials = toPositiveInteger(
      budgets["max_trials"],
      "exploration.budgets.max_trials",
    );
  }
  if (budgets["max_spend_usd"] !== undefined) {
    budget.maxCostUsd = toFiniteNumber(
      budgets["max_spend_usd"],
      "exploration.budgets.max_spend_usd",
    );
    if (budget.maxCostUsd <= 0) {
      throw new ValidationError(
        "exploration.budgets.max_spend_usd must be a positive number.",
      );
    }
  }
  if (budgets["max_wallclock_s"] !== undefined) {
    const seconds = toFiniteNumber(
      budgets["max_wallclock_s"],
      "exploration.budgets.max_wallclock_s",
    );
    if (seconds <= 0) {
      throw new ValidationError(
        "exploration.budgets.max_wallclock_s must be a positive number.",
      );
    }
    budget.maxWallclockMs = Math.round(seconds * 1000);
  }

  return Object.keys(budget).length === 0 ? undefined : budget;
}

function parseStrategyType(exploration: unknown): string | undefined {
  if (!isPlainObject(exploration)) {
    return undefined;
  }
  const strategy = exploration["strategy"];
  if (!isPlainObject(strategy) || typeof strategy["type"] !== "string") {
    return undefined;
  }
  const type = strategy["type"];
  if (
    type !== "grid" &&
    type !== "random" &&
    type !== "bayesian" &&
    type !== "tpe" &&
    type !== "optuna" &&
    type !== "nsga2" &&
    type !== "pareto_optimal"
  ) {
    throw new ValidationError(
      `TVL exploration.strategy.type "${type}" is not supported by the JS SDK.`,
    );
  }
  return type;
}

function parsePromotionPolicy(raw: unknown): PromotionPolicy | undefined {
  if (raw === undefined) {
    return undefined;
  }
  if (!isPlainObject(raw)) {
    throw new ValidationError("promotion_policy must be an object when provided.");
  }

  const normalized: PromotionPolicy = {};

  if (raw["dominance"] !== undefined) {
    if (raw["dominance"] !== "epsilon_pareto") {
      throw new ValidationError(`promotion_policy.dominance must be "epsilon_pareto".`);
    }
    normalized.dominance = "epsilon_pareto";
  }

  if (raw["alpha"] !== undefined) {
    const alpha = toFiniteNumber(raw["alpha"], "promotion_policy.alpha");
    if (alpha <= 0 || alpha >= 1) {
      throw new ValidationError("promotion_policy.alpha must be in (0, 1).");
    }
    normalized.alpha = alpha;
  }

  if (raw["adjust"] !== undefined) {
    if (raw["adjust"] !== "none" && raw["adjust"] !== "BH") {
      throw new ValidationError('promotion_policy.adjust must be "none" or "BH".');
    }
    normalized.adjust = raw["adjust"];
  }

  if (raw["min_effect"] !== undefined) {
    if (!isPlainObject(raw["min_effect"])) {
      throw new ValidationError("promotion_policy.min_effect must be an object.");
    }
    normalized.minEffect = Object.fromEntries(
      Object.entries(raw["min_effect"]).map(([metric, value]) => {
        const epsilon = toFiniteNumber(
          value,
          `promotion_policy.min_effect.${metric}`,
        );
        if (epsilon < 0) {
          throw new ValidationError(
            `promotion_policy.min_effect.${metric} must be non-negative.`,
          );
        }
        return [metric, epsilon];
      }),
    );
  }

  if (raw["chance_constraints"] !== undefined) {
    if (!Array.isArray(raw["chance_constraints"])) {
      throw new ValidationError("promotion_policy.chance_constraints must be an array.");
    }
    normalized.chanceConstraints = raw["chance_constraints"].map((entry, index) => {
      if (!isPlainObject(entry)) {
        throw new ValidationError(
          `promotion_policy.chance_constraints[${index}] must be an object.`,
        );
      }
      const threshold = toFiniteNumber(
        entry["threshold"],
        `promotion_policy.chance_constraints[${index}].threshold`,
      );
      const confidence = toFiniteNumber(
        entry["confidence"],
        `promotion_policy.chance_constraints[${index}].confidence`,
      );
      if (threshold < 0 || threshold > 1) {
        throw new ValidationError(
          `promotion_policy.chance_constraints[${index}].threshold must be in [0, 1].`,
        );
      }
      if (confidence <= 0 || confidence >= 1) {
        throw new ValidationError(
          `promotion_policy.chance_constraints[${index}].confidence must be in (0, 1).`,
        );
      }
      if (typeof entry["name"] !== "string" || entry["name"].trim().length === 0) {
        throw new ValidationError(
          `promotion_policy.chance_constraints[${index}].name must be a non-empty string.`,
        );
      }
      return {
        name: entry["name"].trim(),
        threshold,
        confidence,
      };
    });
  }

  if (raw["tie_breakers"] !== undefined) {
    if (!isPlainObject(raw["tie_breakers"])) {
      throw new ValidationError("promotion_policy.tie_breakers must be an object.");
    }
    normalized.tieBreakers = Object.fromEntries(
      Object.entries(raw["tie_breakers"]).map(([metric, direction]) => {
        if (direction !== "maximize" && direction !== "minimize") {
          throw new ValidationError(
            `promotion_policy.tie_breakers.${metric} must be "maximize" or "minimize".`,
          );
        }
        return [metric, direction];
      }),
    );
  }

  return Object.keys(normalized).length === 0 ? undefined : normalized;
}

export function parseTvlSpec(source: string): LoadedTvlOptimizationSpec {
  const parsed = parseYaml(source) as unknown;
  if (!isPlainObject(parsed)) {
    throw new ValidationError("TVL source must parse to an object.");
  }

  const budget = parseBudget(parsed["exploration"]);
  const constraints = parseConstraints(parsed["constraints"]);
  const promotionPolicy = parsePromotionPolicy(parsed["promotion_policy"]);
  const configurationSpace = parseTvars(parsed["tvars"]);
  const defaultConfig = Object.fromEntries(
    Object.entries(parsed["tvars"] instanceof Array ? parsed["tvars"].reduce<Record<string, unknown>>((acc, entry) => {
      if (isPlainObject(entry) && typeof entry["name"] === "string" && entry["default"] !== undefined) {
        acc[entry["name"]] = entry["default"];
      }
      return acc;
    }, {}) : {}).map(([key, value]) => [key, value]),
  );

  const spec: OptimizationSpec = {
    configurationSpace,
    objectives: parseObjectives(parsed["objectives"]),
    ...(budget ? { budget } : {}),
    ...(constraints ? { constraints } : {}),
    ...(promotionPolicy ? { promotionPolicy } : {}),
    ...(Object.keys(defaultConfig).length > 0 ? { defaultConfig } : {}),
  };

  return {
    spec: normalizeOptimizationSpec(spec),
    metadata: {
      strategyType: parseStrategyType(parsed["exploration"]),
    },
  };
}

export async function loadTvlSpec(
  input: string | TvlLoadOptions,
): Promise<LoadedTvlOptimizationSpec> {
  const options =
    typeof input === "string" ? ({ path: input } satisfies TvlLoadOptions) : input;

  if (options.path) {
    const source = await readFile(options.path, "utf8");
    const loaded = parseTvlSpec(source);
    return {
      ...loaded,
      metadata: {
        ...loaded.metadata,
        path: options.path,
      },
    };
  }

  if (options.source) {
    return parseTvlSpec(options.source);
  }

  throw new ValidationError("loadTvlSpec() requires either path or source.");
}
