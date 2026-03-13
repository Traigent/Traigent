import { Ajv, type ErrorObject } from 'ajv';

export interface ConfigValidationIssue {
  path?: string;
  message: string;
  keyword?: string;
}

export interface ConfigValidationResult {
  ok: boolean;
  issues?: ConfigValidationIssue[];
  summary: string;
  truncated?: boolean;
  total_issues?: number;
}

const MAX_ISSUES = 20;

function toObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function toPath(error: ErrorObject): string | undefined {
  const parts = error.instancePath
    .split('/')
    .filter((part) => part.length > 0)
    .map((part) => part.replace(/~1/g, '/').replace(/~0/g, '~'));

  if (error.keyword === 'required' && toObject(error.params)) {
    const missingProperty = error.params['missingProperty'];
    if (typeof missingProperty === 'string' && missingProperty.length > 0) {
      parts.push(missingProperty);
    }
  }

  return parts.length > 0 ? parts.join('.') : undefined;
}

function toIssues(errors: ErrorObject[] | null | undefined): ConfigValidationIssue[] {
  if (!errors || errors.length === 0) {
    return [];
  }
  return errors.map((error) => ({
    path: toPath(error),
    message: error.message ?? `Schema validation failed (${error.keyword})`,
    keyword: error.keyword,
  }));
}

function truncateIssues(issues: ConfigValidationIssue[]): {
  issues: ConfigValidationIssue[];
  truncated?: boolean;
  totalIssues?: number;
} {
  if (issues.length <= MAX_ISSUES) {
    return { issues };
  }
  return {
    issues: issues.slice(0, MAX_ISSUES),
    truncated: true,
    totalIssues: issues.length,
  };
}

export function validateConfigPayload(
  config: unknown,
  configSchema?: Record<string, unknown>
): ConfigValidationResult {
  if (!toObject(config)) {
    return {
      ok: false,
      issues: [{ message: 'config must be an object' }],
      summary: 'Validation failed: 1 issue(s)',
    };
  }

  if (!configSchema) {
    return {
      ok: true,
      summary: 'Config validation passed',
    };
  }

  const ajv = new Ajv({
    allErrors: true,
    strict: false,
    validateSchema: true,
  });

  const schemaValid = ajv.validateSchema(configSchema);
  if (!schemaValid) {
    const rawIssues = toIssues(ajv.errors);
    const { issues, truncated, totalIssues } = truncateIssues(rawIssues);
    return {
      ok: false,
      issues,
      summary: `Invalid config schema: ${rawIssues.length} issue(s)`,
      ...(truncated ? { truncated } : {}),
      ...(totalIssues !== undefined ? { total_issues: totalIssues } : {}),
    };
  }

  const validate = ajv.compile(configSchema);
  const valid = validate(config);
  if (valid) {
    return {
      ok: true,
      summary: 'Config validation passed',
    };
  }

  const rawIssues = toIssues(validate.errors);
  const { issues, truncated, totalIssues } = truncateIssues(rawIssues);
  return {
    ok: false,
    issues,
    summary: `Validation failed: ${rawIssues.length} issue(s)`,
    ...(truncated ? { truncated } : {}),
    ...(totalIssues !== undefined ? { total_issues: totalIssues } : {}),
  };
}
