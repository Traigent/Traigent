import { parseExpression } from '@babel/parser';
import type {
  BinaryExpression,
  Expression,
  Identifier,
  LogicalExpression,
  MemberExpression,
  Node,
  UnaryExpression,
} from '@babel/types';

import { ValidationError } from '../core/errors.js';
import type { OptimizationConstraint } from './types.js';

const IDENTIFIER_PATTERN = /[A-Za-z_$]/;
const IDENTIFIER_CONTINUATION_PATTERN = /[A-Za-z0-9_$]/;

type ConstraintScope = {
  config: Record<string, unknown>;
  metrics?: Record<string, unknown>;
};

function normalizeConstraintExpressionSyntax(expression: string): string {
  const trimmed = expression.trim();
  if (trimmed.length === 0) {
    throw new ValidationError('Constraint expressions must be non-empty strings.');
  }

  let output = '';
  let index = 0;
  let quote: '"' | "'" | null = null;
  let escaped = false;

  while (index < trimmed.length) {
    const character = trimmed[index]!;

    if (quote) {
      output += character;
      if (escaped) {
        escaped = false;
      } else if (character === '\\') {
        escaped = true;
      } else if (character === quote) {
        quote = null;
      }
      index += 1;
      continue;
    }

    if (character === '"' || character === "'") {
      quote = character;
      output += character;
      index += 1;
      continue;
    }

    if (IDENTIFIER_PATTERN.test(character)) {
      let end = index + 1;
      while (
        end < trimmed.length &&
        IDENTIFIER_CONTINUATION_PATTERN.test(trimmed[end]!)
      ) {
        end += 1;
      }
      const token = trimmed.slice(index, end);
      if (token === 'and') {
        output += '&&';
      } else if (token === 'or') {
        output += '||';
      } else if (token === 'not') {
        output += '!';
      } else {
        output += token;
      }
      index = end;
      continue;
    }

    if (
      character === '=' &&
      trimmed[index - 1] !== '=' &&
      trimmed[index - 1] !== '!' &&
      trimmed[index - 1] !== '<' &&
      trimmed[index - 1] !== '>' &&
      trimmed[index + 1] !== '='
    ) {
      output += '===';
      index += 1;
      continue;
    }

    output += character;
    index += 1;
  }

  if (quote) {
    throw new ValidationError('Constraint expressions must terminate string literals.');
  }

  return output;
}

function parseConstraintAst(expression: string, id: string): Expression {
  try {
    return parseExpression(normalizeConstraintExpressionSyntax(expression), {
      sourceType: 'script',
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'unknown parse failure';
    throw new ValidationError(
      `Constraint "${id}" could not be parsed: ${message}`,
    );
  }
}

function assertSupportedIdentifier(node: Identifier, id: string): void {
  if (
    node.name !== 'config' &&
    node.name !== 'params' &&
    node.name !== 'metrics' &&
    node.name !== 'undefined'
  ) {
    throw new ValidationError(
      `Constraint "${id}" uses unsupported identifier "${node.name}".`,
    );
  }
}

function assertSupportedMemberExpression(
  node: MemberExpression,
  id: string,
): void {
  if (node.computed) {
    throw new ValidationError(
      `Constraint "${id}" cannot use computed property access.`,
    );
  }
  if (node.property.type !== 'Identifier') {
    throw new ValidationError(
      `Constraint "${id}" must use identifier-based property access.`,
    );
  }
  assertSupportedExpressionNode(node.object as Expression, id);
}

function assertSupportedBinaryExpression(
  node: BinaryExpression,
  id: string,
): void {
  if (
    node.operator !== '===' &&
    node.operator !== '!==' &&
    node.operator !== '==' &&
    node.operator !== '!=' &&
    node.operator !== '<' &&
    node.operator !== '<=' &&
    node.operator !== '>' &&
    node.operator !== '>=' &&
    node.operator !== '+' &&
    node.operator !== '-' &&
    node.operator !== '*' &&
    node.operator !== '/' &&
    node.operator !== '%'
  ) {
    throw new ValidationError(
      `Constraint "${id}" uses unsupported binary operator "${node.operator}".`,
    );
  }
  assertSupportedExpressionNode(node.left as Expression, id);
  assertSupportedExpressionNode(node.right as Expression, id);
}

function assertSupportedLogicalExpression(
  node: LogicalExpression,
  id: string,
): void {
  if (node.operator !== '&&' && node.operator !== '||') {
    throw new ValidationError(
      `Constraint "${id}" uses unsupported logical operator "${node.operator}".`,
    );
  }
  assertSupportedExpressionNode(node.left as Expression, id);
  assertSupportedExpressionNode(node.right as Expression, id);
}

function assertSupportedUnaryExpression(
  node: UnaryExpression,
  id: string,
): void {
  if (node.operator !== '!' && node.operator !== '+' && node.operator !== '-') {
    throw new ValidationError(
      `Constraint "${id}" uses unsupported unary operator "${node.operator}".`,
    );
  }
  assertSupportedExpressionNode(node.argument, id);
}

function assertSupportedExpressionNode(node: Node, id: string): void {
  switch (node.type) {
    case 'BooleanLiteral':
    case 'NumericLiteral':
    case 'StringLiteral':
    case 'NullLiteral':
      return;
    case 'Identifier':
      assertSupportedIdentifier(node, id);
      return;
    case 'MemberExpression':
      assertSupportedMemberExpression(node, id);
      return;
    case 'BinaryExpression':
      assertSupportedBinaryExpression(node, id);
      return;
    case 'LogicalExpression':
      assertSupportedLogicalExpression(node, id);
      return;
    case 'UnaryExpression':
      assertSupportedUnaryExpression(node, id);
      return;
    default:
      throw new ValidationError(
        `Constraint "${id}" uses unsupported syntax "${node.type}".`,
      );
  }
}

function expressionReferencesMetrics(node: Node): boolean {
  switch (node.type) {
    case 'Identifier':
      return node.name === 'metrics';
    case 'MemberExpression':
      return expressionReferencesMetrics(node.object as Expression);
    case 'BinaryExpression':
    case 'LogicalExpression':
      return (
        expressionReferencesMetrics(node.left as Expression) ||
        expressionReferencesMetrics(node.right as Expression)
      );
    case 'UnaryExpression':
      return expressionReferencesMetrics(node.argument);
    default:
      return false;
  }
}

function evaluateMemberExpression(
  node: MemberExpression,
  scope: ConstraintScope,
): unknown {
  const target = evaluateExpression(node.object as Expression, scope);
  if (target === null || target === undefined || typeof target !== 'object') {
    return undefined;
  }
  const property = node.property as Identifier;
  return (target as Record<string, unknown>)[property.name];
}

function evaluateBinaryExpression(
  node: BinaryExpression,
  scope: ConstraintScope,
): unknown {
  const left = evaluateExpression(node.left as Expression, scope);
  const right = evaluateExpression(node.right as Expression, scope);

  switch (node.operator) {
    case '===':
    case '==':
      return left === right;
    case '!==':
    case '!=':
      return left !== right;
    case '<':
      return (left as number) < (right as number);
    case '<=':
      return (left as number) <= (right as number);
    case '>':
      return (left as number) > (right as number);
    case '>=':
      return (left as number) >= (right as number);
    case '+':
      return (left as number) + (right as number);
    case '-':
      return (left as number) - (right as number);
    case '*':
      return (left as number) * (right as number);
    case '/':
      return (left as number) / (right as number);
    case '%':
      return (left as number) % (right as number);
    default:
      throw new ValidationError(
        `Unsupported binary operator "${node.operator}".`,
      );
  }
}

function evaluateExpression(node: Expression, scope: ConstraintScope): unknown {
  switch (node.type) {
    case 'BooleanLiteral':
    case 'NumericLiteral':
    case 'StringLiteral':
      return node.value;
    case 'NullLiteral':
      return null;
    case 'Identifier':
      if (node.name === 'config' || node.name === 'params') {
        return scope.config;
      }
      if (node.name === 'metrics') {
        return scope.metrics ?? {};
      }
      if (node.name === 'undefined') {
        return undefined;
      }
      throw new ValidationError(`Unsupported identifier "${node.name}".`);
    case 'MemberExpression':
      return evaluateMemberExpression(node, scope);
    case 'BinaryExpression':
      return evaluateBinaryExpression(node, scope);
    case 'LogicalExpression':
      if (node.operator === '&&') {
        return !!evaluateExpression(node.left as Expression, scope) &&
          !!evaluateExpression(node.right as Expression, scope);
      }
      return !!evaluateExpression(node.left as Expression, scope) ||
        !!evaluateExpression(node.right as Expression, scope);
    case 'UnaryExpression': {
      const argument = evaluateExpression(node.argument, scope);
      switch (node.operator) {
        case '!':
          return !argument;
        case '+':
          return Number(argument);
        case '-':
          return -Number(argument);
        default:
          throw new ValidationError(
            `Unsupported unary operator "${node.operator}".`,
          );
      }
    }
    default:
      throw new ValidationError(`Unsupported syntax "${node.type}".`);
  }
}

export function compileTvlConstraint(
  id: string,
  expression: string,
  errorMessage: string | undefined,
  mode: 'expr' | 'implication',
  whenExpression?: string,
): OptimizationConstraint {
  const expressionAst = parseConstraintAst(expression, id);
  assertSupportedExpressionNode(expressionAst, id);

  const whenAst = whenExpression
    ? parseConstraintAst(whenExpression, id)
    : undefined;
  if (whenAst) {
    assertSupportedExpressionNode(whenAst, id);
  }

  const requiresMetrics =
    expressionReferencesMetrics(expressionAst) ||
    (whenAst ? expressionReferencesMetrics(whenAst) : false);

  const constraint = ((
    config: Record<string, unknown>,
    metrics?: Record<string, unknown>,
  ) => {
    try {
      if (mode === 'implication' && whenAst) {
        return (
          !evaluateExpression(whenAst, { config, metrics }) ||
          !!evaluateExpression(expressionAst, { config, metrics })
        );
      }

      return !!evaluateExpression(expressionAst, { config, metrics });
    } catch (error) {
      const reason =
        error instanceof Error ? error.message : 'unknown constraint failure';
      throw new ValidationError(
        errorMessage
          ? `${errorMessage}: ${reason}`
          : `Constraint "${id}" failed: ${reason}`,
      );
    }
  }) as OptimizationConstraint;

  constraint.requiresMetrics = requiresMetrics;
  Object.defineProperty(constraint, 'name', {
    value: `tvlConstraint_${id}`,
    configurable: true,
  });
  return constraint;
}
