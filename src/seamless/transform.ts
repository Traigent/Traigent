import { parse } from '@babel/parser';
import generatorModule from '@babel/generator';
import traverseModule, { type NodePath } from '@babel/traverse';
import * as t from '@babel/types';

const traverse = traverseModule.default ?? traverseModule;
const generate = generatorModule.default ?? generatorModule;
const PARSER_PLUGINS: NonNullable<Parameters<typeof parse>[1]>['plugins'] = ['typescript', 'jsx'];

type LocalImportMap = Map<string, string>;

export interface SeamlessDiagnostic {
  kind: 'rewritten' | 'rejected';
  message: string;
  filename?: string;
  functionName?: string;
  key?: string;
  line?: number;
  column?: number;
}

export interface SeamlessTransformResult {
  code: string;
  changed: boolean;
  rewrittenCount: number;
  diagnostics: SeamlessDiagnostic[];
}

interface TransformProgramOptions {
  filename?: string;
}

interface RewriteRequest {
  functionPath:
    | NodePath<t.FunctionDeclaration>
    | NodePath<t.FunctionExpression>
    | NodePath<t.ArrowFunctionExpression>;
  configKeys: Set<string>;
  importSource: string;
}

interface PendingRewrite {
  apply: () => void;
  diagnostic: SeamlessDiagnostic;
}

function createDiagnostic(
  kind: SeamlessDiagnostic['kind'],
  message: string,
  node: t.Node | null | undefined,
  filename: string | undefined,
  functionName?: string,
  key?: string
): SeamlessDiagnostic {
  return {
    kind,
    message,
    filename,
    functionName,
    key,
    line: node?.loc?.start.line,
    column: node?.loc?.start.column,
  };
}

function isStaticKeyName(
  property: t.ObjectProperty | t.ObjectMethod | t.ClassMethod | t.ClassProperty
): string | undefined {
  if (property.computed) {
    return undefined;
  }

  if (t.isIdentifier(property.key)) {
    return property.key.name;
  }

  if (t.isStringLiteral(property.key)) {
    return property.key.value;
  }

  return undefined;
}

function getConfigKeysFromSpecObject(
  specNode: t.Expression | t.SpreadElement | t.ArgumentPlaceholder | undefined
): Set<string> | undefined {
  if (!specNode || !t.isObjectExpression(specNode)) {
    return undefined;
  }

  const configurationSpaceProperty = specNode.properties.find((property) => {
    if (!t.isObjectProperty(property)) {
      return false;
    }

    return isStaticKeyName(property) === 'configurationSpace';
  });

  if (
    !configurationSpaceProperty ||
    !t.isObjectProperty(configurationSpaceProperty) ||
    !t.isObjectExpression(configurationSpaceProperty.value)
  ) {
    return undefined;
  }

  const keys = new Set<string>();
  for (const property of configurationSpaceProperty.value.properties) {
    if (!t.isObjectProperty(property)) {
      return undefined;
    }

    const key = isStaticKeyName(property);
    if (!key) {
      return undefined;
    }

    keys.add(key);
  }

  return keys;
}

function getImportDetails(programPath: NodePath<t.Program>): {
  optimizeImports: LocalImportMap;
  getterImports: LocalImportMap;
  importPathsBySource: Map<string, NodePath<t.ImportDeclaration>[]>;
} {
  const optimizeImports: LocalImportMap = new Map();
  const getterImports: LocalImportMap = new Map();
  const importPathsBySource = new Map<string, NodePath<t.ImportDeclaration>[]>();

  for (const statementPath of programPath.get('body')) {
    if (!statementPath.isImportDeclaration()) {
      continue;
    }

    const source = statementPath.node.source.value;
    importPathsBySource.set(source, [...(importPathsBySource.get(source) ?? []), statementPath]);

    for (const specifier of statementPath.node.specifiers) {
      if (!t.isImportSpecifier(specifier)) {
        continue;
      }

      if (t.isIdentifier(specifier.imported) && specifier.imported.name === 'optimize') {
        optimizeImports.set(specifier.local.name, source);
      }

      if (t.isIdentifier(specifier.imported) && specifier.imported.name === 'getTrialParam') {
        getterImports.set(source, specifier.local.name);
      }
    }
  }

  return {
    optimizeImports,
    getterImports,
    importPathsBySource,
  };
}

function getNodeKey(node: t.Node): string {
  return `${node.start ?? -1}:${node.end ?? -1}`;
}

function resolveFunctionPath(
  targetPath: NodePath<t.Node>
):
  | NodePath<t.FunctionDeclaration>
  | NodePath<t.FunctionExpression>
  | NodePath<t.ArrowFunctionExpression>
  | undefined {
  if (
    targetPath.isFunctionDeclaration() ||
    targetPath.isFunctionExpression() ||
    targetPath.isArrowFunctionExpression()
  ) {
    return targetPath;
  }

  if (!targetPath.isIdentifier()) {
    return undefined;
  }

  const binding = targetPath.scope.getBinding(targetPath.node.name);
  if (!binding) {
    return undefined;
  }

  if (binding.path.isFunctionDeclaration()) {
    return binding.path;
  }

  if (!binding.path.isVariableDeclarator()) {
    return undefined;
  }

  const initPath = binding.path.get('init');
  if (initPath.isFunctionExpression() || initPath.isArrowFunctionExpression()) {
    return initPath;
  }

  return undefined;
}

function makeGetterCall(
  getterLocalName: string,
  key: string,
  fallback: t.Expression
): t.CallExpression {
  return t.callExpression(t.identifier(getterLocalName), [t.stringLiteral(key), fallback]);
}

function isSupportedLiteralExpression(node: t.Node | null | undefined): node is t.Expression {
  if (!node) {
    return false;
  }

  if (
    t.isStringLiteral(node) ||
    t.isNumericLiteral(node) ||
    t.isBooleanLiteral(node) ||
    t.isNullLiteral(node)
  ) {
    return true;
  }

  if (t.isUnaryExpression(node)) {
    return (node.operator === '-' || node.operator === '+') && t.isNumericLiteral(node.argument);
  }

  if (t.isTemplateLiteral(node)) {
    return node.expressions.length === 0;
  }

  if (t.isArrayExpression(node)) {
    return node.elements.every((element) => {
      if (!element) {
        return false;
      }
      if (t.isSpreadElement(element)) {
        return false;
      }
      return isSupportedLiteralExpression(element);
    });
  }

  if (t.isObjectExpression(node)) {
    return node.properties.every((property) => {
      if (!t.isObjectProperty(property) || property.computed) {
        return false;
      }

      if (
        !t.isIdentifier(property.key) &&
        !t.isStringLiteral(property.key) &&
        !t.isNumericLiteral(property.key)
      ) {
        return false;
      }

      return isSupportedLiteralExpression(property.value);
    });
  }

  return false;
}

function getPatternIdentifiers(pattern: t.LVal): string[] {
  if (t.isIdentifier(pattern)) {
    return [pattern.name];
  }

  if (t.isRestElement(pattern)) {
    return getPatternIdentifiers(pattern.argument);
  }

  if (t.isAssignmentPattern(pattern)) {
    return getPatternIdentifiers(pattern.left);
  }

  if (t.isArrayPattern(pattern)) {
    return pattern.elements.flatMap((element) =>
      element ? getPatternIdentifiers(element as t.LVal) : []
    );
  }

  if (t.isObjectPattern(pattern)) {
    return pattern.properties.flatMap((property) => {
      if (t.isRestElement(property)) {
        return t.isLVal(property.argument) ? getPatternIdentifiers(property.argument) : [];
      }

      if (t.isObjectProperty(property)) {
        return getPatternIdentifiers(property.value as t.LVal);
      }

      return [];
    });
  }

  return [];
}

export function transformSeamlessFunctionPath(
  functionPath:
    | NodePath<t.FunctionDeclaration>
    | NodePath<t.FunctionExpression>
    | NodePath<t.ArrowFunctionExpression>,
  configKeys: Set<string>,
  getterLocalName: string,
  filename: string | undefined,
  diagnostics: SeamlessDiagnostic[]
): number {
  const pendingRewrites: PendingRewrite[] = [];
  const localDiagnostics: SeamlessDiagnostic[] = [];
  const functionName =
    ('id' in functionPath.node && functionPath.node.id && t.isIdentifier(functionPath.node.id)
      ? functionPath.node.id.name
      : undefined) ||
    (functionPath.parentPath.isVariableDeclarator() &&
    t.isIdentifier(functionPath.parentPath.node.id)
      ? functionPath.parentPath.node.id.name
      : 'anonymous');

  functionPath.traverse({
    Function(path) {
      if (path !== functionPath) {
        path.skip();
      }
    },
    VariableDeclarator(path) {
      if (!t.isLVal(path.node.id)) {
        return;
      }
      const names = getPatternIdentifiers(path.node.id);
      const matchedName = names.find((name) => configKeys.has(name));
      if (!matchedName) {
        return;
      }

      if (!t.isIdentifier(path.node.id)) {
        localDiagnostics.push(
          createDiagnostic(
            'rejected',
            'Destructuring tuned variables is not supported for seamless rewrites.',
            path.node,
            filename,
            functionName,
            matchedName
          )
        );
        return;
      }

      if (!path.node.init || !isSupportedLiteralExpression(path.node.init)) {
        localDiagnostics.push(
          createDiagnostic(
            'rejected',
            `Tuned variable "${matchedName}" must use a literal, array literal, object literal, or template literal initializer.`,
            path.node,
            filename,
            functionName,
            matchedName
          )
        );
        return;
      }

      if (
        t.isCallExpression(path.node.init) &&
        t.isIdentifier(path.node.init.callee, { name: getterLocalName })
      ) {
        return;
      }

      const originalInitializer = path.node.init;
      pendingRewrites.push({
        apply: () => {
          path.node.init = makeGetterCall(getterLocalName, matchedName, originalInitializer);
        },
        diagnostic: createDiagnostic(
          'rewritten',
          `Rewrote tuned variable "${matchedName}" to getTrialParam().`,
          path.node,
          filename,
          functionName,
          matchedName
        ),
      });
    },
    AssignmentExpression(path) {
      if (!t.isIdentifier(path.node.left) || !configKeys.has(path.node.left.name)) {
        return;
      }

      const key = path.node.left.name;
      if (path.node.operator !== '=') {
        localDiagnostics.push(
          createDiagnostic(
            'rejected',
            `Only direct "=" reassignment is supported for tuned variable "${key}".`,
            path.node,
            filename,
            functionName,
            key
          )
        );
        return;
      }

      if (!isSupportedLiteralExpression(path.node.right)) {
        localDiagnostics.push(
          createDiagnostic(
            'rejected',
            `Tuned variable "${key}" can only be reassigned from supported literal values.`,
            path.node,
            filename,
            functionName,
            key
          )
        );
        return;
      }

      const originalRightHandSide = path.node.right;
      pendingRewrites.push({
        apply: () => {
          path.node.right = makeGetterCall(getterLocalName, key, originalRightHandSide);
        },
        diagnostic: createDiagnostic(
          'rewritten',
          `Rewrote tuned variable reassignment "${key}" to getTrialParam().`,
          path.node,
          filename,
          functionName,
          key
        ),
      });
    },
    UpdateExpression(path) {
      if (t.isIdentifier(path.node.argument) && configKeys.has(path.node.argument.name)) {
        localDiagnostics.push(
          createDiagnostic(
            'rejected',
            `Increment/decrement is not supported for tuned variable "${path.node.argument.name}".`,
            path.node,
            filename,
            functionName,
            path.node.argument.name
          )
        );
      }
    },
  });

  const rejectedDiagnostics = localDiagnostics.filter(
    (diagnostic) => diagnostic.kind === 'rejected'
  );

  if (rejectedDiagnostics.length > 0) {
    diagnostics.push(...rejectedDiagnostics);
    if (pendingRewrites.length > 0) {
      diagnostics.push(
        createDiagnostic(
          'rejected',
          `Skipped seamless rewrite for function "${functionName}" because unsupported tuned-variable patterns were found. No partial rewrite was applied.`,
          functionPath.node,
          filename,
          functionName
        )
      );
    }
    return 0;
  }

  for (const rewrite of pendingRewrites) {
    rewrite.apply();
    diagnostics.push(rewrite.diagnostic);
  }

  return pendingRewrites.length;
}

function resolveGetterLocalName(importSource: string, getterImports: LocalImportMap): string {
  return getterImports.get(importSource) ?? 'getTrialParam';
}

function addGetterImport(
  programPath: NodePath<t.Program>,
  importSource: string,
  getterLocalName: string,
  getterImports: LocalImportMap,
  importPathsBySource: Map<string, NodePath<t.ImportDeclaration>[]>
): void {
  if (getterImports.has(importSource)) {
    return;
  }

  const importPaths = importPathsBySource.get(importSource);
  if (importPaths && importPaths.length > 0) {
    importPaths[0]!.node.specifiers.push(
      t.importSpecifier(t.identifier(getterLocalName), t.identifier('getTrialParam'))
    );
    getterImports.set(importSource, getterLocalName);
    return;
  }

  programPath.unshiftContainer(
    'body',
    t.importDeclaration(
      [t.importSpecifier(t.identifier(getterLocalName), t.identifier('getTrialParam'))],
      t.stringLiteral(importSource)
    )
  );
  getterImports.set(importSource, getterLocalName);
}

export function transformSeamlessProgram(
  programPath: NodePath<t.Program>,
  options: TransformProgramOptions = {}
): SeamlessTransformResult {
  const diagnostics: SeamlessDiagnostic[] = [];
  const { optimizeImports, getterImports, importPathsBySource } = getImportDetails(programPath);
  const requests = new Map<string, RewriteRequest>();

  programPath.traverse({
    CallExpression(path) {
      const outerCall = path.node;
      if (outerCall.arguments.length !== 1 || !t.isCallExpression(outerCall.callee)) {
        return;
      }

      const innerCall = outerCall.callee;
      if (!t.isIdentifier(innerCall.callee)) {
        return;
      }

      const importSource = optimizeImports.get(innerCall.callee.name);
      if (!importSource) {
        return;
      }

      const configKeys = getConfigKeysFromSpecObject(innerCall.arguments[0]);
      if (!configKeys || configKeys.size === 0) {
        diagnostics.push(
          createDiagnostic(
            'rejected',
            'Skipping seamless rewrite because optimize() spec is not a static object literal with configurationSpace keys.',
            innerCall.arguments[0] as t.Node | undefined,
            options.filename
          )
        );
        return;
      }

      const targetPath = path.get('arguments.0');
      if (Array.isArray(targetPath)) {
        return;
      }

      const functionPath = resolveFunctionPath(targetPath);
      if (!functionPath) {
        diagnostics.push(
          createDiagnostic(
            'rejected',
            'Skipping seamless rewrite because optimize() target is not a directly resolvable local function.',
            targetPath.node,
            options.filename
          )
        );
        return;
      }

      const requestKey = getNodeKey(functionPath.node);
      const existing = requests.get(requestKey);
      if (existing) {
        for (const key of configKeys) {
          existing.configKeys.add(key);
        }
        return;
      }

      requests.set(requestKey, {
        functionPath,
        configKeys,
        importSource,
      });
    },
  });

  let changed = false;
  let rewrittenCount = 0;

  for (const request of requests.values()) {
    const getterLocalName = resolveGetterLocalName(request.importSource, getterImports);

    const rewritten = transformSeamlessFunctionPath(
      request.functionPath,
      request.configKeys,
      getterLocalName,
      options.filename,
      diagnostics
    );
    if (rewritten > 0) {
      addGetterImport(
        programPath,
        request.importSource,
        getterLocalName,
        getterImports,
        importPathsBySource
      );
      changed = true;
      rewrittenCount += rewritten;
    }
  }

  const code = generate(programPath.node, {
    comments: true,
    retainLines: false,
  }).code;

  return {
    code,
    changed,
    rewrittenCount,
    diagnostics,
  };
}

export function transformSeamlessSource(
  source: string,
  options: TransformProgramOptions = {}
): SeamlessTransformResult {
  const ast = parse(source, {
    sourceType: 'module',
    plugins: PARSER_PLUGINS,
  });

  let result: SeamlessTransformResult = {
    code: source,
    changed: false,
    rewrittenCount: 0,
    diagnostics: [],
  };

  traverse(ast, {
    Program(path) {
      result = transformSeamlessProgram(path, options);
      path.stop();
    },
  });

  return result;
}
