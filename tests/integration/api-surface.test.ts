import { execSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import ts from 'typescript';
import { beforeAll, describe, expect, it } from 'vitest';

const ROOT = resolve(import.meta.dirname, '..', '..');
const FIXTURE_PATH = resolve(ROOT, 'tests/integration/fixtures/api-surface.snapshot.json');

function ensureBuild(): void {
  const entrypoint = resolve(ROOT, 'dist/index.d.ts');
  if (!existsSync(entrypoint)) {
    execSync('npm run build:sdk', { cwd: ROOT, stdio: 'inherit' });
  }
}

function hasExportModifier(node: ts.Node): boolean {
  return (
    ts.canHaveModifiers(node) &&
    (ts.getModifiers(node)?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword) ??
      false)
  );
}

function collectDeclarationExports(filePath: string): string[] {
  const sourceText = readFileSync(filePath, 'utf8');
  const sourceFile = ts.createSourceFile(filePath, sourceText, ts.ScriptTarget.Latest, true);
  const exports = new Set<string>();

  for (const statement of sourceFile.statements) {
    if (ts.isExportDeclaration(statement)) {
      if (statement.exportClause && ts.isNamedExports(statement.exportClause)) {
        for (const element of statement.exportClause.elements) {
          exports.add(element.name.text);
        }
      }
      continue;
    }

    if (!hasExportModifier(statement)) {
      continue;
    }

    if (
      ts.isFunctionDeclaration(statement) ||
      ts.isClassDeclaration(statement) ||
      ts.isInterfaceDeclaration(statement) ||
      ts.isTypeAliasDeclaration(statement) ||
      ts.isEnumDeclaration(statement)
    ) {
      if (statement.name) {
        exports.add(statement.name.text);
      }
      continue;
    }

    if (ts.isVariableStatement(statement)) {
      for (const declaration of statement.declarationList.declarations) {
        if (ts.isIdentifier(declaration.name)) {
          exports.add(declaration.name.text);
        }
      }
    }
  }

  return [...exports].sort((left, right) => left.localeCompare(right));
}

describe('stable API declaration surface', () => {
  beforeAll(() => {
    ensureBuild();
  });

  it('matches the committed stable export snapshot', () => {
    const actual = {
      root: collectDeclarationExports(resolve(ROOT, 'dist/index.d.ts')),
      openai: collectDeclarationExports(resolve(ROOT, 'dist/integrations/openai/index.d.ts')),
      langchain: collectDeclarationExports(resolve(ROOT, 'dist/integrations/langchain/index.d.ts')),
      vercelAi: collectDeclarationExports(resolve(ROOT, 'dist/integrations/vercel-ai/index.d.ts')),
    };

    const expected = JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')) as typeof actual;
    expect(actual).toEqual(expected);
  });
});
