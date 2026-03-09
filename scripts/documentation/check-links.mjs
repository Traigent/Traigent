#!/usr/bin/env node

import { readdir, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    if (
      entry.name === "node_modules" ||
      entry.name === ".git" ||
      entry.name === "dist" ||
      entry.name === "coverage"
    ) {
      continue;
    }
    const fullPath = resolve(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walk(fullPath)));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(fullPath);
    }
  }
  return files;
}

const markdownFiles = await walk(process.cwd());
const linkPattern = /\[[^\]]+\]\(([^)]+)\)/g;
const errors = [];

for (const file of markdownFiles) {
  const content = await readFile(file, "utf8");
  for (const match of content.matchAll(linkPattern)) {
    const rawLink = match[1];
    if (
      rawLink.startsWith("http://") ||
      rawLink.startsWith("https://") ||
      rawLink.startsWith("mailto:") ||
      rawLink.startsWith("#")
    ) {
      continue;
    }

    const linkPath = rawLink.split("#")[0];
    if (!linkPath) continue;

    const resolvedPath = resolve(dirname(file), linkPath);
    if (!existsSync(resolvedPath)) {
      errors.push(`${file}: broken link -> ${rawLink}`);
    }
  }
}

if (errors.length > 0) {
  console.error("Broken documentation links found:");
  for (const error of errors.slice(0, 50)) {
    console.error(`- ${error}`);
  }
  process.exit(1);
}

console.log(`Checked ${markdownFiles.length} markdown files: no broken internal links.`);
