import path, { dirname } from "path";
import { fileURLToPath } from "url";
import type { PackageJson } from "type-fest";

import fs from "fs-extra";

const __filename = fileURLToPath(import.meta.url);
const dirName = dirname(__filename);

function findRoot(startDir: string): string {
  let currentDir = startDir;
  while (true) {
    if (fs.existsSync(path.join(currentDir, "package.json"))) {
      return currentDir;
    }
    const parentDir = path.dirname(currentDir);
    if (parentDir === currentDir) {
      throw new Error("Could not find project root (no package.json found in directory tree)");
    }
    currentDir = parentDir;
  }
}

function getPackageInfo(startDir: string = path.dirname(fileURLToPath(import.meta.url))): PackageJson {
  const rootDir = findRoot(startDir);
  return fs.readJsonSync(path.join(rootDir, "package.json")) as PackageJson;
}

export { getPackageInfo, findRoot, dirName };
