import { exec } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";

import { beforeAll, describe, expect, test } from "vitest";

import { getPackageInfo } from "../src/utils/package-info";

const execAsync = promisify(exec);
const CLI_PATH = path.resolve("./dist/index.js");

describe("CLI Integration", () => {
  beforeAll(async () => {
    // Build the CLI before running tests
    await execAsync("pnpm build");
  });

  test("displays version with -v flag", async () => {
    const { stdout } = await execAsync(`node ${CLI_PATH} -v`);
    expect(stdout.trim()).toBe(getPackageInfo().version);
  });

  test("displays help with no arguments", async () => {
    const { stdout } = await execAsync(`node ${CLI_PATH}`);
    expect(stdout).toContain("Typesense CLI");
    expect(stdout).toContain("Utilities for Typesense");
  });

  test("displays help with --help flag", async () => {
    const { stdout } = await execAsync(`node ${CLI_PATH} --help`);
    expect(stdout).toContain("Typesense CLI");
    expect(stdout).toContain("Utilities for Typesense");
  });

  test("displays version with --version flag", async () => {
    const { stdout } = await execAsync(`node ${CLI_PATH} --version`);
    expect(stdout.trim()).toBe(getPackageInfo().version);
  });

  test("handles invalid commands", async () => {
    await expect(execAsync(`node ${CLI_PATH} invalid-command`)).rejects.toThrow();
  });
});
