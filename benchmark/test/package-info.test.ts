import { tmpdir } from "os";
import path from "path";

import fs from "fs-extra";
import { afterEach, beforeEach, describe, expect, test } from "vitest";

import { findRoot, getPackageInfo } from "../src/utils/package-info";

describe("package info utilities", () => {
  let tempDir: string;
  let subDir: string;
  let deepSubDir: string;

  beforeEach(() => {
    // Create a temporary directory structure for testing
    tempDir = fs.mkdtempSync(path.join(tmpdir(), "package-info-test-"));
    subDir = path.join(tempDir, "subdir");
    deepSubDir = path.join(subDir, "deeper");

    fs.mkdirSync(subDir);
    fs.mkdirSync(deepSubDir);
  });

  afterEach(() => {
    // Clean up temporary directories after each test
    fs.removeSync(tempDir);
  });

  describe("findRoot", () => {
    test("finds root directory containing package.json", () => {
      // Create a package.json in the temp directory
      fs.writeJsonSync(path.join(tempDir, "package.json"), {
        name: "test-package",
        version: "1.0.0",
      });

      // Should find root from nested directories
      expect(findRoot(deepSubDir)).toBe(tempDir);
      expect(findRoot(subDir)).toBe(tempDir);
      expect(findRoot(tempDir)).toBe(tempDir);
    });

    test("finds root when package.json is in intermediate directory", () => {
      // Create package.json in the subdirectory
      fs.writeJsonSync(path.join(subDir, "package.json"), {
        name: "test-package",
        version: "1.0.0",
      });

      expect(findRoot(deepSubDir)).toBe(subDir);
      expect(findRoot(subDir)).toBe(subDir);
    });

    test("throws error when no package.json is found", () => {
      expect(() => findRoot(deepSubDir)).toThrow(
        "Could not find project root (no package.json found in directory tree)",
      );
    });
  });

  describe("getPackageInfo", () => {
    test("reads and returns package.json content", () => {
      const packageData = {
        name: "test-package",
        version: "1.0.0",
        dependencies: {
          "fs-extra": "^10.0.0",
        },
      };

      fs.writeJsonSync(path.join(tempDir, "package.json"), packageData);

      const result = getPackageInfo(deepSubDir);
      expect(result).toEqual(packageData);
    });

    test("handles package.json with minimal fields", () => {
      const packageData = {
        name: "minimal-package",
      };

      fs.writeJsonSync(path.join(tempDir, "package.json"), packageData);

      const result = getPackageInfo(deepSubDir);
      expect(result).toEqual(packageData);
    });

    test("throws error when package.json is not found", () => {
      expect(() => getPackageInfo(deepSubDir)).toThrow(
        "Could not find project root (no package.json found in directory tree)",
      );
    });

    test("throws error when package.json is malformed", () => {
      // Create an invalid JSON file
      fs.writeFileSync(path.join(tempDir, "package.json"), "{invalid json");

      expect(() => getPackageInfo(deepSubDir)).toThrow();
    });
  });
});
