import { describe, expect, it } from "vitest";

import { isStringifiable } from "../src/utils/stringifiable";

describe("isStringifiable", () => {
  it("returns true for strings", () => {
    expect(isStringifiable("")).toBe(true);
    expect(isStringifiable("hello")).toBe(true);
  });

  it("returns true for numbers", () => {
    expect(isStringifiable(0)).toBe(true);
    expect(isStringifiable(-1)).toBe(true);
    expect(isStringifiable(1.5)).toBe(true);
  });

  it("returns true for booleans", () => {
    expect(isStringifiable(true)).toBe(true);
    expect(isStringifiable(false)).toBe(true);
  });

  it("returns true for objects with toString method", () => {
    expect(isStringifiable({})).toBe(true);
    expect(isStringifiable(new Date())).toBe(true);
    expect(isStringifiable(/regex/)).toBe(true);
  });

  it("returns false for null", () => {
    expect(isStringifiable(null)).toBe(false);
  });

  it("returns false for undefined", () => {
    expect(isStringifiable(undefined)).toBe(false);
  });

  it("type narrowing works", () => {
    const value: unknown = "test";
    if (isStringifiable(value)) {
      expect(value.toString()).toBe("test");
    }
  });
});
