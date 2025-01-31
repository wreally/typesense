import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { logger, LoggerClass, LogLevel } from "../src/utils/logger";

vi.mock("chalk", () => ({
  default: {
    red: vi.fn((str) => `RED:${str}`),
    yellow: vi.fn((str) => `YELLOW:${str}`),
    white: vi.fn((str) => `WHITE:${str}`),
    green: vi.fn((str) => `GREEN:${str}`),
    gray: vi.fn((str) => `GRAY:${str}`),
  },
}));

describe("Logger", () => {
  const mockConsole = {
    error: vi.fn(),
    warn: vi.fn(),
    info: vi.fn(),
    log: vi.fn(),
    debug: vi.fn(),
  };

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(mockConsole.error);
    vi.spyOn(console, "warn").mockImplementation(mockConsole.warn);
    vi.spyOn(console, "info").mockImplementation(mockConsole.info);
    vi.spyOn(console, "log").mockImplementation(mockConsole.log);
    vi.spyOn(console, "debug").mockImplementation(mockConsole.debug);

    logger.setLevel(LogLevel.INFO);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Singleton Pattern", () => {
    test("returns same instance when getInstance is called multiple times", () => {
      const instance1 = LoggerClass.getInstance();
      const instance2 = LoggerClass.getInstance();
      expect(instance1).toBe(instance2);
    });

    test("maintains state across getInstance calls", () => {
      const instance1 = LoggerClass.getInstance();
      instance1.setLevel(LogLevel.DEBUG);

      const instance2 = LoggerClass.getInstance();
      expect(instance2.getLevel()).toBe(LogLevel.DEBUG);
    });
  });

  describe("Log Levels", () => {
    test("correctly sets and gets log level", () => {
      logger.setLevel(LogLevel.DEBUG);
      expect(logger.getLevel()).toBe(LogLevel.DEBUG);

      logger.setLevel(LogLevel.ERROR);
      expect(logger.getLevel()).toBe(LogLevel.ERROR);
    });

    test("correctly checks if level is enabled", () => {
      logger.setLevel(LogLevel.WARN);

      expect(logger.isLevelEnabled(LogLevel.ERROR)).toBe(true);
      expect(logger.isLevelEnabled(LogLevel.WARN)).toBe(true);
      expect(logger.isLevelEnabled(LogLevel.INFO)).toBe(false);
      expect(logger.isLevelEnabled(LogLevel.DEBUG)).toBe(false);
    });
  });

  describe("Logging Methods and Colors", () => {
    test("logs errors in red", () => {
      logger.error("Test error");
      expect(mockConsole.error).toHaveBeenCalledWith("RED:\nTest error");
    });

    test("logs warnings in yellow", () => {
      logger.warn("Test warning");
      expect(mockConsole.warn).toHaveBeenCalledWith("YELLOW:\nTest warning");
    });

    test("logs info in white", () => {
      logger.info("Test info");
      expect(mockConsole.info).toHaveBeenCalledWith("WHITE:\nTest info");
    });

    test("logs success in green", () => {
      logger.success("Test success");
      expect(mockConsole.log).toHaveBeenCalledWith("GREEN:\nTest success");
    });

    test("logs debug in gray", () => {
      logger.setLevel(LogLevel.DEBUG);
      logger.debug("Test debug");
      expect(mockConsole.debug).toHaveBeenCalledWith("GRAY:\nTest debug");
    });

    test("handles multiple arguments with consistent coloring", () => {
      logger.error("Error:", "Multiple", "Args");
      expect(mockConsole.error).toHaveBeenCalledWith("RED:\nError: Multiple Args");
    });

    test("does not log when level is too low", () => {
      logger.setLevel(LogLevel.ERROR);
      logger.debug("test debug");
      logger.info("test info");
      logger.warn("test warn");

      expect(mockConsole.debug).not.toHaveBeenCalled();
      expect(mockConsole.info).not.toHaveBeenCalled();
      expect(mockConsole.warn).not.toHaveBeenCalled();
    });
  });

  describe("Message Formatting", () => {
    test("handles Error objects", () => {
      const error = new Error("test error");
      logger.error(error);
      expect(mockConsole.error).toHaveBeenCalledWith(expect.stringContaining("RED:\nError: test error"));
    });

    test("handles objects", () => {
      const obj = { test: "value" };
      logger.info(obj);
      expect(mockConsole.info).toHaveBeenCalledWith("WHITE:\n[object Object]");
    });
  });

  describe("Break Method", () => {
    test("logs a blank line", () => {
      logger.break();
      expect(mockConsole.log).toHaveBeenCalledWith("");
    });

    test("does not log break when level is below INFO", () => {
      logger.setLevel(LogLevel.ERROR);
      logger.break();
      expect(mockConsole.log).not.toHaveBeenCalled();
    });
  });
});
