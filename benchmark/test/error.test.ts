/* eslint-disable @typescript-eslint/unbound-method */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { handleError } from "../src/utils/error";
import { logger } from "../src/utils/logger";

vi.mock("../src/utils/logger", () => ({
  logger: {
    error: vi.fn(),
  },
}));

describe("Error Handler", () => {
  const originalExit = process.exit;

  beforeEach(() => {
    vi.clearAllMocks();
    process.exit = vi.fn() as never;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    process.exit = originalExit;
  });

  describe("handleError", () => {
    it("logs the error message and exits the process", () => {
      const errorMessage = "Test error message";
      handleError(new Error(errorMessage));

      expect(logger.error).toHaveBeenCalledWith(errorMessage);
      expect(process.exit).toHaveBeenCalledWith(1);
    });

    it("handles string errors", () => {
      const errorMessage = "String error";
      handleError(errorMessage);

      expect(logger.error).toHaveBeenCalledWith(errorMessage);
      expect(process.exit).toHaveBeenCalledWith(1);
    });

    it("handles number errors", () => {
      const errorNumber = 42;
      handleError(errorNumber);

      expect(logger.error).toHaveBeenCalledWith("42");
      expect(process.exit).toHaveBeenCalledWith(1);
    });

    it("handles object errors", () => {
      const errorObject = { custom: "error" };
      handleError(errorObject);

      expect(logger.error).toHaveBeenCalledWith('{"custom":"error"}');
      expect(process.exit).toHaveBeenCalledWith(1);
    });

    it("handles null errors", () => {
      handleError(null);

      expect(logger.error).toHaveBeenCalledWith("null");
      expect(process.exit).toHaveBeenCalledWith(1);
    });

    it("handles undefined errors", () => {
      handleError(undefined);

      expect(logger.error).toHaveBeenCalledWith("");
      expect(process.exit).toHaveBeenCalledWith(1);
    });

    it("handles errors that cannot be stringified", () => {
      interface CircularReference {
        circularReference: CircularReference;
      }

      const circularReference: CircularReference = {
        circularReference: {} as CircularReference,
      };

      circularReference.circularReference = circularReference;
      handleError(circularReference);

      expect(logger.error).toHaveBeenCalledWith("[object Object]");
      expect(process.exit).toHaveBeenCalledWith(1);
    });

    it("handles ZodError with single error", () => {
      const schema = z.object({
        name: z.string(),
      });
      const result = schema.safeParse({ name: 123 });
      if (!result.success) {
        handleError(result.error);
      }

      expect(logger.error).toHaveBeenCalledWith("error for argument 'name': Expected string, received number");
      expect(process.exit).toHaveBeenCalledWith(1);
    });

    it("handles ZodError with multiple errors", () => {
      const schema = z.object({
        name: z.string(),
        age: z.number().positive(),
      });
      const result = schema.safeParse({ name: 123, age: -5 });
      if (!result.success) {
        handleError(result.error);
      }

      expect(logger.error).toHaveBeenCalledTimes(2);
      expect(logger.error).toHaveBeenNthCalledWith(1, "error for argument 'name': Expected string, received number");
      expect(logger.error).toHaveBeenNthCalledWith(2, "error for argument 'age': Number must be greater than 0");
      expect(process.exit).toHaveBeenCalledWith(1);
    });

    it("handles ZodError with nested path", () => {
      const schema = z.object({
        user: z.object({
          details: z.object({
            age: z.number().positive(),
          }),
        }),
      });

      const result = schema.safeParse({ user: { details: { age: -5 } } });

      if (!result.success) {
        handleError(result.error);
      }

      expect(logger.error).toHaveBeenCalledWith("error for argument 'user.details.age': Number must be greater than 0");
      expect(process.exit).toHaveBeenCalledWith(1);
    });
  });
});
