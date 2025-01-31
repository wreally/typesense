import { ZodError } from "zod";

import { logger } from "./logger";

interface ErrorWithMessage {
  message: string;
}

function isErrorWithMessage(error: unknown): error is ErrorWithMessage {
  return (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof (error as Record<string, unknown>).message === "string"
  );
}

function toErrorWithMessage(couldBeError: unknown): ErrorWithMessage {
  if (isErrorWithMessage(couldBeError)) return couldBeError;

  try {
    if (typeof couldBeError === "string") {
      return new Error(couldBeError);
    }
    return new Error(JSON.stringify(couldBeError));
  } catch {
    return new Error(String(couldBeError));
  }
}

function getErrorMessage(error: unknown) {
  return toErrorWithMessage(error).message;
}

function handleZodError(error: ZodError) {
  error.errors.map((err) => logger.error(`error for argument '${err.path.join(".")}': ${err.message}`));
}

function handleError(error: unknown) {
  if (error instanceof ZodError) {
    handleZodError(error);
  } else {
    logger.error(getErrorMessage(error));
  }

  process.exit(1);
}

export { handleError, toErrorWithMessage };
export type { ErrorWithMessage };
