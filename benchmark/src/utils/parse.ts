import type { ErrorWithMessage } from "@/utils/error";
import type { ResultAsync } from "neverthrow";
import type { Ora } from "ora";
import type { z, ZodType } from "zod";

import { errAsync, okAsync } from "neverthrow";

function parseOptions<T extends ZodType>(
  options: Record<string, unknown>,
  schema: T,
  spinner: Ora,
): ResultAsync<z.infer<T>, ErrorWithMessage> {
  spinner.start("Parsing options");
  const parsedOpts = schema.safeParse(options);

  if (!parsedOpts.success) {
    return errAsync({
      message: parsedOpts.error.errors
        .map(
          (err) => `Error for argument '${err.path.join(".")}': ${err.message}`,
        )
        .join("\n"),
    });
  }

  return okAsync(parsedOpts.data as z.infer<T>);
}

export { parseOptions };
