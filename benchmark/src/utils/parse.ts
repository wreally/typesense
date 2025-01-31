import type { ErrorWithMessage } from "@/utils/error";
import type { Ora } from "ora";
import type { z, ZodObject, ZodRawShape, ZodType } from "zod";

import { cosmiconfig } from "cosmiconfig";
import { errAsync, okAsync, ResultAsync } from "neverthrow";

import { toErrorWithMessage } from "@/utils/error";
import { logger } from "@/utils/logger";

export function parseOptions<T extends ZodType>(
  options: Record<string, unknown>,
  schema: T,
  spinner: Ora,
): ResultAsync<z.infer<T>, ErrorWithMessage> {
  spinner.start("Parsing options");
  const parsedOpts = schema.safeParse(options);

  if (!parsedOpts.success) {
    return errAsync({
      message: parsedOpts.error.errors
        .map((err) => `Error for argument '${err.path.join(".")}': ${err.message}`)
        .join("\n"),
    });
  }

  return okAsync(parsedOpts.data as z.infer<T>);
}

export type TupleOfLength<T, L extends number> = number extends L ? T[] : TupleOfLengthHelper<T, L, []>;

type TupleOfLengthHelper<T, L extends number, R extends unknown[]> =
  R["length"] extends L ? R : TupleOfLengthHelper<T, L, [T, ...R]>;

export function loadConfig<T extends ZodRawShape>({
  configPath,
  schema,
  defaultSchema,
  spinner,
}: {
  configPath?: string;
  schema: ZodObject<T>;
  defaultSchema: z.infer<typeof schema>;
  spinner: Ora;
}): ResultAsync<z.infer<typeof schema>, ErrorWithMessage> {
  const explorer = cosmiconfig("typesense-benchmark", {
    searchPlaces: [
      "package.json",
      ".typesense-benchmarkrc",
      ".typesense-benchmarkrc.json",
      ".typesense-benchmarkrc.yaml",
      ".typesense-benchmarkrc.yml",
      ".typesense-benchmarkrc.js",
      "typesense-benchmark.config.js",
    ],
  });

  return ResultAsync.fromPromise(
    configPath ? explorer.load(configPath) : explorer.search(),
    toErrorWithMessage,
  ).andThen((result) => {
    if (!result) {
      logger.info("No config file found. Using default configuration.");
      return okAsync(defaultSchema);
    }

    return parseOptions(result.config as Record<string, unknown>, schema, spinner);
  });
}
