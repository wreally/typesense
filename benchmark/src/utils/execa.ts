import type { ErrorWithMessage } from "@/utils/error";
import type { Result as ExecaResult, Options } from "execa";

import { execa } from "execa";
import { ResultAsync } from "neverthrow";

import { toErrorWithMessage } from "@/utils/error";

function safeExeca(
  command: string,
  args: string[],
  opts?: Options,
): ResultAsync<ExecaResult<Options>, ErrorWithMessage> {
  return ResultAsync.fromPromise(execa(command, args, opts), toErrorWithMessage);
}

type StdOut = ExecaResult["stdout"];

export { safeExeca };

export type { StdOut };
