import path from "path";
import type { DockerService } from "@/services/docker";
import type { ErrorWithMessage } from "@/utils/error";
import type { StdOut } from "@/utils/execa";
import type { ResultAsync } from "neverthrow";
import type { Ora } from "ora";

import { errAsync, okAsync } from "neverthrow";

import { DEFAULT_TYPESENSE_DIRECTORY } from "@/services/typesense-dir";
import { safeExeca } from "@/utils/execa";
import { logger } from "@/utils/logger";

export const DEFAULT_TYPESENSE_GIT_URL = "https://github.com/typesense/typesense.git";

export class GitService {
  private readonly url: string;
  private readonly directory: string;

  constructor(
    private spinner: Ora,
    url?: string,
    directory?: string,
    private execContext?: {
      containerName: string;
      dockerService: DockerService;
    },
  ) {
    this.url = url ?? DEFAULT_TYPESENSE_GIT_URL;
    this.directory = directory ?? DEFAULT_TYPESENSE_DIRECTORY;
  }

  private execCommand(args: string[], isClone?: true): ResultAsync<StdOut, ErrorWithMessage> {
    if (this.execContext) {
      return this.execContext.dockerService.execInContainer(this.execContext.containerName, `git ${args.join(" ")}`);
    }

    // Use process.cwd() only for clone operations, this.directory for everything else
    const cwd = isClone ? process.cwd() : this.directory;

    logger.debug(`Executing locally: git ${args.join(" ")}`);
    return safeExeca("git", args, { cwd }).map(({ stdout }) => stdout);
  }

  checkoutToCommit(commitHash?: string): ResultAsync<string, ErrorWithMessage> {
    this.spinner.color = "cyan";

    return this.execCommand(["fetch", "origin"])
      .andThen(() =>
        this.execCommand(["pull", "origin"]).orElse((err) => {
          if (err.message.includes("You are not currently on a branch")) {
            return this.switchBack().andThen(() => this.execCommand(["pull", "origin"]));
          }
          return errAsync(err);
        }),
      )
      .andThen(() => {
        return commitHash ? okAsync(commitHash) : this.revParseHead().map((head) => head);
      })
      .andThen((targetCommit) => {
        this.spinner.start(`Switching to commit ${targetCommit}`);
        return this.execCommand(["checkout", targetCommit]).andThen(() =>
          this.revParseHead().andThen((currentCommit) => {
            if (commitHash) {
              const minLen = Math.min(commitHash.length, currentCommit.length);

              const trimmedCurrentCommit = currentCommit.slice(0, minLen);
              const trimmedCommitHash = commitHash.slice(0, minLen);

              if (trimmedCurrentCommit !== trimmedCommitHash) {
                return this.switchBack().andThen(() => errAsync({ message: "Failed to checkout to commit" }));
              }
            }

            this.spinner.succeed(`Checked out to ${commitHash ? "commit" : "HEAD"} ${targetCommit}`);
            return okAsync(targetCommit);
          }),
        );
      });
  }

  switchBack(): ResultAsync<void, ErrorWithMessage> {
    this.spinner.start("Switching back to the original commit");

    return this.execCommand(["switch", "-"]).map(() => {
      this.spinner.succeed("Switched back to the original commit");
    });
  }

  revParseHead(): ResultAsync<string, ErrorWithMessage> {
    const args = ["rev-parse", "HEAD"];
    logger.debug(`Getting HEAD commit with command: git ${args.join(" ")}`);
    this.spinner.start("Getting HEAD commit");

    return this.execCommand(args).andThen((stdout) => {
      if (typeof stdout === "string") {
        logger.debug(`HEAD commit is: ${stdout}`);
        return okAsync(stdout);
      }

      return errAsync({ message: "Failed to get HEAD commit" });
    });
  }

  markDirectoryAsSafe(directory: string): ResultAsync<string, ErrorWithMessage> {
    const args = ["config", "--global", "--add", "safe.directory", directory];
    logger.debug(`Marking directory as safe with command: git ${args.join(" ")}`);

    this.spinner.start("Marking directory as safe");

    return this.execCommand(args).map(() => {
      this.spinner.succeed("Directory marked as safe");
      return directory;
    });
  }

  cloneRepository(): ResultAsync<string, ErrorWithMessage> {
    const args = ["clone", this.url, this.directory];
    logger.debug(`Cloning into repository with command: git ${args.join(" ")}`);

    this.spinner.start("Cloning git repository");
    return this.execCommand(args, true).map(() => {
      this.spinner.succeed("Repository cloned successfully");
      return path.resolve(this.directory);
    });
  }
}
