import { access, mkdir, readdir, rm, writeFile } from "fs/promises";
import path from "path";
import { Readable } from "stream";
import type { ErrorWithMessage } from "@/utils/error";
import type { Ora } from "ora";

import { emptyDir } from "fs-extra";
import gunzip from "gunzip-maybe";
import inquirer from "inquirer";
import { errAsync, okAsync, ResultAsync } from "neverthrow";
import { extract } from "tar-stream";

import { toErrorWithMessage } from "@/utils/error";

type PlatformFlag = "linux/amd64" | "linux/arm64" | "darwin/amd64" | "windows/amd64";

export function getPlatform(): PlatformFlag {
  const plat = process.arch;

  switch (plat) {
    case "x64":
      return "linux/amd64";
    case "arm":
    case "arm64":
      return "linux/arm64";
    default:
      return "linux/amd64";
  }
}

export class FilesystemService {
  constructor(
    private spinner: Ora,
    private yesToAll?: boolean,
  ) {}

  exists(directory: string, mode?: number): ResultAsync<boolean, ErrorWithMessage> {
    this.spinner.start("Checking directory existence");

    return ResultAsync.fromPromise(
      access(directory, mode)
        .then(() => true)
        .catch(() => false),
      toErrorWithMessage,
    ).map((exists) => {
      return exists;
    });
  }

  removeDirectory(directory: string): ResultAsync<void, ErrorWithMessage> {
    this.spinner.start(`Removing directory ${directory}`);
    return ResultAsync.fromPromise(rm(directory, { recursive: true }), toErrorWithMessage).map(() => {
      this.spinner.succeed(`Removed directory ${directory}`);
    });
  }

  createDirectory(directory: string): ResultAsync<string, ErrorWithMessage> {
    this.spinner.start(`Creating directory ${directory}`);

    const resolved = path.resolve(directory);

    return this.exists(resolved).andThen((exists) => {
      if (exists) {
        if (this.yesToAll) {
          this.spinner.text = `Emptying directory ${resolved}`;
          return ResultAsync.fromPromise(emptyDir(directory), toErrorWithMessage).map(() => {
            this.spinner.succeed(`Directory ${directory} emptied`);

            return resolved;
          });
        }

        this.spinner.stop();
        return ResultAsync.fromPromise(
          inquirer.prompt([
            {
              type: "confirm",
              name: "overwrite",
              message: `Directory ${resolved} already exists. Overwrite?`,
            },
          ]),
          toErrorWithMessage,
        ).andThen((answer) => {
          if (!answer.overwrite) {
            this.spinner.succeed(`Using existing directory ${resolved}`);
            return okAsync(resolved);
          }

          this.spinner.text = `Emptying directory ${resolved}`;
          return ResultAsync.fromPromise(emptyDir(directory), toErrorWithMessage).map(() => {
            this.spinner.succeed(`Directory ${directory} emptied`);

            return resolved;
          });
        });
      }

      return ResultAsync.fromPromise(mkdir(resolved, { recursive: true }), toErrorWithMessage).andThen((res) => {
        if (res === undefined) {
          this.spinner.fail(`Failed to create directory ${directory}`);
          return errAsync({
            message: `Failed to create directory ${directory}`,
          });
        }
        this.spinner.succeed(`Created directory ${directory}`);
        return okAsync(resolved);
      });
    });
  }

  getAbsolutePath(directory: string): string {
    return path.resolve(directory);
  }

  private handleNonEmptyDirectory(directory: string): ResultAsync<string, ErrorWithMessage> {
    if (this.yesToAll) {
      return this.emptyDirectory(directory);
    }

    return ResultAsync.fromPromise(
      inquirer.prompt([
        {
          type: "confirm",
          name: "emptyDir",
          message: `Directory ${directory} is not empty. Empty it?`,
          default: true,
        },
      ]),
      toErrorWithMessage,
    ).andThen((answer) => {
      if (!answer.emptyDir) {
        this.spinner.warn(`Directory ${directory} is not empty. Continuing may overwrite existing files.`);
        return okAsync(directory);
      }

      return this.emptyDirectory(directory);
    });
  }

  private emptyDirectory(directory: string): ResultAsync<string, ErrorWithMessage> {
    this.spinner.text = `Emptying directory ${directory}`;
    return ResultAsync.fromPromise(emptyDir(directory), toErrorWithMessage).map(() => {
      this.spinner.succeed(`Directory ${directory} emptied`);
      return directory;
    });
  }

  private decompressStream(buffer: ArrayBuffer): ResultAsync<void, ErrorWithMessage> {
    this.spinner.start("Decompressing dataset");

    return ResultAsync.fromPromise(
      new Promise<string>((resolve, reject) => {
        const extraction = extract();
        let jsonContent = "";

        extraction.on("entry", (header, stream, next) => {
          if (header.name.endsWith(".jsonl")) {
            stream.on("data", (chunk) => {
              jsonContent += chunk;
            });
          }
          stream.on("end", () => next());
          stream.resume();
        });

        extraction.on("finish", () => {
          resolve(jsonContent);
        });

        extraction.on("error", (err) => {
          // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
          reject(toErrorWithMessage(err));
        });

        const readable = Readable.from(Buffer.from(buffer));
        readable.pipe(gunzip()).pipe(extraction);
      }),
      toErrorWithMessage,
    )
      .andThen((jsonContent) => {
        return ResultAsync.fromPromise(
          mkdir("data", { recursive: true }).then(() => writeFile("data/data.json", jsonContent)),
          toErrorWithMessage,
        );
      })
      .map(() => {
        this.spinner.succeed("Dataset decompressed and saved successfully");
      });
  }

  downloadTypesenseDataset(url: string): ResultAsync<void, ErrorWithMessage> {
    this.spinner.start(`Downloading dataset from ${url}`);

    return ResultAsync.fromPromise(fetch(url), toErrorWithMessage)
      .andThen((response) => {
        if (!response.ok) {
          return errAsync({
            message: `Failed to download dataset: ${response.statusText}`,
          });
        }

        return ResultAsync.fromPromise(response.arrayBuffer(), toErrorWithMessage);
      })
      .andThen((buffer) => this.decompressStream(buffer))
      .map(() => {
        this.spinner.succeed(`Dataset downloaded successfully`);
      });
  }

  private handleNonExistingDirectory(directory: string): ResultAsync<string, ErrorWithMessage> {
    if (this.yesToAll) {
      return ResultAsync.fromPromise(mkdir(directory, { recursive: true }), toErrorWithMessage).map(() => directory);
    }

    return ResultAsync.fromPromise(
      inquirer.prompt([
        {
          type: "confirm",
          name: "createDir",
          message: `Directory ${directory} does not exist. Create directory?`,
          default: true,
        },
      ]),
      toErrorWithMessage,
    ).andThen((answer) => {
      if (!answer.createDir) {
        return errAsync({
          message: `Directory ${directory} does not exist.`,
        });
      }

      this.spinner.text = `Save directory ${directory} does not exist. Creating directory`;
      return ResultAsync.fromPromise(mkdir(directory, { recursive: true }), toErrorWithMessage).map(() => directory);
    });
  }

  validateWorkingDirectory(directory: string): ResultAsync<string, ErrorWithMessage> {
    const resolved = path.resolve(directory);

    return this.exists(resolved).andThen((exists) => {
      if (exists) {
        this.spinner.succeed(`Working directory verified: ${resolved}`);
        return ResultAsync.fromPromise(readdir(resolved), toErrorWithMessage).andThen((files) => {
          if (files.length > 0) {
            return this.handleNonEmptyDirectory(resolved);
          }
          return okAsync(resolved);
        });
      }

      this.spinner.stop();
      return this.handleNonExistingDirectory(resolved);
    });
  }
}
