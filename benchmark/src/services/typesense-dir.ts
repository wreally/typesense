import type { DockerService } from "@/services/docker";
import type { FilesystemService } from "@/services/fs";
import type { GitService } from "@/services/git";
import type { ErrorWithMessage } from "@/utils/error";
import type { Ora } from "ora";

import inquirer from "inquirer";
import { ResultAsync } from "neverthrow";

import { toErrorWithMessage } from "@/utils/error";

export const DEFAULT_TYPESENSE_DIRECTORY = "typesense";

interface GitOptions {
  typesenseGitUrl: string;
  yesToAll?: boolean;
}

export class TypesenseDirectoryManager {
  private readonly directory: string;

  constructor(
    private gitService: GitService,
    private filesystemService: FilesystemService,
    private dockerService: DockerService,
    private spinner: Ora,
    private gitOptions: GitOptions,
    directory?: string,
  ) {
    this.directory = directory ?? DEFAULT_TYPESENSE_DIRECTORY;
  }

  validateDirectory(): ResultAsync<void, ErrorWithMessage> {
    this.spinner.start("Starting directory validation");
    return this.filesystemService.exists(this.directory).andThen((exists) => {
      if (exists) {
        return this.handleExistingDirectory();
      }
      return this.cloneAndVerifyRepository();
    });
  }

  buildAndSave(options: {
    containerName: string;
    destination: string;
    commitHash: string;
  }): ResultAsync<string, ErrorWithMessage> {
    return this.build(options.containerName)
      .andThen(() => this.save(options).map((res) => res))
      .andThen((res) => this.filesystemService.removeDirectory(this.directory).map(() => res));
  }

  private save(options: {
    containerName: string;
    destination: string;
    commitHash: string;
  }): ResultAsync<string, ErrorWithMessage> {
    const { containerName, destination, commitHash } = options;

    this.spinner.start("Saving Typesense build");

    const fullDestination = `${destination}/typesense-server-${commitHash}`;

    return this.dockerService
      .copyFromContainer({
        containerName,
        destination: fullDestination,
        source: "/app/bazel-bin/typesense-server",
      })
      .map(() => fullDestination);
  }

  private build(containerName: string): ResultAsync<void, ErrorWithMessage> {
    this.spinner.start("Building Typesense");
    return this.dockerService
      .execInContainer(
        containerName,
        "bazel build @com_google_protobuf//:protobuf_headers && bazel build @com_google_protobuf//:protobuf_lite && bazel build @com_google_protobuf//:protobuf && bazel build @com_google_protobuf//:protoc && bazel build --verbose_failures --jobs=6 //:typesense-server",
      )
      .map(() => {
        this.spinner.succeed("Typesense built successfully");
      });
  }

  private handleExistingDirectory(): ResultAsync<void, ErrorWithMessage> {
    if (this.gitOptions.yesToAll) {
      return this.forceRecreateDirectory();
    }

    this.spinner.stop();

    return ResultAsync.fromPromise(
      inquirer.prompt([
        {
          type: "confirm",
          name: "rmDir",
          message: `Typesense directory ${this.directory} already exists. Do you want to delete it and clone a fresh copy?`,
          default: true,
        },
      ]),
      toErrorWithMessage,
    ).andThen(({ rmDir }) => (rmDir ? this.forceRecreateDirectory() : this.verifyExistingDirectory()));
  }

  private forceRecreateDirectory(): ResultAsync<void, ErrorWithMessage> {
    return this.filesystemService.removeDirectory(this.directory).andThen(() => {
      this.spinner.start("Preparing to clone repository");
      return this.cloneAndVerifyRepository();
    });
  }

  private cloneAndVerifyRepository(): ResultAsync<void, ErrorWithMessage> {
    return this.gitService.cloneRepository().andThen(() => this.verifyExistingDirectory());
  }

  private verifyExistingDirectory(): ResultAsync<void, ErrorWithMessage> {
    return this.gitService.revParseHead().map(() => {
      this.spinner.succeed("Directory validation complete");
    });
  }
}
