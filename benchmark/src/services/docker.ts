import path from "path";
import type { FilesystemService } from "@/services/fs";
import type { ErrorWithMessage } from "@/utils/error";
import type { StdOut } from "@/utils/execa";
import type { Ora } from "ora";

import chalk from "chalk";
import { err, errAsync, ok, ResultAsync } from "neverthrow";
import ora from "ora";

import { getPlatform } from "@/services/fs";
import { safeExeca as execaAsync } from "@/utils/execa";
import { logger } from "@/utils/logger";

export class DockerService {
  private readonly rootDir: string;

  constructor(
    private spinner: Ora,
    private fsService: FilesystemService,
    rootDir: string,
  ) {
    this.rootDir = rootDir;
  }

  execInContainer(containerName: string, command: string): ResultAsync<StdOut, ErrorWithMessage> {
    logger.debug(`Executing in container ${containerName}: ${command}`);
    return execaAsync("docker", ["exec", containerName, "bash", "-c", command]).map(({ stdout }) => stdout);
  }

  stopContainer(containerName: string): ResultAsync<void, ErrorWithMessage> {
    this.spinner.color = "white";

    const args = ["stop", containerName];
    logger.debug(`Stopping container with command: docker ${args.join(" ")}`);

    this.spinner.start("Stopping container");

    return execaAsync("docker", args).map(() => {
      this.spinner.succeed(`Stopped Docker container ${containerName}`);
    });
  }

  stopContainers(containerNames: string[]): ResultAsync<void, ErrorWithMessage> {
    this.spinner.start("Stopping containers");

    const childSpinners = containerNames.map(() => {
      const spinner = ora();
      spinner.indent = 2;
      return spinner;
    });

    return ResultAsync.combine(
      containerNames.map((containerName, index) => {
        const childSpinner = childSpinners[index];
        childSpinner?.start(`Stopping ${containerName}`);

        return this.stopContainer(containerName).map(() => {
          childSpinner?.succeed();
        });
      }),
    ).map(() => {
      this.spinner.succeed(`Containers ${containerNames.join(" and ")} stopped`);
    });
  }

  validateImage(imageName: string): ResultAsync<string, ErrorWithMessage> {
    this.spinner.start("Verifying image");

    const args = ["inspect", "--format", "{{.Id}}", imageName];
    logger.debug(`\nVerifying image with command: docker ${args.join(" ")}`);

    return execaAsync("docker", args)
      .map(() => {
        this.spinner.succeed(`Image ${imageName} verified`);
        return imageName;
      })
      .orElse((error) => (error.message.includes("No such object") ? this.buildImage(imageName) : err(error)));
  }

  validateContainer(options: {
    containerName: string;
    imageName: string;
  }): ResultAsync<{ containerName: string; imageName: string }, ErrorWithMessage> {
    this.spinner.start("Verifying container");

    const args = ["inspect", "--format", "{{.Config.Image}}", options.containerName];
    logger.debug(`Verifying container with command: docker ${args.join(" ")} for image ${options.imageName}`);

    return execaAsync("docker", args)
      .andThen(({ stdout }) => {
        if (stdout !== options.imageName) {
          return err({
            message: `Container ${options.containerName} is not using the correct image`,
          });
        }

        this.spinner.succeed(`Container ${options.containerName} verified`);
        return ok(options);
      })
      .orElse((error) =>
        error.message.includes("No such object") ? this.buildContainer(options).map(() => options) : err(error),
      );
  }

  copyFromContainer(options: {
    containerName: string;
    source: string;
    destination: string;
  }): ResultAsync<void, ErrorWithMessage> {
    const { containerName, source, destination } = options;

    const args = ["cp", `${containerName}:${source}`, destination];
    logger.debug(`Copying from container with command: docker ${args.join(" ")}`);

    this.spinner.start("Copying from container");

    return execaAsync("docker", args).map(() => {
      this.spinner.succeed(`Copied from container ${containerName}`);
    });
  }

  startContainer(containerName: string): ResultAsync<void, ErrorWithMessage> {
    this.spinner.start("Starting container");

    const args = ["start", containerName];
    logger.debug(`\nStarting container with command: docker ${args.join(" ")}`);

    return execaAsync("docker", args).map(() => {
      this.spinner.succeed(`Container ${containerName} started`);
    });
  }

  private buildImage(imageName: string): ResultAsync<string, ErrorWithMessage> {
    const platform = getPlatform();

    this.spinner.text = chalk.yellow(`Image not found. Building image ${imageName} with platform ${platform}`);

    const args = [
      "buildx",
      "build",
      "--load",
      "--platform",
      platform,
      "--build-arg",
      `TARGETPLATFORM=linux/${platform.split("/")[1]}`,
      "-t",
      imageName,
      this.rootDir,
    ];

    logger.debug(`\nBuilding image with command: docker ${args.join(" ")}`);

    return execaAsync("docker", args).map(() => {
      this.spinner.succeed(`Image ${imageName} built`);
      return imageName;
    });
  }

  private buildContainer(options: { containerName: string; imageName: string }): ResultAsync<
    {
      containerName: string;
      imageName: string;
    },
    ErrorWithMessage
  > {
    this.spinner.color = "yellow";
    this.spinner.text = chalk.yellow(`Container not found. Building container ${options.containerName}...`);

    const absolutePath = path.resolve(`${options.containerName}-typesense`);

    return this.fsService.exists(absolutePath).andThen((exists) => {
      if (!exists) {
        return errAsync({ message: "the path does not exist" });
      }

      const args = ["create", "-it", "--name", options.containerName, "-v", `${absolutePath}:/app`, options.imageName];

      logger.debug(`Building container with command: docker ${args.join(" ")}`);
      return execaAsync("docker", args, { cwd: this.rootDir }).map(() => {
        this.spinner.succeed(`Container ${options.containerName} built`);
        return options;
      });
    });
  }
}
