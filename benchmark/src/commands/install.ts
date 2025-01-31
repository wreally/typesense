import { Command } from "commander";
import { ok } from "neverthrow";
import { z } from "zod";

import { ServiceContainer } from "@/services/container";
import { DEFAULT_TYPESENSE_GIT_URL } from "@/services/git";
import { logger, LogLevel } from "@/utils/logger";
import { dirName, findRoot } from "@/utils/package-info";
import { parseOptions } from "@/utils/parse";

const installOptionsSchema = z.object({
  containerName: z.string(),
  imageName: z.string(),
  typesenseGitUrl: z.string(),
  workingDirectory: z.string(),
  commitHash: z.string().optional(),
  verbose: z.boolean(),
  yes: z.boolean(),
});

type InstallOptions = z.infer<typeof installOptionsSchema>;

const cwd = process.cwd();

const install = new Command()
  .name("install")
  .description("Install a specific Typesense version")
  .option("-n, --container-name <name>", "Name for the Docker container. Defaults to bazel-build", "bazel-build")
  .option("-i, --image-name <image>", "Name for the Docker image. Defaults to ubuntu-build", "ubuntu-build")
  .option(
    "-g, --typesense-git-url <url>",
    "Git URL for the Typesense repo. Defaults to the main Typesense github repo",
    DEFAULT_TYPESENSE_GIT_URL,
  )
  .option(
    "-d, --working-directory <dir>",
    "Directory where the Typesense repo is saved. Defaults to the current directory",
    cwd,
  )
  .option("-c, --commitHash <commit-hash>", "Hash of the commit to install. Defaults to the latest commit")
  .option("-y, --yes", "Answer yes to all prompts", false)
  .option("-v, --verbose", "Verbose output", false)
  .action((options) => {
    logger.info("Installing Typesense");

    const services = new ServiceContainer(findRoot(dirName));
    const spinner = services.getSpinner();

    parseOptions(options as InstallOptions, installOptionsSchema, spinner)
      .andThen((options) => {
        if (options.verbose) {
          logger.setLevel(LogLevel.DEBUG);
        }

        logger.debug("Parsed options");
        return ok(options);
      })
      .andThen((options) => {
        services.initialize({
          directory: options.workingDirectory,
          containerName: options.containerName,
          gitUrl: options.typesenseGitUrl,
          yesToAll: options.yes,
        });

        return services
          .get("typesense")
          .validateDirectory()
          .map(() => options);
      })
      .andThen((options) =>
        services
          .get("fs")
          .validateWorkingDirectory(options.workingDirectory)
          .map(() => options),
      )
      .andThen((options) =>
        services
          .get("docker")
          .validateImage(options.imageName)
          .map(() => options),
      )
      .andThen((options) =>
        services
          .get("docker")
          .validateContainer({
            containerName: options.containerName,
            imageName: options.imageName,
          })
          .map(() => options),
      )
      .andThen((options) =>
        services
          .get("docker")
          .startContainer(options.containerName)
          .map(() => options),
      )
      .andThen((options) =>
        services
          .get("containerGit")
          .markDirectoryAsSafe(options.workingDirectory)
          .map(() => options),
      )
      .andThen((options) =>
        services
          .get("containerGit")
          .checkoutToCommit(options.commitHash)
          .map((commitHash) => ({ ...options, commitHash })),
      )
      .andThen((options) =>
        services
          .get("typesense")
          .buildAndSave({
            commitHash: options.commitHash,
            containerName: options.containerName,
            destination: options.workingDirectory,
          })
          .map(() => options),
      )
      .andThen((options) => services.get("docker").stopContainer(options.containerName))
      .then((res) => {
        if (res.isOk()) {
          logger.success("Typesense installed successfully");
          process.exit(0);
        } else {
          spinner.fail();
          logger.error(res.error.message);
          process.exit(1);
        }
      });
  });

export { install };
