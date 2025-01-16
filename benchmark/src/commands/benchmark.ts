import path from "path";
import type { ErrorWithMessage } from "@/utils/error";
import type { NumericIndices } from "@/utils/types";
import type { IResults } from "influx";
import type { Ora } from "ora";
import type { TableUserConfig } from "table";

import chalk from "chalk";
import { Command } from "commander";
import { ps, upMany } from "docker-compose";
import { InfluxDB } from "influx";
import { errAsync, ok, okAsync, ResultAsync } from "neverthrow";
import { table } from "table";
import { z } from "zod";

import { ServiceContainer } from "@/services/container";
import { DEFAULT_TYPESENSE_GIT_URL } from "@/services/git";
import { K6Benchmarks } from "@/services/k6";
import { TypesenseProcessManager } from "@/services/typesense-process";
import { toErrorWithMessage } from "@/utils/error";
import { logger, LogLevel } from "@/utils/logger";
import { dirName, findRoot } from "@/utils/package-info";
import { parseOptions } from "@/utils/parse";

const cwd = process.cwd();

interface FormattedResults {
  metric: string;
  variable: string;
  oldValue: string;
  newValue: string;
  percentageChange: number;
  formattedPercentageChange: string;
}

const benchmarkOptionSchema = z.object({
  containerName: z.string(),
  imageName: z.string(),
  typesenseGitUrl: z.string(),
  workingDirectory: z.string(),
  commitHashes: z.tuple([z.string(), z.string()]),
  verbose: z.boolean(),
  yes: z.boolean(),
  binaries: z.tuple([z.string(), z.string()]),
  apiKey: z.string(),
  fail: z
    .string()
    .optional()
    .transform((value) => (value ? parseInt(value) : null))
    .pipe(z.number().optional()),
  batchSize: z
    .string()
    .transform((value) => parseInt(value))
    .pipe(z.number().min(0)),
  duration: z.string().refine(
    (value) => {
      return /^\d+[smhd]$/.test(value);
    },
    { message: "Duration must be in the format of <number><s/m/h/d>" },
  ),
});

type IntegrationTestOptions = z.infer<typeof benchmarkOptionSchema>;

type BenchmarkResults = Record<
  string,
  {
    searchBenchmark: {
      p95: number;
      vus: number;
      scenario: string;
    }[];
    indexDuration: number;
  }
>;

interface BenchmarkGroup {
  processManager: TypesenseProcessManager;
  k6Benchmark: K6Benchmarks;
  dataDirectory: string;
}

class Benchmarks {
  private readonly typesenseProcessManagers: [TypesenseProcessManager, TypesenseProcessManager];
  private readonly services: ServiceContainer;
  private readonly workingDirectory: string;
  private readonly batchSize: number;
  private readonly duration: string;
  private readonly apiKey: string;
  private readonly isInCi: boolean;
  private readonly commitHashes: [string, string];
  private readonly failAtPercentage?: number;
  private readonly port: number;
  private readonly spinner: Ora;
  private readonly benchmarkGroupsByCommitHash: Record<string, BenchmarkGroup>;

  constructor(options: {
    typesenseProcessManagers: [TypesenseProcessManager, TypesenseProcessManager];
    batchSize: number;
    duration: string;
    apiKey: string;
    port: number;
    services: ServiceContainer;
    spinner: Ora;
    workingDirectory: string;
    commitHashes: [string, string];
    failAtPercentage?: number;
  }) {
    this.typesenseProcessManagers = options.typesenseProcessManagers;
    this.batchSize = options.batchSize;
    this.duration = options.duration;
    this.apiKey = options.apiKey;
    this.spinner = options.spinner;
    this.services = options.services;
    this.port = options.port;
    this.workingDirectory = options.workingDirectory;
    this.commitHashes = options.commitHashes;
    this.isInCi = Boolean(process.env.CI) || false;
    this.failAtPercentage = options.failAtPercentage;

    this.benchmarkGroupsByCommitHash = Object.fromEntries(
      this.commitHashes.map((hash, index) => [
        hash,
        {
          processManager: this.typesenseProcessManagers[index as NumericIndices<typeof this.commitHashes>],
          k6Benchmark: new K6Benchmarks({
            batchSize: this.batchSize,
            commitHash: hash,
            spinner: this.spinner,
            typesenseProcessManager: this.typesenseProcessManagers[index as NumericIndices<typeof this.commitHashes>],
            duration: this.duration,
            apiKey: this.apiKey,
            port: this.port,
          }),
          dataDirectory: path.join(this.workingDirectory, hash),
        },
      ]),
    );
  }

  private createDataDirectories(): ResultAsync<string[], ErrorWithMessage> {
    this.spinner.start("Validating data directories");
    return this.services
      .get("fs")
      .validateWorkingDirectory(this.workingDirectory)
      .andThen(() => {
        return ResultAsync.combine(
          Object.values(this.benchmarkGroupsByCommitHash).map((group) =>
            this.services.get("fs").createDirectory(group.dataDirectory),
          ),
        );
      })
      .map((directories) => {
        this.spinner.succeed(`Data directories ${Object.values(directories).join(", ")} created`);
        return directories;
      });
  }

  private startProcess(commitHash: string) {
    this.spinner.start(`Starting Typesense process for ${commitHash}`);
    const benchmarkGroup = this.benchmarkGroupsByCommitHash[commitHash];

    if (!benchmarkGroup) {
      return errAsync(new Error(`No benchmark group found for ${commitHash}`));
    }

    return benchmarkGroup.processManager
      .initNode(benchmarkGroup.dataDirectory, this.port)
      .map((node) => benchmarkGroup.processManager.startProcess(node, { multiNode: false }))
      .map(() => {
        this.spinner.succeed(`Typesense process started for ${commitHash}`);
      });
  }

  private delay(ms: number): ResultAsync<void, ErrorWithMessage> {
    return ResultAsync.fromPromise(new Promise((resolve) => setTimeout(resolve, ms)), toErrorWithMessage);
  }

  private handleResults(results: FormattedResults[]): ResultAsync<FormattedResults[], { message: string }> {
    if (!this.failAtPercentage) {
      return okAsync(results);
    }

    const failingRows = results.filter((row) => row.percentageChange > this.failAtPercentage!);

    if (failingRows.length > 0) {
      const failures = failingRows
        .map((row) => `${row.metric} for ${row.variable} changed by ${row.formattedPercentageChange}`)
        .join("\n");

      return errAsync({
        message: `Performance degradation exceeded threshold of ${this.failAtPercentage}%:\n${failures}`,
      });
    }

    return okAsync(results);
  }

  private getResults(): ResultAsync<BenchmarkResults, { message: string; commitHash?: string }> {
    const influx = new InfluxDB({
      host: "localhost",
      port: 8086,
      database: "k6",
    });

    const indexDurationQuery = `
    SELECT mean("value") AS "mean_import_duration"
    FROM "import_duration"
    WHERE ("commitHash" = '${this.commitHashes[0]}' OR "commitHash" = '${this.commitHashes[1]}')
    AND time >= now() - 24h
    GROUP BY "commitHash"
    `;

    const searchDurationQuery = `
      SELECT PERCENTILE("value", 95) AS "p95_value"
      FROM "search_processing_time_ms"
      WHERE "commitHash" = '${this.commitHashes[0]}' OR "commitHash" = '${this.commitHashes[1]}'
      GROUP BY "scenario", "vus", "commitHash"
    `;

    return ResultAsync.combine([
      ResultAsync.fromPromise(
        influx.query<{
          commitHash: string;
          scenario: string;
          vus: string;
          p95_value: number;
        }>(searchDurationQuery),
        toErrorWithMessage,
      ),
      ResultAsync.fromPromise(
        influx.query<{ commitHash: string; mean_import_duration: number }>(indexDurationQuery),
        toErrorWithMessage,
      ),
    ]).andThen((results) => this.mapResults({ indexResults: results[1], searchResults: results[0] }));
  }

  public startContainers(): ResultAsync<void, ErrorWithMessage> {
    if (this.isInCi) {
      return okAsync(undefined);
    }

    this.spinner.start("Starting containers");
    return ResultAsync.fromPromise(ps({ cwd: findRoot(process.cwd()) }), toErrorWithMessage).andThen((res) => {
      const runningServices = new Set(
        res.data.services.map((service) => {
          const match = /^[^-]+-([^-]+)-\d+$/.exec(service.name);
          return match ? match[1] : service.name;
        }),
      );

      const missingServices = K6Benchmarks.REQUIRED_SERVICES.filter((service) => !runningServices.has(service));

      if (missingServices.length === 0) {
        return okAsync(undefined);
      }

      return ResultAsync.fromPromise(
        upMany(missingServices, { cwd: findRoot(process.cwd()) }),
        toErrorWithMessage,
      ).andThen(() => {
        this.spinner.succeed("Containers started");
        return okAsync(undefined);
      });
    });
  }

  private mapResults(results: {
    searchResults: IResults<{
      commitHash: string;
      scenario: string;
      vus: string;
      p95_value: number;
    }>;
    indexResults: IResults<{
      commitHash: string;
      mean_import_duration: number;
    }>;
  }): ResultAsync<BenchmarkResults, { message: string; commitHash?: string }> {
    // First, validate that we have both search and index results
    if (!results.searchResults || !results.indexResults) {
      return errAsync({
        message: "Missing either search or index results",
      });
    }
    // Check for empty results
    if (results.searchResults.length === 0) {
      return errAsync({
        message: "No search results found",
      });
    }
    if (results.indexResults.length === 0) {
      return errAsync({
        message: "No index results found",
      });
    }

    // Validate all required fields in search results
    for (const result of results.searchResults) {
      if (
        !result.commitHash ||
        !result.scenario ||
        typeof Number(result.vus) !== "number" ||
        typeof result.p95_value !== "number"
      ) {
        return errAsync({
          message: `Incomplete search result data for commit ${result.commitHash || "unknown"}`,
          commitHash: result.commitHash,
        });
      }
    }

    // Validate all required fields in index results
    for (const result of results.indexResults) {
      if (!result.commitHash || typeof result.mean_import_duration !== "number") {
        return errAsync({
          message: `Incomplete index result data for commit ${result.commitHash || "unknown"}`,
          commitHash: result.commitHash,
        });
      }
    }

    // Create a Set of all unique scenario+vus combinations that should exist
    const scenarioSet = new Set<string>();
    for (const { scenario, vus } of results.searchResults) {
      scenarioSet.add(`${scenario} (${vus}vu)`);
    }

    // Group results by commit hash for validation
    const resultsByCommit = new Map<string, Set<string>>();
    for (const { commitHash, scenario, vus } of results.searchResults) {
      if (!resultsByCommit.has(commitHash)) {
        resultsByCommit.set(commitHash, new Set());
      }
      resultsByCommit.get(commitHash)?.add(`${scenario} (${vus}vu)`);
    }

    // Validate that each commit has all scenarios
    for (const [commitHash, scenarios] of resultsByCommit) {
      for (const expectedScenario of scenarioSet) {
        if (!scenarios.has(expectedScenario)) {
          return errAsync({
            message: `Commit ${commitHash} is missing results for scenario: ${expectedScenario}`,
            commitHash,
          });
        }
      }
    }

    const resultMap: BenchmarkResults = {};
    const indexMap = new Map(results.indexResults.map((row) => [row.commitHash, row.mean_import_duration]));

    // Process search results and match with index results
    for (const { commitHash, p95_value, vus, scenario } of results.searchResults) {
      const indexDuration = indexMap.get(commitHash);
      if (indexDuration === undefined) {
        return errAsync({
          message: `Commit ${commitHash} has no value for indexing benchmarks`,
          commitHash,
        });
      }
      if (!resultMap[commitHash]) {
        resultMap[commitHash] = {
          searchBenchmark: [],
          indexDuration,
        };
      }
      resultMap[commitHash].searchBenchmark.push({
        p95: p95_value,
        vus: Number(vus),
        scenario,
      });
    }

    // Validate final result map is not empty
    if (Object.keys(resultMap).length < this.commitHashes.length) {
      return errAsync({
        message: "No valid benchmark results could be mapped",
        commitHash: this.commitHashes.find((hash) => !resultMap[hash]),
      });
    }

    return okAsync(resultMap);
  }
  private calculatePercentageChange(oldValue: number, newValue: number) {
    if (oldValue === 0) {
      return newValue === 0 ? 0 : Infinity; // No change if both 0, Infinite increase if old is 0 and new is positive
    }
    return ((newValue - oldValue) / oldValue) * 100;
  }

  private formatPercentageChange(percentageChange: number): string {
    if (!Number.isFinite(percentageChange)) {
      return chalk.red("+∞%");
    }

    const formattedChange = percentageChange.toFixed(2);
    const sign = percentageChange >= 0 ? "+" : "";

    if (formattedChange === "0.00") {
      return `${formattedChange}%`;
    }

    return chalk[percentageChange < 0 ? "green" : "red"](`${sign}${formattedChange}%`);
  }

  private printTable(results: BenchmarkResults): ResultAsync<FormattedResults[], ErrorWithMessage> {
    const formattedRows: FormattedResults[] = [];
    const importDuration = results[this.commitHashes[0]]?.indexDuration;
    const importDurationNew = results[this.commitHashes[1]]?.indexDuration;

    if (!importDuration || !importDurationNew) {
      return errAsync(new Error("No results found"));
    }

    const importPercentageChange = this.calculatePercentageChange(importDuration, importDurationNew);
    formattedRows.push({
      metric: "Time to bulk import",
      variable: "1M records",
      oldValue: `${(importDuration / 1_000).toFixed(5)}s`,
      newValue: `${(importDurationNew / 1_000).toFixed(5)}s`,
      percentageChange: importPercentageChange,
      formattedPercentageChange: this.formatPercentageChange(importPercentageChange),
    });

    const previousSearchResults = results[this.commitHashes[0]]?.searchBenchmark;
    const newSearchResults = results[this.commitHashes[1]]?.searchBenchmark;

    if (!previousSearchResults || !newSearchResults) {
      return errAsync(new Error("Missing search benchmark results"));
    }

    const previousSearchResultsMap = new Map(
      previousSearchResults.map((result) => [`${result.scenario} (${result.vus}vu)`, result.p95]),
    );
    const newSearchResultsMap = new Map(
      newSearchResults.map((result) => [`${result.scenario} (${result.vus}vu)`, result.p95]),
    );

    for (const [key, oldP95] of previousSearchResultsMap) {
      const newP95 = newSearchResultsMap.get(key);
      if (newP95 === undefined) {
        return errAsync({ message: `${key} doesn't have a value for the search benchmark` });
      }
      const percentageChange = this.calculatePercentageChange(oldP95, newP95);
      formattedRows.push({
        metric: "p95 search_time_ms when searching with",
        variable: key,
        oldValue: `${oldP95}ms`,
        newValue: `${newP95}ms`,
        percentageChange: percentageChange,
        formattedPercentageChange: this.formatPercentageChange(percentageChange),
      });
    }

    const columns: [string, string, string, string, string][] = [
      [
        "Metric",
        "Variable",
        `Value for commit ${this.commitHashes[0].slice(0, 7)}`,
        `Value for commit ${this.commitHashes[1].slice(0, 7)}`,
        "Percentage Change",
      ],
      ...formattedRows.map(
        (row) =>
          [row.metric, row.variable, row.oldValue, row.newValue, row.formattedPercentageChange] as [
            string,
            string,
            string,
            string,
            string,
          ],
      ),
    ];

    const tableConfig: TableUserConfig = {
      columns: {
        0: { alignment: "left", width: 40 },
        1: { alignment: "left", width: 35 },
        2: { alignment: "right", width: 25 },
        3: { alignment: "right", width: 25 },
        4: { alignment: "right", width: 20 },
      },
    };

    logger.info(table(columns, tableConfig));
    return okAsync(formattedRows);
  }

  benchmark() {
    logger.info("Running benchmarks");

    return this.startContainers()
      .andThen(() =>
        this.getResults()
          .orElse((error) => {
            if (error.commitHash) {
              return this.performBenchmarks([error.commitHash]);
            }

            return this.performBenchmarks(this.commitHashes);
          })
          .andThen(() => this.getResults())
          .andThen((res) => this.printTable(res)),
      )
      .andThen((results) => this.handleResults(results));
  }

  private performBenchmarks(commitHashes: string[]) {
    return this.createDataDirectories().andThen(() =>
      commitHashes.reduce(
        (promise, commitHash, index) =>
          promise.andThen(() => {
            this.spinner.start("Running benchmarks");
            return this.startProcess(commitHash).andThen(() => {
              const benchmarkGroup = this.benchmarkGroupsByCommitHash[commitHash];
              if (!benchmarkGroup) {
                return errAsync(new Error(`No benchmark group found for ${commitHash}`));
              }
              return benchmarkGroup.k6Benchmark
                .performIndexingBenchmark()
                .andThen(() => benchmarkGroup.k6Benchmark.performSearchBenchmark())
                .map(() => {
                  this.spinner.succeed(`Benchmarks complete for ${commitHash}`);

                  for (const controller of benchmarkGroup.processManager.processes.values()) {
                    controller.cleanup();
                  }
                })
                .andThen(() => {
                  // Only delay if not the last iteration
                  if (index < commitHashes.length - 1) {
                    this.spinner.start("Waiting 10 seconds before next benchmark...");
                    return this.delay(10000).map(() => {
                      this.spinner.succeed("Delay complete");
                    });
                  }
                  return okAsync(undefined);
                });
            });
          }),
        okAsync<void, ErrorWithMessage>(undefined),
      ),
    );
  }
}

const benchmark = new Command()
  .name("benchmark")
  .description("Benchmark Typesense")
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
  .option("-c, --commitHashes <commit-hashes...>", "Hashes of the commits to compare")
  .option("-v, --verbose", "Verbose output", false)
  .option("-y, --yes", "Verbose output", false)
  .option("-b, --binaries <paths...>", "Paths of the pre-built Typesense binaries to compare")
  .option("-f, --fail <percentage>", "Percentage of regression to fail the test", "50")
  .option("--api-key <key>", "API key to use for the Typesense Process.", "xyz")
  .option("--batch-size <num>", "Batch size for indexing operations", "100")
  .option("--duration <num>", "Duration for each search benchmark", "1s")
  .option("--openAI-key <key>", "OpenAI API key. Defaults to OPENAI_API_KEY in PATH", process.env.OPENAI_API_KEY)
  .action((options) => {
    logger.info("Running Typesense Integration tests");
    const services = new ServiceContainer(findRoot(dirName));
    const spinner = services.getSpinner();
    parseOptions(options as IntegrationTestOptions, benchmarkOptionSchema, spinner)
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
          .get("fs")
          .validateWorkingDirectory(options.workingDirectory)
          .map(() => options);
      })
      .andThen((options) => {
        logger.debug("Starting Typesense process");

        const typesenseProcessManagers = [
          new TypesenseProcessManager(
            spinner,
            options.binaries[0],
            options.apiKey,
            options.workingDirectory,
            services.get("fs"),
          ),
          new TypesenseProcessManager(
            spinner,
            options.binaries[1],
            options.apiKey,
            options.workingDirectory,
            services.get("fs"),
          ),
        ] as [TypesenseProcessManager, TypesenseProcessManager];

        const benchmark = new Benchmarks({
          services,
          apiKey: options.apiKey,
          batchSize: options.batchSize,
          duration: options.duration,
          port: 8108,
          spinner,
          commitHashes: options.commitHashes,
          typesenseProcessManagers,
          workingDirectory: options.workingDirectory,
          failAtPercentage: options.fail,
        });

        return ok(benchmark);
      })
      .andThen((benchmark) => benchmark.benchmark())
      .then((result) => {
        if (result.isErr()) {
          spinner.fail();
          logger.error(result.error.message);
          process.exit(1);
        }
        logger.success("Integration tests passed successfully");
        process.exit(0);
      });
  });

export { benchmark };
