import path from "path";
import type { ErrorWithMessage } from "@/utils/error";
import type { TupleOfLength } from "@/utils/parse";
import type { Point } from "@/utils/plot";
import type { NumericIndices } from "@/utils/types";
import type { INanoDate, IResults } from "influx";
import type { Result } from "neverthrow";
import type { Ora } from "ora";
import type { TableUserConfig } from "table";

import * as asciichart from "asciichart";
import chalk from "chalk";
import { Command } from "commander";
import { ps, upMany } from "docker-compose";
import { InfluxDB } from "influx";
import { err, errAsync, ok, okAsync, ResultAsync } from "neverthrow";
import { table } from "table";
import { z } from "zod";

import { searchScenarios } from "@/benchmarks/k6-utils";
import { ServiceContainer } from "@/services/container";
import { DEFAULT_TYPESENSE_GIT_URL } from "@/services/git";
import { K6Benchmarks } from "@/services/k6";
import { TypesenseProcessManager } from "@/services/typesense-process";
import { toErrorWithMessage } from "@/utils/error";
import { logger, LogLevel } from "@/utils/logger";
import { dirName, findRoot } from "@/utils/package-info";
import { loadConfig, parseOptions } from "@/utils/parse";
import { plot } from "@/utils/plot";

const cwd = process.cwd();

interface FormattedIndexResult extends BaseFormattedResult {
  vus: 1;
  scenario: "import";
}

interface BaseFormattedResult {
  metric: string;
  oldValue: string;
  newValue: string;
  percentageChange: number;
  formattedPercentageChange: string;
  displayVariable: string;
}

interface FormattedSearchResult extends BaseFormattedResult {
  scenario: ScenarioNames;
  vus: 50 | 100;
}

type CommitHash = string;

type ScenarioData = Record<string, Point[]>;

type HistoricalData = Record<
  CommitHash,
  {
    scenario: string;
    vus: number;
    p95: number;
  }[]
>;

type ScenarioNames = (typeof searchScenarios)[number]["name"];

type ScenarioLength = (typeof searchScenarios)["length"];

const KeySchema = z.enum([...searchScenarios.map((s) => s.name)] as TupleOfLength<ScenarioNames, ScenarioLength>);

export const BenchmarkConfigSchema = z
  .object({
    failureThresholds: z
      .record(
        KeySchema,
        z.object({
          "50vu": z.number().min(0).max(100),
          "100vu": z.number().min(0).max(100),
        }),
      )
      .refine((obj): obj is Required<typeof obj> => KeySchema.options.every((key) => obj[key] != null)),
  })
  .strict();

type BenchmarkConfig = z.infer<typeof BenchmarkConfigSchema>;

export const defaultConfig: BenchmarkConfig = {
  failureThresholds: {
    filter_complex: {
      "100vu": 50,
      "50vu": 50,
    },
    filter_simple: {
      "100vu": 50,
      "50vu": 50,
    },
    group: {
      "100vu": 50,
      "50vu": 50,
    },
    just_q: {
      "100vu": 50,
      "50vu": 50,
    },
    q_star: {
      "100vu": 50,
      "50vu": 50,
    },
    sort_eval_condition: {
      "100vu": 50,
      "50vu": 50,
    },
    sort_eval_score: {
      "100vu": 50,
      "50vu": 50,
    },
    sort_simple: {
      "100vu": 50,
      "50vu": 50,
    },
    facet: {
      "100vu": 50,
      "50vu": 50,
    },
  },
};

const benchmarkOptionSchema = z.object({
  containerName: z.string(),
  imageName: z.string(),
  typesenseGitUrl: z.string(),
  workingDirectory: z.string(),
  config: z.string().optional(),
  commitHashes: z.tuple([z.string(), z.string()]),
  verbose: z.boolean(),
  yes: z.boolean(),
  binaries: z.tuple([z.string(), z.string()]),
  apiKey: z.string(),
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

const BenchmarkOptionsSchemaWithFailurePoints = benchmarkOptionSchema
  .merge(BenchmarkConfigSchema)
  .strict()
  .omit({
    config: true,
    batchSize: true,
  })
  .merge(
    z.object({
      batchSize: z.number().min(0),
    }),
  );

type BenchmarkOptionsWithFailurePoints = z.infer<typeof BenchmarkOptionsSchemaWithFailurePoints>;
type IntegrationTestOptions = z.infer<typeof benchmarkOptionSchema>;

type BenchmarkResults = Record<
  string,
  {
    searchBenchmark: {
      p95: number;
      vus: 50 | 100;
      scenario: ScenarioNames;
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
  private readonly percentagesForFailure: BenchmarkConfig["failureThresholds"];
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
    failAtPercentage: BenchmarkConfig["failureThresholds"];
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
    this.percentagesForFailure = options.failAtPercentage;

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

  private handleResults(results: { searchResults: FormattedSearchResult[] }): ResultAsync<void, { message: string }> {
    const { searchResults } = results;

    const failingRows = searchResults.filter((row) => {
      const threshold = this.percentagesForFailure[row.scenario][`${row.vus}vu`];
      return row.percentageChange > threshold;
    });

    if (failingRows.length > 0) {
      const failures = failingRows
        .map((row) => {
          const threshold = this.percentagesForFailure[row.scenario][`${row.vus}vu`];
          return `${row.metric} for ${row.displayVariable || `${row.scenario} (${row.vus}vu)`} changed by ${row.formattedPercentageChange} (threshold: ${threshold}%)`;
        })
        .join("\n");

      return errAsync({
        message: `Performance degradation exceeded configured thresholds:\n${failures}`,
      });
    }

    return okAsync(undefined);
  }

  private getComparisonResults(): ResultAsync<BenchmarkResults, { message: string; commitHash?: string }> {
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
        vus: Number(vus) as 50 | 100,
        scenario: scenario as ScenarioNames,
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

  private groupPointsByScenarioAndVU(data: HistoricalData): Map<string, Map<number, Point[]>> {
    const scenarioMap = new Map<string, Map<number, Point[]>>();

    Object.entries(data).forEach(([commitHash, results]) => {
      results.forEach((result) => {
        const { scenario, vus, p95 } = result;
        const point: Point = { x: commitHash, y: p95 };

        // Get or create the VU map for this scenario
        const vuMap = scenarioMap.get(scenario) ?? new Map<number, Point[]>();
        if (!scenarioMap.has(scenario)) {
          scenarioMap.set(scenario, vuMap);
        }

        // Get or create the points array for this VU count
        const points = vuMap.get(vus) ?? [];
        if (!vuMap.has(vus)) {
          vuMap.set(vus, points);
        }

        points.push(point);
      });
    });

    return scenarioMap;
  }

  private formatScenarioData(
    scenarioMap: Map<string, Map<number, Point[]>>,
  ): Result<ScenarioData[][], ErrorWithMessage> {
    const results: ScenarioData[][] = [];

    for (const [scenario, vuMap] of scenarioMap.entries()) {
      const vuEntries = Array.from(vuMap.entries());

      results.push(vuEntries.map((entry) => this.createScenarioDataPoint(scenario, entry)));
    }

    return ok(results);
  }

  private createScenarioDataPoint(scenario: string, vuEntry: [number, Point[]]): ScenarioData {
    const [vuCount, points] = vuEntry;
    const key = `${scenario} (${vuCount}vu)`;
    return { [key]: points };
  }

  private transformHistoricalData(data: HistoricalData): Result<ScenarioData[][], ErrorWithMessage> {
    const scenarioMap = this.groupPointsByScenarioAndVU(data);

    return this.formatScenarioData(scenarioMap);
  }

  private getHistoricalResults() {
    const influx = new InfluxDB({
      host: "localhost",
      port: 8086,
      database: "k6",
    });

    const searchDurationQuery = `
    SELECT PERCENTILE("value", 95) AS "p95_value"
    FROM "search_processing_time_ms"
    WHERE time >= now() - 30d
    GROUP BY "commitHash", "scenario", "vus"
    ORDER BY time ASC
  `;

    return ResultAsync.fromPromise(
      influx.query<{
        commitHash: string;
        scenario: string;
        vus: string;
        p95_value: number;
      }>(searchDurationQuery),
      toErrorWithMessage,
    )
      .map((res) => this.mapHistoricalResults(res))
      .andThen((res) => this.transformHistoricalData(res))
      .map((data) => {
        this.printPlots(data);
        return data;
      });
  }

  printPlots(results: ScenarioData[][]) {
    results.forEach((scenarios) => {
      // Extract labels and points from all scenarios
      const labels = scenarios.map((scenario) => Object.keys(scenario)[0]).filter(Boolean);
      const points = scenarios.map((scenario) => {
        const label = Object.keys(scenario)[0];
        return label ? scenario[label] : null;
      });

      // Check if we have all necessary data
      if (labels.length === 0 || points.some((p) => !p || p.length === 0)) {
        return err({ message: "Missing data for plot" });
      }

      const colorOptions = [
        asciichart.blue,
        asciichart.green,
        asciichart.yellow,
        asciichart.red,
        asciichart.magenta,
        asciichart.cyan,
      ];

      const colors = scenarios.map((_, i) => colorOptions[i % colorOptions.length]);

      logger.info(
        plot(points as Point[][], {
          title: `${labels[0]!.split(" ")[0]} over ${points[0]?.length} commits`,
          xLabel: "Commit Hash",
          yLabel: "p95 search_time_ms",
          lineLabels: labels as string[],
          colors,
          width: 150,
          height: 30,
        }),
      );
    });
  }
  private mapHistoricalResults(
    results: IResults<{
      commitHash: string;
      scenario: string;
      vus: string;
      p95_value: number;
      time: INanoDate;
    }>,
  ): HistoricalData {
    // Track first appearance of each commit
    const firstAppearances = new Map<string, number>();
    for (const result of results) {
      const timestamp = result.time.getTime();
      if (!firstAppearances.has(result.commitHash) || timestamp < firstAppearances.get(result.commitHash)!) {
        firstAppearances.set(result.commitHash, timestamp);
      }
    }

    // Sort commits by first appearance
    const sortedCommits = [...new Set(results.map((r) => r.commitHash))].sort((a, b) => {
      const timeA = firstAppearances.get(a) ?? 0;
      const timeB = firstAppearances.get(b) ?? 0;
      return (
        timeA < timeB ? -1
        : timeA > timeB ? 1
        : 0
      );
    });

    // Build result map in chronological order
    const resultMap: HistoricalData = {};
    for (const commitHash of sortedCommits) {
      resultMap[commitHash] = results
        .filter((r) => r.commitHash === commitHash)
        .map(({ scenario, vus, p95_value }) => ({
          scenario,
          vus: Number(vus),
          p95: p95_value,
        }));
    }

    return resultMap;
  }

  private printTable(
    results: BenchmarkResults,
  ): ResultAsync<{ searchResults: FormattedSearchResult[]; indexingResults: FormattedIndexResult }, ErrorWithMessage> {
    const formatedSearchResults: FormattedSearchResult[] = [];
    const importDuration = results[this.commitHashes[0]]?.indexDuration;
    const importDurationNew = results[this.commitHashes[1]]?.indexDuration;

    if (!importDuration || !importDurationNew) {
      return errAsync(new Error("No results found"));
    }

    const importPercentageChange = this.calculatePercentageChange(importDuration, importDurationNew);

    const FormattedIndexResult: FormattedIndexResult = {
      metric: "Time to bulk import",
      oldValue: `${(importDuration / 1000).toFixed(5)}s`,
      newValue: `${(importDurationNew / 1000).toFixed(5)}s`,
      percentageChange: importPercentageChange,
      formattedPercentageChange: this.formatPercentageChange(importPercentageChange),
      displayVariable: "1M records",
      scenario: "import",
      vus: 1,
    };

    const previousSearchResults = results[this.commitHashes[0]]?.searchBenchmark;
    const newSearchResults = results[this.commitHashes[1]]?.searchBenchmark;

    if (!previousSearchResults || !newSearchResults) {
      return errAsync(new Error("Missing search benchmark results"));
    }

    // Create maps using separate scenario and vus
    const previousSearchResultsMap = new Map(
      previousSearchResults.map((result) => [
        `${result.scenario}-${result.vus}`,
        { p95: result.p95, scenario: result.scenario, vus: result.vus },
      ]),
    );

    const newSearchResultsMap = new Map(
      newSearchResults.map((result) => [
        `${result.scenario}-${result.vus}`,
        { p95: result.p95, scenario: result.scenario, vus: result.vus },
      ]),
    );

    for (const [key, oldResult] of previousSearchResultsMap) {
      const newResult = newSearchResultsMap.get(key);
      if (!newResult) {
        return errAsync({
          message: `${key} doesn't have a value for the search benchmark`,
        });
      }

      const percentageChange = this.calculatePercentageChange(oldResult.p95, newResult.p95);
      formatedSearchResults.push({
        metric: "p95 search_time_ms when searching with",
        scenario: oldResult.scenario,
        vus: oldResult.vus,
        oldValue: `${oldResult.p95}ms`,
        newValue: `${newResult.p95}ms`,
        percentageChange,
        formattedPercentageChange: this.formatPercentageChange(percentageChange),
        displayVariable: `${oldResult.scenario} (${oldResult.vus}vu)`,
      });
    }

    // For display, use the displayVariable
    const columns: [string, string, string, string, string][] = [
      [
        "Metric",
        "Variable",
        `Value for commit ${this.commitHashes[0].slice(0, 7)}`,
        `Value for commit ${this.commitHashes[1].slice(0, 7)}`,
        "Percentage Change",
      ],
      ...formatedSearchResults.map(
        (row) =>
          [row.metric, row.displayVariable, row.oldValue, row.newValue, row.formattedPercentageChange] as [
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
    return okAsync({ searchResults: formatedSearchResults, indexingResults: FormattedIndexResult });
  }

  benchmark() {
    logger.info("Running benchmarks");

    return this.startContainers()
      .andThen(() =>
        this.getComparisonResults()
          .orElse((error) => {
            if (error.commitHash) {
              return this.performBenchmarks([error.commitHash]);
            }

            return this.performBenchmarks(this.commitHashes);
          })
          .andThen(() => this.getComparisonResults())
          .andThen((res) => this.printTable(res)),
      )
      .andThen((results) => this.handleResults(results))
      .andThen(() => this.getHistoricalResults());
  }

  private performBenchmarks(commitHashes: string[]) {
    return this.createDataDirectories()
      .andThen(() =>
        this.services
          .get("fs")
          .downloadTypesenseDataset("https://dl.typesense.org/datasets/musicbrainz-1M-songs.jsonl.tar.gz"),
      )
      .andThen(() =>
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
  .option("--config <path>", "Path for config file")
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

        return loadConfig({
          configPath: options.config,
          schema: BenchmarkConfigSchema,
          defaultSchema: defaultConfig,
          spinner,
        })
          .map((config) => ({
            ...options,
            ...config,
          }))
          .andThen((options) =>
            parseOptions(
              options as BenchmarkOptionsWithFailurePoints,
              BenchmarkOptionsSchemaWithFailurePoints,
              spinner,
            ),
          );
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
          failAtPercentage: options.failureThresholds,
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
        logger.success("Benchmarks completed successfully");
        process.exit(0);
      });
  });

export { benchmark };
