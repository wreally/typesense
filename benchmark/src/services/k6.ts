import path from "path";
import type { TypesenseProcessManager } from "@/services/typesense-process";
import type { ErrorWithMessage } from "@/utils/error";
import type { IDockerComposeResult } from "docker-compose";
import type { Ora } from "ora";

import { run } from "docker-compose";
import { errAsync, okAsync, ResultAsync } from "neverthrow";

import { toErrorWithMessage } from "@/utils/error";
import { logger, LogLevel } from "@/utils/logger";
import { findRoot } from "@/utils/package-info";

interface BaseK6Env {
  API_KEY: string;
  PORT: number;
  COLLECTION_NAME: string;
  HOST: string;
  COMMIT_HASH: string;
}

export interface IndexK6Env extends BaseK6Env {
  BATCH_SIZE: number;
}

export interface SearchK6Env extends BaseK6Env {
  DURATION: string;
}

export type K6Env = IndexK6Env & SearchK6Env;

interface LoadTestConfig {
  batchSize: number;
  duration: string;
  apiKey: string;
  port: number;
  commitHash: string;
  typesenseProcessManager: TypesenseProcessManager;
  spinner: Ora;
}

export class K6Benchmarks {
  private readonly config: LoadTestConfig;
  private readonly isInCi: boolean;
  private static readonly COLLECTION_NAME = "songs";
  public static readonly REQUIRED_SERVICES = ["grafana", "influxdb"];

  constructor(config: LoadTestConfig) {
    this.config = config;
    this.isInCi = Boolean(process.env.CI) || false;
  }

  public performSearchBenchmark(): ResultAsync<void, ErrorWithMessage> {
    return this.getSearchBenchmarkPath()
      .andThen((path) => this.executeK6Benchmark({ name: "search", scriptPath: path }))
      .map(() => {
        this.config.spinner.succeed("Search benchmark complete");
      });
  }

  public performIndexingBenchmark(): ResultAsync<void, ErrorWithMessage> {
    return this.getIndexingBenchmarkPath().andThen((path) => {
      return this.createBenchmarkCollection()
        .andThen(() => this.executeK6Benchmark({ name: "indexing", scriptPath: path }))
        .map(() => {
          this.config.spinner.succeed("Indexing benchmark complete");
        });
    });
  }

  private executeK6Benchmark(options: {
    scriptPath: string;
    name: string;
    additionalVars?: Record<string, unknown>;
  }): ResultAsync<void, ErrorWithMessage> {
    const { scriptPath, name } = options;
    const envVarString = this.buildK6EnvironmentVars(options.additionalVars);
    this.config.spinner.start(`Running ${name} benchmark\n`);

    const command = [
      "run",
      envVarString,
      "--compatibility-mode=experimental_enhanced",
      scriptPath,
      logger.getLevel() <= LogLevel.DEBUG ? "--quiet" : "",
    ].join(" ");

    const errors: string[] = [];
    const warnings: string[] = [];

    return ResultAsync.fromPromise(
      run("k6", command, {
        commandOptions: ["--no-deps", "--remove-orphans"],
        log: logger.getLevel() === LogLevel.DEBUG,
        cwd: findRoot(process.cwd()),
        callback: (chunk) => this.processLogChunk(chunk, errors, warnings),
      }),
      toErrorWithMessage,
    ).andThen((result) => this.handleK6Result(result, errors, warnings));
  }

  private processLogChunk(chunk: Buffer, errors: string[], warnings: string[]): void {
    const log = chunk.toString();
    if (log.includes("level=error")) {
      errors.push(log.trim());
    }
    if (log.includes("level=warn")) {
      warnings.push(log.trim());
    }
  }

  private handleK6Result(
    result: IDockerComposeResult,
    errors: string[],
    warnings: string[],
  ): ResultAsync<void, ErrorWithMessage> {
    // k6 output format is "checks.........................: 100.00% ✓ 21888      ✗ 0"
    // First trim any leading/trailing whitespace
    const cleanOutput = result.out.trim();

    // Find the checks line specifically
    const checksLine = cleanOutput.split("\n").find((line) => line.trim().startsWith("checks"));

    if (!checksLine) {
      return errAsync({
        message: "Could not find checks line in output",
      });
    }

    // Extract the pass rate from the checks line
    const checkMatch = /([0-9.]+)%/.exec(checksLine);
    const checksPassRate = parseFloat(checkMatch?.[1] ?? "0");

    this.config.spinner.stop();
    logger.info(`Checks pass rate: ${checksPassRate}%`);

    if (errors.length > 0) {
      logger.error(`Errors: \n\n${errors.join("\n")}`);
    }
    if (warnings.length > 0) {
      logger.warn(`Warnings: \n\n${warnings.join("\n")}`);
    }

    if (checksPassRate < 100) {
      return errAsync({
        message: `k6 tests failed - ${checksPassRate}% checks passed`,
      });
    }

    this.config.spinner.succeed("Benchmark complete");
    return okAsync(undefined);
  }

  private buildK6EnvironmentVars(additionalVars?: Record<string, unknown>): string {
    const envVarMap = {
      API_KEY: this.config.apiKey,
      DURATION: this.config.duration,
      BATCH_SIZE: this.config.batchSize,
      COLLECTION_NAME: K6Benchmarks.COLLECTION_NAME,
      PORT: this.config.port,
      HOST: "host.docker.internal",
      COMMIT_HASH: this.config.commitHash,
      ...additionalVars,
    };

    return Object.entries(envVarMap)
      .map(([key, value]) => `-e ${key}=${value}`)
      .join(" ");
  }

  private createBenchmarkCollection(): ResultAsync<void, ErrorWithMessage> {
    this.config.spinner.start("Creating benchmark collection");

    const process = this.config.typesenseProcessManager.processes.get(this.config.port);
    if (!process) {
      return errAsync({
        message: `Process not found for port ${this.config.port}`,
      });
    }

    return this.config.typesenseProcessManager
      .createCollection(process, {
        name: K6Benchmarks.COLLECTION_NAME,
        fields: [
          { name: "album_name", type: "string" },
          { name: "country", type: "string", facet: true },
          { name: "genres", type: "string[]", facet: true },
          { name: "primary_artist_name", type: "string", facet: true },
          { name: "release_date", type: "int64" },
          { name: "release_decade", type: "string", facet: true },
          { name: "release_group_types", type: "string[]", facet: true },
          { name: "title", type: "string" },
          { name: "track_id", type: "string" },
          { name: "urls", type: "object[]", optional: true },
        ],
        enable_nested_fields: true,
      })
      .map(() => {
        this.config.spinner.succeed("Benchmark collection created");
      });
  }

  private getSearchBenchmarkPath(): ResultAsync<string, ErrorWithMessage> {
    return okAsync(path.join("/app", "src", "benchmarks", "search.ts"));
  }

  private getIndexingBenchmarkPath(): ResultAsync<string, ErrorWithMessage> {
    return okAsync(path.join("/app", "src", "benchmarks", "index.ts"));
  }
}
