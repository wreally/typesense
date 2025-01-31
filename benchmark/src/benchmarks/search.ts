import type { Album, EnsureExhaustive, EnvVariableKey } from "@/utils/types";
import type { Options as K6Options } from "k6/options";
import type { SearchParams, SearchResponse } from "typesense/lib/Typesense/Documents";

//@ts-expect-error - the text-summary module is not typed
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";
import { check, group } from "k6";
import http from "k6/http";
import { Counter, Trend } from "k6/metrics";

import type { K6Env } from "../services/k6";
import { searchScenarios, validateK6Environment } from "./k6-utils.ts";

interface CharsMap {
  a: "a";
  b: "b";
  c: "c";
  d: "d";
  e: "e";
  f: "f";
  g: "g";
  h: "h";
  i: "i";
  j: "j";
  k: "k";
  l: "l";
  m: "m";
  n: "n";
  o: "o";
  p: "p";
  q: "q";
  r: "r";
  s: "s";
  t: "t";
  u: "u";
  v: "v";
  w: "w";
  x: "x";
  y: "y";
  z: "z";
}

const chars = [
  "a",
  "b",
  "c",
  "d",
  "e",
  "f",
  "g",
  "h",
  "i",
  "j",
  "k",
  "l",
  "m",
  "n",
  "o",
  "p",
  "q",
  "r",
  "s",
  "t",
  "u",
  "v",
  "w",
  "x",
  "y",
  "z",
] as const;

type Validated = EnsureExhaustive<CharsMap, typeof chars>;
const _validated: Validated = true;

const charset = new Set<CharacterSpace>(chars);
type CharacterSpace = CharsMap[keyof CharsMap];

const stopWords = [
  "a",
  "am",
  "an",
  "and",
  "as",
  "at",
  "by",
  "c's",
  "co",
  "do",
  "eg",
  "et",
  "for",
  "he",
  "hi",
  "i",
  "i'd",
  "i'm",
  "ie",
  "if",
  "in",
  "inc",
  "is",
  "it",
  "its",
  "me",
  "my",
  "nd",
  "no",
  "non",
  "nor",
  "not",
  "of",
  "off",
  "oh",
  "ok",
  "on",
  "or",
  "per",
  "que",
  "qv",
  "rd",
  "re",
  "so",
  "sub",
  "t's",
  "th",
  "the",
  "to",
  "too",
  "two",
  "un",
  "up",
  "us",
  "vs",
  "we",
] as const;

type StopWord = (typeof stopWords)[number];

const stopWordSet = new Set<StopWord>(stopWords);

interface Tags {
  [name: string]: string;
  commitHash: string;
  scenario: string;
  vus: string;
}


type ScenarioName = (typeof searchScenarios)[number]["name"];

interface ScenarioConfig {
  executor: "constant-vus";
  duration: string;
  vus: number;

  env: { VUS: string; SCENARIO: ScenarioName };
  exec: "default";
  startTime?: string;
}

type DurationUnit = "s" | "m" | "h" | "d";

type DurationString = `${number}${DurationUnit}`;

function isDurationString(value: string): value is DurationString {
  return /^\d+[smhd]$/.test(value);
}

function parseDuration(duration: string): number {
  if (!isDurationString(duration)) {
    throw new Error(`Invalid duration format: ${duration}`);
  }

  const value = parseInt(duration.slice(0, -1), 10);
  const unit = duration.slice(-1) as DurationUnit;

  switch (unit) {
    case "s":
      return value;
    case "m":
      return value * 60;
    case "h":
      return value * 60 * 60;
    case "d":
      return value * 60 * 60 * 24;
    default:
      throw new Error(`Unknown duration unit: ${unit as string}`);
  }
}

const DURATION = __ENV.DURATION!;

const createScenarioConfig = (
  scenario: (typeof searchScenarios)[number],
  vus: number,
  startTime: number,
): ScenarioConfig => ({
  executor: "constant-vus",
  vus,
  duration: DURATION,
  env: { VUS: vus.toString(), SCENARIO: scenario.name },
  exec: "default",
  startTime: `${startTime}s`,
});

export const options: K6Options = {
  systemTags: [],
  scenarios: Object.fromEntries(
    searchScenarios.flatMap((scenario, index) => {
      const durationInSeconds = parseDuration(DURATION);
      const gap = 5; // 5-second gap between scenarios
      const startTime50vu = index * (durationInSeconds + gap) * 2;
      const startTime100vu = startTime50vu + durationInSeconds + gap;
      return [
        [`${scenario.name}_50vu`, createScenarioConfig(scenario, 50, startTime50vu)],
        [`${scenario.name}_100vu`, createScenarioConfig(scenario, 100, startTime100vu)],
      ];
    }),
  ),
};

function transformParamsToQueryString(params: SearchParams): string {
  return Object.entries(params)
    .map(([key, value]) => `${key}=${encodeURIComponent(String(value))}`)
    .join("&");
}

function performSearch(queryParams: SearchParams, tags: Tags, baseUrl: string, headers: Record<string, string>) {
  const queryParamsString = transformParamsToQueryString(queryParams);
  const url = `${baseUrl}?${queryParamsString}`;
  const response = http.get(url, { headers });

  check(response, { "status was 200": (r) => r.status === 200 }, { quiet: true });

  if (response.status === 200) {
    const body = JSON.parse(response.body as string) as SearchResponse<Album>;

    searchProcessingTimes.add(body.search_time_ms, tags);
  }

  if (response.error?.toLowerCase().includes("timeout")) {
    timeouts.add(1, tags);
  }
}

const timeouts: Counter = new Counter("timeouts");
const searchProcessingTimes: Trend = new Trend("search_processing_time_ms");

export default function (): void {
  const env: EnvVariableKey<K6Env> = {
    API_KEY: __ENV.API_KEY,
    HOST: __ENV.HOST,
    PORT: __ENV.PORT,
    BATCH_SIZE: __ENV.BATCH_SIZE,
    COLLECTION_NAME: __ENV.COLLECTION_NAME,
    COMMIT_HASH: __ENV.COMMIT_HASH,
    DURATION: __ENV.DURATION,
  };

  const scenarioName = __ENV.SCENARIO;

  const validation = validateK6Environment(env);
  if (!validation.isValid) {
    throw new Error(
      `Invalid environment configuration:\n${validation.errors
        .map((err) => `${String(err.key)}: ${err.reason} (got: ${err.value})`)
        .join("\n")}`,
    );
  }

  const headers: Record<string, string> = {
    accept: "application/json, text/plain, */*",
    "x-typesense-api-key": validation.env.API_KEY,
    "user-agent":
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36",
    "content-type": "application/json",
    "accept-encoding": "gzip, deflate, br",
  };

  const scenario = searchScenarios.find((s) => s.name === scenarioName);
  const baseUrl = `http://${validation.env.HOST}:${validation.env.PORT}/collections/${validation.env.COLLECTION_NAME}/documents/search`;

  if (!scenario) {
    throw new Error(`Scenario ${scenarioName} not found`);
  }

  if (!__ENV.VUS) {
    throw new Error(`VUS not found`);
  }

  if (!scenario.wildCardQuery) {
    // Pick search phrase
    const searchPhrase = getSearchPhrase(10);
    // Break the search phrase out into characters to simulate users typing
    const queries = instantSearchQueriesForSearchPhrase(searchPhrase.phrase ?? "aaa");

    queries.forEach((query) => {
      group("", () => {
        const queryParams: SearchParams = { ...scenario.params, q: query };

        performSearch(
          queryParams,
          {
            commitHash: validation.env.COMMIT_HASH,
            scenario: scenario.name,
            vus: __ENV.VUS!,
          },
          baseUrl,
          headers,
        );
      });
    });
  } else {
    group("", () => {
      performSearch(
        scenario.params,
        {
          commitHash: validation.env.COMMIT_HASH,
          scenario: scenario.name,
          vus: __ENV.VUS!,
        },
        baseUrl,
        headers,
      );
    });
  }
}

function instantSearchQueriesForSearchPhrase(searchPhrase: string): string[] {
  return searchPhrase
    .split("")
    .map((_, index) => searchPhrase.slice(0, index + 1))
    .filter((query) => query.length <= 3);
}

function isValidSearchPhrase(phrase: string, charset: Set<CharacterSpace>) {
  return (
    phrase.length === 3 &&
    !stopWordSet.has(phrase as StopWord) &&
    phrase.split("").every((char) => charset.has(char as CharacterSpace))
  );
}

function generatePermutations(length = 3, charset: Set<CharacterSpace>) {
  return generateAllPermutations(length, charset).filter((phrase) => isValidSearchPhrase(phrase, charset));
}

function generateAllPermutations(length = 3, charset: Set<CharacterSpace>): string[] {
  if (length === 1) {
    return [...charset];
  }

  return Array.from(charset).flatMap((char) =>
    generateAllPermutations(length - 1, charset).map((phrase) => char + phrase),
  );
}

function getSearchPhrase(numVus: number): {
  phrase: string | undefined;
  index: number;
} {
  const searches = generatePermutations(3, charset);
  const vuOffset: number = Math.ceil(searches.length / numVus) * (__VU - 1);
  const searchPhraseIndex = (vuOffset + __ITER) % searches.length;

  return { phrase: searches[searchPhraseIndex], index: searchPhraseIndex };
}

export function handleSummary(data: unknown) {
  return {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}
