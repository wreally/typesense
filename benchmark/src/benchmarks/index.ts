import type { K6Env } from "@/services/k6";
import type { EnvVariableKey } from "@/utils/types";
import type { Params } from "k6/http";
import type { Options } from "k6/options";

import { check } from "k6";
import http from "k6/http";
import { Trend } from "k6/metrics";

import { validateK6Environment } from "./k6-utils.ts";

const importDuration = new Trend("import_duration");

const fileContent = open("../../data/data.json", "b");

export const options: Options = {
  vus: 1,
  iterations: 1,
  tags: {
    commitHash: __ENV.COMMIT_HASH ?? "unknown",
  },
};

export default function () {
  const env: EnvVariableKey<K6Env> = {
    API_KEY: __ENV.API_KEY,
    HOST: __ENV.HOST,
    PORT: __ENV.PORT,
    BATCH_SIZE: __ENV.BATCH_SIZE,
    COLLECTION_NAME: __ENV.COLLECTION_NAME,
    COMMIT_HASH: __ENV.COMMIT_HASH,
    DURATION: __ENV.DURATION,
  };

  const validation = validateK6Environment(env);
  if (!validation.isValid) {
    throw new Error(
      `Invalid environment configuration:\n${validation.errors
        .map((err) => `${String(err.key)}: ${err.reason} (got: ${err.value})`)
        .join("\n")}`,
    );
  }

  const baseUrl = `http://${__ENV.HOST}:${__ENV.PORT}/collections/${__ENV.COLLECTION_NAME}/documents/import`;

  const queryParamsMap = {
    batch_size: __ENV.BATCH_SIZE,
  };

  const queryParams = Object.entries(queryParamsMap)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => `${key}=${value}`)
    .join("&");

  const url = queryParams ? `${baseUrl}?${queryParams}` : baseUrl;

  const params: Params = {
    headers: {
      "Content-Type": "application/json",
      "X-TYPESENSE-API-KEY": __ENV.API_KEY ?? "xyz",
    },
  };

  const startTime = new Date().getTime();
  const res = http.post(url, fileContent, params);
  const duration = new Date().getTime() - startTime;

  importDuration.add(duration);

  check(res, {
    "status is 200": (r) => r.status === 200,
    "operation successful": (r) => {
      if (typeof r.body !== "string") {
        return false;
      }

      console.log(url);
      return (
        r.body
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line === '{"success":true}').length ===
        10 ** 6
      );
    },
  });
}
