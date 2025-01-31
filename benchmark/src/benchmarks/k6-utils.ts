import type { SearchParams } from "typesense/lib/Typesense/Documents";
import type { K6Env } from "../services/k6";
import type { EnsureExhaustive, EnvVariableKey } from "../utils/types";

interface ValidationError {
  key: keyof K6Env;
  value: string | number | undefined;
  reason: string;
}

type ValidationResult = { isValid: true; env: K6Env } | { isValid: false; errors: ValidationError[] };

export function validateK6Environment(env: EnvVariableKey<K6Env>): ValidationResult {
  const requiredKeys = [
    "API_KEY",
    "HOST",
    "PORT",
    "BATCH_SIZE",
    "COLLECTION_NAME",
    "COMMIT_HASH",
    "DURATION",
  ] as const satisfies (keyof K6Env)[];

  type Validated = EnsureExhaustive<K6Env, typeof requiredKeys>;

  const _validated: Validated = true;
  const errors: ValidationError[] = [];

  requiredKeys.forEach((key) => {
    const value = env[key];
    if (value === undefined || value === null) {
      errors.push({
        key,
        value: undefined,
        reason: "Value is required but was not provided",
      });
      return;
    }

    switch (key) {
      case "PORT":
      case "BATCH_SIZE": {
        if (typeof value !== "string") {
          errors.push({
            key,
            value,
            reason: "Value must be a string",
          });
        } else if (!parseInt(value)) {
          errors.push({
            key,
            value,
            reason: "Value must be a valid integer",
          });
        }
        break;
      }
      case "API_KEY":
      case "HOST":
      case "COMMIT_HASH":
      case "COLLECTION_NAME": {
        if (typeof value !== "string") {
          errors.push({
            key,
            value,
            reason: "Value must be a string",
          });
        } else if (value.length === 0) {
          errors.push({
            key,
            value,
            reason: "Value cannot be empty",
          });
        }
        break;
      }
      case "DURATION": {
        if (typeof value !== "string") {
          errors.push({
            key,
            value,
            reason: "Value must be a string",
          });
        } else if (!/^\d+[smhd]$/.test(value)) {
          errors.push({
            key,
            value,
            reason: "Duration must be in format: number followed by s/m/h/d",
          });
        }
        break;
      }
    }
  });

  if (errors.length === 0) {
    return { isValid: true, env: env as K6Env };
  }

  return { isValid: false, errors };
}

export const searchScenarios = [
  {
    name: "just_q",
    params: {
      query_by: "primary_artist_name,title,album_name",
      highlight_full_fields: "primary_artist_name,title,album_name",
    },
    wildCardQuery: false,
  },
  {
    name: "q_star",
    params: {
      q: "*",
      query_by: "primary_artist_name,title,album_name",
      highlight_full_fields: "primary_artist_name,title,album_name",
    },
    wildCardQuery: true,
  },
  {
    name: "filter_simple",
    params: {
      q: "*",
      query_by: "primary_artist_name,title,album_name",
      highlight_full_fields: "primary_artist_name,title,album_name",
      filter_by: "genres:Rock",
    },
    wildCardQuery: true,
  },
  {
    name: "filter_complex",
    params: {
      q: "*",
      query_by: "primary_artist_name,title,album_name",
      highlight_full_fields: "primary_artist_name,title,album_name",
      filter_by: "genres:Rock && primary_artist_name:Queen || primary_artist_name:Led Zeppelin",
    },
    wildCardQuery: true,
  },
  {
    name: "sort_simple",
    params: {
      q: "*",
      query_by: "primary_artist_name,title,album_name",
      highlight_full_fields: "primary_artist_name,title,album_name",
      sort_by: "release_date:desc",
    },
    wildCardQuery: true,
  },
  {
    name: "sort_eval_condition",
    params: {
      q: "*",
      query_by: "primary_artist_name,title,album_name",
      highlight_full_fields: "primary_artist_name,title,album_name",
      sort_by: "_eval(primary_artist_name:Queen):desc, release_date:desc",
    },
    wildCardQuery: true,
  },
  {
    name: "sort_eval_score",
    params: {
      q: "*",
      query_by: "primary_artist_name,title,album_name",
      highlight_full_fields: "primary_artist_name,title,album_name",
      sort_by: "_eval([(primary_artist_name:Queen):3, (primary_artist_name:Nirvana):5]):desc, release_date:desc",
    },
    wildCardQuery: true,
  },
  {
    name: "facet",
    params: {
      q: "*",
      query_by: "primary_artist_name,title,album_name",
      highlight_full_fields: "primary_artist_name,title,album_name",
      facet_by: "genres,country,release_decade",
    },
    wildCardQuery: true,
  },
  {
    name: "group",
    params: {
      q: "*",
      query_by: "primary_artist_name,title,album_name",
      highlight_full_fields: "primary_artist_name,title,album_name",
      group_by: "genres",
    },
    wildCardQuery: true,
  },
] as const satisfies Scenario[];

type Scenario =
  | {
      name: string;
      params: Omit<SearchParams, "q">;
      wildCardQuery: false;
    }
  | {
      name: string;
      params: Omit<SearchParams, "q"> & { q: "*" };
      wildCardQuery: true;
    };
