type UnionUpToNumber<T extends number, A extends number[] = []> =
  A["length"] extends T ? A[number] : UnionUpToNumber<T, [...A, A["length"]]>;

export type NumericIndices<T extends unknown[]> =
  T extends { length: infer R extends number } ? UnionUpToNumber<R> : never;

export type EnsureExhaustive<Rec, Arr extends readonly (string | number)[]> =
  Exclude<keyof Rec, Arr[number]> extends never ? true : "Error: Missing required keys";

export interface Album {
  album_name: string;
  country: string;
  genres: string[];
  id: string;
  primary_artist_name: string;
  release_date: number;
  release_decade: string;
  release_group_types: string[];
  title: string;
  track_id: string;
  urls: {
    type: string;
    url: string;
  }[];
}

export type EnvVariableKey<T> = {
  [P in keyof T]: string | number | undefined;
};
