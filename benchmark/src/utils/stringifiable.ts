function isStringifiable(data: unknown): data is { toString(): string } {
  return (
    data !== null &&
    (typeof data === "string" ||
      typeof data === "number" ||
      typeof data === "boolean" ||
      (typeof data === "object" && typeof data.toString === "function"))
  );
}

export { isStringifiable };
