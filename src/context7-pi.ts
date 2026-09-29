import { createRequire } from "node:module";
import path from "node:path";

export const CONTEXT7_TOOLS = ["resolve-library-id", "query-docs"] as const;

export function context7PiRoot(): string {
  return path.dirname(createRequire(import.meta.url).resolve("@upstash/context7-pi/package.json"));
}
