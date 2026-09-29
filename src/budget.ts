export type ParsedBudget = {
  maxSolutions?: number;
  maxMs?: number;
};

export const DEFAULT_SANDBOX_MS = 1_800_000;
export const DEFAULT_PROPOSAL_MS = 1_800_000;

export function parseDuration(text: string, flag: string): number {
  const match = /^(\d+)(ms|s|m|h)$/.exec(text);
  if (!match) throw new Error(`invalid ${flag} ${text} (use 30s, 30m, or 2h)`);
  const n = Number(match[1]);
  const unit = match[2];
  return unit === "ms" ? n : unit === "s" ? n * 1000 : unit === "m" ? n * 60_000 : n * 3_600_000;
}

export function parseBudget(
  text: string | undefined,
  solutions: string | undefined,
  opts: { unlimited?: boolean } = {},
): ParsedBudget {
  const maxSolutions = solutions ? Number(solutions) : opts.unlimited ? undefined : 8;
  if (maxSolutions !== undefined && (!Number.isFinite(maxSolutions) || maxSolutions < 1)) {
    throw new Error(`invalid --solutions ${solutions}`);
  }
  return budgetOf(maxSolutions, text ? parseDuration(text, "--budget") : undefined);
}

export function continueBudget(
  saved: { maxSolutions?: number; maxMs?: number },
  consumedMs: number,
  extra: { budget?: string; solutions?: string; unlimited?: boolean } = {},
): ParsedBudget {
  if (extra.unlimited) {
    return parseBudget(extra.budget, extra.solutions, { unlimited: true });
  }
  const maxSolutions = extra.solutions ? Number(extra.solutions) : saved.maxSolutions;
  if (maxSolutions !== undefined && (!Number.isFinite(maxSolutions) || maxSolutions < 1)) {
    throw new Error(`invalid --solutions ${extra.solutions}`);
  }
  const extraMs = extra.budget ? parseDuration(extra.budget, "--budget") : 0;
  if (saved.maxMs === undefined && !extra.budget) return budgetOf(maxSolutions);
  const remaining = saved.maxMs === undefined ? 0 : Math.max(0, saved.maxMs - consumedMs);
  return budgetOf(maxSolutions, consumedMs + remaining + extraMs);
}

function budgetOf(maxSolutions?: number, maxMs?: number): ParsedBudget {
  const out: ParsedBudget = {};
  if (maxSolutions !== undefined) out.maxSolutions = maxSolutions;
  if (maxMs !== undefined) out.maxMs = maxMs;
  return out;
}

export function parseSandboxMs(
  flag: string | undefined,
  env: string | undefined,
  opts: { unlimited?: boolean } = {},
): number | undefined {
  if (flag) return parseDuration(flag, "--sandbox");
  if (opts.unlimited) return undefined;
  if (env !== undefined && env !== "") {
    const n = Number(env);
    if (!Number.isFinite(n) || n < 1) throw new Error(`invalid HYRA_PI_SANDBOX_MS ${env}`);
    return n;
  }
  return DEFAULT_SANDBOX_MS;
}

export function parseProposalMs(
  flag: string | undefined,
  env: string | undefined,
  opts: { unlimited?: boolean } = {},
): number | undefined {
  if (flag) return parseDuration(flag, "--write");
  if (opts.unlimited) return undefined;
  if (env !== undefined && env !== "") {
    const n = Number(env);
    if (!Number.isFinite(n) || n < 1) throw new Error(`invalid HYRA_PI_PROPOSAL_MS ${env}`);
    return n;
  }
  return DEFAULT_PROPOSAL_MS;
}

export function parseArgs(argv: string[]): Map<string, string | boolean> {
  const out = new Map<string, string | boolean>();
  let command: string | undefined;
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i]!;
    if (!token.startsWith("-") && !command) {
      command = token;
      out.set("_", token);
      continue;
    }
    if (token.startsWith("--")) {
      const key = token.slice(2);
      const next = argv[i + 1];
      if (!next || next.startsWith("-")) {
        out.set(key, true);
      } else {
        out.set(key, next);
        i += 1;
      }
    }
  }
  return out;
}
