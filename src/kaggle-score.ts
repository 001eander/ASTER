import { access, readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import type { Score } from "./experience-bank.js";

export type KaggleConfig = {
  competition: string;
  submission: string;
  higherIsBetter: boolean;
  folds: number;
  minSaveScore: number;
  targetScore: number;
};

export type KaggleCli = (
  args: string[],
) => Promise<{ code: number; stdout: string; stderr: string }>;

export type KaggleSubmission = {
  description: string;
  status: string;
  publicScore: number | null;
};

export async function loadKaggleConfig(taskDir: string): Promise<KaggleConfig | null> {
  const file = path.join(taskDir, "kaggle.json");
  try {
    await access(file);
  } catch {
    return null;
  }
  const raw = JSON.parse(await readFile(file, "utf8")) as {
    competition?: string;
    submission?: string;
    higherIsBetter?: boolean;
    higher_is_better?: boolean;
    folds?: number;
    minSaveScore?: number;
    targetScore?: number;
  };
  if (!raw.competition) throw new Error("kaggle.json missing competition");
  return {
    competition: raw.competition,
    submission: raw.submission ?? "submission.csv",
    higherIsBetter: raw.higherIsBetter ?? raw.higher_is_better ?? true,
    folds: raw.folds ?? 5,
    minSaveScore: raw.minSaveScore ?? 0.9,
    targetScore: raw.targetScore ?? 0.95,
  };
}

export function parseSubmissions(text: string): KaggleSubmission[] {
  const trimmed = text.trim();
  if (!trimmed || trimmed.startsWith("No submissions")) return [];
  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    const raw = JSON.parse(trimmed) as unknown;
    const rows = Array.isArray(raw) ? raw : [raw];
    return rows.map(rowFromUnknown);
  }
  const lines = trimmed.split(/\r?\n/).filter((line) => line.length > 0);
  if (lines.length < 2) return [];
  const header = splitCsv(lines[0]!);
  const descIdx = header.findIndex((name) => /description|message/i.test(name));
  const statusIdx = header.findIndex((name) => /status/i.test(name));
  const scoreIdx = header.findIndex((name) => /publicscore|public.?score/i.test(name));
  return lines.slice(1).map((line) => {
    const cols = splitCsv(line);
    return {
      description: descIdx >= 0 ? cols[descIdx] ?? "" : "",
      status: statusIdx >= 0 ? cols[statusIdx] ?? "" : "",
      publicScore: scoreIdx >= 0 ? parseScore(cols[scoreIdx]) : null,
    };
  });
}

export function pickSubmission(rows: KaggleSubmission[], message: string): KaggleSubmission | undefined {
  return rows.find((row) => row.description === message) ?? rows[0];
}

export async function scoreOnKaggle(opts: {
  workDir: string;
  config: KaggleConfig;
  message: string;
  cli?: KaggleCli;
  sleep?: (ms: number) => Promise<void>;
  now?: () => number;
  timeoutMs?: number;
}): Promise<{ ok: boolean; score: Score | null; log: string }> {
  const cli = opts.cli ?? runKaggle;
  const sleep = opts.sleep ?? ((ms: number) => new Promise((resolve) => setTimeout(resolve, ms)));
  const now = opts.now ?? Date.now;
  const timeoutMs = opts.timeoutMs ?? 180_000;
  const file = path.join(opts.workDir, opts.config.submission);
  try {
    await access(file);
  } catch {
    return { ok: false, score: null, log: `missing ${opts.config.submission} for Kaggle submit` };
  }

  const submitted = await cli([
    "competitions",
    "submit",
    "-c",
    opts.config.competition,
    "-f",
    file,
    "-m",
    opts.message,
  ]);
  const submitLog = `${submitted.stdout}${submitted.stderr}`;
  if (submitted.code !== 0) {
    return { ok: false, score: null, log: submitLog || "kaggle submit failed" };
  }

  const started = now();
  let lastLog = submitLog;
  while (now() - started <= timeoutMs) {
    const listed = await cli([
      "competitions",
      "submissions",
      opts.config.competition,
      "--format",
      "json",
    ]);
    lastLog = `${submitLog}\n${listed.stdout}${listed.stderr}`;
    if (listed.code !== 0) {
      return { ok: false, score: null, log: lastLog };
    }
    const row = pickSubmission(parseSubmissions(listed.stdout), opts.message);
    if (row && /error|fail/i.test(row.status)) {
      return { ok: false, score: null, log: `${lastLog}\nsubmission ${row.status}` };
    }
    if (row && row.publicScore !== null && /complete|scored/i.test(row.status)) {
      return {
        ok: true,
        log: lastLog,
        score: {
          score: row.publicScore,
          higherIsBetter: opts.config.higherIsBetter,
          notes: `kaggle public ${row.publicScore}`,
        },
      };
    }
    await sleep(3000);
  }
  return { ok: false, score: null, log: `${lastLog}\nkaggle score timed out` };
}

function rowFromUnknown(raw: unknown): KaggleSubmission {
  const row = raw as Record<string, unknown>;
  return {
    description: String(row.description ?? row.message ?? ""),
    status: String(row.status ?? ""),
    publicScore: parseScore(row.publicScore ?? row.public_score),
  };
}

function parseScore(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  if (!text || text === "—" || text === "-" || text === "None") return null;
  const n = Number(text);
  return Number.isFinite(n) ? n : null;
}

function splitCsv(line: string): string[] {
  return line.split(",").map((part) => part.trim());
}

function runKaggle(args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn("kaggle", args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}
