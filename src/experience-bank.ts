import { copyFile, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";

export type Score = {
  score: number;
  higherIsBetter: boolean;
  notes: string;
};

export type CommitInput = {
  inspirationId: string;
  solutionDir: string;
  ok: boolean;
  log: string;
  score: Score | null;
};

export type ExperienceRecord = {
  id: string;
  inspirationId: string;
  ok: boolean;
  score: Score | null;
  solutionDir: string;
  logPath: string;
};

type LedgerRow = ExperienceRecord;

export class ExperienceBank {
  readonly runDir: string;
  private records: LedgerRow[] = [];

  private constructor(runDir: string) {
    this.runDir = runDir;
  }

  static async open(runDir: string): Promise<ExperienceBank> {
    const bank = new ExperienceBank(runDir);
    await mkdir(path.join(runDir, "eb", "solutions"), { recursive: true });
    await mkdir(path.join(runDir, "best"), { recursive: true });
    await bank.reload();
    return bank;
  }

  generation(): number {
    return this.records.length;
  }

  async list(): Promise<ExperienceRecord[]> {
    return this.records.map((row) => ({ ...row }));
  }

  async get(id: string): Promise<ExperienceRecord | undefined> {
    const row = this.records.find((item) => item.id === id);
    return row ? { ...row } : undefined;
  }

  async best(): Promise<ExperienceRecord | undefined> {
    let winner: ExperienceRecord | undefined;
    for (const row of this.records) {
      if (!isCandidate(row)) continue;
      if (!winner || isBetter(row.score!, winner.score!)) winner = row;
    }
    return winner ? { ...winner } : undefined;
  }

  async commit(input: CommitInput): Promise<ExperienceRecord> {
    const id = String(this.records.length + 1).padStart(4, "0");
    const destRoot = path.join(this.runDir, "eb", "solutions", id);
    const destSolution = path.join(destRoot, "solution");
    await mkdir(destSolution, { recursive: true });
    await copyDir(input.solutionDir, destSolution);

    const logPath = path.join(destRoot, "run.log");
    await writeFile(logPath, input.log, "utf8");

    const record: ExperienceRecord = {
      id,
      inspirationId: input.inspirationId,
      ok: input.ok,
      score: input.score,
      solutionDir: destSolution,
      logPath,
    };
    this.records.push(record);
    await writeFile(this.ledgerPath(), this.renderLedger(), "utf8");
    await this.refreshBest();
    return { ...record };
  }

  private ledgerPath(): string {
    return path.join(this.runDir, "eb", "index.jsonl");
  }

  private async reload(): Promise<void> {
    try {
      const text = await readFile(this.ledgerPath(), "utf8");
      this.records = text
        .split("\n")
        .filter((line) => line.length > 0)
        .map((line) => JSON.parse(line) as LedgerRow);
    } catch {
      this.records = [];
    }
  }

  private renderLedger(): string {
    return this.records.map((row) => JSON.stringify(row)).join("\n") + (this.records.length ? "\n" : "");
  }

  private async refreshBest(): Promise<void> {
    const winner = await this.best();
    if (!winner) return;
    await mkdir(path.join(this.runDir, "best"), { recursive: true });
    await copyDir(winner.solutionDir, path.join(this.runDir, "best"));
  }
}

function isCandidate(row: ExperienceRecord): boolean {
  return row.ok && row.score !== null;
}

function isBetter(next: Score, current: Score): boolean {
  if (next.higherIsBetter) return next.score > current.score;
  return next.score < current.score;
}

async function copyDir(from: string, to: string): Promise<void> {
  await mkdir(to, { recursive: true });
  const entries = await readdir(from, { withFileTypes: true });
  for (const entry of entries) {
    const src = path.join(from, entry.name);
    const dest = path.join(to, entry.name);
    if (entry.isDirectory()) {
      await copyDir(src, dest);
    } else if (entry.isFile()) {
      await copyFile(src, dest);
    }
  }
}
