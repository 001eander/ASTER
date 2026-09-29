import { mkdir, readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { writeJsonFile } from "./json-file.js";

export type Inspiration = {
  id: string;
  direction: string;
  context: string;
  ebGeneration: number;
};

export type EnqueueInput = {
  direction: string;
  context: string;
  ebGeneration: number;
};

export type EnqueueResult =
  | { ok: true; id: string }
  | { ok: false; reason: "full" | "duplicate" };

type QueueRow = Inspiration & { state?: string };

export class InspirationQueue {
  readonly runDir: string;
  readonly lowWater: number;
  readonly highWater: number;
  private nextId = 1;
  private waiting: Inspiration[] = [];
  private held = new Map<string, Inspiration>();
  private seen = new Set<string>();

  private constructor(runDir: string, lowWater: number, highWater: number) {
    this.runDir = runDir;
    this.lowWater = lowWater;
    this.highWater = highWater;
  }

  static async open(
    runDir: string,
    opts: { lowWater: number; highWater: number; knownIds?: string[] },
  ): Promise<InspirationQueue> {
    if (opts.lowWater < 0 || opts.highWater < 1 || opts.lowWater > opts.highWater) {
      throw new Error("queue marks must satisfy 0 <= lowWater <= highWater");
    }
    const queue = new InspirationQueue(runDir, opts.lowWater, opts.highWater);
    await mkdir(path.join(runDir, "queue"), { recursive: true });
    await queue.reload(opts.knownIds ?? []);
    return queue;
  }

  waitingCount(): number {
    return this.waiting.length;
  }

  needsMore(): boolean {
    return this.waiting.length <= this.lowWater;
  }

  mustStopProducing(): boolean {
    return this.waiting.length >= this.highWater;
  }

  isHeld(id: string): boolean {
    return this.held.has(id);
  }

  peekWaiting(): Inspiration[] {
    return this.waiting.map((item) => ({ ...item }));
  }

  peekHeld(): Inspiration[] {
    return [...this.held.values()].map((item) => ({ ...item }));
  }

  takenDirections(): string[] {
    return [...this.seen];
  }

  async enqueue(input: EnqueueInput): Promise<EnqueueResult> {
    if (this.mustStopProducing()) return { ok: false, reason: "full" };
    const key = normalizeDirection(input.direction);
    if (this.seen.has(key)) return { ok: false, reason: "duplicate" };
    const item: Inspiration = {
      id: `insp-${String(this.nextId).padStart(3, "0")}`,
      direction: input.direction,
      context: input.context,
      ebGeneration: input.ebGeneration,
    };
    this.nextId += 1;
    this.seen.add(key);
    this.waiting.push(item);
    await this.persist(item, "waiting");
    return { ok: true, id: item.id };
  }

  async claim(): Promise<Inspiration | undefined> {
    const item = this.waiting.shift();
    if (!item) return undefined;
    this.held.set(item.id, item);
    await this.persist(item, "held");
    return { ...item };
  }

  releaseHold(id: string): void {
    this.held.delete(id);
  }

  async requeueOrphans(committedIds: Set<string>): Promise<void> {
    const orphans: Inspiration[] = [];
    for (const [id, item] of this.held) {
      if (committedIds.has(id)) continue;
      this.held.delete(id);
      orphans.push(item);
    }
    orphans.sort((a, b) => inspirationSeq(a.id) - inspirationSeq(b.id));
    this.waiting = [...orphans, ...this.waiting];
    for (const item of orphans) await this.persist(item, "waiting");
  }

  private async reload(knownIds: string[]): Promise<void> {
    const dir = path.join(this.runDir, "queue");
    let names: string[] = [];
    try {
      names = await readdir(dir);
    } catch {
      return;
    }
    const rows: QueueRow[] = [];
    for (const name of names) {
      if (!name.endsWith(".json")) continue;
      rows.push(JSON.parse(await readFile(path.join(dir, name), "utf8")) as QueueRow);
    }
    rows.sort((a, b) => inspirationSeq(a.id) - inspirationSeq(b.id));
    for (const row of rows) {
      const item: Inspiration = {
        id: row.id,
        direction: row.direction,
        context: row.context,
        ebGeneration: row.ebGeneration,
      };
      this.seen.add(normalizeDirection(row.direction));
      if (row.state === "held") this.held.set(item.id, item);
      else this.waiting.push(item);
    }
    this.nextId = Math.max(0, ...[...rows.map((row) => row.id), ...knownIds].map(inspirationSeq)) + 1;
  }

  private async persist(item: Inspiration, state: "waiting" | "held"): Promise<void> {
    const file = path.join(this.runDir, "queue", `${item.id}.json`);
    await writeJsonFile(file, { ...item, state });
  }
}

function normalizeDirection(direction: string): string {
  return direction.trim().replace(/\s+/g, " ");
}

function inspirationSeq(id: string): number {
  const match = /^insp-(\d+)$/.exec(id);
  return match ? Number(match[1]) : 0;
}
