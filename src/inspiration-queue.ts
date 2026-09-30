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
  | { ok: false; reason: "full" | "duplicate" | "quota" };

// 同一方向组（「信号源：」前缀相同的灵感）在 waiting + held 里的并发上限。
// 并行 Proposal 会同时挖同一信号源，同质因子堆在一起只会放大共线性，
// 所以队列侧留一道硬闸兜底。
export const DEFAULT_DIRECTION_QUOTA = 2;

type QueueRow = Inspiration & { state?: string };

export class InspirationQueue {
  readonly runDir: string;
  readonly lowWater: number;
  readonly highWater: number;
  readonly directionQuota: number;
  private nextId = 1;
  private waiting: Inspiration[] = [];
  private held = new Map<string, Inspiration>();
  private seen = new Set<string>();
  // 方向组 → 该组 waiting + held 的条数，重启时按 queue/ 里的文件重建。
  private groupCounts = new Map<string, number>();

  private constructor(
    runDir: string,
    lowWater: number,
    highWater: number,
    directionQuota: number,
  ) {
    this.runDir = runDir;
    this.lowWater = lowWater;
    this.highWater = highWater;
    this.directionQuota = directionQuota;
  }

  static async open(
    runDir: string,
    opts: { lowWater: number; highWater: number; directionQuota?: number; knownIds?: string[] },
  ): Promise<InspirationQueue> {
    if (opts.lowWater < 0 || opts.highWater < 1 || opts.lowWater > opts.highWater) {
      throw new Error("queue marks must satisfy 0 <= lowWater <= highWater");
    }
    const directionQuota = opts.directionQuota ?? DEFAULT_DIRECTION_QUOTA;
    if (!Number.isInteger(directionQuota) || directionQuota < 1) {
      throw new Error("directionQuota must be a positive integer");
    }
    const queue = new InspirationQueue(runDir, opts.lowWater, opts.highWater, directionQuota);
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
    const group = directionGroup(input.direction);
    if ((this.groupCounts.get(group) ?? 0) >= this.directionQuota) {
      return { ok: false, reason: "quota" };
    }
    const item: Inspiration = {
      id: `insp-${String(this.nextId).padStart(3, "0")}`,
      direction: input.direction,
      context: input.context,
      ebGeneration: input.ebGeneration,
    };
    this.nextId += 1;
    this.seen.add(key);
    this.bumpGroup(group, 1);
    this.waiting.push(item);
    await this.persist(item, "waiting");
    return { ok: true, id: item.id };
  }

  async claim(): Promise<Inspiration | undefined> {
    const item = this.waiting.shift();
    if (!item) return undefined;
    // waiting → held 不改组计数：配额算的是两者之和。
    this.held.set(item.id, item);
    await this.persist(item, "held");
    return { ...item };
  }

  releaseHold(id: string): void {
    const item = this.held.get(id);
    if (!item) return;
    this.held.delete(id);
    this.bumpGroup(directionGroup(item.direction), -1);
  }

  async requeueOrphans(committedIds: Set<string>): Promise<void> {
    const orphans: Inspiration[] = [];
    for (const [id, item] of this.held) {
      if (committedIds.has(id)) continue;
      this.held.delete(id);
      orphans.push(item);
    }
    // held → waiting 同样不改组计数，崩溃恢复后配额继续按原样生效。
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
      this.bumpGroup(directionGroup(row.direction), 1);
      if (row.state === "held") this.held.set(item.id, item);
      else this.waiting.push(item);
    }
    this.nextId = Math.max(0, ...[...rows.map((row) => row.id), ...knownIds].map(inspirationSeq)) + 1;
  }

  private bumpGroup(group: string, delta: number): void {
    const next = (this.groupCounts.get(group) ?? 0) + delta;
    if (next > 0) this.groupCounts.set(group, next);
    else this.groupCounts.delete(group);
  }

  private async persist(item: Inspiration, state: "waiting" | "held"): Promise<void> {
    const file = path.join(this.runDir, "queue", `${item.id}.json`);
    await writeJsonFile(file, { ...item, state });
  }
}

// 配额按方向分组算：取第一个全角/半角冒号之前的前缀（trim + 小写）当组键，
// 没有冒号就整句当组键。direction 的约定是「信号源：机制」，所以前缀即信号源。
export function directionGroup(direction: string): string {
  const normalized = normalizeDirection(direction);
  const cut = normalized.search(/[：:]/);
  const prefix = cut >= 0 ? normalized.slice(0, cut) : normalized;
  return prefix.trim().toLowerCase();
}

export function normalizeDirection(direction: string): string {
  return direction.trim().replace(/\s+/g, " ");
}

function inspirationSeq(id: string): number {
  const match = /^insp-(\d+)$/.exec(id);
  return match ? Number(match[1]) : 0;
}
