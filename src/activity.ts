import { appendFile, mkdir, readFile } from "node:fs/promises";
import path from "node:path";

export type ActivityActor = "context" | "proposal" | "sandbox";
export type ActivityKind = "think" | "say" | "tool" | "result" | "error";

export type ActivityEvent = {
  at: number;
  actor: ActivityActor;
  id: string;
  kind: ActivityKind;
  text: string;
  tool?: string;
};

export type ActivityBlock = {
  kind: ActivityKind;
  text: string;
  tool?: string;
};

export function groupActivityEvents(events: ActivityEvent[]): ActivityBlock[] {
  const blocks: ActivityBlock[] = [];
  for (const event of events) {
    const last = blocks[blocks.length - 1];
    const joinText = event.kind === "think" || event.kind === "say";
    if (last && last.kind === event.kind && joinText && last.tool === event.tool) {
      last.text += event.text;
      continue;
    }
    const block: ActivityBlock = { kind: event.kind, text: event.text };
    if (event.tool) block.tool = event.tool;
    blocks.push(block);
  }
  return blocks;
}

export function activityFile(runDir: string, id: string): string {
  return path.join(runDir, "activity", `${id}.jsonl`);
}

export async function appendActivity(runDir: string, event: ActivityEvent): Promise<void> {
  const file = activityFile(runDir, event.id);
  await mkdir(path.dirname(file), { recursive: true });
  await appendFile(file, `${JSON.stringify(event)}\n`, "utf8");
}

export async function readActivity(runDir: string, id: string, limit = 80): Promise<ActivityEvent[]> {
  let text: string;
  try {
    text = await readFile(activityFile(runDir, id), "utf8");
  } catch {
    return [];
  }
  const rows: ActivityEvent[] = [];
  for (const line of text.split("\n")) {
    if (!line.length) continue;
    try {
      rows.push(JSON.parse(line) as ActivityEvent);
    } catch {
      continue;
    }
  }
  return rows.length <= limit ? rows : rows.slice(-limit);
}

export function activityFromPiEvent(
  event: unknown,
  ctx: { actor: ActivityActor; id: string; at?: number },
): ActivityEvent | undefined {
  if (!event || typeof event !== "object") return undefined;
  const row = event as PiLikeEvent;
  const at = ctx.at ?? Date.now();
  const base = { at, actor: ctx.actor, id: ctx.id };

  if (row.type === "message_update") {
    const inner = row.assistantMessageEvent;
    if (!inner || typeof inner !== "object") return undefined;
    if (inner.type === "thinking_delta" && inner.delta) {
      return { ...base, kind: "think", text: inner.delta };
    }
    if (inner.type === "text_delta" && inner.delta) {
      return { ...base, kind: "say", text: inner.delta };
    }
    return undefined;
  }

  if (row.type === "tool_execution_start") {
    const tool = row.toolName ?? "tool";
    return { ...base, kind: "tool", tool, text: summarizeTool(tool, row.args) };
  }

  if (row.type === "tool_execution_end") {
    const tool = row.toolName ?? "tool";
    const text = summarizeResult(row.result, row.isError === true);
    return { ...base, kind: row.isError ? "error" : "result", tool, text };
  }

  if (row.type === "bash_execution_update" && row.delta) {
    return { ...base, kind: "result", tool: "bash", text: row.delta };
  }

  return undefined;
}

export class ActivitySink {
  private buffer: ActivityEvent | undefined;
  private writes: Promise<void> = Promise.resolve();
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(private readonly opts: { runDir: string; actor: ActivityActor; id: string }) {}

  handle(event: unknown): void {
    const next = activityFromPiEvent(event, { actor: this.opts.actor, id: this.opts.id });
    if (next && (next.kind === "think" || next.kind === "say")) {
      this.absorb(next);
      this.scheduleFlush();
      return;
    }
    this.enqueue(async () => {
      await this.flushBuffer();
      if (next) await appendActivity(this.opts.runDir, next);
    });
  }

  async flush(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = undefined;
    }
    this.enqueue(async () => {
      await this.flushBuffer();
    });
    await this.writes;
  }

  private absorb(next: ActivityEvent): void {
    if (this.buffer && this.buffer.kind === next.kind && this.buffer.tool === next.tool) {
      this.buffer = { ...this.buffer, text: this.buffer.text + next.text, at: next.at };
      return;
    }
    const previous = this.buffer;
    this.buffer = next;
    if (previous) {
      this.enqueue(async () => {
        await appendActivity(this.opts.runDir, previous);
      });
    }
  }

  private scheduleFlush(): void {
    if (this.timer) return;
    this.timer = setTimeout(() => {
      this.timer = undefined;
      this.enqueue(async () => {
        await this.flushBuffer();
      });
    }, 250);
  }

  private async flushBuffer(): Promise<void> {
    const pending = this.buffer;
    this.buffer = undefined;
    if (pending && pending.text.trim().length > 0) {
      await appendActivity(this.opts.runDir, pending);
    }
  }

  private enqueue(job: () => Promise<void>): void {
    this.writes = this.writes.then(job, job);
  }
}

type PiLikeEvent = {
  type?: string;
  toolName?: string;
  args?: unknown;
  result?: unknown;
  isError?: boolean;
  delta?: string;
  assistantMessageEvent?: { type?: string; delta?: string };
};

function summarizeTool(name: string, args: unknown): string {
  const fields = asRecord(args);
  const file = pick(fields, ["path", "file", "file_path", "filePath"]);
  const command = pick(fields, ["command", "cmd"]);
  const query = pick(fields, ["query", "libraryName", "libraryId", "libraryID"]);
  if (command) return `${name} ${clip(command, 180)}`;
  if (file) return `${name} ${file}`;
  if (query) return `${name} ${clip(query, 140)}`;
  return name;
}

function summarizeResult(result: unknown, isError: boolean): string {
  const text =
    typeof result === "string"
      ? result
      : result && typeof result === "object" && "content" in result && typeof result.content === "string"
        ? result.content
        : result
          ? JSON.stringify(result)
          : "";
  const clipped = clip(text.replace(/\s+/g, " ").trim(), 280);
  if (!clipped) return isError ? "工具失败" : "完成";
  return clipped;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function pick(fields: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = fields[key];
    if (typeof value === "string" && value.length > 0) return value;
  }
  return undefined;
}

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max)}…`;
}
