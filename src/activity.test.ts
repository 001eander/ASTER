import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { ActivitySink, activityFromPiEvent, groupActivityEvents, readActivity } from "./activity.js";

describe("activityFromPiEvent", () => {
  const ctx = { actor: "proposal" as const, id: "insp-002", at: 10 };

  it("turns thinking and text deltas into think/say events", () => {
    expect(
      activityFromPiEvent(
        { type: "message_update", assistantMessageEvent: { type: "thinking_delta", delta: "先看日志" } },
        ctx,
      ),
    ).toEqual({ at: 10, actor: "proposal", id: "insp-002", kind: "think", text: "先看日志" });
    expect(
      activityFromPiEvent(
        { type: "message_update", assistantMessageEvent: { type: "text_delta", delta: "开始写" } },
        ctx,
      ),
    ).toEqual({ at: 10, actor: "proposal", id: "insp-002", kind: "say", text: "开始写" });
  });

  it("summarizes tool calls by the file or command they touch", () => {
    expect(
      activityFromPiEvent(
        { type: "tool_execution_start", toolName: "write", args: { path: "solve.py", content: "print(1)" } },
        ctx,
      ),
    ).toEqual({
      at: 10,
      actor: "proposal",
      id: "insp-002",
      kind: "tool",
      tool: "write",
      text: "write solve.py",
    });
    expect(
      activityFromPiEvent(
        { type: "tool_execution_start", toolName: "bash", args: { command: "python3 -m py_compile solve.py" } },
        ctx,
      )?.text,
    ).toBe("bash python3 -m py_compile solve.py");
  });

  it("records tool results and errors", () => {
    expect(
      activityFromPiEvent(
        { type: "tool_execution_end", toolName: "read", isError: false, result: "#!/bin/sh\necho ok\n" },
        ctx,
      ),
    ).toMatchObject({ kind: "result", tool: "read", text: "#!/bin/sh echo ok" });
    expect(
      activityFromPiEvent(
        { type: "tool_execution_end", toolName: "bash", isError: true, result: "SyntaxError" },
        ctx,
      ),
    ).toMatchObject({ kind: "error", tool: "bash", text: "SyntaxError" });
  });
});

describe("ActivitySink", () => {
  it("joins thinking deltas, then writes the tool that follows", async () => {
    const runDir = await mkdtemp(path.join(tmpdir(), "hyra-pi-act-"));
    const sink = new ActivitySink({ runDir, actor: "proposal", id: "insp-002" });
    sink.handle({
      type: "message_update",
      assistantMessageEvent: { type: "thinking_delta", delta: "先读" },
    });
    sink.handle({
      type: "message_update",
      assistantMessageEvent: { type: "thinking_delta", delta: "现有方案" },
    });
    sink.handle({
      type: "tool_execution_start",
      toolName: "read",
      args: { path: "solve.sh" },
    });
    await sink.flush();

    const events = await readActivity(runDir, "insp-002");
    expect(events.map((row) => row.kind)).toEqual(["think", "tool"]);
    expect(events[0]?.text).toBe("先读现有方案");
    expect(events[1]?.text).toBe("read solve.sh");
  });
});

describe("groupActivityEvents", () => {
  it("joins consecutive thinking and speech into one block each", () => {
    const blocks = groupActivityEvents([
      { at: 1, actor: "proposal", id: "i", kind: "think", text: "先看" },
      { at: 2, actor: "proposal", id: "i", kind: "think", text: "日志" },
      { at: 3, actor: "proposal", id: "i", kind: "say", text: "开始" },
      { at: 4, actor: "proposal", id: "i", kind: "say", text: "写" },
      { at: 5, actor: "proposal", id: "i", kind: "tool", tool: "write", text: "write solve.py" },
      { at: 6, actor: "proposal", id: "i", kind: "think", text: "再检查" },
    ]);
    expect(blocks).toEqual([
      { kind: "think", text: "先看日志" },
      { kind: "say", text: "开始写" },
      { kind: "tool", tool: "write", text: "write solve.py" },
      { kind: "think", text: "再检查" },
    ]);
  });
});
