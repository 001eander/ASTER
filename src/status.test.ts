import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { ExperienceBank } from "./experience-bank.js";
import { buildStatus, renderStatusHtml } from "./status.js";

async function seedRun(): Promise<string> {
  const runDir = await mkdtemp(path.join(tmpdir(), "hyra-pi-st-"));
  await writeFile(
    path.join(runDir, "run.json"),
    JSON.stringify({
      maxProposals: 2,
      maxSandboxes: 2,
      lowWater: 1,
      highWater: 3,
      maxSolutions: 10,
      contextMaxIdleMs: 5000,
    }),
    "utf8",
  );
  await writeFile(
    path.join(runDir, "live.json"),
    JSON.stringify({
      phase: "running",
      writers: [
        { id: "insp-002", direction: "rewrite inner loop" },
        { id: "insp-004", direction: "tune pivot" },
      ],
      sandboxes: [
        { id: "insp-005", direction: "hash join" },
        { id: "insp-003", direction: "cache keys" },
      ],
      contextRunning: false,
      lastContextAt: 1_000,
      startedAt: 0,
    }),
    "utf8",
  );
  await mkdir(path.join(runDir, "queue"), { recursive: true });
  await writeFile(
    path.join(runDir, "queue", "insp-001.json"),
    JSON.stringify({
      id: "insp-001",
      direction: "try merge sort",
      context: "best is 4",
      ebGeneration: 1,
      state: "waiting",
    }),
    "utf8",
  );

  const incoming = path.join(runDir, "incoming");
  await mkdir(incoming, { recursive: true });
  await writeFile(path.join(incoming, "solve.sh"), "#!/bin/sh\necho seeded\n", "utf8");
  const bank = await ExperienceBank.open(runDir);
  await bank.commit({
    inspirationId: "insp-000",
    solutionDir: incoming,
    ok: true,
    log: "ok",
    score: { score: 4, higherIsBetter: true, notes: "seed" },
  });
  return runDir;
}

describe("status view", () => {
  it("builds the page data from the run folder", async () => {
    const runDir = await seedRun();
    const view = await buildStatus(runDir, { now: 1_200 });

    expect(view.phase).toBe("running");
    expect(view.healthy).toBe(true);
    expect(view.budget).toEqual({ maxSolutions: 10, remainingSolutions: 9 });
    expect(view.queue).toEqual([{ id: "insp-001", direction: "try merge sort" }]);
    expect(view.writers).toEqual([
      { id: "insp-002", direction: "rewrite inner loop" },
      { id: "insp-004", direction: "tune pivot" },
    ]);
    expect(view.sandboxes).toEqual([
      { id: "insp-003", direction: "cache keys" },
      { id: "insp-005", direction: "hash join" },
    ]);
    expect(view.context).toEqual({ running: false, lastContextAt: 1_000 });
    expect(view.best).toEqual({ id: "0001", inspirationId: "insp-000", score: 4, higherIsBetter: true });
    expect(view.history).toEqual([
      {
        id: "0001",
        inspirationId: "insp-000",
        ok: true,
        score: 4,
        notes: "seed",
        logTail: "ok",
        stdoutTail: "ok",
      },
    ]);
    expect(view.traces.filter((row) => row.actor === "proposal").map((row) => row.id)).toEqual([
      "insp-001",
      "insp-002",
      "insp-004",
    ]);
  });

  it("puts the writer's thinking and last tool into the live traces", async () => {
    const runDir = await seedRun();
    await mkdir(path.join(runDir, "activity"), { recursive: true });
    await writeFile(
      path.join(runDir, "activity", "insp-002.jsonl"),
      [
        JSON.stringify({
          at: 1_100,
          actor: "proposal",
          id: "insp-002",
          kind: "think",
          text: "先复述说明书：只改目标编码",
        }),
        JSON.stringify({
          at: 1_150,
          actor: "proposal",
          id: "insp-002",
          kind: "tool",
          tool: "write",
          text: "write solve.py",
        }),
      ].join("\n") + "\n",
      "utf8",
    );
    await mkdir(path.join(runDir, "workspaces", "insp-002"), { recursive: true });
    await writeFile(path.join(runDir, "workspaces", "insp-002", "solve.py"), "print(1)\n", "utf8");

    const view = await buildStatus(runDir, { now: 1_200 });
    const writer = view.traces.find((row) => row.id === "insp-002");
    expect(writer?.actor).toBe("proposal");
    expect(writer?.doing).toBe("write solve.py");
    expect(writer?.files).toContain("solve.py");
    expect(writer?.events.map((row) => row.kind)).toEqual(["think", "tool"]);
    expect(renderStatusHtml(view)).toContain("先复述说明书：只改目标编码");
    expect(renderStatusHtml(view)).toContain("write solve.py");
    const card = renderStatusHtml(view).match(/<article[^>]*data-open="insp-002"[\s\S]*?<\/article>/)?.[0];
    expect(card).toContain("write solve.py");
    expect(card).not.toContain("先复述说明书：只改目标编码");
    expect(renderStatusHtml(view)).toContain("<dialog");
  });

  it("joins consecutive thinking fragments into one dialog block", async () => {
    const runDir = await seedRun();
    await mkdir(path.join(runDir, "activity"), { recursive: true });
    await writeFile(
      path.join(runDir, "activity", "insp-002.jsonl"),
      [
        JSON.stringify({ at: 1, actor: "proposal", id: "insp-002", kind: "think", text: "先看" }),
        JSON.stringify({ at: 2, actor: "proposal", id: "insp-002", kind: "think", text: "日志再决定" }),
        JSON.stringify({ at: 3, actor: "proposal", id: "insp-002", kind: "say", text: "开始写" }),
      ].join("\n") + "\n",
      "utf8",
    );

    const html = renderStatusHtml(await buildStatus(runDir, { now: 1_200 }));
    expect(html).toContain('class="block think"');
    expect(html).toContain("先看日志再决定");
    expect(html).toContain('class="block say"');
    expect((html.match(/class="block think"/g) ?? []).length).toBe(1);
    expect(html).not.toContain('class="line think"');
  });

  it("does not put Proposal thinking onto a sandbox card", async () => {
    const runDir = await seedRun();
    await mkdir(path.join(runDir, "activity"), { recursive: true });
    await writeFile(
      path.join(runDir, "activity", "insp-003.jsonl"),
      JSON.stringify({
        at: 1_100,
        actor: "proposal",
        id: "insp-003",
        kind: "think",
        text: "先复述说明书：只改缓存键",
      }) + "\n",
      "utf8",
    );
    await mkdir(path.join(runDir, "workspaces", "insp-003"), { recursive: true });
    await writeFile(path.join(runDir, "workspaces", "insp-003", "eval.log"), "[train] best_iteration=62\n", "utf8");

    const view = await buildStatus(runDir, { now: 1_200 });
    const box = view.traces.find((row) => row.id === "insp-003");
    expect(box?.actor).toBe("sandbox");
    expect(box?.doing).toBe("[train] best_iteration=62");
    expect(box?.events).toEqual([]);
    const html = renderStatusHtml(view);
    expect(html).toContain("[train] best_iteration=62");
    expect(html).not.toContain("先复述说明书：只改缓存键");
  });

  it("lays the Proposal dialog out as inspiration on the left and the live trail on the right", async () => {
    const runDir = await seedRun();
    await writeFile(
      path.join(runDir, "queue", "insp-002.json"),
      JSON.stringify({
        id: "insp-002",
        direction: "rewrite inner loop",
        context: "只改目标编码，别动模型",
        state: "held",
      }),
      "utf8",
    );
    await mkdir(path.join(runDir, "activity"), { recursive: true });
    await writeFile(
      path.join(runDir, "activity", "insp-002.jsonl"),
      JSON.stringify({ at: 1, actor: "proposal", id: "insp-002", kind: "say", text: "开始写 solve.py" }) + "\n",
      "utf8",
    );

    const html = renderStatusHtml(await buildStatus(runDir, { now: 1_200 }));
    const mark = html.indexOf('data-detail="insp-002"');
    const next = html.indexOf("data-detail=", mark + 1);
    const detail = html.slice(mark, next === -1 ? undefined : next);
    expect(detail).toContain('data-layout="proposal"');
    expect(detail.indexOf("只改目标编码，别动模型")).toBeLessThan(detail.indexOf("开始写 solve.py"));
    expect(detail).toContain("灵感");
    expect(detail).toContain("轨迹");
    expect(detail).toContain("sheet-split");
  });

  it("lays the sandbox dialog out as inspiration above split STDOUT and STDERR", async () => {
    const runDir = await seedRun();
    await writeFile(
      path.join(runDir, "queue", "insp-003.json"),
      JSON.stringify({
        id: "insp-003",
        direction: "cache keys",
        context: "只改缓存键",
        state: "held",
      }),
      "utf8",
    );
    await mkdir(path.join(runDir, "workspaces", "insp-003"), { recursive: true });
    await writeFile(
      path.join(runDir, "workspaces", "insp-003", "eval.log"),
      [
        "Rows: train=10",
        "/usr/local/lib/python3.12/site-packages/lightgbm/sklearn.py:1106: LGBMDeprecationWarning: eval_set is deprecated",
        "  eval_set = _validate_eval_set_Xy(eval_set=eval_set)",
        "  fold 1: AUC=0.94",
      ].join("\n") + "\n",
      "utf8",
    );

    const view = await buildStatus(runDir, { now: 1_200 });
    const box = view.traces.find((row) => row.id === "insp-003");
    expect(box?.stdoutTail).toContain("Rows: train=10");
    expect(box?.stdoutTail).toContain("fold 1: AUC=0.94");
    expect(box?.stderrTail).toContain("LGBMDeprecationWarning");
    expect(box?.stderrTail).toContain("eval_set = _validate");
    expect(box?.doing).toBe("fold 1: AUC=0.94");

    const html = renderStatusHtml(view);
    const mark = html.indexOf('data-detail="insp-003"');
    const next = html.indexOf("data-detail=", mark + 1);
    const detail = html.slice(mark, next === -1 ? undefined : next);
    expect(detail).toContain('data-layout="sandbox"');
    expect(detail).toContain("只改缓存键");
    expect(detail).toContain("STDOUT");
    expect(detail).toContain("STDERR");
    expect(detail.indexOf("只改缓存键")).toBeLessThan(detail.indexOf("STDOUT"));
    expect(detail.indexOf("STDOUT")).toBeLessThan(detail.indexOf("STDERR"));
  });

  it("prefers split sandbox logs over the mixed eval.log", async () => {
    const runDir = await seedRun();
    await mkdir(path.join(runDir, "workspaces", "insp-003"), { recursive: true });
    await writeFile(path.join(runDir, "workspaces", "insp-003", "eval.log"), "mixed leftover\n", "utf8");
    await writeFile(path.join(runDir, "workspaces", "insp-003", "eval.stdout"), "printed score\n", "utf8");
    await writeFile(path.join(runDir, "workspaces", "insp-003", "eval.stderr"), "lgbm warn\n", "utf8");

    const box = (await buildStatus(runDir, { now: 1_200 })).traces.find((row) => row.id === "insp-003");
    expect(box?.stdoutTail).toBe("printed score");
    expect(box?.stderrTail).toBe("lgbm warn");
    expect(box?.doing).toBe("printed score");
  });

  it("copies the health reason when the loop looks stuck", async () => {
    const runDir = await seedRun();
    await writeFile(
      path.join(runDir, "live.json"),
      JSON.stringify({
        phase: "running",
        writers: [],
        sandboxes: [],
        contextRunning: false,
        lastContextAt: 1_000,
        startedAt: 0,
      }),
      "utf8",
    );
    const view = await buildStatus(runDir, { now: 1_200 });
    expect(view.healthy).toBe(false);
    expect(view.reason).toBe("队列里有活，但没人领");
  });

  it("uses consumedMs for elapsed time so a paused run does not look like it ran overnight", async () => {
    const runDir = await seedRun();
    await writeFile(
      path.join(runDir, "live.json"),
      JSON.stringify({
        phase: "stopped",
        writers: [],
        sandboxes: [],
        contextRunning: false,
        startedAt: 0,
        consumedMs: 90_000,
      }),
      "utf8",
    );
    const view = await buildStatus(runDir, { now: 1_000_000 });
    expect(view.elapsedMs).toBe(90_000);
    expect(renderStatusHtml(view)).toContain("1 分 30 秒");
  });

  it("says the solution budget is unlimited when run.json has no cap", async () => {
    const runDir = await seedRun();
    await writeFile(
      path.join(runDir, "run.json"),
      JSON.stringify({
        maxProposals: 2,
        maxSandboxes: 2,
        lowWater: 1,
        highWater: 3,
        contextMaxIdleMs: 5000,
      }),
      "utf8",
    );
    const view = await buildStatus(runDir, { now: 1_200 });
    expect(view.budget).toEqual({ unlimited: true });
    expect(renderStatusHtml(view)).toContain("不限份数");
  });

  it("renders a read-only page with the best score and queue direction", async () => {
    const runDir = await seedRun();
    const html = renderStatusHtml(await buildStatus(runDir, { now: 1_200 }));
    expect(html).toContain("try merge sort");
    expect(html).toContain("最好成绩 · insp-000 · 实验 0001");
    expect(html).toContain("4");
    expect(html).toContain("rewrite inner loop");
    expect(html).toContain("cache keys");
    expect(html).not.toContain("kill");
    expect(html).not.toContain("<form");
  });

  it("orders history by inspiration id, not sandbox id", async () => {
    const runDir = await seedRun();
    const incoming = path.join(runDir, "later");
    await mkdir(incoming, { recursive: true });
    await writeFile(path.join(incoming, "solve.sh"), "#!/bin/sh\necho later\n", "utf8");
    const bank = await ExperienceBank.open(runDir);
    await bank.commit({
      inspirationId: "insp-009",
      solutionDir: incoming,
      ok: true,
      log: "later",
      score: { score: 1, higherIsBetter: true, notes: "later" },
    });
    await bank.commit({
      inspirationId: "insp-002",
      solutionDir: incoming,
      ok: true,
      log: "mid",
      score: { score: 2, higherIsBetter: true, notes: "mid" },
    });

    const view = await buildStatus(runDir, { now: 1_200 });
    expect(view.history.map((row) => row.inspirationId)).toEqual([
      "insp-000",
      "insp-002",
      "insp-009",
    ]);
    const html = renderStatusHtml(view);
    const table = html.slice(html.indexOf("历史结果"));
    expect(table.indexOf("insp-000")).toBeLessThan(table.indexOf("insp-002"));
    expect(table.indexOf("insp-002")).toBeLessThan(table.indexOf("insp-009"));
  });

  it("splits history ids and marks the sortable columns", async () => {
    const html = renderStatusHtml(await buildStatus(await seedRun(), { now: 1_200 }));
    const table = html.slice(html.indexOf("历史结果"));
    expect(table).toContain('data-sort="inspiration"');
    expect(table).toContain('data-sort="experiment"');
    expect(table).toContain('data-sort="result"');
    expect(table).toContain('data-sort="score"');
    expect(table).toContain("灵感 ID");
    expect(table).toContain("实验 ID");
    expect(table).toContain('data-experiment="0001"');
    expect(table).toContain('data-inspiration="insp-000"');
    expect(table.indexOf("灵感 ID")).toBeLessThan(table.indexOf("实验 ID"));
    expect(table.indexOf("实验 ID")).toBeLessThan(table.indexOf(">结果<"));
    expect(table.indexOf(">结果<")).toBeLessThan(table.indexOf(">分数<"));
  });

  it("lays the page out as a board: context and status on top, then proposal and sandbox columns", async () => {
    const html = renderStatusHtml(await buildStatus(await seedRun(), { now: 1_200 }));
    const contextAt = html.indexOf(">Context<");
    const statusAt = html.indexOf(">状态<");
    const proposalAt = html.indexOf(">Proposal<");
    const sandboxAt = html.indexOf(">沙盒<");
    const historyAt = html.indexOf(">历史结果<");
    expect(contextAt).toBeGreaterThan(-1);
    expect(statusAt).toBeGreaterThan(contextAt);
    expect(proposalAt).toBeGreaterThan(statusAt);
    expect(sandboxAt).toBeGreaterThan(proposalAt);
    expect(historyAt).toBeGreaterThan(sandboxAt);
    expect(html).toContain("排队等待领取");
    expect(html.indexOf("try merge sort")).toBeLessThan(html.indexOf("rewrite inner loop"));
    expect(html.indexOf("rewrite inner loop")).toBeLessThan(html.indexOf("tune pivot"));
    expect(html.indexOf("rewrite inner loop")).toBeLessThan(html.indexOf("cache keys"));
    expect(html.indexOf("cache keys")).toBeLessThan(html.indexOf("hash join"));
  });

  it("lays history out as inspiration above split STDOUT and STDERR", async () => {
    const runDir = await seedRun();
    await writeFile(
      path.join(runDir, "queue", "insp-009.json"),
      JSON.stringify({
        id: "insp-009",
        direction: "try rules",
        context: "穷举规则，找不到就回退",
        state: "done",
      }),
      "utf8",
    );
    const incoming = path.join(runDir, "later");
    await mkdir(incoming, { recursive: true });
    await writeFile(path.join(incoming, "solve.sh"), "#!/bin/sh\necho later\n", "utf8");
    await writeFile(
      path.join(incoming, "eval.log"),
      [
        "Rows: train=10",
        "/usr/local/lib/python3.12/site-packages/lightgbm/sklearn.py:1106: LGBMDeprecationWarning: eval_set is deprecated",
        "  eval_set = _validate_eval_set_Xy(eval_set=eval_set)",
        "  fold 1: AUC=0.94",
      ].join("\n") + "\n",
      "utf8",
    );
    const bank = await ExperienceBank.open(runDir);
    await bank.commit({
      inspirationId: "insp-009",
      solutionDir: incoming,
      ok: true,
      log: "unused mixed leftover",
      score: { score: 1, higherIsBetter: true, notes: "auc=0.94" },
    });

    const view = await buildStatus(runDir, { now: 1_200 });
    const row = view.history.find((item) => item.inspirationId === "insp-009");
    expect(row?.brief).toBe("穷举规则，找不到就回退");
    expect(row?.stdoutTail).toContain("Rows: train=10");
    expect(row?.stdoutTail).toContain("fold 1: AUC=0.94");
    expect(row?.stderrTail).toContain("LGBMDeprecationWarning");

    const html = renderStatusHtml(view);
    const mark = html.indexOf(`data-detail="hist-${row?.id}"`);
    const next = html.indexOf("data-detail=", mark + 1);
    const detail = html.slice(mark, next === -1 ? undefined : next);
    expect(detail).toContain('data-layout="sandbox"');
    expect(detail).toContain("穷举规则，找不到就回退");
    expect(detail).toContain("STDOUT");
    expect(detail).toContain("STDERR");
    expect(detail.indexOf("穷举规则，找不到就回退")).toBeLessThan(detail.indexOf("STDOUT"));
    expect(detail.indexOf("STDOUT")).toBeLessThan(detail.indexOf("STDERR"));
  });

  it("only pins live panes to the bottom when the reader is already there", async () => {
    const html = renderStatusHtml(await buildStatus(await seedRun(), { now: 1_200 }));
    expect(html).toContain("function nearBottom");
    expect(html).toContain("saved.stick");
    expect(html).not.toContain("follow.forEach((el) => { el.scrollTop = el.scrollHeight; })");
  });
});
