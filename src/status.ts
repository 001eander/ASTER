import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { groupActivityEvents, readActivity, type ActivityActor, type ActivityEvent } from "./activity.js";
import { ExperienceBank } from "./experience-bank.js";
import { assessHealth, type HealthSnapshot } from "./health.js";
import type { LiveState, LoopPhase } from "./inner-loop.js";
import { readJsonFile } from "./json-file.js";

export type StatusView = {
  phase: LoopPhase;
  stopReason?: string;
  healthy: boolean;
  reason?: string;
  startedAt: number;
  elapsedMs: number;
  budget: { maxSolutions?: number; remainingSolutions?: number; unlimited?: boolean };
  queue: Array<{ id: string; direction: string }>;
  writers: Array<{ id: string; direction: string }>;
  sandboxes: Array<{ id: string; direction: string }>;
  context: { running: boolean; lastContextAt?: number };
  best?: { id: string; inspirationId: string; score: number; higherIsBetter: boolean };
  traces: AgentTrace[];
  history: HistoryRow[];
};

export type AgentTrace = {
  id: string;
  actor: ActivityActor;
  title: string;
  direction?: string;
  brief?: string;
  doing: string;
  files?: string[];
  logTail?: string;
  stdoutTail?: string;
  stderrTail?: string;
  events: ActivityEvent[];
  pending?: boolean;
};

export type HistoryRow = {
  id: string;
  inspirationId: string;
  direction?: string;
  brief?: string;
  ok: boolean;
  score: number | null;
  notes?: string;
  logTail?: string;
  stdoutTail?: string;
  stderrTail?: string;
};

type RunConfig = {
  maxProposals: number;
  maxSandboxes: number;
  lowWater: number;
  highWater: number;
  maxSolutions?: number;
  contextMaxIdleMs: number;
};

type QueueRow = {
  id: string;
  direction: string;
  context?: string;
  state?: string;
};

export async function buildStatus(
  runDir: string,
  opts: { now?: number } = {},
): Promise<StatusView> {
  const now = opts.now ?? Date.now();
  const config = await readJsonFile<RunConfig>(path.join(runDir, "run.json"));
  const live = await readJsonFile<LiveState>(path.join(runDir, "live.json"));
  const bank = await ExperienceBank.open(runDir);
  const records = await bank.list();
  const bestRecord = await bank.best();
  const queued = await readQueue(runDir);
  const waiting = queued.filter((row) => row.state === "waiting" || row.state === undefined);
  const briefs = new Map(queued.map((row) => [row.id, row]));

  const writers = live.writers ?? [];
  const sandboxes = live.sandboxes ?? [];
  const healthInput: HealthSnapshot = {
    phase: live.phase,
    waitingCount: waiting.length,
    lowWater: config.lowWater,
    highWater: config.highWater,
    proposalsInFlight: writers.length,
    maxProposals: config.maxProposals,
    sandboxesInFlight: sandboxes.length,
    maxSandboxes: config.maxSandboxes,
    contextRunning: live.contextRunning,
    lastContextAt: live.lastContextAt,
    now,
    contextMaxIdleMs: config.contextMaxIdleMs,
    heldInspirationIds: [...writers, ...sandboxes].map((row) => row.id),
    writers: writers.map((row) => row.id),
  };
  const health = assessHealth(healthInput);

  const history: HistoryRow[] = [];
  for (const row of records) {
    const brief = briefs.get(row.inspirationId);
    const streams = await readHistoryStreams(row);
    const item: HistoryRow = {
      id: row.id,
      inspirationId: row.inspirationId,
      ok: row.ok,
      score: row.score?.score ?? null,
    };
    if (brief?.direction) item.direction = brief.direction;
    if (brief?.context) item.brief = brief.context;
    if (row.score?.notes) item.notes = row.score.notes;
    if (streams.mixed) item.logTail = streams.mixed;
    if (streams.stdout) item.stdoutTail = streams.stdout;
    if (streams.stderr) item.stderrTail = streams.stderr;
    history.push(item);
  }
  history.sort(byInspirationId);

  return {
    phase: live.phase,
    stopReason: live.stopReason,
    healthy: health.healthy,
    reason: health.reason,
    startedAt: live.startedAt ?? 0,
    elapsedMs:
      live.consumedMs !== undefined ? Math.max(0, live.consumedMs) : Math.max(0, now - (live.startedAt ?? now)),
    budget:
      config.maxSolutions === undefined
        ? { unlimited: true }
        : {
            maxSolutions: config.maxSolutions,
            remainingSolutions: Math.max(0, config.maxSolutions - bank.generation()),
          },
    queue: waiting.map((row) => ({ id: row.id, direction: row.direction })).sort(byInspirationId),
    writers: writers.slice().sort(byInspirationId),
    sandboxes: sandboxes.slice().sort(byInspirationId),
    context: { running: live.contextRunning, lastContextAt: live.lastContextAt },
    best: bestRecord?.score
      ? {
          id: bestRecord.id,
          inspirationId: bestRecord.inspirationId,
          score: bestRecord.score.score,
          higherIsBetter: bestRecord.score.higherIsBetter,
        }
      : undefined,
    traces: await buildTraces(runDir, live, briefs),
    history,
  };
}

export function renderStatusHtml(view: StatusView): string {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>hyra-pi 现场</title>
  <style>
    :root { color-scheme: light; }
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #f4f3ef; color: #1b1b1b; }
    #app { max-width: 1440px; margin: 0 auto; padding: 16px 20px 40px; }
    .top, .lanes { display: grid; gap: 12px; }
    .top { grid-template-columns: minmax(0, 1.4fr) minmax(260px, 0.7fr); margin-bottom: 12px; }
    .lanes { grid-template-columns: 1fr 1fr; align-items: start; }
    .lane, .status-card { background: #ece9e2; padding: 8px; min-width: 0; }
    .status-card { background: #fff; border: 1px solid #ddd8ce; padding: 14px 16px; }
    .label { display: flex; justify-content: space-between; align-items: baseline; font-size: 13px; font-weight: 650; margin: 2px 6px 8px; }
    .count { color: #777; font-weight: 500; }
    h1 { font-size: 16px; margin: 0 0 10px; font-weight: 650; }
    .score { font-size: 28px; font-weight: 650; letter-spacing: -0.03em; }
    .meta { color: #555; font-size: 13px; line-height: 1.55; }
    .ok { color: #0a7; }
    .bad { color: #c40; }
    article { background: #fff; border: 1px solid #ddd8ce; padding: 10px 12px; margin: 0 0 8px; cursor: pointer; }
    article:last-child { margin-bottom: 0; }
    article.pending { background: #f7f5f0; }
    article:hover, article:focus { border-color: #b9b3a6; outline: none; }
    article h3, article .doing { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    article h3 { font-size: 14px; margin: 0 0 4px; font-weight: 600; }
    .who { font-size: 11px; color: #888; letter-spacing: 0.03em; margin-bottom: 4px; }
    .doing { font-size: 13px; color: #222; margin: 0; }
    .brief, .files, .log { font-size: 12px; color: #444; line-height: 1.5; white-space: pre-wrap; word-break: break-word; margin: 0 0 8px; }
    .files { color: #666; }
    .events { display: flex; flex-direction: column; gap: 8px; }
    .block { padding: 8px 10px; border: 1px solid #ece8df; }
    .block-label { font-size: 11px; color: #888; margin-bottom: 4px; }
    .block-text { font: 12px/1.5 ui-sans-serif, system-ui, sans-serif; white-space: pre-wrap; word-break: break-word; }
    .block.think { background: #f3f1eb; }
    .block.think .block-text { color: #5c564c; }
    .block.say { background: #f7f7f4; }
    .block.tool { background: #eef3f6; }
    .block.tool .block-text { color: #134e6a; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .block.result { background: #eef3ee; }
    .block.result .block-text { color: #3d4a3d; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .block.error { background: #f7eeee; }
    .block.error .block-text { color: #a32; }
    dialog { width: min(760px, 92vw); max-height: 86vh; border: 1px solid #ddd8ce; padding: 0; background: #fff; overflow: hidden; overscroll-behavior: none; }
    dialog[data-layout="proposal"], dialog[data-layout="sandbox"] { width: min(1120px, 96vw); height: min(86vh, 920px); }
    dialog[open] { display: flex; flex-direction: column; }
    dialog::backdrop { background: #0003; }
    .sheet-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; flex: 0 0 auto; padding: 12px 16px; border-bottom: 1px solid #ece8df; background: #fff; }
    .sheet-head h2 { font-size: 15px; margin: 0; font-weight: 650; line-height: 1.4; }
    .sheet-close { border: 0; background: transparent; color: #555; width: 28px; height: 28px; padding: 0; flex: 0 0 auto; font: 20px/1 ui-sans-serif, system-ui, sans-serif; cursor: pointer; }
    .sheet-body { padding: 16px; overflow: auto; min-height: 0; flex: 1 1 auto; overscroll-behavior: contain; }
    .sheet-body:has(.sheet-split), .sheet-body:has(.sheet-sandbox) { overflow: hidden; display: flex; }
    .sheet-split, .sheet-sandbox { flex: 1 1 auto; min-height: 0; width: 100%; }
    .sheet-split { display: grid; grid-template-columns: 2fr 3fr; gap: 12px; }
    .sheet-sandbox { display: flex; flex-direction: column; gap: 12px; }
    .sheet-sandbox .inspire { flex: 0 1 auto; max-height: 30%; }
    .sheet-sandbox .streams { flex: 1 1 auto; min-height: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .pane { min-width: 0; min-height: 0; overflow: auto; padding: 10px 12px; border: 1px solid #ddd8ce; background: #f7f5f0; }
    .pane-label { font-size: 11px; color: #888; margin-bottom: 6px; }
    .pane .brief, .pane .files, .pane .log { margin-bottom: 0; }
    .pane.trail { background: #faf9f6; }
    .pane.stdout { background: #f3f6f3; }
    .pane.stderr { background: #f7f1f1; }
    .pane pre { margin: 0; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; word-break: break-word; }
    .pane .empty { margin: 0; }
    tr[data-open] { cursor: pointer; }
    tr[data-open]:hover { background: #f7f5f0; }
    .empty { color: #888; font-size: 13px; margin: 8px; }
    .history { margin-top: 20px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }
    td, th { border-bottom: 1px solid #ddd; text-align: left; padding: 7px 8px; vertical-align: top; }
    th[data-sort] { cursor: pointer; user-select: none; white-space: nowrap; }
    th[data-sort]:hover, th[data-sort]:focus { background: #f7f5f0; outline: none; }
    th[data-sort][aria-sort="ascending"]::after { content: " ↑"; color: #888; font-weight: 500; }
    th[data-sort][aria-sort="descending"]::after { content: " ↓"; color: #888; font-weight: 500; }
    .sub { display: block; color: #666; font-size: 12px; margin-top: 3px; white-space: pre-wrap; }
    @media (max-width: 900px) {
      .top, .lanes, .sheet-split, .sheet-sandbox .streams { grid-template-columns: 1fr; }
      dialog[data-layout="proposal"], dialog[data-layout="sandbox"] { height: auto; max-height: 86vh; }
    }
  </style>
</head>
<body>
  <div id="app">${renderStatusPanel(view)}</div>
  <dialog id="sheet">
    <div class="sheet-head">
      <h2 id="sheet-title"></h2>
      <button type="button" class="sheet-close" id="sheet-close" aria-label="关闭">×</button>
    </div>
    <div class="sheet-body" id="sheet-body"></div>
  </dialog>
  <script>
    let openId = null;
    let historySort = { key: "inspiration", dir: "asc" };
    const sheet = document.getElementById("sheet");
    const paneScroll = {};
    function nearBottom(el) {
      return el.scrollHeight - el.clientHeight - el.scrollTop < 48;
    }
    function rememberPanes() {
      document.querySelectorAll("#sheet-body [data-pane]").forEach((el) => {
        paneScroll[el.getAttribute("data-pane")] = { top: el.scrollTop, stick: nearBottom(el) };
      });
      const body = document.getElementById("sheet-body");
      if (body) paneScroll.body = { top: body.scrollTop, stick: nearBottom(body) };
    }
    function restorePanes(first) {
      const body = document.getElementById("sheet-body");
      if (!body) return;
      const panes = body.querySelectorAll("[data-pane]");
      if (panes.length) {
        panes.forEach((el) => {
          const saved = paneScroll[el.getAttribute("data-pane")];
          const follow = el.hasAttribute("data-follow");
          if (follow && (first || (saved && saved.stick))) el.scrollTop = el.scrollHeight;
          else if (saved) el.scrollTop = saved.top;
        });
        return;
      }
      const saved = paneScroll.body;
      if (first || (saved && saved.stick)) body.scrollTop = body.scrollHeight;
      else if (saved) body.scrollTop = saved.top;
    }
    function fillSheet(first) {
      if (!openId) return;
      const src = document.querySelector('[data-detail="' + openId + '"]');
      if (!src) {
        sheet.close();
        openId = null;
        return;
      }
      document.getElementById("sheet-title").textContent = src.getAttribute("data-title") || "";
      sheet.dataset.layout = src.getAttribute("data-layout") || "plain";
      const body = document.getElementById("sheet-body");
      const next = src.innerHTML;
      if (!first && body.innerHTML === next) return;
      if (!first) rememberPanes();
      body.innerHTML = next;
      restorePanes(!!first);
    }
    function openSheet(id) {
      openId = id;
      for (const key of Object.keys(paneScroll)) delete paneScroll[key];
      fillSheet(true);
      if (!sheet.open) sheet.showModal();
    }
    function closeSheet() {
      openId = null;
      if (sheet.open) sheet.close();
    }
    document.getElementById("sheet-close").onclick = closeSheet;
    sheet.addEventListener("close", () => { openId = null; });
    sheet.addEventListener("click", (e) => { if (e.target === sheet) closeSheet(); });
    function historyValue(row, key) {
      if (key === "score") {
        const n = Number(row.getAttribute("data-score"));
        return Number.isFinite(n) ? n : null;
      }
      if (key === "result") return Number(row.getAttribute("data-result"));
      return row.getAttribute("data-" + key) || "";
    }
    function applyHistorySort() {
      const tbody = document.querySelector(".history tbody");
      if (!tbody) return;
      const { key, dir } = historySort;
      const sign = dir === "desc" ? -1 : 1;
      [...tbody.querySelectorAll("tr")].sort((a, b) => {
        const av = historyValue(a, key);
        const bv = historyValue(b, key);
        if (av === null && bv === null) return 0;
        if (av === null) return 1;
        if (bv === null) return -1;
        if (typeof av === "number" && typeof bv === "number") return (av - bv) * sign;
        return String(av).localeCompare(String(bv), "en", { numeric: true }) * sign;
      }).forEach((row) => tbody.appendChild(row));
      document.querySelectorAll(".history th[data-sort]").forEach((th) => {
        const on = th.getAttribute("data-sort") === key;
        th.setAttribute("aria-sort", on ? (dir === "desc" ? "descending" : "ascending") : "none");
      });
    }
    function toggleHistorySort(key) {
      if (historySort.key === key) historySort.dir = historySort.dir === "asc" ? "desc" : "asc";
      else historySort = { key, dir: key === "score" ? "desc" : "asc" };
      applyHistorySort();
    }
    document.getElementById("app").addEventListener("click", (e) => {
      const head = e.target.closest("th[data-sort]");
      if (head) {
        toggleHistorySort(head.getAttribute("data-sort"));
        return;
      }
      const card = e.target.closest("[data-open]");
      if (card) openSheet(card.getAttribute("data-open"));
    });
    document.getElementById("app").addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      const head = e.target.closest("th[data-sort]");
      if (head) {
        e.preventDefault();
        toggleHistorySort(head.getAttribute("data-sort"));
        return;
      }
      const card = e.target.closest("[data-open]");
      if (!card) return;
      e.preventDefault();
      openSheet(card.getAttribute("data-open"));
    });
    async function tick() {
      const html = await fetch("/panel").then((r) => r.text());
      const y = window.scrollY;
      document.getElementById("app").innerHTML = html;
      window.scrollTo(0, y);
      applyHistorySort();
      if (openId && sheet.open) fillSheet(false);
    }
    setInterval(tick, 1000);
  </script>
</body>
</html>`;
}

export function renderStatusPanel(view: StatusView): string {
  const best = view.best ? formatScore(view.best.score) : "还没有";
  const health = view.healthy ? "健康" : `不健康${view.reason ? ` — ${esc(view.reason)}` : ""}`;
  const contextTrace = view.traces.find((row) => row.actor === "context");
  const proposals = view.traces.filter((row) => row.actor === "proposal").sort(byInspirationId);
  const sandboxes = view.traces.filter((row) => row.actor === "sandbox").sort(byInspirationId);
  const historyRows = view.history
    .map((row) => {
      const title = row.direction ? esc(row.direction) : esc(row.inspirationId);
      const score = row.score === null ? "" : String(row.score);
      return `<tr data-open="hist-${esc(row.id)}" data-inspiration="${esc(row.inspirationId)}" data-experiment="${esc(row.id)}" data-result="${row.ok ? "1" : "0"}" data-score="${esc(score)}" role="button" tabindex="0">
        <td><code>${esc(row.inspirationId)}</code></td>
        <td><code>${esc(row.id)}</code></td>
        <td>${title}</td>
        <td class="${row.ok ? "ok" : "bad"}">${row.ok ? "ok" : "fail"}</td>
        <td>${row.score === null ? "" : formatScore(row.score)}</td>
      </tr>`;
    })
    .join("");

  return `
  <div class="top">
    <section class="lane">
      <div class="label"><span>Context</span></div>
      ${contextTrace ? renderCard(contextTrace) : `<article><p class="doing">还没开始整理</p></article>`}
    </section>
    <section class="status-card">
      <h1>状态</h1>
      <div class="score">${esc(best)}</div>
      <p class="meta">最好成绩${view.best ? ` · ${esc(view.best.inspirationId)} · 实验 ${esc(view.best.id)}` : ""}</p>
      <div class="meta">
        <div>阶段：${esc(view.phase)}${view.stopReason ? ` · ${esc(view.stopReason)}` : ""}</div>
        <div class="${view.healthy ? "ok" : "bad"}">${esc(health)}</div>
        <div>${
          view.budget.unlimited
            ? "预算：不限份数"
            : `预算：还能再评 ${view.budget.remainingSolutions} / ${view.budget.maxSolutions} 份`
        }</div>
        <div>已跑 ${esc(formatElapsed(view.elapsedMs))}</div>
        <div>${view.context.running ? "Context 正在整理" : `Context 空闲${view.context.lastContextAt ? `（上次 ${esc(new Date(view.context.lastContextAt).toLocaleString())}）` : ""}`}</div>
      </div>
    </section>
  </div>
  <div class="lanes">
    <section class="lane">
      <div class="label"><span>Proposal</span><span class="count">${proposals.length}</span></div>
      ${renderLane(proposals, "没有人在写")}
    </section>
    <section class="lane">
      <div class="label"><span>沙盒</span><span class="count">${sandboxes.length}</span></div>
      ${renderLane(sandboxes, "沙盒空闲")}
    </section>
  </div>
  <section class="history">
    <div class="label"><span>历史结果</span><span class="count">${view.history.length}</span></div>
    <table>
      <thead><tr>
        <th data-sort="inspiration" tabindex="0" aria-sort="ascending">灵感 ID</th>
        <th data-sort="experiment" tabindex="0" aria-sort="none">实验 ID</th>
        <th>实验</th>
        <th data-sort="result" tabindex="0" aria-sort="none">结果</th>
        <th data-sort="score" tabindex="0" aria-sort="none">分数</th>
      </tr></thead>
      <tbody>${historyRows}</tbody>
    </table>
  </section>
  <div hidden id="details">${[...view.traces, ...view.history].map((row) =>
    "inspirationId" in row ? renderHistoryDetail(row) : renderDetail(row),
  ).join("")}</div>`;
}

function renderLane(traces: AgentTrace[], empty: string): string {
  if (traces.length === 0) return `<p class="empty">${esc(empty)}</p>`;
  return traces.map(renderCard).join("");
}

async function buildTraces(
  runDir: string,
  live: LiveState,
  briefs: Map<string, QueueRow>,
): Promise<AgentTrace[]> {
  const traces: AgentTrace[] = [];
  const contextEvents = await readActivity(runDir, "context", 200);
  traces.push({
    id: "context",
    actor: "context",
    title: live.contextRunning ? "正在整理经验" : "空闲",
    doing: headline(contextEvents) || (live.contextRunning ? "读经验库、出实验" : "等待下一轮补货"),
    events: contextEvents,
  });

  const busy = new Set([...(live.writers ?? []), ...(live.sandboxes ?? [])].map((row) => row.id));
  for (const writer of live.writers ?? []) {
    const events = await readActivity(runDir, writer.id, 200);
    const files = await listWorkspace(path.join(runDir, "workspaces", writer.id));
    traces.push({
      id: writer.id,
      actor: "proposal",
      title: writer.direction,
      direction: writer.direction,
      brief: briefs.get(writer.id)?.context,
      doing: headline(events) || (files.latest ? `在写 ${files.latest}` : "在写方案"),
      files: files.names,
      events,
    });
  }

  for (const waiting of [...briefs.values()].filter((row) => !busy.has(row.id) && (row.state === "waiting" || row.state === undefined))) {
    traces.push({
      id: waiting.id,
      actor: "proposal",
      title: waiting.direction,
      direction: waiting.direction,
      brief: waiting.context,
      doing: "排队等待领取",
      events: [],
      pending: true,
    });
  }

  for (const box of live.sandboxes ?? []) {
    const workspace = path.join(runDir, "workspaces", box.id);
    const streams = await readSandboxStreams(workspace);
    traces.push({
      id: box.id,
      actor: "sandbox",
      title: box.direction,
      direction: box.direction,
      brief: briefs.get(box.id)?.context,
      doing: lastLine(streams.stdout) || lastLine(streams.stderr) || lastLine(streams.mixed) || "沙盒评分中",
      logTail: streams.mixed,
      stdoutTail: streams.stdout,
      stderrTail: streams.stderr,
      events: [],
    });
  }
  const context = traces.filter((row) => row.actor === "context");
  const proposals = traces.filter((row) => row.actor === "proposal").sort(byInspirationId);
  const boxes = traces.filter((row) => row.actor === "sandbox").sort(byInspirationId);
  return [...context, ...proposals, ...boxes];
}

function cardWho(trace: AgentTrace): string {
  if (trace.actor === "context") return "Context";
  if (trace.pending) return `等待 · ${esc(trace.id)}`;
  if (trace.actor === "proposal") return `正在写 · ${esc(trace.id)}`;
  return `评分中 · ${esc(trace.id)}`;
}

function renderCard(trace: AgentTrace): string {
  const pending = trace.pending ? ` class="pending"` : "";
  return `<article${pending} data-open="${esc(trace.id)}" role="button" tabindex="0">
    <div class="who">${cardWho(trace)}</div>
    <h3>${esc(trace.title)}</h3>
    <p class="doing">${esc(trace.doing)}</p>
  </article>`;
}

function renderDetail(trace: AgentTrace): string {
  if (trace.actor === "proposal") return renderProposalDetail(trace);
  if (trace.actor === "sandbox") return renderSandboxDetail(trace);
  return renderPlainDetail(trace);
}

function renderProposalDetail(trace: AgentTrace): string {
  const trail = renderEvents(trace);
  return `<div data-detail="${esc(trace.id)}" data-title="${esc(trace.title)}" data-layout="proposal">
    <div class="sheet-split">
      <div class="pane inspire" data-pane="inspire">${renderInspiration(trace)}</div>
      <div class="pane trail" data-pane="trail" data-follow><div class="pane-label">轨迹</div>${trail || `<p class="empty">还没有轨迹</p>`}</div>
    </div>
  </div>`;
}

function renderSandboxDetail(trace: AgentTrace): string {
  return renderSandboxSheet({
    detailId: trace.id,
    title: trace.title,
    brief: trace.brief,
    extra: trace.files && trace.files.length > 0 ? `文件：${trace.files.join(" · ")}` : undefined,
    stdout: trace.stdoutTail,
    stderr: trace.stderrTail,
  });
}

function renderPlainDetail(trace: AgentTrace): string {
  const events = renderEvents(trace);
  return `<div data-detail="${esc(trace.id)}" data-title="${esc(trace.title)}" data-layout="plain">${events}</div>`;
}

function renderInspiration(trace: AgentTrace): string {
  const brief = trace.brief ? `<p class="brief">${esc(trace.brief)}</p>` : `<p class="empty">没有说明书</p>`;
  const files =
    trace.files && trace.files.length > 0 ? `<p class="files">文件：${esc(trace.files.join(" · "))}</p>` : "";
  return `<div class="pane-label">灵感</div>${brief}${files}`;
}

function renderEvents(trace: AgentTrace): string {
  if (trace.events.length === 0) return "";
  return `<div class="events" data-id="${esc(trace.id)}">${groupActivityEvents(trace.events)
    .map(
      (row) =>
        `<div class="block ${esc(row.kind)}"><div class="block-label">${esc(kindLabel(row.kind))}</div><div class="block-text">${esc(clip(row.text, 8000))}</div></div>`,
    )
    .join("")}</div>`;
}

function renderStream(text: string | undefined): string {
  return text ? `<pre>${esc(text)}</pre>` : `<p class="empty">暂无</p>`;
}

function renderHistoryDetail(row: HistoryRow): string {
  return renderSandboxSheet({
    detailId: `hist-${row.id}`,
    title: row.direction ?? row.inspirationId,
    brief: row.brief,
    extra: row.notes,
    stdout: row.stdoutTail,
    stderr: row.stderrTail,
  });
}

function renderSandboxSheet(opts: {
  detailId: string;
  title: string;
  brief?: string;
  extra?: string;
  stdout?: string;
  stderr?: string;
}): string {
  const brief = opts.brief ? `<p class="brief">${esc(opts.brief)}</p>` : `<p class="empty">没有说明书</p>`;
  const extra = opts.extra ? `<p class="files">${esc(opts.extra)}</p>` : "";
  return `<div data-detail="${esc(opts.detailId)}" data-title="${esc(opts.title)}" data-layout="sandbox">
    <div class="sheet-sandbox">
      <div class="pane inspire" data-pane="inspire"><div class="pane-label">灵感</div>${brief}${extra}</div>
      <div class="streams">
        <div class="pane stdout" data-pane="stdout" data-follow><div class="pane-label">STDOUT</div>${renderStream(opts.stdout)}</div>
        <div class="pane stderr" data-pane="stderr" data-follow><div class="pane-label">STDERR</div>${renderStream(opts.stderr)}</div>
      </div>
    </div>
  </div>`;
}

function headline(events: ActivityEvent[]): string {
  const blocks = groupActivityEvents(events);
  const last = blocks[blocks.length - 1];
  return last ? clip(last.text.replace(/\s+/g, " ").trim(), 160) : "";
}

function kindLabel(kind: ActivityEvent["kind"]): string {
  if (kind === "think") return "思考";
  if (kind === "say") return "说";
  if (kind === "tool") return "做";
  if (kind === "result") return "回";
  return "错";
}

async function readQueue(runDir: string): Promise<QueueRow[]> {
  const dir = path.join(runDir, "queue");
  let names: string[] = [];
  try {
    names = await readdir(dir);
  } catch {
    return [];
  }
  const items: QueueRow[] = [];
  for (const name of names) {
    if (!name.endsWith(".json")) continue;
    try {
      items.push(await readJsonFile<QueueRow>(path.join(dir, name)));
    } catch {
      continue;
    }
  }
  return items;
}

async function listWorkspace(dir: string): Promise<{ names: string[]; latest?: string }> {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return { names: [] };
  }
  const files: Array<{ name: string; mtime: number }> = [];
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    if (entry.name === "eval.log" || entry.name === "eval.stdout" || entry.name === "eval.stderr" || entry.name.startsWith(".")) continue;
    const info = await stat(path.join(dir, entry.name));
    files.push({ name: entry.name, mtime: info.mtimeMs });
  }
  files.sort((a, b) => b.mtime - a.mtime);
  return { names: files.map((row) => row.name).slice(0, 12), latest: files[0]?.name };
}

async function readHistoryStreams(row: { solutionDir: string; logPath: string }): Promise<{
  stdout: string;
  stderr: string;
  mixed: string;
}> {
  const fromSolution = await readSandboxStreams(row.solutionDir);
  if (fromSolution.stdout || fromSolution.stderr || fromSolution.mixed) return fromSolution;
  const mixed = await tailFile(row.logPath, 8000);
  const split = classifyMixedLog(mixed);
  return { stdout: split.stdout, stderr: split.stderr, mixed };
}

async function readSandboxStreams(workspace: string): Promise<{ stdout: string; stderr: string; mixed: string }> {
  const [stdoutFile, stderrFile, mixed] = await Promise.all([
    tailFile(path.join(workspace, "eval.stdout"), 8000),
    tailFile(path.join(workspace, "eval.stderr"), 8000),
    tailFile(path.join(workspace, "eval.log"), 8000),
  ]);
  if (stdoutFile || stderrFile) return { stdout: stdoutFile, stderr: stderrFile, mixed };
  const split = classifyMixedLog(mixed);
  return { stdout: split.stdout, stderr: split.stderr, mixed };
}

export function classifyMixedLog(text: string): { stdout: string; stderr: string } {
  const out: string[] = [];
  const err: string[] = [];
  let prev: "out" | "err" = "out";
  for (const line of text.split("\n")) {
    const stream = classifyLogLine(line, prev);
    (stream === "err" ? err : out).push(line);
    if (line.trim()) prev = stream;
  }
  return { stdout: out.join("\n").trim(), stderr: err.join("\n").trim() };
}

function classifyLogLine(line: string, prev: "out" | "err"): "out" | "err" {
  if (/site-packages|\bWarning\b|_log_warning|Traceback|\[LightGBM\]|\[XGBoost\]|\[catboost\]/.test(line)) {
    return "err";
  }
  if (prev === "err" && /Please use |has been found|will be ignored/.test(line)) return "err";
  if (
    prev === "err" &&
    /^\s+/.test(line) &&
    !/^\s+(fold |\[|\d|AUC|auc=|OOF|Early |Rows:|Positive)/.test(line)
  ) {
    return "err";
  }
  return "out";
}

async function tailFile(file: string, maxChars: number): Promise<string> {
  try {
    const text = await readFile(file, "utf8");
    return text.length <= maxChars ? text.trim() : text.slice(-maxChars).trim();
  } catch {
    return "";
  }
}

function lastLine(text: string | undefined): string {
  if (!text) return "";
  const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
  return lines[lines.length - 1] ?? "";
}

function formatScore(score: number): string {
  return Number.isInteger(score) ? String(score) : score.toPrecision(7);
}

function formatElapsed(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} 分 ${seconds % 60} 秒`;
  return `${Math.floor(minutes / 60)} 时 ${minutes % 60} 分`;
}

function byInspirationId(a: { id?: string; inspirationId?: string }, b: { id?: string; inspirationId?: string }): number {
  const left = a.inspirationId ?? a.id ?? "";
  const right = b.inspirationId ?? b.id ?? "";
  return left.localeCompare(right, "en", { numeric: true });
}

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max)}…`;
}

function esc(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
