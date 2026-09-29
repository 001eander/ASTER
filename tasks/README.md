# tasks — 内循环任务目录

每个子目录是一个任务包，含 `TASK.md`（题目正文，Context 与 Proposal 两个角色都会读到）与 `eval.sh`（沙盒评分入口）。

内循环与沙盒的执行约定见 [`../src/inner-loop.ts`](../src/inner-loop.ts) 与 [`../src/sandbox.ts`](../src/sandbox.ts)。
