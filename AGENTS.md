# AGENTS

A 股日频量化系统：Agent 挖因子 + AutoML 合成信号 + 凸优化组合 → 每日目标持仓人工跟单。fork 自 `001eander/hyra-pi`，单仓库双语言：

- `src/`、`prompts/`：TS 编排层（内循环、调度）
- `quant/`：Python 执行层（数据、因子、回测、组合、跑批）
- `tasks/`、`docker/`：Agent 任务与沙盒镜像
- `factor_library/`：因子库，入 git
- `data/`、`runs/`：本地数据与运行记录，不入 git

## 技术栈

Python：polars（禁用 pandas）、cvxpy、AutoGluon（GPU 加速，本机有显卡）、akshare（锁版本）、pytest、uv。
TS：Node ≥ 20、vitest、strict。

## 代码

- Python：全量类型标注；polars 向量化，禁逐行循环；配置集中模块顶部；pytest 覆盖规则分支（回测器的 T+1/涨跌停/整手/停牌/费用必测）。
- TS：沿用 hyra-pi 风格；只做调度与 prompt，量化逻辑全在 Python。

## Git

远程 SSH：`git@github.com:001eander/ASTER.git`。提交信息中文，一句话说清改动与原因。单测全绿才合入。

## 语言风格

不用「不是……而是……」对举句式；不用三段排比；中文交流。

## Agent skills

### Issue tracker

Issue 与 spec 存 GitHub Issues，用 `gh` CLI 操作。见 `docs/agents/issue-tracker.md`。

### Triage labels

每个 issue 挂一个类别（`bug`/`enhancement`）加一个状态（`needs-triage`/`needs-info`/`ready-for-agent`/`ready-for-human`/`wontfix`），模块 label（`data`/`factor`/`backtest`/`portfolio`/`automl`/`agent`/`infra`）另计。见 `docs/agents/triage-labels.md`。

### Domain docs

单上下文：根 `GLOSSARY.md` + `docs/adr/`，有术语或决策落地时才创建。见 `docs/agents/domain.md`。
