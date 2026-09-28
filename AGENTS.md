# AGENTS

架构与决策参照：`C:\Users\陶唐\.opencode\plan\aster-architecture.md`。

A 股日频量化系统：Agent 挖因子 + AutoML 合成信号 + 凸优化组合 → 每日目标持仓人工跟单。fork 自 `001eander/hyra-pi`，单仓库双语言：

- `src/`、`prompts/`：TS 编排层（内循环、调度）
- `quant/`：Python 执行层（数据、因子、回测、组合、跑批）
- `tasks/`、`docker/`：Agent 任务与沙盒镜像
- `factor_library/`：因子库，入 git
- `data/`、`runs/`：本地数据与运行记录，不入 git

## 技术栈

Python：polars（禁用 pandas）、cvxpy、LightGBM、akshare（锁版本）、pytest、uv。
TS：Node ≥ 20、vitest、strict。
红线：不引入 Qlib；数据源只用 akshare。

## 流程

- Issue 驱动（M0–M4 共 37 个），动手前把 issue 移到 In Progress。
- 分支命名 `issue号-简述`，一个 PR 关一个 issue，描述写 `Closes #N`。
- Milestone 按 DoD 验收后打 tag（`m0`、`m1`…）。
- M0→M3 严格按序，M4 可与 M3 并行。
- 例外：`prompts/`、`factor_library/` 日常变更直接提 main。

## 量化纪律（违反即 bug）

1. 无前视：cutoff = T 日收盘，成交 = T+1 开盘价 + 滑点，标签 = T+1 开盘 → T+2 开盘收益。
2. 标签只在 `quant/labels/` 定义；IC/RankIC/ICIR/分层只在 `quant/eval/metrics.py` 实现。
3. 因子是一个 `.py` 实现 `compute(data: pl.DataFrame) -> pl.DataFrame`，禁 IO、禁随机、禁 cutoff 后数据。
4. 奖励信号一致：入库规则必须体现在 score.json，不允许「分数高但被拒」的隐藏规则。
5. 账户模拟器同时服务回测与跟单虚拟账户，一套逻辑。
6. 跑批先过 `quant/data/validate.py`，不过即中止。

## 代码

- Python：全量类型标注；polars 向量化，禁逐行循环；配置集中模块顶部；pytest 覆盖规则分支（回测器的 T+1/涨跌停/整手/停牌/费用必测）。
- TS：沿用 hyra-pi 风格；只做调度与 prompt，量化逻辑全在 Python。

## Git

远程 SSH：`git@github.com:001eander/ASTER.git`。提交信息中文，一句话说清改动与原因。单测全绿才合入。

## 语言风格

不用「不是……而是……」对举句式；不用三段排比；中文交流。
