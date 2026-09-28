# AGENTS — ASTER 开发规范

> 本文件面向在本仓库工作的所有人与 AI agent。开工前必读。
> 架构与决策的唯一权威参照：`C:\Users\陶唐\.opencode\plan\aster-architecture.md`（本地规划文档）。

## 项目是什么

ASTER 是个人量化私募系统：A 股日频，Agent 内循环挖因子 + AutoML 合成信号 + 凸优化组合，产出每日目标持仓供人工跟单。仓库 fork 自 `001eander/hyra-pi`，单仓库双语言：

- `src/`、`prompts/`：TypeScript 编排层（hyra-pi 血统，内循环、经验库、灵感队列、沙盒调度）
- `quant/`：Python 执行层（数据、因子、回测、组合、跑批）
- `tasks/`、`docker/`：Agent 任务定义与沙盒镜像
- `factor_library/`：因子库（代码 + 元信息，**纳入 git**）
- `data/`、`runs/`：本地数据与运行记录（**不入 git**）

## 技术栈与工具

| 层 | 栈 |
|---|---|
| Python | polars（一律不用 pandas）、cvxpy、LightGBM、akshare（锁版本）、pytest |
| TypeScript | Node ≥ 20、vitest、strict mode |
| 包管理 | Python 一律用 `uv` / `uvx` |
| 依赖红线 | **不引入 Qlib**（运行时与测试均不依赖）；数据源只依赖 akshare，保持人人可用 |

## 开发流程

1. **Issue 驱动**。开发任务对应 GitHub issue（M0–M4 共 37 个，见 milestone）。动手前把 issue 移到 In Progress，确认验收标准。
2. **分支与 PR**。每个 issue 一个分支，命名 `issue号-简述`（如 `3-parquet-cache`）。一个 PR 关一个 issue，描述里写 `Closes #N`。
3. **Milestone 验收**。按 DoD 验收，通过后打 tag（`m0`、`m1`…）。
4. **例外**：`prompts/` 迭代、`factor_library/` 的日常入库与整库直接提 main，不走 issue。
5. **顺序约束**：M0 → M1 → M2 → M3 严格按序，M4 可与 M3 并行。不要提前动下游模块。

## 量化纪律（违反即 bug）

1. **无前视**。信号 cutoff = T 日收盘；成交假设 = T+1 开盘价 + 滑点；标签 = T+1 开盘 → T+2 开盘收益。三者对齐，禁止用 T 收盘 → T+1 收盘当标签。
2. **标签唯一口径**。`quant/labels/` 是全局唯一的标签定义处，任何模块不得自行构造收益标签。
3. **指标唯一实现**。IC / RankIC / ICIR / 分层单调性只在 `quant/eval/metrics.py` 实现，因子级与模型级评估共用。
4. **因子接口**。因子是一个 `.py` 文件，实现 `compute(data: pl.DataFrame) -> pl.DataFrame`；输入输出列、排序、缺失值约定见 `quant/factor_api/`。因子内禁止网络/文件 IO、禁止随机性、禁止使用 cutoff 之后的数据。
5. **奖励信号一致性**。任何影响因子入库的规则（门控、共线性折扣）必须体现在 score.json 的分数里，Agent 只能看到分数，不允许存在「分数高但被拒」的隐藏规则。
6. **回测与实盘同构**。账户模拟器（`quant/backtest/`）同时服务历史回测与每日跟单虚拟账户，不允许两边维护两套逻辑。
7. **数据校验熔断**。每日跑批先过 `quant/data/validate.py`，校验不过就中止报警，不带病跑。

## 代码规范

### Python

- 全量类型标注；公共函数写 docstring（说明输入输出与口径，不用逐行注释）。
- 数据处理用 polars 表达式，避免 Python 层逐行循环；日频截面计算必须向量化。
- 配置（费率、约束参数、窗口长度）集中放模块顶部的 dataclass 或常量，不散落魔法数字。
- 测试用 pytest，每个规则分支都要有用例（回测器的 T+1/涨跌停/整手/停牌/费用是强制覆盖项）。

### TypeScript

- 沿用 hyra-pi 既有风格：ESM、strict、vitest。
- 编排层不碰量化计算，只做调度、状态、prompt 组装；量化逻辑全部在 Python 侧。

## Git

- 远程一律 SSH：`git@github.com:001eander/ASTER.git`。
- 提交信息用中文，一句话说清改动与原因，如「broker: 涨跌停拒单改为严格版，避免乐观成交假设」。
- 单测全绿才能合入。

## 语言风格

所有输出（回复、文档、提交信息、代码注释）遵守：

- 不用「不是……而是……」「并非……而是……」这类对举句式。
- 不用三段式排比，能一句自然的话说完就一句说完。
- 与陶唐（001eander）用中文交流。
