# 课题体系：任务单元、双通道提出与并行语义

Status: accepted（2026-10-16，issue #130，决策地图 #128）

新因子挖掘系统的任务单元定为课题，取代 hyra-pi 按单任务触发的层级流水线。整体结构参考 QuantaAlpha（planning 一次 LLM 调用产方向、每方向一个带 trace 的 loop、确定性 controller 轮转），按源码精读笔记 `docs/research/quantaalpha-source-notes.md`（issue #129）只取结构不取实现：其论文宣传的变异定位与片段交叉在源码中不存在，轨迹池只写不读，方向互补全靠 prompt。

## 决定

**两层结构**：课题是灵感的上位容器。课题持有方向边界（信号源加时间尺度，可给范围）、禁区、验收标准（含排序键）、股票池（默认 zz1000，可覆盖），四项必填；假设与构造思路选填。灵感是课题内的一次尝试（具体假设加构造思路）。课题载体为 `topics/<课题id>.md`（frontmatter 带状态、方向边界、预算、验收、排序键），入 git 为唯一真源；issue 只作讨论入口。排序键为字典序字段列表，默认 `[rank_ic desc, icir desc, max_corr asc]`，课题可覆盖。

**课题内循环**：每次尝试走单 loop 五步（propose → construct → calculate → backtest → feedback），带 trace，feedback 产出的 New Hypothesis 作为下一轮 propose 输入。现有 Context/Proposal 双 agent 的 prompt 资产拆解进对应步骤；双模型分档降级为步骤级配置（propose、feedback 走强模型，construct 走快模型）。节拍为轮次制：一轮一次 propose 产 3 条假设，后续步骤并行，共用课题 trace；调度器在课题间以一轮为轮转单位。

**课题层管理为纯代码**：建题、结题判定、休眠复活、优先级轮转全部在编排层确定性代码，不新增 LLM 角色（QuantaAlpha 的 controller 同样如此）。结题三条件任一触发：入库数达验收线（完成，默认 10 个低相关 pool 因子，可覆盖）；连续 3 轮无过门控产出（枯竭，转休眠，可复活）；30 轮硬上限（熔断）。预算按课题隔离（token 与沙盒时长独立账户，字段必填，默认值由首个试跑课题校准）；活跃课题并发上限 3；优先级为人工整数。

**方向错开走 prompt 软约束**：废除灵感级方向硬配额与 deferred 机制，课题激活不做重叠校验。direction-map 渲染升级为「活跃课题方向占用图 + 因子库密度 + 各课题近期被拒率」，注入 planning 与 propose 的 prompt。结题摘要附方向分布回顾，供人观测错开是否生效。

**双通道提出**：两个通道共用同一简报 schema。人工课题直接激活；agent 从库空隙提出的课题进待批，人工一键批准，自动激活开关默认关。空隙扫描在结题后、每日跑批后、活跃课题数低于下限时触发。研报复现类课题由人工提炼结构化要点填简报，PDF 仅作附件引用，不自动解析。经验共享：trace 与评估明细课题内共享；跨课题只共享入库因子与结题摘要。

## Considered Options

- 方向错开硬校验：初案为课题激活时对方向边界重叠做硬拒绝（灵感级配额保留）。用户判定硬约束太强会导致挖不到东西，改为纯 prompt 软约束，灵感级配额随之退役。代价是错开失效时没有兜底，只靠结题回顾事后观测。
- 课题层 Planner LLM 角色：QuantaAlpha 论文有规划叙事，但其源码的 controller 是纯代码状态机，planning 只在初始化调一次 LLM。结题、轮转这类规则性判断交 LLM 可控性差，故维持编排层代码。
- 排序键加权和：会把 ADR 0001 刚废除的单一 score 换形式请回，且权重本身成为新的调参面。
- 课题载体用 GitHub issue：调度器每轮读课题状态会把外网依赖引进内循环热路径，状态迁移用 git diff 追踪更干净。
- 课题内单线串行（纯 QuantaAlpha 式，并行全靠课题间）：课题并发上限仅 3，课题内再无批量则吞吐过低。

## Consequences

- `src/inspiration-queue.ts` 的方向配额与 deferred 机制退役；`src/direction-map.ts` 渲染内容升级；`prompts/context.md` 与 `prompts/proposal.md` 拆解进 propose / construct / feedback 步骤 prompt。
- ADR 0001 留下的「排序键交课题层」在本票落地为字典序字段列表。
- 变异、交叉与轨迹结构归 issue #131 辩；语言选择倾向（参考 QuantaAlpha 的 Python 一体化编排）作为输入记入 issue #138，本票不做决定。
- 首个试跑课题的选定与验收标准另出实现票（地图 #128 的 Not yet specified）。
