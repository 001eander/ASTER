# ASTER 架构规划

> 个人量化私募项目：A 股日频，Agent 挖因子 + AutoML + 凸优化组合，半自动信号输出（每日目标持仓，人工跟单）。
>
> 本文档是架构与量化设计参照。定稿日期：2026-09-28。

## 1. 核心决策汇总

| 决策点 | 结论 |
|---|---|
| 仓库形态 | fork `001eander/hyra-pi` 为 ASTER 主仓，单仓库发展。TS 编排层 + Python 执行层 |
| Qlib | 完全移除，运行时与测试均不依赖 |
| 数据源 | akshare → Parquet 本地缓存。抽象 DataSource 协议，基础校验（完整性/异常值），不做对拍 |
| 因子表达 | Python 函数接口（每因子一个 `.py` 文件，实现 `compute(data)`），不用 DSL 表达式 |
| 前视防护 | 截断重算检测（行为级）+ 接口约定（数据 cutoff）|
| 共线性 | 四层防御：Context 源头分流 → Proposal 差异化 → 评分相关性折扣 → 入库门控与定期整库 |
| 回测 | 三层体系：因子级 / 模型级 / 账户级。统一 open-to-open 标签口径 |
| 成交假设 | T+1 开盘价 + 滑点（贴合人工跟单习惯） |
| 组合优化 | cvxpy 凸优化，自研 |
| 生产形态 | 账户模拟器兼任「虚拟账户」：每日产出可执行调仓单，人工跟单后回录实际成交 |

## 2. 架构总览

```
┌────────────────────────────────────────────────────────────┐
│ 研究层（离线，hyra-pi 内循环，连续跑）                        │
│                                                            │
│  Context(强模型) ──灵感(方向配额)──▶ N×Proposal(快模型)     │
│       ▲                              │ 因子 .py            │
│       │                              ▼                     │
│  因子库/经验库 ◀──score── Docker沙盒: eval.sh              │
│       ▲                        门控 + 三层指标 + 相关折扣    │
│       │                                                    │
│  共线性四层防御贯穿全程                                      │
└────────────────────────────────────────────────────────────┘
                   │ 产出：过门控的因子入因子库
                   ▼
┌────────────────────────────────────────────────────────────┐
│ 生产层（每日 T 日收盘后跑批）                                │
│                                                            │
│  数据增量更新 → 因子重算 → AutoML(AutoGluon, GPU) → 截面打分 │
│  → cvxpy 凸优化 → 虚拟账户生成 T+1 调仓单(已滤不可交易)      │
│  → 人工跟单 → 回录实际成交 → 简报(调仓/敞口/因子表现)        │
└────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│ 数据层                                                     │
│  akshare → 校验 → Parquet 缓存（首抓全量 + 每日增量）        │
│  预计算：涨跌停上下限、复权价、交易日历、分红送转             │
└────────────────────────────────────────────────────────────┘
```

研究层与生产层解耦，因子库是唯一接口。Agent 可以连跑几天挖因子，生产每日定时跑批只消费已入库因子。

## 3. 模块设计

### 3.1 数据层 `quant/data/`

```
quant/data/
├── source/
│   ├── base.py      # DataSource 协议：daily_bars / trade_calendar /
│   │                #   corporate_actions / instrument_info
│   └── akshare.py   # 主数据源实现
├── cache/           # Parquet 缓存：首次全量抓取（断点续传）+ 每日增量
├── validate.py      # 校验：完整性、异常值、复权口径一致性
└── schema.py        # 统一字段定义
```

- 上层只面向 `base.py` 协议编程，换数据源只加一个实现类。
- 行情表预计算 `limit_up` / `limit_down` 两列，broker 直接使用。涨跌停幅度按「(板块, 日期, ST状态)」规则表判定：主板 10%、创业板 20%（2020-08-24 起）、科创板 20%（开市起）、北交所 30%、ST 5%。
- akshare 已知风险与对策：接口可能变动（锁版本 + 校验报警 + 抽象层隔离）；退市股历史覆盖不全（第一版接受幸存者偏差，文档明示）；首抓慢（断点续传，一次性成本）。

### 3.2 标签 `quant/labels/`

全局唯一口径，三层回测共用：

```
信号生成：T 日收盘后（数据 cutoff = T 日收盘）
成交假设：T+1 开盘价 + 滑点
预测标签：T+1 开盘 → T+2 开盘收益（或 T+1 开盘起 N 日）
```

错用「T 收盘 → T+1 收盘」做标签会系统性虚高 IC。此模块是唯一定义处。

### 3.3 因子接口 `quant/factor_api/`

每个因子一个 `.py` 文件：

```python
import polars as pl

def compute(data: pl.DataFrame) -> pl.DataFrame:
    """
    输入：长表面板数据，列固定为
        date, instrument, open, high, low, close, vwap, volume, amount, adjfactor
        已按 (instrument, date) 排序；价格未复权，复权用 adjfactor 自行处理
    输出：date, instrument, value 三列长表，缺失用 null，不做截面标准化
    禁止：网络/文件 IO、随机性（必须可复现）、使用 cutoff 之后的数据
    """
```

配套检测：

- **截断重算检测**（防前视，必做）：抽样若干交易日 t，数据截断到 t 重算因子，对比全量计算在 t 日的值。不一致即零分。注意容差：检测点避开 rolling 窗口预热期。
- **复杂度度量**：圈复杂度（radon）+ AST 节点数 + 使用字段数，阈值卡控。
- **schema 校验**：输入输出列、索引、类型。

### 3.4 评估 `quant/eval/`

```
quant/eval/
├── metrics.py   # 共享指标内核：IC / RankIC / ICIR / 分层单调性 / 换手
├── factor.py    # 因子级评估（eval.sh 调用，性能敏感：数据常驻内存、向量化）
└── model.py     # 模型级评估：walk-forward 拼接纯样本外预测 + 因子贡献度
```

三层回测对照：

| | 因子回测 | 模型回测 | 组合回测 |
|---|---|---|---|
| 对象 | 单因子值 | AutoML 合成打分 | 端到端策略 |
| 指标 | RankIC/ICIR/分层/换手 | 同左 + 滚动稳定性 + 贡献度 | 净值/超额/回撤/夏普/费用占比 |
| 成本 | 无 | 无（换手惩罚近似） | 完整 |
| 速度 | 秒级（循环内万次级） | 分钟级（每次重训） | 分钟级（低频） |
| 触发 | 每个 Proposal | 每次模型训练 | 策略变更/定期复核 |

### 3.5 账户级回测 `quant/backtest/`

```
quant/backtest/
├── account.py     # 现金、持仓（volume / sellable 分离）、成本
├── broker.py      # 撮合：停牌、涨跌停、T+1、整手、成交价假设
├── fee.py         # 佣金(双边+最低5元)、印花税(卖出千0.5)、滑点
├── engine.py      # 主循环：每日推进
└── corporate.py   # 分红送转：除权日调持仓数量与成本
```

主循环时序：

```
T 日收盘：数据更新 → 因子打分 → 凸优化目标持仓
T+1 开盘前：目标 vs 当前持仓 → 调仓订单（按手取整）
T+1 撮合：停牌拒单 / 涨跌停拒单(严格版) / 现金不足缩单 / T+1冻结拒卖
T+1 日终：费用结算、昨日买入解冻、公司行为调整、记净值
```

A 股规则：T+1（当日买入次日解冻）；涨跌停分板块分时段；买入 100 股整数倍（科创板 200 起）、卖出允许零股出清；停牌不可交易按最后价估值。

**设计统一：账户模拟器兼任生产虚拟账户。** 回测模式由历史数据驱动批量推进；生产模式持久化虚拟账户（SQLite/JSON），每日产出可执行调仓单，人工跟单后回录实际成交。回测与实盘的滑点偏差因此可量化。

### 3.6 组合优化 `quant/portfolio/`

cvxpy 凸优化：

```
min  -αᵀw + λ·wᵀΣw + κ·|w − w_prev|₁
s.t. Σw = 1,  0 ≤ w ≤ w_max,
     |w − w_prev|₁ ≤ 双边换手上限
     （行业偏离约束可选，第一版不做）
```

- Σ 第一版用 Ledoit-Wolf 收缩协方差，后续可换结构化风险模型。
- 权重上限留 buffer（如目标 5% 则约束 4.8%），吸收股数取整带来的偏差。
- κ 内含印花税与佣金，抑制无效换手。

### 3.7 共线性四层防御（hyra-pi 定制核心）

| 层 | 位置 | 机制 |
|---|---|---|
| 1 源头分流 | `prompts/context.md` | 注入因子库方向分布地图（各方向数量/质量/近期被拒率），要求新灵感走低密度方向 |
| 2 生成避让 | `prompts/proposal.md` | 附同方向已入库因子摘要（假设+思路，不附代码），要求机制差异 |
| 3 评分折扣 | `quant/factor_eval` | `score = quality × (1 − max_corr_with_pool)` |
| 4 入库与整库 | `quant/factor_lib` | 行为相关性 > 0.7 拒入；定期整库（同簇留强）；容量上限 50% |

原则：**奖励信号一致性**——共线性惩罚必须体现在 score.json 里，Agent 才能学到「高相关 = 低分」。门控只是兜底。

hyra-pi 并行架构特有的补充：**inspiration-queue 方向配额**，同一方向并发灵感数设上限，避免 N 个 Proposal 一窝蜂挖同质方向。

因子库 schema（血统链）：

```json
{
  "factor_id": "f_20260928_001",
  "hypothesis": "缩量下跌后的放量反弹预示短期反转",
  "code_path": "factor_library/f_20260928_001.py",
  "metrics": {"rank_ic": 0.047, "icir": 0.38, "max_corr": 0.31},
  "direction": {"signal_source": "volume", "time_scale": "short", "mechanism": "reversal"},
  "lineage": {"op": "mutation", "parents": ["f_20260927_003"], "run_id": "r_09", "generation": 4},
  "status": "pool"
}
```

## 4. 目录结构

```
ASTER/
├── src/                     # [TS] 编排层，hyra-pi 血统
│   ├── inner-loop.ts
│   ├── experience-bank.ts   # 扩展：方向标签、相关性元数据、被拒原因
│   ├── inspiration-queue.ts # 扩展：方向配额
│   └── ...
├── prompts/
│   ├── context.md           # Context：方向分布地图 + 互补分流
│   └── proposal.md          # Proposal：接口规范 + 字段文档 + 示范因子 + 同方向避让
├── quant/                   # [Python] 执行层
│   ├── data/  labels/  factor_api/  eval/  backtest/
│   ├── factor_lib/  automl/  portfolio/  daily/
├── tasks/factor-mining/
│   ├── TASK.md
│   └── eval.sh              # 沙盒内入口 → quant.eval.factor → score.json
├── docker/                  # 沙盒镜像：Python + 依赖，数据只读挂载
├── factor_library/          # 因子代码 + 元信息（纳入 git）
├── data/                    # gitignore
└── runs/                    # gitignore
```

## 5. GitHub 开发进度管理

> **本节（5.1–5.5）自 2026-10-14 起废止。** 开发流程已切换至 mattpocock 工作流（grill-with-docs → to-spec → to-tickets → implement），label 体系已重建（类型与优先级 label 删除，保留模块 label），M0–M5 milestone 已全部关闭。以下内容仅作历史记录保留，不再维护。

### 5.1 Milestone 划分（对应实施阶段）

| Milestone | 名称 | 验收标准（DoD） |
|---|---|---|
| M0 | 数据管线 | akshare 全量+增量跑通，Parquet 缓存落地，校验通过，含涨跌停预计算 |
| M1 | 端到端基线（无 Agent） | 手工因子 + AutoGluon + 凸优化 + 账户回测 + 每日调仓单，全链路可跑 |
| M1.5 | 股票池与指数增强 | 可控股票池（沪深300/中证500/中证1000/自定义，PIT）+ 指数增强优化器 + 基准绩效与风险四表，zz1000 池指增端到端出合理超额曲线 |
| M2 | 因子挖掘内循环 | 沙盒镜像就绪，TASK.md/eval.sh 可用，一次 run 产出 ≥1 个过门控因子 |
| M3 | 进化机制与共线性 | 四层防御全部上线，多代进化后入库因子质量单调提升且相关性受控 |
| M4 | AutoML 增强 | walk-forward 滚动重训自动化，模型级评估报告产出，信号稳定优于基线 |
| M5 | AutoResearch（远期） | Agent 调参与网络结构演化循环跑通，参照 RD-Agent 协同演化 |

顺序依赖：M0 → M1 → M1.5 → M2 → M3；M4 与 M3 可部分并行，M5 依赖 M4。M1 刻意不用 Agent：先调通链路本身，否则 M2 出问题时分不清是循环的 bug 还是量化的 bug。M1.5 插在 M2 之前（2026-09-29 用户拍板）：策略层就绪后，M2 起因子评估直接在指定池内进行。三个口径决策：指数权重用中证官方月度权重 + 本地按日收益漂移；行业分类用东财行业；期货对冲（IM overlay）与 DRO 目标函数只记 backlog（#72、#73）。

### 5.2 Issue 拆分

**M0 数据管线**

| # | Issue | Label |
|---|---|---|
| 1 | DataSource 协议与统一 schema 定义 | data, feat, P0 |
| 2 | akshare 实现：日线/日历/复权/分红送转抓取 | data, feat, P0 |
| 3 | Parquet 缓存：全量首抓 + 断点续传 | data, feat, P0 |
| 4 | 每日增量更新脚本 | data, feat, P0 |
| 5 | 涨跌停上下限预计算（板块/时段/ST 规则表） | data, feat, P0 |
| 6 | 数据校验：完整性、异常值、复权一致性 | data, test, P1 |

**M1 端到端基线**

| # | Issue | Label |
|---|---|---|
| 7 | labels 模块：open-to-open 标签构造 | backtest, feat, P0 |
| 8 | eval/metrics.py：IC/RankIC/ICIR/分层指标内核 | factor, feat, P0 |
| 9 | backtest：account + fee（费用模型） | backtest, feat, P0 |
| 10 | backtest：broker（T+1/涨跌停/整手/停牌撮合） | backtest, feat, P0 |
| 11 | backtest：engine 主循环 + corporate 公司行为 | backtest, feat, P0 |
| 12 | portfolio：cvxpy 凸优化器 + 权重取整可行化 | portfolio, feat, P0 |
| 13 | 基线因子集：手工实现 Alpha158 风格子集 | factor, feat, P1 |
| 14 | automl：AutoGluon 基线训练管线（GPU） | automl, feat, P0 |
| 15 | daily：每日跑批 + 虚拟账户持久化 + 调仓单生成 | infra, feat, P0 |
| 16 | 端到端验证：全链路回测出合理净值曲线 | test, P0 |

**M1.5 股票池与指数增强**

| # | Issue | Label |
|---|---|---|
| 64 | 指数成分与权重历史数据落地（PIT，官方月度权重+本地漂移） | data, feat, P0 |
| 65 | 行业分类数据落地（东财行业口径） | data, feat, P0 |
| 66 | universe 模块：命名股票池与自定义池的 PIT 解析 | data, feat, P0 |
| 67 | 指数日行情落地与回测基准绩效（超额/跟踪误差/IR） | backtest, feat, P1 |
| 68 | 指数增强优化器：基准锚定约束族 + 放松阶梯 + 失败漂移 | portfolio, feat, P0 |
| 69 | 持仓风险分析四表：指数/市值/风格/主动行业暴露 | portfolio, feat, P1 |
| 70 | universe 与策略配置接线：训练/回测/跑批按池工作 | infra, feat, P1 |
| 71 | 指增端到端验证：zz1000 池 + 000852 基准出超额净值曲线 | test, P0 |

依赖：64/65/67 并行 → 66 等 64 → 68 等 64+65 → 69 等 68 → 70 等 66+68 → 71 收口。范式参照 `reference-portfolio-optimization.py`（指增约束族与放松阶梯）、`reference-analyze-weights.py`（四表）、`reference-hedge-backtest.py`（对冲，backlog #72），均在本目录。

**M2 因子挖掘内循环**

| # | Issue | Label |
|---|---|---|
| 17 | factor_api：接口规范 + schema 校验 | factor, feat, P0 |
| 18 | factor_api：截断重算检测（含预热期容差） | factor, feat, P0 |
| 19 | factor_api：复杂度度量（radon + AST） | factor, feat, P1 |
| 20 | eval/factor.py：因子级评估管线（性能预算秒级） | factor, feat, P0 |
| 21 | docker：沙盒镜像（Python + 依赖 + 数据只读挂载） | infra, feat, P0 |
| 22 | tasks/factor-mining：TASK.md + eval.sh | agent, feat, P0 |
| 23 | prompts：context.md / proposal.md 因子挖掘改造 | agent, feat, P0 |
| 24 | 内循环冒烟：一次 run 产出 ≥1 个入库因子 | agent, test, P0 |

**M3 进化机制与共线性**

| # | Issue | Label |
|---|---|---|
| 25 | factor_lib：因子库 schema + 血统链持久化 | factor, feat, P0 |
| 26 | factor_lib：行为相关性查重 | factor, feat, P0 |
| 27 | 评分函数：共线性折扣进 score.json | factor, feat, P0 |
| 28 | Context：方向分布地图注入 prompt | agent, feat, P1 |
| 29 | Proposal：同方向已有因子摘要注入 | agent, feat, P1 |
| 30 | inspiration-queue：方向配额 | agent, feat, P1 |
| 31 | factor_lib：定期整库 + 容量上限 | factor, feat, P2 |
| 32 | 多代进化验证：质量单调性 + 相关性受控 | agent, test, P1 |

**M4 AutoML 增强**

| # | Issue | Label |
|---|---|---|
| 33 | eval/model.py：walk-forward 模型评估 | automl, feat, P0 |
| 34 | automl：滚动重训自动化 | automl, feat, P0 |
| 35 | automl：AutoGluon 配方调优（preset/HPO/集成深度） | automl, feat, P1 |
| 36 | automl：因子贡献度分析 | automl, feat, P2 |
| 37 | daily：简报生成（调仓明细/风险敞口/因子表现） | infra, feat, P1 |

**M5 AutoResearch（远期）**

| # | Issue | Label |
|---|---|---|
| 46 | agent：训练配置的 AutoResearch 循环（Agent 调参） | agent, automl, feat, P2 |
| 47 | agent：网络结构演化（Agent 改模型结构） | agent, automl, feat, P2 |

建模器选型：AutoGluon Tabular（本机有 GPU，启用 GPU 加速与内置 HPO/stacking）。AutoGluon 只负责单窗口训练，walk-forward 分割归 `quant/eval/model.py`，标签归 `quant/labels/`。M5 的两个方向复用 hyra-pi 内循环：Proposal 改训练配置或模型结构代码，eval 以 walk-forward 样本外 IC 打分。

### 5.3 Label 体系

```
模块：data / factor / backtest / portfolio / automl / agent / infra
类型：feat / fix / test / docs / chore
优先级：P0(阻塞主线) / P1(重要) / P2(增强)
```

### 5.4 工作流

- 每个 issue 一个 feature 分支 + 一个 PR，PR 描述关联 `Closes #N`。
- GitHub Projects 看板：Backlog / In Progress / Review / Done。
- Milestone 完成以 DoD 验收为准，验收后打 tag（`m0`、`m1`…）。
- 研究层的 prompt 迭代、因子库变化直接提交 main，不走 issue。

### 5.5 初始化命令参考（gh CLI）

```bash
# milestone
gh api repos/001eander/ASTER/milestones -f title="M0: 数据管线" -f description="akshare → Parquet，校验与涨跌停预计算"
gh api repos/001eander/ASTER/milestones -f title="M1: 端到端基线" -f description="手工因子+AutoGluon+凸优化+账户回测+每日调仓单"
gh api repos/001eander/ASTER/milestones -f title="M2: 因子挖掘内循环" -f description="沙盒+任务包+prompt改造，首个入库因子"
gh api repos/001eander/ASTER/milestones -f title="M3: 进化机制与共线性" -f description="四层防御+方向配额+整库"
gh api repos/001eander/ASTER/milestones -f title="M4: AutoML增强" -f description="walk-forward+滚动重训+贡献度"

# label（模块）
gh label create data      --repo 001eander/ASTER --color 1d76db
gh label create factor    --repo 001eander/ASTER --color 0e8a16
gh label create backtest  --repo 001eander/ASTER --color d93f0b
gh label create portfolio --repo 001eander/ASTER --color fbca04
gh label create automl    --repo 001eander/ASTER --color 5319e7
gh label create agent     --repo 001eander/ASTER --color d4c5f9
gh label create infra     --repo 001eander/ASTER --color c2e0c6
# label（优先级）
gh label create P0 --repo 001eander/ASTER --color b60205
gh label create P1 --repo 001eander/ASTER --color e99695
gh label create P2 --repo 001eander/ASTER --color fef2c0
```

## 6. 风险清单

| 风险 | 缓解 |
|---|---|
| 沙盒启动开销拖慢内循环 | 数据只读卷挂载 + 沙盒复用 |
| LLM token 成本（连续跑数天） | hyra-pi `--budget` 机制 + Context/Proposal 双模型分档 |
| Agent 挖出样本内过拟合因子 | 截断重算检测 + 留出样本外段 + 分层单调性要求 |
| akshare 接口变动 | 锁版本 + 校验报警 + 抽象层隔离（坏了只改 akshare.py） |
| akshare 退市股覆盖不全 → 幸存者偏差 | 第一版接受并在文档明示，后续可补源 |
| 自研回测器正确性 | 单元测试覆盖规则分支（T+1/涨跌停/整手/停牌/费用），小样本人工核对若干交易日 |
| 并行 Proposal 放大共线性 | inspiration-queue 方向配额 |

## 7. 关键设计原则（回顾）

1. 研究层与生产层解耦，因子库是唯一接口。
2. 三层回测共享同一套指标内核与同一套 open-to-open 标签。
3. 奖励信号一致性：入库标准必须体现在 score.json。
4. 账户模拟器同时服务回测与跟单，调仓单天生可执行。
5. 每个模块面向协议编程（数据源、因子、模型），实现可替换。
